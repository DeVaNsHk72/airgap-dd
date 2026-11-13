import pickle
import random

import lightgbm as lgb

from src.data.protocols import build_bonafide_manifest, build_fake_manifest
from src.eval.leakage_check import assert_no_heldout_leakage
from src.eval.metrics import compute_eer
from src.features.extract import extract_features_for_manifest

NON_FEATURE_COLS = {"label", "path"}

if __name__ == "__main__":
    random.seed(0)
    bf = build_bonafide_manifest()
    fk = build_fake_manifest()
    train_utts = random.sample([u for u in bf if u.split == "train"], 5000) + random.sample(
        [u for u in fk if u.split == "train"], 5000
    )
    dev_utts = random.sample([u for u in bf if u.split == "dev"], 1000) + random.sample(
        [u for u in fk if u.split == "dev"], 1000
    )
    assert_no_heldout_leakage(train_utts)
    assert_no_heldout_leakage(dev_utts)

    train_df = extract_features_for_manifest(train_utts, seed=None)
    dev_df = extract_features_for_manifest(dev_utts, seed=None)

    feature_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS]
    X_train, y_train = train_df[feature_cols], train_df["label"]
    X_dev, y_dev = dev_df[feature_cols], dev_df["label"]

    model = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, random_state=0, verbose=-1)
    model.fit(X_train, y_train, eval_set=[(X_dev, y_dev)], callbacks=[lgb.early_stopping(30, verbose=False)])

    dev_scores = model.predict_proba(X_dev)[:, 1]
    eer, _ = compute_eer(dev_scores[y_dev.values == 1], dev_scores[y_dev.values == 0])
    print(f"clean LightGBM dev EER (in-domain, M-AILABS/MLAAD): {eer*100:.2f}%")

    with open("checkpoints/lgbm_clean_baseline.pkl", "wb") as f:
        pickle.dump(model, f)
    print("saved checkpoints/lgbm_clean_baseline.pkl")
