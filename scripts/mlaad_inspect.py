"""Inspect the MLAAD repo history for a revision matching ReplayDF's v5 lineage.

ReplayDF's spoofed audio comes from MLAAD v5, architectures bark / vits /
xtts_v1.1 / xtts_v2. HF `main` now serves v9, whose English split contains
none of those. HF dataset repos keep full git history, so the v5 files may
still be reachable by commit SHA.

Read-only. Downloads nothing but the file listing.
"""

from huggingface_hub import HfApi

REPO = "mueller91/MLAAD"
V5_MARKERS = ("bark", "vits", "xtts_v1.1", "xtts_v2", "xtts")


def main() -> None:
    api = HfApi()
    print("auth:", api.whoami()["name"])

    commits = api.list_repo_commits(REPO, repo_type="dataset")
    print(f"\n{len(commits)} commits on main\n")
    for c in commits:
        print(f"  {c.created_at.date()}  {c.commit_id[:10]}  {c.title[:64]}")

    print("\nscanning revisions for v5 English architectures...")
    for c in commits:
        try:
            files = api.list_repo_files(REPO, repo_type="dataset", revision=c.commit_id)
        except Exception as exc:
            print(f"  {c.commit_id[:10]}  unreadable: {type(exc).__name__}")
            continue
        models = {f.split("/")[2] for f in files if f.startswith("fake/en/") and f.count("/") > 2}
        hits = {m for m in models if any(k in m.lower() for k in V5_MARKERS)}
        flag = "  <-- v5 candidate" if hits else ""
        print(f"  {c.commit_id[:10]}  en models={len(models):3d}  v5-arch={sorted(hits)}{flag}")


if __name__ == "__main__":
    main()
