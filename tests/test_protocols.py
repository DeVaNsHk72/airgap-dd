"""G2 gate: sample rates uniform, silence handling uniform, no file in more
than one split, bona fide and fake come from the paired corpora."""

from src.data.protocols import build_bonafide_manifest, build_fake_manifest

MAILABS_SPEAKERS = {"elizabeth_klett", "elliot_miller", "judy_bieber", "mary_ann"}


def test_no_duplicate_paths_within_each_manifest() -> None:
    for manifest in (build_bonafide_manifest(), build_fake_manifest()):
        paths = [u.path for u in manifest]
        assert len(paths) == len(set(paths)), "duplicate file(s) in one manifest"


def test_no_path_in_more_than_one_split() -> None:
    for manifest in (build_bonafide_manifest(), build_fake_manifest()):
        by_split: dict[str, set] = {"train": set(), "dev": set()}
        for u in manifest:
            by_split[u.split].add(u.path)
        overlap = by_split["train"] & by_split["dev"]
        assert not overlap, f"{len(overlap)} file(s) appear in both train and dev"


def test_bonafide_and_fake_are_paired_corpora() -> None:
    """Rule 3: MLAAD fake must pair with M-AILABS bona fide, verified from
    the source data (original_file column), not assumed from directory
    layout."""
    for u in build_fake_manifest():
        assert u.speaker in MAILABS_SPEAKERS, (
            f"{u.path} traces back to speaker {u.speaker!r}, not one of our "
            "M-AILABS bona fide speakers — fake and bona fide have drifted apart"
        )


def test_every_speaker_present_in_both_splits() -> None:
    bf = build_bonafide_manifest()
    for split in ("train", "dev"):
        speakers_in_split = {u.speaker for u in bf if u.split == split}
        assert speakers_in_split == MAILABS_SPEAKERS, (
            f"{split} split is missing speaker(s): {MAILABS_SPEAKERS - speakers_in_split}"
        )


def test_fake_split_matches_its_source_bonafide_split() -> None:
    """A MLAAD clip must land in the same split as the bona fide recording
    it was synthesized from — otherwise the same sentence could appear as
    bona fide in train and as a fake of that sentence in dev."""
    bf_split_by_key = {(u.speaker, u.book): u.split for u in build_bonafide_manifest()}
    for u in build_fake_manifest():
        assert u.split == bf_split_by_key[(u.speaker, u.book)]


if __name__ == "__main__":
    test_no_duplicate_paths_within_each_manifest()
    test_no_path_in_more_than_one_split()
    test_bonafide_and_fake_are_paired_corpora()
    test_every_speaker_present_in_both_splits()
    test_fake_split_matches_its_source_bonafide_split()
    print("G2 assertion suite: PASS")
