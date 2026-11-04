"""Regression test for scripts/train_production_artifact.py's version pruning. A raw string
sort put "v2026w10_..." before "v2026w7_...", so the first double-digit week's retrain deleted
its own fresh artifact and left latest.json dangling (2026-11-03 weekly picks failure)."""
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

tpa = importlib.import_module("train_production_artifact")

# The exact production dir contents from the failed run.
EXISTING = [
    "v2026w6_20261006T200634Z",
    "v2026w6_20261013T151328Z",
    "v2026w7_20261013T203709Z",
    "v2026w7_20261020T151006Z",
    "v2026w7_20261022T133036Z",
    "v2026w8_20261020T190447Z",
    "v2026w8_20261026T193823Z",
    "v2026w9_20261027T195733Z",
]
NEW = "v2026w10_20261104T013751Z"


def _make(tmp_path: Path, names: list[str], latest: str) -> None:
    for n in names:
        (tmp_path / n).mkdir()
        (tmp_path / n / "metadata.json").write_text("{}")
    (tmp_path / "latest.json").write_text(json.dumps({"version": latest}))


def test_double_digit_week_survives_prune(tmp_path):
    _make(tmp_path, EXISTING + [NEW], latest=NEW)
    deleted = tpa._prune_old_versions(tmp_path, keep=8)
    assert [d.name for d in deleted] == ["v2026w6_20261006T200634Z"]
    assert (tmp_path / NEW).is_dir()


def test_orders_across_seasons_numerically(tmp_path):
    names = ["v2025w15_20251201T000000Z", "v2026w2_20260910T000000Z", "v2026w11_20261110T000000Z"]
    _make(tmp_path, names, latest=names[-1])
    deleted = tpa._prune_old_versions(tmp_path, keep=2)
    assert [d.name for d in deleted] == ["v2025w15_20251201T000000Z"]


def test_never_deletes_version_latest_points_to(tmp_path):
    _make(tmp_path, EXISTING, latest=EXISTING[0])
    deleted = tpa._prune_old_versions(tmp_path, keep=1)
    assert EXISTING[0] not in {d.name for d in deleted}
    assert (tmp_path / EXISTING[0]).is_dir()
    assert (tmp_path / EXISTING[-1]).is_dir()


def test_ignores_unrecognized_dirs(tmp_path):
    _make(tmp_path, EXISTING[:2], latest=EXISTING[1])
    (tmp_path / "vscode_scratch").mkdir()
    tpa._prune_old_versions(tmp_path, keep=1)
    assert (tmp_path / "vscode_scratch").is_dir()
