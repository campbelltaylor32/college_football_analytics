"""Central path constants, resolved relative to the project root.

Every script/module that reads or writes a file under data/ or outputs/ should import its
target directory from here rather than constructing a relative path locally -- this is what
lets scripts be invoked from any working directory, and is the direct fix for the stale
hardcoded os.chdir() confirmed in Week_Predictions.ipynb.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = PROJECT_ROOT.parent

CONFIG_DIR = PROJECT_ROOT / "config"
DOCS_DIR = PROJECT_ROOT / "docs"

# The modeling input lives one level up, in the repo-level Data/ folder shared with the R
# scripts and the current notebook pipeline -- this project reads it, never writes to it.
REPO_DATA_DIR = REPO_ROOT / "Data"

DATA_DIR = PROJECT_ROOT / "data"
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DATA_INVENTORY = OUTPUTS_DIR / "data_inventory"
OUTPUTS_EDA = OUTPUTS_DIR / "eda"
OUTPUTS_FEATURE_ANALYSIS = OUTPUTS_DIR / "feature_analysis"
OUTPUTS_MODEL_COMPARISON = OUTPUTS_DIR / "model_comparison"
OUTPUTS_THRESHOLD_SELECTION = OUTPUTS_DIR / "threshold_selection"
OUTPUTS_PREDICTIONS = OUTPUTS_DIR / "predictions"
OUTPUTS_MODELS = OUTPUTS_DIR / "models"
OUTPUTS_CALIBRATION = OUTPUTS_DIR / "calibration"

ALL_OUTPUT_DIRS = (
    DATA_INTERIM_DIR,
    DATA_PROCESSED_DIR,
    OUTPUTS_DATA_INVENTORY,
    OUTPUTS_EDA,
    OUTPUTS_FEATURE_ANALYSIS,
    OUTPUTS_MODEL_COMPARISON,
    OUTPUTS_THRESHOLD_SELECTION,
    OUTPUTS_PREDICTIONS,
    OUTPUTS_MODELS,
    OUTPUTS_CALIBRATION,
)


def ensure_dirs() -> None:
    """Create every data/output directory this project writes to, if missing."""
    for directory in ALL_OUTPUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
