"""Central path constants, resolved relative to the project root.

Every script/module that reads or writes a file under data/ or outputs/ should import its
target directory from here rather than constructing a relative path locally -- this is what
lets scripts be invoked from any working directory.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG_DIR = PROJECT_ROOT / "config"
DOCS_DIR = PROJECT_ROOT / "docs"

DATA_DIR = PROJECT_ROOT / "data"
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DATA_INVENTORY = OUTPUTS_DIR / "data_inventory"
OUTPUTS_EDA = OUTPUTS_DIR / "eda"
OUTPUTS_FEATURE_ANALYSIS = OUTPUTS_DIR / "feature_analysis"
OUTPUTS_MODEL_COMPARISON = OUTPUTS_DIR / "model_comparison"
OUTPUTS_DIAGNOSTICS = OUTPUTS_DIR / "diagnostics"
OUTPUTS_PREDICTIONS = OUTPUTS_DIR / "predictions"
OUTPUTS_MODELS = OUTPUTS_DIR / "models"

# Exploratory diagnostics for the win-total compression investigation. Kept physically
# separate from OUTPUTS_MODEL_COMPARISON/OUTPUTS_MODELS/OUTPUTS_PREDICTIONS -- nothing here
# is read by the production pipeline (train_models.py/evaluate_models.py/generate_predictions.py).
OUTPUTS_DIAGNOSTICS_COMPRESSION = OUTPUTS_DIR / "diagnostics_compression"
OUTPUTS_DIAGNOSTICS_COMPRESSION_TABLES = OUTPUTS_DIAGNOSTICS_COMPRESSION / "tables"
OUTPUTS_DIAGNOSTICS_COMPRESSION_PLOTS = OUTPUTS_DIAGNOSTICS_COMPRESSION / "plots"
OUTPUTS_DIAGNOSTICS_COMPRESSION_EXPERIMENTS = OUTPUTS_DIAGNOSTICS_COMPRESSION / "experiments"
OUTPUTS_DIAGNOSTICS_COMPRESSION_LOGS = OUTPUTS_DIAGNOSTICS_COMPRESSION / "logs"

ALL_OUTPUT_DIRS = (
    DATA_INTERIM_DIR,
    DATA_PROCESSED_DIR,
    OUTPUTS_DATA_INVENTORY,
    OUTPUTS_EDA,
    OUTPUTS_FEATURE_ANALYSIS,
    OUTPUTS_MODEL_COMPARISON,
    OUTPUTS_DIAGNOSTICS,
    OUTPUTS_PREDICTIONS,
    OUTPUTS_MODELS,
    OUTPUTS_DIAGNOSTICS_COMPRESSION_TABLES,
    OUTPUTS_DIAGNOSTICS_COMPRESSION_PLOTS,
    OUTPUTS_DIAGNOSTICS_COMPRESSION_EXPERIMENTS,
    OUTPUTS_DIAGNOSTICS_COMPRESSION_LOGS,
)


def ensure_dirs() -> None:
    """Create every data/output directory this project writes to, if missing."""
    for directory in ALL_OUTPUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
