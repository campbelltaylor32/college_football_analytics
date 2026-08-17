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
OUTPUTS_EDA = OUTPUTS_DIR / "eda"
OUTPUTS_MODEL_COMPARISON = OUTPUTS_DIR / "model_comparison"
OUTPUTS_MODELS = OUTPUTS_DIR / "models"
OUTPUTS_RATINGS = OUTPUTS_DIR / "ratings"

ALL_OUTPUT_DIRS = (
    DATA_INTERIM_DIR,
    DATA_PROCESSED_DIR,
    OUTPUTS_EDA,
    OUTPUTS_MODEL_COMPARISON,
    OUTPUTS_MODELS,
    OUTPUTS_RATINGS,
)


def ensure_dirs() -> None:
    """Create every data/output directory this project writes to, if missing."""
    for directory in ALL_OUTPUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
