"""Central path constants, resolved relative to the project root, so scripts can be invoked
from any working directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG_DIR = PROJECT_ROOT / "config"

DATA_DIR = PROJECT_ROOT / "data"
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_MODEL_COMPARISON = OUTPUTS_DIR / "model_comparison"
OUTPUTS_MODELS = OUTPUTS_DIR / "models"
OUTPUTS_RANKINGS = OUTPUTS_DIR / "rankings"
OUTPUTS_FIGURES = OUTPUTS_DIR / "figures"

ALL_OUTPUT_DIRS = (
    DATA_INTERIM_DIR,
    DATA_PROCESSED_DIR,
    OUTPUTS_MODEL_COMPARISON,
    OUTPUTS_MODELS,
    OUTPUTS_RANKINGS,
    OUTPUTS_FIGURES,
)


def ensure_output_dirs() -> None:
    for d in ALL_OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
