"""Machine-learning package (Phase 3)."""

from src.ml.dataset import MODEL_FEATURE_COLUMNS, build_dataset
from src.ml.model_registry import build_logistic_pipeline
from src.ml.trainer import ModelTrainer

__all__ = [
    "MODEL_FEATURE_COLUMNS",
    "ModelTrainer",
    "build_dataset",
    "build_logistic_pipeline",
]
