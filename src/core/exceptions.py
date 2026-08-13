"""Custom exceptions for data acquisition, features, and ML training."""


class DataProviderError(Exception):
    """Base error for market data provider failures."""


class DataFetchError(DataProviderError):
    """Raised when market data cannot be fetched from a remote source."""


class EmptyDataError(DataProviderError):
    """Raised when a fetch succeeds but returns zero rows."""


class DataSaveError(DataProviderError):
    """Raised when fetched data cannot be persisted to disk."""


class FeatureError(Exception):
    """Base error for the feature-generation pipeline."""


class DataValidationError(FeatureError):
    """Raised when input OHLCV data fails quality checks."""


class FeatureGenerationError(FeatureError):
    """Raised when feature computation or processed CSV persistence fails."""


class MLError(Exception):
    """Base error for the machine-learning pipeline."""


class DatasetError(MLError):
    """Raised when dataset loading or target generation fails."""


class SplitError(MLError):
    """Raised when chronological train/validation/test splitting fails."""


class TrainingError(MLError):
    """Raised when model training, evaluation, or persistence fails."""
