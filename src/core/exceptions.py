"""
Custom Exceptions for Bull BC1 Bot

Provides a hierarchy of exceptions for specific error handling.
"""


class BullBotException(Exception):
    """Base exception for Bull BC1 bot."""
    pass


class ConfigurationError(BullBotException):
    """Configuration-related errors."""
    pass


class DataFetchError(BullBotException):
    """Data fetching errors from APIs."""
    pass


class ModelError(BullBotException):
    """Model prediction errors."""
    pass


class ExecutionError(BullBotException):
    """Trade execution errors."""
    pass


class RateLimitError(BullBotException):
    """API rate limit exceeded."""
    pass


class InsufficientDataError(BullBotException):
    """Not enough data for analysis."""
    pass


class RiskValidationError(BullBotException):
    """Risk management validation failed."""
    pass


class ScenarioDetectionError(BullBotException):
    """Error during scenario detection."""
    pass


class BAITScoringError(BullBotException):
    """Error during BAIT scoring."""
    pass
