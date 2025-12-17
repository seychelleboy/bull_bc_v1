"""
Core module - Reusable infrastructure for Bull BC1 bot
"""
from .base_engine import BaseEngine, Signal
from .base_scenario import BaseScenario, ScenarioResult
from .thread_safe_cache import ThreadSafeCache
from .rate_limiter import RateLimiter, MultiEndpointRateLimiter
from .exceptions import (
    BullBotException,
    ConfigurationError,
    DataFetchError,
    ModelError,
    ExecutionError,
    RateLimitError,
    InsufficientDataError,
    RiskValidationError,
    ScenarioDetectionError,
    BAITScoringError
)

__all__ = [
    'BaseEngine',
    'Signal',
    'BaseScenario',
    'ScenarioResult',
    'ThreadSafeCache',
    'RateLimiter',
    'MultiEndpointRateLimiter',
    'BullBotException',
    'ConfigurationError',
    'DataFetchError',
    'ModelError',
    'ExecutionError',
    'RateLimitError',
    'InsufficientDataError',
    'RiskValidationError',
    'ScenarioDetectionError',
    'BAITScoringError'
]
