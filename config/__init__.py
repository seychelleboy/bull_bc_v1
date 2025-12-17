"""
Configuration module for Bull BC1 Bot.

Centralized configuration with priority:
1. Environment variables (.env) - secrets only
2. config.local.yaml - local overrides
3. defaults.yaml - all default values
"""
from .settings import (
    Config,
    get_config,
    # API configs
    APIConfig,
    EODHDConfig,
    CoinGeckoConfig,
    # Trading & Risk
    TradingConfig,
    RiskConfig,
    CircuitBreakersConfig,
    # Cache
    CacheConfig,
    # BAIT Scoring
    ScoringConfig,
    # Scenarios (BULLISH)
    ScenarioConfig,
    BullishDivergenceConfig,
    SuccessfulBreakoutConfig,
    OversoldConfig,
    VolumeBreakoutConfig,
    FundingExtremeConfig,
    FearGreedConfig,
    # Data sources
    DataSourcesConfig,
    DataConfig,
    MT5Config,
    # Database & Paths
    DatabaseConfig,
    PathsConfig,
    LoggingConfig,
    BacktestConfig,
)
from .logging_config import setup_logging, get_logger

__all__ = [
    # Main config
    'Config',
    'get_config',
    # API
    'APIConfig',
    'EODHDConfig',
    'CoinGeckoConfig',
    # Trading
    'TradingConfig',
    'RiskConfig',
    'CircuitBreakersConfig',
    # Cache
    'CacheConfig',
    # Scoring
    'ScoringConfig',
    # Scenarios
    'ScenarioConfig',
    'BullishDivergenceConfig',
    'SuccessfulBreakoutConfig',
    'OversoldConfig',
    'VolumeBreakoutConfig',
    'FundingExtremeConfig',
    'FearGreedConfig',
    # Data
    'DataSourcesConfig',
    'DataConfig',
    'MT5Config',
    'DatabaseConfig',
    'PathsConfig',
    'LoggingConfig',
    'BacktestConfig',
    # Logging
    'setup_logging',
    'get_logger'
]
