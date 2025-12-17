"""
Central Configuration System for Bull BC1 Bot

Configuration is loaded from multiple sources with the following priority:
1. Environment variables (highest - for secrets only)
2. config.local.yaml (local overrides, git-ignored)
3. defaults.yaml (all default values, version controlled)
4. Dataclass defaults (lowest - fallback only)

Usage:
    config = Config.load()
    print(config.trading.min_confidence_threshold)  # 80.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import os
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Project root directory
CONFIG_DIR = Path(__file__).parent
PROJECT_ROOT = CONFIG_DIR.parent


def _load_yaml(path: Path) -> dict:
    """Load YAML file if it exists."""
    if path.exists():
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, override takes precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _get_nested(data: dict, *keys, default=None):
    """Safely get nested dictionary value."""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, default)
        else:
            return default
    return data if data is not None else default


# =============================================================================
# API CONFIGURATION
# =============================================================================

@dataclass
class EODHDConfig:
    """EODHD API configuration"""
    api_key: str = ''
    base_url: str = 'https://eodhd.com/api'
    ws_url: str = 'wss://ws.eodhistoricaldata.com/ws/crypto'
    calls_per_second: int = 5
    calls_per_minute: int = 100
    calls_per_day: int = 100000
    burst_size: int = 10
    websocket_timeout_seconds: int = 30


@dataclass
class CoinGeckoConfig:
    """CoinGecko API configuration"""
    base_url: str = 'https://api.coingecko.com/api/v3'
    calls_per_second: int = 5
    calls_per_minute: int = 30
    calls_per_day: int = 50000
    burst_size: int = 5


@dataclass
class APIConfig:
    """All API configurations"""
    eodhd: EODHDConfig = field(default_factory=EODHDConfig)
    coingecko: CoinGeckoConfig = field(default_factory=CoinGeckoConfig)
    fear_greed_url: str = 'https://api.alternative.me/fng/'
    cryptopanic_api_key: str = ''
    cryptopanic_url: str = 'https://cryptopanic.com/api/v1/posts/'

    # Legacy aliases for backward compatibility
    @property
    def eodhd_api_key(self) -> str:
        return self.eodhd.api_key

    @property
    def eodhd_base_url(self) -> str:
        return self.eodhd.base_url

    @property
    def eodhd_ws_url(self) -> str:
        return self.eodhd.ws_url

    @property
    def coingecko_base_url(self) -> str:
        return self.coingecko.base_url

    @property
    def eodhd_calls_per_day(self) -> int:
        return self.eodhd.calls_per_day

    @property
    def eodhd_calls_per_minute(self) -> int:
        return self.eodhd.calls_per_minute

    @property
    def cryptopanic_token(self) -> str:
        return self.cryptopanic_api_key


# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

@dataclass
class CacheConfig:
    """TTL cache durations (seconds)"""
    quote_ttl: int = 5              # Real-time price quotes
    price_ttl: int = 30             # Bitcoin current price
    intraday_ttl: int = 60          # Intraday bars
    features_ttl: int = 300         # Calculated features
    news_ttl: int = 300             # News sentiment
    fear_greed_ttl: int = 3600      # Fear & Greed Index
    fundamentals_ttl: int = 3600    # Market fundamentals
    bitcoin_history_ttl: int = 300  # Historical data
    global_data_ttl: int = 300      # CoinGecko global
    derivatives_ttl: int = 300      # Derivatives data

    # Legacy aliases
    @property
    def price_data_ttl(self) -> int:
        return self.price_ttl

    @property
    def intraday_data_ttl(self) -> int:
        return self.intraday_ttl


# =============================================================================
# TRADING CONFIGURATION
# =============================================================================

@dataclass
class TradingConfig:
    """Trading execution settings - LONG only for Bull Bot"""
    symbol: str = 'BTCUSD'
    direction: str = 'LONG'  # Bull bot ONLY does LONG positions
    auto_execute: bool = True
    paper_trade: bool = False
    min_confidence_threshold: float = 80.0

    # Timeframes
    primary_timeframe: str = '4h'
    entry_timeframe: str = '1h'
    trend_timeframe: str = '1d'

    # Position management
    hold_overnight: bool = True
    use_trailing_stop: bool = True
    trailing_stop_atr_multiplier: float = 2.0
    initial_stop_atr_multiplier: float = 2.5
    atr_buffer_multiplier: float = 0.5

    # Scanning
    scan_interval_seconds: int = 300


# =============================================================================
# RISK CONFIGURATION
# =============================================================================

@dataclass
class RiskConfig:
    """Risk management parameters"""
    # Account settings
    initial_balance: float = 10000.0
    max_leverage: float = 3.0

    # Position sizing
    risk_per_trade_percent: float = 2.0
    max_position_size_usd: float = 10000.0
    max_position_percent: float = 0.25

    # Daily limits
    max_daily_loss_percent: float = 6.0
    max_daily_trades: int = 10

    # Drawdown protection
    max_drawdown_percent: float = 15.0

    # Trade quality
    min_reward_risk_ratio: float = 2.0
    max_open_positions: int = 3

    # Stop distances
    min_stop_distance_pct: float = 0.5
    max_stop_distance_pct: float = 5.0

    # Kelly criterion
    kelly_default_win_rate: float = 0.65
    kelly_default_win_loss_ratio: float = 2.0
    kelly_safety_multiplier: float = 0.5
    kelly_min: float = 0.01
    kelly_max: float = 0.25

    # Legacy alias
    @property
    def min_confidence_threshold(self) -> float:
        return 80.0  # Use trading.min_confidence_threshold instead


# =============================================================================
# CIRCUIT BREAKERS CONFIGURATION
# =============================================================================

@dataclass
class CircuitBreakersConfig:
    """Risk circuit breaker settings"""
    # Daily loss limit
    enable_daily_loss_limit: bool = True
    max_daily_loss_percent: float = 6.0

    # Maximum positions limit
    enable_max_positions_limit: bool = True
    max_positions: int = 3

    # Connection health monitoring
    enable_connection_check: bool = True
    connection_check_interval_seconds: int = 60
    connection_max_failures: int = 3


# =============================================================================
# BACKTESTING CONFIGURATION
# =============================================================================

@dataclass
class BacktestConfig:
    """Backtesting engine settings"""
    enabled: bool = True
    default_days_back: int = 365
    default_timeframe: str = "4h"
    initial_balance: float = 10000.0
    output_dir: str = "data/backtest"
    use_paper_slippage: bool = False


# =============================================================================
# BAIT SCORING CONFIGURATION - LONG OPTIMIZED
# =============================================================================

@dataclass
class ScoringConfig:
    """
    BAIT scoring configuration for LONG-optimized confidence.

    INVERTED from SHORT scoring:
    - Higher scores = more bullish = BETTER for LONG
    - Score >= 75 = STRONG buy signal
    - Score >= 65 = MODERATE buy signal
    - Score >= 55 = WEAK buy signal
    - Score < 55 = AVOID (too bearish for longs)
    """
    # Component weights for LONG signals
    behavioral_weight: float = 0.30      # Fear & Greed (fear = contrarian buy)
    analytical_weight: float = 0.25      # Funding rate (negative = bullish)
    informational_weight: float = 0.25   # News catalyst detection
    technical_weight: float = 0.20       # RSI, MACD, OBV, Bollinger

    # Confidence thresholds for LONG signals (INVERTED - higher = more bullish)
    strong_threshold: float = 75.0       # Score >= 75 = STRONG buy signal
    moderate_threshold: float = 65.0     # Score >= 65 = MODERATE buy signal
    weak_threshold: float = 55.0         # Score >= 55 = WEAK buy signal
    # Score < 55 = AVOID (too bearish for longs)

    # Catalyst keywords for informational scoring
    bullish_keywords: List[str] = field(default_factory=lambda: [
        'etf', 'approval', 'adoption', 'institution', 'treasury',
        'halving', 'upgrade', 'partnership', 'integration', 'launch',
        'accumulation', 'buying', 'bullish', 'breakout', 'rally',
        'milestone', 'record', 'growth', 'expansion', 'strategic'
    ])
    bearish_keywords: List[str] = field(default_factory=lambda: [
        'hack', 'exploit', 'bankruptcy', 'crash', 'sec', 'lawsuit',
        'ban', 'regulation', 'investigation', 'fraud', 'ponzi',
        'sell-off', 'dump', 'liquidation', 'delisting', 'audit'
    ])

    # Cache TTLs for scoring components (seconds)
    behavioral_cache_ttl: int = 900      # 15 min
    analytical_cache_ttl: int = 21600    # 6 hours
    informational_cache_ttl: int = 600   # 10 min
    technical_cache_ttl: int = 300       # 5 min

    # Ensemble integration
    ensemble_confidence_threshold: float = 80.0


# =============================================================================
# SCENARIO CONFIGURATION - BULLISH PATTERNS
# =============================================================================

@dataclass
class BullishDivergenceConfig:
    """Bullish Divergence scenario thresholds (63.6% win rate)"""
    rsi_threshold: float = 40.0          # RSI must be below this (oversold zone)
    obv_lookback: int = 20
    lookback_period: int = 20
    divergence_lookback: int = 14
    price_low_tolerance: float = 0.005   # 0.5% tolerance for "lower low"


@dataclass
class SuccessfulBreakoutConfig:
    """Successful Breakout scenario thresholds (70% win rate - BEST)"""
    volume_surge_threshold: float = 1.5  # 50% volume increase for breakout
    price_breakout_threshold: float = 0.02  # 2% above resistance = breakout
    breakout_hold_threshold: float = 0.01   # Must hold 1% above resistance
    rsi_not_overbought: int = 70         # RSI should not be extremely overbought
    confirmation_bars: int = 2           # Bars to confirm breakout holds
    confirmation_pct: float = 1.01       # Must be above 101% of resistance
    magnitude_threshold: float = 0.03    # 3% breakout = significant
    lookback_period: int = 20


@dataclass
class OversoldConfig:
    """Oversold scenario thresholds (55-60% win rate)"""
    vwap_threshold: float = -0.03        # 3% below VWAP = oversold
    atr_multiplier: float = 2.5
    rsi_threshold: int = 30              # RSI oversold level
    bb_threshold: float = 0.0            # Below lower Bollinger Band
    sma_deviation_threshold: float = -5.0  # 5% below SMA20
    mean_reversion_target_pct: float = 1.03  # 3% reversion target


@dataclass
class VolumeBreakoutConfig:
    """Volume Breakout scenario thresholds (60% win rate)"""
    obv_lookback: int = 20
    volume_spike: float = 2.0            # 2x average = spike
    volume_multiplier: float = 1.5
    breakout_threshold: float = 0.01     # 1% price breakout
    cmf_threshold: float = 0.1           # CMF above 0.1 = buying pressure
    lookback_period: int = 20


@dataclass
class FundingExtremeConfig:
    """
    Funding Extreme scenario thresholds (65-70% win rate, Bitcoin-specific).

    For LONG: NEGATIVE funding = shorts overcrowded = bullish
    """
    threshold: float = -0.001            # -0.1% funding rate = shorts overcrowded
    volume_spike_threshold: float = 2.0
    target_pct: float = 1.03             # 3% take profit target


@dataclass
class FearGreedConfig:
    """Fear & Greed classification thresholds"""
    extreme_fear_threshold: int = 25     # <= 25 = extreme fear (BULLISH for LONG)
    extreme_greed_threshold: int = 75    # >= 75 = extreme greed (bearish)
    neutral_low: int = 40
    neutral_high: int = 60


@dataclass
class ScenarioConfig:
    """All scenario detection thresholds for BULLISH patterns"""
    # Shared thresholds
    price_near_lows_pct: float = 1.02    # Within 2% of recent lows
    divergence_slope_threshold: float = 0.001

    # Individual bullish scenarios
    bullish_divergence: BullishDivergenceConfig = field(default_factory=BullishDivergenceConfig)
    successful_breakout: SuccessfulBreakoutConfig = field(default_factory=SuccessfulBreakoutConfig)
    oversold: OversoldConfig = field(default_factory=OversoldConfig)
    volume_breakout: VolumeBreakoutConfig = field(default_factory=VolumeBreakoutConfig)
    funding_extreme: FundingExtremeConfig = field(default_factory=FundingExtremeConfig)
    fear_greed: FearGreedConfig = field(default_factory=FearGreedConfig)

    # Bitcoin-specific thresholds
    btc_dominance_high_threshold: float = 55.0   # >55% = flight to safety
    btc_dominance_low_threshold: float = 42.0    # <42% = alt season
    ath_near_threshold: float = 10.0             # Within 10% of ATH
    ath_oversold_threshold: float = 50.0         # 50%+ from ATH = oversold (BULLISH)

    # Legacy flat accessors
    @property
    def bullish_divergence_rsi_threshold(self) -> float:
        return self.bullish_divergence.rsi_threshold

    @property
    def bullish_divergence_obv_lookback(self) -> int:
        return self.bullish_divergence.obv_lookback

    @property
    def successful_breakout_volume_surge_threshold(self) -> float:
        return self.successful_breakout.volume_surge_threshold

    @property
    def successful_breakout_price_breakout_threshold(self) -> float:
        return self.successful_breakout.price_breakout_threshold

    @property
    def oversold_vwap_threshold(self) -> float:
        return self.oversold.vwap_threshold

    @property
    def oversold_atr_multiplier(self) -> float:
        return self.oversold.atr_multiplier

    @property
    def volume_breakout_obv_lookback(self) -> int:
        return self.volume_breakout.obv_lookback

    @property
    def volume_breakout_volume_spike(self) -> float:
        return self.volume_breakout.volume_spike

    @property
    def funding_rate_extreme_threshold(self) -> float:
        return self.funding_extreme.threshold


# =============================================================================
# OTHER CONFIGURATIONS
# =============================================================================

@dataclass
class CorrelationSourcesConfig:
    """Per-asset data source configuration"""
    gold: str = 'mt5'
    sp500: str = 'mt5'
    dxy: str = 'mt5'
    nasdaq: str = 'mt5'


@dataclass
class DataSourcesConfig:
    """
    Hybrid data architecture configuration.

    Allows switching between MT5 (free, direct broker data) and EODHD (paid API).
    MT5 is preferred for BTC and correlation assets when available.
    """
    # Primary BTC data source: "mt5" or "eodhd"
    btc_data_source: str = 'mt5'

    # Per-asset correlation sources
    correlation_sources: CorrelationSourcesConfig = field(default_factory=CorrelationSourcesConfig)

    # Fallback behavior
    enable_fallback: bool = True
    fallback_source: str = 'eodhd'


def _safe_int(value: str, default: int = 0) -> int:
    """Safely convert string to int, return default if not possible."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


@dataclass
class MT5Config:
    """MetaTrader 5 connection and symbol settings"""
    # Connection settings (from environment for security)
    path: str = field(default_factory=lambda: os.getenv(
        'MT5_PATH',
        r'C:\Program Files\Pepperstone MetaTrader 5\terminal64.exe'
    ))
    login: int = field(default_factory=lambda: _safe_int(os.getenv('MT5_LOGIN', '0')))
    password: str = field(default_factory=lambda: os.getenv('MT5_PASSWORD', ''))
    server: str = field(default_factory=lambda: os.getenv('MT5_SERVER', 'PepperstoneUK-Demo'))

    # Trading symbols (Pepperstone UK - Spread Bet variants)
    btc_symbol: str = 'BTCUSD_SB'
    gold_symbol: str = 'XAUUSD_SB'
    sp500_symbol: str = 'US500_SB'
    dxy_symbol: str = 'USDX_SB'
    nasdaq_symbol: str = 'NAS100_SB'

    # Connection settings
    timeout_seconds: int = 60
    retry_attempts: int = 3
    max_bars_request: int = 10000


@dataclass
class DatabaseConfig:
    """Database configuration"""
    path: str = 'data/database/bull_bc_1.db'
    wal_mode: bool = True


@dataclass
class PathsConfig:
    """File paths configuration"""
    project_root: str = field(default_factory=lambda: str(PROJECT_ROOT))
    models_dir: str = 'data/models'
    logs_dir: str = 'data/logs'
    database_dir: str = 'data/database'


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = 'INFO'
    log_file: str = 'data/logs/bull_bc_1.log'
    max_bytes: int = 10_000_000
    backup_count: int = 5
    format: str = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'


@dataclass
class DataConfig:
    """Data fetching configuration"""
    cache_ttl: int = 60
    lookback_days: int = 90


# =============================================================================
# MASTER CONFIG
# =============================================================================

@dataclass
class Config:
    """
    Master Configuration Container for Bull BC1 Bot

    Load priority:
    1. Environment variables (.env) - secrets only
    2. config.local.yaml - local overrides
    3. defaults.yaml - all default values
    """
    api: APIConfig = field(default_factory=APIConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    circuit_breakers: CircuitBreakersConfig = field(default_factory=CircuitBreakersConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    scenarios: ScenarioConfig = field(default_factory=ScenarioConfig)
    data_sources: DataSourcesConfig = field(default_factory=DataSourcesConfig)
    mt5: MT5Config = field(default_factory=MT5Config)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> 'Config':
        """
        Load configuration from YAML files and environment.

        Priority (highest to lowest):
        1. Environment variables (secrets only)
        2. config.local.yaml (local overrides)
        3. defaults.yaml (version controlled defaults)
        4. Dataclass defaults (fallback)
        """
        # Load YAML files
        defaults_path = CONFIG_DIR / 'defaults.yaml'
        local_path = CONFIG_DIR / 'config.local.yaml'

        defaults = _load_yaml(defaults_path)
        local = _load_yaml(local_path)

        # Merge: defaults <- local overrides
        yaml_config = _deep_merge(defaults, local)

        # Create config instance
        config = cls()

        # Apply YAML configuration
        config = cls._apply_yaml_config(config, yaml_config)

        # Apply environment variables (secrets)
        config = cls._apply_env_secrets(config)

        return config

    @classmethod
    def _apply_yaml_config(cls, config: 'Config', yaml_config: dict) -> 'Config':
        """Apply YAML configuration to dataclasses."""
        # API config
        api_cfg = yaml_config.get('api', {})
        if 'eodhd' in api_cfg:
            for k, v in api_cfg['eodhd'].items():
                if hasattr(config.api.eodhd, k):
                    setattr(config.api.eodhd, k, v)
        if 'coingecko' in api_cfg:
            for k, v in api_cfg['coingecko'].items():
                if hasattr(config.api.coingecko, k):
                    setattr(config.api.coingecko, k, v)
        if 'fear_greed' in api_cfg:
            config.api.fear_greed_url = api_cfg['fear_greed'].get('url', config.api.fear_greed_url)
        if 'cryptopanic' in api_cfg:
            config.api.cryptopanic_url = api_cfg['cryptopanic'].get('url', config.api.cryptopanic_url)

        # Cache config
        cache_cfg = yaml_config.get('cache', {})
        for k, v in cache_cfg.items():
            if hasattr(config.cache, k):
                setattr(config.cache, k, v)

        # Trading config
        trading_cfg = yaml_config.get('trading', {})
        for k, v in trading_cfg.items():
            if hasattr(config.trading, k):
                setattr(config.trading, k, v)

        # Risk config
        risk_cfg = yaml_config.get('risk', {})
        for k, v in risk_cfg.items():
            if hasattr(config.risk, k):
                setattr(config.risk, k, v)

        # Circuit breakers config
        circuit_breakers_cfg = yaml_config.get('circuit_breakers', {})
        for k, v in circuit_breakers_cfg.items():
            if hasattr(config.circuit_breakers, k):
                setattr(config.circuit_breakers, k, v)

        # Backtest config
        backtest_cfg = yaml_config.get('backtest', {})
        for k, v in backtest_cfg.items():
            if hasattr(config.backtest, k):
                setattr(config.backtest, k, v)

        # Scoring config (BAIT)
        scoring_cfg = yaml_config.get('scoring', {})
        for k, v in scoring_cfg.items():
            if hasattr(config.scoring, k):
                setattr(config.scoring, k, v)

        # Scenarios config (BULLISH patterns)
        scenarios_cfg = yaml_config.get('scenarios', {})
        for k, v in scenarios_cfg.items():
            if k == 'bullish_divergence' and isinstance(v, dict):
                for sk, sv in v.items():
                    if hasattr(config.scenarios.bullish_divergence, sk):
                        setattr(config.scenarios.bullish_divergence, sk, sv)
            elif k == 'successful_breakout' and isinstance(v, dict):
                for sk, sv in v.items():
                    if hasattr(config.scenarios.successful_breakout, sk):
                        setattr(config.scenarios.successful_breakout, sk, sv)
            elif k == 'oversold' and isinstance(v, dict):
                for sk, sv in v.items():
                    if hasattr(config.scenarios.oversold, sk):
                        setattr(config.scenarios.oversold, sk, sv)
            elif k == 'volume_breakout' and isinstance(v, dict):
                for sk, sv in v.items():
                    if hasattr(config.scenarios.volume_breakout, sk):
                        setattr(config.scenarios.volume_breakout, sk, sv)
            elif k == 'funding_extreme' and isinstance(v, dict):
                for sk, sv in v.items():
                    if hasattr(config.scenarios.funding_extreme, sk):
                        setattr(config.scenarios.funding_extreme, sk, sv)
            elif k == 'fear_greed' and isinstance(v, dict):
                for sk, sv in v.items():
                    if hasattr(config.scenarios.fear_greed, sk):
                        setattr(config.scenarios.fear_greed, sk, sv)
            elif hasattr(config.scenarios, k):
                setattr(config.scenarios, k, v)

        # Logging config
        logging_cfg = yaml_config.get('logging', {})
        for k, v in logging_cfg.items():
            if hasattr(config.logging, k):
                setattr(config.logging, k, v)

        # Database config
        db_cfg = yaml_config.get('database', {})
        for k, v in db_cfg.items():
            if hasattr(config.database, k):
                setattr(config.database, k, v)

        # Paths config
        paths_cfg = yaml_config.get('paths', {})
        for k, v in paths_cfg.items():
            if hasattr(config.paths, k):
                setattr(config.paths, k, v)

        # Data sources config (hybrid architecture)
        ds_cfg = yaml_config.get('data_sources', {})
        if 'btc_data_source' in ds_cfg:
            config.data_sources.btc_data_source = ds_cfg['btc_data_source']
        if 'enable_fallback' in ds_cfg:
            config.data_sources.enable_fallback = ds_cfg['enable_fallback']
        if 'fallback_source' in ds_cfg:
            config.data_sources.fallback_source = ds_cfg['fallback_source']
        if 'correlation_sources' in ds_cfg:
            cs = ds_cfg['correlation_sources']
            for k, v in cs.items():
                if hasattr(config.data_sources.correlation_sources, k):
                    setattr(config.data_sources.correlation_sources, k, v)

        # MT5 config
        mt5_cfg = yaml_config.get('mt5', {})
        for k, v in mt5_cfg.items():
            if hasattr(config.mt5, k) and k not in ('login', 'password', 'server', 'path'):
                setattr(config.mt5, k, v)

        return config

    @classmethod
    def _apply_env_secrets(cls, config: 'Config') -> 'Config':
        """Apply environment variables for secrets."""
        # API keys from environment
        eodhd_key = os.getenv('EODHD_API_KEY', '')
        if eodhd_key:
            config.api.eodhd.api_key = eodhd_key

        cryptopanic_key = os.getenv('CRYPTOPANIC_API_KEY', '')
        if cryptopanic_key:
            config.api.cryptopanic_api_key = cryptopanic_key

        return config

    def to_dict(self) -> dict:
        """Convert config to dictionary (for logging/debugging)."""
        from dataclasses import asdict
        result = asdict(self)
        # Mask secrets
        if result.get('api', {}).get('eodhd', {}).get('api_key'):
            result['api']['eodhd']['api_key'] = '***MASKED***'
        if result.get('api', {}).get('cryptopanic_api_key'):
            result['api']['cryptopanic_api_key'] = '***MASKED***'
        return result

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Check API key
        if not self.api.eodhd.api_key:
            errors.append("EODHD_API_KEY not set in environment")

        # Check thresholds are in valid ranges
        if not 0 <= self.trading.min_confidence_threshold <= 100:
            errors.append(f"min_confidence_threshold must be 0-100, got {self.trading.min_confidence_threshold}")

        if not 0 < self.risk.risk_per_trade_percent <= 10:
            errors.append(f"risk_per_trade_percent should be 0-10%, got {self.risk.risk_per_trade_percent}")

        # Check BAIT scoring weights sum to ~1.0
        weight_sum = (self.scoring.behavioral_weight + self.scoring.analytical_weight +
                     self.scoring.informational_weight + self.scoring.technical_weight)
        if not 0.99 <= weight_sum <= 1.01:
            errors.append(f"BAIT scoring weights should sum to 1.0, got {weight_sum}")

        # Bull bot specific: direction must be LONG
        if self.trading.direction != 'LONG':
            errors.append(f"Bull bot direction must be LONG, got {self.trading.direction}")

        return errors


# Convenience function
def get_config() -> Config:
    """Get the default configuration."""
    return Config.load()
