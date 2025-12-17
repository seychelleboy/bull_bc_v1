"""
Data Aggregator - Combines all data sources with async parallel fetching

Central data layer that:
1. Fetches data from multiple sources concurrently
2. Caches results to minimize API calls
3. Provides unified data structure for feature engineering
4. Handles errors gracefully with fallbacks

Hybrid Data Architecture:
- Primary: MT5 Pepperstone (FREE) for BTC and correlation assets
- Fallback: EODHD API ($79.99/mo) - also used for news
- Free: CoinGecko for market metrics, Fear & Greed for sentiment
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import pandas as pd

from .eodhd_client import EODHDClient, OHLCVBar
from .fear_greed_client import FearGreedClient, FearGreedReading
from .coingecko_client import CoinGeckoClient, BTCMarketData
from .news_client import NewsAggregator, CryptoPanicClient
from .mt5_data_client import MT5DataClient
from ..core import ThreadSafeCache

logger = logging.getLogger(__name__)


@dataclass
class AggregatedData:
    """
    Complete aggregated data for analysis.

    Contains all data needed for:
    - Technical indicator calculation
    - BAIT scoring
    - Scenario detection (LONG patterns)
    """
    # Price data
    prices_4h: pd.DataFrame  # Primary timeframe
    prices_1h: pd.DataFrame  # Entry timing
    prices_daily: pd.DataFrame  # Trend
    current_price: float

    # Sentiment
    fear_greed: Dict[str, Any]

    # Market metrics
    btc_market: Dict[str, Any]

    # Correlation assets
    correlation_data: Dict[str, pd.DataFrame]

    # News sentiment
    news_sentiment: Dict[str, Any]

    # Metadata
    timestamp: datetime
    fetch_duration_ms: int
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/storage."""
        return {
            'current_price': self.current_price,
            'fear_greed': self.fear_greed,
            'btc_market': self.btc_market,
            'news_sentiment': self.news_sentiment,
            'timestamp': self.timestamp.isoformat(),
            'fetch_duration_ms': self.fetch_duration_ms,
            'errors': self.errors,
            'prices_4h_rows': len(self.prices_4h),
            'prices_1h_rows': len(self.prices_1h),
            'prices_daily_rows': len(self.prices_daily)
        }

    def is_valid(self) -> bool:
        """Check if data is valid for analysis."""
        return (
            self.current_price > 0 and
            len(self.prices_4h) >= 50 and
            bool(self.btc_market)
        )


class DataAggregator:
    """
    Central data aggregation service with hybrid data architecture.

    Supports two data modes:
    - MT5: Primary source for BTC and correlation data (FREE, direct broker)
    - EODHD: Fallback source (paid API, $79.99/mo)

    Example:
        aggregator = DataAggregator.from_config(config)
        await aggregator.initialize()

        # Get all data for analysis
        data = await aggregator.get_analysis_data()

        # Quick price check
        price = await aggregator.get_current_price()
    """

    def __init__(
        self,
        eodhd_api_key: str,
        cryptopanic_token: Optional[str] = None,
        cache_ttl: int = 60,
        config: Optional[Any] = None
    ):
        """
        Initialize data aggregator.

        Args:
            eodhd_api_key: EODHD API key
            cryptopanic_token: Optional CryptoPanic API token
            cache_ttl: Default cache TTL in seconds
            config: Full config object (for MT5 and data source settings)
        """
        self.config = config

        # Shared cache for all clients
        self.cache = ThreadSafeCache(default_ttl=cache_ttl)

        # Initialize EODHD client (fallback and news)
        self.eodhd = EODHDClient(
            api_key=eodhd_api_key,
            cache=self.cache
        )

        # Initialize MT5 client (primary data source)
        self.mt5: Optional[MT5DataClient] = None
        self._mt5_available = False

        # Data source configuration
        self._btc_source = 'mt5'  # Default to MT5
        self._enable_fallback = True

        if config and hasattr(config, 'data_sources'):
            self._btc_source = config.data_sources.btc_data_source
            self._enable_fallback = config.data_sources.enable_fallback

        # Initialize other clients
        self.fear_greed = FearGreedClient(cache=self.cache)
        self.coingecko = CoinGeckoClient(cache=self.cache)
        self.news = NewsAggregator(
            eodhd_client=self.eodhd,
            cryptopanic_client=CryptoPanicClient(
                auth_token=cryptopanic_token,
                cache=self.cache
            )
        )

        self._initialized = False

    @classmethod
    def from_config(cls, config) -> 'DataAggregator':
        """
        Create aggregator from config object.

        Args:
            config: Config object with API settings

        Returns:
            Configured DataAggregator instance
        """
        return cls(
            eodhd_api_key=config.api.eodhd_api_key,
            cryptopanic_token=config.api.cryptopanic_token,
            cache_ttl=config.data.cache_ttl,
            config=config
        )

    async def initialize(self) -> bool:
        """
        Initialize and verify all connections.

        Returns:
            True if all critical services are available
        """
        logger.info("Initializing data aggregator...")
        logger.info(f"Primary BTC data source: {self._btc_source}")

        # Initialize MT5 if it's the primary source
        if self._btc_source == 'mt5':
            try:
                self.mt5 = MT5DataClient(self.config)
                if await self.mt5.initialize():
                    self._mt5_available = True
                    price = await self.mt5.get_current_price('BTC-USD')
                    logger.info(f"MT5 connected (PRIMARY). BTC: ${price:,.2f}")
                else:
                    logger.warning("MT5 initialization failed, will use fallback")
                    self._mt5_available = False
            except Exception as e:
                logger.warning(f"MT5 unavailable: {e}")
                self._mt5_available = False

            # If MT5 failed and fallback is disabled, fail initialization
            if not self._mt5_available and not self._enable_fallback:
                logger.error("MT5 unavailable and fallback is disabled")
                return False

        # Test EODHD connection (fallback or if MT5 unavailable)
        if not self._mt5_available or self._btc_source == 'eodhd':
            try:
                price = await self.eodhd.get_quote('BTC-USD.CC')
                if price:
                    label = "PRIMARY" if self._btc_source == 'eodhd' else "FALLBACK"
                    logger.info(f"EODHD connected ({label}). BTC: ${price.get('close', 0):,.2f}")
                else:
                    logger.error("EODHD returned empty response")
                    return False
            except Exception as e:
                logger.error(f"EODHD connection failed: {e}")
                if not self._mt5_available:
                    return False

        # Test Fear & Greed (non-critical)
        try:
            fng = await self.fear_greed.get_current()
            logger.info(f"Fear & Greed: {fng.value} ({fng.classification.value})")
        except Exception as e:
            logger.warning(f"Fear & Greed unavailable: {e}")

        # Test CoinGecko (non-critical)
        try:
            btc_price = await self.coingecko.get_btc_price()
            logger.info(f"CoinGecko connected. BTC: ${btc_price:,.2f}")
        except Exception as e:
            logger.warning(f"CoinGecko unavailable: {e}")

        self._initialized = True
        source_info = "MT5" if self._mt5_available else "EODHD"
        logger.info(f"Data aggregator initialized successfully (using {source_info} for BTC)")
        return True

    async def close(self) -> None:
        """Close all client connections."""
        if self.mt5:
            await self.mt5.close()
        await self.eodhd.close()
        await self.fear_greed.close()
        await self.coingecko.close()
        await self.news.close()
        logger.info("Data aggregator connections closed")

    async def get_current_price(self) -> float:
        """
        Get current BTC price.

        Uses MT5 if available, falls back to EODHD, then CoinGecko.

        Returns:
            Current BTC-USD price
        """
        # Try MT5 first if available
        if self._mt5_available and self.mt5:
            try:
                price = await self.mt5.get_current_price('BTC-USD')
                if price > 0:
                    return price
            except Exception as e:
                logger.warning(f"Failed to get price from MT5: {e}")

        # Fallback to EODHD
        try:
            quote = await self.eodhd.get_quote('BTC-USD.CC')
            return float(quote.get('close', 0))
        except Exception as e:
            logger.warning(f"Failed to get price from EODHD: {e}")

        # Final fallback to CoinGecko
        try:
            return await self.coingecko.get_btc_price()
        except Exception as e2:
            logger.error(f"All price sources failed: {e2}")
            return 0.0

    async def get_analysis_data(
        self,
        lookback_days: int = 90,
        include_correlation: bool = True,
        include_news: bool = True
    ) -> AggregatedData:
        """
        Get all data needed for analysis.

        Fetches from all sources in parallel for efficiency.

        Args:
            lookback_days: Days of historical data to fetch
            include_correlation: Whether to fetch DXY, Gold, S&P500
            include_news: Whether to fetch news sentiment

        Returns:
            AggregatedData with all fetched data
        """
        start_time = datetime.now(timezone.utc)
        errors = []

        # Build list of tasks to run in parallel
        tasks = {
            'prices_4h': self._fetch_prices('4h', lookback_days),
            'prices_1h': self._fetch_prices('1h', min(lookback_days, 30)),
            'prices_daily': self._fetch_prices('d', lookback_days),
            'quote': self._fetch_quote(),
            'fear_greed': self._fetch_fear_greed(),
            'btc_market': self._fetch_btc_market(),
        }

        if include_correlation:
            tasks['correlation'] = self._fetch_correlation_data(lookback_days)

        if include_news:
            tasks['news'] = self._fetch_news_sentiment()

        # Execute all tasks concurrently
        results = {}
        task_objects = {k: asyncio.create_task(v) for k, v in tasks.items()}

        for name, task in task_objects.items():
            try:
                results[name] = await task
            except Exception as e:
                logger.warning(f"Failed to fetch {name}: {e}")
                errors.append(f"{name}: {str(e)}")
                results[name] = None

        # Extract current price
        current_price = 0.0
        if results.get('quote'):
            current_price = float(results['quote'].get('close', 0))

        # Build aggregated data
        fetch_duration = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

        data = AggregatedData(
            prices_4h=results.get('prices_4h', pd.DataFrame()),
            prices_1h=results.get('prices_1h', pd.DataFrame()),
            prices_daily=results.get('prices_daily', pd.DataFrame()),
            current_price=current_price,
            fear_greed=results.get('fear_greed', {}),
            btc_market=results.get('btc_market', {}),
            correlation_data=results.get('correlation', {}),
            news_sentiment=results.get('news', {}),
            timestamp=start_time,
            fetch_duration_ms=fetch_duration,
            errors=errors
        )

        logger.info(f"Data aggregation completed in {fetch_duration}ms with {len(errors)} errors")
        return data

    async def _fetch_quote(self) -> Dict:
        """Fetch current BTC quote from best available source."""
        # Try MT5 first
        if self._mt5_available and self.mt5:
            try:
                quote = await self.mt5.get_quote('BTC-USD')
                if quote:
                    return quote
            except Exception as e:
                logger.warning(f"MT5 quote fetch failed: {e}")

        # Fallback to EODHD
        return await self.eodhd.get_quote('BTC-USD.CC')

    async def _fetch_prices(
        self,
        interval: str,
        days: int
    ) -> pd.DataFrame:
        """
        Fetch price data from best available source.

        Uses MT5 if available, falls back to EODHD.
        """
        # Calculate bars needed (approximate)
        bars_per_day = {'1m': 1440, '5m': 288, '1h': 24, '4h': 6, 'd': 1, '1d': 1}
        multiplier = bars_per_day.get(interval.lower(), 6)
        bars_needed = days * multiplier + 50  # Buffer

        # Try MT5 first
        if self._mt5_available and self.mt5:
            try:
                df = await self.mt5.get_historical('BTC-USD', interval, bars_needed)
                if not df.empty:
                    logger.debug(f"Fetched {len(df)} {interval} bars from MT5")
                    return df
            except Exception as e:
                logger.warning(f"MT5 price fetch failed for {interval}: {e}")

        # Fallback to EODHD
        bars = await self.eodhd.get_historical(
            symbol='BTC-USD.CC',
            interval=interval,
            days=days
        )
        return EODHDClient.bars_to_dataframe(bars)

    async def _fetch_fear_greed(self) -> Dict:
        """Fetch Fear & Greed sentiment data."""
        try:
            return await self.fear_greed.get_sentiment_signal()
        except Exception as e:
            logger.warning(f"Fear & Greed fetch failed: {e}")
            return {
                'current': 50,
                'classification': 'Neutral',
                'is_extreme_fear': False,
                'is_extreme_greed': False,
                'long_favorable': False  # For Bull Bot: extreme fear is favorable
            }

    async def _fetch_btc_market(self) -> Dict:
        """Fetch BTC market metrics."""
        try:
            btc_data = await self.coingecko.get_btc_market()
            if btc_data:
                return {
                    'price_usd': btc_data.price,
                    'market_cap': btc_data.market_cap,
                    'volume_24h': btc_data.volume_24h,
                    'price_change_24h': btc_data.price_change_24h,
                    'price_change_7d': btc_data.price_change_7d,
                    'btc_dominance': btc_data.btc_dominance,
                    'ath_distance_pct': btc_data.ath_distance_pct if hasattr(btc_data, 'ath_distance_pct') else 0,
                }
            return {}
        except Exception as e:
            logger.warning(f"BTC market fetch failed: {e}")
            return {}

    async def _fetch_correlation_data(self, days: int) -> Dict[str, pd.DataFrame]:
        """
        Fetch correlation asset data (DXY, Gold, S&P500).

        Uses MT5 if available (per asset), falls back to EODHD.
        """
        result = {}
        bars_needed = days + 50  # Buffer for daily data

        # Assets to fetch: internal name -> MT5 symbol
        assets = {
            'dxy': 'dxy',
            'gold': 'gold',
            'sp500': 'sp500'
        }

        # Try MT5 first for each asset
        if self._mt5_available and self.mt5:
            for asset_name, mt5_alias in assets.items():
                try:
                    df = await self.mt5.get_historical(mt5_alias, 'D1', bars_needed)
                    if not df.empty:
                        result[asset_name] = df
                        logger.debug(f"Fetched {asset_name} from MT5")
                except Exception as e:
                    logger.debug(f"MT5 {asset_name} fetch failed: {e}")

        # Fetch remaining assets from EODHD
        missing_assets = [a for a in assets if a not in result]
        if missing_assets:
            try:
                eodhd_data = await self.eodhd.get_correlation_assets(days=days)
                for symbol, bars in eodhd_data.items():
                    if symbol not in result:
                        result[symbol] = EODHDClient.bars_to_dataframe(bars)
                        logger.debug(f"Fetched {symbol} from EODHD")
            except Exception as e:
                logger.warning(f"EODHD correlation data fetch failed: {e}")

        return result

    async def _fetch_news_sentiment(self) -> Dict:
        """Fetch news sentiment analysis."""
        try:
            articles = await self.news.get_all_btc_news(limit=30)
            sentiment = self.news.calculate_aggregate_sentiment(articles, hours=24)

            # Include headlines for BAIT informational scoring
            headlines = [a.title for a in articles if hasattr(a, 'title')]
            sentiment['headlines'] = headlines

            return sentiment
        except Exception as e:
            logger.warning(f"News sentiment fetch failed: {e}")
            return {
                'sentiment_score': 0.5,
                'dominant_sentiment': 'NEUTRAL',
                'long_favorable': False,
                'headlines': []
            }

    async def get_quick_snapshot(self) -> Dict:
        """
        Get quick market snapshot (minimal API calls).

        Returns:
            Dictionary with essential metrics
        """
        tasks = {
            'price': self.get_current_price(),
            'fear_greed': self.fear_greed.get_current(),
        }

        results = {}
        for name, coro in tasks.items():
            try:
                results[name] = await coro
            except Exception as e:
                logger.warning(f"Snapshot {name} failed: {e}")
                results[name] = None

        price = results.get('price', 0)
        fng = results.get('fear_greed')

        return {
            'price': price,
            'fear_greed_value': fng.value if fng else 50,
            'fear_greed_class': fng.classification.value if fng else 'Neutral',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    @property
    def cache_stats(self) -> Dict:
        """Get cache statistics."""
        return self.cache.stats
