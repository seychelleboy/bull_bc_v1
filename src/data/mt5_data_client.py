"""
MT5 Data Client - Data fetching interface using MT5 connector.

This client wraps the MT5Connector to provide a consistent interface
for the DataAggregator, matching the EODHD client interface for easy swapping.

Usage:
    client = MT5DataClient(config)
    await client.initialize()

    # Fetch BTC data
    df = await client.get_historical('BTCUSD_SB', '4h', 100)

    # Get current price
    price = await client.get_current_price()
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import pandas as pd

from ..execution.mt5_connector import MT5Connector
from ..core.thread_safe_cache import ThreadSafeCache

logger = logging.getLogger(__name__)


class MT5DataClient:
    """
    MT5 Data Client for fetching market data.

    Provides a consistent interface matching EODHDClient for seamless
    integration with the DataAggregator.
    """

    def __init__(self, config=None):
        """
        Initialize MT5 data client.

        Args:
            config: Configuration object with MT5 settings
        """
        self.config = config
        self._connector: Optional[MT5Connector] = None
        self._initialized = False

        # Get TTLs from config or use defaults
        self._quote_ttl = 5         # Real-time quotes
        self._price_ttl = 30        # Current price
        self._historical_ttl = 60   # Historical bars

        if config and hasattr(config, 'cache'):
            self._quote_ttl = getattr(config.cache, 'quote_ttl', 5)
            self._price_ttl = getattr(config.cache, 'price_ttl', 30)
            self._historical_ttl = getattr(config.cache, 'intraday_ttl', 60)

        self._cache = ThreadSafeCache(default_ttl=self._price_ttl)

        logger.debug(
            f"MT5DataClient initialized with TTLs: quote={self._quote_ttl}s, "
            f"price={self._price_ttl}s, historical={self._historical_ttl}s"
        )

        # Symbol mapping
        self._symbols = {
            'btc': 'BTCUSD_SB',
            'gold': 'XAUUSD_SB',
            'sp500': 'US500_SB',
            'dxy': 'USDX_SB',
            'nasdaq': 'NAS100_SB',
        }

        # Override from config if available
        if config and hasattr(config, 'mt5'):
            self._symbols['btc'] = getattr(config.mt5, 'btc_symbol', 'BTCUSD_SB')
            self._symbols['gold'] = getattr(config.mt5, 'gold_symbol', 'XAUUSD_SB')
            self._symbols['sp500'] = getattr(config.mt5, 'sp500_symbol', 'US500_SB')
            self._symbols['dxy'] = getattr(config.mt5, 'dxy_symbol', 'USDX_SB')
            self._symbols['nasdaq'] = getattr(config.mt5, 'nasdaq_symbol', 'NAS100_SB')

    async def initialize(self) -> bool:
        """
        Initialize the MT5 connection.

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        try:
            self._connector = MT5Connector(self.config)
            result = await self._connector.initialize()
            self._initialized = result

            if result:
                logger.info("MT5DataClient initialized successfully")
            else:
                logger.error("MT5DataClient initialization failed")

            return result

        except Exception as e:
            logger.error(f"MT5DataClient initialization error: {e}")
            return False

    async def close(self) -> None:
        """Close the MT5 connection."""
        if self._connector:
            await self._connector.shutdown()
            self._initialized = False
            logger.info("MT5DataClient closed")

    def _resolve_symbol(self, symbol: str) -> str:
        """
        Resolve symbol alias to MT5 symbol.

        Args:
            symbol: Symbol alias (e.g., 'BTC-USD', 'btc') or MT5 symbol

        Returns:
            MT5 symbol name
        """
        # Check if it's an alias
        symbol_lower = symbol.lower().replace('-', '').replace('/', '').replace('.cc', '')

        if 'btc' in symbol_lower or 'bitcoin' in symbol_lower:
            return self._symbols['btc']
        elif 'xau' in symbol_lower or 'gold' in symbol_lower:
            return self._symbols['gold']
        elif 'sp500' in symbol_lower or 'us500' in symbol_lower or 'spx' in symbol_lower:
            return self._symbols['sp500']
        elif 'dxy' in symbol_lower or 'usdx' in symbol_lower or 'dollar' in symbol_lower:
            return self._symbols['dxy']
        elif 'nas' in symbol_lower or 'tech' in symbol_lower:
            return self._symbols['nasdaq']

        # Return as-is if no alias match
        return symbol

    def _timeframe_to_mt5(self, timeframe: str) -> str:
        """
        Convert common timeframe strings to MT5 format.

        Args:
            timeframe: Timeframe string (e.g., '4h', '1d', 'H4')

        Returns:
            MT5 compatible timeframe string
        """
        mapping = {
            '1m': 'M1', '5m': 'M5', '15m': 'M15', '30m': 'M30',
            '1h': 'H1', '4h': 'H4',
            'd': 'D1', '1d': 'D1',
            'w': 'W1', '1w': 'W1',
        }
        return mapping.get(timeframe.lower(), timeframe)

    async def get_historical(
        self,
        symbol: str,
        timeframe: str,
        bars: int = 100
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data.

        Args:
            symbol: Symbol (e.g., 'BTC-USD', 'BTCUSD_SB')
            timeframe: Timeframe (e.g., '4h', 'H4', '1d')
            bars: Number of bars to fetch

        Returns:
            DataFrame with datetime, open, high, low, close, volume columns
        """
        if not self._initialized:
            if not await self.initialize():
                return pd.DataFrame()

        # Resolve symbol and timeframe
        mt5_symbol = self._resolve_symbol(symbol)
        mt5_tf = self._timeframe_to_mt5(timeframe)

        # Check cache
        cache_key = f"historical:{mt5_symbol}:{mt5_tf}:{bars}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            df = await self._connector.get_bars(mt5_symbol, mt5_tf, bars)

            if not df.empty:
                # Cache using configured TTL
                self._cache.set(cache_key, df, ttl=self._historical_ttl)

            return df

        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            return pd.DataFrame()

    async def get_quote(self, symbol: str) -> Dict:
        """
        Get current quote for a symbol.

        Args:
            symbol: Symbol (e.g., 'BTC-USD', 'BTCUSD_SB')

        Returns:
            Dictionary with quote data (close, open, high, low, volume, timestamp)
        """
        if not self._initialized:
            if not await self.initialize():
                return {}

        mt5_symbol = self._resolve_symbol(symbol)

        # Check cache
        cache_key = f"quote:{mt5_symbol}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            tick = await self._connector.get_tick(mt5_symbol)

            if not tick:
                return {}

            # Format to match EODHD response format
            result = {
                'close': tick.get('bid', 0),
                'open': tick.get('bid', 0),
                'high': tick.get('ask', 0),
                'low': tick.get('bid', 0),
                'volume': tick.get('volume', 0),
                'timestamp': tick.get('time', datetime.now(timezone.utc)).isoformat(),
                'bid': tick.get('bid', 0),
                'ask': tick.get('ask', 0),
                'spread': tick.get('spread', 0),
            }

            # Cache using configured TTL
            self._cache.set(cache_key, result, ttl=self._quote_ttl)
            return result

        except Exception as e:
            logger.error(f"Failed to get quote: {e}")
            return {}

    async def get_current_price(self, symbol: str = 'BTC-USD') -> float:
        """
        Get current price for a symbol.

        Args:
            symbol: Symbol (defaults to BTC)

        Returns:
            Current price (bid)
        """
        quote = await self.get_quote(symbol)
        return quote.get('close', 0.0)

    async def get_intraday(
        self,
        symbol: str,
        interval: str = '1h',
        bars: int = 100
    ) -> pd.DataFrame:
        """
        Get intraday data (alias for get_historical with intraday timeframes).

        Args:
            symbol: Symbol
            interval: Interval (e.g., '1m', '5m', '1h')
            bars: Number of bars

        Returns:
            DataFrame with OHLCV data
        """
        return await self.get_historical(symbol, interval, bars)

    async def get_correlation_assets(self, days: int = 90) -> Dict[str, pd.DataFrame]:
        """
        Get correlation asset data (DXY, Gold, S&P500).

        Args:
            days: Number of days of history

        Returns:
            Dictionary mapping asset name to DataFrame
        """
        if not self._initialized:
            if not await self.initialize():
                return {}

        # Calculate approximate bars needed for daily data
        bars = days + 10  # Add buffer

        result = {}

        # Fetch all correlation assets in parallel
        tasks = {
            'dxy': self.get_historical(self._symbols['dxy'], 'D1', bars),
            'gold': self.get_historical(self._symbols['gold'], 'D1', bars),
            'sp500': self.get_historical(self._symbols['sp500'], 'D1', bars),
        }

        for asset, task in tasks.items():
            try:
                df = await task
                if not df.empty:
                    result[asset] = df
            except Exception as e:
                logger.warning(f"Failed to fetch {asset} data: {e}")

        return result

    async def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols.

        Returns:
            List of symbol names
        """
        if not self._initialized:
            if not await self.initialize():
                return []

        return await self._connector.get_available_symbols("*")

    async def check_connection(self) -> bool:
        """
        Check if connection is alive.

        Returns:
            True if connected
        """
        if not self._connector:
            return False
        return await self._connector.check_connection()

    @property
    def symbols(self) -> Dict[str, str]:
        """Get symbol mapping."""
        return self._symbols.copy()
