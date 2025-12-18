"""
EODHD Client - REST and WebSocket client for EODHD All-In-One API

Provides access to:
- Historical OHLCV data
- Intraday data (1m, 5m, 1h intervals)
- Real-time quotes via WebSocket
- Financial news
- Fundamentals data

Rate limit: 100,000 API calls/day
"""

import asyncio
import aiohttp
import websockets
import json
import logging
from datetime import datetime, date, timezone, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd

from ..core import ThreadSafeCache, RateLimiter

logger = logging.getLogger(__name__)


class Interval(Enum):
    """EODHD supported intervals."""
    MINUTE_1 = '1m'
    MINUTE_5 = '5m'
    HOUR_1 = '1h'
    HOUR_4 = '4h'
    DAILY = 'd'


@dataclass
class OHLCVBar:
    """Single OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""
    def __init__(self, message: str, retry_after: int = 60):
        self.retry_after = retry_after
        super().__init__(message)


class EODHDClient:
    """
    EODHD REST API Client.

    Example:
        client = EODHDClient(api_key='your_key')
        bars = await client.get_historical('BTC-USD.CC', interval='1h', days=30)
        df = client.bars_to_dataframe(bars)
    """

    BASE_URL = "https://eodhd.com/api"

    def __init__(
        self,
        api_key: str,
        rate_limiter: Optional[RateLimiter] = None,
        cache: Optional[ThreadSafeCache] = None,
        cache_config: Optional['CacheConfig'] = None
    ):
        """
        Initialize EODHD client.

        Args:
            api_key: EODHD API key
            rate_limiter: Optional rate limiter (creates default if not provided)
            cache: Optional cache for responses
            cache_config: Optional cache configuration with TTL settings
        """
        self.api_key = api_key
        self.rate_limiter = rate_limiter or RateLimiter(
            calls_per_second=5,
            calls_per_minute=100,
            calls_per_day=100000,
            burst_size=10
        )

        # Get TTLs from config or use defaults
        self._quote_ttl = 5            # Real-time quotes
        self._intraday_ttl = 60        # Intraday data
        self._history_ttl = 300        # Historical data
        self._news_ttl = 300           # News articles

        if cache_config is not None:
            self._quote_ttl = getattr(cache_config, 'quote_ttl', 5)
            self._intraday_ttl = getattr(cache_config, 'intraday_ttl', 60)
            self._history_ttl = getattr(cache_config, 'bitcoin_history_ttl', 300)
            self._news_ttl = getattr(cache_config, 'news_ttl', 300)

        default_ttl = self._intraday_ttl
        self.cache = cache or ThreadSafeCache(default_ttl=default_ttl)
        self._session: Optional[aiohttp.ClientSession] = None

        logger.debug(
            f"EODHDClient initialized with TTLs: quote={self._quote_ttl}s, "
            f"intraday={self._intraday_ttl}s, history={self._history_ttl}s, news={self._news_ttl}s"
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        use_cache: bool = True,
        cache_ttl: Optional[int] = None
    ) -> Any:
        """
        Make an API request.

        Args:
            endpoint: API endpoint
            params: Query parameters
            use_cache: Whether to use cache
            cache_ttl: Custom cache TTL

        Returns:
            JSON response
        """
        params = params or {}
        params['api_token'] = self.api_key
        params['fmt'] = 'json'

        url = f"{self.BASE_URL}/{endpoint}"
        cache_key = f"{endpoint}:{json.dumps(params, sort_keys=True)}"

        # Check cache
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for {endpoint}")
                return cached

        # Rate limit
        await self.rate_limiter.acquire_async()

        session = await self._get_session()

        try:
            async with session.get(url, params=params) as response:
                if response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited. Retry after {retry_after}s")
                    raise RateLimitError(f"Rate limited", retry_after)

                response.raise_for_status()
                data = await response.json()

                # Cache successful response
                if use_cache:
                    self.cache.set(cache_key, data, cache_ttl)

                return data

        except aiohttp.ClientError as e:
            logger.error(f"EODHD API error: {e}")
            raise

    # =========================================================================
    # HISTORICAL DATA
    # =========================================================================

    async def get_historical(
        self,
        symbol: str = 'BTC-USD.CC',
        interval: str = 'd',
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        days: Optional[int] = None
    ) -> List[OHLCVBar]:
        """
        Get historical OHLCV data.

        Args:
            symbol: Trading symbol (default: BTC-USD.CC)
            interval: Data interval ('d', '1h', '4h', '5m', '1m')
            start_date: Start date
            end_date: End date
            days: Number of days back from today (alternative to start_date)

        Returns:
            List of OHLCVBar objects
        """
        # Calculate dates
        if days is not None:
            end_date = date.today()
            start_date = end_date - timedelta(days=days)
        elif start_date is None:
            start_date = date.today() - timedelta(days=365)
        if end_date is None:
            end_date = date.today()

        params = {
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'period': interval
        }

        # Different endpoint for intraday vs daily
        if interval in ['1m', '5m', '1h', '4h']:
            endpoint = f"intraday/{symbol}"
            params['interval'] = interval
            del params['period']
        else:
            endpoint = f"eod/{symbol}"

        data = await self._request(endpoint, params, cache_ttl=self._history_ttl)

        # Parse response
        bars = []
        if not isinstance(data, list):
            return bars

        for item in data:
            try:
                # Handle different timestamp formats
                if 'datetime' in item:
                    ts = datetime.fromisoformat(item['datetime'].replace('Z', '+00:00'))
                elif 'date' in item:
                    ts = datetime.strptime(item['date'], '%Y-%m-%d')
                    ts = ts.replace(tzinfo=timezone.utc)
                else:
                    continue

                bar = OHLCVBar(
                    timestamp=ts,
                    open=float(item.get('open') or 0),
                    high=float(item.get('high') or 0),
                    low=float(item.get('low') or 0),
                    close=float(item.get('close') or 0),
                    volume=float(item.get('volume') or 0)
                )
                bars.append(bar)
            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to parse bar: {e}")
                continue

        logger.info(f"Fetched {len(bars)} bars for {symbol} ({interval})")
        return bars

    async def get_intraday(
        self,
        symbol: str = 'BTC-USD.CC',
        interval: str = '1h',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[OHLCVBar]:
        """
        Get intraday data.

        Args:
            symbol: Trading symbol
            interval: '1m', '5m', '1h', or '4h'
            start_date: Start datetime
            end_date: End datetime

        Returns:
            List of OHLCVBar objects
        """
        params = {'interval': interval}

        if start_date:
            params['from'] = int(start_date.timestamp())
        if end_date:
            params['to'] = int(end_date.timestamp())

        data = await self._request(f"intraday/{symbol}", params, cache_ttl=self._intraday_ttl)

        bars = []
        if not isinstance(data, list):
            return bars

        for item in data:
            try:
                if 'datetime' in item:
                    ts = datetime.fromisoformat(item['datetime'].replace('Z', '+00:00'))
                elif 'timestamp' in item:
                    ts = datetime.fromtimestamp(item['timestamp'], tz=timezone.utc)
                else:
                    continue

                bar = OHLCVBar(
                    timestamp=ts,
                    open=float(item.get('open') or 0),
                    high=float(item.get('high') or 0),
                    low=float(item.get('low') or 0),
                    close=float(item.get('close') or 0),
                    volume=float(item.get('volume') or 0)
                )
                bars.append(bar)
            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to parse intraday bar: {e}")
                continue

        return bars

    # =========================================================================
    # REAL-TIME DATA
    # =========================================================================

    async def get_quote(self, symbol: str = 'BTC-USD.CC') -> Dict:
        """
        Get real-time quote.

        Args:
            symbol: Trading symbol

        Returns:
            Quote dictionary with price, change, volume, etc.
        """
        data = await self._request(f"real-time/{symbol}", cache_ttl=self._quote_ttl)
        return data if isinstance(data, dict) else {}

    async def get_quotes_bulk(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get quotes for multiple symbols.

        Args:
            symbols: List of symbols

        Returns:
            Dictionary of symbol -> quote data
        """
        symbol_str = ','.join(symbols)
        data = await self._request(f"real-time/{symbol_str}", cache_ttl=self._quote_ttl)

        # Handle single vs multiple response
        if isinstance(data, dict):
            return {symbols[0]: data}

        if isinstance(data, list):
            return {item.get('code', ''): item for item in data}

        return {}

    # =========================================================================
    # NEWS
    # =========================================================================

    async def get_news(
        self,
        symbol: str = 'BTC-USD.CC',
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict]:
        """
        Get financial news for a symbol.

        Args:
            symbol: Trading symbol
            limit: Number of news items
            offset: Pagination offset

        Returns:
            List of news articles
        """
        params = {
            's': symbol,
            'limit': limit,
            'offset': offset
        }
        data = await self._request('news', params, cache_ttl=self._news_ttl)
        return data if isinstance(data, list) else []

    async def get_crypto_news(self, limit: int = 50) -> List[Dict]:
        """
        Get cryptocurrency news (all crypto).

        Args:
            limit: Number of news items

        Returns:
            List of news articles
        """
        params = {
            't': 'crypto',
            'limit': limit
        }
        data = await self._request('news', params, cache_ttl=self._news_ttl)
        return data if isinstance(data, list) else []

    # =========================================================================
    # CORRELATION DATA (DXY, Gold, S&P500)
    # =========================================================================

    async def get_correlation_assets(
        self,
        days: int = 30
    ) -> Dict[str, List[OHLCVBar]]:
        """
        Get data for correlation analysis assets.

        Returns:
            Dictionary with 'dxy', 'gold', 'sp500' data
        """
        tasks = [
            self.get_historical('DX-Y.NYB', interval='d', days=days),  # DXY
            self.get_historical('GC.COMEX', interval='d', days=days),  # Gold
            self.get_historical('GSPC.INDX', interval='d', days=days),  # S&P 500
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        data = {}
        symbols = ['dxy', 'gold', 'sp500']

        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to fetch {symbol}: {result}")
                data[symbol] = []
            else:
                data[symbol] = result

        return data

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def bars_to_dataframe(bars: List[OHLCVBar]) -> pd.DataFrame:
        """
        Convert bars to pandas DataFrame.

        Args:
            bars: List of OHLCVBar objects

        Returns:
            DataFrame with OHLCV data, indexed by timestamp
        """
        if not bars:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        df = pd.DataFrame([bar.to_dict() for bar in bars])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df


class EODHDWebSocket:
    """
    EODHD WebSocket client for real-time data streaming.

    Example:
        ws = EODHDWebSocket(api_key='your_key')
        ws.on_quote = lambda data: print(f"Quote: {data}")
        await ws.connect(['BTC-USD'])
    """

    WS_URL = "wss://ws.eodhistoricaldata.com/ws/crypto"

    def __init__(self, api_key: str):
        """
        Initialize WebSocket client.

        Args:
            api_key: EODHD API key
        """
        self.api_key = api_key
        self._ws = None
        self._running = False
        self._subscribed_symbols: List[str] = []

        # Callbacks
        self.on_quote: Optional[Callable[[Dict], None]] = None
        self.on_trade: Optional[Callable[[Dict], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        self.on_connect: Optional[Callable[[], None]] = None
        self.on_disconnect: Optional[Callable[[], None]] = None

    async def connect(self, symbols: List[str]) -> None:
        """
        Connect and subscribe to symbols.

        Args:
            symbols: List of symbols to subscribe to (e.g., ['BTC-USD'])
        """
        self._running = True
        self._subscribed_symbols = symbols

        url = f"{self.WS_URL}?api_token={self.api_key}"

        try:
            async with websockets.connect(url) as ws:
                self._ws = ws
                logger.info("WebSocket connected")

                if self.on_connect:
                    self.on_connect()

                # Subscribe to symbols
                for symbol in symbols:
                    subscribe_msg = {
                        "action": "subscribe",
                        "symbols": symbol
                    }
                    await ws.send(json.dumps(subscribe_msg))
                    logger.info(f"Subscribed to {symbol}")

                # Listen for messages
                while self._running:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=30)
                        await self._handle_message(message)
                    except asyncio.TimeoutError:
                        # Send ping to keep alive
                        await ws.ping()
                    except websockets.ConnectionClosed:
                        logger.warning("WebSocket connection closed")
                        break

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            if self.on_error:
                self.on_error(e)
        finally:
            self._ws = None
            if self.on_disconnect:
                self.on_disconnect()

    async def _handle_message(self, message: str) -> None:
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)

            # Determine message type
            msg_type = data.get('type', data.get('t', 'unknown'))

            if msg_type == 'quote' or 'p' in data:  # Quote update
                if self.on_quote:
                    self.on_quote(data)
            elif msg_type == 'trade':
                if self.on_trade:
                    self.on_trade(data)
            else:
                logger.debug(f"Unknown message type: {msg_type}")

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse WebSocket message: {message}")

    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
            logger.info("WebSocket disconnected")

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._ws is not None and self._running
