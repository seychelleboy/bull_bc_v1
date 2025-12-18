"""
CoinGecko Client - Free BTC market data API

Provides comprehensive Bitcoin market metrics for analysis.
API: https://api.coingecko.com/api/v3
Rate limit: ~50 calls/minute for free tier
"""

import aiohttp
import logging
from datetime import datetime, timezone
from typing import Dict, Optional
from dataclasses import dataclass

from ..core import ThreadSafeCache

logger = logging.getLogger(__name__)


@dataclass
class BTCMarketData:
    """Bitcoin market data from CoinGecko."""
    price: float
    market_cap: float
    volume_24h: float
    price_change_24h: float
    price_change_7d: float
    ath: float
    ath_change_percentage: float
    btc_dominance: float
    timestamp: datetime

    @property
    def is_near_ath(self) -> bool:
        """Check if price is within 10% of ATH."""
        return self.ath_change_percentage > -10

    @property
    def is_far_from_ath(self) -> bool:
        """Check if price is >30% below ATH (potential value)."""
        return self.ath_change_percentage < -30

    def to_dict(self) -> Dict:
        return {
            'price': self.price,
            'market_cap': self.market_cap,
            'volume_24h': self.volume_24h,
            'price_change_24h': self.price_change_24h,
            'price_change_7d': self.price_change_7d,
            'ath': self.ath,
            'ath_change_percentage': self.ath_change_percentage,
            'btc_dominance': self.btc_dominance,
            'is_near_ath': self.is_near_ath,
            'is_far_from_ath': self.is_far_from_ath,
            'timestamp': self.timestamp.isoformat()
        }


class CoinGeckoClient:
    """
    CoinGecko API Client for BTC market data.

    Provides:
    - Current BTC price and market cap
    - Price changes (24h, 7d)
    - All-time high comparison
    - BTC dominance
    - Volume metrics

    Example:
        client = CoinGeckoClient()
        market = await client.get_btc_market()
        print(f"BTC: ${market.price:,.2f} ({market.price_change_24h:+.2f}%)")
    """

    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(
        self,
        cache: Optional[ThreadSafeCache] = None,
        cache_config: Optional['CacheConfig'] = None
    ):
        """
        Initialize CoinGecko client.

        Args:
            cache: Optional cache for responses
            cache_config: Optional cache configuration with TTL settings
        """
        # Get TTLs from config or use defaults
        self._price_ttl = 30           # Real-time price
        self._fundamentals_ttl = 60    # Market data
        self._global_ttl = 300         # Global metrics
        self._history_ttl = 3600       # Historical data
        self._derivatives_ttl = 300    # Derivatives data

        if cache_config is not None:
            self._price_ttl = getattr(cache_config, 'price_ttl', 30)
            self._fundamentals_ttl = getattr(cache_config, 'fundamentals_ttl', 60)
            self._global_ttl = getattr(cache_config, 'global_data_ttl', 300)
            self._history_ttl = getattr(cache_config, 'bitcoin_history_ttl', 3600)
            self._derivatives_ttl = getattr(cache_config, 'derivatives_ttl', 300)

        default_ttl = self._fundamentals_ttl
        self.cache = cache or ThreadSafeCache(default_ttl=default_ttl)
        self._session: Optional[aiohttp.ClientSession] = None

        logger.debug(
            f"CoinGeckoClient initialized with TTLs: price={self._price_ttl}s, "
            f"fundamentals={self._fundamentals_ttl}s, global={self._global_ttl}s"
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
        cache_key: Optional[str] = None,
        cache_ttl: int = 60
    ) -> Dict:
        """
        Make an API request.

        Args:
            endpoint: API endpoint
            params: Query parameters
            cache_key: Cache key (None to skip cache)
            cache_ttl: Cache TTL in seconds

        Returns:
            JSON response
        """
        # Check cache
        if cache_key:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"CoinGecko cache hit: {cache_key}")
                return cached

        session = await self._get_session()
        url = f"{self.BASE_URL}/{endpoint}"

        try:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()

                if cache_key:
                    self.cache.set(cache_key, data, cache_ttl)

                return data

        except aiohttp.ClientError as e:
            logger.error(f"CoinGecko API error: {e}")
            raise

    async def get_btc_price(self) -> float:
        """
        Get current BTC price in USD.

        Returns:
            Current BTC price
        """
        data = await self._request(
            'simple/price',
            params={'ids': 'bitcoin', 'vs_currencies': 'usd'},
            cache_key='btc_price',
            cache_ttl=self._price_ttl
        )
        return data['bitcoin']['usd']

    async def get_btc_market(self) -> BTCMarketData:
        """
        Get comprehensive BTC market data.

        Returns:
            BTCMarketData with all metrics
        """
        # Get coin data
        coin_data = await self._request(
            'coins/bitcoin',
            params={
                'localization': 'false',
                'tickers': 'false',
                'community_data': 'false',
                'developer_data': 'false'
            },
            cache_key='btc_coin',
            cache_ttl=self._fundamentals_ttl
        )

        # Get global data for dominance
        global_data = await self._request(
            'global',
            cache_key='global',
            cache_ttl=self._global_ttl
        )

        market = coin_data['market_data']

        return BTCMarketData(
            price=market['current_price']['usd'],
            market_cap=market['market_cap']['usd'],
            volume_24h=market['total_volume']['usd'],
            price_change_24h=market['price_change_percentage_24h'] or 0,
            price_change_7d=market['price_change_percentage_7d'] or 0,
            ath=market['ath']['usd'],
            ath_change_percentage=market['ath_change_percentage']['usd'] or 0,
            btc_dominance=global_data['data']['market_cap_percentage']['btc'],
            timestamp=datetime.now(timezone.utc)
        )

    async def get_btc_market_dict(self) -> Dict:
        """
        Get BTC market data as dictionary.

        Returns:
            Dictionary with market metrics
        """
        market = await self.get_btc_market()
        return market.to_dict()

    async def get_global_metrics(self) -> Dict:
        """
        Get global cryptocurrency market metrics.

        Returns:
            Dictionary with global metrics
        """
        data = await self._request('global', cache_key='global', cache_ttl=self._global_ttl)

        return {
            'total_market_cap': data['data']['total_market_cap']['usd'],
            'total_volume_24h': data['data']['total_volume']['usd'],
            'btc_dominance': data['data']['market_cap_percentage']['btc'],
            'eth_dominance': data['data']['market_cap_percentage'].get('eth', 0),
            'market_cap_change_24h': data['data']['market_cap_change_percentage_24h_usd'],
            'active_cryptocurrencies': data['data']['active_cryptocurrencies'],
            'markets': data['data']['markets']
        }

    async def get_btc_history(self, days: int = 30) -> Dict:
        """
        Get BTC price history.

        Args:
            days: Number of days of history

        Returns:
            Dictionary with price history
        """
        data = await self._request(
            'coins/bitcoin/market_chart',
            params={'vs_currency': 'usd', 'days': days},
            cache_key=f'btc_history_{days}',
            cache_ttl=self._history_ttl
        )

        return {
            'prices': data['prices'],  # [[timestamp, price], ...]
            'market_caps': data['market_caps'],
            'total_volumes': data['total_volumes']
        }

    async def get_derivatives_data(self) -> Dict:
        """
        Get derivatives exchange data (open interest, funding).

        Returns:
            Dictionary with derivatives metrics
        """
        try:
            data = await self._request(
                'derivatives/exchanges',
                params={'per_page': 10},
                cache_key='derivatives',
                cache_ttl=self._derivatives_ttl
            )

            # Sum open interest across top exchanges
            total_oi = sum(
                ex.get('open_interest_btc', 0) or 0
                for ex in data
            )

            return {
                'total_open_interest_btc': total_oi,
                'exchanges': len(data),
                'data': data[:5]  # Top 5 exchanges
            }
        except Exception as e:
            logger.warning(f"Failed to get derivatives data: {e}")
            return {}
