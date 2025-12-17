"""
Fear & Greed Index Client - Alternative.me API

The Fear & Greed Index is a market sentiment indicator that measures
investor emotions on a scale of 0 (Extreme Fear) to 100 (Extreme Greed).

For LONG signals:
- Extreme Fear (0-25) = Potential buying opportunity
- Extreme Greed (75-100) = Caution, market may be overheated

API: https://alternative.me/crypto/fear-and-greed-index/
Rate limit: No official limit, but recommended ~1 request per minute
"""

import aiohttp
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from ..core import ThreadSafeCache

logger = logging.getLogger(__name__)


class FearGreedLevel(Enum):
    """Fear & Greed index classification."""
    EXTREME_FEAR = 'Extreme Fear'
    FEAR = 'Fear'
    NEUTRAL = 'Neutral'
    GREED = 'Greed'
    EXTREME_GREED = 'Extreme Greed'


@dataclass
class FearGreedReading:
    """Single Fear & Greed reading."""
    value: int  # 0-100
    classification: FearGreedLevel
    timestamp: datetime

    @property
    def is_extreme_fear(self) -> bool:
        """Check if market is in extreme fear (bullish signal for longs)."""
        return self.value <= 25

    @property
    def is_extreme_greed(self) -> bool:
        """Check if market is in extreme greed (potential top - caution for longs)."""
        return self.value >= 75

    @property
    def is_neutral(self) -> bool:
        """Check if market is neutral."""
        return 40 <= self.value <= 60

    @property
    def is_fear_zone(self) -> bool:
        """Check if market is in fear zone (good for LONG entries)."""
        return self.value <= 40

    def to_dict(self) -> Dict:
        return {
            'value': self.value,
            'classification': self.classification.value,
            'timestamp': self.timestamp.isoformat(),
            'is_extreme_fear': self.is_extreme_fear,
            'is_fear_zone': self.is_fear_zone
        }


class FearGreedClient:
    """
    Fear & Greed Index API Client.

    For Bull BC1 (LONG signals):
    - Extreme fear often precedes rallies (good for longs)
    - Fear zone (0-40) is generally favorable for entries

    Example:
        client = FearGreedClient()
        current = await client.get_current()
        print(f"Fear & Greed: {current.value} ({current.classification.value})")

        if current.is_extreme_fear:
            print("Potential buying opportunity!")
    """

    BASE_URL = "https://api.alternative.me/fng/"

    def __init__(self, cache: Optional[ThreadSafeCache] = None):
        """
        Initialize Fear & Greed client.

        Args:
            cache: Optional cache for responses
        """
        self.cache = cache or ThreadSafeCache(default_ttl=300)  # 5 min cache
        self._session: Optional[aiohttp.ClientSession] = None

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

    @staticmethod
    def _classify_value(value: int) -> FearGreedLevel:
        """Classify a Fear & Greed value."""
        if value <= 25:
            return FearGreedLevel.EXTREME_FEAR
        elif value <= 45:
            return FearGreedLevel.FEAR
        elif value <= 55:
            return FearGreedLevel.NEUTRAL
        elif value <= 75:
            return FearGreedLevel.GREED
        else:
            return FearGreedLevel.EXTREME_GREED

    async def _request(
        self,
        params: Optional[Dict] = None,
        use_cache: bool = True
    ) -> Dict:
        """
        Make an API request.

        Args:
            params: Query parameters
            use_cache: Whether to use cache

        Returns:
            JSON response
        """
        params = params or {}
        cache_key = f"fng:{params.get('limit', 1)}"

        # Check cache
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug("Fear & Greed cache hit")
                return cached

        session = await self._get_session()

        try:
            async with session.get(self.BASE_URL, params=params) as response:
                response.raise_for_status()
                data = await response.json()

                # Validate response
                if data.get('metadata', {}).get('error'):
                    raise ValueError(data['metadata']['error'])

                if use_cache:
                    self.cache.set(cache_key, data)

                return data

        except aiohttp.ClientError as e:
            logger.error(f"Fear & Greed API error: {e}")
            raise

    async def get_current(self) -> FearGreedReading:
        """
        Get current Fear & Greed index.

        Returns:
            Current FearGreedReading
        """
        data = await self._request({'limit': 1})
        item = data['data'][0]

        return FearGreedReading(
            value=int(item['value']),
            classification=self._classify_value(int(item['value'])),
            timestamp=datetime.fromtimestamp(int(item['timestamp']), tz=timezone.utc)
        )

    async def get_historical(self, days: int = 30) -> List[FearGreedReading]:
        """
        Get historical Fear & Greed data.

        Args:
            days: Number of days of history

        Returns:
            List of FearGreedReading objects (most recent first)
        """
        data = await self._request({'limit': days}, use_cache=True)

        readings = []
        for item in data.get('data', []):
            try:
                reading = FearGreedReading(
                    value=int(item['value']),
                    classification=self._classify_value(int(item['value'])),
                    timestamp=datetime.fromtimestamp(int(item['timestamp']), tz=timezone.utc)
                )
                readings.append(reading)
            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to parse Fear & Greed data: {e}")
                continue

        logger.info(f"Fetched {len(readings)} days of Fear & Greed data")
        return readings

    async def get_average(self, days: int = 7) -> float:
        """
        Get average Fear & Greed over a period.

        Args:
            days: Number of days to average

        Returns:
            Average Fear & Greed value
        """
        readings = await self.get_historical(days)
        if not readings:
            return 50.0  # Default neutral

        return sum(r.value for r in readings) / len(readings)

    async def get_trend(self, days: int = 7) -> str:
        """
        Determine Fear & Greed trend direction.

        Args:
            days: Number of days to analyze

        Returns:
            'INCREASING', 'DECREASING', or 'STABLE'
        """
        readings = await self.get_historical(days)
        if len(readings) < 2:
            return 'STABLE'

        # Compare first half avg to second half avg
        mid = len(readings) // 2
        recent_avg = sum(r.value for r in readings[:mid]) / mid
        older_avg = sum(r.value for r in readings[mid:]) / (len(readings) - mid)

        diff = recent_avg - older_avg
        if diff > 5:
            return 'INCREASING'  # Moving toward greed
        elif diff < -5:
            return 'DECREASING'  # Moving toward fear
        else:
            return 'STABLE'

    async def get_sentiment_signal(self) -> Dict:
        """
        Get comprehensive sentiment analysis for LONG signals.

        Returns:
            Dictionary with sentiment metrics for trading decisions
        """
        current = await self.get_current()
        history = await self.get_historical(30)

        # Calculate metrics
        avg_7d = sum(r.value for r in history[:7]) / min(7, len(history)) if history else 50
        avg_30d = sum(r.value for r in history) / len(history) if history else 50

        # Determine if conditions favor longs
        # Extreme fear often precedes rallies (good for longs)
        # Fear zone recovery is bullish
        long_favorable = current.is_extreme_fear or (current.value < 40 and avg_7d < avg_30d)

        return {
            'current': current.value,
            'classification': current.classification.value,
            'avg_7d': round(avg_7d, 1),
            'avg_30d': round(avg_30d, 1),
            'trend': await self.get_trend(),
            'is_extreme_fear': current.is_extreme_fear,
            'is_extreme_greed': current.is_extreme_greed,
            'is_fear_zone': current.is_fear_zone,
            'long_favorable': long_favorable,
            'timestamp': current.timestamp.isoformat()
        }
