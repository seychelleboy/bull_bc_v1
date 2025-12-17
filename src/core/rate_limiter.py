"""
Rate Limiter - Thread-safe API rate limiting

Implements token bucket algorithm with sliding window tracking.
Essential for respecting API limits (EODHD: 100K calls/day).
"""

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for a rate limit."""
    calls_per_second: float = 10.0
    calls_per_minute: float = 100.0
    calls_per_day: int = 100000  # EODHD limit
    burst_size: int = 20  # Allow bursts up to this size


class RateLimiter:
    """
    Thread-safe rate limiter using token bucket algorithm.

    Features:
    - Multiple time windows (second, minute, day)
    - Token bucket for burst handling
    - Async and sync support
    - Per-endpoint tracking
    - Automatic retry-after calculation

    Example:
        limiter = RateLimiter(calls_per_second=10, calls_per_day=100000)

        # Sync usage
        limiter.acquire()  # Blocks until allowed
        make_api_call()

        # Async usage
        await limiter.acquire_async()
        await make_async_api_call()
    """

    def __init__(
        self,
        calls_per_second: float = 10.0,
        calls_per_minute: float = 100.0,
        calls_per_day: int = 100000,
        burst_size: int = 20
    ):
        """
        Initialize rate limiter.

        Args:
            calls_per_second: Max calls per second
            calls_per_minute: Max calls per minute
            calls_per_day: Max calls per day (EODHD = 100K)
            burst_size: Maximum burst size allowed
        """
        self._calls_per_second = calls_per_second
        self._calls_per_minute = calls_per_minute
        self._calls_per_day = calls_per_day
        self._burst_size = burst_size

        # Token bucket state
        self._tokens = float(burst_size)
        self._last_refill = time.time()
        self._refill_rate = calls_per_second  # tokens per second

        # Sliding window tracking
        self._second_window: deque = deque()  # timestamps in last second
        self._minute_window: deque = deque()  # timestamps in last minute
        self._day_window: deque = deque()     # timestamps in last day

        # Thread safety
        self._lock = threading.RLock()
        self._async_lock: Optional[asyncio.Lock] = None

        # Statistics
        self._total_calls = 0
        self._throttled_calls = 0
        self._daily_reset_time = self._get_day_start()

    def _get_day_start(self) -> float:
        """Get timestamp of start of current day (UTC)."""
        import datetime
        now = datetime.datetime.utcnow()
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return day_start.timestamp()

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(
            self._burst_size,
            self._tokens + (elapsed * self._refill_rate)
        )
        self._last_refill = now

    def _cleanup_windows(self, now: float) -> None:
        """Remove expired entries from sliding windows."""
        # Clean second window (keep last 1 second)
        second_cutoff = now - 1.0
        while self._second_window and self._second_window[0] < second_cutoff:
            self._second_window.popleft()

        # Clean minute window (keep last 60 seconds)
        minute_cutoff = now - 60.0
        while self._minute_window and self._minute_window[0] < minute_cutoff:
            self._minute_window.popleft()

        # Clean day window (keep last 24 hours)
        day_cutoff = now - 86400.0
        while self._day_window and self._day_window[0] < day_cutoff:
            self._day_window.popleft()

        # Reset daily counter at day boundary
        current_day_start = self._get_day_start()
        if current_day_start > self._daily_reset_time:
            self._day_window.clear()
            self._daily_reset_time = current_day_start
            logger.info("Daily rate limit counter reset")

    def _calculate_wait_time(self, now: float) -> float:
        """
        Calculate how long to wait before next request is allowed.

        Returns:
            Wait time in seconds (0 if request allowed now)
        """
        wait_times = []

        # Check per-second limit
        if len(self._second_window) >= self._calls_per_second:
            oldest = self._second_window[0]
            wait_times.append(oldest + 1.0 - now)

        # Check per-minute limit
        if len(self._minute_window) >= self._calls_per_minute:
            oldest = self._minute_window[0]
            wait_times.append(oldest + 60.0 - now)

        # Check per-day limit
        if len(self._day_window) >= self._calls_per_day:
            # Wait until day resets
            next_day_start = self._daily_reset_time + 86400.0
            wait_times.append(next_day_start - now)
            logger.warning("Daily API limit reached!")

        # Check token bucket
        if self._tokens < 1.0:
            tokens_needed = 1.0 - self._tokens
            wait_times.append(tokens_needed / self._refill_rate)

        return max(0.0, max(wait_times)) if wait_times else 0.0

    def _record_call(self, now: float) -> None:
        """Record a successful API call."""
        self._second_window.append(now)
        self._minute_window.append(now)
        self._day_window.append(now)
        self._tokens -= 1.0
        self._total_calls += 1

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to make an API call (blocking).

        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)

        Returns:
            True if acquired, False if timeout
        """
        start_time = time.time()

        while True:
            with self._lock:
                now = time.time()
                self._refill_tokens()
                self._cleanup_windows(now)

                wait_time = self._calculate_wait_time(now)

                if wait_time <= 0:
                    self._record_call(now)
                    return True

                # Check timeout
                if timeout is not None:
                    elapsed = now - start_time
                    if elapsed + wait_time > timeout:
                        self._throttled_calls += 1
                        return False

            # Wait outside lock
            time.sleep(min(wait_time, 0.1))  # Sleep in small increments

    async def acquire_async(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to make an API call (async).

        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)

        Returns:
            True if acquired, False if timeout
        """
        # Lazy init async lock
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()

        start_time = time.time()

        while True:
            async with self._async_lock:
                with self._lock:
                    now = time.time()
                    self._refill_tokens()
                    self._cleanup_windows(now)

                    wait_time = self._calculate_wait_time(now)

                    if wait_time <= 0:
                        self._record_call(now)
                        return True

                    # Check timeout
                    if timeout is not None:
                        elapsed = now - start_time
                        if elapsed + wait_time > timeout:
                            self._throttled_calls += 1
                            return False

            # Wait outside locks
            await asyncio.sleep(min(wait_time, 0.1))

    def try_acquire(self) -> bool:
        """
        Try to acquire without blocking.

        Returns:
            True if acquired, False if would need to wait
        """
        with self._lock:
            now = time.time()
            self._refill_tokens()
            self._cleanup_windows(now)

            if self._calculate_wait_time(now) <= 0:
                self._record_call(now)
                return True

            self._throttled_calls += 1
            return False

    @property
    def stats(self) -> Dict:
        """Get rate limiter statistics."""
        with self._lock:
            now = time.time()
            self._cleanup_windows(now)

            return {
                'total_calls': self._total_calls,
                'throttled_calls': self._throttled_calls,
                'calls_last_second': len(self._second_window),
                'calls_last_minute': len(self._minute_window),
                'calls_today': len(self._day_window),
                'daily_limit': self._calls_per_day,
                'daily_remaining': self._calls_per_day - len(self._day_window),
                'tokens_available': self._tokens,
                'burst_size': self._burst_size
            }

    @property
    def remaining_daily(self) -> int:
        """Get remaining daily API calls."""
        with self._lock:
            self._cleanup_windows(time.time())
            return self._calls_per_day - len(self._day_window)

    def __repr__(self) -> str:
        stats = self.stats
        return (
            f"RateLimiter(calls_today={stats['calls_today']}, "
            f"remaining={stats['daily_remaining']})"
        )


class MultiEndpointRateLimiter:
    """
    Manages rate limits for multiple API endpoints.

    Example:
        limiter = MultiEndpointRateLimiter()
        limiter.add_endpoint('eodhd', calls_per_second=10, calls_per_day=100000)
        limiter.add_endpoint('coingecko', calls_per_second=5, calls_per_minute=50)

        await limiter.acquire('eodhd')
        await eodhd_api_call()
    """

    def __init__(self):
        """Initialize multi-endpoint rate limiter."""
        self._limiters: Dict[str, RateLimiter] = {}
        self._lock = threading.Lock()

    def add_endpoint(
        self,
        name: str,
        calls_per_second: float = 10.0,
        calls_per_minute: float = 100.0,
        calls_per_day: int = 100000,
        burst_size: int = 20
    ) -> None:
        """
        Add a new endpoint with its rate limits.

        Args:
            name: Endpoint identifier
            calls_per_second: Rate limit per second
            calls_per_minute: Rate limit per minute
            calls_per_day: Rate limit per day
            burst_size: Burst allowance
        """
        with self._lock:
            self._limiters[name] = RateLimiter(
                calls_per_second=calls_per_second,
                calls_per_minute=calls_per_minute,
                calls_per_day=calls_per_day,
                burst_size=burst_size
            )
            logger.info(f"Added rate limiter for endpoint: {name}")

    def acquire(self, endpoint: str, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission for an endpoint (blocking).

        Args:
            endpoint: Endpoint name
            timeout: Maximum wait time

        Returns:
            True if acquired, False if timeout or unknown endpoint
        """
        limiter = self._limiters.get(endpoint)
        if limiter is None:
            logger.warning(f"Unknown endpoint: {endpoint}")
            return True  # Allow if no limiter configured

        return limiter.acquire(timeout)

    async def acquire_async(
        self,
        endpoint: str,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Acquire permission for an endpoint (async).

        Args:
            endpoint: Endpoint name
            timeout: Maximum wait time

        Returns:
            True if acquired, False if timeout or unknown endpoint
        """
        limiter = self._limiters.get(endpoint)
        if limiter is None:
            logger.warning(f"Unknown endpoint: {endpoint}")
            return True

        return await limiter.acquire_async(timeout)

    def get_stats(self, endpoint: Optional[str] = None) -> Dict:
        """
        Get statistics for endpoint(s).

        Args:
            endpoint: Specific endpoint or None for all

        Returns:
            Statistics dictionary
        """
        if endpoint:
            limiter = self._limiters.get(endpoint)
            return limiter.stats if limiter else {}

        return {
            name: limiter.stats
            for name, limiter in self._limiters.items()
        }
