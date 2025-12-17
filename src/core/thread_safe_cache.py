"""
Thread-Safe TTL Cache - Race-compliant caching for concurrent access

Provides a thread-safe cache with TTL (time-to-live) expiration.
Essential for caching API responses in a multi-threaded environment.
"""

import threading
import time
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with value and expiration time."""
    value: Any
    expires_at: float
    created_at: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        return time.time() > self.expires_at


class ThreadSafeCache:
    """
    Thread-safe cache with TTL expiration.

    Features:
    - Thread-safe read/write operations using RLock
    - Configurable TTL per entry or default
    - Automatic cleanup of expired entries
    - Statistics tracking for monitoring

    Example:
        cache = ThreadSafeCache(default_ttl=300)  # 5 min default
        cache.set('btc_price', 42000.0)
        price = cache.get('btc_price')  # Returns 42000.0 or None if expired
    """

    def __init__(self, default_ttl: int = 300, max_size: int = 1000):
        """
        Initialize the cache.

        Args:
            default_ttl: Default time-to-live in seconds (default: 300 = 5 min)
            max_size: Maximum number of entries before cleanup (default: 1000)
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._default_ttl = default_ttl
        self._max_size = max_size

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from cache.

        Args:
            key: Cache key
            default: Default value if key not found or expired

        Returns:
            Cached value or default
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return default

            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                self._evictions += 1
                return default

            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override (uses default if not specified)
        """
        ttl = ttl if ttl is not None else self._default_ttl
        expires_at = time.time() + ttl

        with self._lock:
            # Cleanup if at max size
            if len(self._cache) >= self._max_size:
                self._cleanup_expired()

                # If still at max, remove oldest entries
                if len(self._cache) >= self._max_size:
                    self._evict_oldest(self._max_size // 4)

            self._cache[key] = CacheEntry(value=value, expires_at=expires_at)

    def delete(self, key: str) -> bool:
        """
        Delete a key from cache.

        Args:
            key: Cache key to delete

        Returns:
            True if key existed and was deleted, False otherwise
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[int] = None
    ) -> Any:
        """
        Get value from cache or compute and store it.

        Thread-safe: Only one thread will compute the value if cache miss.

        Args:
            key: Cache key
            factory: Callable that produces the value if not cached
            ttl: Optional TTL override

        Returns:
            Cached or newly computed value
        """
        # First check without computing
        value = self.get(key)
        if value is not None:
            return value

        # Need to compute - acquire lock
        with self._lock:
            # Double-check after acquiring lock
            value = self.get(key)
            if value is not None:
                return value

            # Compute and store
            value = factory()
            self.set(key, value, ttl)
            return value

    async def get_or_set_async(
        self,
        key: str,
        factory: Callable,
        ttl: Optional[int] = None
    ) -> Any:
        """
        Async version of get_or_set.

        Args:
            key: Cache key
            factory: Async callable that produces the value
            ttl: Optional TTL override

        Returns:
            Cached or newly computed value
        """
        value = self.get(key)
        if value is not None:
            return value

        # Compute value (outside lock for async)
        value = await factory()

        with self._lock:
            # Check again in case another task computed it
            existing = self.get(key)
            if existing is not None:
                return existing

            self.set(key, value, ttl)
            return value

    def clear(self) -> int:
        """
        Clear all entries from cache.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def _cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        now = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if now > entry.expires_at
        ]

        for key in expired_keys:
            del self._cache[key]
            self._evictions += 1

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

        return len(expired_keys)

    def _evict_oldest(self, count: int) -> int:
        """
        Evict oldest entries by creation time.

        Args:
            count: Number of entries to evict

        Returns:
            Number of entries evicted
        """
        if not self._cache:
            return 0

        # Sort by creation time
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].created_at
        )

        evicted = 0
        for key, _ in sorted_entries[:count]:
            del self._cache[key]
            evicted += 1
            self._evictions += 1

        logger.debug(f"Evicted {evicted} oldest cache entries")
        return evicted

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

            return {
                'size': len(self._cache),
                'max_size': self._max_size,
                'hits': self._hits,
                'misses': self._misses,
                'evictions': self._evictions,
                'hit_rate': f"{hit_rate:.1f}%",
                'default_ttl': self._default_ttl
            }

    def __len__(self) -> int:
        """Return number of entries in cache."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return self.get(key) is not None
