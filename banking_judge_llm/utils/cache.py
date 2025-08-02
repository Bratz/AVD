from cachetools import TTLCache
from config.settings import Settings
import logging

class Cache:
    """In-memory cache using TTLCache."""
    
    def __init__(self):
        self.settings = Settings()
        self.logger = logging.getLogger(__name__)
        self.cache = TTLCache(
            maxsize=1000,
            ttl=self.settings.cache_ttl
        )

    async def get(self, key: str) -> str | None:
        """Get value from cache."""
        try:
            value = self.cache.get(key)
            self.logger.debug(f"Cache {'hit' if value else 'miss'} for key: {key}")
            return value
        except Exception as e:
            self.logger.error(f"Cache get error: {str(e)}")
            return None

    async def set(self, key: str, value: str, ttl: int | None = None):
        """Set value in cache."""
        try:
            self.cache[key] = value
            self.logger.debug(f"Cache set for key: {key}")
        except Exception as e:
            self.logger.error(f"Cache set error: {str(e)}")