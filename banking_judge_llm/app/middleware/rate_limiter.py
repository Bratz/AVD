import asyncio
import time
from functools import wraps
from typing import Callable, Any
from fastapi import HTTPException, Request
from cachetools import TTLCache

class RateLimiter:
    """In-memory rate limiter using token bucket algorithm."""
    def __init__(self, limit: int, window: int):
        self.limit = limit
        self.window = window
        self.buckets = TTLCache(maxsize=1000, ttl=window)
        self.lock = asyncio.Lock()

    async def acquire(self, key: str) -> bool:
        """Check if a request is allowed under the rate limit."""
        async with self.lock:
            now = time.time()
            bucket = self.buckets.get(key, {"tokens": self.limit, "last_refill": now})

            # Refill tokens based on elapsed time
            elapsed = now - bucket["last_refill"]
            new_tokens = min(self.limit, bucket["tokens"] + (elapsed * self.limit / self.window))
            bucket["tokens"] = new_tokens
            bucket["last_refill"] = now

            # Check if request is allowed
            if new_tokens >= 1:
                bucket["tokens"] -= 1
                self.buckets[key] = bucket
                return True
            return False

def rate_limit(limit: int, window: int) -> Callable:
    """Decorator for rate-limiting FastAPI endpoints."""
    limiter = RateLimiter(limit=limit, window=window)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs) -> Any:
            client_ip = request.client.host
            key = f"{client_ip}:{func.__name__}"
            if not await limiter.acquire(key):
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded: {limit} requests per {window} seconds"
                )
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator