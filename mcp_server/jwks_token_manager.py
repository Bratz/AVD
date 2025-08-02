#!/usr/bin/env python3
# jwks_token_manager.py
"""
Robust JWKS and token management for OAuth-enabled MCP
Handles caching, validation, and automatic refresh
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from functools import lru_cache
import httpx
import jwt
from jwt.algorithms import RSAAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cachetools import TTLCache, LRUCache
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class JWKSKey:
    """Represents a single JWK"""
    kid: str
    algorithm: str
    key: Any
    not_before: Optional[int] = None
    not_after: Optional[int] = None
    
    def is_valid(self) -> bool:
        """Check if key is within valid time range"""
        now = int(time.time())
        if self.not_before and now < self.not_before:
            return False
        if self.not_after and now > self.not_after:
            return False
        return True

@dataclass
class TokenInfo:
    """Cached token information"""
    claims: Dict[str, Any]
    user_info: Dict[str, Any]
    cached_at: datetime
    expires_at: datetime
    
    def is_expired(self) -> bool:
        """Check if token info has expired"""
        return datetime.now() >= self.expires_at

class JWKSManager:
    """Manages JWKS fetching, caching, and validation"""
    
    def __init__(
        self,
        jwks_url: str,
        issuer: str,
        refresh_interval: timedelta = timedelta(hours=1),
        max_retries: int = 3
    ):
        self.jwks_url = jwks_url
        self.issuer = issuer
        self.refresh_interval = refresh_interval
        self.max_retries = max_retries
        
        # Key storage
        self._keys: Dict[str, JWKSKey] = {}
        self._last_refresh: Optional[datetime] = None
        self._refresh_lock = asyncio.Lock()
        
        # Stats
        self.stats = {
            "refreshes": 0,
            "refresh_failures": 0,
            "key_lookups": 0,
            "key_hits": 0,
            "key_misses": 0
        }
    
    async def get_key(self, kid: str) -> Optional[JWKSKey]:
        """Get a key by kid, refreshing if necessary"""
        self.stats["key_lookups"] += 1
        
        # Check if we have the key and it's valid
        if kid in self._keys and self._keys[kid].is_valid():
            self.stats["key_hits"] += 1
            return self._keys[kid]
        
        # Check if we need to refresh
        if self._should_refresh() or kid not in self._keys:
            await self.refresh_keys()
        
        # Try again after refresh
        if kid in self._keys and self._keys[kid].is_valid():
            self.stats["key_hits"] += 1
            return self._keys[kid]
        
        self.stats["key_misses"] += 1
        return None
    
    def _should_refresh(self) -> bool:
        """Check if JWKS should be refreshed"""
        if not self._last_refresh:
            return True
        return datetime.now() - self._last_refresh > self.refresh_interval
    
    async def refresh_keys(self) -> None:
        """Refresh JWKS from the provider"""
        async with self._refresh_lock:
            # Double-check inside lock
            if not self._should_refresh():
                return
            
            self.stats["refreshes"] += 1
            
            for attempt in range(self.max_retries):
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            self.jwks_url,
                            timeout=10.0,
                            headers={"Accept": "application/json"}
                        )
                        response.raise_for_status()
                        jwks_data = response.json()
                    
                    # Parse and store keys
                    new_keys = {}
                    for key_data in jwks_data.get("keys", []):
                        try:
                            jwks_key = self._parse_key(key_data)
                            if jwks_key:
                                new_keys[jwks_key.kid] = jwks_key
                        except Exception as e:
                            logger.error(f"Failed to parse key {key_data.get('kid')}: {e}")
                    
                    # Update keys atomically
                    self._keys = new_keys
                    self._last_refresh = datetime.now()
                    
                    logger.info(f"JWKS refreshed successfully. Loaded {len(new_keys)} keys")
                    return
                    
                except Exception as e:
                    logger.error(f"JWKS refresh attempt {attempt + 1} failed: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            self.stats["refresh_failures"] += 1
            raise Exception("Failed to refresh JWKS after all retries")
    
    def _parse_key(self, key_data: Dict[str, Any]) -> Optional[JWKSKey]:
        """Parse a single JWK"""
        kid = key_data.get("kid")
        kty = key_data.get("kty")
        alg = key_data.get("alg", "RS256")
        
        if not kid:
            return None
        
        if kty == "RSA":
            try:
                key = RSAAlgorithm.from_jwk(json.dumps(key_data))
                return JWKSKey(
                    kid=kid,
                    algorithm=alg,
                    key=key,
                    not_before=key_data.get("nbf"),
                    not_after=key_data.get("exp")
                )
            except Exception as e:
                logger.error(f"Failed to parse RSA key {kid}: {e}")
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics"""
        stats = self.stats.copy()
        stats["cached_keys"] = len(self._keys)
        stats["last_refresh"] = self._last_refresh.isoformat() if self._last_refresh else None
        stats["cache_hit_rate"] = (
            self.stats["key_hits"] / self.stats["key_lookups"] 
            if self.stats["key_lookups"] > 0 else 0
        )
        return stats

class TokenValidator:
    """Validates and caches OAuth tokens"""
    
    def __init__(
        self,
        jwks_manager: JWKSManager,
        issuer: str,
        audience: Optional[List[str]] = None,
        cache_size: int = 1000,
        cache_ttl: int = 300  # 5 minutes
    ):
        self.jwks_manager = jwks_manager
        self.issuer = issuer
        self.audience = audience or []
        
        # Token cache - uses token hash as key for security
        self._token_cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
        self._cache_lock = asyncio.Lock()
        
        # Stats
        self.stats = {
            "validations": 0,
            "cache_hits": 0,
            "validation_errors": 0,
            "expired_tokens": 0
        }
    
    def _hash_token(self, token: str) -> str:
        """Create a hash of the token for cache key"""
        return hashlib.sha256(token.encode()).hexdigest()
    
    async def validate_token(
        self,
        token: str,
        required_scopes: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Validate an OAuth token"""
        self.stats["validations"] += 1
        
        # Check cache first
        token_hash = self._hash_token(token)
        
        async with self._cache_lock:
            cached = self._token_cache.get(token_hash)
            if cached and not cached.is_expired():
                self.stats["cache_hits"] += 1
                return cached.user_info
        
        try:
            # Decode header to get kid
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            
            if not kid:
                logger.error("Token missing kid in header")
                self.stats["validation_errors"] += 1
                return None
            
            # Get the key
            jwks_key = await self.jwks_manager.get_key(kid)
            if not jwks_key:
                logger.error(f"No key found for kid: {kid}")
                self.stats["validation_errors"] += 1
                return None
            
            # Validate token
            claims = jwt.decode(
                token,
                jwks_key.key,
                algorithms=[jwks_key.algorithm],
                issuer=self.issuer,
                audience=self.audience if self.audience else None,
                options={
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_iss": True,
                    "verify_aud": bool(self.audience)
                }
            )
            
            # Extract user info
            user_info = self._extract_user_info(claims)
            
            # Check required scopes
            if required_scopes:
                token_scopes = set(claims.get("scope", "").split())
                if not set(required_scopes).issubset(token_scopes):
                    logger.warning(f"Token missing required scopes: {required_scopes}")
                    self.stats["validation_errors"] += 1
                    return None
            
            # Cache the result
            token_info = TokenInfo(
                claims=claims,
                user_info=user_info,
                cached_at=datetime.now(),
                expires_at=datetime.fromtimestamp(claims.get("exp", 0))
            )
            
            async with self._cache_lock:
                self._token_cache[token_hash] = token_info
            
            return user_info
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            self.stats["expired_tokens"] += 1
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            self.stats["validation_errors"] += 1
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            self.stats["validation_errors"] += 1
        
        return None
    
    def _extract_user_info(self, claims: Dict[str, Any]) -> Dict[str, Any]:
        """Extract standardized user info from claims"""
        return {
            "sub": claims.get("sub"),
            "username": claims.get("preferred_username", claims.get("sub")),
            "email": claims.get("email"),
            "email_verified": claims.get("email_verified", False),
            "name": claims.get("name"),
            "given_name": claims.get("given_name"),
            "family_name": claims.get("family_name"),
            "scopes": claims.get("scope", "").split(),
            "roles": self._extract_roles(claims),
            "client_id": claims.get("azp", claims.get("client_id")),
            "session_state": claims.get("session_state"),
            "auth_time": claims.get("auth_time"),
            "exp": claims.get("exp"),
            "iat": claims.get("iat")
        }
    
    def _extract_roles(self, claims: Dict[str, Any]) -> List[str]:
        """Extract roles from various claim formats"""
        roles = []
        
        # Keycloak format
        realm_access = claims.get("realm_access", {})
        roles.extend(realm_access.get("roles", []))
        
        # Resource access
        resource_access = claims.get("resource_access", {})
        for resource, access in resource_access.items():
            resource_roles = access.get("roles", [])
            roles.extend([f"{resource}:{role}" for role in resource_roles])
        
        # Simple roles claim
        if "roles" in claims:
            roles.extend(claims["roles"])
        
        return list(set(roles))  # Remove duplicates
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validator statistics"""
        stats = self.stats.copy()
        stats["cache_size"] = len(self._token_cache)
        stats["cache_hit_rate"] = (
            self.stats["cache_hits"] / self.stats["validations"]
            if self.stats["validations"] > 0 else 0
        )
        return stats

class OAuthManager:
    """High-level OAuth management combining JWKS and token validation"""
    
    def __init__(
        self,
        keycloak_url: str,
        realm: str,
        audience: Optional[List[str]] = None,
        **kwargs
    ):
        self.keycloak_url = keycloak_url
        self.realm = realm
        
        # URLs
        self.jwks_url = f"{keycloak_url}/realms/{realm}/protocol/openid-connect/certs"
        self.issuer = f"{keycloak_url}/realms/{realm}"
        self.token_endpoint = f"{keycloak_url}/realms/{realm}/protocol/openid-connect/token"
        
        # Initialize managers
        self.jwks_manager = JWKSManager(
            jwks_url=self.jwks_url,
            issuer=self.issuer,
            **kwargs.get("jwks_options", {})
        )
        
        self.token_validator = TokenValidator(
            jwks_manager=self.jwks_manager,
            issuer=self.issuer,
            audience=audience,
            **kwargs.get("token_options", {})
        )
    
    async def initialize(self) -> None:
        """Initialize the manager (pre-fetch JWKS)"""
        await self.jwks_manager.refresh_keys()
    
    async def validate_token(
        self,
        token: str,
        required_scopes: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Validate a token"""
        return await self.token_validator.validate_token(token, required_scopes)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics"""
        return {
            "jwks": self.jwks_manager.get_stats(),
            "tokens": self.token_validator.get_stats(),
            "urls": {
                "jwks": self.jwks_url,
                "issuer": self.issuer,
                "token": self.token_endpoint
            }
        }
    
    async def exchange_code_for_token(
        self,
        code: str,
        redirect_uri: str,
        client_id: str,
        client_secret: Optional[str] = None,
        code_verifier: Optional[str] = None
    ) -> Dict[str, Any]:
        """Exchange authorization code for tokens"""
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id
        }
        
        if client_secret:
            data["client_secret"] = client_secret
        if code_verifier:
            data["code_verifier"] = code_verifier
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.token_endpoint,
                data=data,
                headers={"Accept": "application/json"}
            )
            response.raise_for_status()
            return response.json()
    
    async def refresh_token(
        self,
        refresh_token: str,
        client_id: str,
        client_secret: Optional[str] = None
    ) -> Dict[str, Any]:
        """Refresh an access token"""
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id
        }
        
        if client_secret:
            data["client_secret"] = client_secret
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.token_endpoint,
                data=data,
                headers={"Accept": "application/json"}
            )
            response.raise_for_status()
            return response.json()

# Example usage
async def example_usage():
    """Example of using the OAuth manager"""
    
    # Initialize manager
    oauth_manager = OAuthManager(
        keycloak_url="http://localhost:8080",
        realm="banking-mcp",
        audience=["banking-mcp-server"],
        jwks_options={
            "refresh_interval": timedelta(hours=1),
            "max_retries": 3
        },
        token_options={
            "cache_size": 1000,
            "cache_ttl": 300
        }
    )
    
    # Initialize (pre-fetch JWKS)
    await oauth_manager.initialize()
    
    # Validate a token
    token = "eyJ..."  # Your JWT token
    user_info = await oauth_manager.validate_token(
        token,
        required_scopes=["banking:read"]
    )
    
    if user_info:
        print(f"Valid token for user: {user_info['username']}")
        print(f"Scopes: {user_info['scopes']}")
        print(f"Roles: {user_info['roles']}")
    else:
        print("Invalid token")
    
    # Get statistics
    stats = oauth_manager.get_stats()
    print(f"Stats: {json.dumps(stats, indent=2)}")

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())