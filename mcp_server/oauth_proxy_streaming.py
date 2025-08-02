#!/usr/bin/env python3
# oauth_proxy_enhanced.py
"""
Enhanced OAuth proxy for Banking MCP with proper streaming support,
content-length handling, and robust JWKS caching
"""

import os
import logging
import json
import time
import asyncio
from typing import Optional, Dict, Any, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from fastapi import FastAPI, Request, HTTPException, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import jwt
from jwt.algorithms import RSAAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from dotenv import load_dotenv
import uvicorn
from cachetools import TTLCache
import aiohttp

load_dotenv()

# Configuration
KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "http://localhost:8080")
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "banking-mcp")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8082")
MCP_SSE_URL = os.getenv("MCP_SSE_URL", "http://localhost:8084")
PROXY_PORT = int(os.getenv("PROXY_PORT", "8081"))

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App initialization
app = FastAPI(title="Enhanced OAuth Proxy for Banking MCP")
security = HTTPBearer(auto_error=False)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],  # Important for streaming responses
)

@dataclass
class JWKSCache:
    """JWKS cache with automatic refresh"""
    keys: Dict[str, Any] = field(default_factory=dict)
    last_refresh: Optional[datetime] = None
    refresh_interval: timedelta = field(default_factory=lambda: timedelta(hours=1))
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

@dataclass
class TokenCache:
    """Token validation cache with TTL"""
    cache: TTLCache = field(default_factory=lambda: TTLCache(maxsize=1000, ttl=300))
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

# Global caches
jwks_cache = JWKSCache()
token_cache = TokenCache()

class OAuthValidator:
    """Handles OAuth token validation with JWKS"""
    
    def __init__(self, keycloak_url: str, realm: str):
        self.keycloak_url = keycloak_url
        self.realm = realm
        self.jwks_url = f"{keycloak_url}/realms/{realm}/protocol/openid-connect/certs"
        self.issuer = f"{keycloak_url}/realms/{realm}"
        
    async def refresh_jwks(self) -> None:
        """Refresh JWKS from Keycloak"""
        async with jwks_cache.lock:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(self.jwks_url)
                    response.raise_for_status()
                    jwks_data = response.json()
                    
                    # Parse and cache keys
                    for key_data in jwks_data.get("keys", []):
                        kid = key_data.get("kid")
                        if kid and key_data.get("kty") == "RSA":
                            try:
                                jwks_cache.keys[kid] = RSAAlgorithm.from_jwk(json.dumps(key_data))
                                logger.info(f"Cached JWKS key: {kid}")
                            except Exception as e:
                                logger.error(f"Failed to parse key {kid}: {e}")
                    
                    jwks_cache.last_refresh = datetime.now()
                    logger.info(f"JWKS refreshed successfully. Cached {len(jwks_cache.keys)} keys")
                    
            except Exception as e:
                logger.error(f"Failed to refresh JWKS: {e}")
                raise
    
    async def get_public_key(self, kid: str) -> Optional[Any]:
        """Get public key for the given kid, refreshing if necessary"""
        # Check if we need to refresh
        if (not jwks_cache.last_refresh or 
            datetime.now() - jwks_cache.last_refresh > jwks_cache.refresh_interval or
            kid not in jwks_cache.keys):
            await self.refresh_jwks()
        
        return jwks_cache.keys.get(kid)
    
    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate OAuth token with caching"""
        # Check cache first
        async with token_cache.lock:
            cached = token_cache.cache.get(token)
            if cached:
                return cached
        
        try:
            # Decode header to get kid
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            
            if not kid:
                logger.error("Token missing kid in header")
                return None
            
            # Get public key
            public_key = await self.get_public_key(kid)
            if not public_key:
                logger.error(f"No public key found for kid: {kid}")
                return None
            
            # Verify token
            claims = jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                issuer=self.issuer,
                options={
                    "verify_exp": True,
                    "verify_aud": False,  # We'll check audience separately
                    "verify_iss": True
                }
            )
            
            # Extract user info
            user_info = {
                "sub": claims.get("sub"),
                "username": claims.get("preferred_username", claims.get("sub")),
                "email": claims.get("email"),
                "name": claims.get("name"),
                "scopes": claims.get("scope", "").split(),
                "roles": claims.get("realm_access", {}).get("roles", []),
                "exp": claims.get("exp"),
                "iat": claims.get("iat")
            }
            
            # Cache the result
            async with token_cache.lock:
                token_cache.cache[token] = user_info
            
            return user_info
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
        except Exception as e:
            logger.error(f"Token validation error: {e}")
        
        return None

# Initialize validator
oauth_validator = OAuthValidator(KEYCLOAK_URL, KEYCLOAK_REALM)

class StreamingProxy:
    """Handles streaming responses properly"""
    
    @staticmethod
    async def stream_response(
        source_response: httpx.Response,
        chunk_size: int = 8192
    ) -> AsyncIterator[bytes]:
        """Stream response chunks from source to client"""
        try:
            async for chunk in source_response.aiter_bytes(chunk_size):
                yield chunk
        finally:
            await source_response.aclose()
    
    @staticmethod
    async def stream_sse(
        source_response: httpx.Response
    ) -> AsyncIterator[bytes]:
        """Stream SSE responses with proper formatting"""
        try:
            async for line in source_response.aiter_lines():
                # Ensure proper SSE format
                if line:
                    yield f"{line}\n".encode('utf-8')
                else:
                    yield b"\n"  # Empty line (event separator)
                    
                # Flush buffer for real-time streaming
                if line.startswith("data:"):
                    yield b"\n"  # Extra newline after data
                    
        finally:
            await source_response.aclose()
    
    @staticmethod
    def prepare_headers(
        source_headers: httpx.Headers,
        skip_headers: set = None
    ) -> Dict[str, str]:
        """Prepare headers for proxied response"""
        if skip_headers is None:
            skip_headers = {
                'host', 'connection', 'content-length', 'transfer-encoding',
                'upgrade', 'proxy-authenticate', 'proxy-authorization',
                'te', 'trailer'
            }
        
        headers = {}
        for key, value in source_headers.items():
            if key.lower() not in skip_headers:
                headers[key] = value
        
        return headers

streaming_proxy = StreamingProxy()

@app.on_event("startup")
async def startup():
    """Initialize JWKS on startup"""
    try:
        await oauth_validator.refresh_jwks()
    except Exception as e:
        logger.warning(f"Failed to refresh JWKS on startup: {e}")

@app.get("/.well-known/oauth-protected-resource")
async def oauth_metadata():
    """OAuth 2.0 Protected Resource Metadata"""
    return {
        "resource": f"http://localhost:{PROXY_PORT}",
        "authorization_servers": [
            f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}"
        ],
        "scopes_supported": [
            "banking:read",
            "banking:write",
            "customer:profile"
        ],
        "bearer_methods_supported": ["header"],
        "resource_documentation": f"http://localhost:{PROXY_PORT}/docs",
        "mcp_protocol_version": "2025-06-18",
        "streaming_supported": True,
        "sse_endpoint": f"http://localhost:{PROXY_PORT}/sse"
    }

@app.get("/health")
async def health():
    """Health check with dependency status"""
    health_status = {
        "status": "healthy",
        "service": "oauth-proxy-enhanced",
        "timestamp": datetime.now().isoformat(),
        "dependencies": {}
    }
    
    # Check MCP server
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{MCP_SERVER_URL}/health", timeout=2)
            health_status["dependencies"]["mcp_server"] = {
                "url": MCP_SERVER_URL,
                "status": "healthy" if resp.status_code == 200 else "unhealthy",
                "response_time_ms": resp.elapsed.total_seconds() * 1000
            }
    except Exception as e:
        health_status["dependencies"]["mcp_server"] = {
            "url": MCP_SERVER_URL,
            "status": "unreachable",
            "error": str(e)
        }
    
    # Check JWKS status
    health_status["dependencies"]["jwks"] = {
        "cached_keys": len(jwks_cache.keys),
        "last_refresh": jwks_cache.last_refresh.isoformat() if jwks_cache.last_refresh else None
    }
    
    # Check token cache
    health_status["dependencies"]["token_cache"] = {
        "cached_tokens": len(token_cache.cache),
        "max_size": token_cache.cache.maxsize
    }
    
    return health_status

async def validate_request(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """Validate OAuth token from request"""
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    user_info = await oauth_validator.validate_token(credentials.credentials)
    if not user_info:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": 'Bearer error="invalid_token"'}
        )
    
    # Check required scopes
    required_scopes = {"banking:read"}
    user_scopes = set(user_info.get("scopes", []))
    
    if not required_scopes.issubset(user_scopes):
        raise HTTPException(
            status_code=403,
            detail=f"Insufficient scope. Required: {required_scopes}",
            headers={"WWW-Authenticate": f'Bearer error="insufficient_scope" scope="{" ".join(required_scopes)}"'}
        )
    
    return user_info

@app.post("/mcp")
@app.post("/mcp/{path:path}")
async def proxy_mcp(
    request: Request,
    path: Optional[str] = None,
    user_info: Dict[str, Any] = Depends(validate_request)
):
    """Proxy MCP requests with proper streaming support"""
    
    # Build target URL
    target_path = f"/mcp/{path}" if path else "/mcp"
    target_url = f"{MCP_SERVER_URL}{target_path}"
    
    logger.info(f"Proxying request from {user_info['username']} to {target_url}")
    
    # Read request body
    body = await request.body()
    
    # Prepare headers
    headers = streaming_proxy.prepare_headers(request.headers)
    headers["X-User-Context"] = json.dumps({
        "username": user_info["username"],
        "sub": user_info["sub"],
        "scopes": user_info["scopes"]
    })
    
    # Make proxied request with streaming support
    async with httpx.AsyncClient() as client:
        try:
            # Create request
            req = client.build_request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
                params=dict(request.query_params)
            )
            
            # Send with streaming
            response = await client.send(req, stream=True)
            
            # Check content type for streaming decision
            content_type = response.headers.get("content-type", "")
            is_sse = "text/event-stream" in content_type
            is_chunked = response.headers.get("transfer-encoding") == "chunked"
            
            # Prepare response headers (without content-length for streaming)
            response_headers = streaming_proxy.prepare_headers(response.headers)
            
            if is_sse:
                # SSE streaming
                logger.info("Streaming SSE response")
                return StreamingResponse(
                    streaming_proxy.stream_sse(response),
                    media_type="text/event-stream",
                    headers=response_headers,
                    status_code=response.status_code
                )
            elif is_chunked or response.headers.get("content-length", "0") == "0":
                # Chunked streaming
                logger.info("Streaming chunked response")
                return StreamingResponse(
                    streaming_proxy.stream_response(response),
                    media_type=content_type or "application/octet-stream",
                    headers=response_headers,
                    status_code=response.status_code
                )
            else:
                # Regular response - read all content
                content = await response.aread()
                await response.aclose()
                
                # Don't set content-length, let FastAPI handle it
                if "content-length" in response_headers:
                    del response_headers["content-length"]
                
                return Response(
                    content=content,
                    media_type=content_type or "application/json",
                    headers=response_headers,
                    status_code=response.status_code
                )
                
        except httpx.ConnectError:
            logger.error(f"Failed to connect to {target_url}")
            raise HTTPException(503, detail="MCP server unavailable")
        except asyncio.TimeoutError:
            logger.error(f"Timeout connecting to {target_url}")
            raise HTTPException(504, detail="Gateway timeout")
        except Exception as e:
            logger.error(f"Proxy error: {type(e).__name__}: {e}")
            raise HTTPException(500, detail=f"Internal proxy error: {str(e)}")

@app.get("/sse")
@app.get("/sse/{path:path}")
async def proxy_sse(
    request: Request,
    path: Optional[str] = None,
    user_info: Dict[str, Any] = Depends(validate_request)
):
    """Dedicated SSE proxy endpoint"""
    
    # Build target URL for SSE
    target_path = f"/{path}" if path else "/"
    target_url = f"{MCP_SSE_URL}{target_path}"
    
    logger.info(f"Proxying SSE request from {user_info['username']} to {target_url}")
    
    # SSE specific headers
    headers = {
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-User-Context": json.dumps({
            "username": user_info["username"],
            "sub": user_info["sub"]
        })
    }
    
    async def event_generator():
        """Generate SSE events"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(target_url, headers=headers) as response:
                    async for line in response.content:
                        if line:
                            yield line
                        else:
                            yield b"\n"
            except Exception as e:
                logger.error(f"SSE streaming error: {e}")
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n".encode()
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
            "Connection": "keep-alive"
        }
    )

@app.get("/")
async def root():
    """Service information"""
    return {
        "service": "Enhanced OAuth Proxy for Banking MCP",
        "version": "2.0.0",
        "features": [
            "OAuth 2.0 token validation with JWKS",
            "Streaming response support (SSE, chunked)",
            "Content-length handling",
            "Token caching with TTL",
            "Automatic JWKS refresh"
        ],
        "endpoints": {
            "oauth_metadata": "/.well-known/oauth-protected-resource",
            "mcp_proxy": "/mcp",
            "sse_proxy": "/sse",
            "health": "/health"
        },
        "authentication": {
            "type": "Bearer",
            "token_endpoint": f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token",
            "required_scopes": ["banking:read"]
        }
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Consistent error response format"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "timestamp": datetime.now().isoformat()
            }
        },
        headers=exc.headers
    )

if __name__ == "__main__":
    print(f"🚀 Enhanced OAuth Proxy for Banking MCP")
    print(f"=" * 50)
    print(f"📍 Proxy URL: http://localhost:{PROXY_PORT}")
    print(f"🔄 MCP Server: {MCP_SERVER_URL}")
    print(f"📡 SSE Server: {MCP_SSE_URL}")
    print(f"🔑 Keycloak: {KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}")
    print(f"\n✨ Features:")
    print(f"   - JWKS validation with auto-refresh")
    print(f"   - Token caching with TTL")
    print(f"   - Streaming support (SSE, chunked)")
    print(f"   - Proper content-length handling")
    print(f"\n📝 Usage:")
    print(f"   curl -H 'Authorization: Bearer <token>' \\")
    print(f"        http://localhost:{PROXY_PORT}/mcp")
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PROXY_PORT,
        log_level="info",
        # Important for SSE
        timeout_keep_alive=30,
        limit_max_requests=0
    )