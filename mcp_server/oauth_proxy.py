# oauth_proxy_fixed.py
"""
Fixed OAuth proxy that properly handles MCP protocol and streaming
"""

import os
import logging
import json
from typing import Optional, AsyncIterator
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse, Response
import httpx
import jwt
from dotenv import load_dotenv
import uvicorn
import asyncio

load_dotenv()

# Configuration
KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "http://localhost:8080")
KEYCLOAK_REALM = "banking-mcp"
MCP_SERVER_URL = "http://localhost:8082"
PROXY_PORT = 8081

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OAuth Proxy for Banking MCP")
security = HTTPBearer(auto_error=False)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Token cache (simple in-memory cache for demo)
token_cache = {}

async def validate_token(token: str) -> Optional[dict]:
    """Validate OAuth token and return user info"""
    
    # Check cache first
    if token in token_cache:
        cached = token_cache[token]
        # Simple expiry check (you'd want better logic in production)
        return cached
    
    try:
        # For testing, decode without verification
        # In production, verify with Keycloak JWKS
        claims = jwt.decode(token, options={"verify_signature": False})
        
        user_info = {
            "sub": claims.get("sub"),
            "username": claims.get("preferred_username", claims.get("sub")),
            "email": claims.get("email"),
            "scopes": claims.get("scope", "").split(),
            "roles": claims.get("realm_access", {}).get("roles", []),
            "exp": claims.get("exp")
        }
        
        # Cache it
        token_cache[token] = user_info
        
        return user_info
        
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        return None

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
        "mcp_protocol_version": "2025-06-18"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    # Check if MCP server is reachable
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{MCP_SERVER_URL}/health", timeout=2)
            mcp_healthy = resp.status_code == 200
    except:
        mcp_healthy = False
    
    return {
        "status": "healthy",
        "service": "oauth-proxy",
        "mcp_server": {
            "url": MCP_SERVER_URL,
            "healthy": mcp_healthy
        }
    }

@app.post("/mcp")
@app.post("/mcp/{path:path}")
async def proxy_mcp(
    request: Request,
    path: Optional[str] = None,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """Proxy MCP requests with OAuth validation"""
    
    # Validate token
    if not credentials:
        return JSONResponse(
            status_code=401,
            content={
                "error": "unauthorized",
                "message": "Missing authorization header"
            }
        )
    
    user_info = await validate_token(credentials.credentials)
    if not user_info:
        return JSONResponse(
            status_code=401,
            content={
                "error": "unauthorized",
                "message": "Invalid or expired token"
            }
        )
    
    # Log the request
    logger.info(f"Proxying MCP request from user: {user_info['username']}")
    
    # Build target URL
    target_path = f"/mcp/{path}" if path else "/mcp"
    target_url = f"{MCP_SERVER_URL}{target_path}"
    
    # Get request body
    body = await request.body()
    
    # Prepare headers (remove hop-by-hop headers)
    headers = {}
    skip_headers = {
        'host', 'connection', 'keep-alive', 'transfer-encoding',
        'upgrade', 'proxy-authenticate', 'proxy-authorization',
        'te', 'trailer', 'authorization'
    }
    
    for key, value in request.headers.items():
        if key.lower() not in skip_headers:
            headers[key] = value
    
    # Add user context header
    headers['X-User-Info'] = json.dumps({
        "username": user_info['username'],
        "sub": user_info['sub'],
        "scopes": user_info['scopes']
    })
    
    # Make the proxied request
    try:
        async with httpx.AsyncClient() as client:
            # Build request
            req = client.build_request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
                params=dict(request.query_params)
            )
            
            # Send with streaming
            response = await client.send(req, stream=True)
            
            # Check if it's a streaming response
            is_streaming = (
                response.headers.get('transfer-encoding') == 'chunked' or
                'text/event-stream' in response.headers.get('content-type', '')
            )
            
            if is_streaming:
                # Stream the response
                async def stream_generator():
                    try:
                        async for chunk in response.aiter_bytes():
                            yield chunk
                    finally:
                        await response.aclose()
                
                # Prepare response headers
                response_headers = {}
                for key, value in response.headers.items():
                    if key.lower() not in skip_headers:
                        response_headers[key] = value
                
                return StreamingResponse(
                    stream_generator(),
                    status_code=response.status_code,
                    headers=response_headers,
                    media_type=response.headers.get('content-type')
                )
            else:
                # Regular response
                content = await response.aread()
                await response.aclose()
                
                # Try to parse as JSON for better error handling
                try:
                    json_content = json.loads(content)
                    return JSONResponse(
                        content=json_content,
                        status_code=response.status_code,
                        headers=dict(response.headers)
                    )
                except:
                    # Return as-is if not JSON
                    return Response(
                        content=content,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        media_type=response.headers.get('content-type', 'application/octet-stream')
                    )
                    
    except httpx.ConnectError:
        logger.error(f"Failed to connect to MCP server at {target_url}")
        return JSONResponse(
            status_code=503,
            content={
                "error": "service_unavailable",
                "message": "MCP server is not reachable",
                "target": target_url
            }
        )
    except Exception as e:
        logger.error(f"Proxy error: {type(e).__name__}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "proxy_error",
                "message": str(e),
                "type": type(e).__name__
            }
        )

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "OAuth Proxy for Banking MCP",
        "version": "1.0.0",
        "endpoints": {
            "oauth_metadata": "/.well-known/oauth-protected-resource",
            "mcp_proxy": "/mcp",
            "health": "/health"
        },
        "authentication": {
            "type": "Bearer",
            "token_endpoint": f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token"
        }
    }

# Handle preflight requests
@app.options("/{path:path}")
async def handle_options(path: str):
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

if __name__ == "__main__":
    print(f"🔐 OAuth Proxy for Banking MCP")
    print(f"=" * 50)
    print(f"📍 Proxy URL: http://localhost:{PROXY_PORT}")
    print(f"🔄 MCP Server: {MCP_SERVER_URL}")
    print(f"🔑 Keycloak: {KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}")
    print(f"\n👉 Use http://localhost:{PROXY_PORT}/mcp with your OAuth token")
    print(f"👉 OAuth metadata: http://localhost:{PROXY_PORT}/.well-known/oauth-protected-resource\n")
    
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT, log_level="info")