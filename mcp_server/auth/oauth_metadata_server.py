# mcp_server/oauth_metadata_server.py
"""
OAuth Metadata Server for Banking MCP
Serves the OAuth metadata endpoints that MCP Inspector looks for
"""

import os
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="MCP OAuth Metadata Server")

# Add CORS support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "http://localhost:8080")
KEYCLOAK_REALM = "banking-mcp"
MCP_HTTP_ENDPOINT = os.getenv("MCP_HTTP_ENDPOINT", "http://localhost:8082")

@app.get("/.well-known/oauth-protected-resource")
async def oauth_protected_resource():
    """OAuth 2.0 Protected Resource Metadata (RFC 9728)"""
    return {
        "resource": MCP_HTTP_ENDPOINT,
        "authorization_servers": [
            f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/.well-known/openid-configuration"
        ],
        "scopes_supported": [
            "banking:read",
            "banking:write",
            "customer:profile"
        ],
        "bearer_methods_supported": ["header"],
        "resource_documentation": f"{MCP_HTTP_ENDPOINT}/docs",
        "mcp_protocol_version": "2025-06-18",
        "resource_type": "mcp-server"
    }

@app.get("/.well-known/oauth-authorization-server")
async def oauth_authorization_server():
    """Redirect to Keycloak's authorization server metadata"""
    return JSONResponse(
        status_code=302,
        headers={
            "Location": f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/.well-known/openid-configuration"
        }
    )

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "oauth-metadata"}

# Handle the OPTIONS requests that are causing errors
@app.options("/.well-known/oauth-protected-resource")
@app.options("/.well-known/oauth-protected-resource/sse")
@app.options("/.well-known/oauth-authorization-server")
async def handle_options():
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

# Return 404 for the SSE variant (it's not a real endpoint)
@app.get("/.well-known/oauth-protected-resource/sse")
async def oauth_sse_not_found():
    return JSONResponse(
        status_code=404,
        content={"detail": "SSE endpoint not available for metadata"}
    )

if __name__ == "__main__":
    port = int(MCP_HTTP_ENDPOINT.split(":")[-1])  # Use same port as MCP
    print(f"Starting OAuth Metadata Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)