"""
Enhanced MCP server - Preserves ALL original functionality
Adds OAuth support without breaking existing features
"""

import logging
import os
import asyncio
from urllib.parse import urlparse
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from fastmcp import FastMCP
from mcp_server.tools.tcs_bancs_real_tools import register_bancs_tools
import yaml
import jwt
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

load_dotenv()

class BankingMCPServer:
    """
    Pure MCP server with OPTIONAL OAuth support.
    Preserves all original functionality - OAuth is only activated if ENABLE_OAUTH=true
    """

    def __init__(
        self,
        config_path: str = "mcp_server/config/tools_config.yaml",
        streamable_http_endpoint: str = os.getenv("MCP_HTTP_ENDPOINT", "http://localhost:8082"),
        sse_endpoint: str = os.getenv("MCP_SSE_ENDPOINT", "http://localhost:8084/mcp")
    ):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.streamable_http_endpoint = streamable_http_endpoint
        self.sse_endpoint = sse_endpoint
        
        # Check if OAuth is enabled
        self.oauth_enabled = os.getenv("ENABLE_OAUTH", "false").lower() == "true"
        
        # Initialize FastMCP with custom middleware if OAuth is enabled
        if self.oauth_enabled:
            self.logger.info("OAuth protection enabled for MCP server")
            self.mcp = FastMCP(
                name="BankingMCP",
                stateless_http=True,
                middleware=self._create_oauth_middleware()
            )
        else:
            # Original initialization - no changes
            self.mcp = FastMCP(name="BankingMCP", stateless_http=True)
        
        # Original API key check
        self.api_key = os.getenv("MCP_API_KEY")
        if not self.api_key or self.api_key == "your-api-key":
            raise ValueError("MCP_API_KEY not set or invalid")
        
        # OAuth configuration (only used if enabled)
        self.keycloak_url = os.getenv("KEYCLOAK_URL", "http://localhost:8080")
        self.keycloak_realm = "banking-mcp"

    def _create_oauth_middleware(self):
        """Create OAuth middleware for FastMCP"""
        async def oauth_middleware(request: Dict[str, Any], call_next):
            """Validate OAuth token if present"""
            
            # Extract headers from request
            headers = request.get("headers", {})
            
            # Check for Authorization header
            auth_header = None
            for key, value in headers.items():
                if key.lower() == "authorization":
                    auth_header = value
                    break
            
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header[7:]  # Remove "Bearer " prefix
                
                # Validate token
                user_info = await self._validate_token(token)
                if user_info:
                    # Add user context to request
                    request["user"] = user_info
                else:
                    # Invalid token
                    return {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32603,
                            "message": "Unauthorized: Invalid token"
                        },
                        "id": request.get("id")
                    }
            elif self.oauth_enabled:
                # OAuth is enabled but no token provided
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": "Unauthorized: Bearer token required"
                    },
                    "id": request.get("id")
                }
            
            # Continue with request processing
            return await call_next(request)
        
        return oauth_middleware

    async def _validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate OAuth token - simplified for testing"""
        try:
            # In production, verify with Keycloak JWKS
            # For now, decode without verification
            claims = jwt.decode(token, options={"verify_signature": False})
            
            return {
                "sub": claims.get("sub"),
                "username": claims.get("preferred_username", claims.get("sub")),
                "email": claims.get("email"),
                "scopes": claims.get("scope", "").split(),
                "roles": claims.get("realm_access", {}).get("roles", [])
            }
        except Exception as e:
            self.logger.error(f"Token validation error: {e}")
            return None

    async def initialize(self):
        """Initialize MCP server - UNCHANGED from original"""
        self.logger.info("Initializing Banking MCP server")
        
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            # ORIGINAL: Register banking tools
            register_bancs_tools(self.mcp)
            
            # ORIGINAL: Add simple health check
            @self.mcp.tool()
            async def health_check() -> dict:
                """Simple health check for MCP server."""
                return {
                    "status": "healthy",
                    "service": "Banking MCP Server",
                    "oauth_enabled": self.oauth_enabled,
                    "endpoints": {
                        "http": self.streamable_http_endpoint,
                        "sse": self.sse_endpoint
                    }
                }
            
            # NEW: Add OAuth metadata endpoint if OAuth is enabled
            if self.oauth_enabled:
                @self.mcp.tool()
                async def oauth_info() -> dict:
                    """Get OAuth configuration information."""
                    return {
                        "oauth_enabled": True,
                        "token_endpoint": f"{self.keycloak_url}/realms/{self.keycloak_realm}/protocol/openid-connect/token",
                        "authorization_endpoint": f"{self.keycloak_url}/realms/{self.keycloak_realm}/protocol/openid-connect/auth",
                        "required_scopes": ["banking:read", "banking:write", "customer:profile"]
                    }
            
            self.logger.info("Banking MCP server initialized successfully")
            
        except Exception as e:
            self.logger.exception("MCP server initialization failed: %s", str(e))
            raise

    async def run(self):
        """Run MCP server with multiple transports - UNCHANGED from original"""
        parsed_http = urlparse(self.streamable_http_endpoint)
        http_host = parsed_http.hostname or "localhost"
        http_port = parsed_http.port or 8082
        
        parsed_sse = urlparse(self.sse_endpoint)
        sse_host = parsed_sse.hostname or "localhost"
        sse_port = parsed_sse.port or 8084

        self.logger.info("Starting Banking MCP Server")
        self.logger.info("HTTP: %s:%d, SSE: %s:%d", http_host, http_port, sse_host, sse_port)
        if self.oauth_enabled:
            self.logger.info("OAuth protection: ENABLED")
        
        try:
            tasks = [
                self._start_transport("streamable-http", http_host, http_port),
                self._start_transport("sse", sse_host, sse_port),
                self._start_transport("stdio")
            ]
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error("Server runtime error: %s", str(e))
            raise

    async def _start_transport(self, transport: str, host: str = None, port: int = None):
        """Start individual transport - UNCHANGED from original"""
        try:
            if transport == "stdio":
                await self.mcp.run_async(transport="stdio")
            else:
                await self.mcp.run_async(transport=transport, host=host, port=port)
                self.logger.info("%s transport started on %s:%d", transport, host, port)
        except Exception as e:
            self.logger.error("Failed to start %s transport: %s", transport, e)
            raise


# For backward compatibility - can be imported and used exactly as before
if __name__ == "__main__":
    async def main():
        server = BankingMCPServer()
        await server.initialize()
        await server.run()
    
    asyncio.run(main())