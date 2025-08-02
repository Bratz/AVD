"""
Simplified MCP server - ONLY handles MCP protocol and tool registration
NO agent logic, NO complex workflows, NO session management
"""

import logging
import os
import asyncio
from urllib.parse import urlparse
from dotenv import load_dotenv
from fastmcp import FastMCP
# from mcp_server.tools.banking_tools import register_banking_tools
from mcp_server.tools.tcs_bancs_real_tools import register_bancs_tools
import yaml

load_dotenv()

class BankingMCPServer:
    """Pure MCP server - only tool registration and protocol compliance."""

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
        self.mcp = FastMCP(name="BankingMCP",stateless_http=True)
        
        self.api_key = os.getenv("MCP_API_KEY")
        if not self.api_key or self.api_key == "your-api-key":
            raise ValueError("MCP_API_KEY not set or invalid")

    async def initialize(self):
        """Initialize MCP server with simple tool registration only."""
        self.logger.info("Initializing Banking MCP server")
        
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            # ONLY register simple MCP tools
            # register_banking_tools(self.mcp, self.config_path)
            register_bancs_tools(self.mcp)
            
            # Add simple health check
            @self.mcp.tool()
            async def health_check() -> dict:
                """Simple health check for MCP server."""
                return {
                    "status": "healthy",
                    "service": "Banking MCP Server",
                    "endpoints": {
                        "http": self.streamable_http_endpoint,
                        "sse": self.sse_endpoint
                    }
                }
            
            self.logger.info("Banking MCP server initialized successfully")
            
        except Exception as e:
            self.logger.exception("MCP server initialization failed: %s", str(e))
            raise

    async def run(self):
        """Run MCP server with multiple transports."""
        parsed_http = urlparse(self.streamable_http_endpoint)
        http_host = parsed_http.hostname or "localhost"
        http_port = parsed_http.port or 8082
        
        parsed_sse = urlparse(self.sse_endpoint)
        sse_host = parsed_sse.hostname or "localhost"
        sse_port = parsed_sse.port or 8084

        self.logger.info("Starting Banking MCP Server")
        self.logger.info("HTTP: %s:%d, SSE: %s:%d", http_host, http_port, sse_host, sse_port)
        
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
        """Start individual transport."""
        try:
            if transport == "stdio":
                await self.mcp.run_async(transport="stdio")
            else:
                await self.mcp.run_async(transport=transport, host=host, port=port)
                self.logger.info("%s transport started on %s:%d", transport, host, port)
        except Exception as e:
            self.logger.error("Failed to start %s transport: %s", transport, e)
            raise