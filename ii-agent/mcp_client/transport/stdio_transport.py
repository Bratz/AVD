import logging
import sys
import json
from .base_transport import BaseTransport
from mcp_client.protocol.messages import MCPRequest, MCPResponse

class StdioTransport(BaseTransport):
    """MCP-compliant stdio transport for client communication."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def connect(self):
        """Initialize stdio transport."""
        self.logger.info("Stdio transport connected")

    async def send_request(self, request: MCPRequest) -> MCPResponse:
        """Send MCP request via stdio and return response."""
        try:
            sys.stdout.write(json.dumps(request.to_dict()) + "\n")
            sys.stdout.flush()
            response_line = sys.stdin.readline().strip()
            return MCPResponse.from_dict(json.loads(response_line))
        except Exception as e:
            self.logger.error(f"Error sending request: {str(e)}")
            raise

    async def disconnect(self):
        """Close stdio transport."""
        self.logger.info("Stdio transport disconnected")