from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from mcp_client.protocol.messages import MCPRequest, MCPResponse

class BaseTransport(ABC):
    """Abstract base class for MCP transports."""

    @abstractmethod
    async def connect(self):
        pass

    @abstractmethod
    async def send_request(self, request: MCPRequest) -> MCPResponse:
        pass

    @abstractmethod
    async def disconnect(self):
        pass