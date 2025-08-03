"""
Standard MCP client - fully protocol compliant
"""
import logging
from typing import Dict, Any, List, Optional
from .transport.base_transport import BaseTransport
from .protocol.messages import (
    InitializeRequest, ListToolsRequest, CallToolRequest,
    MCPRequest, MCPResponse
)

class ListPromptsRequest(MCPRequest):
    """Request to list available prompts."""
    def __init__(self):
        super().__init__(method="prompts/list", params={})

class GetPromptRequest(MCPRequest):
    """Request to get a specific prompt."""
    def __init__(self, params: Dict[str, Any]):
        super().__init__(method="prompts/get", params=params)
        
class StandardMCPClient:
    """MCP-compliant client implementation."""

    def __init__(self, transport: BaseTransport):
        self.transport = transport
        self.server_capabilities: Optional[Dict[str, Any]] = None
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> Dict[str, Any]:
        """Perform MCP initialization handshake."""
        self.logger.info("Initializing MCP client")
        
        await self.transport.connect()
        
        init_request = InitializeRequest()
        response = await self.transport.send_request(init_request)
        
        if response.error:
            raise MCPClientError(f"Initialization failed: {response.error}")
        
        self.server_capabilities = response.result.get("capabilities", {})
        
        self.logger.info("MCP client initialized successfully")
        return response.result

    async def list_prompts(self) -> List[Dict[str, Any]]:
        """List available prompts from MCP server."""
        if not self.server_capabilities:
            raise MCPClientError("Client not initialized")

        request = ListPromptsRequest()
        response = await self.transport.send_request(request)
        
        if response.error:
            raise MCPClientError(f"List prompts failed: {response.error}")
        
        prompts = response.result.get("prompts", [])
        self.logger.debug("Listed %d prompts", len(prompts))
        return prompts

    async def get_prompt(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get a specific prompt with arguments."""
        if not self.server_capabilities:
            raise MCPClientError("Client not initialized")

        request = GetPromptRequest(params={
            "name": name,
            "arguments": arguments or {}
        })
        
        response = await self.transport.send_request(request)
        
        if response.error:
            raise MCPClientError(f"Get prompt failed: {response.error}")
        
        self.logger.debug("Retrieved prompt '%s' successfully", name)
        return response.result

    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from MCP server."""
        if not self.server_capabilities:
            raise MCPClientError("Client not initialized")

        request = ListToolsRequest()
        response = await self.transport.send_request(request)
        
        if response.error:
            raise MCPClientError(f"List tools failed: {response.error}")
        
        tools = response.result.get("tools", [])
        self.logger.debug("Listed %d tools", len(tools))
        return tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool on the MCP server."""
        if not self.server_capabilities:
            raise MCPClientError("Client not initialized")

        request = CallToolRequest(params={
            "name": name,
            "arguments": arguments
        })
        
        response = await self.transport.send_request(request)
        
        if response.error:
            raise MCPClientError(f"Tool call failed: {response.error}")
        
        self.logger.debug("Tool '%s' executed successfully", name)
        return response.result

    async def close(self) -> None:
        """Close MCP client connection."""
        await self.transport.disconnect()
        self.server_capabilities = None
        self.logger.info("MCP client closed")

    



class MCPClientError(Exception):
    """MCP client-specific error."""
    pass
