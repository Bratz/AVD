"""
Corrected MCP protocol messages with proper request formatting
"""
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field
import uuid

class MCPMessage(BaseModel):
    """Base MCP protocol message."""
    jsonrpc: str = Field(default="2.0", description="JSON-RPC version")

class MCPRequest(MCPMessage):
    """Standard MCP request message with required ID."""
    method: str = Field(description="MCP method name")
    params: Optional[Dict[str, Any]] = Field(default=None, description="Method parameters")
    id: Union[str, int] = Field(description="Request ID - REQUIRED for requests")

    def __init__(self, **data):
        # Ensure every request has an ID
        if 'id' not in data or data['id'] is None:
            data['id'] = str(uuid.uuid4())
        super().__init__(**data)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPRequest':
        return cls.model_validate(data)

class MCPResponse(MCPMessage):
    """Standard MCP response message."""
    result: Optional[Dict[str, Any]] = Field(default=None, description="Success result")
    error: Optional[Dict[str, Any]] = Field(default=None, description="Error details")
    id: Optional[Union[str, int]] = Field(default=None, description="Request ID")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPResponse':
        return cls.model_validate(data)

class MCPNotification(MCPMessage):
    """Standard MCP notification (no response expected, no ID)."""
    method: str = Field(description="Notification method")
    params: Optional[Dict[str, Any]] = Field(default=None, description="Notification parameters")
    # Note: Notifications MUST NOT have an 'id' field

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)

# Standard MCP requests with proper formatting
class InitializeRequest(MCPRequest):
    """MCP initialize handshake request with correct format."""
    method: str = Field(default="initialize")
    
    def __init__(self, **data):
        # Provide proper MCP 2024-11-05 initialize parameters
        if 'params' not in data:
            data['params'] = {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "BankingMCPClient",
                    "version": "1.0.0"
                }
            }
        super().__init__(**data)

class ListToolsRequest(MCPRequest):
    """MCP tools/list request."""
    method: str = Field(default="tools/list")
    
    def __init__(self, **data):
        if 'params' not in data:
            data['params'] = {}
        super().__init__(**data)

class CallToolRequest(MCPRequest):
    """MCP tools/call request."""
    method: str = Field(default="tools/call")