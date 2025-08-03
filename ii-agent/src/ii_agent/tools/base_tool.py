"""
II-Agent Base Tool Interface
Adapted from II-Agent repository patterns
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import uuid

@dataclass
class ToolResult:
    """Result of tool execution."""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseTool(ABC):
    """Base class for all II-Agent tools."""
    
    def __init__(self, name: str, description: str, category: str = "general"):
        self.name = name
        self.description = description
        self.category = category
        self.tool_id = str(uuid.uuid4())
        
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
        
    @abstractmethod
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters."""
        pass
        
    def get_tool_info(self) -> Dict[str, Any]:
        """Get information about this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "tool_id": self.tool_id,
            "parameters_schema": self.get_parameters_schema()
        }
        
    async def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters against schema."""
        # Basic validation - can be enhanced with jsonschema
        schema = self.get_parameters_schema()
        required = schema.get("required", [])
        
        for field in required:
            if field not in parameters:
                return False
                
        return True