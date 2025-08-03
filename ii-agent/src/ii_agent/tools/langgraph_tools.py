# src/ii_agent/tools/langgraph_tools.py
from typing import Any, Dict, List, Optional, Union
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.prebuilt import ToolExecutor
import json

class LangGraphToolAdapter:
    """Adapts ii-agent tools for use with LangGraph"""
    
    def __init__(self, ii_agent_tool):
        self.ii_tool = ii_agent_tool
    
    def to_langchain_tool(self) -> BaseTool:
        """Convert ii-agent tool to LangChain tool"""
        
        # Create input schema from ii-agent tool definition
        class InputSchema(BaseModel):
            """Dynamic input schema for tool"""
            pass
        
        # Add fields based on ii-agent tool parameters
        for param_name, param_info in self.ii_tool.parameters.items():
            setattr(
                InputSchema,
                param_name,
                Field(
                    default=param_info.get("default", ...),
                    description=param_info.get("description", "")
                )
            )
        
        # Create the LangChain tool
        return StructuredTool(
            name=self.ii_tool.name,
            description=self.ii_tool.description,
            func=self._execute_ii_tool,
            args_schema=InputSchema
        )
    
    def _execute_ii_tool(self, **kwargs) -> Any:
        """Execute ii-agent tool with LangChain interface"""
        return self.ii_tool.execute(**kwargs)

class MCPToolAdapter:
    """Adapts MCP tools for use with LangGraph"""
    
    def __init__(self, mcp_client, server_name: str):
        self.mcp_client = mcp_client
        self.server_name = server_name
    
    async def get_langchain_tools(self) -> List[BaseTool]:
        """Get all MCP tools as LangChain tools"""
        tools = []
        
        # List available tools from MCP server
        response = await self.mcp_client.list_tools()
        
        for tool in response.tools:
            # Create LangChain tool for each MCP tool
            lc_tool = StructuredTool(
                name=f"{self.server_name}:{tool.name}",
                description=tool.description,
                func=lambda **kwargs: self._execute_mcp_tool(tool.name, kwargs),
                args_schema=self._create_schema_from_mcp(tool.inputSchema)
            )
            tools.append(lc_tool)
        
        return tools
    
    def _create_schema_from_mcp(self, mcp_schema: Dict[str, Any]) -> type:
        """Create Pydantic schema from MCP tool schema"""
        
        class DynamicSchema(BaseModel):
            pass
        
        # Convert MCP schema to Pydantic fields
        if "properties" in mcp_schema:
            for prop_name, prop_info in mcp_schema["properties"].items():
                field_type = self._mcp_type_to_python(prop_info.get("type", "string"))
                setattr(
                    DynamicSchema,
                    prop_name,
                    Field(
                        default=... if prop_name in mcp_schema.get("required", []) else None,
                        description=prop_info.get("description", "")
                    )
                )
        
        return DynamicSchema
    
    async def _execute_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute MCP tool"""
        result = await self.mcp_client.call_tool(tool_name, arguments)
        return result.content[0].text if result.content else None

# Unified tool executor for LangGraph
class UnifiedToolExecutor(ToolExecutor):
    """Executes both ii-agent and MCP tools in LangGraph"""
    
    def __init__(self):
        self.tools = {}
        self.mcp_adapters = {}
        
    def register_ii_agent_tools(self, tools: List[Any]):
        """Register ii-agent tools"""
        for tool in tools:
            adapter = LangGraphToolAdapter(tool)
            lc_tool = adapter.to_langchain_tool()
            self.tools[lc_tool.name] = lc_tool
    
    async def register_mcp_server(self, server_name: str, mcp_client):
        """Register MCP server tools"""
        adapter = MCPToolAdapter(mcp_client, server_name)
        tools = await adapter.get_langchain_tools()
        for tool in tools:
            self.tools[tool.name] = tool