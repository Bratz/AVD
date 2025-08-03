"""
MCP Tool Adapter Module
Provides LLMTool adapter for MCP tools with proper validation and execution
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional

from src.ii_agent.tools.base import LLMTool, ToolImplOutput
from src.ii_agent.tools.banking_tool_registry import ToolMetadata,MCPToolRegistry,mcp_tool_registry
from wrappers.mcp_client_wrapper import MCPClientWrapper


class EnhancedMCPToolAdapter(LLMTool):
    """Enhanced adapter for MCP tools with proper validation and metadata"""
    
    def __init__(self, tool_metadata: ToolMetadata, mcp_wrapper: MCPClientWrapper):
        """Initialize adapter from registry metadata"""
        # Use metadata from registry
        self.name = tool_metadata.name
        self.description = tool_metadata.description
        self.input_schema = tool_metadata.input_schema
        
        # Store additional metadata for rich functionality
        self.category = tool_metadata.category
        self.tags = tool_metadata.tags
        self.deprecated = tool_metadata.deprecated
        self.method = tool_metadata.method
        self.path = tool_metadata.path
        self.operation_id = tool_metadata.operation_id
        self.auth_required = tool_metadata.auth_required
        self.examples = tool_metadata.examples
        
        self.mcp_wrapper = mcp_wrapper
        self.tool_metadata = tool_metadata
        self.logger = logging.getLogger(f'MCPTool.{self.name}')
        
    def run_impl(self, tool_input: Dict[str, Any], message_history=None) -> ToolImplOutput:
        """Execute MCP tool with proper async handling"""
        # Create new event loop for synchronous context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Log deprecation warning
            if self.deprecated:
                self.logger.warning(f"Using deprecated tool: {self.name}")
            
            # Execute tool through wrapper (validation happens there)
            result = loop.run_until_complete(
                self.mcp_wrapper.execute_tool(self.name, tool_input)
            )
            
            # Format result based on response type
            if isinstance(result, dict):
                if result.get('error'):
                    return ToolImplOutput(
                        tool_output=f"Error: {result.get('message', 'Unknown error')}",
                        tool_result_message=f"Tool '{self.name}' failed"
                    )
                output = json.dumps(result, indent=2)
                message = result.get('status', f"Tool '{self.name}' completed successfully")
            else:
                output = str(result)
                message = f"Tool '{self.name}' completed successfully"
            
            return ToolImplOutput(
                tool_output=output,
                tool_result_message=message
            )
            
        except ValueError as e:
            # Validation errors from the wrapper
            return ToolImplOutput(
                tool_output=f"Validation error: {str(e)}",
                tool_result_message=f"Tool '{self.name}' failed validation"
            )
        except Exception as e:
            self.logger.error(f"Tool execution error: {e}", exc_info=True)
            return ToolImplOutput(
                tool_output=f"Execution error: {str(e)}",
                tool_result_message=f"Tool '{self.name}' failed with exception"
            )
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    
    def get_tool_start_message(self, tool_input: Dict[str, Any]) -> str:
        """Generate informative start message"""
        tags_str = f" [{', '.join(self.tags)}]" if self.tags else ""
        deprecated_str = " [DEPRECATED]" if self.deprecated else ""
        return f"Executing {self.method} {self.name}{tags_str}{deprecated_str} ({self.category})"
    
    def get_detailed_help(self) -> str:
        """Get detailed help information about the tool"""
        help_lines = [
            f"Tool: {self.name}",
            f"Description: {self.description}",
            f"Category: {self.category}",
            f"Method: {self.method} {self.path}",
            f"Tags: {', '.join(self.tags) if self.tags else 'None'}",
            f"Deprecated: {'Yes' if self.deprecated else 'No'}",
            f"Auth Required: {'Yes' if self.auth_required else 'No'}",
        ]
        
        if self.operation_id:
            help_lines.append(f"Operation ID: {self.operation_id}")
        
        if self.examples:
            help_lines.append("\nExamples:")
            for example_name, example_data in self.examples.items():
                help_lines.append(f"  {example_name}: {json.dumps(example_data, indent=4)}")
        
        return "\n".join(help_lines)
    
    def __repr__(self):
        return f"<MCPToolAdapter {self.name} [{self.category}]>"


def create_mcp_tool_adapters(mcp_wrapper: MCPClientWrapper, 
                           include_deprecated: bool = False,
                           categories: Optional[List[str]] = None,
                           tags: Optional[List[str]] = None) -> List[EnhancedMCPToolAdapter]:
    """
    Create MCP tool adapters from the wrapper's registry
    
    Args:
        mcp_wrapper: Initialized MCP client wrapper
        include_deprecated: Whether to include deprecated tools
        categories: Filter by specific categories
        tags: Filter by specific tags
        
    Returns:
        List of EnhancedMCPToolAdapter instances
    """
    adapters = []
    
    # Get tools from registry with filters
    tools = mcp_wrapper.tool_registry.list_tools(include_deprecated=include_deprecated)
    
    for tool_metadata in tools:
        # Apply category filter
        if categories and tool_metadata.category not in categories:
            continue
            
        # Apply tag filter
        if tags and not any(tag in tool_metadata.tags for tag in tags):
            continue
        
        try:
            adapter = EnhancedMCPToolAdapter(tool_metadata, mcp_wrapper)
            adapters.append(adapter)
        except Exception as e:
            logging.error(f"Failed to create adapter for tool '{tool_metadata.name}': {e}")
    
    return adapters


class BankingCoreToolAdapter(LLMTool):
    """Special adapter for the three core banking tools"""
    
    def __init__(self, tool_type: str, mcp_wrapper: MCPClientWrapper):
        self.mcp_wrapper = mcp_wrapper
        self.tool_type = tool_type
        self.logger = logging.getLogger(f'BankingCore.{tool_type}')
        
        # Define the three core tools
        if tool_type == "list_apis":
            self.name = "list_banking_apis"
            self.mcp_tool_name = "list_api_endpoints"
            self.description = """Discover available banking APIs. Use filters to narrow results:
            - category: Filter by category (accounts, transfers, cards, loans, admin)
            - search: Search term to filter APIs
            Always use this tool first unless you know the exact API."""
            self.input_schema = {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["accounts", "transfers", "cards", "loans", "admin", "all"],
                        "description": "Filter APIs by category"
                    },
                    "search": {
                        "type": "string",
                        "description": "Search term to filter APIs"
                    }
                },
                "required": []
            }
            
        elif tool_type == "get_structure":
            self.name = "get_api_structure"
            self.mcp_tool_name = "get_api_endpoint_schema"
            self.description = "Get detailed request/response structure for a specific API"
            self.input_schema = {
                "type": "object",
                "properties": {
                    "api_name": {
                        "type": "string",
                        "description": "Name of the API to get structure for"
                    }
                },
                "required": ["api_name"]
            }
            
        elif tool_type == "invoke":
            self.name = "invoke_banking_api"
            self.mcp_tool_name = "invoke_banking_endpoint"
            self.description = "Execute a banking API with validated parameters"
            self.input_schema = {
                "type": "object",
                "properties": {
                    "api_name": {
                        "type": "string",
                        "description": "Name of the API to invoke"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Parameters for the API call"
                    }
                },
                "required": ["api_name", "parameters"]
            }
    
    def run(self, tool_input: Dict[str, Any], message_history=None) -> str:
        """Run method that returns string as expected by tool_manager"""
        impl_result = self.run_impl(tool_input, message_history)
        return impl_result.tool_output
    
    def run_impl(self, tool_input: Dict[str, Any], message_history=None) -> ToolImplOutput:
        """Execute the core banking tool"""
        try:
            if self.tool_type == "list_apis":
                return self._run_list_apis(tool_input)
            elif self.tool_type == "get_structure":
                return self._run_get_structure(tool_input)
            elif self.tool_type == "invoke":
                return self._run_invoke_api(tool_input)
        except Exception as e:
            self.logger.error(f"Tool execution error: {e}")
            return ToolImplOutput(
                tool_output=f"Error: {str(e)}",
                tool_result_message=f"Tool '{self.name}' failed"
            )
    
    def _run_list_apis(self, tool_input: Dict[str, Any]) -> ToolImplOutput:
        """List available APIs with filtering"""
        category = tool_input.get('category')
        search = tool_input.get('search')
        
        # Use the registry directly
        if category and category != 'all':
            tools = mcp_tool_registry.list_tools(category=category)
        elif search:
            tools = mcp_tool_registry.search_tools(search)
        else:
            tools = mcp_tool_registry.list_tools()
        
        # Format output
        output = f"Found {len(tools)} APIs:\n\n"
        by_category = {}
        for tool in tools:
            cat = tool.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(tool)
        
        for cat, cat_tools in sorted(by_category.items()):
            output += f"{cat.upper()} ({len(cat_tools)} APIs):\n"
            for tool in cat_tools:
                output += f"  - {tool.name}: {tool.description}\n"
            output += "\n"
        
        return ToolImplOutput(
            tool_output=output,
            tool_result_message=f"Listed {len(tools)} available APIs"
        )
    
    def _run_get_structure(self, tool_input: Dict[str, Any]) -> ToolImplOutput:
        """Get API structure"""
        api_name = tool_input['api_name']
        tool_metadata = mcp_tool_registry.get_tool(api_name)
        
        if not tool_metadata:
            return ToolImplOutput(
                tool_output=f"API '{api_name}' not found",
                tool_result_message="API not found"
            )
        
        # Format structure
        output = f"API: {api_name}\n"
        output += f"Description: {tool_metadata.description}\n"
        output += f"Category: {tool_metadata.category}\n\n"
        output += "Parameters:\n"
        
        schema = tool_metadata.input_schema
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        
        for param, details in properties.items():
            req = " (REQUIRED)" if param in required else ""
            output += f"  - {param}{req}: {details.get('type')} - {details.get('description', '')}\n"
        
        return ToolImplOutput(
            tool_output=output,
            tool_result_message=f"Retrieved structure for {api_name}"
        )
    
    def _run_invoke_api(self, tool_input: Dict[str, Any]) -> ToolImplOutput:
        """Invoke an API through MCP"""
        api_name = tool_input['api_name']
        parameters = tool_input.get('parameters', {})
        
        # Execute through MCP wrapper
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                self.mcp_wrapper.execute_tool(api_name, parameters)
            )
            
            output = json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
            
            return ToolImplOutput(
                tool_output=output,
                tool_result_message=f"Successfully executed {api_name}"
            )
        finally:
            loop.close()
    
    


# Add this function to mcp_tool_adapter.py:

def create_banking_core_tools(mcp_wrapper: MCPClientWrapper) -> List[LLMTool]:
    """Create adapters that wrap the actual MCP tools with banking-friendly names"""
    logger = logging.getLogger(__name__)
    
    # Get the actual registered tools from the registry
    actual_tools = mcp_tool_registry.list_tools()
    logger.info(f"DEBUG: MCP Registry has {len(actual_tools)} tools")
    
    # Log first few tool names
    for i, tool in enumerate(actual_tools[:5]):
        logger.info(f"DEBUG: Tool {i}: {tool.name}")
    
    tool_map = {tool.name: tool for tool in actual_tools}
    
    adapters = []
    
    # Check what tools we're looking for vs what we have
    tool_mappings = {
        'list_api_endpoints': 'list_banking_apis',
        'get_api_endpoint_schema': 'get_api_structure',
        'invoke_api_endpoint': 'invoke_banking_api'
    }
    
    logger.info(f"DEBUG: Looking for tools: {list(tool_mappings.keys())}")
    logger.info(f"DEBUG: Available tools: {list(tool_map.keys())[:10]}")
    
    for mcp_name, banking_name in tool_mappings.items():
        if mcp_name in tool_map:
            adapter = EnhancedMCPToolAdapter(tool_map[mcp_name], mcp_wrapper)
            adapter.name = banking_name
            adapter.tool_metadata.name = banking_name
            adapters.append(adapter)
            logger.info(f"✓ Created adapter: '{banking_name}' from '{mcp_name}'")
        else:
            logger.error(f"✗ Expected MCP tool '{mcp_name}' not found in registry")
    
    logger.info(f"DEBUG: Created {len(adapters)} banking adapters")
    return adapters


# Modify the existing create_mcp_tool_adapters function
def create_mcp_tool_adapters(
    mcp_wrapper: MCPClientWrapper = None,
    registry: MCPToolRegistry = None,
    include_deprecated: bool = False,
    categories: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    banking_mode: bool = False  # Add this parameter
) -> List[LLMTool]:
    """
    Create MCP tool adapters
    
    If banking_mode is True, returns only the three core banking tools.
    Otherwise, creates adapters for all tools in the registry.
    """
    if banking_mode and mcp_wrapper:
        # For banking mode, create the tools directly
        from src.ii_agent.tools.banking_core_tools import (
            ListBankingAPIs, 
            GetAPIStructure, 
            InvokeBankingAPI
        )
        
        # Create the banking tools directly
        banking_tools = [
            ListBankingAPIs(mcp_wrapper),
            GetAPIStructure(mcp_wrapper),
            InvokeBankingAPI(mcp_wrapper)
        ]
        
        logging.info(f"Created {len(banking_tools)} banking tools directly")
        for tool in banking_tools:
            logging.info(f"  - {tool.name}")
        
        return banking_tools
    
    # Original logic for non-banking mode
    adapters = []
    
    # Use provided registry or wrapper's registry
    if registry:
        tools = registry.list_tools(include_deprecated=include_deprecated)
    elif mcp_wrapper:
        tools = mcp_wrapper.tool_registry.list_tools(include_deprecated=include_deprecated)
    else:
        raise ValueError("Either registry or mcp_wrapper must be provided")
    
    # Continue with original filtering and adapter creation...
    for tool_metadata in tools:
        if categories and tool_metadata.category not in categories:
            continue
        if tags and not any(tag in tool_metadata.tags for tag in tags):
            continue
        
        try:
            adapter = EnhancedMCPToolAdapter(tool_metadata, mcp_wrapper)
            adapters.append(adapter)
        except Exception as e:
            logging.error(f"Failed to create adapter for tool '{tool_metadata.name}': {e}")
    
    return adapters