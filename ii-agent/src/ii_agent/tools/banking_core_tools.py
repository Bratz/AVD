"""
Core Banking Tools for TCS BaNCS Platform
Provides wrapper tools that understand the two-tier MCP structure
"""

from datetime import datetime
import json
import logging
from typing import Dict, Any, List, Optional
from src.ii_agent.tools.base import LLMTool, ToolImplOutput
from src.ii_agent.llm.message_history import MessageHistory

import nest_asyncio
nest_asyncio.apply()


class ListBankingAPIs(LLMTool):
    """Tool for discovering available banking API endpoints"""
    
    name = "list_banking_apis"
    description = """Discover available banking API endpoints using the MCP server.
    This searches through actual banking APIs (not MCP tools).

    IMPORTANT: Start with list_banking_apis() (no parameters) to see available tags!

    Use filters:
    - search_query: Search in API names/descriptions (use SINGLE WORDS: "balance", "transfer")
    - tag: Filter by category - use LOWERCASE tags from available_tags list!
    - method: Filter by HTTP method (GET, POST, etc.)

    Common lowercase tags: customermanagement, accountmanagement, loans, deposits
    
    Example workflow:
    1. list_banking_apis() -> see available_tags
    2. list_banking_apis(tag="customermanagement", search_query="details")"""
    
    input_schema = {
        "type": "object",
        "properties": {
            "search_query": {
                "type": "string",
                "description": "Search term - use SINGLE WORDS (e.g., 'balance' not 'account balance')"
            },
            "tag": {
                "type": "string",
                "description": "Filter by category - use LOWERCASE (e.g., 'customermanagement' not 'CustomerManagement')"
            },
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", ""],
                "description": "Filter by HTTP method"
            },
            "include_deprecated": {
                "type": "boolean",
                "description": "Include deprecated APIs",
                "default": False
            }
        },
        "required": []
    }

    def __init__(self, mcp_wrapper):
        self.mcp_wrapper = mcp_wrapper
        self.logger = logging.getLogger("BankingTools.ListAPIs")
    
        
    async def run_impl_async(self, tool_input: Dict[str, Any], message_history: MessageHistory) -> ToolImplOutput:
        """Execute API discovery through MCP's list_api_endpoints"""
        try:
            # Map our input to MCP tool parameters
            mcp_params = {
                "search_query": tool_input.get('search_query', ''),
                "tag": tool_input.get('tag', ''),
                "method": tool_input.get('method', ''),
                "include_deprecated": tool_input.get('include_deprecated', False)
            }
            
            # Call the actual MCP tool
            result = await self.mcp_wrapper.execute_tool('list_api_endpoints', mcp_params)
            
                    # Format the response concisely
            if isinstance(result, dict) and 'apis' in result:
                apis = result.get('apis', [])
                
                # Group by tags/categories
                by_category = {}
                for api in apis:
                    # Handle both dict and string formats
                    if isinstance(api, dict):
                        name = api.get('name', 'Unknown')
                        tags = api.get('tags', ['Other'])
                        category = tags[0] if tags else 'Other'
                    else:
                        name = str(api)
                        category = 'Other'
                    
                    if category not in by_category:
                        by_category[category] = []
                    by_category[category].append(name)
                
                # Build concise output
                output_lines = [f"Found {len(apis)} APIs:\n"]
                
                for category, names in sorted(by_category.items()):
                    output_lines.append(f"\n{category} ({len(names)} APIs):")
                    # Just list the names, max 10 per category
                    for name in names[:10]:
                        output_lines.append(f"  - {name}")
                    if len(names) > 10:
                        output_lines.append(f"  ... and {len(names) - 10} more")
                
                output = "\n".join(output_lines)

            # The result should contain the discovered APIs
            if isinstance(result, dict) and 'error' in result:
                return ToolImplOutput(
                    tool_output=f"Error discovering APIs: {result.get('error', 'Unknown error')}",
                    tool_result_message="API discovery failed"
                )
            
            # Format the response for the agent
            return ToolImplOutput(
                tool_output=result if isinstance(result, str) else json.dumps(result, indent=2),
                tool_result_message="Successfully discovered banking APIs"
            )
            
        except Exception as e:
            self.logger.error(f"Error listing APIs: {e}")
            return ToolImplOutput(
                tool_output=f"Error listing APIs: {str(e)}",
                tool_result_message="Failed to list APIs"
            )
    
    def run_impl(self, tool_input: Dict[str, Any], message_history: MessageHistory) -> ToolImplOutput:
        """Sync wrapper for async implementation"""
        import asyncio
        # loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(loop)
        # try:
        #     return loop.run_until_complete(self.run_impl_async(tool_input, message_history))
        # finally:
        #     loop.close()
        return asyncio.run(self.run_impl_async(tool_input, message_history))


class GetAPIStructure(LLMTool):
    """Tool for getting detailed API endpoint structure"""
    
    name = "get_api_structure"
    description = """Get detailed schema for a specific banking API endpoint.
    Use this after discovering APIs with list_banking_apis.
    
    Provide either:
    - endpoint_name: The API name exactly as returned by list_banking_apis
    - operation_id: The OpenAPI operation ID
    
    Example:
    get_api_structure(endpoint_name="cbpet0108_fetch_bpdetails_postusing_get")"""
    
    input_schema = {
        "type": "object",
        "properties": {
            "endpoint_name": {
                "type": "string",
                "description": "Name of the API endpoint"
            },
            "operation_id": {
                "type": "string",
                "description": "OpenAPI operation ID (alternative to endpoint_name)"
            }
        },
        "required": []
    }
    
    def __init__(self, mcp_wrapper):
        self.mcp_wrapper = mcp_wrapper
        self.logger = logging.getLogger("BankingTools.GetStructure")
        
    async def run_impl_async(self, tool_input: Dict[str, Any], message_history: MessageHistory) -> ToolImplOutput:
        """Get API structure through MCP's get_api_endpoint_schema"""
        try:
            # Use the MCP tool
            result = await self.mcp_wrapper.execute_tool('get_api_endpoint_schema', tool_input)
            
            if isinstance(result, dict) and 'error' in result:
                return ToolImplOutput(
                    tool_output=f"Error getting API structure: {result.get('error', 'Unknown error')}",
                    tool_result_message="Failed to get API structure"
                )
            
            return ToolImplOutput(
                tool_output=result if isinstance(result, str) else json.dumps(result, indent=2),
                tool_result_message="Successfully retrieved API structure"
            )
            
        except Exception as e:
            self.logger.error(f"Error getting API structure: {e}")
            return ToolImplOutput(
                tool_output=f"Error getting API structure: {str(e)}",
                tool_result_message="Failed to get API structure"
            )
    
    def run_impl(self, tool_input: Dict[str, Any], message_history: MessageHistory) -> ToolImplOutput:
        """Sync wrapper for async implementation"""
        import asyncio
        # loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(loop)
        # try:
        #     return loop.run_until_complete(self.run_impl_async(tool_input, message_history))
        # finally:
        #     loop.close()
        return asyncio.run(self.run_impl_async(tool_input, message_history))

class InvokeBankingAPI(LLMTool):
    """Tool for executing banking API endpoints"""
    
    name = "invoke_banking_api"
    description = """Execute a banking API endpoint discovered through list_banking_apis.
    
    Required:
    - endpoint_name OR operation_id: The API to execute
    - params: ALL parameters in a single object (headers, query, body all go here)
    
    Example:
    invoke_banking_api(
        endpoint_name="cbpet0108_fetch_bpdetails_postusing_get",
        params={"CustomerID": 100205}
    )"""
    
    input_schema = {
        "type": "object",
        "properties": {
            "endpoint_name": {
                "type": "string",
                "description": "Name of the API endpoint to invoke"
            },
            "operation_id": {
                "type": "string",
                "description": "OpenAPI operation ID (alternative to endpoint_name)"
            },
            "params": {
                "type": "object",
                "description": "Parameters for the API call"
            }
        },
        "required": ["params"]
    }
    
    def __init__(self, mcp_wrapper):
        self.mcp_wrapper = mcp_wrapper
        self.logger = logging.getLogger("BankingTools.InvokeAPI")
        self._execution_history = []
        
    async def run_impl_async(self, tool_input: Dict[str, Any], message_history: MessageHistory) -> ToolImplOutput:
        """Execute API through MCP's invoke_api_endpoint"""
        try:
            # Use the MCP tool
            result = await self.mcp_wrapper.execute_tool('invoke_api_endpoint', tool_input)
            
            # Track execution
            self._execution_history.append({
                'endpoint': tool_input.get('endpoint_name', tool_input.get('operation_id')),
                'params': tool_input.get('params', {}),
                'timestamp': datetime.now().isoformat(),
                'success': not (isinstance(result, dict) and 'error' in result)
            })
            
            if isinstance(result, dict) and 'error' in result:
                return ToolImplOutput(
                    tool_output=f"API execution failed: {result.get('error', 'Unknown error')}",
                    tool_result_message="API execution failed"
                )
            
            # Format response
            response = "API executed successfully!\n\n"
            response += "Result:\n"
            response += result if isinstance(result, str) else json.dumps(result, indent=2)
            
            return ToolImplOutput(
                tool_output=response,
                tool_result_message="Successfully executed API"
            )
            
        except Exception as e:
            self.logger.error(f"Error executing API: {e}")
            
            # Track failed execution
            self._execution_history.append({
                'endpoint': tool_input.get('endpoint_name', tool_input.get('operation_id')),
                'params': tool_input.get('params', {}),
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            })
            
            return ToolImplOutput(
                tool_output=f"Error executing API: {str(e)}",
                tool_result_message="Failed to execute API"
            )
    
    def run_impl(self, tool_input: Dict[str, Any], message_history: MessageHistory) -> ToolImplOutput:
        """Sync wrapper for async implementation"""
        import asyncio
        # loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(loop)
        # try:
        #     return loop.run_until_complete(self.run_impl_async(tool_input, message_history))
        # finally:
        #     loop.close()
        return asyncio.run(self.run_impl_async(tool_input, message_history))

def get_banking_core_tools(mcp_wrapper) -> List[LLMTool]:
    """Get the three core banking tools that wrap MCP's two-tier structure"""
    return [
        ListBankingAPIs(mcp_wrapper),
        GetAPIStructure(mcp_wrapper),
        InvokeBankingAPI(mcp_wrapper)
    ]