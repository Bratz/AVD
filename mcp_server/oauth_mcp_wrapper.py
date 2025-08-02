#!/usr/bin/env python3
# oauth_mcp_wrapper.py
"""
OAuth-compliant wrapper for Banking MCP Server with tool discovery and audience fix
"""

import os
import logging
import asyncio
import json
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import jwt
from jwt.algorithms import RSAAlgorithm
import uvicorn
from dotenv import load_dotenv
from cachetools import TTLCache

# Import existing server components
try:
    from mcp_server.server import BankingMCPServer
    from mcp_server.tools.tcs_bancs_real_tools import register_bancs_tools
except ImportError:
    try:
        from server import BankingMCPServer
        from tools.tcs_bancs_real_tools import register_bancs_tools
    except ImportError:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from mcp_server.server import BankingMCPServer
        from mcp_server.tools.tcs_bancs_real_tools import register_bancs_tools

load_dotenv()

# Configuration
MCP_ENDPOINT = os.getenv("MCP_HTTP_ENDPOINT", "http://localhost:8082/mcp")
parsed_url = urlparse(MCP_ENDPOINT)
MCP_PORT = parsed_url.port or 8082
MCP_BASE_URL = f"{parsed_url.scheme}://{parsed_url.hostname}:{MCP_PORT}"

KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "http://localhost:8080")
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "banking-mcp")
ENABLE_OAUTH = os.getenv("ENABLE_OAUTH", "false").lower() == "true"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
app_state = {
    "mcp_server": None,
    "mcp_handler": None,
    "oauth_validator": None,
    "jwks_cache": {},
    "jwks_last_refresh": None,
    "token_cache": TTLCache(maxsize=100, ttl=300)
}

class SimpleOAuthValidator:
    """Simple OAuth validator with JWKS support"""
    
    def __init__(self, keycloak_url: str, realm: str):
        self.keycloak_url = keycloak_url
        self.realm = realm
        self.jwks_url = f"{keycloak_url}/realms/{realm}/protocol/openid-connect/certs"
        self.issuer = f"{keycloak_url}/realms/{realm}"
        
    async def fetch_jwks(self) -> None:
        """Fetch JWKS from Keycloak"""
        if app_state["jwks_last_refresh"] and datetime.now() - app_state["jwks_last_refresh"] < timedelta(hours=1):
            return
            
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(self.jwks_url)
                resp.raise_for_status()
                jwks_data = resp.json()
                
                app_state["jwks_cache"].clear()
                
                for key_data in jwks_data.get("keys", []):
                    kid = key_data.get("kid")
                    if kid and key_data.get("kty") == "RSA":
                        try:
                            app_state["jwks_cache"][kid] = RSAAlgorithm.from_jwk(json.dumps(key_data))
                        except Exception as e:
                            logger.error(f"Failed to parse key {kid}: {e}")
                
                app_state["jwks_last_refresh"] = datetime.now()
                logger.info(f"Loaded {len(app_state['jwks_cache'])} keys from JWKS")
                
        except Exception as e:
            logger.error(f"Failed to fetch JWKS: {e}")
            if not app_state["jwks_cache"]:
                raise
    
    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate OAuth token with audience validation disabled"""
        cached = app_state["token_cache"].get(token)
        if cached:
            return cached
            
        try:
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            
            if not kid:
                logger.error("Token missing kid in header")
                return None
            
            if kid not in app_state["jwks_cache"]:
                await self.fetch_jwks()
                
            if kid not in app_state["jwks_cache"]:
                logger.error(f"Unknown kid: {kid}")
                return None
            
            # First decode without validation to see what's in the token
            try:
                claims_preview = jwt.decode(
                    token,
                    options={"verify_signature": False}
                )
                logger.info(f"Token audience: {claims_preview.get('aud')}")
                logger.info(f"Token azp: {claims_preview.get('azp')}")
            except:
                pass
            
            # Decode with audience validation disabled
            try:
                claims = jwt.decode(
                    token,
                    app_state["jwks_cache"][kid],
                    algorithms=["RS256"],
                    issuer=self.issuer,
                    options={
                        "verify_exp": True, 
                        "verify_iss": True,
                        "verify_aud": False  # Disable audience validation
                    }
                )
            except jwt.ExpiredSignatureError:
                logger.warning("Token expired")
                return None
            except jwt.InvalidTokenError as e:
                logger.warning(f"Invalid token: {e}")
                return None
            
            user_info = {
                "sub": claims.get("sub"),
                "username": claims.get("preferred_username", claims.get("sub")),
                "email": claims.get("email"),
                "scopes": claims.get("scope", "").split(),
                "roles": claims.get("realm_access", {}).get("roles", []),
                "exp": claims.get("exp")
            }
            
            app_state["token_cache"][token] = user_info
            return user_info
            
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return None

class MCPRequestHandler:
    """Handles MCP JSON-RPC requests with proper FastMCP integration"""
    
    def __init__(self, mcp_server: BankingMCPServer):
        self.mcp_server = mcp_server
        self.mcp = mcp_server.mcp
        self._discover_tools_and_prompts()
        
    def _discover_tools_and_prompts(self):
        """Discover where tools and prompts are stored"""
        self.tools_location = None
        self.prompts_location = None
        self.tools_getter = None
        self.prompts_getter = None
        
        logger.info("=== Tool and Prompt Discovery ===")
        
        # Check for _tool_manager first (FastMCP specific)
        if hasattr(self.mcp, '_tool_manager'):
            tm = self.mcp._tool_manager
            logger.info(f"Found _tool_manager: {type(tm)}")
            
            # Check different ways to get tools from ToolManager
            if hasattr(tm, '_tools'):
                self.tools_location = 'tool_manager._tools'
                self.tools_getter = lambda: tm._tools
                logger.info(f"  _tool_manager._tools: {len(tm._tools)} tools")
                logger.info(f"  Tool names: {list(tm._tools.keys())}")
            elif hasattr(tm, 'tools'):
                if callable(getattr(tm, 'tools', None)):
                    try:
                        tools_dict = tm.tools()
                        self.tools_location = 'tool_manager.tools()'
                        self.tools_getter = tm.tools
                        logger.info(f"  _tool_manager.tools(): {len(tools_dict)} tools")
                    except:
                        pass
                else:
                    self.tools_location = 'tool_manager.tools'
                    self.tools_getter = lambda: tm.tools
                    logger.info(f"  _tool_manager.tools: {len(tm.tools)} tools")
        
        # Check for _prompt_manager (FastMCP specific)
        if hasattr(self.mcp, '_prompt_manager'):
            pm = self.mcp._prompt_manager
            logger.info(f"Found _prompt_manager: {type(pm)}")
            
            # Check different ways to get prompts from PromptManager
            if hasattr(pm, '_prompts'):
                self.prompts_location = 'prompt_manager._prompts'
                self.prompts_getter = lambda: pm._prompts
                logger.info(f"  _prompt_manager._prompts: {len(pm._prompts)} prompts")
                logger.info(f"  Prompt names: {list(pm._prompts.keys())}")
            elif hasattr(pm, 'prompts'):
                if callable(getattr(pm, 'prompts', None)):
                    try:
                        prompts_dict = pm.prompts()
                        self.prompts_location = 'prompt_manager.prompts()'
                        self.prompts_getter = pm.prompts
                        logger.info(f"  _prompt_manager.prompts(): {len(prompts_dict)} prompts")
                    except:
                        pass
                else:
                    self.prompts_location = 'prompt_manager.prompts'
                    self.prompts_getter = lambda: pm.prompts
                    logger.info(f"  _prompt_manager.prompts: {len(pm.prompts)} prompts")
        
        # Check for get_tools() and get_prompts() methods on MCP object
        if not self.tools_location and hasattr(self.mcp, 'get_tools'):
            try:
                tools_dict = self.mcp.get_tools()
                if tools_dict:
                    self.tools_location = 'get_tools()'
                    self.tools_getter = self.mcp.get_tools
                    logger.info(f"Tools found via get_tools(): {len(tools_dict)} tools")
                    logger.info(f"  Tool names: {list(tools_dict.keys())}")
            except Exception as e:
                logger.error(f"Error calling get_tools(): {e}")
        
        if not self.prompts_location and hasattr(self.mcp, 'get_prompts'):
            try:
                prompts_dict = self.mcp.get_prompts()
                if prompts_dict:
                    self.prompts_location = 'get_prompts()'
                    self.prompts_getter = self.mcp.get_prompts
                    logger.info(f"Prompts found via get_prompts(): {len(prompts_dict)} prompts")
                    logger.info(f"  Prompt names: {list(prompts_dict.keys())}")
            except Exception as e:
                logger.error(f"Error calling get_prompts(): {e}")
        
        # Check for _server (older FastMCP versions)
        if not self.tools_location and hasattr(self.mcp, '_server'):
            server = self.mcp._server
            logger.info(f"Found _server: {type(server)}")
            
            if hasattr(server, 'tool_manager'):
                tm = server.tool_manager
                logger.info(f"  Found tool_manager: {type(tm)}")
                if hasattr(tm, 'tools'):
                    self.tools_location = 'server.tool_manager'
                    self.tools_getter = lambda: tm.tools
                    logger.info(f"    tool_manager.tools: {len(tm.tools)} tools")
        
        # Check for _registry
        if not self.tools_location and hasattr(self.mcp, '_registry'):
            registry = self.mcp._registry
            logger.info(f"Found _registry: {type(registry)}")
            
            if hasattr(registry, 'tools'):
                self.tools_location = 'registry'
                self.tools_getter = lambda: registry.tools
                logger.info(f"  _registry.tools: {len(registry.tools)} tools")
                
            if hasattr(registry, 'prompts'):
                self.prompts_location = 'registry'
                self.prompts_getter = lambda: registry.prompts
                logger.info(f"  _registry.prompts: {len(registry.prompts)} prompts")
        
        logger.info(f"Discovery complete - Tools: {self.tools_location}, Prompts: {self.prompts_location}")
        logger.info("=== End Tool Discovery ===")
    
    def _get_tools(self) -> Dict[str, Any]:
        """Get tools based on discovered location"""
        if self.tools_getter:
            try:
                return self.tools_getter()
            except Exception as e:
                logger.error(f"Error getting tools: {e}")
        return {}
    
    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts based on discovered location"""
        if self.prompts_getter:
            try:
                return self.prompts_getter()
            except Exception as e:
                logger.error(f"Error getting prompts: {e}")
        return {}
        
    async def handle_request(self, json_body: Dict[str, Any], user_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process MCP JSON-RPC request"""
        method = json_body.get("method", "")
        params = json_body.get("params", {})
        request_id = json_body.get("id")
        
        try:
            if method == "initialize":
                return await self._handle_initialize(params, request_id)
            elif method == "tools/list":
                return await self._handle_tools_list(request_id)
            elif method == "tools/call":
                return await self._handle_tool_call(params, request_id, user_info)
            elif method == "prompts/list":
                return await self._handle_prompts_list(request_id)
            elif method == "prompts/get":
                return await self._handle_prompt_get(params, request_id)
            else:
                return self._error_response(-32601, f"Method not found: {method}", request_id)
                
        except Exception as e:
            logger.error(f"Request handling error: {e}", exc_info=True)
            return self._error_response(-32603, f"Internal error: {str(e)}", request_id)
    
    async def _handle_initialize(self, params: Dict[str, Any], request_id: Any) -> Dict[str, Any]:
        """Handle initialize method"""
        return {
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": True},
                    "resources": {"subscribe": False, "listChanged": False},
                    "prompts": {"listChanged": True},
                    "logging": {}
                },
                "serverInfo": {
                    "name": "BankingMCP",
                    "version": "2.0.0"
                }
            },
            "id": request_id
        }
    
    async def _handle_tools_list(self, request_id: Any) -> Dict[str, Any]:
        """Handle tools/list method"""
        tools = []
        tool_registry = self._get_tools()
        
        for tool_name, tool_info in tool_registry.items():
            try:
                # Handle FunctionTool from FastMCP
                if hasattr(tool_info, '__class__') and tool_info.__class__.__name__ == 'FunctionTool':
                    # Extract tool definition from FunctionTool
                    tool_def = {
                        "name": tool_name,
                        "description": getattr(tool_info, 'description', '') or f"Tool {tool_name}",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                    
                    # Try to get schema from FunctionTool
                    if hasattr(tool_info, 'schema'):
                        schema = tool_info.schema
                        if isinstance(schema, dict):
                            tool_def["description"] = schema.get("description", tool_def["description"])
                            tool_def["inputSchema"] = schema.get("parameters", tool_def["inputSchema"])
                    
                    # Try to extract from function if available
                    if hasattr(tool_info, 'fn') or hasattr(tool_info, 'func') or hasattr(tool_info, 'handler'):
                        func = getattr(tool_info, 'fn', None) or getattr(tool_info, 'func', None) or getattr(tool_info, 'handler', None)
                        if func and callable(func):
                            # Extract from function signature
                            extracted = self._extract_tool_definition(tool_name, func)
                            # Use extracted description if better
                            if extracted["description"] and extracted["description"] != f"Tool {tool_name}":
                                tool_def["description"] = extracted["description"]
                            # Use extracted schema if more detailed
                            if extracted["inputSchema"]["properties"]:
                                tool_def["inputSchema"] = extracted["inputSchema"]
                    
                    tools.append(tool_def)
                    
                elif hasattr(tool_info, 'schema') and hasattr(tool_info, 'handler'):
                    # FastMCP format with schema
                    schema = tool_info.schema
                    tools.append({
                        "name": tool_name,
                        "description": schema.get("description", f"Tool {tool_name}"),
                        "inputSchema": schema.get("parameters", {
                            "type": "object",
                            "properties": {},
                            "required": []
                        })
                    })
                elif callable(tool_info):
                    # Plain function format
                    tools.append(self._extract_tool_definition(tool_name, tool_info))
                elif isinstance(tool_info, dict) and "description" in tool_info:
                    # Already formatted as dict
                    tools.append(tool_info)
                else:
                    logger.warning(f"Unknown tool format for {tool_name}: {type(tool_info)}")
                    # Try to extract anything we can
                    fallback_tool = {
                        "name": tool_name,
                        "description": str(getattr(tool_info, 'description', f"Tool {tool_name}")),
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                    tools.append(fallback_tool)
                    
            except Exception as e:
                logger.error(f"Error processing tool {tool_name}: {e}", exc_info=True)
        
        logger.info(f"Returning {len(tools)} tools")
        return {
            "jsonrpc": "2.0",
            "result": {"tools": tools},
            "id": request_id
        }
    
    async def _handle_prompts_list(self, request_id: Any) -> Dict[str, Any]:
        """Handle prompts/list method"""
        prompts = []
        prompt_registry = self._get_prompts()
        
        for prompt_name, prompt_info in prompt_registry.items():
            try:
                # Handle PromptArgument objects
                arguments = []
                if hasattr(prompt_info, 'arguments'):
                    for arg in prompt_info.arguments:
                        if hasattr(arg, '__dict__'):
                            # Convert PromptArgument to dict
                            arguments.append({
                                "name": getattr(arg, 'name', ''),
                                "description": getattr(arg, 'description', ''),
                                "required": getattr(arg, 'required', False)
                            })
                        elif isinstance(arg, dict):
                            arguments.append(arg)
                
                prompts.append({
                    "name": prompt_name,
                    "description": getattr(prompt_info, 'description', f"Prompt {prompt_name}"),
                    "arguments": arguments
                })
            except Exception as e:
                logger.error(f"Error processing prompt {prompt_name}: {e}")
        
        return {
            "jsonrpc": "2.0",
            "result": {"prompts": prompts},
            "id": request_id
        }
    
    async def _handle_tool_call(self, params: Dict[str, Any], request_id: Any, user_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle tools/call method"""
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})
        
        tool_registry = self._get_tools()
        if tool_name not in tool_registry:
            return self._error_response(-32601, f"Unknown tool: {tool_name}", request_id)
        
        tool_info = tool_registry[tool_name]
        tool_func = None
        
        # Extract function from different formats
        if hasattr(tool_info, '__class__') and tool_info.__class__.__name__ == 'FunctionTool':
            # Handle FunctionTool from FastMCP
            if hasattr(tool_info, 'fn'):
                tool_func = tool_info.fn
            elif hasattr(tool_info, 'func'):
                tool_func = tool_info.func
            elif hasattr(tool_info, 'handler'):
                tool_func = tool_info.handler
            elif hasattr(tool_info, '__call__'):
                # FunctionTool might be callable itself
                tool_func = tool_info
        elif hasattr(tool_info, 'handler'):
            tool_func = tool_info.handler
        elif callable(tool_info):
            tool_func = tool_info
        elif isinstance(tool_info, dict):
            tool_func = tool_info.get('function') or tool_info.get('handler')
        
        if not tool_func:
            # Last resort - check if tool_info itself is callable
            if hasattr(tool_info, '__call__'):
                tool_func = tool_info
            else:
                return self._error_response(-32603, f"Tool {tool_name} is not callable", request_id)
        
        try:
            # Execute tool
            import inspect
            
            # If it's a bound method or has special calling convention
            if hasattr(tool_func, '__self__') or (hasattr(tool_info, '__call__') and tool_func == tool_info):
                # Call directly without introspection
                try:
                    if asyncio.iscoroutinefunction(tool_func):
                        result = await tool_func(**tool_args)
                    else:
                        result = tool_func(**tool_args)
                except TypeError as e:
                    # Try without arguments if it fails
                    if not tool_args:
                        if asyncio.iscoroutinefunction(tool_func):
                            result = await tool_func()
                        else:
                            result = tool_func()
                    else:
                        raise e
            else:
                # Regular function - inspect signature
                if asyncio.iscoroutinefunction(tool_func):
                    result = await tool_func(**tool_args)
                else:
                    result = tool_func(**tool_args)
            
            # Format result according to MCP spec
            if isinstance(result, dict) and "content" in result:
                return {
                    "jsonrpc": "2.0",
                    "result": result,
                    "id": request_id
                }
            else:
                # Convert to MCP content format
                if isinstance(result, str):
                    content = [{"type": "text", "text": result}]
                elif isinstance(result, dict):
                    content = [{"type": "text", "text": json.dumps(result, indent=2)}]
                elif isinstance(result, list) and all(isinstance(item, dict) and "type" in item for item in result):
                    # Already formatted as content array
                    content = result
                else:
                    content = [{"type": "text", "text": str(result)}]
                
                return {
                    "jsonrpc": "2.0",
                    "result": {"content": content},
                    "id": request_id
                }
                
        except Exception as e:
            logger.error(f"Tool execution error for {tool_name}: {e}", exc_info=True)
            return self._error_response(-32603, f"Tool execution error: {str(e)}", request_id)
    
    async def _handle_prompt_get(self, params: Dict[str, Any], request_id: Any) -> Dict[str, Any]:
        """Handle prompts/get method"""
        prompt_name = params.get("name")
        prompt_args = params.get("arguments", {})
        
        prompt_registry = self._get_prompts()
        if prompt_name not in prompt_registry:
            return self._error_response(-32601, f"Unknown prompt: {prompt_name}", request_id)
        
        prompt_info = prompt_registry[prompt_name]
        
        try:
            if hasattr(prompt_info, 'handler'):
                result = await prompt_info.handler(**prompt_args)
            elif callable(prompt_info):
                result = await prompt_info(**prompt_args)
            else:
                result = {"messages": [{"role": "user", "content": str(prompt_info)}]}
            
            return {
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id
            }
        except Exception as e:
            return self._error_response(-32603, f"Prompt execution error: {str(e)}", request_id)
    
    def _extract_tool_definition(self, tool_name: str, tool_func: Any) -> Dict[str, Any]:
        """Extract tool definition from function"""
        import inspect
        
        description = inspect.getdoc(tool_func) or f"Tool {tool_name}"
        sig = inspect.signature(tool_func)
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name in ['self', '_user_context']:
                continue
                
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
            
            properties[param_name] = {
                "type": param_type,
                "description": f"Parameter {param_name}"
            }
            
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        return {
            "name": tool_name,
            "description": description.strip(),
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    
    def _error_response(self, code: int, message: str, request_id: Any) -> Dict[str, Any]:
        """Create JSON-RPC error response"""
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": code,
                "message": message
            },
            "id": request_id
        }

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Banking MCP Server...")
    
    try:
        # Initialize OAuth if enabled
        if ENABLE_OAUTH:
            logger.info("Initializing OAuth...")
            app_state["oauth_validator"] = SimpleOAuthValidator(KEYCLOAK_URL, KEYCLOAK_REALM)
            await app_state["oauth_validator"].fetch_jwks()
            logger.info("OAuth initialized successfully")
        
        # Initialize MCP server
        logger.info("Initializing MCP server...")
        app_state["mcp_server"] = BankingMCPServer(
            config_path=os.path.join("mcp_server", "config", "tools_config.yaml"),
            streamable_http_endpoint=MCP_ENDPOINT,
            sse_endpoint=os.getenv("MCP_SSE_ENDPOINT", "http://localhost:8084/mcp")
        )
        
        await app_state["mcp_server"].initialize()
        
        # Debug: Deep inspection of FastMCP structure
        logger.info("=== DEBUGGING FASTMCP STRUCTURE ===")
        mcp = app_state["mcp_server"].mcp
        logger.info(f"MCP type: {type(mcp)}")
        logger.info(f"MCP dir: {[attr for attr in dir(mcp) if not attr.startswith('__')]}")
        
        # Check all possible locations
        for attr in ['_registry', '_tools', 'tools', 'tool_registry', '_server', '_tool_manager', '_prompt_manager']:
            if hasattr(mcp, attr):
                value = getattr(mcp, attr)
                logger.info(f"Found {attr}: type={type(value)}")
                
                # Special handling for _tool_manager
                if attr == '_tool_manager':
                    if hasattr(value, '_tools'):
                        logger.info(f"  _tool_manager._tools: {type(value._tools)}")
                        if hasattr(value._tools, '__len__'):
                            logger.info(f"    Length: {len(value._tools)}")
                        if hasattr(value._tools, 'keys'):
                            logger.info(f"    Keys: {list(value._tools.keys())}")
                
                # Special handling for _prompt_manager
                elif attr == '_prompt_manager':
                    if hasattr(value, '_prompts'):
                        logger.info(f"  _prompt_manager._prompts: {type(value._prompts)}")
                        if hasattr(value._prompts, '__len__'):
                            logger.info(f"    Length: {len(value._prompts)}")
                        if hasattr(value._prompts, 'keys'):
                            logger.info(f"    Keys: {list(value._prompts.keys())}")
                
                # Generic handling
                else:
                    if hasattr(value, '__len__'):
                        try:
                            logger.info(f"  Length: {len(value)}")
                        except:
                            pass
                    if hasattr(value, 'keys'):
                        try:
                            keys = list(value.keys())
                            logger.info(f"  Keys ({len(keys)}): {keys[:5]}")  # First 5 keys
                        except:
                            pass
                    # Check nested attributes
                    for nested in ['tools', 'prompts', 'tool_manager', 'prompt_manager', '_tools', '_prompts']:
                        if hasattr(value, nested):
                            nested_value = getattr(value, nested)
                            logger.info(f"  {attr}.{nested} exists: {type(nested_value)}")
                            if hasattr(nested_value, 'keys'):
                                try:
                                    nested_keys = list(nested_value.keys())
                                    logger.info(f"    Keys: {nested_keys[:5]}")
                                except:
                                    pass
        
        logger.info("=== END DEBUG ===")
        
        # Initialize handler
        app_state["mcp_handler"] = MCPRequestHandler(app_state["mcp_server"])
        
        # Log summary
        tools = app_state["mcp_handler"]._get_tools()
        prompts = app_state["mcp_handler"]._get_prompts()
        logger.info(f"MCP server initialized with {len(tools)} tools and {len(prompts)} prompts")
        
        if tools:
            logger.info(f"Available tools: {list(tools.keys())}")
        if prompts:
            logger.info(f"Available prompts: {list(prompts.keys())}")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Banking MCP Server...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Banking MCP Server (OAuth-Enabled)",
    description="MCP server with optional OAuth 2.0 protection",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.get("/.well-known/oauth-protected-resource")
async def oauth_protected_resource():
    """OAuth 2.0 Protected Resource Metadata"""
    return {
        "resource": MCP_BASE_URL,
        "authorization_servers": [
            f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}"
        ],
        "scopes_supported": [
            "banking:read",
            "banking:write",
            "customer:profile"
        ],
        "bearer_methods_supported": ["header"],
        "resource_documentation": f"{MCP_BASE_URL}/docs"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    health_info = {
        "status": "healthy",
        "service": "banking-mcp",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "oauth_enabled": ENABLE_OAUTH
    }
    
    if app_state["mcp_handler"]:
        tools = app_state["mcp_handler"]._get_tools()
        prompts = app_state["mcp_handler"]._get_prompts()
        health_info["mcp"] = {
            "tools_count": len(tools),
            "prompts_count": len(prompts),
            "initialized": True
        }
    
    return health_info

@app.post("/mcp")
@app.post("/mcp/{path:path}")
async def handle_mcp(request: Request, path: Optional[str] = None):
    """Handle MCP requests with optional OAuth validation"""
    
    user_info = None
    
    # OAuth validation if enabled
    if ENABLE_OAUTH and app_state["oauth_validator"]:
        auth_header = request.headers.get("Authorization", "")
        
        if not auth_header.startswith("Bearer "):
            return Response(
                status_code=401,
                headers={
                    "WWW-Authenticate": f'Bearer realm="{KEYCLOAK_REALM}"'
                },
                content=json.dumps({
                    "error": "unauthorized",
                    "error_description": "Bearer token required"
                }),
                media_type="application/json"
            )
        
        token = auth_header[7:]
        user_info = await app_state["oauth_validator"].validate_token(token)
        
        if not user_info:
            return Response(
                status_code=401,
                headers={
                    "WWW-Authenticate": f'Bearer realm="{KEYCLOAK_REALM}", error="invalid_token"'
                },
                content=json.dumps({
                    "error": "invalid_token",
                    "error_description": "Token validation failed"
                }),
                media_type="application/json"
            )
        
        # Check required scopes
        if "banking:read" not in user_info.get("scopes", []):
            return Response(
                status_code=403,
                headers={
                    "WWW-Authenticate": f'Bearer realm="{KEYCLOAK_REALM}", error="insufficient_scope", scope="banking:read"'
                },
                content=json.dumps({
                    "error": "insufficient_scope",
                    "error_description": "Missing required scope: banking:read"
                }),
                media_type="application/json"
            )
    
    # Process request
    try:
        body = await request.body()
        json_body = json.loads(body) if body else {}
        
        if app_state["mcp_handler"]:
            response = await app_state["mcp_handler"].handle_request(json_body, user_info)
            return JSONResponse(content=response)
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": "Service not initialized"
                    },
                    "id": json_body.get("id")
                }
            )
            
    except json.JSONDecodeError as e:
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {str(e)}"
                },
                "id": None
            }
        )
    except Exception as e:
        logger.error(f"Request error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                },
                "id": None
            }
        )

@app.get("/")
async def root():
    """Service information"""
    info = {
        "service": "Banking MCP Server",
        "version": "2.0.0",
        "endpoints": {
            "mcp": "/mcp",
            "health": "/health",
            "oauth_metadata": "/.well-known/oauth-protected-resource"
        },
        "oauth_enabled": ENABLE_OAUTH,
        "mcp_spec": "2024-11-05"
    }
    
    if ENABLE_OAUTH:
        info["oauth"] = {
            "authorization": f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/auth",
            "token": f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token",
            "jwks": f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/certs"
        }
    
    return info

if __name__ == "__main__":
    print(f"🚀 Banking MCP Server (OAuth-Compliant)")
    print(f"=" * 50)
    print(f"📍 Server URL: {MCP_BASE_URL}")
    print(f"🔐 OAuth: {'ENABLED' if ENABLE_OAUTH else 'DISABLED'}")
    
    if ENABLE_OAUTH:
        print(f"🔑 Auth Server: {KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}")
        print(f"📋 OAuth Metadata: {MCP_BASE_URL}/.well-known/oauth-protected-resource")
    
    print(f"\n📡 Endpoints:")
    print(f"   - MCP: {MCP_BASE_URL}/mcp")
    print(f"   - Health: {MCP_BASE_URL}/health")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=MCP_PORT, log_level="info")