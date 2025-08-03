"""
MCP Client Wrapper with proper OAuth injection for StreamableHTTPTransport
"""
import asyncio
import os
import logging
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid
import httpx

import jsonschema
from mcp_client.client import StandardMCPClient
from mcp_client.transport.streamable_http_transport import StreamableHTTPTransport
from mcp_client.protocol.messages import MCPRequest, MCPResponse

# Import the global registry
from src.ii_agent.tools.banking_tool_registry import mcp_tool_registry, ToolMetadata

# Try to import OAuth utilities
try:
    from src.ii_agent.utils.oauth_utils import OAuthTokenManager
    OAUTH_AVAILABLE = True
except ImportError:
    OAUTH_AVAILABLE = False


class OAuthStreamableHTTPTransport(StreamableHTTPTransport):
    """Extended StreamableHTTPTransport that supports OAuth headers"""
    
    def __init__(self, endpoint: str, api_key: str, sse_endpoint: Optional[str] = None, 
                 oauth_headers: Optional[Dict[str, str]] = None):
        super().__init__(endpoint, api_key, sse_endpoint)
        self.oauth_headers = oauth_headers or {}
        
        # Debug logging
        self.logger.info(f"OAuthStreamableHTTPTransport initialized")
        self.logger.info(f"OAuth headers keys: {list(self.oauth_headers.keys())}")
        if "Authorization" in self.oauth_headers:
            self.logger.info(f"Authorization header present: {self.oauth_headers['Authorization'][:30]}...")
    
    async def connect(self):
        """Complete override of connect to include OAuth headers"""
        # Create client with default headers
        self.client = httpx.AsyncClient(follow_redirects=True, timeout=20.0)
        
        # Build headers including OAuth
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        # Add OAuth headers
        headers.update(self.oauth_headers)
        
        # Log headers being sent
        self.logger.info(f"Sending connect request with headers: {list(headers.keys())}")
        
        # Determine endpoint URL
        url = f"{self.endpoint}/mcp" if not self.endpoint.endswith('/mcp') else self.endpoint
        
        # Define payload variations
        payloads = [
            {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {},
                "id": "init-1"
            },
            {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "capabilities": {
                        "jsonrpc": "2.0",
                        "streaming": True,
                        "methods": ["initialize", "tools/list", "query"]
                    }
                },
                "id": "init-2"
            }
        ]

        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            payload = payloads[attempt % len(payloads)]
            try:
                self.logger.debug(f"Attempt {attempt}/{max_attempts}: Connecting to: {url}")
                
                # Send request with OAuth headers
                response = await self.client.post(
                    url,
                    json=payload,
                    headers=headers
                )
                
                self.logger.debug(f"Response received: status={response.status_code}")
                response.raise_for_status()
                
                # Handle response
                content_type = response.headers.get("content-type", "").lower()
                if "application/json" in content_type:
                    from mcp_client.protocol.messages import MCPResponse
                    init_response = MCPResponse.from_dict(response.json())
                else:
                    init_response = None
                
                # Set session ID
                self.session_id = response.headers.get("Mcp-Session-Id")
                self.logger.info(f"Initialized session: {self.session_id}")
                
                # Start SSE if available
                if self.sse_endpoint:
                    import asyncio
                    self.sse_task = asyncio.create_task(self._receive_sse())
                
                return
                
            except httpx.HTTPStatusError as e:
                self.logger.error(f"Attempt {attempt}/{max_attempts} failed: {str(e)}")
                if attempt == max_attempts:
                    await self.disconnect()
                    from mcp_client.transport.streamable_http_transport import StreamableHTTPError
                    raise StreamableHTTPError(f"HTTP {e.response.status_code}: {e.response.text}")
                import asyncio
                await asyncio.sleep(4)
            except Exception as e:
                self.logger.error(f"Attempt {attempt}/{max_attempts} failed: {str(e)}")
                if attempt == max_attempts:
                    await self.disconnect()
                    from mcp_client.transport.streamable_http_transport import StreamableHTTPError
                    raise StreamableHTTPError(f"Transport failed: {str(e)}")
                import asyncio
                await asyncio.sleep(4)
    
    async def send_request(self, request):
        """Override to include OAuth headers in all requests"""
        if not self.client:
            raise RuntimeError("Transport not connected")
            
        # Build headers for the request
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        # Add OAuth headers
        headers.update(self.oauth_headers)
        
        # Add session ID if available
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
            
        url = f"{self.endpoint}/mcp" if not self.endpoint.endswith('/mcp') else self.endpoint
        
        # Handle batch vs single request
        is_batch = isinstance(request, list)
        payload = [r.to_dict() for r in request] if is_batch else request.to_dict()
        
        try:
            self.logger.debug(f"Sending request to: {url} with OAuth headers")
            response = await self.client.post(
                url,
                json=payload,
                headers=headers
            )
            
            response.raise_for_status()
            if hasattr(request, 'method') and request.method == 'tools/list':
                self.logger.info("=== TOOLS/LIST RESPONSE DEBUG ===")
                self.logger.info(f"Response headers: {dict(response.headers)}")
                self.logger.info(f"Response status: {response.status_code}")
                self.logger.info(f"Response body preview: {response.text[:500]}")
            
            # Handle response
            content_type = response.headers.get("content-type", "").lower()
            body = response.text or "<empty>"
            
            if not response.text and "text/event-stream" not in content_type:
                self.logger.error("Empty response body received")
                raise ValueError("MCP server returned empty response")
                
            if "application/json" in content_type:
                from mcp_client.protocol.messages import MCPResponse
                response_data = response.json()
                if is_batch:
                    return [MCPResponse.from_dict(item) for item in response_data]
                return MCPResponse.from_dict(response_data)
            elif "text/event-stream" in content_type:
                self.logger.debug("SSE stream response received")
                return None
            else:
                self.logger.warning(f"Non-JSON/SSE response received")
                return None
                
        except httpx.HTTPStatusError as e:
            self.logger.error(f"Request failed: HTTP {e.response.status_code}: {e.response.text}")
            from mcp_client.transport.streamable_http_transport import StreamableHTTPError
            raise StreamableHTTPError(f"HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            self.logger.error(f"Request failed: {str(e)}")
            raise
    
    async def send_progress(self, session_id: str, progress: Dict[str, Any]) -> None:
        """Send progress update with OAuth headers"""
        if not self.client:
            raise RuntimeError("Transport not connected")
            
        from mcp_client.protocol.messages import ProgressMessage
        progress_message = ProgressMessage(session_id=session_id, content=progress)
        
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        # Add OAuth headers
        headers.update(self.oauth_headers)
        
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
            
        url = f"{self.endpoint}/mcp" if not self.endpoint.endswith('/mcp') else self.endpoint
        
        try:
            response = await self.client.post(
                url,
                json=progress_message.to_dict(),
                headers=headers
            )
            response.raise_for_status()
            self.logger.debug(f"Sent progress for session {session_id}")
        except Exception as e:
            self.logger.error(f"Error sending progress: {str(e)}")
            raise
    
    async def _receive_sse(self):
        """Receive SSE stream with OAuth headers"""
        headers = {
            "X-API-Key": self.api_key,
            "Accept": "text/event-stream"
        }
        
        # Add OAuth headers
        headers.update(self.oauth_headers)
        
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
            
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                self.logger.debug(f"Starting SSE stream at: {self.sse_endpoint}")
                async with client.stream(
                    "GET",
                    self.sse_endpoint,
                    headers=headers,
                    timeout=None
                ) as response:
                    if response.status_code != 200:
                        self.logger.error(f"Failed to start SSE stream: {response.status_code}")
                        raise ValueError(f"SSE stream failed with status {response.status_code}")
                        
                    self.logger.debug("SSE stream established")
                    
                    # Process SSE events (simplified)
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            data = line[5:].strip()
                            if data:
                                try:
                                    import json
                                    event_data = json.loads(data)
                                    if event_data.get("type") == "ProgressMessage":
                                        await self.progress_queue.put(event_data.get("content", {}))
                                except json.JSONDecodeError:
                                    self.logger.warning(f"Failed to parse SSE data: {data}")
                                    
        except Exception as e:
            self.logger.error(f"SSE stream error: {str(e)}")
    
    async def receive_progress(self) -> Optional[Dict[str, Any]]:
        """Receive progress update from SSE stream"""
        try:
            import asyncio
            return await asyncio.wait_for(self.progress_queue.get(), timeout=5.0)
        except asyncio.TimeoutError:
            return None
    
    async def disconnect(self):
        """Disconnect and cleanup"""
        if hasattr(self, 'sse_task') and self.sse_task:
            self.sse_task.cancel()
            try:
                import asyncio
                await self.sse_task
            except asyncio.CancelledError:
                pass
        if self.client:
            await self.client.aclose()
            self.client = None
        self.logger.info("Streamable HTTP transport disconnected")

# class OAuthStreamableHTTPTransport(StreamableHTTPTransport):
#     """Extended StreamableHTTPTransport that supports OAuth headers"""
    
#     def __init__(self, endpoint: str, api_key: str, sse_endpoint: Optional[str] = None, 
#                  oauth_headers: Optional[Dict[str, str]] = None):
#         super().__init__(endpoint, api_key, sse_endpoint)
#         self.oauth_headers = oauth_headers or {}
    
#     # async def send_request(self, request):
#     #     """Override to ensure OAuth headers are included"""
#     #     if not self.client:
#     #         raise RuntimeError("Transport not connected")
            
#     #     # Temporarily add OAuth headers
#     #     original_headers = dict(self.client.headers)
#     #     try:
#     #         self.client.headers.update(self.oauth_headers)
#     #         return await super().send_request(request)
#     #     finally:
#     #         # Restore original headers
#     #         self.client.headers.clear()
#     #         self.client.headers.update(original_headers)
        
#     async def connect(self):
#         """Override connect to inject OAuth headers"""
#         # Create client with OAuth headers
#         self.client = httpx.AsyncClient(
#             follow_redirects=True, 
#             timeout=20.0,
#             headers=self.oauth_headers  # Add OAuth headers here
#         )
        
#         # Enhanced headers for better compatibility
#         headers = {
#             "X-API-Key": self.api_key,
#             "Content-Type": "application/json",
#             "Accept": "application/json, text/event-stream",
#             "User-Agent": "BankingMCPClient/2.0.0"
#         }
        
#         # Add OAuth headers
#         headers.update(self.oauth_headers)
        
#         # Determine endpoint URL
#         url = f"{self.endpoint}/mcp" if not self.endpoint.endswith('/mcp') else self.endpoint

#         # Updated protocol version to latest
#         payloads = [
#             {
#                 "jsonrpc": "2.0",
#                 "method": "initialize",
#                 "params": {
#                     "protocolVersion": "2024-11-05",  # Latest version
#                     "capabilities": {
#                         "experimental": {},
#                         "sampling": {},
#                         "tools": {},
#                         "prompts": {},
#                         "resources": {}
#                     },
#                     "clientInfo": {
#                         "name": "BankingMCPClient",
#                         "version": "2.0.0"
#                     }
#                 },
#                 "id": "init-1"
#             },
#             # Fallback with older version if needed
#             {
#                 "jsonrpc": "2.0", 
#                 "method": "initialize",
#                 "params": {
#                     "protocolVersion": "2024-10-07",  # Fallback version
#                     "capabilities": {
#                         "jsonrpc": "2.0",
#                         "streaming": True,
#                         "methods": ["initialize", "tools/list", "tools/call"]
#                     },
#                     "clientInfo": {
#                         "name": "BankingMCPClient",
#                         "version": "2.0.0"
#                     }
#                 },
#                 "id": "init-2"
#             }
#         ]

#         max_attempts = 4
#         for attempt in range(1, max_attempts + 1):
#             payload = payloads[(attempt - 1) % len(payloads)]
            
#             try:
#                 self.logger.debug(f"Attempt {attempt}/{max_attempts}: Connecting to: {url}")
#                 self.logger.debug(f"Using protocol version: {payload['params']['protocolVersion']}")
#                 self.logger.debug(f"OAuth enabled: {bool(self.oauth_headers.get('Authorization'))}")
                
#                 response = await self.client.post(url, json=payload, headers=headers)
                
#                 self.logger.debug(f"Response status: {response.status_code}")
#                 self.logger.debug(f"Response headers: {dict(response.headers)}")
                
#                 response.raise_for_status()
                
#                 # Handle different response formats
#                 content_type = response.headers.get("content-type", "").lower()
#                 response_text = response.text
                
#                 self.logger.debug(f"Content-Type: {content_type}")
#                 self.logger.debug(f"Response text preview: {response_text[:200]}")
                
#                 if not response_text:
#                     self.logger.error("Empty response body")
#                     raise ValueError("Empty response from MCP server")
                
#                 # Parse response based on format
#                 init_response = None
                
#                 if "text/event-stream" in content_type or "event:" in response_text:
#                     self.logger.debug("Processing SSE response format")
#                     init_response = self._parse_sse_response(response_text)
#                 elif "application/json" in content_type:
#                     self.logger.debug("Processing JSON response format")
#                     try:
#                         init_response = response.json()
#                         from mcp_client.protocol.messages import MCPResponse
#                         init_response = MCPResponse.from_dict(init_response)
#                     except json.JSONDecodeError as e:
#                         self.logger.error(f"JSON decode error: {e}")
#                         raise ValueError(f"Invalid JSON response: {e}")
#                 else:
#                     # Try SSE parsing first, then JSON
#                     if "event:" in response_text or "data:" in response_text:
#                         self.logger.debug("Detected SSE format in non-SSE content-type")
#                         init_response = self._parse_sse_response(response_text)
#                     else:
#                         # Last resort: try JSON parsing
#                         try:
#                             json_data = response.json()
#                             from mcp_client.protocol.messages import MCPResponse
#                             init_response = MCPResponse.from_dict(json_data)
#                         except json.JSONDecodeError as e:
#                             self.logger.error(f"Failed to parse response as JSON: {e}")
#                             self.logger.error(f"Response text: {response_text}")
#                             raise ValueError(f"Invalid JSON response: {e}")
                
#                 if init_response and init_response.result:
#                     # Extract session ID if provided
#                     self.session_id = response.headers.get("Mcp-Session-Id")
#                     if not self.session_id:
#                         # Try to get from response
#                         self.session_id = init_response.result.get("sessionId")
                    
#                     self.logger.info(f"Initialized session: {self.session_id}")
                    
#                     # Check protocol version compatibility
#                     server_version = init_response.result.get("protocolVersion")
#                     if server_version:
#                         self.logger.info(f"Server protocol version: {server_version}")
                    
#                     # Start SSE if available
#                     if self.sse_endpoint:
#                         import asyncio
#                         self.sse_task = asyncio.create_task(self._receive_sse())
                    
#                     return
#                 else:
#                     raise ValueError("Invalid initialization response")
                    
#             except httpx.HTTPStatusError as e:
#                 self.logger.error(f"Attempt {attempt} failed: HTTP {e.response.status_code}")
#                 error_text = e.response.text if hasattr(e.response, 'text') else str(e)
#                 self.logger.error(f"Error details: {error_text}")
                
#                 if e.response.status_code == 401:
#                     self.logger.error("Authentication failed - check OAuth token")
#                 elif e.response.status_code == 404:
#                     self.logger.error("MCP endpoint not found - check server URL")
#                 elif e.response.status_code == 400:
#                     self.logger.error("Bad request - check protocol version or request format")
                
#                 if attempt == max_attempts:
#                     await self.disconnect()
#                     from mcp_client.transport.streamable_http_transport import StreamableHTTPError
#                     raise StreamableHTTPError(f"HTTP {e.response.status_code}: {error_text}")
                    
#                 import asyncio
#                 await asyncio.sleep(2)
                
#             except Exception as e:
#                 self.logger.error(f"Attempt {attempt} failed: {e}")
#                 if attempt == max_attempts:
#                     await self.disconnect()
#                     from mcp_client.transport.streamable_http_transport import StreamableHTTPError
#                     raise StreamableHTTPError(f"Transport failed: {str(e)}")
                    
#                 import asyncio
#                 await asyncio.sleep(2)


class MCPClientWrapper:
    """Enhanced MCP Client Wrapper with OAuth support and improved registry"""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8082",
        sse_endpoint: str = "http://localhost:8084/mcp", 
        api_key: str = "test-api-key-123",
        registry: Optional[Any] = None,  # Can be MCPToolRegistry or EnhancedMCPToolRegistry
        headers: Optional[Dict[str, str]] = None,  # OAuth headers support
        oauth_token: Optional[str] = None  # Direct OAuth token support
    ):
        # Initialize basic attributes first
        self.base_url = base_url
        self.sse_endpoint = sse_endpoint
        self.api_key = api_key
        self.client: Optional[StandardMCPClient] = None
        
        # Initialize OAuth headers
        self.headers = headers or {}
        
        # If OAuth token provided directly, add to headers
        if oauth_token and "Authorization" not in self.headers:
            self.headers["Authorization"] = f"Bearer {oauth_token}"
        
        # Initialize logger BEFORE any logging calls
        self.logger = logging.getLogger(__name__)
        
        # Setup interaction logger
        self.interaction_logger = self._setup_interaction_logger()
        
        # Use provided registry or default to global singleton
        if registry is None:
            # Use the global singleton registry
            self.tool_registry = mcp_tool_registry
            self.logger.info("Using global MCP tool registry")
        else:
            # Use the provided registry
            self.tool_registry = registry
            self.logger.info(f"Using provided registry: {type(registry).__name__}")
        
        # Track initialization state
        self._initialized = False
        
        # OAuth token manager (optional)
        self._oauth_token_manager = None
        
        # Store transport for reuse
        self._transport = None
        
        self.logger.info(
            f"MCPClientWrapper initialized: base_url={base_url}, "
            f"oauth_enabled={bool(self.headers.get('Authorization'))}"
        )

    def _setup_interaction_logger(self):
        """Setup detailed interaction logging"""
        logger = logging.getLogger("MCPToolInteractions")
        
        # Only add handler if it doesn't exist
        if not logger.handlers:
            file_handler = logging.FileHandler('mcp_tool_interactions.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.setLevel(logging.INFO)
            
        return logger

    def set_oauth_token_manager(self, token_manager):
        """Set an OAuth token manager for automatic token refresh"""
        self._oauth_token_manager = token_manager
        self.logger.info("OAuth token manager configured")

    async def _get_headers(self) -> Dict[str, str]:
        """Get headers with updated OAuth token if token manager is available"""
        headers = self.headers.copy()
        
        if self._oauth_token_manager:
            try:
                fresh_token = await self._oauth_token_manager.get_valid_token()
                if fresh_token:
                    headers["Authorization"] = f"Bearer {fresh_token}"
                    self.logger.debug("Using fresh OAuth token from token manager")
                else:
                    self.logger.warning("Token manager returned no token, using existing headers")
            except Exception as e:
                self.logger.warning(f"Failed to refresh OAuth token: {e}")
                # Continue with existing token if refresh fails
        
        return headers

    async def initialize(self):
        """Initialize the MCP client and discover tools"""
        if self._initialized:
            self.logger.warning("MCP client already initialized")
            return
            
        try:
            # Get current headers (with fresh OAuth token if available)
  
            oauth_headers = await self._get_headers()
            self.logger.info(f"OAuth headers to transport: {list(oauth_headers.keys())}")
            if "Authorization" in oauth_headers:
                self.logger.info(f"Auth value: {oauth_headers['Authorization'][:40]}...")            
            
            # Use our custom transport that supports OAuth
            self._transport = OAuthStreamableHTTPTransport(
                endpoint=self.base_url,
                api_key=self.api_key,
                sse_endpoint=self.sse_endpoint,
                oauth_headers=oauth_headers
            )
            
            # Create MCP client
            self.client = StandardMCPClient(self._transport)
            await self.client.initialize()
            
            # Discover and register tools
            await self.discover_tools()
            # Also discover prompts
            await self.discover_prompts()
            
            self._initialized = True
            self.logger.info("MCP Client Wrapper initialized successfully")
            
            # Log OAuth status
            if oauth_headers.get("Authorization"):
                self.logger.info(f"OAuth authentication is enabled with token: {oauth_headers['Authorization'][:20]}...")
            else:
                self.logger.info("OAuth authentication is not configured")
            
        except Exception as e:
            # Enhanced error logging for OAuth issues
            if "401" in str(e) or "unauthorized" in str(e).lower():
                self.logger.error(
                    f"Authentication failed - check OAuth token: {e}. "
                    "Ensure OAUTH_TOKEN is valid or OAuth is disabled on server."
                )
            else:
                self.logger.error(f"Failed to initialize MCP client: {e}")
            
            # Cleanup on failure
            await self.close()
            raise
    
    async def _update_transport_headers(self):
        """Update transport headers with fresh token"""
        if self._transport and hasattr(self._transport, 'oauth_headers'):
            headers = await self._get_headers()
            self._transport.oauth_headers = headers
            # Also update the client if it exists
            if hasattr(self._transport, 'client') and self._transport.client:
                self._transport.client.headers.update(headers)
            self.logger.debug("Updated transport headers")
    
    async def list_available_tools(self) -> List[Dict[str, Any]]:
        """List all available tools from MCP server."""
        if not self.client:
            raise RuntimeError("MCP client not initialized")
            
        # Ensure we have fresh headers for the request
        await self._update_transport_headers()
            
        return await self.client.list_tools()
    
    async def discover_tools(self):
        """Dynamically discover and register tools from MCP server"""
        if not self.client:
            raise RuntimeError("MCP client not initialized")
            
        try:
            mcp_tools = await self.client.list_tools()
            self.logger.info(f"MCP server returned {len(mcp_tools)} tools")
            
            registered_count = 0
            
            for tool_config in mcp_tools:
                try:
                    # Extract all available metadata
                    name = tool_config.get('name')
                    if not name:
                        self.logger.warning("Tool without name found, skipping")
                        continue
                    
                    # Handle both input_schema and inputSchema
                    schema = tool_config.get('input_schema') or tool_config.get('inputSchema', {})
                    
                    # Ensure schema has basic structure
                    if not isinstance(schema, dict):
                        schema = {"type": "object", "properties": {}, "required": []}
                    elif 'type' not in schema:
                        schema['type'] = 'object'
                    if 'properties' not in schema:
                        schema['properties'] = {}
                    # ============ ADD THIS FIX HERE ============
                    # Fix the schema for invoke_api_endpoint
                    if name == 'invoke_api_endpoint':
                        self.logger.warning(f"Fixing schema for {name}: params should be object, not string")
                        if 'properties' in schema and 'params' in schema['properties']:
                            # Change params from string to object type
                            schema['properties']['params'] = {
                                "type": "object",
                                "description": "Parameters to pass to the API endpoint",
                                "additionalProperties": True
                            }
                            # Also make it optional if it's marked as required
                            if 'required' in schema and 'params' in schema['required']:
                                schema['required'].remove('params')
                    # ============ END OF FIX ============

                    # Extract enhanced metadata
                    tool_metadata = ToolMetadata(
                        name=name,
                        description=tool_config.get('description', f'MCP tool: {name}'),
                        input_schema=schema,
                        category=tool_config.get('category', 'banking'),
                        permissions=tool_config.get('permissions', []),
                        method=tool_config.get('method', 'POST'),
                        path=tool_config.get('path', f'/{name}'),
                        tags=tool_config.get('tags', []),
                        deprecated=tool_config.get('deprecated', False),
                        auth_required=tool_config.get('auth_required', True),
                        operation_id=tool_config.get('operation_id'),
                        parameters=tool_config.get('parameters', []),
                        examples=tool_config.get('examples', {})
                    )
                    
                    self.tool_registry.register_tool(tool_metadata)
                    registered_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to register tool {tool_config.get('name', 'unknown')}: {e}")
                    
            self.logger.info(f"Tool discovery completed: {registered_count}/{len(mcp_tools)} tools registered")
            
            # Log statistics
            stats = self.tool_registry.get_statistics()
            self.logger.info(f"Registry statistics: {stats}")
            
        except Exception as e:
            # Enhanced error handling for OAuth issues
            if "401" in str(e) or "unauthorized" in str(e).lower():
                self.logger.error(
                    "Tool discovery failed due to authentication error. "
                    "Check OAuth token validity."
                )
            self.logger.error(f"Tool discovery failed: {e}")
            raise

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with comprehensive validation and logging"""
        if not self._initialized:
            raise RuntimeError("MCP client not initialized")
            
        start_time = datetime.now()
        
        # Retrieve tool metadata
        tool_metadata = self.tool_registry.get_tool(tool_name)
        if not tool_metadata:
            # Try to find by operation ID
            tool_metadata = self.tool_registry.get_tool_by_operation_id(tool_name)
            if not tool_metadata:
                raise ValueError(f"Tool {tool_name} not found in registry")
            tool_name = tool_metadata.name

        try:
            # Validate input parameters
            jsonschema.validate(instance=parameters, schema=tool_metadata.input_schema)
            
            # Check deprecation
            if tool_metadata.deprecated:
                self.logger.warning(f"Executing deprecated tool: {tool_name}")
            
            # Log tool call
            self.interaction_logger.info(
                f"Executing Tool: {tool_name} [{tool_metadata.method} {tool_metadata.path}], "
                f"Category: {tool_metadata.category}, "
                f"Tags: {tool_metadata.tags}, "
                f"Parameters: {self._mask_sensitive_data(parameters)}, "
                f"OAuth: {bool(self.headers.get('Authorization'))}"
            )
            
            # Update headers before tool execution
            await self._update_transport_headers()
            
            # Execute tool
            result = await self.client.call_tool(tool_name, parameters)
            
            # Update tool metadata
            tool_metadata.last_used = datetime.now()
            tool_metadata.usage_count += 1
            
            # Log successful execution
            execution_time = (datetime.now() - start_time).total_seconds()
            self.interaction_logger.info(
                f"Tool Execution Complete: {tool_name}, "
                f"Execution Time: {execution_time:.2f}s, "
                f"Total Usage: {tool_metadata.usage_count}"
            )
            
            return result
        
        except jsonschema.ValidationError as ve:
            # Log validation error with details
            self.interaction_logger.error(
                f"Tool Validation Error: {tool_name}, "
                f"Error: {str(ve)}, "
                f"Failed at: {ve.json_path}"
            )
            raise ValueError(f"Invalid parameters for tool {tool_name}: {ve.message}")
        
        except Exception as e:
            # Enhanced error logging for OAuth issues
            error_msg = str(e).lower()
            if "401" in error_msg or "unauthorized" in error_msg:
                self.interaction_logger.error(
                    f"Tool Execution Authentication Error: {tool_name}, "
                    f"OAuth token may be expired or invalid"
                )
                raise RuntimeError(
                    f"Authentication failed for tool {tool_name}. "
                    "Please check OAuth token."
                )
            else:
                # Log execution error
                self.interaction_logger.error(
                    f"Tool Execution Error: {tool_name}, "
                    f"Error Type: {type(e).__name__}, "
                    f"Error: {str(e)}"
                )
            raise
    
    def _is_response_too_large(self, result):
        """Check if response might overwhelm context"""
        result_str = json.dumps(result) if isinstance(result, dict) else str(result)
        return len(result_str) > 5000  # 5KB threshold
        
    def _create_summary_response(self, result):
        """Create a summary for large responses"""
        return {
            "summary": "Response too large - showing summary",
            "total_results": result.get('total_found', 'many'),
            "message": "Please use more specific filters (tag + search_query)",
            "available_tags": result.get('available_tags', []),
            "sample_apis": self._extract_sample_apis(result, limit=10)
        }
        
    def _extract_sample_apis(self, result, limit=10):
        """Extract sample APIs from result"""
        # Implementation depends on your result structure
        if isinstance(result, dict) and 'apis' in result:
            return result['apis'][:limit]
        return []
        
    def _mask_sensitive_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive information in tool parameters"""
        if not params:
            return {}
            
        masked_params = params.copy()
        sensitive_keys = [
            'password', 'token', 'secret', 'api_key', 'apikey',
            'auth', 'authorization', 'credential', 'private_key',
            'access_token', 'refresh_token', 'session_id'
        ]
        
        def mask_dict(d: Dict[str, Any]):
            for key, value in d.items():
                key_lower = key.lower()
                if any(sensitive in key_lower for sensitive in sensitive_keys):
                    d[key] = '****'
                elif isinstance(value, dict):
                    mask_dict(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            mask_dict(item)
        
        mask_dict(masked_params)
        return masked_params
    
    async def list_prompts(self) -> List[Dict[str, Any]]:
        """List all available prompts from the MCP server."""
        if not self.client:
            raise RuntimeError("MCP client not initialized")
        
        try:
            # Update headers before request
            await self._update_transport_headers()
                
            # Check if client has list_prompts method
            if hasattr(self.client, 'list_prompts'):
                prompts = await self.client.list_prompts()
                return prompts
            else:
                # Client doesn't support prompts
                self.logger.warning("MCP client doesn't have list_prompts method")
                return []
        except Exception as e:
            if "401" in str(e) or "unauthorized" in str(e).lower():
                self.logger.error("Failed to list prompts due to authentication error")
            else:
                self.logger.error(f"Failed to list prompts: {e}")
            return []

    async def get_prompt(self, name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get a specific prompt with arguments filled in."""
        if not self.client:
            raise RuntimeError("MCP client not initialized")
        
        try:
            # Update headers before request
            await self._update_transport_headers()
                
            # Check if client has get_prompt method
            if hasattr(self.client, 'get_prompt'):
                result = await self.client.get_prompt(name, arguments)
                return result
            else:
                raise RuntimeError("MCP client doesn't have get_prompt method")
        except Exception as e:
            if "401" in str(e) or "unauthorized" in str(e).lower():
                self.logger.error(f"Authentication failed when getting prompt '{name}'")
                raise RuntimeError(f"Authentication failed for prompt '{name}'")
            else:
                self.logger.error(f"Failed to get prompt '{name}': {e}")
            raise

    async def discover_prompts(self):
        """Discover and cache available prompts"""
        try:
            self.available_prompts = await self.list_prompts()
            if self.available_prompts:
                self.logger.info(f"Discovered {len(self.available_prompts)} MCP prompts")
                for prompt in self.available_prompts:
                    self.logger.info(f"  - {prompt['name']}: {prompt['description']}")
            else:
                self.logger.info("No MCP prompts discovered")
        except Exception as e:
            self.logger.error(f"Failed to discover prompts: {e}")
            self.available_prompts = []
            
    async def get_contextual_prompt(self, context: str):
        """Get appropriate prompt based on context"""
        if "error" in context:
            return await self.get_prompt("banking_error_handling")
        elif "discover" in context:
            return await self.get_prompt("discover_banking_apis_by_domain")
        else:
            return await self.get_prompt("banking_api_quick_start")

    async def close(self):
        """Clean up resources"""
        try:
            if self._transport and hasattr(self._transport, 'disconnect'):
                await self._transport.disconnect()
            
            if self.client:
                # Add any cleanup if needed
                self._initialized = False
                
            self.logger.info("MCP client closed")
        except Exception as e:
            self.logger.error(f"Error during close: {e}")
    
    def update_oauth_token(self, token: str):
        """Update OAuth token after initialization"""
        self.headers["Authorization"] = f"Bearer {token}"
        
        # Update transport headers if it exists
        if self._transport and hasattr(self._transport, 'oauth_headers'):
            self._transport.oauth_headers["Authorization"] = f"Bearer {token}"
            # Also update the client if it exists
            if hasattr(self._transport, 'client') and self._transport.client:
                self._transport.client.headers.update({"Authorization": f"Bearer {token}"})
            
        self.logger.info("OAuth token updated")


########################
# import json
# import logging
# from typing import Dict, Any, List, Optional, Set, Union
# from collections import defaultdict
# from dataclasses import dataclass, field
# from datetime import datetime

# import jsonschema
# from mcp_client.client import StandardMCPClient
# from mcp_client.transport.streamable_http_transport import StreamableHTTPTransport

# # Import the global registry
# from src.ii_agent.tools.banking_tool_registry import mcp_tool_registry, ToolMetadata


# class MCPClientWrapper:
#     """Enhanced MCP Client Wrapper with improved registry and error handling"""
    
#     def __init__(
#         self, 
#         base_url: str = "http://localhost:8082",
#         sse_endpoint: str = "http://localhost:8084/mcp", 
#         api_key: str = "test-api-key-123",
#         registry: Optional[Any] = None  # Can be MCPToolRegistry or EnhancedMCPToolRegistry
#     ):
#         # Initialize basic attributes first
#         self.base_url = base_url
#         self.sse_endpoint = sse_endpoint
#         self.api_key = api_key
#         self.client: Optional[StandardMCPClient] = None
        
#         # Initialize logger BEFORE any logging calls
#         self.logger = logging.getLogger(__name__)
        
#         # Setup interaction logger
#         self.interaction_logger = self._setup_interaction_logger()
        
#         # Use provided registry or default to global singleton
#         if registry is None:
#             # Use the global singleton registry
#             self.tool_registry = mcp_tool_registry
#             self.logger.info("Using global MCP tool registry")
#         else:
#             # Use the provided registry
#             self.tool_registry = registry
#             self.logger.info(f"Using provided registry: {type(registry).__name__}")
        
#         # Track initialization state
#         self._initialized = False
        
#         self.logger.info(f"MCPClientWrapper initialized: base_url={base_url}")

#     def _setup_interaction_logger(self):
#         """Setup detailed interaction logging"""
#         logger = logging.getLogger("MCPToolInteractions")
        
#         # Only add handler if it doesn't exist
#         if not logger.handlers:
#             file_handler = logging.FileHandler('mcp_tool_interactions.log')
#             formatter = logging.Formatter(
#                 '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#             )
#             file_handler.setFormatter(formatter)
#             logger.addHandler(file_handler)
#             logger.setLevel(logging.INFO)
            
#         return logger

#     async def initialize(self):
#         """Initialize the MCP client and discover tools"""
#         if self._initialized:
#             self.logger.warning("MCP client already initialized")
#             return
            
#         try:
#             transport = StreamableHTTPTransport(
#                 endpoint=self.base_url,
#                 api_key=self.api_key,
#                 sse_endpoint=self.sse_endpoint
#             )
            
#             self.client = StandardMCPClient(transport)
#             await self.client.initialize()
            
#             # Discover and register tools
#             await self.discover_tools()
#              # Also discover prompts
#             await self.discover_prompts()
            
#             self._initialized = True
#             self.logger.info("MCP Client Wrapper initialized successfully")
            
#         except Exception as e:
#             self.logger.error(f"Failed to initialize MCP client: {e}")
#             raise
    
#     async def list_available_tools(self) -> List[Dict[str, Any]]:
#         """List all available tools from MCP server."""
#         if not self.client:
#             raise RuntimeError("MCP client not initialized")
#         return await self.client.list_tools()
    
#     async def discover_tools(self):
#         """Dynamically discover and register tools from MCP server"""
#         if not self.client:
#             raise RuntimeError("MCP client not initialized")
            
#         try:
#             mcp_tools = await self.client.list_tools()
#             self.logger.info(f"MCP server returned {len(mcp_tools)} tools")
            
#             registered_count = 0
            
#             for tool_config in mcp_tools:
#                 try:
#                     # Extract all available metadata
#                     name = tool_config.get('name')
#                     if not name:
#                         self.logger.warning("Tool without name found, skipping")
#                         continue
                    
#                     # Handle both input_schema and inputSchema
#                     schema = tool_config.get('input_schema') or tool_config.get('inputSchema', {})
                    
#                     # Ensure schema has basic structure
#                     if not isinstance(schema, dict):
#                         schema = {"type": "object", "properties": {}, "required": []}
#                     elif 'type' not in schema:
#                         schema['type'] = 'object'
#                     if 'properties' not in schema:
#                         schema['properties'] = {}
                    
#                     # Extract enhanced metadata
#                     tool_metadata = ToolMetadata(
#                         name=name,
#                         description=tool_config.get('description', f'MCP tool: {name}'),
#                         input_schema=schema,
#                         category=tool_config.get('category', 'banking'),
#                         permissions=tool_config.get('permissions', []),
#                         method=tool_config.get('method', 'POST'),
#                         path=tool_config.get('path', f'/{name}'),
#                         tags=tool_config.get('tags', []),
#                         deprecated=tool_config.get('deprecated', False),
#                         auth_required=tool_config.get('auth_required', True),
#                         operation_id=tool_config.get('operation_id'),
#                         parameters=tool_config.get('parameters', []),
#                         examples=tool_config.get('examples', {})
#                     )
                    
#                     self.tool_registry.register_tool(tool_metadata)
#                     registered_count += 1
                    
#                 except Exception as e:
#                     self.logger.error(f"Failed to register tool {tool_config.get('name', 'unknown')}: {e}")
                    
#             self.logger.info(f"Tool discovery completed: {registered_count}/{len(mcp_tools)} tools registered")
            
#             # Log statistics
#             stats = self.tool_registry.get_statistics()
#             self.logger.info(f"Registry statistics: {stats}")
            
#         except Exception as e:
#             self.logger.error(f"Tool discovery failed: {e}")
#             raise

#     async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
#         """Execute a tool with comprehensive validation and logging"""
#         if not self._initialized:
#             raise RuntimeError("MCP client not initialized")
            
#         start_time = datetime.now()
        
#         # if tool_name == 'list_api_endpoints':
#         #     # Force compact mode unless explicitly set to False
#         #     if 'compact' not in parameters:
#         #         parameters['compact'] = True
#         #     if 'max_results' not in parameters:
#         #         parameters['max_results'] = 50
#         # Retrieve tool metadata
#         tool_metadata = self.tool_registry.get_tool(tool_name)
#         if not tool_metadata:
#             # Try to find by operation ID
#             tool_metadata = self.tool_registry.get_tool_by_operation_id(tool_name)
#             if not tool_metadata:
#                 raise ValueError(f"Tool {tool_name} not found in registry")
#             tool_name = tool_metadata.name

#         try:
#             # Validate input parameters
#             jsonschema.validate(instance=parameters, schema=tool_metadata.input_schema)
            
#             # Check deprecation
#             if tool_metadata.deprecated:
#                 self.logger.warning(f"Executing deprecated tool: {tool_name}")
            
#             # Log tool call
#             self.interaction_logger.info(
#                 f"Executing Tool: {tool_name} [{tool_metadata.method} {tool_metadata.path}], "
#                 f"Category: {tool_metadata.category}, "
#                 f"Tags: {tool_metadata.tags}, "
#                 f"Parameters: {self._mask_sensitive_data(parameters)}"
#             )
            
#             # Execute tool
#             result = await self.client.call_tool(tool_name, parameters)
            
#             # Update tool metadata
#             tool_metadata.last_used = datetime.now()
#             tool_metadata.usage_count += 1
            
#             # Log successful execution
#             execution_time = (datetime.now() - start_time).total_seconds()
#             self.interaction_logger.info(
#                 f"Tool Execution Complete: {tool_name}, "
#                 f"Execution Time: {execution_time:.2f}s, "
#                 f"Total Usage: {tool_metadata.usage_count}"
#             )
            
#             return result
        
#         except jsonschema.ValidationError as ve:
#             # Log validation error with details
#             self.interaction_logger.error(
#                 f"Tool Validation Error: {tool_name}, "
#                 f"Error: {str(ve)}, "
#                 f"Failed at: {ve.json_path}"
#             )
#             raise ValueError(f"Invalid parameters for tool {tool_name}: {ve.message}")
        
#         except Exception as e:
#             # Log execution error
#             self.interaction_logger.error(
#                 f"Tool Execution Error: {tool_name}, "
#                 f"Error Type: {type(e).__name__}, "
#                 f"Error: {str(e)}"
#             )
#             raise
    
#     def _is_response_too_large(self, result):
#         """Check if response might overwhelm context"""
#         result_str = json.dumps(result) if isinstance(result, dict) else str(result)
#         return len(result_str) > 5000  # 5KB threshold
        
#     def _create_summary_response(self, result):
#         """Create a summary for large responses"""
#         return {
#             "summary": "Response too large - showing summary",
#             "total_results": result.get('total_found', 'many'),
#             "message": "Please use more specific filters (tag + search_query)",
#             "available_tags": result.get('available_tags', []),
#             "sample_apis": self._extract_sample_apis(result, limit=10)
#         }
#     def _mask_sensitive_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
#         """Mask sensitive information in tool parameters"""
#         if not params:
#             return {}
            
#         masked_params = params.copy()
#         sensitive_keys = [
#             'password', 'token', 'secret', 'api_key', 'apikey',
#             'auth', 'authorization', 'credential', 'private_key',
#             'access_token', 'refresh_token', 'session_id'
#         ]
        
#         def mask_dict(d: Dict[str, Any]):
#             for key, value in d.items():
#                 key_lower = key.lower()
#                 if any(sensitive in key_lower for sensitive in sensitive_keys):
#                     d[key] = '****'
#                 elif isinstance(value, dict):
#                     mask_dict(value)
#                 elif isinstance(value, list):
#                     for item in value:
#                         if isinstance(item, dict):
#                             mask_dict(item)
        
#         mask_dict(masked_params)
#         return masked_params
    
#     async def list_prompts(self) -> List[Dict[str, Any]]:
#         """List all available prompts from the MCP server."""
#         if not self.client:
#             raise RuntimeError("MCP client not initialized")
        
#         try:
#             # Check if client has list_prompts method
#             if hasattr(self.client, 'list_prompts'):
#                 prompts = await self.client.list_prompts()
#                 return prompts
#             else:
#                 # Client doesn't support prompts
#                 self.logger.warning("MCP client doesn't have list_prompts method")
#                 return []
#         except Exception as e:
#             self.logger.error(f"Failed to list prompts: {e}")
#             return []

#     async def get_prompt(self, name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
#         """Get a specific prompt with arguments filled in."""
#         if not self.client:
#             raise RuntimeError("MCP client not initialized")
        
#         try:
#             # Check if client has get_prompt method
#             if hasattr(self.client, 'get_prompt'):
#                 result = await self.client.get_prompt(name, arguments)
#                 return result
#             else:
#                 raise RuntimeError("MCP client doesn't have get_prompt method")
#         except Exception as e:
#             self.logger.error(f"Failed to get prompt '{name}': {e}")
#             raise

#     async def discover_prompts(self):
#         """Discover and cache available prompts"""
#         try:
#             self.available_prompts = await self.list_prompts()
#             if self.available_prompts:
#                 self.logger.info(f"Discovered {len(self.available_prompts)} MCP prompts")
#                 for prompt in self.available_prompts:
#                     self.logger.info(f"  - {prompt['name']}: {prompt['description']}")
#             else:
#                 self.logger.info("No MCP prompts discovered")
#         except Exception as e:
#             self.logger.error(f"Failed to discover prompts: {e}")
#             self.available_prompts = []
            
#     async def get_contextual_prompt(self, context: str):
#         """Get appropriate prompt based on context"""
        
#         if "error" in context:
#             return await self.mcp_wrapper.get_prompt("banking_error_handling")
#         elif "discover" in context:
#             return await self.mcp_wrapper.get_prompt("discover_banking_apis_by_domain")
#         else:
#             return await self.mcp_wrapper.get_prompt("banking_api_quick_start")

#     async def close(self):
#         """Clean up resources"""
#         if self.client:
#             # Add any cleanup if needed
#             self._initialized = False
#             self.logger.info("MCP client closed")