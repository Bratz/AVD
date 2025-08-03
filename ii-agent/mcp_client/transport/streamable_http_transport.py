"""
Fixed Streamable HTTP transport with SSE support and latest protocol version
"""
import logging
import httpx
import asyncio
import json
import re
from typing import Dict, Any, Optional, List, Union
from .base_transport import BaseTransport
from ..protocol.messages import MCPRequest, MCPResponse

class StreamableHTTPTransport(BaseTransport):
    """Fixed Streamable HTTP transport with SSE response handling."""

    def __init__(self, endpoint: str, api_key: str, sse_endpoint: Optional[str] = None):
        self.endpoint = endpoint.rstrip('/')
        self.sse_endpoint = sse_endpoint or f"{self.endpoint}/sse"
        self.api_key = api_key
        self.client = None
        self.session_id = None
        self.sse_task = None
        self.progress_queue = asyncio.Queue()
        self.logger = logging.getLogger(__name__)

    async def connect(self):
        """Establish connection with enhanced error handling."""
        self.client = httpx.AsyncClient(follow_redirects=True, timeout=20.0)
        
        # Enhanced headers for better compatibility
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": "BankingMCPClient/2.0.0"
        }
        
        # Determine endpoint URL
        url = f"{self.endpoint}/mcp" if not self.endpoint.endswith('/mcp') else self.endpoint

        # Updated protocol version to latest
        payloads = [
            {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",  # Latest version
                    "capabilities": {
                        "experimental": {},
                        "sampling": {},
                        "tools": {},
                        "prompts": {},
                        "resources": {}
                    },
                    "clientInfo": {
                        "name": "BankingMCPClient",
                        "version": "2.0.0"
                    }
                },
                "id": "init-1"
            },
            # Fallback with older version if needed
            {
                "jsonrpc": "2.0", 
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-10-07",  # Fallback version
                    "capabilities": {
                        "jsonrpc": "2.0",
                        "streaming": True,
                        "methods": ["initialize", "tools/list", "tools/call"]
                    },
                    "clientInfo": {
                        "name": "BankingMCPClient",
                        "version": "2.0.0"
                    }
                },
                "id": "init-2"
            }
        ]

        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            payload = payloads[(attempt - 1) % len(payloads)]
            
            try:
                self.logger.debug(f"Attempt {attempt}/{max_attempts}: Connecting to: {url}")
                self.logger.debug(f"Using protocol version: {payload['params']['protocolVersion']}")
                
                response = await self.client.post(url, json=payload, headers=headers)
                
                self.logger.debug(f"Response status: {response.status_code}")
                self.logger.debug(f"Response headers: {dict(response.headers)}")
                
                response.raise_for_status()
                
                # Handle different response formats
                content_type = response.headers.get("content-type", "").lower()
                response_text = response.text
                
                self.logger.debug(f"Content-Type: {content_type}")
                self.logger.debug(f"Response text preview: {response_text[:200]}")
                
                if not response_text:
                    self.logger.error("Empty response body")
                    raise ValueError("Empty response from MCP server")
                
                # Parse response based on format
                init_response = None
                
                if "text/event-stream" in content_type or "event:" in response_text:
                    self.logger.debug("Processing SSE response format")
                    init_response = self._parse_sse_response(response_text)
                elif "application/json" in content_type:
                    self.logger.debug("Processing JSON response format")
                    try:
                        init_response = MCPResponse.from_dict(response.json())
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON decode error: {e}")
                        raise ValueError(f"Invalid JSON response: {e}")
                else:
                    # Try SSE parsing first, then JSON
                    if "event:" in response_text or "data:" in response_text:
                        self.logger.debug("Detected SSE format in non-SSE content-type")
                        init_response = self._parse_sse_response(response_text)
                    else:
                        # Last resort: try JSON parsing
                        try:
                            json_data = response.json()
                            init_response = MCPResponse.from_dict(json_data)
                        except json.JSONDecodeError as e:
                            self.logger.error(f"Failed to parse response as JSON: {e}")
                            self.logger.error(f"Response text: {response_text}")
                            raise ValueError(f"Invalid JSON response: {e}")
                
                if init_response and init_response.result:
                    # Extract session ID if provided
                    self.session_id = response.headers.get("Mcp-Session-Id")
                    if not self.session_id:
                        # Try to get from response
                        self.session_id = init_response.result.get("sessionId")
                    
                    self.logger.info(f"Initialized session: {self.session_id}")
                    
                    # Check protocol version compatibility
                    server_version = init_response.result.get("protocolVersion")
                    if server_version:
                        self.logger.info(f"Server protocol version: {server_version}")
                        if server_version != payload['params']['protocolVersion']:
                            self.logger.warning(f"Protocol version mismatch: client={payload['params']['protocolVersion']}, server={server_version}")
                    
                    # Start SSE if available
                    if self.sse_endpoint:
                        self.sse_task = asyncio.create_task(self._receive_sse())
                    
                    return
                else:
                    raise ValueError("Invalid initialization response")
                    
            except httpx.HTTPStatusError as e:
                self.logger.error(f"Attempt {attempt} failed: HTTP {e.response.status_code}")
                error_text = e.response.text if hasattr(e.response, 'text') else str(e)
                self.logger.error(f"Error details: {error_text}")
                
                if e.response.status_code == 404:
                    self.logger.error("MCP endpoint not found - check server URL")
                elif e.response.status_code == 400:
                    self.logger.error("Bad request - check protocol version or request format")
                
                if attempt == max_attempts:
                    await self.disconnect()
                    raise StreamableHTTPError(f"HTTP {e.response.status_code}: {error_text}")
                    
                await asyncio.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Attempt {attempt} failed: {e}")
                if attempt == max_attempts:
                    await self.disconnect()
                    raise StreamableHTTPError(f"Transport failed: {str(e)}")
                    
                await asyncio.sleep(2)

    def _parse_sse_response(self, sse_text: str) -> MCPResponse:
        """Parse Server-Sent Events response format."""
        self.logger.debug("Parsing SSE response")
        
        try:
            # SSE format:
            # event: message
            # data: {"jsonrpc":"2.0","id":"...","result":{...}}
            
            lines = sse_text.strip().split('\n')
            json_data = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('data: '):
                    json_str = line[6:]  # Remove 'data: ' prefix
                    try:
                        json_data = json.loads(json_str)
                        break
                    except json.JSONDecodeError:
                        continue
            
            if json_data is None:
                # Alternative parsing: find JSON pattern
                json_match = re.search(r'\{.*\}', sse_text, re.DOTALL)
                if json_match:
                    try:
                        json_data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass
            
            if json_data is None:
                raise ValueError("No valid JSON found in SSE response")
            
            self.logger.debug(f"Extracted JSON from SSE: {json_data}")
            return MCPResponse.from_dict(json_data)
            
        except Exception as e:
            self.logger.error(f"SSE parsing failed: {e}")
            self.logger.debug(f"SSE text: {sse_text}")
            raise ValueError(f"SSE parse error: {e}")

    async def send_request(self, request: Union[MCPRequest, List[MCPRequest]]) -> Union[MCPResponse, List[MCPResponse]]:
        """Send request with SSE response handling."""
        if not self.client:
            raise RuntimeError("Transport not connected")
            
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
            
        url = f"{self.endpoint}/mcp" if not self.endpoint.endswith('/mcp') else self.endpoint
        is_batch = isinstance(request, list)
        payload = [r.to_dict() for r in request] if is_batch else request.to_dict()
        
        try:
            self.logger.debug(f"Sending request to: {url}")
            
            response = await self.client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            content_type = response.headers.get("content-type", "").lower()
            response_text = response.text
            
            if not response_text:
                raise ValueError("Empty response from server")
            
            # Handle response format
            if "text/event-stream" in content_type or "event:" in response_text:
                # SSE response
                if is_batch:
                    # For batch requests, we might get multiple SSE messages
                    # This is a simplified implementation
                    parsed_response = self._parse_sse_response(response_text)
                    return [parsed_response]  # Simplified for now
                else:
                    return self._parse_sse_response(response_text)
            else:
                # JSON response
                response_data = response.json()
                if is_batch:
                    return [MCPResponse.from_dict(item) for item in response_data]
                return MCPResponse.from_dict(response_data)
                
        except httpx.HTTPStatusError as e:
            error_text = e.response.text if hasattr(e.response, 'text') else str(e)
            self.logger.error(f"Request failed: HTTP {e.response.status_code}: {error_text}")
            raise StreamableHTTPError(f"HTTP {e.response.status_code}: {error_text}")
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            raise StreamableHTTPError(f"Request failed: {str(e)}")

    async def send_progress(self, session_id: str, progress: Dict[str, Any]) -> None:
        """Send progress notification."""
        # Implementation for progress notifications
        pass

    async def _receive_sse(self):
        """Receive SSE stream for notifications."""
        if not self.sse_endpoint:
            return
            
        headers = {
            "X-API-Key": self.api_key,
            "Accept": "text/event-stream"
        }
        
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
            
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                self.logger.debug(f"Starting SSE stream at: {self.sse_endpoint}")
                
                async with client.stream("GET", self.sse_endpoint, headers=headers, timeout=None) as response:
                    if response.status_code != 200:
                        self.logger.error(f"SSE stream failed: {response.status_code}")
                        return
                        
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                await self.progress_queue.put(data)
                            except json.JSONDecodeError:
                                pass
                                
        except Exception as e:
            self.logger.error(f"SSE stream error: {e}")

    async def receive_progress(self) -> Optional[Dict[str, Any]]:
        """Receive progress update."""
        try:
            return await asyncio.wait_for(self.progress_queue.get(), timeout=5.0)
        except asyncio.TimeoutError:
            return None

    async def disconnect(self):
        """Close connection."""
        if self.sse_task:
            self.sse_task.cancel()
            
        if self.client:
            await self.client.aclose()
            self.client = None
            
        self.session_id = None
        self.logger.info("Streamable HTTP transport disconnected")

class StreamableHTTPError(Exception):
    """Streamable HTTP transport error."""
    pass
