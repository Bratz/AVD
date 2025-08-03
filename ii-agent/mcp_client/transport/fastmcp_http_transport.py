"""
Enhanced FastMCP HTTP transport with comprehensive debugging for 500 errors
"""
import logging
import httpx
import uuid
import json
import re
from typing import Dict, Any, Optional
from .base_transport import BaseTransport
from ..protocol.messages import MCPRequest, MCPResponse, MCPNotification

class FastMCPHTTPTransport(BaseTransport):
    """FastMCP HTTP transport with enhanced 500 error debugging."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0):
        # Smart URL handling - remove trailing slashes to prevent redirects
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.client: Optional[httpx.AsyncClient] = None
        self.session_id: Optional[str] = None
        self.logger = logging.getLogger(__name__)
        
        # Determine correct endpoint
        if self.base_url.endswith('/mcp'):
            self.mcp_endpoint = self.base_url
        else:
            self.mcp_endpoint = f"{self.base_url}/mcp"
        
        # Remove trailing slash to prevent redirects
        self.mcp_endpoint = self.mcp_endpoint.rstrip('/')
        
        self.logger.debug("FastMCP Transport initialized:")
        self.logger.debug("  Base URL: %s", self.base_url)
        self.logger.debug("  MCP Endpoint: %s", self.mcp_endpoint)

    async def connect(self) -> None:
        """Establish HTTP client connection."""
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            follow_redirects=False  # Don't follow redirects to debug URL issues
        )
        
        self.session_id = str(uuid.uuid4())
        self.logger.info("FastMCP HTTP transport connected to %s", self.mcp_endpoint)
        self.logger.debug("Generated session ID: %s", self.session_id)

    async def send_request(self, request: MCPRequest) -> MCPResponse:
        """Send MCP request with comprehensive 500 error debugging."""
        if not self.client:
            raise RuntimeError("Transport not connected")

        # Build proper headers for FastMCP
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "X-API-Key": self.api_key,
            "User-Agent": "FastMCPClient/1.0.0"
        }

        # Prepare request payload
        request_payload = request.to_dict()
        
        # Add ID if missing (required for proper JSON-RPC)
        if "id" not in request_payload or request_payload["id"] is None:
            request_payload["id"] = str(uuid.uuid4())

        try:
            self.logger.debug("=" * 60)
            self.logger.debug("SENDING FASTMCP REQUEST")
            self.logger.debug("Endpoint: %s", self.mcp_endpoint)
            self.logger.debug("Method: POST")
            self.logger.debug("Headers: %s", {k: v for k, v in headers.items() if k != "X-API-Key"})
            self.logger.debug("API Key: %s...", self.api_key[:10] if self.api_key else "None")
            self.logger.debug("Payload: %s", json.dumps(request_payload, indent=2))
            self.logger.debug("=" * 60)
            
            response = await self.client.post(
                self.mcp_endpoint,
                json=request_payload,
                headers=headers
            )
            
            self.logger.debug("=" * 60)
            self.logger.debug("RECEIVED RESPONSE")
            self.logger.debug("Status Code: %s", response.status_code)
            self.logger.debug("Response Headers: %s", dict(response.headers))
            self.logger.debug("Response Text (first 1000 chars): %s", response.text[:1000])
            self.logger.debug("=" * 60)
            
            # Handle different response codes
            if response.status_code == 500:
                self.logger.error("🚨 HTTP 500 INTERNAL SERVER ERROR")
                self.logger.error("This usually indicates a server-side issue.")
                
                # Try to extract error details
                try:
                    error_content = response.text
                    self.logger.error("Server Error Response: %s", error_content)
                    
                    # Check if it's JSON error
                    if error_content.strip().startswith('{'):
                        try:
                            error_json = response.json()
                            self.logger.error("Parsed Error JSON: %s", json.dumps(error_json, indent=2))
                        except:
                            pass
                    
                    # Look for common FastMCP error patterns
                    if "tool" in error_content.lower():
                        self.logger.error("❌ Possible tool-related error")
                    if "validation" in error_content.lower():
                        self.logger.error("❌ Possible request validation error")
                    if "config" in error_content.lower():
                        self.logger.error("❌ Possible server configuration error")
                    
                except Exception as e:
                    self.logger.error("Could not parse error response: %s", str(e))
                
                # Check server logs suggestion
                self.logger.error("🔍 Debugging suggestions:")
                self.logger.error("1. Check MCP server logs for detailed error")
                self.logger.error("2. Verify server configuration")
                self.logger.error("3. Check if tools are properly registered")
                self.logger.error("4. Verify request format matches server expectations")
            
            elif response.status_code == 307:
                redirect_location = response.headers.get('Location', 'Unknown')
                self.logger.warning("HTTP 307 Redirect to: %s", redirect_location)
                self.logger.warning("Endpoint URL might need adjustment")
            
            elif response.status_code == 404:
                self.logger.error("HTTP 404 Not Found - endpoint doesn't exist")
                self.logger.error("Current endpoint: %s", self.mcp_endpoint)
            
            # Raise for status (will raise exception for 4xx, 5xx)
            response.raise_for_status()
            
            if not response.content:
                raise ValueError("Empty response from FastMCP server")

            # Handle response format
            content_type = response.headers.get("content-type", "").lower()
            response_text = response.text
            
            if "text/event-stream" in content_type or "event:" in response_text:
                # Handle SSE response
                self.logger.debug("Processing SSE response")
                return self._parse_sse_response(response_text, request_payload.get("id"))
            elif "application/json" in content_type:
                # Handle JSON response
                self.logger.debug("Processing JSON response")
                response_data = response.json()
                return MCPResponse.from_dict(response_data)
            else:
                # Try SSE parsing first
                if "event:" in response_text or "data:" in response_text:
                    return self._parse_sse_response(response_text, request_payload.get("id"))
                else:
                    try:
                        response_data = response.json()
                        return MCPResponse.from_dict(response_data)
                    except json.JSONDecodeError:
                        raise ValueError(f"Unsupported response format: {content_type}")

        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_detail = e.response.text
            except:
                pass
            
            self.logger.error("HTTP Error %s: %s", e.response.status_code, error_detail)
            
            # Enhanced error context for 500 errors
            if e.response.status_code == 500:
                self.logger.error("🚨 SERVER ERROR CONTEXT:")
                self.logger.error("  Request URL: %s", self.mcp_endpoint)
                self.logger.error("  Request Method: POST")
                self.logger.error("  Request Payload: %s", json.dumps(request_payload, indent=2))
                self.logger.error("  API Key: %s...", self.api_key[:10] if self.api_key else "None")
            
            raise MCPTransportError(f"HTTP {e.response.status_code}: {error_detail}")
        except Exception as e:
            self.logger.error("Transport error: %s", str(e))
            raise MCPTransportError(f"Request failed: {str(e)}")

    def _parse_sse_response(self, sse_text: str, request_id: Optional[str] = None) -> MCPResponse:
        """Parse SSE response format."""
        self.logger.debug("Parsing SSE response...")
        
        try:
            lines = sse_text.strip().split('\n')
            json_data = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('data: '):
                    json_str = line[6:]
                    try:
                        json_data = json.loads(json_str)
                        break
                    except json.JSONDecodeError:
                        continue
            
            if json_data is None:
                # Try direct JSON parsing
                json_match = re.search(r'\{.*\}', sse_text, re.DOTALL)
                if json_match:
                    try:
                        json_data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass
            
            if json_data is None:
                raise ValueError("No valid JSON found in SSE response")
            
            return MCPResponse.from_dict(json_data)
            
        except Exception as e:
            self.logger.error("Failed to parse SSE response: %s", str(e))
            return MCPResponse(
                jsonrpc="2.0",
                id=request_id,
                error={
                    "code": -32700,
                    "message": f"Parse error: {str(e)}",
                    "data": {"raw_response": sse_text[:500]}
                }
            )

    async def disconnect(self) -> None:
        """Close transport."""
        if self.client:
            await self.client.aclose()
            self.client = None
        self.session_id = None
        self.logger.info("FastMCP HTTP transport disconnected")

class MCPTransportError(Exception):
    """Transport error."""
    pass
