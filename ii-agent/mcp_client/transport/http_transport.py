"""
Fixed HTTP transport with correct URL handling and request formatting
"""
import logging
import httpx
import uuid
import json
import re
from typing import Dict, Any, Optional
from .base_transport import BaseTransport
from ..protocol.messages import MCPRequest, MCPResponse, MCPNotification

class StandardHTTPTransport(BaseTransport):
    """MCP-compliant HTTP transport with corrected URL and request handling."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0):
        # Fix URL handling - remove trailing slashes and normalize
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.client: Optional[httpx.AsyncClient] = None
        self.session_id: Optional[str] = None
        self.logger = logging.getLogger(__name__)
        
        # Determine the correct endpoint URL
        # Based on server logs, it's expecting /mcp/ but we should try /mcp first
        if self.base_url.endswith('/mcp'):
            # URL already has /mcp - use as-is
            self.mcp_endpoint = self.base_url
        else:
            # URL doesn't have /mcp - add it
            self.mcp_endpoint = f"{self.base_url}/mcp"
        
        self.logger.debug("Base URL (input): %s", base_url)
        self.logger.debug("Base URL (normalized): %s", self.base_url)
        self.logger.debug("MCP Endpoint (final): %s", self.mcp_endpoint)

    async def connect(self) -> None:
        """Establish HTTP client connection."""
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            follow_redirects=True
        )
        
        # Generate session ID for this connection
        self.session_id = str(uuid.uuid4())
        
        self.logger.info("Standard HTTP transport connected to %s", self.mcp_endpoint)
        self.logger.debug("Created session ID: %s", self.session_id)

    async def send_request(self, request: MCPRequest) -> MCPResponse:
        """Send MCP request with proper formatting."""
        if not self.client:
            raise RuntimeError("Transport not connected")

        # Ensure request has proper format
        if not hasattr(request, 'id') or request.id is None:
            request.id = str(uuid.uuid4())
            self.logger.debug("Added missing request ID: %s", request.id)

        # Build headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "X-API-Key": self.api_key,
            "User-Agent": "BankingMCPClient/1.0.0"
        }

        # Prepare request data
        request_data = request.to_dict()
        
        # Ensure the request is properly formatted
        if 'jsonrpc' not in request_data:
            request_data['jsonrpc'] = '2.0'
        
        # Ensure ID is present for requests (not notifications)
        if 'id' not in request_data:
            request_data['id'] = str(uuid.uuid4())

        try:
            self.logger.debug("Sending MCP request: %s to %s", request.method, self.mcp_endpoint)
            self.logger.debug("Request data: %s", json.dumps(request_data, indent=2))
            
            response = await self.client.post(
                self.mcp_endpoint,
                json=request_data,
                headers=headers
            )
            
            self.logger.debug("Response status: %s", response.status_code)
            self.logger.debug("Response headers: %s", dict(response.headers))
            
            # Handle different response status codes
            if response.status_code == 202:
                self.logger.info("Received 202 Accepted - server is processing request")
                # For 202, we might need to wait for SSE response
                response_text = response.text
                self.logger.debug("202 response text: %s", response_text)
                
                # If there's content, try to parse it
                if response_text:
                    return self._parse_response_content(response_text, request.id)
                else:
                    # Return a pending response
                    return MCPResponse(
                        jsonrpc="2.0",
                        id=request.id,
                        result={"status": "accepted", "message": "Request accepted for processing"}
                    )
            elif response.status_code == 200:
                response_text = response.text
                self.logger.debug("200 response text: %s", response_text[:500])
                return self._parse_response_content(response_text, request.id)
            else:
                response.raise_for_status()

        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_detail = e.response.text
            except:
                pass
            
            self.logger.error("HTTP error %s: %s", e.response.status_code, error_detail)
            raise MCPTransportError(f"HTTP {e.response.status_code}: {error_detail}")
        except Exception as e:
            self.logger.error("Transport error: %s", str(e))
            raise MCPTransportError(f"Request failed: {str(e)}")

    def _parse_response_content(self, content: str, request_id: Optional[str] = None) -> MCPResponse:
        """Parse response content (JSON or SSE)."""
        
        # Check if it's SSE format
        if "event:" in content or "data:" in content:
            return self._parse_sse_response(content, request_id)
        else:
            # Try to parse as JSON
            try:
                response_data = json.loads(content)
                return MCPResponse.from_dict(response_data)
            except json.JSONDecodeError:
                self.logger.warning("Could not parse response as JSON: %s", content[:200])
                # Return a wrapper response
                return MCPResponse(
                    jsonrpc="2.0",
                    id=request_id,
                    result={"raw_content": content}
                )

    def _parse_sse_response(self, sse_text: str, request_id: Optional[str] = None) -> MCPResponse:
        """Parse SSE (Server-Sent Events) response format."""
        self.logger.debug("Parsing SSE response: %s", sse_text[:200])
        
        try:
            # Extract the JSON data from SSE format
            lines = sse_text.strip().split('\n')
            json_data = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('data: '):
                    # Extract JSON after 'data: '
                    json_str = line[6:]  # Remove 'data: ' prefix
                    try:
                        json_data = json.loads(json_str)
                        break
                    except json.JSONDecodeError as e:
                        self.logger.warning("Failed to parse JSON data line: %s", line)
                        continue
            
            if json_data is None:
                # Try alternative parsing - look for JSON pattern
                json_match = re.search(r'\{.*\}', sse_text, re.DOTALL)
                if json_match:
                    try:
                        json_data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass
            
            if json_data is None:
                raise ValueError("No valid JSON found in SSE response")
            
            self.logger.debug("Extracted JSON from SSE: %s", json_data)
            
            # Convert to MCPResponse
            mcp_response = MCPResponse.from_dict(json_data)
            self.logger.debug("Successfully parsed SSE response")
            
            return mcp_response
            
        except Exception as e:
            self.logger.error("Failed to parse SSE response: %s", str(e))
            
            # Return error response
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
        """Close HTTP transport."""
        if self.client:
            await self.client.aclose()
            self.client = None
            
        self.session_id = None
        self.logger.info("Standard HTTP transport disconnected")

class MCPTransportError(Exception):
    """MCP transport-specific error."""
    pass