#!/usr/bin/env python3
# sse_streaming_handler.py
"""
Specialized SSE (Server-Sent Events) handler for MCP with OAuth
Handles streaming responses, reconnection, and buffering issues
"""

import asyncio
import json
import logging
from typing import AsyncIterator, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import aiohttp
from aiohttp import web
import backoff
from enum import Enum

logger = logging.getLogger(__name__)

class SSEEventType(Enum):
    """SSE event types for MCP"""
    MESSAGE = "message"
    NOTIFICATION = "notification"
    PROGRESS = "progress"
    ERROR = "error"
    PING = "ping"
    TOOLS_LIST_CHANGED = "tools_list_changed"
    RESOURCE_UPDATED = "resource_updated"

@dataclass
class SSEEvent:
    """Represents a single SSE event"""
    id: Optional[str] = None
    event: Optional[str] = None
    data: str = ""
    retry: Optional[int] = None
    
    def format(self) -> bytes:
        """Format event for SSE protocol"""
        lines = []
        
        if self.id:
            lines.append(f"id: {self.id}")
        if self.event:
            lines.append(f"event: {self.event}")
        if self.retry:
            lines.append(f"retry: {self.retry}")
        
        # Handle multi-line data
        for line in self.data.split('\n'):
            lines.append(f"data: {line}")
        
        # End with double newline
        return '\n'.join(lines).encode('utf-8') + b'\n\n'

class SSEClient:
    """Client for consuming SSE streams with reconnection"""
    
    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        reconnect_interval: int = 3000,  # milliseconds
        max_retries: int = 5
    ):
        self.url = url
        self.headers = headers or {}
        self.reconnect_interval = reconnect_interval
        self.max_retries = max_retries
        self.last_event_id: Optional[str] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
    
    async def __aenter__(self):
        """Async context manager entry"""
        self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def close(self):
        """Close the client"""
        self._running = False
        if self._session:
            await self._session.close()
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=5,
        max_time=60
    )
    async def _connect(self) -> aiohttp.ClientResponse:
        """Connect to SSE endpoint with retry logic"""
        headers = self.headers.copy()
        headers['Accept'] = 'text/event-stream'
        headers['Cache-Control'] = 'no-cache'
        
        # Add Last-Event-ID for reconnection
        if self.last_event_id:
            headers['Last-Event-ID'] = self.last_event_id
        
        if not self._session:
            self._session = aiohttp.ClientSession()
        
        response = await self._session.get(
            self.url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=None, sock_read=30)
        )
        
        if response.status != 200:
            text = await response.text()
            raise aiohttp.ClientResponseError(
                request_info=response.request_info,
                history=response.history,
                status=response.status,
                message=f"SSE connection failed: {text}"
            )
        
        return response
    
    async def events(self) -> AsyncIterator[SSEEvent]:
        """Stream SSE events with automatic reconnection"""
        self._running = True
        
        while self._running:
            try:
                response = await self._connect()
                logger.info(f"Connected to SSE endpoint: {self.url}")
                
                async for event in self._parse_events(response):
                    if event.id:
                        self.last_event_id = event.id
                    yield event
                    
            except aiohttp.ClientError as e:
                logger.error(f"SSE connection error: {e}")
                if self._running:
                    await asyncio.sleep(self.reconnect_interval / 1000)
            except Exception as e:
                logger.error(f"Unexpected SSE error: {e}")
                raise
    
    async def _parse_events(self, response: aiohttp.ClientResponse) -> AsyncIterator[SSEEvent]:
        """Parse SSE events from response stream"""
        event = SSEEvent()
        
        async for line in response.content:
            line = line.decode('utf-8').rstrip('\r\n')
            
            if not line:  # Empty line = end of event
                if event.data:  # Only yield if there's data
                    yield event
                    event = SSEEvent()
                continue
            
            if line.startswith(':'):  # Comment
                continue
            
            # Parse field
            if ':' in line:
                field, value = line.split(':', 1)
                value = value.lstrip()
            else:
                field = line
                value = ''
            
            # Handle fields
            if field == 'id':
                event.id = value
            elif field == 'event':
                event.event = value
            elif field == 'data':
                if event.data:
                    event.data += '\n'
                event.data += value
            elif field == 'retry':
                try:
                    event.retry = int(value)
                    self.reconnect_interval = int(value)
                except ValueError:
                    pass

class SSEServer:
    """Server for sending SSE streams with proper buffering and error handling"""
    
    def __init__(self, 
                 heartbeat_interval: int = 30,
                 event_queue_size: int = 100):
        self.heartbeat_interval = heartbeat_interval
        self.event_queue_size = event_queue_size
        self.clients: Dict[str, 'SSEConnection'] = {}
    
    async def handle_request(self, request: web.Request) -> web.StreamResponse:
        """Handle SSE request"""
        # Create response with proper headers
        response = web.StreamResponse(
            status=200,
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',  # Disable Nginx buffering
                'Access-Control-Allow-Origin': '*'
            }
        )
        
        await response.prepare(request)
        
        # Create connection
        connection = SSEConnection(response, request)
        connection_id = connection.id
        self.clients[connection_id] = connection
        
        try:
            # Send initial connection event
            await connection.send_event(SSEEvent(
                event="connected",
                data=json.dumps({
                    "connection_id": connection_id,
                    "timestamp": datetime.now().isoformat()
                })
            ))
            
            # Start heartbeat
            heartbeat_task = asyncio.create_task(
                self._heartbeat(connection)
            )
            
            # Keep connection alive
            await connection.wait_closed()
            
        finally:
            # Cleanup
            heartbeat_task.cancel()
            del self.clients[connection_id]
            logger.info(f"SSE client disconnected: {connection_id}")
        
        return response
    
    async def _heartbeat(self, connection: 'SSEConnection'):
        """Send periodic heartbeat to keep connection alive"""
        while not connection.closed:
            try:
                await connection.send_event(SSEEvent(
                    event="ping",
                    data=json.dumps({
                        "timestamp": datetime.now().isoformat()
                    })
                ))
                await asyncio.sleep(self.heartbeat_interval)
            except Exception:
                break
    
    async def broadcast_event(self, event: SSEEvent):
        """Broadcast event to all connected clients"""
        disconnected = []
        
        for client_id, connection in self.clients.items():
            try:
                await connection.send_event(event)
            except Exception:
                disconnected.append(client_id)
        
        # Remove disconnected clients
        for client_id in disconnected:
            self.clients.pop(client_id, None)
    
    async def send_to_client(self, client_id: str, event: SSEEvent):
        """Send event to specific client"""
        connection = self.clients.get(client_id)
        if connection:
            await connection.send_event(event)

@dataclass
class SSEConnection:
    """Represents a single SSE connection"""
    response: web.StreamResponse
    request: web.Request
    id: str = field(default_factory=lambda: f"sse-{int(datetime.now().timestamp() * 1000)}")
    created_at: datetime = field(default_factory=datetime.now)
    last_event_id: Optional[str] = None
    closed: bool = False
    _write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    def __post_init__(self):
        """Extract Last-Event-ID from request"""
        self.last_event_id = self.request.headers.get('Last-Event-ID')
    
    async def send_event(self, event: SSEEvent):
        """Send an event to the client"""
        if self.closed:
            raise ConnectionError("Connection is closed")
        
        async with self._write_lock:
            try:
                await self.response.write(event.format())
                await self.response.drain()
            except (ConnectionResetError, ConnectionAbortedError):
                self.closed = True
                raise
    
    async def wait_closed(self):
        """Wait until connection is closed"""
        while not self.closed:
            await asyncio.sleep(1)

class MCPSSEProxy:
    """Specialized SSE proxy for MCP with OAuth integration"""
    
    def __init__(
        self,
        mcp_sse_url: str,
        oauth_validator: Optional[Callable] = None
    ):
        self.mcp_sse_url = mcp_sse_url
        self.oauth_validator = oauth_validator
        self.sse_server = SSEServer()
    
    async def handle_sse_proxy(self, request: web.Request) -> web.StreamResponse:
        """Proxy SSE requests to MCP server with OAuth validation"""
        
        # Validate OAuth token if validator provided
        if self.oauth_validator:
            auth_header = request.headers.get('Authorization', '')
            if not auth_header.startswith('Bearer '):
                return web.Response(
                    status=401,
                    text='Unauthorized: Bearer token required'
                )
            
            token = auth_header[7:]
            user_info = await self.oauth_validator(token)
            
            if not user_info:
                return web.Response(
                    status=401,
                    text='Unauthorized: Invalid token'
                )
        else:
            user_info = None
        
        # Create SSE response
        response = web.StreamResponse(
            status=200,
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
                'Access-Control-Allow-Origin': '*'
            }
        )
        
        await response.prepare(request)
        
        # Proxy headers
        proxy_headers = {
            'Accept': 'text/event-stream',
            'Cache-Control': 'no-cache'
        }
        
        if user_info:
            proxy_headers['X-User-Context'] = json.dumps({
                'username': user_info.get('username'),
                'sub': user_info.get('sub'),
                'scopes': user_info.get('scopes', [])
            })
        
        # Connect to upstream SSE
        async with SSEClient(
            self.mcp_sse_url,
            headers=proxy_headers
        ) as client:
            try:
                # Initial connection event
                await response.write(SSEEvent(
                    event="proxy-connected",
                    data=json.dumps({
                        "upstream": self.mcp_sse_url,
                        "timestamp": datetime.now().isoformat()
                    })
                ).format())
                
                # Proxy events
                async for event in client.events():
                    # Optionally transform event
                    transformed = await self._transform_event(event, user_info)
                    await response.write(transformed.format())
                    await response.drain()
                    
            except Exception as e:
                logger.error(f"SSE proxy error: {e}")
                error_event = SSEEvent(
                    event="error",
                    data=json.dumps({
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
                )
                await response.write(error_event.format())
        
        return response
    
    async def _transform_event(
        self,
        event: SSEEvent,
        user_info: Optional[Dict[str, Any]]
    ) -> SSEEvent:
        """Transform event before sending to client"""
        # Add user context to certain events if needed
        if user_info and event.event in ["tools_list_changed", "resource_updated"]:
            try:
                data = json.loads(event.data)
                data["_user_context"] = {
                    "username": user_info.get("username"),
                    "sub": user_info.get("sub")
                }
                event.data = json.dumps(data)
            except json.JSONDecodeError:
                pass  # Keep original data if not JSON
        
        return event

# Example usage
async def example_sse_server():
    """Example SSE server setup"""
    app = web.Application()
    sse_server = SSEServer()
    
    # Route for SSE
    app.router.add_get('/events', sse_server.handle_request)
    
    # Route to send events
    async def send_event(request: web.Request):
        event_type = request.match_info.get('type', 'message')
        data = await request.json()
        
        event = SSEEvent(
            event=event_type,
            data=json.dumps(data),
            id=str(int(datetime.now().timestamp() * 1000))
        )
        
        await sse_server.broadcast_event(event)
        return web.json_response({"status": "sent"})
    
    app.router.add_post('/send/{type}', send_event)
    
    return app

async def example_sse_client():
    """Example SSE client usage"""
    async with SSEClient(
        url="http://localhost:8084/events",
        headers={"Authorization": "Bearer your-token"}
    ) as client:
        async for event in client.events():
            print(f"Event: {event.event}")
            print(f"Data: {event.data}")
            print(f"ID: {event.id}")
            print("-" * 40)

if __name__ == "__main__":
    # Run example server
    app = asyncio.run(example_sse_server())
    web.run_app(app, host="0.0.0.0", port=8084)