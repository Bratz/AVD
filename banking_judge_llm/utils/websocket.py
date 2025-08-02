from fastapi import WebSocket
from typing import Dict, List
import logging

class WebSocketManager:
    """Manage WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def initialize(self):
        """Initialize WebSocket manager."""
        self.logger.info("WebSocketManager initialized")

    async def cleanup(self):
        """Clean up all WebSocket connections."""
        for session_id in list(self.active_connections.keys()):
            for websocket in self.active_connections[session_id]:
                await self.disconnect(websocket, session_id)
        self.logger.info("WebSocketManager cleaned up")

    async def connect(self, websocket: WebSocket, session_id: str):
        """Connect a WebSocket client."""
        await websocket.accept()
        self.active_connections.setdefault(session_id, []).append(websocket)
        self.logger.debug(f"WebSocket connected for session: {session_id}")

    async def disconnect(self, websocket: WebSocket, session_id: str):
        """Disconnect a WebSocket client."""
        if session_id in self.active_connections:
            if websocket in self.active_connections[session_id]:
                self.active_connections[session_id].remove(websocket)
                await websocket.close()
                self.logger.debug(f"WebSocket disconnected for session: {session_id}")
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]

    async def send_progress(self, session_id: str, message: dict):
        """Send progress update to connected WebSocket clients."""
        if session_id in self.active_connections:
            for websocket in self.active_connections[session_id]:
                try:
                    await websocket.send_json(message)
                    self.logger.debug(f"Sent WebSocket message to {session_id}: {message}")
                except Exception as e:
                    self.logger.error(f"Failed to send WebSocket message: {str(e)}")
                    await self.disconnect(websocket, session_id)