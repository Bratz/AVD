"""
A2A (Agent-to-Agent) Communication Server for II-Agent
Implements Google A2A protocol server-side for receiving agent communications
"""
import json
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime
from enum import Enum
import logging
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import httpx

from src.ii_agent.utils.logging_config import get_logger
from src.ii_agent.core.agent import IIAgent, AgentContext


class A2AServer:
    """
    A2A Server for inbound agent communication.
    Handles incoming requests from other agents via JSON-RPC 2.0.
    """
    
    def __init__(
        self,
        agent: IIAgent,
        port: int = 8001,
        host: str = "0.0.0.0"
    ):
        self.agent = agent
        self.port = port
        self.host = host
        self.logger = get_logger(f"A2AServer.{agent.agent_id}")
        
        # Create FastAPI app for this agent
        self.app = FastAPI(
            title=f"A2A Server - {agent.name}",
            description=f"Agent-to-Agent communication endpoint for {agent.name}",
            version="1.0.0"
        )
        
        # Method handlers
        self.method_handlers: Dict[str, Callable] = {
            "execute": self._handle_execute,
            "query": self._handle_query,
            "ping": self._handle_ping,
            "discover": self._handle_discover,
            "subscribe": self._handle_subscribe,
            "unsubscribe": self._handle_unsubscribe
        }
        
        # Subscription management
        self.subscriptions: Dict[str, Dict[str, Any]] = {}
        
        # Setup routes
        self._setup_routes()
        
        # Agent card
        self.agent_card = self._create_agent_card()
    
    def _create_agent_card(self) -> Dict[str, Any]:
        """Create agent discovery card"""
        return {
            "agent_id": self.agent.agent_id,
            "name": self.agent.name,
            "description": self.agent.description,
            "version": "1.0.0",
            "capabilities": self.agent.capabilities,
            "endpoints": {
                "rpc": f"http://{self.host}:{self.port}/a2a/rpc",
                "notify": f"http://{self.host}:{self.port}/a2a/notify",
                "discover": f"http://{self.host}:{self.port}/a2a/discover",
                "websocket": f"ws://{self.host}:{self.port}/a2a/ws"
            },
            "methods": list(self.method_handlers.keys()),
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "state": self.agent.state.value if hasattr(self.agent, 'state') else "active"
            }
        }
    
    def _setup_routes(self):
        """Setup FastAPI routes for A2A communication"""
        
        @self.app.get("/")
        async def root():
            return {"message": f"A2A Server for {self.agent.name}"}
        
        @self.app.get("/a2a/discover")
        async def discover():
            """Agent discovery endpoint"""
            return self.agent_card
        
        @self.app.post("/a2a/rpc")
        async def handle_rpc(request: Request):
            """Handle JSON-RPC requests"""
            try:
                body = await request.json()
                self.logger.debug(f"Received RPC request: {body}")
                
                # Validate JSON-RPC format
                if "jsonrpc" not in body or body["jsonrpc"] != "2.0":
                    return JSONResponse(
                        status_code=400,
                        content={
                            "jsonrpc": "2.0",
                            "error": {
                                "code": -32600,
                                "message": "Invalid Request"
                            },
                            "id": body.get("id")
                        }
                    )
                
                method = body.get("method")
                params = body.get("params", {})
                request_id = body.get("id")
                
                # Check if method exists
                if method not in self.method_handlers:
                    return JSONResponse(
                        content={
                            "jsonrpc": "2.0",
                            "error": {
                                "code": -32601,
                                "message": f"Method '{method}' not found"
                            },
                            "id": request_id
                        }
                    )
                
                # Execute method
                try:
                    result = await self.method_handlers[method](params)
                    return JSONResponse(
                        content={
                            "jsonrpc": "2.0",
                            "result": result,
                            "id": request_id
                        }
                    )
                except Exception as e:
                    self.logger.error(f"Method execution error: {e}")
                    return JSONResponse(
                        content={
                            "jsonrpc": "2.0",
                            "error": {
                                "code": -32603,
                                "message": str(e)
                            },
                            "id": request_id
                        }
                    )
                    
            except Exception as e:
                self.logger.error(f"RPC handling error: {e}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32700,
                            "message": "Parse error"
                        }
                    }
                )
        
        @self.app.post("/a2a/notify")
        async def handle_notification(request: Request):
            """Handle notifications (no response expected)"""
            try:
                body = await request.json()
                self.logger.info(f"Received notification: {body}")
                
                method = body.get("method")
                params = body.get("params", {})
                
                # Process notification asynchronously
                asyncio.create_task(self._process_notification(method, params))
                
                return {"status": "accepted"}
                
            except Exception as e:
                self.logger.error(f"Notification handling error: {e}")
                return {"status": "error", "message": str(e)}
        
        @self.app.websocket("/a2a/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time communication"""
            await websocket.accept()
            connection_id = str(uuid.uuid4())
            
            try:
                self.logger.info(f"WebSocket connection established: {connection_id}")
                
                while True:
                    data = await websocket.receive_json()
                    
                    # Handle WebSocket message
                    response = await self._handle_websocket_message(data, connection_id)
                    
                    if response:
                        await websocket.send_json(response)
                        
            except WebSocketDisconnect:
                self.logger.info(f"WebSocket disconnected: {connection_id}")
                # Clean up any subscriptions for this connection
                self._cleanup_connection_subscriptions(connection_id)
    
    async def _handle_execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task execution requests"""
        task = params.get("task", "")
        context = params.get("context", {})
        requesting_agent = params.get("requesting_agent", "unknown")
        
        self.logger.info(f"Execute request from {requesting_agent}: {task}")
        
        try:
            # Execute task using the agent
            result = await self.agent.execute(task)
            
            return {
                "success": True,
                "result": result,
                "agent_id": self.agent.agent_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_id": self.agent.agent_id
            }
    
    async def _handle_query(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle information queries"""
        query_type = params.get("query", "")
        query_params = params.get("params", {})
        requesting_agent = params.get("requesting_agent", "unknown")
        
        self.logger.info(f"Query from {requesting_agent}: {query_type}")
        
        # Handle different query types
        if query_type == "capabilities":
            return {
                "capabilities": self.agent.capabilities,
                "agent_id": self.agent.agent_id
            }
        
        elif query_type == "state":
            return {
                "state": self.agent.state.value if hasattr(self.agent, 'state') else "active",
                "agent_id": self.agent.agent_id
            }
        
        elif query_type == "memory":
            # Return agent's memory state (if accessible)
            if hasattr(self.agent, 'context') and self.agent.context:
                return {
                    "working_memory": self.agent.context.working_memory,
                    "current_goal": self.agent.context.current_goal,
                    "agent_id": self.agent.agent_id
                }
            return {"error": "Memory not accessible"}
        
        else:
            return {
                "error": f"Unknown query type: {query_type}",
                "available_queries": ["capabilities", "state", "memory"]
            }
    
    async def _handle_ping(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ping requests"""
        return {
            "pong": True,
            "agent_id": self.agent.agent_id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_discover(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle discovery requests"""
        return self.agent_card
    
    async def _handle_subscribe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle event subscription requests"""
        event_type = params.get("event_type", "")
        subscription_id = params.get("subscription_id", str(uuid.uuid4()))
        callback_url = params.get("callback_url", "")
        
        if not event_type or not callback_url:
            return {
                "success": False,
                "error": "event_type and callback_url are required"
            }
        
        # Store subscription
        self.subscriptions[subscription_id] = {
            "event_type": event_type,
            "callback_url": callback_url,
            "created_at": datetime.now().isoformat(),
            "active": True
        }
        
        self.logger.info(f"New subscription: {subscription_id} for {event_type}")
        
        return {
            "success": True,
            "subscription_id": subscription_id
        }
    
    async def _handle_unsubscribe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle unsubscribe requests"""
        subscription_id = params.get("subscription_id", "")
        
        if subscription_id in self.subscriptions:
            self.subscriptions[subscription_id]["active"] = False
            del self.subscriptions[subscription_id]
            
            return {
                "success": True,
                "message": f"Unsubscribed: {subscription_id}"
            }
        
        return {
            "success": False,
            "error": "Subscription not found"
        }
    
    async def _process_notification(self, method: str, params: Dict[str, Any]):
        """Process incoming notifications asynchronously"""
        try:
            # Log notification
            self.logger.info(f"Processing notification: {method}")
            
            # Handle different notification types
            if method == "agent_status_update":
                # Another agent is updating us about their status
                agent_id = params.get("agent_id")
                status = params.get("status")
                self.logger.info(f"Agent {agent_id} status: {status}")
                
            elif method == "task_completed":
                # Another agent completed a task we might be interested in
                task_id = params.get("task_id")
                result = params.get("result")
                self.logger.info(f"Task {task_id} completed with result: {result}")
                
            # Add more notification handlers as needed
            
        except Exception as e:
            self.logger.error(f"Notification processing error: {e}")
    
    async def _handle_websocket_message(
        self,
        message: Dict[str, Any],
        connection_id: str
    ) -> Optional[Dict[str, Any]]:
        """Handle WebSocket messages"""
        message_type = message.get("type", "")
        
        if message_type == "subscribe":
            # WebSocket subscription
            event_type = message.get("event_type", "")
            subscription_id = f"ws_{connection_id}_{uuid.uuid4().hex[:8]}"
            
            self.subscriptions[subscription_id] = {
                "event_type": event_type,
                "connection_id": connection_id,
                "type": "websocket",
                "created_at": datetime.now().isoformat(),
                "active": True
            }
            
            return {
                "type": "subscription_confirmed",
                "subscription_id": subscription_id
            }
        
        elif message_type == "ping":
            return {
                "type": "pong",
                "timestamp": datetime.now().isoformat()
            }
        
        return None
    
    def _cleanup_connection_subscriptions(self, connection_id: str):
        """Clean up subscriptions for a disconnected connection"""
        to_remove = []
        for sub_id, sub in self.subscriptions.items():
            if sub.get("connection_id") == connection_id:
                to_remove.append(sub_id)
        
        for sub_id in to_remove:
            del self.subscriptions[sub_id]
            self.logger.info(f"Cleaned up subscription: {sub_id}")
    
    async def emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event to all subscribers"""
        self.logger.debug(f"Emitting event: {event_type}")
        
        # Find all active subscriptions for this event type
        for sub_id, subscription in self.subscriptions.items():
            if subscription["active"] and subscription["event_type"] == event_type:
                if subscription.get("type") == "websocket":
                    # Handle WebSocket subscription
                    # This would need WebSocket connection management
                    pass
                else:
                    # HTTP callback
                    asyncio.create_task(
                        self._send_event_callback(
                            subscription["callback_url"],
                            event_type,
                            data
                        )
                    )
    
    async def _send_event_callback(
        self,
        callback_url: str,
        event_type: str,
        data: Dict[str, Any]
    ):
        """Send event to HTTP callback"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    callback_url,
                    json={
                        "event_type": event_type,
                        "data": data,
                        "timestamp": datetime.now().isoformat(),
                        "agent_id": self.agent.agent_id
                    },
                    timeout=5.0
                )
                response.raise_for_status()
                
        except Exception as e:
            self.logger.error(f"Failed to send event callback: {e}")
    
    async def start(self):
        """Start the A2A server"""
        import uvicorn
        
        self.logger.info(f"Starting A2A server on {self.host}:{self.port}")
        
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    def run(self):
        """Run the A2A server (blocking)"""
        import uvicorn
        
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port
        )


class A2ARegistry:
    """
    Central registry for discovering agents in the network.
    Can be extended to use distributed discovery mechanisms.
    """
    
    def __init__(self):
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.logger = get_logger("A2ARegistry")
    
    def register_agent(self, agent_card: Dict[str, Any]):
        """Register an agent in the registry"""
        agent_id = agent_card["agent_id"]
        self.agents[agent_id] = {
            **agent_card,
            "last_seen": datetime.now().isoformat()
        }
        self.logger.info(f"Registered agent: {agent_id}")
    
    def unregister_agent(self, agent_id: str):
        """Remove an agent from registry"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self.logger.info(f"Unregistered agent: {agent_id}")
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent information"""
        return self.agents.get(agent_id)
    
    def list_agents(self, capability: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all agents, optionally filtered by capability"""
        agents = list(self.agents.values())
        
        if capability:
            agents = [
                agent for agent in agents
                if capability in agent.get("capabilities", [])
            ]
        
        return agents
    
    def find_agents_by_name(self, name_pattern: str) -> List[Dict[str, Any]]:
        """Find agents by name pattern"""
        import re
        pattern = re.compile(name_pattern, re.IGNORECASE)
        
        return [
            agent for agent in self.agents.values()
            if pattern.search(agent.get("name", ""))
        ]