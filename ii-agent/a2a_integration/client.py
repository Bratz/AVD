"""
A2A (Agent-to-Agent) Communication Client for II-Agent
Implements Google A2A protocol for inter-agent communication
"""
import json
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum
import httpx
from pydantic import BaseModel, Field

from src.ii_agent.utils.logging_config import get_logger


class A2AMessageType(Enum):
    """Types of A2A messages"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    STREAM = "stream"
    ERROR = "error"


class A2AMethod(Enum):
    """Standard A2A methods"""
    EXECUTE = "execute"
    QUERY = "query"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    DISCOVER = "discover"
    PING = "ping"


class A2AMessage(BaseModel):
    """A2A message format following JSON-RPC 2.0"""
    jsonrpc: str = "2.0"
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    method: str
    params: Dict[str, Any] = {}
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class A2ARequest(A2AMessage):
    """A2A request message"""
    pass


class A2AResponse(BaseModel):
    """A2A response message"""
    jsonrpc: str = "2.0"
    id: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class A2ANotification(BaseModel):
    """A2A notification (no response expected)"""
    jsonrpc: str = "2.0"
    method: str
    params: Dict[str, Any] = {}
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentCard(BaseModel):
    """Agent discovery card"""
    agent_id: str
    name: str
    description: str
    version: str = "1.0.0"
    capabilities: List[str] = []
    endpoints: Dict[str, str] = {}
    methods: List[str] = []
    metadata: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.now)
    last_seen: Optional[datetime] = None


class A2AClient:
    """
    A2A Client for outbound agent communication.
    Implements JSON-RPC 2.0 over HTTP(S) with agent discovery.
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        base_url: str,
        auth_token: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = get_logger(f"A2AClient.{agent_id}")
        
        # HTTP client with connection pooling
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers=self._get_default_headers()
        )
        
        # Agent registry cache
        self.agent_registry: Dict[str, AgentCard] = {}
        self.registry_last_update: Optional[datetime] = None
        
        # Active subscriptions
        self.subscriptions: Dict[str, asyncio.Task] = {}
        
        # Response handlers
        self.response_handlers: Dict[str, Callable] = {}
        
        self.logger.info(f"Initialized A2A client for agent {agent_name}")
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default HTTP headers"""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"II-Agent-A2A/{self.agent_id}",
            "X-Agent-ID": self.agent_id,
            "X-Agent-Name": self.agent_name
        }
        
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        return headers
    
    async def close(self):
        """Close the client and cleanup resources"""
        # Cancel all subscriptions
        for task in self.subscriptions.values():
            task.cancel()
        
        # Close HTTP client
        await self.client.aclose()
        
        self.logger.info("A2A client closed")
    
    async def discover_agents(self, force_refresh: bool = False) -> List[AgentCard]:
        """
        Discover available agents in the network.
        
        Args:
            force_refresh: Force refresh of agent registry
            
        Returns:
            List of discovered agent cards
        """
        # Check cache first
        if not force_refresh and self.registry_last_update:
            cache_age = (datetime.now() - self.registry_last_update).seconds
            if cache_age < 300:  # 5 minute cache
                return list(self.agent_registry.values())
        
        try:
            self.logger.info("Discovering agents...")
            
            # Broadcast discovery request
            discovery_url = f"{self.base_url}/a2a/discover"
            response = await self.client.get(discovery_url)
            response.raise_for_status()
            
            agents_data = response.json()
            
            # Update registry
            self.agent_registry.clear()
            for agent_data in agents_data.get("agents", []):
                agent_card = AgentCard(**agent_data)
                self.agent_registry[agent_card.agent_id] = agent_card
            
            self.registry_last_update = datetime.now()
            self.logger.info(f"Discovered {len(self.agent_registry)} agents")
            
            return list(self.agent_registry.values())
            
        except Exception as e:
            self.logger.error(f"Agent discovery failed: {e}")
            return list(self.agent_registry.values())  # Return cached data
    
    async def get_agent_card(self, agent_id: str) -> Optional[AgentCard]:
        """Get agent card by ID"""
        # Check cache
        if agent_id in self.agent_registry:
            return self.agent_registry[agent_id]
        
        # Try discovery
        await self.discover_agents()
        return self.agent_registry.get(agent_id)
    
    async def send_request(
        self,
        target_agent: str,
        method: str,
        params: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> A2AResponse:
        """
        Send a request to another agent and wait for response.
        
        Args:
            target_agent: Target agent ID
            method: Method to call
            params: Method parameters
            timeout: Request timeout (overrides default)
            
        Returns:
            A2A response
        """
        # Get target agent info
        agent_card = await self.get_agent_card(target_agent)
        if not agent_card:
            return A2AResponse(
                id="error",
                error={
                    "code": -32601,
                    "message": f"Agent {target_agent} not found"
                }
            )
        
        # Prepare request
        request = A2ARequest(
            method=method,
            params=params,
            metadata={
                "source_agent": self.agent_id,
                "target_agent": target_agent,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Determine endpoint
        endpoint = agent_card.endpoints.get("rpc", f"{self.base_url}/a2a/agents/{target_agent}/rpc")
        
        # Send request with retries
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Sending request to {target_agent}: {method}")
                
                response = await self.client.post(
                    endpoint,
                    json=request.dict(),
                    timeout=timeout or self.timeout
                )
                response.raise_for_status()
                
                # Parse response
                response_data = response.json()
                a2a_response = A2AResponse(**response_data)
                
                if a2a_response.error:
                    self.logger.warning(f"Agent {target_agent} returned error: {a2a_response.error}")
                
                return a2a_response
                
            except httpx.TimeoutException:
                self.logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    return A2AResponse(
                        id=request.id,
                        error={
                            "code": -32603,
                            "message": "Request timeout"
                        }
                    )
            except Exception as e:
                self.logger.error(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    return A2AResponse(
                        id=request.id,
                        error={
                            "code": -32603,
                            "message": str(e)
                        }
                    )
            
            # Exponential backoff
            await asyncio.sleep(2 ** attempt)
        
        return A2AResponse(
            id=request.id,
            error={
                "code": -32603,
                "message": "Max retries exceeded"
            }
        )
    
    async def send_notification(
        self,
        target_agent: str,
        method: str,
        params: Dict[str, Any]
    ) -> bool:
        """
        Send a notification (fire-and-forget) to another agent.
        
        Args:
            target_agent: Target agent ID
            method: Notification method
            params: Method parameters
            
        Returns:
            True if sent successfully
        """
        # Get target agent info
        agent_card = await self.get_agent_card(target_agent)
        if not agent_card:
            self.logger.warning(f"Agent {target_agent} not found for notification")
            return False
        
        # Prepare notification
        notification = A2ANotification(
            method=method,
            params=params,
            metadata={
                "source_agent": self.agent_id,
                "target_agent": target_agent,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Determine endpoint
        endpoint = agent_card.endpoints.get("notify", f"{self.base_url}/a2a/agents/{target_agent}/notify")
        
        try:
            self.logger.debug(f"Sending notification to {target_agent}: {method}")
            
            response = await self.client.post(
                endpoint,
                json=notification.dict(),
                timeout=5.0  # Short timeout for notifications
            )
            response.raise_for_status()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")
            return False
    
    async def execute_on_agent(
        self,
        target_agent: str,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a task on another agent.
        
        Args:
            target_agent: Target agent ID
            task: Task description
            context: Additional context
            
        Returns:
            Execution result
        """
        params = {
            "task": task,
            "context": context or {},
            "requesting_agent": self.agent_id
        }
        
        response = await self.send_request(
            target_agent,
            A2AMethod.EXECUTE.value,
            params
        )
        
        if response.error:
            return {
                "success": False,
                "error": response.error["message"],
                "agent": target_agent
            }
        
        return response.result or {"success": True, "agent": target_agent}
    
    async def query_agent(
        self,
        target_agent: str,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query information from another agent.
        
        Args:
            target_agent: Target agent ID
            query: Query type
            params: Query parameters
            
        Returns:
            Query result
        """
        query_params = {
            "query": query,
            "params": params or {},
            "requesting_agent": self.agent_id
        }
        
        response = await self.send_request(
            target_agent,
            A2AMethod.QUERY.value,
            query_params
        )
        
        if response.error:
            return {
                "success": False,
                "error": response.error["message"],
                "agent": target_agent
            }
        
        return response.result or {"success": True, "agent": target_agent}
    
    async def subscribe_to_agent(
        self,
        target_agent: str,
        event_type: str,
        handler: Callable[[Dict[str, Any]], None]
    ) -> str:
        """
        Subscribe to events from another agent.
        
        Args:
            target_agent: Target agent ID
            event_type: Event type to subscribe to
            handler: Callback for handling events
            
        Returns:
            Subscription ID
        """
        subscription_id = str(uuid.uuid4())
        
        params = {
            "event_type": event_type,
            "subscription_id": subscription_id,
            "callback_url": f"{self.base_url}/a2a/agents/{self.agent_id}/events"
        }
        
        response = await self.send_request(
            target_agent,
            A2AMethod.SUBSCRIBE.value,
            params
        )
        
        if response.error:
            raise Exception(f"Subscription failed: {response.error['message']}")
        
        # Store handler
        self.response_handlers[subscription_id] = handler
        
        # Start event listener task
        task = asyncio.create_task(
            self._listen_for_events(subscription_id, target_agent)
        )
        self.subscriptions[subscription_id] = task
        
        self.logger.info(f"Subscribed to {event_type} events from {target_agent}")
        return subscription_id
    
    async def unsubscribe(self, subscription_id: str):
        """Unsubscribe from events"""
        if subscription_id in self.subscriptions:
            # Cancel listener task
            self.subscriptions[subscription_id].cancel()
            del self.subscriptions[subscription_id]
            
            # Remove handler
            if subscription_id in self.response_handlers:
                del self.response_handlers[subscription_id]
            
            self.logger.info(f"Unsubscribed: {subscription_id}")
    
    async def _listen_for_events(self, subscription_id: str, target_agent: str):
        """Listen for events from subscription"""
        # This would typically use WebSocket or SSE
        # For now, we'll simulate with polling
        while subscription_id in self.subscriptions:
            try:
                # Poll for events
                await asyncio.sleep(5)
                
                # In real implementation, this would receive events
                # and call the handler
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Event listener error: {e}")
    
    async def ping_agent(self, target_agent: str) -> bool:
        """
        Ping another agent to check if it's alive.
        
        Args:
            target_agent: Target agent ID
            
        Returns:
            True if agent is alive
        """
        try:
            response = await self.send_request(
                target_agent,
                A2AMethod.PING.value,
                {},
                timeout=5.0
            )
            
            return not response.error
            
        except Exception:
            return False
    
    async def broadcast_to_agents(
        self,
        method: str,
        params: Dict[str, Any],
        agent_filter: Optional[Callable[[AgentCard], bool]] = None
    ) -> Dict[str, A2AResponse]:
        """
        Broadcast a request to multiple agents.
        
        Args:
            method: Method to call
            params: Method parameters
            agent_filter: Optional filter function for agents
            
        Returns:
            Dictionary of agent_id -> response
        """
        # Discover agents
        agents = await self.discover_agents()
        
        # Apply filter
        if agent_filter:
            agents = [a for a in agents if agent_filter(a)]
        
        # Send requests concurrently
        tasks = {
            agent.agent_id: self.send_request(agent.agent_id, method, params)
            for agent in agents
            if agent.agent_id != self.agent_id  # Don't send to self
        }
        
        responses = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        # Map responses
        result = {}
        for agent_id, response in zip(tasks.keys(), responses):
            if isinstance(response, Exception):
                result[agent_id] = A2AResponse(
                    id="error",
                    error={
                        "code": -32603,
                        "message": str(response)
                    }
                )
            else:
                result[agent_id] = response
        
        return result


class A2AConnectionPool:
    """
    Connection pool for managing multiple A2A clients.
    Useful when an agent needs to communicate with many other agents.
    """
    
    def __init__(self, agent_id: str, agent_name: str, base_url: str):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.base_url = base_url
        self.clients: Dict[str, A2AClient] = {}
        self.logger = get_logger(f"A2APool.{agent_id}")
    
    async def get_client(self, target_agent: str) -> A2AClient:
        """Get or create a client for target agent"""
        if target_agent not in self.clients:
            self.clients[target_agent] = A2AClient(
                agent_id=self.agent_id,
                agent_name=self.agent_name,
                base_url=self.base_url
            )
        
        return self.clients[target_agent]
    
    async def close_all(self):
        """Close all clients"""
        for client in self.clients.values():
            await client.close()
        self.clients.clear()