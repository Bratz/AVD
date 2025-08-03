"""
A2A Discovery Service for Agent Network
Provides agent discovery and registry functionality
"""
import asyncio
import json
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from src.ii_agent.utils.logging_config import get_logger
from a2a_integration.models import AgentCard, A2ACapability


class DiscoveryConfig(BaseModel):
    """Configuration for discovery service"""
    host: str = "0.0.0.0"
    port: int = 7000
    heartbeat_interval: int = 60  # seconds
    agent_timeout: int = 300  # seconds before marking agent as inactive
    cleanup_interval: int = 600  # seconds between cleanup runs


class A2ADiscoveryService:
    """
    Central discovery service for A2A agent network.
    Provides agent registration, discovery, and health monitoring.
    """
    
    def __init__(self, config: Optional[DiscoveryConfig] = None):
        self.config = config or DiscoveryConfig()
        self.logger = get_logger("A2ADiscovery")
        
        # Agent registry
        self.agents: Dict[str, AgentCard] = {}
        self.agent_health: Dict[str, datetime] = {}
        
        # Create FastAPI app
        self.app = FastAPI(
            title="A2A Discovery Service",
            description="Central discovery service for agent network",
            version="1.0.0"
        )
        
        # Setup routes
        self._setup_routes()
        
        # Background tasks
        self._cleanup_task = None
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "service": "A2A Discovery Service",
                "version": "1.0.0",
                "agents_registered": len(self.agents),
                "agents_active": len([a for a in self.agent_health.values() 
                                    if (datetime.now() - a).seconds < self.config.agent_timeout])
            }
        
        @self.app.post("/register")
        async def register_agent(agent_card: AgentCard, background_tasks: BackgroundTasks):
            """Register or update an agent"""
            try:
                agent_id = agent_card.agent_id
                
                # Update registry
                self.agents[agent_id] = agent_card
                self.agent_health[agent_id] = datetime.now()
                
                # Schedule health check
                background_tasks.add_task(self._monitor_agent_health, agent_id)
                
                self.logger.info(f"Registered agent: {agent_id} - {agent_card.name}")
                
                return {
                    "status": "registered",
                    "agent_id": agent_id,
                    "message": f"Agent {agent_card.name} registered successfully"
                }
                
            except Exception as e:
                self.logger.error(f"Registration failed: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.delete("/unregister/{agent_id}")
        async def unregister_agent(agent_id: str):
            """Unregister an agent"""
            if agent_id in self.agents:
                del self.agents[agent_id]
                if agent_id in self.agent_health:
                    del self.agent_health[agent_id]
                
                self.logger.info(f"Unregistered agent: {agent_id}")
                
                return {
                    "status": "unregistered",
                    "agent_id": agent_id
                }
            
            raise HTTPException(status_code=404, detail="Agent not found")
        
        @self.app.get("/agents")
        async def list_agents(
            capability: Optional[str] = None,
            tag: Optional[str] = None,
            active_only: bool = True
        ):
            """List all agents with optional filters"""
            agents = list(self.agents.values())
            
            # Filter by capability
            if capability:
                agents = [a for a in agents if capability in a.capabilities]
            
            # Filter by tag
            if tag:
                agents = [a for a in agents if tag in a.tags]
            
            # Filter by active status
            if active_only:
                active_cutoff = datetime.now() - timedelta(seconds=self.config.agent_timeout)
                agents = [
                    a for a in agents
                    if a.agent_id in self.agent_health and
                    self.agent_health[a.agent_id] > active_cutoff
                ]
            
            # Add health status to each agent
            result = []
            for agent in agents:
                agent_dict = agent.dict()
                if agent.agent_id in self.agent_health:
                    last_seen = self.agent_health[agent.agent_id]
                    agent_dict["last_seen"] = last_seen.isoformat()
                    agent_dict["is_active"] = (datetime.now() - last_seen).seconds < self.config.agent_timeout
                else:
                    agent_dict["is_active"] = False
                
                result.append(agent_dict)
            
            return {
                "agents": result,
                "total": len(result)
            }
        
        @self.app.get("/agents/{agent_id}")
        async def get_agent(agent_id: str):
            """Get specific agent details"""
            if agent_id not in self.agents:
                raise HTTPException(status_code=404, detail="Agent not found")
            
            agent = self.agents[agent_id]
            agent_dict = agent.dict()
            
            # Add health status
            if agent_id in self.agent_health:
                last_seen = self.agent_health[agent_id]
                agent_dict["last_seen"] = last_seen.isoformat()
                agent_dict["is_active"] = (datetime.now() - last_seen).seconds < self.config.agent_timeout
            else:
                agent_dict["is_active"] = False
            
            return agent_dict
        
        @self.app.post("/heartbeat/{agent_id}")
        async def heartbeat(agent_id: str):
            """Update agent heartbeat"""
            if agent_id not in self.agents:
                raise HTTPException(status_code=404, detail="Agent not found")
            
            self.agent_health[agent_id] = datetime.now()
            
            return {
                "status": "ok",
                "agent_id": agent_id,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/discover")
        async def discover_agents(query: Dict[str, Any]):
            """Advanced agent discovery with complex queries"""
            # This could be extended with more sophisticated query capabilities
            # For now, we'll implement basic filtering
            
            agents = list(self.agents.values())
            
            # Filter by multiple capabilities (AND operation)
            if "required_capabilities" in query:
                required = set(query["required_capabilities"])
                agents = [
                    a for a in agents
                    if required.issubset(set(a.capabilities))
                ]
            
            # Filter by any capability (OR operation)
            if "any_capability" in query:
                any_caps = set(query["any_capability"])
                agents = [
                    a for a in agents
                    if any_caps.intersection(set(a.capabilities))
                ]
            
            # Filter by performance requirements
            if "min_performance" in query:
                perf_reqs = query["min_performance"]
                agents = [
                    a for a in agents
                    if self._meets_performance_requirements(a, perf_reqs)
                ]
            
            return {
                "agents": [a.dict() for a in agents],
                "total": len(agents)
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "agents_registered": len(self.agents),
                "agents_active": len([
                    a for a in self.agent_health.values()
                    if (datetime.now() - a).seconds < self.config.agent_timeout
                ])
            }
    
    def _meets_performance_requirements(
        self,
        agent: AgentCard,
        requirements: Dict[str, Any]
    ) -> bool:
        """Check if agent meets performance requirements"""
        if not agent.performance:
            return False
        
        for key, required_value in requirements.items():
            agent_value = agent.performance.get(key)
            if agent_value is None:
                return False
            
            # Handle different comparison types
            if key.endswith("_ms") and agent_value > required_value:
                return False
            elif key == "availability" and agent_value < required_value:
                return False
            elif key == "throughput_rps" and agent_value < required_value:
                return False
        
        return True
    
    async def _monitor_agent_health(self, agent_id: str):
        """Monitor agent health via heartbeat"""
        while agent_id in self.agents:
            try:
                agent = self.agents[agent_id]
                
                # Check if agent has heartbeat endpoint
                if "heartbeat" in agent.endpoints:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            agent.endpoints["heartbeat"],
                            timeout=5.0
                        )
                        if response.status_code == 200:
                            self.agent_health[agent_id] = datetime.now()
                
                # Wait for next check
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                self.logger.warning(f"Health check failed for {agent_id}: {e}")
                await asyncio.sleep(self.config.heartbeat_interval)
    
    async def _cleanup_inactive_agents(self):
        """Periodically clean up inactive agents"""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                
                cutoff_time = datetime.now() - timedelta(seconds=self.config.agent_timeout * 2)
                inactive_agents = []
                
                for agent_id, last_seen in self.agent_health.items():
                    if last_seen < cutoff_time:
                        inactive_agents.append(agent_id)
                
                for agent_id in inactive_agents:
                    self.logger.info(f"Removing inactive agent: {agent_id}")
                    if agent_id in self.agents:
                        del self.agents[agent_id]
                    if agent_id in self.agent_health:
                        del self.agent_health[agent_id]
                
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
    
    async def start(self):
        """Start the discovery service"""
        import uvicorn
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_inactive_agents())
        
        # Start web server
        config = uvicorn.Config(
            app=self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    def run(self):
        """Run the discovery service (blocking)"""
        import uvicorn
        
        # Start cleanup task in background
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.create_task(self._cleanup_inactive_agents())
        
        # Run web server
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port
        )


class A2ADiscoveryClient:
    """
    Client for interacting with the discovery service.
    Used by agents to register themselves and discover other agents.
    """
    
    def __init__(self, discovery_url: str = "http://localhost:7000"):
        self.discovery_url = discovery_url.rstrip('/')
        self.logger = get_logger("A2ADiscoveryClient")
        self.client = httpx.AsyncClient(timeout=10.0)
    
    async def register(self, agent_card: AgentCard) -> Dict[str, Any]:
        """Register agent with discovery service"""
        try:
            response = await self.client.post(
                f"{self.discovery_url}/register",
                json=agent_card.dict()
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Registration failed: {e}")
            raise
    
    async def unregister(self, agent_id: str) -> Dict[str, Any]:
        """Unregister agent from discovery service"""
        try:
            response = await self.client.delete(
                f"{self.discovery_url}/unregister/{agent_id}"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Unregistration failed: {e}")
            raise
    
    async def list_agents(
        self,
        capability: Optional[str] = None,
        tag: Optional[str] = None,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """List agents with optional filters"""
        try:
            params = {"active_only": active_only}
            if capability:
                params["capability"] = capability
            if tag:
                params["tag"] = tag
            
            response = await self.client.get(
                f"{self.discovery_url}/agents",
                params=params
            )
            response.raise_for_status()
            return response.json()["agents"]
        except Exception as e:
            self.logger.error(f"Failed to list agents: {e}")
            return []
    
    async def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get specific agent details"""
        try:
            response = await self.client.get(
                f"{self.discovery_url}/agents/{agent_id}"
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        except Exception as e:
            self.logger.error(f"Failed to get agent: {e}")
            return None
    
    async def heartbeat(self, agent_id: str) -> bool:
        """Send heartbeat for agent"""
        try:
            response = await self.client.post(
                f"{self.discovery_url}/heartbeat/{agent_id}"
            )
            response.raise_for_status()
            return True
        except Exception as e:
            self.logger.error(f"Heartbeat failed: {e}")
            return False
    
    async def discover(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Discover agents with complex query"""
        try:
            response = await self.client.get(
                f"{self.discovery_url}/discover",
                params={"query": json.dumps(query)}
            )
            response.raise_for_status()
            return response.json()["agents"]
        except Exception as e:
            self.logger.error(f"Discovery failed: {e}")
            return []
    
    async def close(self):
        """Close the client"""
        await self.client.aclose()


# Standalone discovery service runner
if __name__ == "__main__":
    import sys
    
    config = DiscoveryConfig()
    if len(sys.argv) > 1:
        config.port = int(sys.argv[1])
    
    service = A2ADiscoveryService(config)
    print(f"Starting A2A Discovery Service on {config.host}:{config.port}")
    service.run()