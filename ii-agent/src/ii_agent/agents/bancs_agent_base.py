"""
BaNCS-specific II-Agent Base Classes
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

# Import II-Agent base classes
try:
    from src.ii_agent.core.agent import BaseAgent as IIBaseAgent
except ImportError:
    from src.ii_agent.agents.base import BaseAgent as IIBaseAgent

from app.core.mcp_service import MCPService
from app.services.progress_service import ProgressService
from app.services.llm_service import LLMService

class BaNCSAgentState(Enum):
    """States for BaNCS banking agents."""
    INITIALIZING = "initializing"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    ERROR = "error"
    WAITING = "waiting"

@dataclass
class BaNCSContext:
    """Banking context for II-Agent operations."""
    customer_id: str
    session_id: str
    entity_code: str = "GPRDTTSTOU"
    user_id: str = "1"
    language_code: str = "1"
    account_reference: Optional[str] = None

class BaNCSAgentBase(IIBaseAgent):
    """Base class for all BaNCS banking agents using II-Agent framework."""
    
    def __init__(
        self,
        name: str,
        session_id: str,
        mcp_service: MCPService,
        progress_service: Optional[ProgressService] = None,
        llm_service: Optional[LLMService] = None,
        bancs_context: Optional[BaNCSContext] = None
    ):
        super().__init__(name=name)
        
        self.session_id = session_id
        self.mcp_service = mcp_service
        self.progress_service = progress_service
        self.llm_service = llm_service
        self.bancs_context = bancs_context or BaNCSContext(
            customer_id="default",
            session_id=session_id
        )
        
        self.state = BaNCSAgentState.INITIALIZING
        self.thoughts = []
        self.logger = logging.getLogger(f"bancs.agent.{name}")
        self.agent_id = str(uuid.uuid4())
    
    async def add_thought(self, thought_type: str, content: str, context: Optional[Dict[str, Any]] = None):
        """Add a thought to the agent's thought trail."""
        thought = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "agent_name": self.name,
            "type": thought_type,
            "content": content,
            "context": context or {}
        }
        
        self.thoughts.append(thought)
        
        if self.progress_service:
            await self.progress_service.send_update(
                self.session_id,
                {
                    "agent": self.name,
                    "state": self.state.value,
                    "thought": thought
                }
            )
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return agent capabilities."""
        pass
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task using II-Agent framework."""
        try:
            await self.add_thought("observation", f"Starting task: {task.get('description', 'Unknown')}")
            
            # Plan execution
            self.state = BaNCSAgentState.PLANNING
            plan = await self.plan(task.get("description", ""), task.get("parameters", {}))
            
            # Execute plan
            self.state = BaNCSAgentState.EXECUTING
            results = []
            
            for step in plan:
                await self.add_thought("action", f"Executing: {step.get('description', 'Unknown step')}")
                result = await self._execute_step(step)
                results.append(result)
            
            # Reflect on results
            self.state = BaNCSAgentState.REFLECTING
            reflection = await self.reflect(results)
            
            self.state = BaNCSAgentState.COMPLETED
            
            return {
                "task": task,
                "plan": plan,
                "results": results,
                "reflection": reflection,
                "thoughts": self.thoughts,
                "agent_id": self.agent_id
            }
            
        except Exception as e:
            self.state = BaNCSAgentState.ERROR
            await self.add_thought("observation", f"Task failed: {str(e)}")
            raise
    
    async def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step - to be implemented by subclasses."""
        return {"step": step, "result": "executed"}
