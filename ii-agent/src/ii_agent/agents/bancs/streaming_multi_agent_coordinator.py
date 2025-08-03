# src/ii_agent/workflows/streaming_multi_agent_coordinator.py
"""
Streaming utilities for ROWBOAT workflow creation
Handles large workflow definitions without token limits
"""

import json
import asyncio
import logging
from typing import Dict, Any, List, AsyncGenerator, Optional
from datetime import datetime
import uuid
import os
import re
import aiohttp

from src.ii_agent.workflows.rowboat_types import OutputVisibility, ControlType, StreamEventType

logger = logging.getLogger(__name__)


class WorkflowStreamEvent:
    """Event types for workflow creation streaming"""
    AGENT_CREATED = "agent_created"
    TOOL_CREATED = "tool_created"
    PROMPT_CREATED = "prompt_created"
    EDGE_CREATED = "edge_created"
    WORKFLOW_COMPLETE = "workflow_complete"
    ERROR = "error"
    PROGRESS = "progress"


class StreamingWorkflowCreator:
    """Handles streaming creation of workflows from natural language descriptions"""
    
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.copilot = coordinator.copilot
        self.logger = logging.getLogger(__name__)
        
        # Get copilot URL from environment or use default
        self.copilot_url = os.environ.get('COPILOT_API_URL', 'http://localhost:3002')
    
    async def create_workflow_streaming(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream workflow creation events as they are generated
        Uses Rowboat copilot service for compatibility
        """
        workflow_id = str(uuid.uuid4())
        
        try:
            # Initialize workflow tracking
            workflow_data = {
                "agents": [],
                "tools": [],
                "prompts": [],
                "edges": [],
                "startAgent": None
            }
            
            # Stream progress event
            yield self._create_event(
                WorkflowStreamEvent.PROGRESS,
                {"message": "Analyzing workflow requirements...", "progress": 0}
            )
            
            # Use Rowboat copilot for streaming generation
            async for component in self._stream_copilot_components(description, context):
                event_type = None
                
                if component["type"] == "agent":
                    workflow_data["agents"].append(component["data"])
                    event_type = WorkflowStreamEvent.AGENT_CREATED
                    
                    # Set start agent if it's the first user-facing agent
                    if not workflow_data["startAgent"] and \
                       component["data"].get("outputVisibility") == "user_facing":
                        workflow_data["startAgent"] = component["data"]["name"]
                
                elif component["type"] == "tool":
                    workflow_data["tools"].append(component["data"])
                    event_type = WorkflowStreamEvent.TOOL_CREATED
                
                elif component["type"] == "prompt":
                    workflow_data["prompts"].append(component["data"])
                    event_type = WorkflowStreamEvent.PROMPT_CREATED
                
                # Yield the event
                if event_type:
                    yield self._create_event(event_type, component["data"])
            
            # Create edges from connected_agents
            yield self._create_event(
                WorkflowStreamEvent.PROGRESS,
                {"message": "Creating agent connections...", "progress": 80}
            )
            
            edges = self._create_edges_from_agents(workflow_data["agents"])
            for edge in edges:
                workflow_data["edges"].append(edge)
                yield self._create_event(WorkflowStreamEvent.EDGE_CREATED, edge)
            
            # Finalize workflow
            yield self._create_event(
                WorkflowStreamEvent.PROGRESS,
                {"message": "Finalizing workflow...", "progress": 90}
            )
            
            # Create the actual workflow in the coordinator
            workflow_def = self.coordinator._convert_to_workflow_definition(workflow_data)
            created_id = await self.coordinator.create_workflow(workflow_def)
            
            # Complete event
            yield self._create_event(
                WorkflowStreamEvent.WORKFLOW_COMPLETE,
                {
                    "workflow_id": created_id,
                    "agent_count": len(workflow_data["agents"]),
                    "tool_count": len(workflow_data["tools"]),
                    "edge_count": len(workflow_data["edges"])
                }
            )
            
        except Exception as e:
            logger.error(f"Error in streaming workflow creation: {e}")
            yield self._create_event(
                WorkflowStreamEvent.ERROR,
                {"error": str(e), "workflow_id": workflow_id}
            )
    
    async def _stream_copilot_components(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream components from Rowboat copilot service
        """
        # Prepare copilot request matching Rowboat format
        copilot_request = {
            "messages": [{"role": "user", "content": description}],
            "workflow_schema": self._get_workflow_schema(),
            "current_workflow_config": "{}",  # Empty for new workflow
            "context": context,
            "dataSources": []
        }
        
        # Stream from copilot
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.copilot_url}/chat_stream",
                json=copilot_request,
                headers={'Content-Type': 'application/json'}
            ) as response:
                buffer = ""
                async for chunk in response.content:
                    if chunk:
                        buffer += chunk.decode('utf-8')
                        
                        # Parse SSE events
                        lines = buffer.split('\n')
                        for i, line in enumerate(lines[:-1]):  # Process complete lines
                            if line.startswith('data: '):
                                try:
                                    data = json.loads(line[6:])
                                    content = data.get('content', '')
                                    
                                    # Extract copilot_change blocks
                                    for change in self._extract_copilot_changes(content):
                                        if change["action"] == "create_new":
                                            yield {
                                                "type": change["config_type"],
                                                "data": self._normalize_component(
                                                    change["config_type"],
                                                    change["config_changes"]
                                                )
                                            }
                                except:
                                    pass
                        
                        # Keep incomplete line in buffer
                        buffer = lines[-1]
    
    def _extract_copilot_changes(self, content: str) -> List[Dict[str, Any]]:
        """Extract copilot_change blocks from streamed content"""
        changes = []
        
        # Pattern matching Rowboat copilot format
        pattern = r'```copilot_change\n//\s*action:\s*(\w+)\n//\s*config_type:\s*(\w+)\n//\s*name:\s*(.+?)\n(.*?)\n```'
        
        matches = re.finditer(pattern, content, re.DOTALL)
        for match in matches:
            action = match.group(1)
            config_type = match.group(2)
            name = match.group(3)
            json_content = match.group(4)
            
            try:
                data = json.loads(json_content)
                changes.append({
                    "action": action,
                    "config_type": config_type,
                    "name": name,
                    **data
                })
            except:
                pass
        
        return changes
    
    def _normalize_component(self, component_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize component data to match ii-agent format"""
        if component_type == "agent":
            # Ensure enum values are strings
            if "outputVisibility" in data:
                if hasattr(data["outputVisibility"], "value"):
                    data["outputVisibility"] = data["outputVisibility"].value
            else:
                data["outputVisibility"] = "user_facing"
            
            if "controlType" in data:
                if hasattr(data["controlType"], "value"):
                    data["controlType"] = data["controlType"].value
            else:
                data["controlType"] = "retain"
            
            # Ensure lists exist
            data.setdefault("tools", [])
            data.setdefault("prompts", [])
            data.setdefault("examples", [])
            data.setdefault("connected_agents", [])
            data.setdefault("model", self.coordinator.default_model)
        
        return data
    
    def _create_edges_from_agents(self, agents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create edges based on connected_agents in each agent"""
        edges = []
        
        for agent in agents:
            for connected in agent.get("connected_agents", []):
                edges.append({
                    "from_agent": agent["name"],
                    "to_agent": connected
                })
        
        return edges
    
    def _create_event(self, event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a standardized stream event"""
        return {
            "type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
    
    def _get_workflow_schema(self) -> str:
        """Get the workflow schema matching Rowboat format"""
        return json.dumps({
            "type": "object",
            "properties": {
                "agents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "instructions": {"type": "string"},
                            "outputVisibility": {"type": "string", "enum": ["user_facing", "internal"]},
                            "controlType": {"type": "string", "enum": ["retain", "relinquish_to_parent", "start_agent"]},
                            "tools": {"type": "array", "items": {"type": "string"}},
                            "prompts": {"type": "array", "items": {"type": "string"}},
                            "examples": {"type": "array"},
                            "connected_agents": {"type": "array", "items": {"type": "string"}},
                            "model": {"type": "string"}
                        }
                    }
                },
                "tools": {"type": "array"},
                "prompts": {"type": "array"},
                "startAgent": {"type": "string"}
            }
        })