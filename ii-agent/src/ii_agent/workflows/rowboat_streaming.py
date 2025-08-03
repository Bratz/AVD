# ============================================================
# FILE 2: src/ii_agent/workflows/rowboat_streaming.py
# ============================================================
"""
ROWBOAT Streaming Infrastructure
Handles event streaming and formatting for real-time workflow execution
"""

import json
import asyncio
import os
import re
from typing import Dict, Any, Optional, AsyncGenerator, List
from datetime import datetime
from enum import Enum
import logging
import uuid

# Note: Import from rowboat_types in actual implementation
from src.ii_agent.workflows.rowboat_types import StreamEvent, StreamEventType, OutputVisibility

logger = logging.getLogger(__name__)

class StreamFormat(Enum):
    """Supported streaming formats"""
    SSE = "sse"  # Server-Sent Events
    WEBSOCKET = "websocket"
    JSON = "json"
    NDJSON = "ndjson"  # Newline-delimited JSON

class ROWBOATStreamProcessor:
    """Process and format streaming events for different protocols"""
    
    def __init__(self, format_type: str = "sse"):
        self.format_type = StreamFormat(format_type)
        self.event_counter = 0
        self.event_buffer: List[StreamEvent] = []
        self.metadata = {
            "session_start": datetime.utcnow(),
            "total_events": 0,
            "events_by_type": {}
        }
    
    def format_event(self, event: StreamEvent) -> str:
        """Format event based on configured format type"""
        
        self.event_counter += 1
        self.metadata["total_events"] += 1
        
        # Track event types
        event_type = event.type
        self.metadata["events_by_type"][event_type] = \
            self.metadata["events_by_type"].get(event_type, 0) + 1
        
        # Format based on type
        if self.format_type == StreamFormat.SSE:
            return self._format_sse(event)
        elif self.format_type == StreamFormat.WEBSOCKET:
            return self._format_websocket(event)
        elif self.format_type == StreamFormat.NDJSON:
            return self._format_ndjson(event)
        else:
            return self._format_json(event)
    
    def _format_sse(self, event: StreamEvent) -> str:
        """Format as Server-Sent Event"""
        
        # SSE format: event: <event_type>\ndata: <json_data>\n\n
        data = self._prepare_event_data(event)
        
        lines = [f"event: {event.type}"]
        
        # Add ID for reconnection support
        lines.append(f"id: {self.event_counter}")
        
        # Add retry hint for connection issues
        if event.type == StreamEventType.ERROR.value:
            lines.append("retry: 5000")  # 5 seconds
        
        # Add data
        lines.append(f"data: {json.dumps(data)}")
        
        return "\n".join(lines) + "\n\n"
    
    def _format_websocket(self, event: StreamEvent) -> str:
        """Format for WebSocket transmission"""
        
        data = self._prepare_event_data(event)
        
        message = {
            "type": "event",
            "event": event.type,
            "data": data,
            "metadata": {
                "id": self.event_counter,
                "timestamp": datetime.utcnow().isoformat(),
                "session_events": self.metadata["total_events"]
            }
        }
        
        return json.dumps(message)
    
    def _format_ndjson(self, event: StreamEvent) -> str:
        """Format as newline-delimited JSON"""
        
        data = self._prepare_event_data(event)
        data["_event_id"] = self.event_counter
        data["_event_type"] = event.type
        
        return json.dumps(data) + "\n"
    
    def _format_json(self, event: StreamEvent) -> str:
        """Format as plain JSON"""
        
        data = self._prepare_event_data(event)
        return json.dumps(data)
    
    def _prepare_event_data(self, event: StreamEvent) -> Dict[str, Any]:
        """Prepare event data for formatting"""
        
        base_data = {
            "type": event.type,
            "timestamp": event.timestamp.isoformat(),
            "data": event.data
        }
        
        # Add optional fields
        if event.agent_name:
            base_data["agent"] = event.agent_name
        
        if event.visibility:
            base_data["visibility"] = event.visibility.value
        
        if event.response_type:
            base_data["response_type"] = event.response_type.value
        
        # Add event-specific formatting
        if event.type == StreamEventType.MESSAGE.value:
            base_data["is_internal"] = (
                event.visibility == OutputVisibility.INTERNAL
                if event.visibility else False
            )
        
        return base_data
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of streaming session"""
        
        return {
            "session_duration": (
                datetime.utcnow() - self.metadata["session_start"]
            ).total_seconds(),
            "total_events": self.metadata["total_events"],
            "events_by_type": self.metadata["events_by_type"],
            "format": self.format_type.value
        }

class ROWBOATStreamHandler:
    """Handle streaming for ROWBOAT workflows with buffering and error handling"""
    
    def __init__(self, coordinator, buffer_size: int = 100):
        self.coordinator = coordinator
        self.buffer_size = buffer_size
        self.active_streams: Dict[str, Dict[str, Any]] = {}
    
    async def stream_workflow_execution(
        self,
        workflow_id: str,
        input_data: Dict[str, Any],
        format_type: str = "sse",
        filter_internal: bool = True
    ) -> AsyncGenerator[str, None]:
        """Stream workflow execution with formatted events and filtering"""
        
        # Create stream session
        stream_id = str(datetime.utcnow().timestamp())
        processor = ROWBOATStreamProcessor(format_type)
        
        self.active_streams[stream_id] = {
            "workflow_id": workflow_id,
            "processor": processor,
            "start_time": datetime.utcnow(),
            "event_count": 0
        }
        
        try:
            # Stream execution events
            async for event in self.coordinator.execute_workflow_with_streaming(
                workflow_id, input_data
            ):
                # Apply filtering
                if filter_internal and self._should_filter_event(event):
                    logger.debug(f"Filtered internal event: {event.type} from {event.agent_name}")
                    continue
                
                # Format and yield event
                formatted = processor.format_event(event)
                self.active_streams[stream_id]["event_count"] += 1
                
                yield formatted
                
                # Handle special events
                if event.type == StreamEventType.ERROR.value:
                    # Add error context
                    error_context = self._create_error_context(event, stream_id)
                    yield processor.format_event(StreamEvent(
                        type="error_context",
                        data=error_context
                    ))
                
                elif event.type == StreamEventType.TURN_END.value:
                    # Add session summary
                    summary_event = StreamEvent(
                        type="session_summary",
                        data=processor.get_session_summary()
                    )
                    yield processor.format_event(summary_event)
        
        except Exception as e:
            logger.error(f"Stream error for workflow {workflow_id}: {e}")
            
            # Send error event
            error_event = StreamEvent(
                type=StreamEventType.ERROR.value,
                data={
                    "error": str(e),
                    "workflow_id": workflow_id,
                    "stream_id": stream_id
                }
            )
            yield processor.format_event(error_event)
            
            raise
        
        finally:
            # Cleanup stream session
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
    
    def _should_filter_event(self, event: StreamEvent) -> bool:
        """Determine if event should be filtered based on visibility"""
        
        # Filter internal messages unless explicitly configured
        if event.type == StreamEventType.MESSAGE.value:
            return event.visibility == OutputVisibility.INTERNAL
        
        # Filter internal control transitions
        if event.type == StreamEventType.CONTROL_TRANSITION.value:
            return event.data.get("internal_only", False)
        
        return False
    
    def _create_error_context(self, error_event: StreamEvent, stream_id: str) -> Dict[str, Any]:
        """Create detailed error context for debugging"""
        
        stream_info = self.active_streams.get(stream_id, {})
        
        return {
            "error_details": error_event.data,
            "stream_context": {
                "workflow_id": stream_info.get("workflow_id"),
                "events_processed": stream_info.get("event_count", 0),
                "stream_duration": (
                    datetime.utcnow() - stream_info.get("start_time", datetime.utcnow())
                ).total_seconds() if stream_info else 0
            },
            "recovery_suggestions": self._get_recovery_suggestions(error_event.data)
        }
    
    def _get_recovery_suggestions(self, error_data: Dict[str, Any]) -> List[str]:
        """Generate recovery suggestions based on error type"""
        
        error_msg = str(error_data.get("error", "")).lower()
        suggestions = []
        
        if "timeout" in error_msg:
            suggestions.append("Consider increasing workflow timeout")
            suggestions.append("Check if agents are stuck in loops")
        elif "limit" in error_msg:
            suggestions.append("Reduce concurrent workflow executions")
            suggestions.append("Check parent-child call limits")
        elif "not found" in error_msg:
            suggestions.append("Verify workflow ID is correct")
            suggestions.append("Ensure workflow was created successfully")
        else:
            suggestions.append("Check workflow configuration")
            suggestions.append("Review agent instructions for errors")
        
        return suggestions

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
        
        # Use the existing stream processor for consistent formatting
        self.stream_processor = ROWBOATStreamProcessor("sse")
    
    async def create_workflow_streaming(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None,
        format_type: str = "sse"
    ) -> AsyncGenerator[str, None]:
        """
        Stream workflow creation events as they are generated
        Uses Rowboat copilot service for compatibility
        
        Args:
            description: Natural language description of the workflow
            context: Optional context (examples, model preferences, etc.)
            format_type: Output format (sse, json, websocket, ndjson)
            
        Yields:
            Formatted stream events
        """
        workflow_id = str(uuid.uuid4())
        self.stream_processor = ROWBOATStreamProcessor(format_type)
        
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
            yield self.stream_processor.format_event(StreamEvent(
                type=WorkflowStreamEvent.PROGRESS,
                data={"message": "Analyzing workflow requirements...", "progress": 0}
            ))
            
            # Use Rowboat copilot for streaming generation
            component_count = 0
            async for component in self._stream_copilot_components(description, context):
                component_count += 1
                event_type = None
                
                if component["type"] == "agent":
                    workflow_data["agents"].append(component["data"])
                    event_type = WorkflowStreamEvent.AGENT_CREATED
                    
                    # Set start agent if it's the first user-facing agent
                    if not workflow_data["startAgent"] and \
                       component["data"].get("outputVisibility") == "user_facing":
                        workflow_data["startAgent"] = component["data"]["name"]
                    
                    # Progress update
                    progress = min(10 + (component_count * 10), 70)
                    yield self.stream_processor.format_event(StreamEvent(
                        type=WorkflowStreamEvent.PROGRESS,
                        data={
                            "message": f"Created agent: {component['data']['name']}",
                            "progress": progress
                        }
                    ))
                
                elif component["type"] == "tool":
                    workflow_data["tools"].append(component["data"])
                    event_type = WorkflowStreamEvent.TOOL_CREATED
                
                elif component["type"] == "prompt":
                    workflow_data["prompts"].append(component["data"])
                    event_type = WorkflowStreamEvent.PROMPT_CREATED
                
                # Yield the component creation event
                if event_type:
                    yield self.stream_processor.format_event(StreamEvent(
                        type=event_type,
                        data=component["data"],
                        agent_name=component["data"].get("name") if component["type"] == "agent" else None
                    ))
            
            # Create edges from connected_agents
            yield self.stream_processor.format_event(StreamEvent(
                type=WorkflowStreamEvent.PROGRESS,
                data={"message": "Creating agent connections...", "progress": 80}
            ))
            
            edges = self._create_edges_from_agents(workflow_data["agents"])
            for edge in edges:
                workflow_data["edges"].append(edge)
                yield self.stream_processor.format_event(StreamEvent(
                    type=WorkflowStreamEvent.EDGE_CREATED,
                    data=edge
                ))
            
            # Finalize workflow
            yield self.stream_processor.format_event(StreamEvent(
                type=WorkflowStreamEvent.PROGRESS,
                data={"message": "Finalizing workflow...", "progress": 90}
            ))
            
            # Create the actual workflow in the coordinator
            workflow_def = self.coordinator._convert_to_workflow_definition(workflow_data)
            created_id = await self.coordinator.create_workflow(workflow_def)
            
            # Complete event with session summary
            yield self.stream_processor.format_event(StreamEvent(
                type=WorkflowStreamEvent.WORKFLOW_COMPLETE,
                data={
                    "workflow_id": created_id,
                    "agent_count": len(workflow_data["agents"]),
                    "tool_count": len(workflow_data["tools"]),
                    "edge_count": len(workflow_data["edges"]),
                    "session_summary": self.stream_processor.get_session_summary()
                }
            ))
            
        except Exception as e:
            logger.error(f"Error in streaming workflow creation: {e}")
            yield self.stream_processor.format_event(StreamEvent(
                type=WorkflowStreamEvent.ERROR,
                data={
                    "error": str(e),
                    "workflow_id": workflow_id,
                    "context": {
                        "description": description[:100] + "..." if len(description) > 100 else description,
                        "components_created": component_count
                    }
                }
            ))
    
    async def _stream_copilot_components(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream components from Rowboat copilot service
        """
        import aiohttp
        import re
        import uuid
        
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