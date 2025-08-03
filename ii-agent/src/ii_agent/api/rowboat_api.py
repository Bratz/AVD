# src/ii_agent/api/rowboat_api.py
"""
ROWBOAT API Compatibility Layer
Provides local ROWBOAT-compatible APIs for the enhanced coordinator
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum
import os

logger = logging.getLogger(__name__)

# Try to import Quart, fall back to Flask if not available
try:
    from quart import Quart, request, jsonify, Response
    ASYNC_FRAMEWORK = "quart"
    logger.info("Using Quart for async API server")
except ImportError:
    logger.warning("Quart not available, falling back to Flask")
    from flask import Flask as Quart, request, jsonify, Response
    ASYNC_FRAMEWORK = "flask"

from functools import wraps

from src.ii_agent.agents.bancs.multi_agent_coordinator import ROWBOATCoordinator
from src.ii_agent.workflows.rowboat_types import StreamEvent, StreamEventType
from src.ii_agent.workflows.definitions import WorkflowDefinition

from src.ii_agent.workflows.rowboat_streaming import WorkflowStreamEvent
# ===== API Configuration =====

@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 4040
    api_key: Optional[str] = None
    enable_tracing: bool = False
    enable_cors: bool = True
    max_message_size: int = 10 * 1024 * 1024  # 10MB

# ===== API Application =====

class ROWBOATAPIServer:
    """ROWBOAT-compatible API server"""
    
    def __init__(self, coordinator: ROWBOATCoordinator, config: APIConfig):
        self.coordinator = coordinator
        self.config = config
        self.app = Quart(__name__)
        self._setup_routes()
        self._setup_middleware()
        
        # Track active sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    def _setup_middleware(self):
        """Setup middleware for CORS, auth, etc."""
        
        if ASYNC_FRAMEWORK == "quart":
            @self.app.before_request
            async def before_request():
                # CORS headers
                if self.config.enable_cors:
                    @self.app.after_request
                    async def after_request(response):
                        response.headers['Access-Control-Allow-Origin'] = '*'
                        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
                        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
                        return response
                
                # Size limit
                if request.content_length and request.content_length > self.config.max_message_size:
                    return jsonify({'error': 'Request too large'}), 413
        else:
            # Flask middleware
            @self.app.before_request
            def before_request():
                if request.content_length and request.content_length > self.config.max_message_size:
                    return jsonify({'error': 'Request too large'}), 413
            
            if self.config.enable_cors:
                @self.app.after_request
                def after_request(response):
                    response.headers['Access-Control-Allow-Origin'] = '*'
                    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
                    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
                    return response
    
    def _setup_routes(self):
        """Setup API routes matching local ROWBOAT"""
        
        # ===== Health Check =====
        @self.app.route("/health", methods=["GET"])
        def health():
            return jsonify({
                "status": "ok",
                "version": "2.0",
                "coordinator": "enhanced_rowboat",
                "framework": ASYNC_FRAMEWORK,
                "features": {
                    "visibility_control": True,
                    "streaming": True,
                    "parent_child_limits": True,
                    "natural_language": True
                }
            })
        
        # ===== Authentication Decorator =====
        def require_api_key(f):
            if ASYNC_FRAMEWORK == "quart":
                @wraps(f)
                async def decorated(*args, **kwargs):
                    if not self.config.api_key:
                        return await f(*args, **kwargs)
                    
                    auth_header = request.headers.get('Authorization')
                    if not auth_header or not auth_header.startswith('Bearer '):
                        return jsonify({'error': 'Missing or invalid authorization header'}), 401
                    
                    token = auth_header.split('Bearer ')[1]
                    if token != self.config.api_key:
                        return jsonify({'error': 'Invalid API key'}), 403
                    
                    return await f(*args, **kwargs)
            else:
                @wraps(f)
                def decorated(*args, **kwargs):
                    if not self.config.api_key:
                        return f(*args, **kwargs)
                    
                    auth_header = request.headers.get('Authorization')
                    if not auth_header or not auth_header.startswith('Bearer '):
                        return jsonify({'error': 'Missing or invalid authorization header'}), 401
                    
                    token = auth_header.split('Bearer ')[1]
                    if token != self.config.api_key:
                        return jsonify({'error': 'Invalid API key'}), 403
                    
                    return f(*args, **kwargs)
            return decorated
        
        # ===== Main Chat API (Local ROWBOAT Compatible) =====
        if ASYNC_FRAMEWORK == "quart":
            @self.app.route("/chat", methods=["POST"])
            @require_api_key
            async def chat():
                """Main chat endpoint - compatible with local ROWBOAT"""
                
                try:
                    request_data = await request.get_json()
                    logger.info(f"Chat request: {json.dumps(request_data, indent=2)}")
                    
                    # Extract components from request
                    messages = request_data.get("messages", [])
                    agents = request_data.get("agents", [])
                    tools = request_data.get("tools", [])
                    prompts = request_data.get("prompts", [])
                    start_agent = request_data.get("startAgent", "")
                    state = request_data.get("state", {})
                    
                    # Filter agent transfer messages (compatibility)
                    filtered_messages = self._filter_agent_transfer_messages(messages)
                    
                    # Create or get workflow
                    workflow_id = await self._get_or_create_workflow(
                        agents, tools, prompts, start_agent, request_data
                    )
                    
                    # Execute workflow
                    result = await self.coordinator.execute_workflow(
                        workflow_id,
                        {"message": self._extract_user_message(filtered_messages)},
                        stream_events=False
                    )
                    
                    # Format response for compatibility
                    response_messages = self._format_response_messages(result)
                    
                    return jsonify({
                        "messages": response_messages,
                        "state": result.get("result", {}).get("state", state)
                    })
                
                except Exception as e:
                    logger.error(f"Chat error: {str(e)}")
                    return jsonify({"error": str(e)}), 500
        else:
            @self.app.route("/chat", methods=["POST"])
            @require_api_key
            def chat():
                """Main chat endpoint - Flask sync version"""
                
                try:
                    request_data = request.get_json()
                    logger.info(f"Chat request: {json.dumps(request_data, indent=2)}")
                    
                    # Extract components from request
                    messages = request_data.get("messages", [])
                    agents = request_data.get("agents", [])
                    tools = request_data.get("tools", [])
                    prompts = request_data.get("prompts", [])
                    start_agent = request_data.get("startAgent", "")
                    state = request_data.get("state", {})
                    
                    # Filter agent transfer messages (compatibility)
                    filtered_messages = self._filter_agent_transfer_messages(messages)
                    
                    # Create workflow synchronously
                    workflow_def = self._create_workflow_definition_from_request(
                        agents, tools, prompts, start_agent
                    )
                    
                    # For Flask, we'll simulate a simpler response
                    response_messages = [{
                        "role": "assistant",
                        "content": "I'm running in Flask compatibility mode. Full async features require Quart.",
                        "sender": start_agent or "Assistant"
                    }]
                    
                    return jsonify({
                        "messages": response_messages,
                        "state": state
                    })
                
                except Exception as e:
                    logger.error(f"Chat error: {str(e)}")
                    return jsonify({"error": str(e)}), 500
        

        # ===== Streaming Workflow Creation Endpoint =====
        @self.app.route("/rowboat/workflows/from-description/stream", methods=["POST"])
        @require_api_key
        def create_workflow_stream():
            """Streaming endpoint for workflow creation from natural language"""
            
            if ASYNC_FRAMEWORK == "quart":
                request_data = asyncio.run_coroutine_threadsafe(
                    request.get_json(), 
                    asyncio.get_event_loop()
                ).result()
            else:
                request_data = request.get_json()
            
            if not request_data or "description" not in request_data:
                return jsonify({"error": "Description is required"}), 400
            
            description = request_data.get("description", "")
            context = request_data.get("context", {})
            
            def generate():
                """Generator for SSE streaming"""
                try:
                    if ASYNC_FRAMEWORK == "quart":
                        # For Quart, use async streaming
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        # Stream workflow creation
                        async def stream_creation():
                            async for event in self.coordinator.create_workflow_with_streaming(
                                description, context
                            ):
                                # Format event for SSE
                                event_type = event.get("type", "")
                                
                                # Map internal event types to client-friendly format
                                if event_type == WorkflowStreamEvent.AGENT_CREATED:
                                    yield self._format_sse({
                                        "type": "agent",
                                        "data": event["data"]
                                    }, "create")
                                
                                elif event_type == WorkflowStreamEvent.TOOL_CREATED:
                                    yield self._format_sse({
                                        "type": "tool",
                                        "data": event["data"]
                                    }, "create")
                                
                                elif event_type == WorkflowStreamEvent.PROMPT_CREATED:
                                    yield self._format_sse({
                                        "type": "prompt",
                                        "data": event["data"]
                                    }, "create")
                                
                                elif event_type == WorkflowStreamEvent.EDGE_CREATED:
                                    yield self._format_sse({
                                        "type": "edge",
                                        "data": event["data"]
                                    }, "create")
                                
                                elif event_type == WorkflowStreamEvent.PROGRESS:
                                    yield self._format_sse({
                                        "message": event["data"]["message"],
                                        "progress": event["data"]["progress"]
                                    }, "progress")
                                
                                elif event_type == WorkflowStreamEvent.WORKFLOW_COMPLETE:
                                    yield self._format_sse({
                                        "workflow_id": event["data"]["workflow_id"],
                                        "summary": {
                                            "agents": event["data"]["agent_count"],
                                            "tools": event["data"]["tool_count"],
                                            "edges": event["data"]["edge_count"]
                                        }
                                    }, "complete")
                                
                                elif event_type == WorkflowStreamEvent.ERROR:
                                    yield self._format_sse({
                                        "error": event["data"]["error"],
                                        "workflow_id": event["data"].get("workflow_id")
                                    }, "error")
                        
                        # Run the async generator
                        gen = stream_creation()
                        try:
                            while True:
                                event_data = loop.run_until_complete(gen.__anext__())
                                yield event_data
                        except StopAsyncIteration:
                            pass
                        finally:
                            loop.close()
                    
                    else:
                        # Flask fallback - return a simple message
                        yield self._format_sse({
                            "message": "Streaming workflow creation requires Quart for async support",
                            "description": description
                        }, "info")
                        
                        # Create a simple workflow synchronously
                        workflow_def = self._create_workflow_definition_from_request(
                            agents=[{
                                "name": "SimpleAgent",
                                "instructions": "A simple agent created from: " + description,
                                "outputVisibility": "user_facing",
                                "controlType": "retain"
                            }],
                            tools=[],
                            prompts=[],
                            start_agent="SimpleAgent"
                        )
                        
                        yield self._format_sse({
                            "workflow_id": str(datetime.utcnow().timestamp()),
                            "summary": {"agents": 1, "tools": 0, "edges": 0}
                        }, "complete")
                
                except Exception as e:
                    logger.error(f"Streaming workflow creation error: {str(e)}")
                    yield self._format_sse({"error": str(e)}, "error")
            
            return Response(
                generate(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no',
                    'Connection': 'keep-alive'
                }
            )
        
        # ===== Get Workflow Status Endpoint =====
        @self.app.route("/rowboat/workflows/<workflow_id>/status", methods=["GET"])
        @require_api_key
        def get_workflow_status(workflow_id):
            """Get status of a workflow"""
            
            if workflow_id in self.coordinator.active_workflows:
                workflow_info = self.coordinator.active_workflows[workflow_id]
                
                return jsonify({
                    "workflow_id": workflow_id,
                    "status": workflow_info.get("status", "unknown"),
                    "created_at": workflow_info.get("created_at", datetime.utcnow()).isoformat(),
                    "agent_count": len(workflow_info.get("definition", {}).agents) if "definition" in workflow_info else 0,
                    "metadata": workflow_info.get("metadata", {})
                })
            else:
                return jsonify({"error": "Workflow not found"}), 404
        # ===== Streaming Chat API (Local ROWBOAT Compatible) =====
        @self.app.route("/chat_stream", methods=["POST"])
        @require_api_key
        def chat_stream():
            """Streaming chat endpoint - compatible with local ROWBOAT SSE format"""
            
            if ASYNC_FRAMEWORK == "quart":
                request_data = asyncio.run_coroutine_threadsafe(
                    request.get_json(), 
                    asyncio.get_event_loop()
                ).result()
            else:
                request_data = request.get_json()
            
            def generate():
                """Generator for SSE streaming"""
                try:
                    if ASYNC_FRAMEWORK == "quart":
                        # For Quart, we can use async properly
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        # Setup workflow
                        workflow_id = loop.run_until_complete(
                            self._get_or_create_workflow(
                                request_data.get("agents", []),
                                request_data.get("tools", []),
                                request_data.get("prompts", []),
                                request_data.get("startAgent", ""),
                                request_data
                            )
                        )
                        
                        # Filter messages
                        messages = self._filter_agent_transfer_messages(
                            request_data.get("messages", [])
                        )
                        
                        # Stream execution
                        async def stream_events():
                            async for event in self.coordinator.execute_workflow_with_streaming(
                                workflow_id,
                                {"message": self._extract_user_message(messages)}
                            ):
                                # Convert to local ROWBOAT format
                                if event.type == StreamEventType.MESSAGE.value:
                                    yield self._format_sse(event.data, "message")
                                
                                elif event.type == StreamEventType.TURN_END.value:
                                    done_data = {
                                        "state": {
                                            "last_agent_name": event.data.get("last_agent", ""),
                                            "tokens": event.data.get("tokens", {}),
                                            **request_data.get("state", {})
                                        }
                                    }
                                    yield self._format_sse(done_data, "done")
                                
                                elif event.type == StreamEventType.ERROR.value:
                                    yield self._format_sse(event.data, "error")
                        
                        # Run the async generator
                        gen = stream_events()
                        try:
                            while True:
                                event_data = loop.run_until_complete(gen.__anext__())
                                yield event_data
                        except StopAsyncIteration:
                            pass
                    else:
                        # Flask fallback - simple mock streaming
                        yield self._format_sse({
                            "content": "Streaming requires Quart for full functionality",
                            "role": "assistant"
                        }, "message")
                        
                        yield self._format_sse({
                            "state": request_data.get("state", {})
                        }, "done")
                
                except Exception as e:
                    logger.error(f"Streaming error: {str(e)}")
                    yield self._format_sse({"error": str(e)}, "error")
            
            return Response(generate(), mimetype='text/event-stream')
        
        # ===== Additional endpoints =====
        
        @self.app.route("/api/v1/create_workflow", methods=["POST"])
        @require_api_key
        def create_workflow_nl():
            """Create workflow from natural language description"""
            
            try:
                if ASYNC_FRAMEWORK == "quart":
                    request_data = asyncio.run_coroutine_threadsafe(
                        request.get_json(), 
                        asyncio.get_event_loop()
                    ).result()
                else:
                    request_data = request.get_json()
                
                description = request_data.get("description", "")
                
                if not description:
                    return jsonify({"error": "Description required"}), 400
                
                # For Flask, return a mock workflow
                if ASYNC_FRAMEWORK == "flask":
                    return jsonify({
                        "workflow_id": str(datetime.utcnow().timestamp()),
                        "message": "Natural language workflow creation requires Quart for async support"
                    })
                
                # Create workflow using coordinator
                loop = asyncio.new_event_loop()
                workflow_id = loop.run_until_complete(
                    self.coordinator.create_workflow_from_description(
                        description,
                        request_data.get("context", {})
                    )
                )
                
                return jsonify({
                    "workflow_id": workflow_id,
                    "status": "created"
                })
            
            except Exception as e:
                logger.error(f"Workflow creation error: {str(e)}")
                return jsonify({"error": str(e)}), 500
    
    # ===== Helper Methods =====
    
    def _filter_agent_transfer_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out agent transfer messages for compatibility"""
        
        def is_agent_transfer(msg):
            if (msg.get("role") == "assistant" and
                msg.get("content") is None and
                msg.get("tool_calls") is not None and
                len(msg.get("tool_calls")) > 0 and
                msg.get("tool_calls")[0].get("function", {}).get("name") == "transfer_to_agent"):
                return True
            
            if (msg.get("role") == "tool" and
                msg.get("tool_name") == "transfer_to_agent"):
                return True
            
            return False
        
        return [msg for msg in messages if not is_agent_transfer(msg)]
    
    def _extract_user_message(self, messages: List[Dict[str, Any]]) -> str:
        """Extract last user message from message list"""
        
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        
        return ""
    
    async def _get_or_create_workflow(
        self,
        agents: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        prompts: List[Dict[str, Any]],
        start_agent: str,
        request_data: Dict[str, Any]
    ) -> str:
        """Get existing or create new workflow from request data"""
        
        workflow_def = self._create_workflow_definition_from_request(
            agents, tools, prompts, start_agent
        )
        
        return await self.coordinator.create_workflow(workflow_def)
    
    # Update _create_workflow_definition_from_request in rowboat_api.py

    def _create_workflow_definition_from_request(
        self,
        agents: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        prompts: List[Dict[str, Any]],
        start_agent: str,
        edges: Optional[List[Dict[str, Any]]] = None  # Add edges parameter
    ) -> WorkflowDefinition:
        """Create workflow definition from request components"""
        
        from src.ii_agent.workflows.definitions import AgentConfig, AgentRole, WorkflowEdge, EdgeConditionType
        from src.ii_agent.workflows.rowboat_types import OutputVisibility, ControlType
        
        # Convert agents
        agent_configs = []
        
        for agent_data in agents:
            # Determine role
            role = AgentRole.CUSTOM
            role_str = agent_data.get("role", "").lower()
            if "research" in role_str:
                role = AgentRole.RESEARCHER
            elif "analyze" in role_str:
                role = AgentRole.ANALYZER
            elif "support" in role_str:
                role = AgentRole.CUSTOMER_SUPPORT
            
            # Handle outputVisibility and controlType
            output_vis = agent_data.get("outputVisibility", "user_facing")
            control = agent_data.get("controlType", "retain")
            
            # Create agent config
            agent_config = AgentConfig(
                name=agent_data.get("name", "Agent"),
                role=role,
                description=agent_data.get("description", ""),
                instructions=agent_data.get("instructions", ""),
                model=agent_data.get("model"),
                temperature=agent_data.get("temperature", 0.7),
                tools=agent_data.get("tools", []),
                output_visibility=output_vis,  # Pass the string value directly
                control_type=control,  # Pass the string value directly
                metadata={
                    "connected_agents": agent_data.get("connected_agents", []),
                    "examples": agent_data.get("examples", [])
                }
            )
            agent_configs.append(agent_config)
        
        # Convert edges - use provided edges, don't generate from connected_agents
        workflow_edges = []
        if edges:
            for edge_data in edges:
                # Get edge data with proper keys
                from_agent = edge_data.get("from_agent")
                to_agent = edge_data.get("to_agent")
                
                # Skip invalid edges
                if not from_agent or not to_agent:
                    logger.warning(f"Skipping invalid edge: {edge_data}")
                    continue
                
                # Create WorkflowEdge
                workflow_edges.append(WorkflowEdge(
                    from_agent=from_agent,
                    to_agent=to_agent,
                    condition_type=EdgeConditionType.ALWAYS
                ))
        
        return WorkflowDefinition(
            name="API Workflow",
            description="Workflow created from API request",
            agents=agent_configs,
            edges=workflow_edges,
            entry_point=start_agent or (agent_configs[0].name if agent_configs else "")
        )

    # Also update the calling code to pass edges
    async def _get_or_create_workflow(
        self,
        agents: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        prompts: List[Dict[str, Any]],
        start_agent: str,
        request_data: Dict[str, Any]
    ) -> str:
        """Get existing or create new workflow from request data"""
        
        # Pass edges from request_data
        edges = request_data.get("edges", [])
        
        workflow_def = self._create_workflow_definition_from_request(
            agents, tools, prompts, start_agent, edges
        )
        
        return await self.coordinator.create_workflow(workflow_def)
    
    def _format_response_messages(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format execution result as messages for API response"""
        
        messages = []
        
        if result.get("success"):
            workflow_result = result.get("result", {})
            
            # Extract messages from result
            if "messages" in workflow_result:
                messages = workflow_result["messages"]
            else:
                # Create a single response message
                messages = [{
                    "role": "assistant",
                    "content": "Workflow executed successfully"
                }]
        
        return messages
    
    def _format_sse(self, data: Dict[str, Any], event: Optional[str] = None) -> str:
        """Format data as Server-Sent Event"""
        
        msg = f"data: {json.dumps(data)}\n\n"
        if event is not None:
            msg = f"event: {event}\n{msg}"
        return msg
    
    def run(self):
        """Run the API server"""
        
        logger.info(f"Starting ROWBOAT API server on {self.config.host}:{self.config.port}")
        
        if ASYNC_FRAMEWORK == "quart":
            # Try to use hypercorn if available
            try:
                import hypercorn.asyncio
                from hypercorn.config import Config
                
                config = Config()
                config.bind = [f"{self.config.host}:{self.config.port}"]
                
                asyncio.run(hypercorn.asyncio.serve(self.app, config))
            except ImportError:
                logger.warning("Hypercorn not available, using Quart's built-in server")
                self.app.run(host=self.config.host, port=self.config.port)
        else:
            # Use Flask's built-in server
            logger.warning("Running in Flask compatibility mode - async features limited")
            self.app.run(host=self.config.host, port=self.config.port, debug=False)

# ===== Standalone Functions =====

async def create_api_server(
    coordinator: Optional[ROWBOATCoordinator] = None,
    workspace_manager=None,
    message_queue=None,
    context_manager=None,
    client=None
) -> ROWBOATAPIServer:
    """Create API server instance"""
    
    if not coordinator:
        # Create required dependencies if not provided
        if not workspace_manager:
            from src.ii_agent.utils.workspace_manager import WorkspaceManager
            workspace_manager = WorkspaceManager()
        if not message_queue:
            import asyncio
            message_queue = asyncio.Queue()
        if not context_manager:
            from src.ii_agent.llm.context_manager import ContextManager
            context_manager = ContextManager()
        if not client:
            from src.ii_agent.llm.model_registry import ChutesModelRegistry
            client = ChutesModelRegistry.create_llm_client(
                model_key="deepseek-v3",
                use_native_tools=True
            )
        
        # Create coordinator using async factory
        coordinator = await ROWBOATCoordinator.create(
            client=client,
            tools=[],
            workspace_manager=workspace_manager,
            message_queue=message_queue,
            logger_for_agent_logs=logging.getLogger(__name__),
            context_manager=context_manager
        )
    
    config = APIConfig(
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "4040")),
        api_key=os.getenv("API_KEY"),
        enable_tracing=os.getenv("ENABLE_TRACING", "false").lower() == "true"
    )
    
    return ROWBOATAPIServer(coordinator, config)

# Update main section
if __name__ == "__main__":
    async def main():
        server = await create_api_server()
        server.run()
    
    asyncio.run(main())
