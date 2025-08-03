"""
II-Agent Banking System - Main Server with ROWBOAT Integration and OAuth Support
Combines HTTP API, WebSocket endpoints, and ROWBOAT multi-agent workflows
"""
import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import your components
from src.ii_agent.core.agent import AgentContext, ThoughtStep
from src.ii_agent.utils.logging_config import setup_logging, get_logger
from src.ii_agent.agents.tcs_bancs_specialist_agent import TCSBancsSpecialistAgent
from wrappers.mcp_client_wrapper import MCPClientWrapper
from wrappers.ollama_wrapper import OllamaWrapper, OllamaConfig
from src.ii_agent.core.event_stream import RealtimeEvent
from src.ii_agent.utils.workspace_manager import WorkspaceManager

# Import ROWBOAT integration
from api.rowboat_routes import integrate_rowboat_with_ii_agent
from src.ii_agent.config.rowboat_config import rowboat_config

from dotenv import load_dotenv

# Load environment variables before anything else
load_dotenv()

# Debug: Print OAuth-related environment variables
print(f"ENABLE_OAUTH from env: {os.getenv('ENABLE_OAUTH')}")
print(f"OAUTH_TOKEN exists: {bool(os.getenv('OAUTH_TOKEN'))}")
print(f"Current working directory: {os.getcwd()}")

# OAuth imports
try:
    from src.ii_agent.utils.oauth_utils import OAuthTokenManager
    OAUTH_AVAILABLE = True
except ImportError:
    OAUTH_AVAILABLE = False
    logging.warning("OAuth utilities not available. OAuth features will be disabled.")

# Setup logging
setup_logging(level=os.getenv("LOG_LEVEL", "DEBUG"))
logger = get_logger("main")

# Global state
active_sessions: Dict[str, Dict[str, Any]] = {}
websocket_connections: Dict[str, WebSocket] = {}
mcp_wrapper: Optional[MCPClientWrapper] = None
ollama_wrapper: Optional[OllamaWrapper] = None
workspace_manager: Optional[WorkspaceManager] = None
oauth_token_manager: Optional[OAuthTokenManager] = None

class AgentRequest(BaseModel):
    goal: str
    session_id: Optional[str] = None
    customer_id: Optional[str] = None
    agent_type: str = "tcs_bancs"
    oauth_token: Optional[str] = None  # Add OAuth token field

class AgentResponse(BaseModel):
    session_id: str
    status: str
    result: Dict[str, Any]
    thought_trail: List[Dict[str, Any]]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan management with OAuth support."""
    # Startup
    logger.info("🚀 Starting II-Agent Banking System with ROWBOAT Multi-Agent Support")
    
    global mcp_wrapper, ollama_wrapper, workspace_manager, oauth_token_manager
    
    try:
        # Initialize OAuth token manager if enabled
        oauth_token = None
        oauth_enabled = os.getenv("ENABLE_OAUTH", "false").lower() == "true"
        
        if OAUTH_AVAILABLE and oauth_enabled:
            oauth_token_manager = OAuthTokenManager.from_env()
            if oauth_token_manager:
                logger.info("✅ OAuth token manager initialized")
                try:
                    # Try to get initial token
                    token = await oauth_token_manager.get_valid_token()
                    if token:
                        logger.info("✅ OAuth token available and valid")
                        oauth_token = token
                except Exception as e:
                    logger.warning(f"⚠️  Failed to get initial OAuth token: {e}")
                    # Fall back to environment variable
                    oauth_token = os.getenv("OAUTH_TOKEN")
            else:
                logger.info("⚠️  OAuth enabled but token manager not configured")
                # Fall back to environment variable
                oauth_token = os.getenv("OAUTH_TOKEN")
        elif oauth_enabled:
            # OAuth enabled but utilities not available, use env token
            oauth_token = os.getenv("OAUTH_TOKEN")
            logger.info("ℹ️  Using OAuth token from environment variable")
        else:
            logger.info("ℹ️  OAuth support disabled")

        # Initialize services with OAuth support
        # Fix: Include all required parameters
        mcp_wrapper = MCPClientWrapper(
            base_url=os.getenv("MCP_BASE_URL", "http://localhost:8082"),
            sse_endpoint=os.getenv("MCP_SSE_ENDPOINT", os.getenv("MCP_SSE_URL", "http://localhost:8084/mcp")),
            api_key=os.getenv("MCP_API_KEY", "test-api-key-123")
            oauth_token=oauth_token  # Pass the OAuth token
        )
        
        # Set token manager for auto-refresh if available
        if oauth_token_manager:
            mcp_wrapper.set_oauth_token_manager(oauth_token_manager)
            
        await mcp_wrapper.initialize()
        logger.info("✅ MCP wrapper initialized" + (" with OAuth" if oauth_token else ""))
        
        # ollama_config = OllamaConfig()
        # ollama_wrapper = OllamaWrapper(ollama_config)
        # await ollama_wrapper.initialize()
        
        # Initialize workspace manager
        from pathlib import Path
        workspace_path = Path(os.getenv("WORKSPACE_PATH", "./workspace"))
        workspace_manager = WorkspaceManager(root=workspace_path)
        
        # Store in app state for ROWBOAT integration
        app.state.mcp_wrapper = mcp_wrapper
        app.state.ollama_wrapper = ollama_wrapper
        app.state.workspace_manager = workspace_manager
        app.state.oauth_token_manager = oauth_token_manager
        
        # ===== ADD THIS SECTION FOR ROWBOAT COORDINATOR =====
        # Initialize ROWBOAT coordinator
        try:
            from src.ii_agent.agents.bancs.multi_agent_coordinator import ROWBOATCoordinator
            from src.ii_agent.llm.model_registry import ChutesModelRegistry
            from src.ii_agent.llm.context_manager.llm_summarizing import LLMSummarizingContextManager
            from src.ii_agent.llm.token_counter import TokenCounter
            import asyncio
            
            # Create message queue for ROWBOAT
            message_queue = asyncio.Queue()
            
            # Create client for ROWBOAT using model registry
            model_id = rowboat_config.default_agent_model
            model_key = None

            # Check if this model is already registered
            for key, info in ChutesModelRegistry.AVAILABLE_MODELS.items():
                if info.model_id == model_id:  # Fixed: Direct attribute access
                    model_key = key
                    break

            # Create client using registry with automatic fallback generation
            if model_key:
                client = ChutesModelRegistry.create_llm_client(
                    model_key=model_key,
                    use_native_tools=True,
                    auto_fallbacks=True  # Let registry handle fallback generation
                )
                logger.info(f"Created client using model key: {model_key}")
            else:
                # Create client directly with model ID and auto-generated fallbacks
                client = ChutesModelRegistry.create_llm_client_by_model_id(
                    model_id=model_id,
                    use_native_tools=True,
                    auto_fallbacks=True  # Generate fallbacks based on model characteristics
                )
                logger.info(f"Created client directly with model ID: {model_id}")
            
            # Create context manager
            token_counter = TokenCounter()
            context_manager = LLMSummarizingContextManager(
                client=client,
                token_counter=token_counter,
                logger=logger,
                token_budget=8192  # Adjust based on your model
            )
            
            # Create ROWBOAT coordinator using async factory
            app.state.rowboat_coordinator = await ROWBOATCoordinator.create(
                client=client,
                tools=[],  # Add tools as needed
                workspace_manager=workspace_manager,
                message_queue=message_queue,
                logger_for_agent_logs=logger,
                context_manager=context_manager,
                config={
                    "mcp_wrapper": mcp_wrapper,
                    "ollama_wrapper": ollama_wrapper,
                    "user_role": "coordinator",
                    "oauth_enabled": oauth_enabled
                }
            )
            
            # Store dependencies in app state for other components
            app.state.message_queue = message_queue
            app.state.context_manager = context_manager
            
            # Log configuration summary
            logger.info("✅ ROWBOAT coordinator initialized")
            rowboat_config.log_config()  # Log full configuration

        except Exception as e:
            logger.error(f"❌ Failed to initialize ROWBOAT coordinator: {e}")
            import traceback
            traceback.print_exc()
            app.state.rowboat_coordinator = None
        # ===== END OF ROWBOAT SECTION =====            
        
        logger.info("✅ All services initialized")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down...")
    try:
        if mcp_wrapper:
            await mcp_wrapper.close()
        if ollama_wrapper:
            await ollama_wrapper.close()
        logger.info("✅ Shutdown complete")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

# Create FastAPI app
app = FastAPI(
    title="II-Agent Banking System with ROWBOAT",
    version="2.0.0",
    description="Intelligent Banking Agent with Multi-Agent Workflow Support and OAuth",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],  # Expose all headers
)

# Integrate ROWBOAT multi-agent routes
app = integrate_rowboat_with_ii_agent(app)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "II-Agent Banking System",
        "version": "2.0.0",
        "features": [
            "Single Agent Execution",
            "Multi-Agent Workflows (ROWBOAT)",
            "Natural Language Workflow Creation",
            "Real-time WebSocket Support",
            "MCP Tool Integration",
            "OAuth Authentication Support"
        ],
        "endpoints": {
            "health": "/health",
            "single_agent": "/agent/execute",
            "multi_agent": "/rowboat/workflows",
            "oauth_status": "/oauth/status",
            "docs": "/docs"
        },
        "oauth_enabled": os.getenv("ENABLE_OAUTH", "false").lower() == "true"
    }

@app.post("/oauth/refresh")
async def refresh_oauth_token():
    """Manually trigger OAuth token refresh"""
    if not oauth_token_manager:
        raise HTTPException(status_code=400, detail="OAuth not configured")
    
    try:
        token = await oauth_token_manager.get_valid_token()
        if token:
            return {
                "status": "success",
                "token_preview": token[:20] + "...",
                "expires_at": oauth_token_manager._token_expires.isoformat() if oauth_token_manager._token_expires else None
            }
        else:
            raise HTTPException(status_code=401, detail="Failed to refresh token")
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Enhanced health check including ROWBOAT and OAuth status."""
    rowboat_status = {
        "coordinator": hasattr(app.state, "rowboat_coordinator") and app.state.rowboat_coordinator is not None,
        "workflows": len(getattr(app.state, "workflows", {})) if hasattr(app.state, "workflows") else 0
    }
    
    oauth_status = {
        "available": OAUTH_AVAILABLE,
        "enabled": os.getenv("ENABLE_OAUTH", "false").lower() == "true",
        "token_manager": oauth_token_manager is not None,
        "has_valid_token": False,
        "mcp_has_oauth": False
    }
    
    if oauth_token_manager:
        try:
            token = await oauth_token_manager.get_valid_token()
            oauth_status["has_valid_token"] = bool(token)
        except:
            pass
    
    # Check if MCP wrapper has OAuth configured
    if mcp_wrapper and hasattr(mcp_wrapper, 'headers'):
        oauth_status["mcp_has_oauth"] = bool(mcp_wrapper.headers.get("Authorization"))
    
    return {
        "status": "healthy",
        "services": {
            "mcp": mcp_wrapper is not None,
            "ollama": ollama_wrapper is not None,
            "workspace": workspace_manager is not None,
            "rowboat": rowboat_status,
            "oauth": oauth_status
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/oauth/status")
async def get_oauth_status():
    """Get detailed OAuth status."""
    status = {
        "oauth_available": OAUTH_AVAILABLE,
        "oauth_enabled": os.getenv("ENABLE_OAUTH", "false").lower() == "true",
        "token_manager_configured": oauth_token_manager is not None,
        "has_valid_token": False,
        "token_expires_at": None,
        "keycloak_url": os.getenv("KEYCLOAK_URL"),
        "keycloak_realm": os.getenv("KEYCLOAK_REALM"),
        "client_id": os.getenv("OAUTH_CLIENT_ID"),
        "mcp_oauth_configured": False
    }
    
    if oauth_token_manager:
        try:
            token = await oauth_token_manager.get_valid_token()
            if token:
                status["has_valid_token"] = True
                if hasattr(oauth_token_manager, '_token_expires'):
                    status["token_expires_at"] = oauth_token_manager._token_expires.isoformat() if oauth_token_manager._token_expires else None
        except Exception as e:
            status["token_error"] = str(e)
            logger.warning(f"Failed to check token validity: {e}")
    
    # Check MCP wrapper OAuth status
    if mcp_wrapper and hasattr(mcp_wrapper, 'headers'):
        status["mcp_oauth_configured"] = bool(mcp_wrapper.headers.get("Authorization"))
    
    return status

@app.post("/agent/execute")
async def execute_agent(request: AgentRequest):
    """Execute a single agent (original functionality) with OAuth support."""
    session_id = request.session_id or str(uuid.uuid4())
    
    context = AgentContext(
        session_id=session_id,
        working_memory={"customer_id": request.customer_id} if request.customer_id else {}
    )
    
    # Use the shared message queue and context manager from app state
    message_queue = app.state.message_queue if hasattr(app.state, 'message_queue') else asyncio.Queue()
    context_manager = app.state.context_manager if hasattr(app.state, 'context_manager') else None
    
    # If context manager doesn't exist, create one
    if not context_manager:
        from src.ii_agent.llm.context_manager.llm_summarizing import LLMSummarizingContextManager
        from src.ii_agent.llm.token_counter import TokenCounter
        
        token_counter = TokenCounter()
        context_manager = LLMSummarizingContextManager(
            client=mcp_wrapper,
            token_counter=token_counter,
            logger=logger,
            token_budget=8192
        )
    
    # Handle OAuth token
    oauth_token = request.oauth_token
    if not oauth_token and oauth_token_manager:
        try:
            oauth_token = await oauth_token_manager.get_valid_token()
        except Exception as e:
            logger.warning(f"Failed to get OAuth token: {e}")
    
    # Update MCP wrapper with token if provided
    if oauth_token and mcp_wrapper:
        mcp_wrapper.update_oauth_token(oauth_token)
    
    # Create agent using async factory method
    agent = await TCSBancsSpecialistAgent.create(
        client=mcp_wrapper,  # or ollama_wrapper based on request
        tools=[],  # Add your tools here
        workspace_manager=workspace_manager,
        message_queue=message_queue,
        logger_for_agent_logs=logger,
        context_manager=context_manager,
        user_role=request.agent_type if hasattr(request, 'agent_type') else "customer",
        mcp_wrapper=mcp_wrapper,
        use_mcp_prompts=True,
        max_output_tokens_per_turn=8192,
        max_turns=10,
        session_id=session_id,
        interactive_mode=False
    )
    
    # Store session
    active_sessions[session_id] = {
        "agent": agent,
        "context": context,
        "status": "running",
        "type": "single_agent"
    }
    
    if session_id in websocket_connections:
        async def stream_events(event: RealtimeEvent):
            await notify_websocket(session_id, {
                "type": "agent_event",
                "event": event.dict()
            })
        
        agent.event_stream.subscribe(stream_events)

    # Execute
    try:
        result = await agent.run_agent(request.goal)
    except AttributeError:
        # If run_agent doesn't exist, try other common method names
        if hasattr(agent, 'execute'):
            result = await agent.execute(request.goal)
        elif hasattr(agent, 'process'):
            result = await agent.process(request.goal)
        else:
            # Fallback to synchronous method if async not available
            result = agent.run_agent(request.goal)
    except Exception as e:
        # Check for auth errors
        if "401" in str(e) or "unauthorized" in str(e).lower():
            raise HTTPException(
                status_code=401,
                detail="Authentication failed. OAuth token may be expired or invalid."
            )
        raise
    
    # Update session
    active_sessions[session_id]["status"] = "completed"
    active_sessions[session_id]["result"] = result
    
    # Notify WebSocket if connected
    if session_id in websocket_connections:
        await notify_websocket(session_id, {
            "type": "agent_completed",
            "result": result
        })
    
    return AgentResponse(
        session_id=session_id,
        status="completed",
        result=result,
        thought_trail=agent.get_thought_trail() if hasattr(agent, 'get_thought_trail') else []
    )

@app.websocket("/ws/agent/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time communication with OAuth support."""
    await websocket.accept()
    websocket_connections[session_id] = websocket
    
    logger.info(f"🔌 WebSocket connected: {session_id}")
    
    try:
        # Send connection confirmation with OAuth status
        oauth_status = {
            "enabled": os.getenv("ENABLE_OAUTH", "false").lower() == "true",
            "has_token": False
        }
        
        if oauth_token_manager:
            try:
                token = await oauth_token_manager.get_valid_token()
                oauth_status["has_token"] = bool(token)
            except:
                pass
        
        await websocket.send_text(json.dumps({
            "type": "connected",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "capabilities": ["single_agent", "multi_agent", "workflow_streaming"],
            "oauth_status": oauth_status
        }))
        
        # Handle incoming messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message = json.loads(data)
                
                await handle_websocket_message(session_id, message, websocket)
                
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_text(json.dumps({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                }))
                
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
    finally:
        websocket_connections.pop(session_id, None)

async def handle_websocket_message(session_id: str, message: Dict[str, Any], websocket: WebSocket):
    """Handle incoming WebSocket messages with OAuth support."""
    msg_type = message.get("type")
    
    if msg_type == "get_status":
        if session_id in active_sessions:
            session = active_sessions[session_id]
            await websocket.send_text(json.dumps({
                "type": "status_update",
                "session_id": session_id,
                "status": session["status"],
                "session_type": session.get("type", "single_agent")
            }))
    
    elif msg_type == "update_oauth_token":
        # Handle OAuth token update
        new_token = message.get("oauth_token")
        if new_token and mcp_wrapper:
            mcp_wrapper.update_oauth_token(new_token)
            logger.info("OAuth token updated via WebSocket")
            
            await websocket.send_text(json.dumps({
                "type": "oauth_updated",
                "success": True
            }))
        else:
            await websocket.send_text(json.dumps({
                "type": "oauth_updated",
                "success": False,
                "error": "Invalid token or MCP not initialized"
            }))
    
    elif msg_type == "get_thoughts":
        if session_id in active_sessions:
            agent = active_sessions[session_id].get("agent")
            if agent and hasattr(agent, "get_thought_trail"):
                await websocket.send_text(json.dumps({
                    "type": "thought_trail",
                    "thoughts": agent.get_thought_trail()
                }))
    
    elif msg_type == "get_workflows":
        # Return available workflows
        workflows = getattr(app.state, "workflows", {})
        await websocket.send_text(json.dumps({
            "type": "workflow_list",
            "workflows": list(workflows.keys()),
            "count": len(workflows)
        }))
    
    elif msg_type == "execute_workflow":
        # Execute a workflow via WebSocket
        workflow_id = message.get("workflow_id")
        input_data = message.get("input", {})
        oauth_token = message.get("oauth_token")
        
        # Update OAuth token if provided
        if oauth_token and mcp_wrapper:
            mcp_wrapper.update_oauth_token(oauth_token)
        
        if workflow_id and hasattr(app.state, "rowboat_coordinator"):
            coordinator = app.state.rowboat_coordinator
            
            # Store workflow session
            active_sessions[session_id] = {
                "workflow_id": workflow_id,
                "status": "running",
                "type": "multi_agent_workflow"
            }
            
            # Execute with streaming
            try:
                async for event in coordinator.execute_workflow_stream(workflow_id, input_data):
                    await websocket.send_text(json.dumps({
                        "type": "workflow_event",
                        "workflow_id": workflow_id,
                        "event": event
                    }))
                
                await websocket.send_text(json.dumps({
                    "type": "workflow_completed",
                    "workflow_id": workflow_id
                }))
                
                active_sessions[session_id]["status"] = "completed"
                
            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg or "unauthorized" in error_msg.lower():
                    await websocket.send_text(json.dumps({
                        "type": "workflow_error",
                        "workflow_id": workflow_id,
                        "error": "Authentication failed",
                        "auth_error": True
                    }))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "workflow_error",
                        "workflow_id": workflow_id,
                        "error": error_msg
                    }))
                active_sessions[session_id]["status"] = "failed"

async def notify_websocket(session_id: str, message: Dict[str, Any]):
    """Send message to WebSocket if connected."""
    if session_id in websocket_connections:
        try:
            websocket = websocket_connections[session_id]
            # Add event type for client-side handling
            message["event_stream"] = True
            message["timestamp"] = datetime.now().isoformat()
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.warning(f"Failed to send WebSocket message: {e}")

# Additional endpoints for hybrid operations
@app.post("/hybrid/execute")
async def execute_hybrid(request: Dict[str, Any]):
    """Execute either single agent or workflow based on request."""
    if "workflow_id" in request:
        # Execute workflow
        from api.rowboat_routes import execute_workflow, WorkflowExecuteRequest
        
        return await execute_workflow(
            workflow_id=request["workflow_id"],
            request=WorkflowExecuteRequest(**request.get("params", {})),
            background_tasks=None,
            coordinator=app.state.rowboat_coordinator,
            ii_context={"app": app}
        )
    else:
        # Execute single agent
        return await execute_agent(AgentRequest(**request))

def main():
    """Main entry point."""
    host = os.getenv("II_AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("II_AGENT_PORT", "9000"))
    
    logger.info(f"🚀 Starting II-Agent with ROWBOAT on {host}:{port}")
    logger.info(f"📚 API Documentation available at http://{host}:{port}/docs")
    logger.info(f"🔧 ROWBOAT endpoints available at http://{host}:{port}/rowboat/")
    logger.info(f"🔐 OAuth support: {'Enabled' if os.getenv("ENABLE_OAUTH", "false").lower() == "true" else 'Disabled'}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    main()