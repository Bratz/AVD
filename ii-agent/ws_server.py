#!/usr/bin/env python3
"""
FastAPI WebSocket Server for the Agent with OAuth support.

This script provides a WebSocket interface for interacting with the Agent,
allowing real-time communication with a frontend application.
"""

from datetime import datetime
import io
import os
import argparse
import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

import jwt
import uvicorn
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    Request,
    HTTPException,
)

from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import anyio
import base64
from sqlalchemy import asc, text

from src.ii_agent.core.event import RealtimeEvent, EventType
from src.ii_agent.db.models import Event
from src.ii_agent.utils.constants import DEFAULT_MODEL, TOKEN_BUDGET, UPLOAD_FOLDER_NAME
from utils import parse_common_args, create_workspace_manager_for_connection
from src.ii_agent.agents.anthropic_fc import AnthropicFC
from src.ii_agent.agents.tcs_bancs_specialist_agent import TCSBancsSpecialistAgent
from wrappers.mcp_client_wrapper import MCPClientWrapper
from src.ii_agent.agents.base import BaseAgent
from src.ii_agent.llm.base import LLMClient
from src.ii_agent.utils import WorkspaceManager
from src.ii_agent.llm import get_client
from src.ii_agent.utils.prompt_generator import enhance_user_prompt

from fastapi.staticfiles import StaticFiles

from src.ii_agent.llm.context_manager.llm_summarizing import LLMSummarizingContextManager
from src.ii_agent.llm.token_counter import TokenCounter
from src.ii_agent.db.manager import DatabaseManager
from src.ii_agent.tools import get_system_tools
from src.ii_agent.prompts.system_prompt import SYSTEM_PROMPT, SYSTEM_PROMPT_WITH_SEQ_THINKING

# OAuth imports
try:
    from src.ii_agent.utils.oauth_utils import OAuthTokenManager
    OAUTH_AVAILABLE = True
except ImportError:
    OAUTH_AVAILABLE = False
    logging.warning("OAuth utilities not available. OAuth features will be disabled.")

MAX_OUTPUT_TOKENS_PER_TURN = 32000
MAX_TURNS = 200

# Create FastAPI app
app = FastAPI(title="Agent WebSocket API with OAuth Support")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create a logger
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
logger = logging.getLogger("websocket_server")
logger.setLevel(logging.DEBUG)

# Add file handler for persistent logging
handler = logging.FileHandler('websocket_server.log', encoding='utf-8')
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Active WebSocket connections
active_connections: Set[WebSocket] = set()

# Active agents for each connection
active_agents: Dict[WebSocket, BaseAgent] = {}

# Active agent tasks
active_tasks: Dict[WebSocket, asyncio.Task] = {}

# Store message processors for each connection
message_processors: Dict[WebSocket, asyncio.Task] = {}

# Store MCP wrappers for each connection
active_mcp_wrappers: Dict[WebSocket, MCPClientWrapper] = {}

# Connection to session mapping
connection_to_session: Dict[WebSocket, uuid.UUID] = {}
session_to_agent: Dict[uuid.UUID, BaseAgent] = {}

# Store global args for use in endpoint
global_args = None

# OAuth token manager (global)
oauth_token_manager = None

# Branding configuration
BRANDING_CONFIG = {
    "app_name": os.getenv("APP_NAME", "chatGBP"),
    "agent_name": os.getenv("AGENT_NAME", "II-Agent"),
    "connection_message": os.getenv("CONNECTION_MESSAGE", "Connected to Agent WebSocket Server"),
}

@app.on_event("startup")
async def startup():
    """Initialize OAuth token manager on startup"""
    global oauth_token_manager
    
    if OAUTH_AVAILABLE and os.getenv("ENABLE_OAUTH", "false").lower() == "true":
        oauth_token_manager = OAuthTokenManager.from_env()
        if oauth_token_manager:
            logger.info("OAuth token manager initialized for WebSocket server")
            try:
                # Try to get initial token
                token = await oauth_token_manager.get_valid_token()
                if token:
                    logger.info("OAuth token available and valid")
            except Exception as e:
                logger.warning(f"Failed to get initial OAuth token: {e}")
        else:
            logger.info("OAuth enabled but token manager not configured")
    else:
        logger.info("OAuth support disabled or not available")

async def send_event(websocket: WebSocket, event: RealtimeEvent):
    """Send a RealtimeEvent to the websocket."""
    try:
        message = {
            "type": event.type.value,  # Direct enum value
            "content": event.content or {}
        }
        await websocket.send_json(message)
        logger.debug(f"Sent event: type={event.type.value}, content_keys={list(event.content.keys()) if event.content else []}")
    except Exception as e:
        logger.error(f"Error sending event: {e}")


async def send_error_event(websocket: WebSocket, message: str, error_type: str = "error", suggestions: List[str] = None, **kwargs):
    """Send error event in expected format."""
    content = {
        "message": message,
        "error_type": error_type
    }
    
    if suggestions:
        content["suggestions"] = suggestions
        
    # Add any additional fields
    content.update(kwargs)
    
    event = RealtimeEvent(type=EventType.ERROR, content=content)
    await send_event(websocket, event)


def normalize_model_name(model_name: str) -> str:
    """Normalize frontend model names to backend names."""
    model_mapping = {
        "chutes-deepseek-ai/DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-R1-0528": "deepseek-ai/DeepSeek-R1",
        "claude-sonnet-4@20250514": "claude-3-5-sonnet-20241022",
        "claude-opus-4@20250514": "claude-3-opus-20240229",
        "claude-3-7-sonnet@20250219": "claude-3-sonnet-20240229",
        "gemini-2.5-pro-preview-05-06": "gemini-2.0-flash-exp",
        "gpt-4.1": "gpt-4-turbo-preview",
    }
    return model_mapping.get(model_name, model_name)


def map_model_name_to_client(model_name: str, ws_content: Dict[str, Any]) -> LLMClient:
    """Create an LLM client based on the model name and configuration.
    
    Args:
        model_name: The name of the model to use
        ws_content: Dictionary containing configuration options like thinking_tokens
        
    Returns:
        LLMClient: Configured LLM client instance
        
    Raises:
        ValueError: If the model name is not supported
    """
    # Normalize model name
    model_name = normalize_model_name(model_name)
    
    # Check for Chutes models first
    chutes_models = [
        "deepseek-ai/DeepSeek-V3-0324",
        "deepseek-ai/DeepSeek-V3-Chat",
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-R1-0528",
        "NVIDIA/Nemotron-4-340B-Chat",
        "Qwen/Qwen3-72B-Instruct",
        "Qwen/Qwen3-235B-A22B",
        "Qwen/Qwen2.5-VL-32B-Instruct",
        "chutesai/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "claude-opus-4-0",
        "google/gemini-2.5-pro-preview",
        "openai/gpt-4.1",
        "google/gemini-2.5-flash-preview-05-20:thinking",
    ]
    
    if model_name in chutes_models or "chutes" in ws_content.get("provider", "").lower():
        # Import Chutes client
        from src.ii_agent.llm.chutes_openai import ChutesOpenAIClient
        
        return ChutesOpenAIClient(
            model_name=model_name,
            max_retries=ws_content.get("max_retries", 3),
            use_caching=ws_content.get("use_caching", True),
            fallback_models=ws_content.get("fallback_models"),
            test_mode=ws_content.get("test_mode", False),
            no_fallback=ws_content.get("no_fallback", False),
            use_native_tool_calling=ws_content.get("use_native_tool_calling", False),
            thinking_tokens=ws_content.get('tool_args', {}).get('thinking_tokens'),
        )
    elif "claude" in model_name:
        return get_client(
            "anthropic-direct",
            model_name=model_name,
            use_caching=False,
            project_id=global_args.project_id,
            region=global_args.region,
            thinking_tokens=ws_content.get('tool_args', {}).get('thinking_tokens', False),
        )
    elif "gemini" in model_name:
        return get_client(
            "gemini-direct",
            model_name=model_name,
            project_id=global_args.project_id,
            region=global_args.region,
        )
    elif model_name in ["o3", "o4-mini", "gpt-4.1", "gpt-4o"]:
        return get_client(
            "openai-direct",
            model_name=model_name,
            azure_model=ws_content.get("azure_model", True),
            cot_model=ws_content.get("cot_model", False),
        )
    elif model_name in ["llama3.1:8b", "mistral:latest", "phi3:latest"] or any(m in model_name.lower() for m in ["llama", "mistral", "phi"]):
        return get_client(
            "ollama",
            model_name=model_name,
            base_url=ws_content.get("ollama_base_url", "http://localhost:11434"),
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


async def create_agent_for_connection(  # Make it async
    client: LLMClient,
    session_id: uuid.UUID,
    workspace_manager: WorkspaceManager,
    websocket: WebSocket,
    tool_args: Dict[str, Any],
    mcp_wrapper: Optional[MCPClientWrapper] = None,
):
    """Create a new agent instance for a websocket connection."""
    global global_args
    device_id = websocket.query_params.get("device_id")
    
    # Setup logging (same as before)
    logger_for_agent_logs = logging.getLogger(f"agent_logs_{id(websocket)}")
    logger_for_agent_logs.setLevel(logging.DEBUG)
    logger_for_agent_logs.propagate = False

    if not logger_for_agent_logs.handlers:
        logger_for_agent_logs.addHandler(logging.FileHandler(global_args.logs_path))
        if not global_args.minimize_stdout_logs:
            logger_for_agent_logs.addHandler(logging.StreamHandler())

    # Initialize database manager
    db_manager = DatabaseManager()

    # Create a new session
    db_manager.create_session(
        device_id=device_id,
        session_uuid=session_id,
        workspace_path=workspace_manager.root,
    )
    logger_for_agent_logs.info(
        f"Created new session {session_id} with workspace at {workspace_manager.root}"
    )

    # Initialize token counter
    token_counter = TokenCounter()

    # Choose context manager (same as before)
    if client.model_name and any(model in client.model_name.lower() for model in ["llama", "mistral", "phi"]):
        from src.ii_agent.llm.ollama import OllamaContextManager, OllamaAdaptiveContextManager
        
        context_strategy = tool_args.get("context_strategy", "adaptive")
        
        if context_strategy == "adaptive":
            context_manager = OllamaAdaptiveContextManager(
                token_counter=token_counter,
                logger=logger_for_agent_logs,
                token_budget=4096,
                model_name=client.model_name,
                client=client
            )
            logger_for_agent_logs.info(f"Using adaptive context strategy for Ollama model: {client.model_name}")
        else:
            context_manager = OllamaContextManager.create(
                token_counter=token_counter,
                logger=logger_for_agent_logs,
                token_budget=4096,
                strategy=context_strategy,
                client=client if context_strategy == "summarizing" else None
            )
            logger_for_agent_logs.info(f"Using {context_strategy} context strategy for Ollama")
    else:
        context_manager = LLMSummarizingContextManager(
            client=client,
            token_counter=token_counter,
            logger=logger_for_agent_logs,
            token_budget=TOKEN_BUDGET,
        )

    # Initialize queue
    queue = asyncio.Queue()
    
    # Prepare MCP tools if enabled (matching CLI)
    mcp_tools = []
    if mcp_wrapper and tool_args.get("enable_mcp", False):
        try:
            from src.ii_agent.tools.mcp_tool_adapter import create_mcp_tool_adapters
            
            if tool_args.get("banking_mode", False):
                # Use banking mode to get only the 3 core tools
                mcp_tools = create_mcp_tool_adapters(
                    mcp_wrapper=mcp_wrapper,
                    banking_mode=True
                )
                logger_for_agent_logs.info(f"Created {len(mcp_tools)} banking tool adapters")
                for tool in mcp_tools:
                    logger_for_agent_logs.info(f"  - Banking tool: {tool.name}")
            else:
                # Standard mode - get all tools
                mcp_tools = create_mcp_tool_adapters(
                    mcp_wrapper=mcp_wrapper,
                    include_deprecated=False,
                    categories=tool_args.get("mcp_categories"),
                    tags=tool_args.get("mcp_tags")
                )
                logger_for_agent_logs.info(f"Standard mode: Created {len(mcp_tools)} MCP tool adapters")
                
        except Exception as e:
            logger_for_agent_logs.error(f"Failed to create MCP tools: {e}")
            import traceback
            traceback.print_exc()
    
    # Before creating tools, ensure tool_args includes sequential_thinking for banking mode
    if tool_args.get("banking_mode", False):
        tool_args["sequential_thinking"] = True
        tool_args["memory_tool"] = "compactify"  # Add this line

    # Get system tools based on configuration
    if tool_args.get("mcp_tools_only", False) and mcp_tools:
        # Use only MCP tools plus essential communication tools
        from src.ii_agent.tools.message_tool import MessageTool
        from src.ii_agent.tools.complete_tool import ReturnControlToUserTool, CompleteTool
        
        tools = [
            MessageTool(),
            ReturnControlToUserTool() if websocket else CompleteTool(),
        ] + mcp_tools
        
        logger_for_agent_logs.info(f"MCP-only mode: {len(tools)} tools total")
    else:
        # Get standard system tools
        tools = get_system_tools(
            client=client,
            workspace_manager=workspace_manager,
            message_queue=queue,
            container_id=global_args.docker_container_id,
            ask_user_permission=global_args.needs_permission,
            tool_args=tool_args,
        )
        
        # Add MCP tools if available
        if mcp_tools:
            tools.extend(mcp_tools)
            logger_for_agent_logs.info(f"Mixed mode: {len(tools)} tools total ({len(mcp_tools)} MCP + {len(tools) - len(mcp_tools)} system)")

    # Determine system prompt
    system_prompt = SYSTEM_PROMPT_WITH_SEQ_THINKING if tool_args.get("sequential_thinking", False) else SYSTEM_PROMPT

    # Create appropriate agent based on mode - MATCHING CLI EXACTLY
    if tool_args.get("banking_mode", False) and mcp_wrapper:
        # Banking mode: Use TCSBancsSpecialistAgent (extends AnthropicFC)
        logger_for_agent_logs.info("Creating TCS BaNCS Specialist Agent...")
        
        agent = await TCSBancsSpecialistAgent.create(
            client=client,
            tools=tools,
            workspace_manager=workspace_manager,
            message_queue=queue,
            logger_for_agent_logs=logger_for_agent_logs,
            context_manager=context_manager,
            user_role=tool_args.get("user_role", "customer"),
            mcp_wrapper=mcp_wrapper,
            use_mcp_prompts=True,  # Enable MCP prompts
            max_output_tokens_per_turn=MAX_OUTPUT_TOKENS_PER_TURN,
            max_turns=MAX_TURNS,
            session_id=session_id,
            interactive_mode=True,
            websocket=websocket
        )
        
        # Log MCP prompt status
        if agent.is_mcp_enabled():
            available_prompts = await agent.get_available_prompts()
            logger_for_agent_logs.info(f"MCP prompts loaded: {available_prompts}")
        else:
            logger_for_agent_logs.info("Using local prompts (MCP prompts not available)")
            
        logger_for_agent_logs.info("TCS BaNCS Specialist Agent created")
        
    else:
        # Standard mode: Use AnthropicFC
        logger_for_agent_logs.info("Creating standard AnthropicFC agent...")
        
        agent = AnthropicFC(
            system_prompt=system_prompt,
            client=client,
            tools=tools,
            workspace_manager=workspace_manager,
            message_queue=queue,
            logger_for_agent_logs=logger_for_agent_logs,
            context_manager=context_manager,
            max_output_tokens_per_turn=MAX_OUTPUT_TOKENS_PER_TURN,
            max_turns=MAX_TURNS,
            websocket=websocket,
            session_id=session_id,
            interactive_mode=True
        )
        
        logger_for_agent_logs.info("AnthropicFC agent created")

    # Store references
    agent.session_id = session_id
    agent.db_manager = db_manager
    agent.mcp_wrapper = mcp_wrapper

    # Log final tool configuration
    if hasattr(agent, 'tool_manager') and agent.tool_manager:
        all_tools = agent.tool_manager.get_tools()
        logger_for_agent_logs.info(f"Agent initialized with {len(all_tools)} tools:")
        tool_types = {}
        for tool in all_tools:
            tool_type = type(tool).__name__
            tool_types[tool_type] = tool_types.get(tool_type, 0) + 1
        for tool_type, count in sorted(tool_types.items()):
            logger_for_agent_logs.info(f"  - {tool_type}: {count}")

    return agent
    
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Get auth token from query params
    # auth_token = websocket.query_params.get("auth_token")
    
    # if auth_token:
    #     try:
    #         # Verify JWT token
    #         payload = jwt.decode(auth_token, os.getenv("JWT_SECRET"), algorithms=["HS256"])
    #         user_email = payload.get("email")
    #         logger.info(f"Authenticated WebSocket connection for: {user_email}")
    #     except jwt.InvalidTokenError:
    #         logger.warning("Invalid auth token provided")
            
    active_connections.add(websocket)

    workspace_manager, session_uuid = create_workspace_manager_for_connection(
        global_args.workspace, global_args.use_container_workspace
    )

    # Store the session mapping
    connection_to_session[websocket] = session_uuid

    logger.info(f"WebSocket connected: {id(websocket)}, session: {session_uuid}")
    print(f"Workspace manager created: {workspace_manager}")

    try:    
        # Initial connection message - matching frontend expectations
        event = RealtimeEvent(
            type=EventType.CONNECTION_ESTABLISHED,
            content={
                "message": BRANDING_CONFIG["connection_message"],
                "workspace_path": str(workspace_manager.root),
                "session_id": str(session_uuid),
                "timestamp": datetime.now().isoformat(),
                "oauth_available": OAUTH_AVAILABLE,
                "oauth_enabled": os.getenv("ENABLE_OAUTH", "false").lower() == "true"
            }
        )
        
        await send_event(websocket, event)

        # Process messages from the client
        while True:
            # Receive and parse message
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                msg_type = message.get("type")
                content = message.get("content", {})
                
                logger.info(f"Received message type: {msg_type}")
                logger.debug(f"Message content: {content}")

                if msg_type == "init_agent":
                    model_name = content.get("model_name", DEFAULT_MODEL)
                    enable_mcp = content.get("enable_mcp", False)
                    mcp_config = content.get("mcp_config", {})
                    tool_args = content.get("tool_args", {})
                    
                    # Store MCP wrapper reference
                    mcp_wrapper = None
                    mcp_init_success = False
                    mcp_error = None
                    
                    # Check for Chutes-specific options
                    chutes_options = content.get("chutes_options", {})
                    if chutes_options:
                        # Merge Chutes options into content for client creation
                        content.update(chutes_options)
                        logger.info(f"Using Chutes-specific options: {chutes_options}")

                    # Initialize MCP if enabled
                    if enable_mcp:
                        try:
                            logger.info(f"Initializing MCP with config: {mcp_config}")
                            
                            # Get OAuth token from client or token manager
                            oauth_token = mcp_config.get("oauth_token") or content.get("oauth_token")
                            
                            # If no direct token but we have token manager, get fresh token
                            if not oauth_token and oauth_token_manager:
                                try:
                                    oauth_token = await oauth_token_manager.get_valid_token()
                                    logger.info("Using OAuth token from token manager")
                                except Exception as e:
                                    logger.warning(f"Failed to get token from manager: {e}")
                            
                            mcp_wrapper = MCPClientWrapper(
                                base_url=mcp_config.get("base_url", "http://localhost:8082"),
                                sse_endpoint=mcp_config.get("sse_endpoint", "http://localhost:8084/mcp"),
                                api_key=mcp_config.get("api_key", "test-api-key-123"),
                                oauth_token=oauth_token  # Pass OAuth token
                            )
                            
                            # Set token manager for auto-refresh if available
                            if oauth_token_manager:
                                mcp_wrapper.set_oauth_token_manager(oauth_token_manager)
                                logger.info("OAuth token manager set for MCP wrapper")
                            
                            # Initialize and discover tools
                            await mcp_wrapper.initialize()
                            mcp_init_success = True
                            
                            # Store MCP wrapper for this connection
                            active_mcp_wrappers[websocket] = mcp_wrapper
                            
                            # Get statistics
                            stats = mcp_wrapper.tool_registry.get_statistics()
                            
                            # Send MCP initialization success
                            event = RealtimeEvent(
                                type=EventType.SYSTEM,
                                content={
                                    "message": "MCP initialized successfully",
                                    "mcp_stats": stats,
                                    "total_tools": stats.get("total_tools", 0),
                                    "categories": stats.get("categories", {}),
                                    "oauth_enabled": bool(oauth_token)
                                }
                            )
                            await send_event(websocket, event)

                            logger.info(f"MCP initialized with {stats.get('total_tools', 0)} tools")
                            
                        except Exception as e:
                            mcp_error = str(e)
                            logger.error(f"Failed to initialize MCP: {e}")
                            import traceback
                            traceback.print_exc()
                            
                            # Check if it's an auth error
                            auth_error = "401" in mcp_error or "unauthorized" in mcp_error.lower()
                            
                            event = RealtimeEvent(
                                type=EventType.SYSTEM,
                                content={
                                    "message": f"MCP initialization failed: {mcp_error}",
                                    "error": mcp_error,
                                    "auth_error": auth_error,
                                    "continuing_without_mcp": True,
                                    "warning": True  # Flag to indicate warning
                                }
                            )
                            await send_event(websocket, event)

                            mcp_wrapper = None
                            enable_mcp = False
                    
                    # Initialize LLM client
                    try:
                        client = map_model_name_to_client(model_name, content)
                    except Exception as e:
                        await send_error_event(
                            websocket,
                            f"Failed to initialize LLM client: {str(e)}",
                            "llm_init_failed",
                            model=model_name
                        )
                        continue

                    # Update tool_args with MCP status
                    tool_args["enable_mcp"] = enable_mcp and mcp_init_success
                    
                    # Create agent
                    try:
                        agent = await create_agent_for_connection(
                            client=client,
                            session_id=session_uuid,
                            workspace_manager=workspace_manager,
                            websocket=websocket,
                            tool_args=tool_args,
                            mcp_wrapper=mcp_wrapper
                        )
                        
                        # Store agent
                        active_agents[websocket] = agent
                        session_to_agent[session_uuid] = agent
                        logger.info(f"Agent stored for websocket {id(websocket)}, total active agents: {len(active_agents)}")

                        # Start the message processor - CRITICAL
                        message_processor = asyncio.create_task(
                            process_agent_messages(agent, websocket)
                        )
                        message_processors[websocket] = message_processor
                        logger.info(f"Started message processor for session {session_uuid}")

                        # Prepare tool summary
                        tool_summary = {"total": 0, "by_type": {}, "mcp_tools": [], "system_tools": []}
                        
                        if hasattr(agent, 'tool_manager') and agent.tool_manager:
                            tools = agent.tool_manager.get_tools()
                            tool_summary["total"] = len(tools)
                            
                            for tool in tools:
                                tool_type = type(tool).__name__
                                tool_summary["by_type"][tool_type] = tool_summary["by_type"].get(tool_type, 0) + 1
                                
                                # Categorize tools
                                if hasattr(tool, 'mcp_wrapper') or 'MCP' in tool_type or tool.name in ['list_banking_apis', 'get_api_structure', 'invoke_banking_api']:
                                    tool_summary["mcp_tools"].append(tool.name)
                                else:
                                    tool_summary["system_tools"].append(tool.name)
                        
                        # Send initialization success
                        event = RealtimeEvent(
                            type=EventType.AGENT_INITIALIZED,
                            content={
                                "message": "Agent initialized successfully",
                                "model": model_name,
                                "mcp_enabled": enable_mcp and mcp_init_success,
                                "mcp_error": mcp_error,
                                "banking_mode": tool_args.get("banking_mode", False),
                                "tool_summary": tool_summary,
                                "session_id": str(session_uuid),
                                "workspace_path": str(workspace_manager.root),
                                "oauth_status": "enabled" if oauth_token else "disabled"
                            }
                        )
                        await send_event(websocket, event)
                        
                        logger.info(f"Agent initialized: model={model_name}, mcp={enable_mcp}, banking={tool_args.get('banking_mode', False)}, tools={tool_summary['total']}")
                        
                    except Exception as e:
                        logger.error(f"Failed to create agent: {e}")
                        import traceback
                        traceback.print_exc()
                        
                        await send_error_event(
                            websocket,
                            f"Failed to create agent: {str(e)}",
                            "agent_creation_failed"
                        )

                elif msg_type == "query":
                    # First check if agent exists
                    agent = active_agents.get(websocket)
                    if not agent:
                        await send_error_event(
                            websocket,
                            "No agent initialized. Please send 'init_agent' message first.",
                            "agent_not_initialized",
                            help="Send a message with type 'init_agent' and include model_name in content"
                        )
                        continue
                    
                    # Check if there's an active task for this connection
                    if websocket in active_tasks and not active_tasks[websocket].done():
                        await send_error_event(
                            websocket,
                            "A query is already being processed",
                            "query_in_progress"
                        )
                        continue

                    # Process a query to the agent
                    user_input = content.get("text", "")
                    resume = content.get("resume", False)
                    files = content.get("files", [])

                    # Send acknowledgment
                    event = RealtimeEvent(
                        type=EventType.PROCESSING,
                        content={"message": "Processing your request..."}
                    )
                    await send_event(websocket, event)

                    # Run the agent with the query in a separate task
                    task = asyncio.create_task(
                        run_agent_async(websocket, user_input, resume, files)
                    )
                    active_tasks[websocket] = task

                elif msg_type == "get_status":
                    # Get agent status
                    agent = active_agents.get(websocket)
                    if not agent:
                        await send_error_event(
                            websocket,
                            "No active agent for this connection",
                            "no_agent"
                        )
                        continue
                    
                    # Determine current status
                    status = "idle"  # Default status
                    
                    # Check if there's an active task
                    if websocket in active_tasks and not active_tasks[websocket].done():
                        status = "executing"
                    # Check if agent has completed
                    elif hasattr(agent, '_is_cancelled') and agent._is_cancelled:
                        status = "cancelled"
                    elif hasattr(agent, 'turn_count') and agent.turn_count >= agent.max_turns:
                        status = "completed"
                    
                    event = RealtimeEvent(
                        type=EventType.SYSTEM,
                        content={
                            "status": status,
                            "session_id": str(session_uuid),
                            "agent_initialized": True,
                            "status_update": True  # Flag to indicate this is a status update
                        }
                    )
                    await send_event(websocket, event)

                elif msg_type == "update_oauth_token":
                    # Handle OAuth token update
                    new_token = content.get("oauth_token")
                    if new_token and websocket in active_mcp_wrappers:
                        mcp_wrapper = active_mcp_wrappers[websocket]
                        mcp_wrapper.update_oauth_token(new_token)
                        logger.info("OAuth token updated for MCP wrapper")
                        
                        event = RealtimeEvent(
                            type=EventType.SYSTEM,
                            content={
                                "message": "OAuth token updated successfully",
                                "oauth_update": True
                            }
                        )
                        await send_event(websocket, event)
                    else:
                        await send_error_event(
                            websocket,
                            "No MCP wrapper to update token for",
                            "no_mcp_wrapper"
                        )

                elif msg_type == "get_thoughts":
                    # Get thought trail
                    agent = active_agents.get(websocket)
                    if not agent:
                        await send_error_event(
                            websocket,
                            "No active agent for this connection",
                            "no_agent"
                        )
                        continue
                    
                    # Extract thoughts from agent history or events
                    thoughts = []
                    
                    # Try to get from database events first
                    if hasattr(agent, 'db_manager') and agent.session_id:
                        try:
                            with agent.db_manager.get_session() as db_session:
                                events = (
                                    db_session.query(Event)
                                    .filter(Event.session_id == str(agent.session_id))
                                    .order_by(asc(Event.timestamp))
                                    .all()
                                )
                                
                                for event in events:
                                    event_data = json.loads(event.event_payload) if isinstance(event.event_payload, str) else event.event_payload
                                    
                                    # Extract thought-like events
                                    if event.event_type in ["agent_thinking", "tool_call", "tool_result"]:
                                        thought_type = "thought"
                                        if event.event_type == "tool_call":
                                            thought_type = "action"
                                        elif event.event_type == "tool_result":
                                            thought_type = "observation"
                                        
                                        content_text = ""
                                        if isinstance(event_data, dict) and "content" in event_data:
                                            content_data = event_data["content"]
                                            if isinstance(content_data, dict):
                                                content_text = content_data.get("text", str(content_data))
                                            else:
                                                content_text = str(content_data)
                                        else:
                                            content_text = str(event_data)
                                        
                                        thoughts.append({
                                            "id": str(event.id),
                                            "type": thought_type,
                                            "content": content_text,
                                            "timestamp": event.timestamp.isoformat(),
                                            "metadata": event_data.get("content", {}) if isinstance(event_data, dict) else {}
                                        })
                        except Exception as e:
                            logger.error(f"Error fetching thoughts from database: {e}")
                    
                    event = RealtimeEvent(
                        type=EventType.SYSTEM,
                        content={
                            "thoughts": thoughts,
                            "thought_trail": True  # Flag to indicate this is a thought trail
                        }
                    )
                    await send_event(websocket, event)

                elif msg_type == "check_agent":
                    # Debug: Check if agent exists
                    agent = active_agents.get(websocket)
                    session_id = connection_to_session.get(websocket)
                    
                    event = RealtimeEvent(
                        type=EventType.SYSTEM,
                        content={
                            "agent_exists": agent is not None,
                            "session_id": str(session_id) if session_id else None,
                            "active_agents_count": len(active_agents),
                            "message": "Agent is initialized" if agent else "No agent found"
                        }
                    )
                    await send_event(websocket, event)

                elif msg_type == "list_agent_tools":
                    # List all tools available to the agent
                    agent = active_agents.get(websocket)
                    if not agent:
                        await send_error_event(
                            websocket,
                            "No active agent for this connection",
                            "no_agent"
                        )
                        continue
                    
                    tool_list = []
                    if hasattr(agent, 'tool_manager') and agent.tool_manager:
                        tools = agent.tool_manager.get_tools()
                        for tool in tools:
                            tool_info = {
                                "name": tool.name,
                                "type": type(tool).__name__,
                                "description": tool.description[:100] + "..." if len(tool.description) > 100 else tool.description,
                                "is_mcp": hasattr(tool, 'mcp_wrapper') or 'MCP' in type(tool).__name__ or tool.name in ['list_banking_apis', 'get_api_structure', 'invoke_banking_api']
                            }
                            
                            # Add extra info for MCP tools
                            if tool_info["is_mcp"] and hasattr(tool, 'category'):
                                tool_info["category"] = getattr(tool, 'category', 'unknown')
                                tool_info["tags"] = getattr(tool, 'tags', [])
                            
                            tool_list.append(tool_info)
                    
                    event = RealtimeEvent(
                        type=EventType.SYSTEM,
                        content={
                            "tools": tool_list,
                            "total": len(tool_list),
                            "mcp_count": sum(1 for t in tool_list if t["is_mcp"]),
                            "system_count": sum(1 for t in tool_list if not t["is_mcp"]),
                            "tool_list": True  # Flag to indicate this is a tool list
                        }
                    )
                    await send_event(websocket, event)

                elif msg_type == "workspace_info":
                    # Send information about the current workspace
                    if workspace_manager:
                        event = RealtimeEvent(
                            type=EventType.WORKSPACE_INFO,
                            content={
                                "path": str(workspace_manager.root),
                                "session_id": str(session_uuid),
                            }
                        )
                        await send_event(websocket, event)
                    else:
                        await send_error_event(
                            websocket,
                            "Workspace not initialized",
                            "no_workspace"
                        )

                elif msg_type == "ping":
                    # Simple ping to keep connection alive
                    event = RealtimeEvent(type=EventType.PONG, content={})
                    await send_event(websocket, event)

                elif msg_type == "cancel":
                    # Get the agent for this connection
                    agent = active_agents.get(websocket)
                    if not agent:
                        await send_error_event(
                            websocket,
                            "No active agent for this connection",
                            "no_agent"
                        )
                        continue

                    agent.cancel()

                    # Send acknowledgment that cancellation was received
                    event = RealtimeEvent(
                        type=EventType.SYSTEM,
                        content={"message": "Query cancelled"}
                    )
                    await send_event(websocket, event)

                elif msg_type == "enhance_prompt":
                    # Process a request to enhance a prompt using an LLM
                    model_name = content.get("model_name", DEFAULT_MODEL)
                    user_input = content.get("text", "")
                    files = content.get("files", [])
                    
                    # Initialize LLM client
                    try:
                        client = map_model_name_to_client(model_name, content)
                    except Exception as e:
                        await send_error_event(
                            websocket,
                            f"Failed to initialize LLM client: {str(e)}",
                            "llm_init_failed"
                        )
                        continue
                    
                    # Call the enhance_prompt function from the module
                    success, message, enhanced_prompt = await enhance_user_prompt(
                        client=client,
                        user_input=user_input,
                        files=files,
                    )

                    if success and enhanced_prompt:
                        # Send the enhanced prompt back to the client
                        event = RealtimeEvent(
                            type=EventType.PROMPT_GENERATED,
                            content={
                                "result": enhanced_prompt,
                                "original_request": user_input,
                            }
                        )
                        await send_event(websocket, event)
                    else:
                        # Send error message
                        await send_error_event(
                            websocket,
                            message,
                            "prompt_enhancement_failed"
                        )

                # MCP-related message handlers
                elif msg_type == "get_mcp_prompt":
                    # Handle MCP prompt requests
                    mcp_wrapper = active_mcp_wrappers.get(websocket)
                    if not mcp_wrapper:
                        await send_error_event(
                            websocket,
                            "MCP not initialized for this connection",
                            "mcp_not_initialized"
                        )
                        continue
                    
                    prompt_name = content.get("prompt_name")
                    prompt_args = content.get("arguments", {})
                    
                    try:
                        prompt_result = await mcp_wrapper.get_prompt(prompt_name, prompt_args)
                        event = RealtimeEvent(
                            type=EventType.SYSTEM,
                            content={
                                "prompt_name": prompt_name,
                                "prompt_content": prompt_result,
                                "success": True,
                                "mcp_prompt": True  # Flag to indicate this is an MCP prompt
                            }
                        )
                        await send_event(websocket, event)
                    except Exception as e:
                        await send_error_event(
                            websocket,
                            f"Failed to get MCP prompt: {str(e)}",
                            "mcp_prompt_failed",
                            prompt_name=prompt_name
                        )

                elif msg_type == "list_mcp_prompts":
                    # List available MCP prompts
                    mcp_wrapper = active_mcp_wrappers.get(websocket)
                    if not mcp_wrapper:
                        await send_error_event(
                            websocket,
                            "MCP not initialized for this connection",
                            "mcp_not_initialized"
                        )
                        continue
                    
                    try:
                        prompts = await mcp_wrapper.list_prompts()
                        event = RealtimeEvent(
                            type=EventType.SYSTEM,
                            content={
                                "prompts": prompts,
                                "count": len(prompts),
                                "mcp_prompts_list": True  # Flag to indicate this is an MCP prompts list
                            }
                        )
                        await send_event(websocket, event)
                    except Exception as e:
                        await send_error_event(
                            websocket,
                            f"Failed to list MCP prompts: {str(e)}",
                            "mcp_list_failed"
                        )

                else:
                    # Unknown message type
                    await send_error_event(
                        websocket,
                        f"Unknown message type: {msg_type}",
                        "unknown_message_type"
                    )

            except json.JSONDecodeError as e:
                await send_error_event(
                    websocket,
                    f"Invalid JSON format: {str(e)}",
                    "json_decode_error"
                )
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}", exc_info=True)
                await send_error_event(
                    websocket,
                    f"Error processing request: {str(e)}",
                    "processing_error"
                )

    except WebSocketDisconnect:
        # Handle disconnection
        logger.info("Client disconnected")
        cleanup_connection(websocket)
    except Exception as e:
        # Handle other exceptions
        logger.error(f"WebSocket error: {str(e)}")
        cleanup_connection(websocket)


async def process_agent_messages(agent: BaseAgent, websocket: WebSocket):
    """Process messages from agent's queue with special handling for UI presentation."""
    try:
        while True:
            try:
                # Get event from agent's message queue
                event = await agent.message_queue.get()
                
                # Save to database if we have a session
                if agent.session_id is not None:
                    agent.db_manager.save_event(agent.session_id, event)
                
                # Handle special cases for UI presentation
                if event.type == EventType.TOOL_CALL:
                    tool_name = event.content.get("tool_name")
                    
                    # Convert sequential thinking to AGENT_THINKING
                    if tool_name == "sequential_thinking":
                        thought = event.content.get("tool_input", {}).get("thought", "")
                        if thought:
                            thinking_event = RealtimeEvent(
                                type=EventType.AGENT_THINKING,
                                content={"text": thought}
                            )
                            await send_event(websocket, thinking_event)
                        continue  # Don't send the original tool call
                    
                    # Skip UI-irrelevant tools
                    elif tool_name in ["return_control_to_user", "message_user"]:
                        # Get the message content
                        message = event.content.get("tool_input", {}).get("message", "") or event.content.get("tool_input", {}).get("text", "")
                        if message:
                            # Send as agent response
                            response_event = RealtimeEvent(
                                type=EventType.AGENT_RESPONSE,
                                content={"text": message}
                            )
                            await send_event(websocket, response_event)            
                    # For all other tools, send as normal
                    else:
                        await send_event(websocket, event)
                
                elif event.type == EventType.TOOL_RESULT:
                    tool_name = event.content.get("tool_name")
                    
                    # Skip results for thinking and control tools
                    if tool_name in ["sequential_thinking", "return_control_to_user", "message_user"]:
                        continue
                    
                    # Send other tool results
                    await send_event(websocket, event)
                
                else:
                    # Send all other events as-is
                    await send_event(websocket, event)
                    
            except asyncio.CancelledError:
                logger.info("Message processor cancelled")
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                
    except Exception as e:
        logger.error(f"Fatal error in message processor: {e}")


async def run_agent_async(
    websocket: WebSocket, user_input: str, resume: bool = False, files: List[str] = []
):
    """Run the agent asynchronously and send results back to the websocket."""

    logger.info(f"run_agent_async called for websocket {id(websocket)}")
    logger.info(f"Active agents: {len(active_agents)}, WebSocket IDs: {[id(ws) for ws in active_agents.keys()]}")

    # Try to get agent by websocket first
    agent = active_agents.get(websocket)

    # If not found, try by session
    if not agent and websocket in connection_to_session:
        session_id = connection_to_session[websocket]
        agent = session_to_agent.get(session_id)
        
        if agent:
            logger.info(f"Found agent by session ID: {session_id}")
            # Update the websocket mapping
            active_agents[websocket] = agent

    if not agent:
        logger.error(f"No agent found for websocket {id(websocket)}")
        await send_error_event(
            websocket,
            "Agent not initialized for this connection. Please initialize the agent first by sending an 'init_agent' message.",
            "agent_not_initialized",
            session_id=str(connection_to_session.get(websocket)) if websocket in connection_to_session else None
        )
        return

    try:
        # Run the agent with the query
        await anyio.to_thread.run_sync(
            agent.run_agent, user_input, files, resume, abandon_on_cancel=True
        )
        
        # Send completion event
        event = RealtimeEvent(
            type=EventType.AGENT_RESPONSE,
            content={
                "message": "Task completed",
                "status": "completed",
                "completed": True  # Flag to indicate completion
            }
        )
        await send_event(websocket, event)

    except RuntimeError as e:
        error_msg = str(e).lower()
        logger.error(f"Runtime error in agent: {str(e)}")
        
        # Handle auth errors
        if "401" in error_msg or "unauthorized" in error_msg:
            await send_error_event(
                websocket,
                "Authentication failed. OAuth token may be expired or invalid.",
                "auth_error",
                suggestions=[
                    "Refresh your OAuth token",
                    "Re-authenticate with the server",
                    "Check if OAuth is properly configured"
                ]
            )
        # Handle Ollama-specific errors
        elif "out of memory" in error_msg:
            await send_error_event(
                websocket,
                "Model ran out of memory. Try a smaller model or reduce context length.",
                "ollama_oom",
                suggestions=[
                    "Switch to a smaller model (e.g., from llama3.1:8b to mistral:latest)",
                    "Clear conversation history and start fresh",
                    "Reduce the complexity of your request"
                ]
            )
        elif "context length" in error_msg:
            await send_error_event(
                websocket,
                "Context length exceeded. The conversation is too long for this model.",
                "context_exceeded",
                suggestions=[
                    "Start a new conversation",
                    "Use a model with larger context window",
                    "Enable context summarization"
                ]
            )
        elif "cannot connect to ollama" in error_msg:
            await send_error_event(
                websocket,
                "Cannot connect to Ollama server. Please ensure Ollama is running.",
                "ollama_connection",
                suggestions=[
                    "Start Ollama with: ollama serve",
                    "Check if Ollama is running on the correct port",
                    "Verify the Ollama base URL configuration"
                ]
            )
        else:
            # Generic runtime error
            await send_error_event(
                websocket,
                f"Runtime error: {str(e)}",
                "runtime_error"
            )
            
    except ValueError as e:
        # Handle validation errors
        logger.error(f"Validation error in agent: {str(e)}")
        await send_error_event(
            websocket,
            f"Invalid input: {str(e)}",
            "validation_error"
        )
        
    except asyncio.CancelledError:
        # Handle cancellation
        logger.info("Agent task was cancelled")
        event = RealtimeEvent(
            type=EventType.SYSTEM,
            content={"message": "Task cancelled by user"}
        )
        await send_event(websocket, event)
        raise  # Re-raise to properly handle cancellation
        
    except Exception as e:
        logger.error(f"Unexpected error running agent: {str(e)}")
        import traceback
        traceback.print_exc()
        
        await send_error_event(
            websocket,
            f"Unexpected error: {str(e)}",
            "unexpected_error",
            details=traceback.format_exc() if getattr(global_args, 'debug', False) else None
        )
    finally:
        # Clean up the task reference
        if websocket in active_tasks:
            del active_tasks[websocket]


def cleanup_connection(websocket: WebSocket):
    """Clean up resources associated with a websocket connection."""
    
    # Cancel message processor first
    if websocket in message_processors:
        processor = message_processors[websocket]
        processor.cancel()
        
        # Wait briefly for cancellation
        try:
            asyncio.wait_for(processor, timeout=1.0)
        except:
            pass
            
        del message_processors[websocket]
    
    # Cleanup MCP wrapper
    if websocket in active_mcp_wrappers:
        del active_mcp_wrappers[websocket]
    
    # Then cleanup agent
    if websocket in active_agents:
        agent = active_agents[websocket]
        
        # Drain any remaining messages
        while not agent.message_queue.empty():
            try:
                agent.message_queue.get_nowait()
            except:
                break
                
        del active_agents[websocket]
    
    # Remove from active connections
    active_connections.discard(websocket)


def setup_workspace(app, workspace_path):
    try:
        app.mount(
            "/workspace",
            StaticFiles(directory=workspace_path, html=True),
            name="workspace",
        )
    except RuntimeError:
        # Directory might not exist yet
        os.makedirs(workspace_path, exist_ok=True)
        app.mount(
            "/workspace",
            StaticFiles(directory=workspace_path, html=True),
            name="workspace",
        )


def main():
    """Main entry point for the WebSocket server."""
    global global_args

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="WebSocket Server for interacting with the Agent"
    )
    parser = parse_common_args(parser)
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the server on",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9001,
        help="Port to run the server on",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    args = parser.parse_args()
    global_args = args

    setup_workspace(app, args.workspace)

    # Log OAuth status
    logger.info(f"OAuth support: {'Available' if OAUTH_AVAILABLE else 'Not available'}")
    if OAUTH_AVAILABLE:
        logger.info(f"OAuth enabled: {os.getenv('ENABLE_OAUTH', 'false')}")

    # Start the FastAPI server
    logger.info(f"Starting WebSocket server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


# HTTP endpoints remain the same...
# [Keep all the existing HTTP endpoints like upload, sessions, etc. as they are]

@app.post("/api/upload")
async def upload_file_endpoint(request: Request):
    """API endpoint for uploading a single file to the workspace."""
    try:
        data = await request.json()
        session_id = data.get("session_id")
        file_info = data.get("file")

        if not session_id:
            return JSONResponse(
                status_code=400, content={"error": "session_id is required"}
            )

        if not file_info:
            return JSONResponse(
                status_code=400, content={"error": "No file provided for upload"}
            )

        # Find the workspace path for this session
        workspace_path = Path(global_args.workspace).resolve() / session_id
        if not workspace_path.exists():
            return JSONResponse(
                status_code=404,
                content={"error": f"Workspace not found for session: {session_id}"},
            )

        # Create the upload directory if it doesn't exist
        upload_dir = workspace_path / UPLOAD_FOLDER_NAME
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = file_info.get("path", "")
        file_content = file_info.get("content", "")

        if not file_path:
            return JSONResponse(
                status_code=400, content={"error": "File path is required"}
            )

        # Ensure the file path is relative to the workspace
        if Path(file_path).is_absolute():
            file_path = Path(file_path).name

        # Create the full path within the upload directory
        original_path = upload_dir / file_path
        full_path = original_path

        # Handle filename collision by adding a suffix
        if full_path.exists():
            base_name = full_path.stem
            extension = full_path.suffix
            counter = 1

            # Keep incrementing counter until we find a unique filename
            while full_path.exists():
                new_filename = f"{base_name}_{counter}{extension}"
                full_path = upload_dir / new_filename
                counter += 1

            # Update the file_path to reflect the new name
            file_path = f"{full_path.relative_to(upload_dir)}"

        # Ensure any subdirectories exist
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if content is base64 encoded (for binary files)
        if file_content.startswith("data:"):
            # Handle data URLs (e.g., "data:application/pdf;base64,...")
            # Split the header from the base64 content
            header, encoded = file_content.split(",", 1)

            # Decode the content
            decoded = base64.b64decode(encoded)

            # Write binary content
            with open(full_path, "wb") as f:
                f.write(decoded)
        else:
            # Write text content
            with open(full_path, "w") as f:
                f.write(file_content)

        # Log the upload
        logger.info(f"File uploaded to {full_path}")

        # Return the path relative to the workspace for client use
        relative_path = f"/{UPLOAD_FOLDER_NAME}/{file_path}"

        return {
            "message": "File uploaded successfully",
            "file": {"path": relative_path, "saved_path": str(full_path)},
        }

    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return JSONResponse(
            status_code=500, content={"error": f"Error uploading file: {str(e)}"}
        )


@app.get("/api/sessions/{device_id}")
async def get_sessions_by_device_id(device_id: str):
    """Get all sessions for a specific device ID."""
    try:
        # Initialize database manager
        db_manager = DatabaseManager()

        # Get all sessions for this device, sorted by created_at descending
        with db_manager.get_session() as session:
            # Use raw SQL query to get sessions with their first user message
            query = text("""
            SELECT 
                session.id AS session_id,
                session.*, 
                event.id AS first_event_id,
                event.event_payload AS first_message,
                event.timestamp AS first_event_time
            FROM session
            LEFT JOIN event ON session.id = event.session_id
            WHERE event.id IN (
                SELECT e.id
                FROM event e
                WHERE e.event_type = "user_message" 
                AND e.timestamp = (
                    SELECT MIN(e2.timestamp)
                    FROM event e2
                    WHERE e2.session_id = e.session_id
                    AND e2.event_type = "user_message"
                )
            )
            AND session.device_id = :device_id
            ORDER BY session.created_at DESC
            """)

            # Execute the raw query with parameters
            result = session.execute(query, {"device_id": device_id})

            # Convert result to a list of dictionaries
            sessions = []
            for row in result:
                session_data = {
                    "id": row.id,
                    "workspace_dir": row.workspace_dir,
                    "created_at": row.created_at,
                    "device_id": row.device_id,
                    "first_message": json.loads(row.first_message)
                    .get("content", {})
                    .get("text", "")
                    if row.first_message
                    else "",
                }
                sessions.append(session_data)

            return {"sessions": sessions}

    except Exception as e:
        logger.error(f"Error retrieving sessions: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving sessions: {str(e)}"
        )


@app.get("/api/sessions/{session_id}/events")
async def get_session_events(session_id: str):
    """Get all events for a specific session ID."""
    try:
        # Initialize database manager
        db_manager = DatabaseManager()

        # Get all events for this session, sorted by timestamp ascending
        with db_manager.get_session() as session:
            events = (
                session.query(Event)
                .filter(Event.session_id == session_id)
                .order_by(asc(Event.timestamp))
                .all()
            )

            # Convert events to a list of dictionaries
            event_list = []
            for e in events:
                event_list.append(
                    {
                        "id": e.id,
                        "session_id": e.session_id,
                        "timestamp": e.timestamp.isoformat(),
                        "event_type": e.event_type,
                        "event_payload": e.event_payload,
                       "workspace_dir": e.session.workspace_dir,
                    }
                )

            return {"events": event_list}

    except Exception as e:
        logger.error(f"Error retrieving events: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving events: {str(e)}"
        )


@app.get("/api/mcp/status")
async def get_mcp_status():
    """Get the status of MCP integration."""
    try:
        # Check if any active agents have MCP enabled
        mcp_agents = []
        for ws, agent in active_agents.items():
            if hasattr(agent, 'tool_manager'):
                tools = agent.tool_manager.get_tools()
                mcp_tools = [t for t in tools if hasattr(t, 'mcp_wrapper') or 'MCP' in type(t).__name__ or t.name in ['list_banking_apis', 'get_api_structure', 'invoke_banking_api']]
                if mcp_tools:
                    mcp_agents.append({
                        "session_id": str(agent.session_id),
                        "mcp_tools_count": len(mcp_tools),
                        "mcp_tool_names": [t.name for t in mcp_tools]
                    })
        
        # Get active MCP wrapper count
        active_mcp_count = len(active_mcp_wrappers)
        
        return {
            "mcp_enabled_sessions": len(mcp_agents),
            "active_mcp_wrappers": active_mcp_count,
            "sessions": mcp_agents,
            "oauth_enabled": OAUTH_AVAILABLE and os.getenv("ENABLE_OAUTH", "false").lower() == "true",
            "oauth_token_manager": oauth_token_manager is not None
        }
        
    except Exception as e:
        logger.error(f"Error getting MCP status: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting MCP status: {str(e)}"
        )


@app.get("/api/oauth/status")
async def get_oauth_status():
    """Get OAuth status and configuration."""
    try:
        status = {
            "oauth_available": OAUTH_AVAILABLE,
            "oauth_enabled": os.getenv("ENABLE_OAUTH", "false").lower() == "true",
            "token_manager_configured": oauth_token_manager is not None,
            "has_valid_token": False,
            "token_expires_at": None
        }
        
        if oauth_token_manager:
            try:
                token = await oauth_token_manager.get_valid_token()
                if token:
                    status["has_valid_token"] = True
                    # Try to get expiry from token manager
                    if hasattr(oauth_token_manager, 'token_expires_at'):
                        status["token_expires_at"] = oauth_token_manager.token_expires_at.isoformat() if oauth_token_manager.token_expires_at else None
            except Exception as e:
                logger.warning(f"Failed to check token validity: {e}")
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting OAuth status: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting OAuth status: {str(e)}"
        )


if __name__ == "__main__":
    main()