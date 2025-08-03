# src/ii_agent/api/rowboat_routes.py
"""
ROWBOAT API Routes for II-Agent Framework
Integrates multi-agent workflow capabilities into existing II-Agent API server
"""

from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import asyncio
import json
import logging
import uuid
import os
import aiohttp
from typing import AsyncGenerator
from fastapi.responses import StreamingResponse
import time

from src.ii_agent.utils.metrics_collector import MetricsCollector

# Import only what we need
from src.ii_agent.agents.bancs.multi_agent_coordinator import ROWBOATCoordinator, WorkflowStatus
from src.ii_agent.workflows.definitions import WorkflowDefinition, AgentConfig, AgentRole
from src.ii_agent.copilot.workflow_builder import WorkflowCopilot,WorkflowStreamEvent
from src.ii_agent.llm.chutes_openai import ChutesOpenAIClient
from src.ii_agent.config.rowboat_config import rowboat_config
from src.ii_agent.utils.logging_config import get_logger


logger = get_logger("rowboat_api")

# Pydantic Models
class WorkflowCreateRequest(BaseModel):
    name: str
    description: str
    agents: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = {}

class WorkflowFromDescriptionRequest(BaseModel):
    description: str
    examples: Optional[List[Dict[str, str]]] = []
    documents: Optional[List[str]] = []
    model_preferences: Optional[Dict[str, str]] = {}

class WorkflowExecuteRequest(BaseModel):
    input: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    context: Optional[Dict[str, Any]] = {}
    stream: bool = False
    mode: str = "execute"
    state: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    workflow_id: str
    state: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None

class WorkflowTestRequest(BaseModel):
    scenario_id: Optional[str] = None
    custom_input: Optional[str] = None
    mock_responses: Optional[Dict[str, Any]] = {}

class EnhanceWorkflowRequest(BaseModel):
    enhancement_type: str = "all"

# Create router
rowboat_router = APIRouter(prefix="/rowboat", tags=["rowboat"])

# Dependency for getting II-Agent components
def get_ii_agent_context(request: Request):
    """Get II-Agent context from request"""
    return {
        "app": request.app,
        "mcp_wrapper": getattr(request.app.state, "mcp_wrapper", None),
        "ollama_wrapper": getattr(request.app.state, "ollama_wrapper", None),
        "workspace_manager": getattr(request.app.state, "workspace_manager", None)
    }

# Initialize ROWBOAT coordinator
async def get_coordinator(ii_context: Dict = Depends(get_ii_agent_context)) -> ROWBOATCoordinator:
    """Get or create ROWBOAT coordinator"""
    
    app = ii_context["app"]
    
    # Return existing coordinator if available
    if hasattr(app.state, "rowboat_coordinator") and app.state.rowboat_coordinator:
        return app.state.rowboat_coordinator
    
    # If no coordinator exists, raise an error
    # The coordinator should be initialized in main.py during startup
    raise HTTPException(
        status_code=503,
        detail="ROWBOAT coordinator not initialized. Please check server startup logs."
    )

# Workflow Management Endpoints

@rowboat_router.get("/workflows")
async def list_workflows(
    category: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    ii_context: Dict = Depends(get_ii_agent_context),
    coordinator: ROWBOATCoordinator = Depends(get_coordinator)
):
    """List all ROWBOAT workflows"""
    
    try:
        # Get workflows from both app state and coordinator
        stored_workflows = ii_context["app"].state.workflows if hasattr(ii_context["app"].state, "workflows") else {}
        
        # Also check coordinator's active workflows
        if coordinator and hasattr(coordinator, "active_workflows"):
            for workflow_id, workflow_info in coordinator.active_workflows.items():
                if workflow_id not in stored_workflows:
                    # Convert to serializable format
                    workflow_def = workflow_info.get("definition", {})
                    
                    # Create serializable workflow data
                    workflow_data = {
                        "id": workflow_id,
                        "name": getattr(workflow_def, "name", "Unnamed Workflow"),
                        "description": getattr(workflow_def, "description", ""),
                        "version": getattr(workflow_def, "version", "1.0.0"),
                        "status": str(workflow_info.get("status", WorkflowStatus.INITIALIZING).value if hasattr(workflow_info.get("status", WorkflowStatus.INITIALIZING), "value") else workflow_info.get("status", "initializing")),
                        "created_at": workflow_info.get("created_at", datetime.utcnow()).isoformat() if hasattr(workflow_info.get("created_at", datetime.utcnow()), "isoformat") else str(workflow_info.get("created_at", datetime.utcnow())),
                        "updated_at": datetime.utcnow().isoformat(),
                        "agents": [],
                        "edges": [],
                        "metadata": workflow_info.get("rowboat_metadata", {})
                    }
                    
                    # Convert agents to serializable format
                    if hasattr(workflow_def, "agents"):
                        for agent in workflow_def.agents:
                            agent_data = {
                                "name": getattr(agent, "name", str(agent)),
                                "role": str(agent.role.value) if hasattr(agent, "role") and hasattr(agent.role, "value") else "custom",
                                "description": getattr(agent, "description", ""),
                                "instructions": getattr(agent, "instructions", ""),
                                "tools": getattr(agent, "tools", []),
                                "model": getattr(agent, "model", None)
                            }
                            workflow_data["agents"].append(agent_data)
                    
                    # Convert edges to serializable format
                    if hasattr(workflow_def, "edges"):
                        for edge in workflow_def.edges:
                            if hasattr(edge, "dict"):
                                workflow_data["edges"].append(edge.dict())
                            else:
                                edge_data = {
                                    "from_agent": getattr(edge, "from_agent", ""),
                                    "to_agent": getattr(edge, "to_agent", ""),
                                    "condition": getattr(edge, "condition", None)
                                }
                                workflow_data["edges"].append(edge_data)
                    
                    stored_workflows[workflow_id] = workflow_data
        
        # Filter by category if provided
        if category:
            workflows = {
                k: v for k, v in stored_workflows.items()
                if v.get("metadata", {}).get("category") == category
            }
        else:
            workflows = stored_workflows
        
        # Convert to list and apply pagination
        workflow_list = []
        for workflow in list(workflows.values())[offset:offset + limit]:
            # Ensure workflow is serializable - remove any non-serializable fields
            serializable_workflow = {
                "id": workflow.get("id", ""),
                "name": workflow.get("name", ""),
                "description": workflow.get("description", ""),
                "version": workflow.get("version", "1.0.0"),
                "status": workflow.get("status", "active"),
                "created_at": workflow.get("created_at", datetime.utcnow().isoformat()),
                "updated_at": workflow.get("updated_at", datetime.utcnow().isoformat()),
                "agents": workflow.get("agents", []),
                "edges": workflow.get("edges", []),
                "metadata": workflow.get("metadata", {})
            }
            # Remove langgraph field if present
            serializable_workflow.pop("langgraph", None)
            workflow_list.append(serializable_workflow)
        
        return {
            "workflows": workflow_list,
            "total": len(workflows),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error listing workflows: {e}", exc_info=True)
        # Return empty list on error
        return {
            "workflows": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
            "error": str(e)
        }
    
@rowboat_router.post("/workflows")
async def create_workflow(
    request: WorkflowCreateRequest,
    coordinator: ROWBOATCoordinator = Depends(get_coordinator),
    ii_context: Dict = Depends(get_ii_agent_context)
):
    """Create a new workflow from definition"""
    
    workflow_id = str(uuid.uuid4())
    
    # Store workflow
    workflow_data = {
        "id": workflow_id,
        "name": request.name,
        "description": request.description,
        "agents": request.agents,
        "edges": request.edges,
        "metadata": request.metadata,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }
    
    # Create LangGraph workflow using coordinator's workflow engine
    try:
        langgraph_workflow = coordinator.workflow_engine.create_workflow(workflow_data)
        workflow_data["langgraph"] = langgraph_workflow
        
        # Store in app state
        ii_context["app"].state.workflows[workflow_id] = workflow_data
        
        logger.info(f"Created workflow: {workflow_id} - {request.name}")
        
    except Exception as e:
        logger.error(f"Failed to create workflow: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    return {
        "workflow_id": workflow_id,
        "status": "created",
        "message": f"Workflow '{request.name}' created successfully"
    }

@rowboat_router.post("/workflows/from-description")
async def create_workflow_from_description(
    request: WorkflowFromDescriptionRequest,
    coordinator: ROWBOATCoordinator = Depends(get_coordinator),
    ii_context: Dict = Depends(get_ii_agent_context)
):
    """Create workflow from natural language description using copilot"""
    
    try:
        # Add II-Agent context to user context
        user_context = {
            "examples": request.examples,
            "documents": request.documents,
            "model_preferences": request.model_preferences,
            "mcp_available": ii_context.get("mcp_wrapper") is not None,
            "ollama_available": ii_context.get("ollama_wrapper") is not None
        }
        
        # Use coordinator to create workflow
        workflow_id = await coordinator.create_workflow_from_description(
            description=request.description,
            user_context=user_context
        )
        
        # Get the created workflow
        workflow_info = coordinator.active_workflows.get(workflow_id)
        
        if workflow_info:
            # Store in II-Agent workflow storage
            workflow_data = {
                "id": workflow_id,
                "name": workflow_info["definition"].name,
                "description": workflow_info["definition"].description,
                "agents": [agent.dict() for agent in workflow_info["definition"].agents],
                "edges": workflow_info["definition"].edges,
                "metadata": workflow_info["definition"].metadata,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "langgraph": workflow_info["langgraph"]
            }
            
            ii_context["app"].state.workflows[workflow_id] = workflow_data
        
        logger.info(f"Created workflow from description: {workflow_id}")
        
        return {
            "workflow_id": workflow_id,
            "status": "created",
            "message": "Workflow created from description successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create workflow from description: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@rowboat_router.get("/workflows/{workflow_id}")
async def get_workflow(
    workflow_id: str,
    ii_context: Dict = Depends(get_ii_agent_context)
):
    """Get workflow details"""
    
    workflows = ii_context["app"].state.workflows
    workflow = workflows.get(workflow_id)
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Remove internal fields
    response_workflow = workflow.copy()
    response_workflow.pop("langgraph", None)
    
    return response_workflow

@rowboat_router.delete("/workflows/{workflow_id}")
async def delete_workflow(
    workflow_id: str,
    coordinator: ROWBOATCoordinator = Depends(get_coordinator),
    ii_context: Dict = Depends(get_ii_agent_context)
):
    """Delete a workflow"""
    
    workflows = ii_context["app"].state.workflows
    
    if workflow_id not in workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Remove from storage
    del workflows[workflow_id]
    
    # Remove from coordinator if active
    if workflow_id in coordinator.active_workflows:
        del coordinator.active_workflows[workflow_id]
    
    logger.info(f"Deleted workflow: {workflow_id}")
    
    return {
        "status": "deleted",
        "message": f"Workflow '{workflow_id}' deleted successfully"
    }

# Workflow Execution Endpoints

@rowboat_router.post("/workflows/{workflow_id}/execute")
async def execute_workflow(
    workflow_id: str,
    request: WorkflowExecuteRequest,
    background_tasks: BackgroundTasks,
    coordinator: ROWBOATCoordinator = Depends(get_coordinator),
    ii_context: Dict = Depends(get_ii_agent_context)
):
    """Execute a workflow"""
    
    workflows = ii_context["app"].state.workflows
    workflow = workflows.get(workflow_id)
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
        # Ensure workflow is in coordinator's active workflows
    if workflow_id not in coordinator.active_workflows:
        logger.info(f"Workflow {workflow_id} not in coordinator, reconstructing...")
        
        from src.ii_agent.workflows.definitions import WorkflowDefinition, AgentConfig
        
        workflow_def = WorkflowDefinition(
            name=workflow.get("name", "Unknown"),
            description=workflow.get("description", ""),
            agents=[AgentConfig(**agent) for agent in workflow.get("agents", [])],
            edges=workflow.get("edges", []),
            metadata=workflow.get("metadata", {})
        )
        
        coordinator.active_workflows[workflow_id] = {
            "definition": workflow_def,
            "langgraph": workflow.get("langgraph"),
            "created_at": workflow.get("created_at"),
            "status": WorkflowStatus.INITIALIZING,
            "execution_history": [],
            "metrics": {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "average_duration_ms": 0
            }
        }
    # Generate execution ID
    execution_id = str(uuid.uuid4())
    
    # Prepare input with II-Agent context
    input_data = {
        "message": request.input or "",
        "context": request.context,
        "metadata": {
            "mode": request.mode,
            "mcp_enabled": ii_context.get("mcp_wrapper") is not None,
            "ollama_enabled": ii_context.get("ollama_wrapper") is not None,
            "execution_id": execution_id,
            "thread_id": f"thread_{execution_id}"  # Pass thread_id in metadata
        }
    }
    
    if request.messages:
        input_data["messages"] = request.messages
    
    if request.state:
        input_data["state"] = request.state
    
    # Execute workflow
    try:
        if request.stream:
            # Return streaming response
            async def generate():
                async for event in coordinator.execute_workflow_stream(
                    workflow_id,
                    input_data
                ):
                    yield f"data: {json.dumps(event)}\n\n"
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            # Regular execution - NO config parameter
            result = await coordinator.execute_workflow(
                workflow_id,
                input_data,
                stream_events=False
            )
            
            logger.info(f"Executed workflow {workflow_id}: {execution_id}")
            
            return {
                "success": result.get("success", False),
                "execution_id": execution_id,
                "result": result.get("result", {}),
                "agents_used": list(result.get("result", {}).get("agent_outputs", {}).keys()),
                "metadata": result.get("execution_metrics", {})
            }
            
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@rowboat_router.post("/chat")
async def chat(
    request: ChatRequest,
    coordinator: ROWBOATCoordinator = Depends(get_coordinator),
    ii_context: Dict = Depends(get_ii_agent_context)
):
    """Stateless chat endpoint integrated with II-Agent"""
    
    workflows = ii_context["app"].state.workflows
    workflow = workflows.get(request.workflow_id)
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
        # Ensure workflow is in coordinator's active workflows
    if request.workflow_id not in coordinator.active_workflows:
        logger.info(f"Workflow {request.workflow_id} not in coordinator, reconstructing...")
        
        from src.ii_agent.workflows.definitions import WorkflowDefinition, AgentConfig
        
        workflow_def = WorkflowDefinition(
            name=workflow.get("name", "Unknown"),
            description=workflow.get("description", ""),
            agents=[AgentConfig(**agent) for agent in workflow.get("agents", [])],
            edges=workflow.get("edges", []),
            metadata=workflow.get("metadata", {})
        )
        
        coordinator.active_workflows[request.workflow_id] = {
            "definition": workflow_def,
            "langgraph": workflow.get("langgraph"),
            "created_at": workflow.get("created_at"),
            "status": WorkflowStatus.INITIALIZING,
            "execution_history": [],
            "metrics": {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "average_duration_ms": 0
            }
        }
    # Execute as chat with II-Agent context
    input_data = {
        "messages": request.messages,
        "state": request.state,
        "session_id": request.session_id or str(uuid.uuid4()),
        "metadata": {
            "mode": "chat",
            "ii_agent_integration": True
        }
    }
    
    try:
        result = await coordinator.execute_workflow(
            request.workflow_id,
            input_data,
            stream_events=False
        )
        
        # Extract response
        response_content = ""
        new_state = request.state or {}
        
        if result.get("success"):
            agent_outputs = result.get("result", {}).get("agent_outputs", {})
            if agent_outputs:
                last_agent_output = list(agent_outputs.values())[-1]
                response_content = last_agent_output.get("output", "")
            
            new_state = result.get("result", {}).get("workflow_metadata", {}).get("state", new_state)
        
        return {
            "response": response_content,
            "state": new_state,
            "session_id": input_data["session_id"],
            "agents_used": list(result.get("result", {}).get("agent_outputs", {}).keys())
        }
        
    except Exception as e:
        logger.error(f"Chat execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Testing and Development Endpoints

@rowboat_router.post("/workflows/{workflow_id}/test")
async def test_workflow(
    workflow_id: str,
    request: WorkflowTestRequest,
    coordinator: ROWBOATCoordinator = Depends(get_coordinator),
    ii_context: Dict = Depends(get_ii_agent_context)
):
    """Test workflow in playground"""
    
    logger.info(f"=== TEST WORKFLOW START: {workflow_id} ===")
    logger.info(f"Test request: scenario_id={request.scenario_id}, custom_input={request.custom_input}")


    workflows = ii_context["app"].state.workflows
    logger.info(f"App state has {len(workflows)} workflows")
    logger.info(f"App state workflow IDs: {list(workflows.keys())}")
    workflow = workflows.get(workflow_id)

    
    if not workflow:
        logger.error(f"Workflow {workflow_id} not found in app state")
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    logger.info(f"Found workflow in app state: name={workflow.get('name')}, created_at={workflow.get('created_at')}")
    try:
        # Ensure workflow is in coordinator's active workflows
        if workflow_id not in coordinator.active_workflows:
            logger.info(f"Workflow {workflow_id} not in coordinator, reconstructing...")
            
            # Import the required classes
            from src.ii_agent.workflows.definitions import WorkflowDefinition, AgentConfig
            
            # Recreate the workflow definition from stored data
            workflow_def = WorkflowDefinition(
                name=workflow.get("name", "Unknown"),
                description=workflow.get("description", ""),
                agents=[AgentConfig(**agent) for agent in workflow.get("agents", [])],
                edges=workflow.get("edges", []),
                metadata=workflow.get("metadata", {})
            )
            
            # Add to coordinator's active workflows
            coordinator.active_workflows[workflow_id] = {
                "definition": workflow_def,
                "langgraph": workflow.get("langgraph"),
                "created_at": workflow.get("created_at"),
                "status": WorkflowStatus.INITIALIZING,
                "execution_history": [],
                "metrics": {
                    "total_executions": 0,
                    "successful_executions": 0,
                    "failed_executions": 0,
                    "average_duration_ms": 0
                }
            }
            
            logger.info(f"Reconstructed workflow {workflow_id} in coordinator")
        
        # Simple test execution using coordinator
        test_input = request.custom_input or "Test input"
        
        # Execute with mock mode
        result = await coordinator.execute_workflow(
            workflow_id,
            {
                "message": test_input,
                "metadata": {
                    "test_mode": True,
                    "mock_responses": request.mock_responses
                }
            },
            stream_events=False
        )
        
        return {
            "test_result": result,
            "scenario_id": request.scenario_id,
            "success": result.get("success", False)
        }
        
    except Exception as e:
        logger.error(f"=== TEST WORKFLOW FAILED: {workflow_id} ===")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Full error details:", exc_info=True)
        logger.error(f"Workflow test failed: {e}", exc_info=True)
        # Try to get more details about the error
        import traceback
        tb_str = traceback.format_exc()
        logger.error(f"Full traceback:\n{tb_str}")
        raise HTTPException(status_code=500, detail=str(e))

@rowboat_router.get("/templates")
async def list_templates(
    coordinator: ROWBOATCoordinator = Depends(get_coordinator)
):
    """List available workflow templates"""
    
    # Get templates from coordinator if available
    templates = {
        "customer_support": {
            "name": "Customer Support Assistant",
            "description": "Handle customer inquiries with intelligent routing",
            "agents": [
                {
                    "name": "classifier",
                    "role": "analyzer",
                    "instructions": "Classify incoming customer inquiries into categories: billing, technical, general. Route to @billing_agent, @tech_agent, or @general_agent accordingly."
                },
                {
                    "name": "billing_agent",
                    "role": "specialist",
                    "instructions": "Handle billing inquiries. Access customer account data and resolve billing issues."
                },
                {
                    "name": "tech_agent",
                    "role": "specialist",
                    "instructions": "Handle technical support requests. Troubleshoot issues and provide solutions."
                }
            ],
            "edges": [
                {"from_agent": "entry", "to_agent": "classifier"},
                {"from_agent": "classifier", "to_agent": "billing_agent", "condition": "category == 'billing'"},
                {"from_agent": "classifier", "to_agent": "tech_agent", "condition": "category == 'technical'"}
            ]
        },
        "research_assistant": {
            "name": "Research Assistant",
            "description": "Comprehensive research with analysis and report generation",
            "agents": [
                {
                    "name": "researcher",
                    "role": "researcher",
                    "instructions": "Search for information on the given topic. Gather data from multiple sources. Send findings to @analyzer."
                },
                {
                    "name": "analyzer",
                    "role": "analyzer",
                    "instructions": "Analyze research findings. Identify patterns and insights. Send analysis to @writer."
                },
                {
                    "name": "writer",
                    "role": "writer",
                    "instructions": "Create a comprehensive report based on research and analysis. Format in markdown."
                }
            ],
            "edges": [
                {"from_agent": "entry", "to_agent": "researcher"},
                {"from_agent": "researcher", "to_agent": "analyzer"},
                {"from_agent": "analyzer", "to_agent": "writer"}
            ]
        },
        "ii_agent_banking": {
            "name": "II-Agent Banking Assistant",
            "description": "Banking workflow with TCS BaNCS integration",
            "agents": [
                {
                    "name": "intent_classifier",
                    "role": "analyzer",
                    "instructions": "Classify banking requests: account, transaction, loan, card. Route to @account_agent, @transaction_agent, @loan_agent, or @card_agent."
                },
                {
                    "name": "account_agent",
                    "role": "specialist",
                    "instructions": "Handle account inquiries using BaNCS core banking APIs.",
                    "tools": ["bancs_account_api", "mcp:banking_server:get_account"]
                },
                {
                    "name": "transaction_agent",
                    "role": "specialist",
                    "instructions": "Process transaction requests and queries.",
                    "tools": ["bancs_transaction_api", "mcp:banking_server:list_transactions"]
                }
            ],
            "edges": [
                {"from_agent": "entry", "to_agent": "intent_classifier"},
                {"from_agent": "intent_classifier", "to_agent": "account_agent", "condition": "intent == 'account'"},
                {"from_agent": "intent_classifier", "to_agent": "transaction_agent", "condition": "intent == 'transaction'"}
            ]
        }
    }
    
    return {"templates": templates}

@rowboat_router.post("/workflows/{workflow_id}/enhance")
async def enhance_workflow(
    workflow_id: str,
    request: EnhanceWorkflowRequest,
    coordinator: ROWBOATCoordinator = Depends(get_coordinator),
    ii_context: Dict = Depends(get_ii_agent_context)
):
    """Use copilot to enhance workflow with II-Agent capabilities"""
    
    workflows = ii_context["app"].state.workflows
    workflow = workflows.get(workflow_id)
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    try:
        # Get suggestions from coordinator
        suggestions = await coordinator.suggest_workflow_improvements(workflow_id)
        
        # Add II-Agent specific enhancements
        enhancements = {
            "suggestions": suggestions.get("suggestions", []),
            "metrics": suggestions.get("metrics", {}),
            "tool_recommendations": []
        }
        
        # Suggest MCP tools if available
        if ii_context.get("mcp_wrapper"):
            enhancements["tool_recommendations"].extend([
                "mcp:banking_server:get_balance",
                "mcp:banking_server:transfer_funds"
            ])
        
        return {
            "workflow_id": workflow_id,
            "enhancements": enhancements,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Workflow enhancement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Additional ROWBOAT-specific endpoints

@rowboat_router.get("/workflows/{workflow_id}/visualization")
async def get_workflow_visualization(
    workflow_id: str,
    coordinator: ROWBOATCoordinator = Depends(get_coordinator)
):
    """Get workflow visualization data"""
    
    return coordinator.get_workflow_visualization_data(workflow_id)

@rowboat_router.post("/workflows/{workflow_id}/suggestions")
async def get_workflow_suggestions(
    workflow_id: str,
    coordinator: ROWBOATCoordinator = Depends(get_coordinator)
):
    """Get AI-powered workflow improvement suggestions"""
    
    return await coordinator.suggest_workflow_improvements(workflow_id)

# Visual Builder Endpoints

@rowboat_router.get("/visual-builder/config")
async def get_visual_builder_config(
    ii_context: Dict = Depends(get_ii_agent_context)
):
    """Get visual builder configuration with II-Agent additions"""
    
    config = {
        "nodes": [
            {
                "type": "agent_researcher",
                "label": "Researcher",
                "color": "#4A90E2",
                "icon": "🔍",
                "ports": {
                    "inputs": [{"id": "query", "label": "Query"}],
                    "outputs": [{"id": "findings", "label": "Findings"}]
                }
            },
            {
                "type": "agent_analyzer",
                "label": "Analyzer",
                "color": "#7B68EE",
                "icon": "📊",
                "ports": {
                    "inputs": [{"id": "data", "label": "Data"}],
                    "outputs": [{"id": "analysis", "label": "Analysis"}]
                }
            },
            {
                "type": "agent_writer",
                "label": "Writer",
                "color": "#50C878",
                "icon": "✍️",
                "ports": {
                    "inputs": [{"id": "content", "label": "Content"}],
                    "outputs": [{"id": "document", "label": "Document"}]
                }
            },
            {
                "type": "agent_bancs_specialist",
                "label": "BaNCS Specialist",
                "color": "#FF5733",
                "icon": "🏦",
                "ports": {
                    "inputs": [{"id": "query", "label": "Banking Query"}],
                    "outputs": [{"id": "response", "label": "Banking Response"}]
                },
                "properties": {
                    "description": "TCS BaNCS banking specialist",
                    "tools": ["bancs_api", "mcp:banking_server"],
                    "requires_auth": True
                }
            }
        ],
        "edges": [
            {
                "type": "direct",
                "label": "Direct Flow",
                "style": {"strokeWidth": 2, "stroke": "#333"}
            },
            {
                "type": "conditional",
                "label": "Conditional",
                "style": {"strokeWidth": 2, "stroke": "#FF9800", "strokeDasharray": "5,5"}
            },
            {
                "type": "mention",
                "label": "@Mention",
                "style": {"strokeWidth": 2, "stroke": "#4CAF50", "strokeDasharray": "3,3"}
            }
        ]
    }
    
    return config

# ===== Streaming Workflow Creation =====
@rowboat_router.post("/workflows/from-description/stream")
async def create_workflow_from_description_stream(
    request: WorkflowFromDescriptionRequest,
    coordinator: ROWBOATCoordinator = Depends(get_coordinator),
    ii_context: Dict = Depends(get_ii_agent_context)
):
    """Create workflow from natural language with SSE streaming"""
    
    async def generate_events():
        try:
            # Variables to capture workflow data during streaming
            workflow_id = None
            workflow_definition = None
            
            # Initial progress
            yield f"data: {json.dumps({'type': 'progress', 'data': {'message': 'Initializing workflow creation...', 'progress': 10}})}\n\n"
            await asyncio.sleep(0.1)
            
            # Parse description
            yield f"data: {json.dumps({'type': 'progress', 'data': {'message': 'Analyzing your description...', 'progress': 20}})}\n\n"
            
            # Create workflow using copilot
            # First create the LLM client
            model_config = rowboat_config.get_model_config()
            copilot_model = request.model_preferences.get("copilot_model", model_config["copilot_model"])
            
            copilot_llm = ChutesOpenAIClient(
                model_name=copilot_model,
                use_native_tool_calling=True,
                fallback_models=model_config.get("fallback_models", [])
            )
            
            workflow_copilot = WorkflowCopilot(
                copilot_llm,
                default_model=model_config.get("default_agent_model")
            )
            
            # Stream agent creation events
            agent_count = 0
            async for event in workflow_copilot.build_workflow_stream(
                description=request.description,
                examples=request.examples,
                documents=request.documents
            ):
                # Capture workflow definition when complete
                if event["type"] == WorkflowStreamEvent.WORKFLOW_COMPLETE.value:
                    workflow_definition = await workflow_copilot.get_workflow_definition()
                
                if event["type"] == WorkflowStreamEvent.AGENT_CREATED.value:
                    agent_count += 1
                    yield f"data: {json.dumps({'type': 'agent', 'data': event['data']})}\n\n"
                    progress = min(30 + (agent_count * 10), 70)
                    yield f"data: {json.dumps({'type': 'progress', 'data': {'message': f'Created agent: {event["data"]["name"]}', 'progress': progress}})}\n\n"
                
                elif event["type"] == WorkflowStreamEvent.TOOL_ADDED.value:
                    yield f"data: {json.dumps({'type': 'tool', 'data': event['data']})}\n\n"
                
                elif event["type"] == WorkflowStreamEvent.EDGE_CREATED.value:
                    yield f"data: {json.dumps({'type': 'edge', 'data': event['data']})}\n\n"
            
            # Finalize workflow
            yield f"data: {json.dumps({'type': 'progress', 'data': {'message': 'Finalizing workflow configuration...', 'progress': 80}})}\n\n"
            
            # Create user context (same as non-streaming)
            user_context = {
                "examples": request.examples,
                "documents": request.documents,
                "model_preferences": request.model_preferences,
                "mcp_available": ii_context.get("mcp_wrapper") is not None,
                "ollama_available": ii_context.get("ollama_wrapper") is not None
            }
            
            # Now create the workflow in the coordinator
            created_workflow_id = await coordinator.create_workflow(workflow_definition, user_context)
            
            # Get workflow info from coordinator (same as non-streaming)
            workflow_info = coordinator.active_workflows.get(created_workflow_id)
            
            if workflow_info:
                # Store in II-Agent workflow storage (exact same as non-streaming)
                workflow_data = {
                    "id": created_workflow_id,
                    "name": workflow_info["definition"].name,
                    "description": workflow_info["definition"].description,
                    "agents": [agent.dict() for agent in workflow_info["definition"].agents],
                    "edges": workflow_info["definition"].edges,  # No .dict() call
                    "metadata": workflow_info["definition"].metadata,
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                    "langgraph": workflow_info["langgraph"]
                }
                
                ii_context["app"].state.workflows[created_workflow_id] = workflow_data
                logger.info(f"Created workflow from description: {created_workflow_id}")
            
            # Complete
            yield f"data: {json.dumps({'type': 'progress', 'data': {'message': 'Workflow created successfully!', 'progress': 100}})}\n\n"
            
            yield f"data: {json.dumps({'type': 'complete', 'data': {'workflow_id': created_workflow_id, 'summary': {'agents': len(workflow_definition.agents), 'tools': sum(len(a.tools) for a in workflow_definition.agents), 'edges': len(workflow_definition.edges)}}})}\n\n"
            
        except Exception as e:
            logger.error(f"Failed to create workflow from description (streaming): {e}")
            yield f"data: {json.dumps({'type': 'error', 'data': {'error': str(e)}})}\n\n"
    
    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable Nginx buffering
        }
    )

# ===== Workflow Metrics =====
@rowboat_router.get("/workflows/{workflow_id}/metrics")
async def get_workflow_metrics(
    workflow_id: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    coordinator: ROWBOATCoordinator = Depends(get_coordinator),
    ii_context: Dict = Depends(get_ii_agent_context)
):
    """Get workflow execution metrics"""
    
    workflows = ii_context["app"].state.workflows
    if workflow_id not in workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Get metrics from coordinator
    workflow_info = coordinator.active_workflows.get(workflow_id, {})
    
    # Calculate real metrics
    metrics = {
        "workflow_id": workflow_id,
        "total_executions": workflow_info.get("metrics", {}).get("total_executions", 0),
        "successful_executions": workflow_info.get("metrics", {}).get("successful_executions", 0),
        "failed_executions": workflow_info.get("metrics", {}).get("failed_executions", 0),
        "avg_duration": workflow_info.get("metrics", {}).get("average_duration_ms", 0) / 1000,  # Convert to seconds
        "success_rate": 0,
        "execution_history": [],
        "agent_metrics": {},
        "time_range": {
            "start": start_date.isoformat() if start_date else None,
            "end": end_date.isoformat() if end_date else None
        }
    }
    
    # Calculate success rate
    if metrics["total_executions"] > 0:
        metrics["success_rate"] = metrics["successful_executions"] / metrics["total_executions"]
    
    # Get execution history
    if "execution_history" in workflow_info:
        history = workflow_info["execution_history"]
        
        # Filter by date range if provided
        if start_date or end_date:
            filtered_history = []
            for execution in history:
                exec_time = datetime.fromisoformat(execution.get("completed_at", execution.get("started_at")))
                if start_date and exec_time < start_date:
                    continue
                if end_date and exec_time > end_date:
                    continue
                filtered_history.append(execution)
            history = filtered_history
        
        # Add to metrics
        metrics["execution_history"] = [
            {
                "execution_id": exec.get("execution_id"),
                "started_at": exec.get("started_at"),
                "completed_at": exec.get("completed_at"),
                "duration_ms": exec.get("duration_ms"),
                "status": "success" if not exec.get("error") else "failed",
                "agent_count": exec.get("handoff_count", 0) + 1
            }
            for exec in history[-10:]  # Last 10 executions
        ]
    
    # Calculate per-agent metrics
    for agent in workflows[workflow_id].get("agents", []):
        agent_name = agent["name"]
        metrics["agent_metrics"][agent_name] = {
            "invocations": 0,
            "avg_duration_ms": 0,
            "error_count": 0
        }
    
    return metrics

# ===== MCP Server Management =====
@rowboat_router.get("/mcp/servers")
async def get_mcp_servers(
    ii_context: Dict = Depends(get_ii_agent_context)
):
    """Get available MCP servers with real status"""
    
    # logger.info("MCP servers check initiated")
    
    mcp_wrapper = ii_context.get("mcp_wrapper")
    servers = []
    
    # Get configuration from environment
    base_url = os.getenv("MCP_BASE_URL", "http://localhost:8082")
    sse_url = os.getenv("MCP_SSE_URL", "http://localhost:8084/mcp")
    api_key = os.getenv("MCP_API_KEY", "test-api-key-123")
    
    # Base MCP server info
    base_server = {
        "name": "mcp_base_server",
        "url": base_url,
        "tools": [],
        "tool_count": 0,
        "categories": {},
        "status": "disconnected",
        "error": None
    }
    
    try:
        if mcp_wrapper:
            # Check if wrapper has tool_registry (indicates it's initialized)
            if hasattr(mcp_wrapper, 'tool_registry'):
                base_server["status"] = "connected"
                # logger.info("MCP base server connected via wrapper")
                
                # Get tool statistics
                if hasattr(mcp_wrapper.tool_registry, 'get_statistics'):
                    stats = mcp_wrapper.tool_registry.get_statistics()
                    base_server["tool_count"] = stats.get("total_tools", 0)
                    base_server["categories"] = stats.get("categories", {})
                    # logger.debug(f"Base server has {base_server['tool_count']} tools")
                
                # Get actual tool names (limit to avoid large response)
                if hasattr(mcp_wrapper.tool_registry, 'list_tools'):
                    try:
                        tools = mcp_wrapper.tool_registry.list_tools()
                        base_server["tools"] = [
                            getattr(tool, 'name', str(tool)) 
                            for tool in tools[:10]  # First 10 tools only
                        ]
                        if len(tools) > 10:
                            base_server["tools"].append(f"... and {len(tools) - 10} more")
                    except Exception as e:
                        logger.warning(f"Could not list tools: {e}")
            else:
                # Wrapper exists but not initialized
                base_server["status"] = "not_initialized"
                
                # Check if it has URL configuration
                if hasattr(mcp_wrapper, 'base_url'):
                    base_server["url"] = mcp_wrapper.base_url
                elif hasattr(mcp_wrapper, '_base_url'):
                    base_server["url"] = mcp_wrapper._base_url
        else:
            # No wrapper, try to check if server is reachable
            logger.debug("No MCP wrapper, checking server availability")
            
            import aiohttp
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(
                        f"{base_url}/health",
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=aiohttp.ClientTimeout(total=2)
                    ) as response:
                        if response.status == 200:
                            base_server["status"] = "available"
                            # logger.info("MCP base server is available but not connected")
                        else:
                            logger.debug(f"Health check returned status {response.status}")
                except Exception as e:
                    # logger.debug(f"Health check failed: {e}")
                    base_server["status"] = "unreachable"
                    
    except Exception as e:
        logger.error(f"Error checking MCP base server: {e}")
        base_server["error"] = str(e)
        base_server["status"] = "error"
    
    servers.append(base_server)
    
    # SSE server info (if configured differently)
    if sse_url and sse_url != base_url:
        sse_server = {
            "name": "mcp_sse_server",
            "url": sse_url,
            "tools": [],
            "tool_count": 0,
            "categories": {},
            "status": "disconnected",
            "error": None,
            "description": "Server-Sent Events endpoint for real-time updates"
        }
        
        try:
            if mcp_wrapper:
                # Check if wrapper has SSE endpoint configured
                if hasattr(mcp_wrapper, 'sse_endpoint'):
                    sse_server["url"] = mcp_wrapper.sse_endpoint
                    sse_server["status"] = "configured"
                elif hasattr(mcp_wrapper, '_sse_endpoint'):
                    sse_server["url"] = mcp_wrapper._sse_endpoint
                    sse_server["status"] = "configured"
                    
        except Exception as e:
            logger.error(f"Error checking MCP SSE server: {e}")
            sse_server["error"] = str(e)
        
        servers.append(sse_server)
    
    # Add summary
    total_tools = sum(server.get("tool_count", 0) for server in servers)
    connected_servers = sum(1 for server in servers if server["status"] == "connected")
    
    # logger.info(f"MCP servers check completed: {connected_servers}/{len(servers)} connected, {total_tools} total tools")
    
    return {
        "servers": servers,
        "summary": {
            "total_servers": len(servers),
            "connected": connected_servers,
            "total_tools": total_tools
        }
    }

# ===== MCP Status Endpoint =====
# @rowboat_router.get("/mcp/status")
@rowboat_router.get("/mcp/status")
async def get_mcp_status(
    ii_context: Dict = Depends(get_ii_agent_context)
):
    """Get detailed MCP integration status"""
    
    # logger.info("MCP status check initiated")
    
    mcp_wrapper = ii_context.get("mcp_wrapper")
    # logger.debug(f"MCP wrapper present: {bool(mcp_wrapper)}")
    # logger.debug(f"MCP wrapper type: {type(mcp_wrapper).__name__ if mcp_wrapper else 'None'}")
    
    status = {
        "mcp_enabled": bool(mcp_wrapper),
        "configuration": {
            "base_url": os.getenv("MCP_BASE_URL", "http://localhost:8082"),
            "sse_url": os.getenv("MCP_SSE_URL", "http://localhost:8084/mcp"),
            "api_key_configured": bool(os.getenv("MCP_API_KEY"))
        },
        "connection_status": "disconnected",
        "available_tools": [],
        "tool_statistics": {},
        "active_sessions": 0,
        "last_error": None
    }
    
    # logger.debug(f"MCP configuration: base_url={status['configuration']['base_url']}, "
                # f"sse_url={status['configuration']['sse_url']}, "
                # f"api_key_configured={status['configuration']['api_key_configured']}")
    
    if mcp_wrapper:
        try:
            # Check if wrapper has tool_registry (indicates it's initialized)
            if hasattr(mcp_wrapper, 'tool_registry'):
                status["connection_status"] = "connected"
                # logger.info("MCP wrapper has tool_registry - marked as connected")
                
                # Get tool statistics
                if hasattr(mcp_wrapper.tool_registry, 'get_statistics'):
                    stats = mcp_wrapper.tool_registry.get_statistics()
                    status["tool_statistics"] = stats
                    # logger.info(f"Retrieved tool statistics: {stats}")
                    
                    # Extract total tools count
                    total_tools = stats.get("total_tools", 0)
                    # logger.info(f"Total MCP tools available: {total_tools}")
                    
                    # Get tool categories
                    categories = stats.get("categories", {})
                    # logger.debug(f"Tool categories: {categories}")
                
                # Try to get actual tool list
                if hasattr(mcp_wrapper.tool_registry, 'list_tools'):
                    try:
                        tools = mcp_wrapper.tool_registry.list_tools()
                        # Convert tool objects to serializable format
                        status["available_tools"] = [
                            {
                                "name": getattr(tool, 'name', str(tool)),
                                "description": getattr(tool, 'description', ''),
                                "category": getattr(tool, 'category', 'general')
                            }
                            for tool in tools[:20]  # Limit to first 20 tools
                        ]
                        logger.info(f"Retrieved {len(tools)} tools, showing first {len(status['available_tools'])}")
                    except Exception as e:
                        logger.warning(f"Could not list tools: {e}")
            else:
                # Wrapper exists but not initialized
                status["connection_status"] = "not_initialized"
                # logger.warning("MCP wrapper exists but tool_registry not found - not initialized")
                
                # Check if it has base_url (configuration indicator)
                if hasattr(mcp_wrapper, 'base_url'):
                    status["configuration"]["detected_base_url"] = mcp_wrapper.base_url
                elif hasattr(mcp_wrapper, '_base_url'):
                    status["configuration"]["detected_base_url"] = mcp_wrapper._base_url
                
        except Exception as e:
            status["last_error"] = str(e)
            # logger.error(f"Error getting MCP status: {e}", exc_info=True)
    else:
        logger.warning("MCP wrapper not found in context - MCP functionality disabled")
    
    # Add summary info
    status["summary"] = {
        "enabled": status["mcp_enabled"],
        "connected": status["connection_status"] == "connected",
        "total_tools": status["tool_statistics"].get("total_tools", 0),
        "categories": list(status["tool_statistics"].get("categories", {}).keys())
    }
    
    # logger.info(f"MCP status check completed: enabled={status['mcp_enabled']}, "
            #    f"connection={status['connection_status']}, "
            #    f"total_tools={status['summary']['total_tools']}")
    
    # logger.debug(f"Complete MCP status: {json.dumps(status, indent=2)}")
    
    return status

# WebSocket endpoint for real-time workflow execution

@rowboat_router.websocket("/ws/{workflow_id}")
async def websocket_workflow_execution(
    websocket: WebSocket,
    workflow_id: str,
):
    """WebSocket endpoint for real-time workflow execution"""
    
    # Get dependencies directly from websocket.app
    app = websocket.app
    
    # Get coordinator
    if not hasattr(app.state, "rowboat_coordinator") or not app.state.rowboat_coordinator:
        await websocket.close(code=4503, reason="ROWBOAT coordinator not initialized")
        return
    
    coordinator = app.state.rowboat_coordinator
    
    # Get II context directly
    ii_context = {
        "app": app,
        "mcp_wrapper": getattr(app.state, "mcp_wrapper", None),
        "ollama_wrapper": getattr(app.state, "ollama_wrapper", None),
        "workspace_manager": getattr(app.state, "workspace_manager", None)
    }
    
    await websocket.accept()
    
    workflows = ii_context["app"].state.workflows
    workflow = workflows.get(workflow_id)
    
    if not workflow:
        await websocket.close(code=4004, reason="Workflow not found")
        return
    
    # Ensure workflow is in coordinator's active workflows
    if workflow_id not in coordinator.active_workflows:
        logger.info(f"Workflow {workflow_id} not in coordinator, reconstructing...")
        
        from src.ii_agent.workflows.definitions import WorkflowDefinition, AgentConfig
        
        workflow_def = WorkflowDefinition(
            name=workflow.get("name", "Unknown"),
            description=workflow.get("description", ""),
            agents=[AgentConfig(**agent) for agent in workflow.get("agents", [])],
            edges=workflow.get("edges", []),
            metadata=workflow.get("metadata", {})
        )
        
        coordinator.active_workflows[workflow_id] = {
            "definition": workflow_def,
            "langgraph": workflow.get("langgraph"),
            "created_at": workflow.get("created_at"),
            "status": WorkflowStatus.INITIALIZING,
            "execution_history": [],
            "metrics": {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "average_duration_ms": 0
            }
        }
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            # Generate execution ID
            execution_id = str(uuid.uuid4())
            
            # Prepare input data similar to the REST endpoint
            input_data = {
                "message": data.get("message", ""),
                "context": data.get("context", {}),
                "metadata": {
                    "mode": "websocket",
                    "mcp_enabled": ii_context.get("mcp_wrapper") is not None,
                    "ollama_enabled": ii_context.get("ollama_wrapper") is not None,
                    "execution_id": execution_id,
                    "thread_id": f"thread_{execution_id}",
                    "session_id": data.get("session_id", str(uuid.uuid4()))
                }
            }
            
            if data.get("messages"):
                input_data["messages"] = data["messages"]
            
            if data.get("state"):
                input_data["state"] = data["state"]
            
            # Execute workflow
            try:
                # Send start event
                await websocket.send_json({
                    "type": "execution_started",
                    "execution_id": execution_id,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Execute workflow
                result = await coordinator.execute_workflow(
                    workflow_id,
                    input_data,
                    stream_events=False
                )
                
                # Convert result to JSON-serializable format
                def serialize_result(obj):
                    """Convert LangChain objects to serializable format"""
                    if hasattr(obj, 'dict'):
                        return obj.dict()
                    elif hasattr(obj, 'content'):
                        # For HumanMessage, AIMessage, etc.
                        return {
                            "type": obj.__class__.__name__,
                            "content": obj.content,
                            "additional_kwargs": getattr(obj, 'additional_kwargs', {})
                        }
                    elif isinstance(obj, dict):
                        return {k: serialize_result(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [serialize_result(item) for item in obj]
                    elif isinstance(obj, (str, int, float, bool, type(None))):
                        return obj
                    else:
                        return str(obj)
                
                # Serialize the result
                serialized_result = serialize_result(result)
                
                # Extract the final response
                final_response = ""
                if "result" in serialized_result and "agent_outputs" in serialized_result["result"]:
                    agent_outputs = serialized_result["result"]["agent_outputs"]
                    if agent_outputs:
                        # Get the last agent's output
                        last_agent = list(agent_outputs.keys())[-1]
                        last_output = agent_outputs[last_agent]
                        if isinstance(last_output, dict) and "output" in last_output:
                            final_response = last_output["output"]
                        elif isinstance(last_output, dict) and "content" in last_output:
                            final_response = last_output["content"]
                
                # Send result
                await websocket.send_json({
                    "type": "execution_result",
                    "data": {
                        "success": serialized_result.get("success", False),
                        "execution_id": execution_id,
                        "response": final_response,  # Add the actual response
                        "result": serialized_result.get("result", {}),
                        "agents_used": list(serialized_result.get("result", {}).get("agent_outputs", {}).keys()),
                        "metadata": serialized_result.get("execution_metrics", {})
                    }
                })
                
                # Send completion
                await websocket.send_json({
                    "type": "complete",
                    "session_id": input_data["metadata"]["session_id"],
                    "execution_id": execution_id
                })
                
            except Exception as e:
                logger.error(f"Workflow execution failed: {e}", exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "error": str(e),
                    "execution_id": execution_id
                })
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for workflow {workflow_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        await websocket.close(code=4000, reason=str(e))

# Add these routes to your rowboat_routes.py or create a new mcp_routes.py file

# @rowboat_router.get("/mcp/status")
# async def get_mcp_status(ii_context: Dict = Depends(get_ii_agent_context)):
#     """Get MCP service status"""
#     mcp_wrapper = ii_context.get("mcp_wrapper")
    
#     if not mcp_wrapper:
#         return {
#             "status": "disabled",
#             "message": "MCP not initialized",
#             "timestamp": datetime.now().isoformat()
#         }
    
#     try:
#         # Check if MCP wrapper is connected
#         is_connected = hasattr(mcp_wrapper, 'is_connected') and mcp_wrapper.is_connected
        
#         return {
#             "status": "connected" if is_connected else "disconnected",
#             "base_url": getattr(mcp_wrapper, 'base_url', 'http://localhost:8082'),
#             "timestamp": datetime.now().isoformat()
#         }
#     except Exception as e:
#         logger.error(f"Error checking MCP status: {e}")
#         return {
#             "status": "error",
#             "message": str(e),
#             "timestamp": datetime.now().isoformat()
#         }

# @rowboat_router.get("/mcp/servers")
# async def get_mcp_servers(ii_context: Dict = Depends(get_ii_agent_context)):
#     """Get list of available MCP servers"""
#     mcp_wrapper = ii_context.get("mcp_wrapper")
    
#     if not mcp_wrapper:
#         return {
#             "servers": [],
#             "message": "MCP not initialized",
#             "timestamp": datetime.now().isoformat()
#         }
    
#     try:
#         # Get available MCP servers/tools
#         servers = []
        
#         # Check if MCP wrapper has list_available_tools method
#         if hasattr(mcp_wrapper, 'list_available_tools'):
#             tools = await mcp_wrapper.list_available_tools()
            
#             # Group tools by server/category
#             server_map = {}
#             for tool in tools:
#                 server_name = tool.get('server', 'default')
#                 if server_name not in server_map:
#                     server_map[server_name] = {
#                         "name": server_name,
#                         "status": "connected",
#                         "tools": []
#                     }
#                 server_map[server_name]["tools"].append({
#                     "name": tool.get('name'),
#                     "description": tool.get('description', '')
#                 })
            
#             servers = list(server_map.values())
#         else:
#             # Fallback: return default server info
#             servers = [{
#                 "name": "core_banking",
#                 "status": "connected" if mcp_wrapper else "disconnected",
#                 "url": getattr(mcp_wrapper, 'base_url', 'http://localhost:8082'),
#                 "tools": []
#             }]
        
#         return {
#             "servers": servers,
#             "total": len(servers),
#             "timestamp": datetime.now().isoformat()
#         }
#     except Exception as e:
#         logger.error(f"Error getting MCP servers: {e}")
#         return {
#             "servers": [],
#             "error": str(e),
#             "timestamp": datetime.now().isoformat()
#         }

# Optional: Add a health check endpoint specifically for MCP
@rowboat_router.get("/mcp/health")
async def check_mcp_health(ii_context: Dict = Depends(get_ii_agent_context)):
    """Check health of MCP services"""
    mcp_wrapper = ii_context.get("mcp_wrapper")
    
    health_status = {
        "mcp_wrapper": mcp_wrapper is not None,
        "services": {},
        "timestamp": datetime.now().isoformat()
    }
    
    if mcp_wrapper:
        # Check individual MCP services if available
        try:
            # Try to ping the MCP server
            if hasattr(mcp_wrapper, 'ping') or hasattr(mcp_wrapper, 'health_check'):
                health_status["services"]["core_banking"] = {
                    "status": "healthy",
                    "url": getattr(mcp_wrapper, 'base_url', 'http://localhost:8082')
                }
            else:
                health_status["services"]["core_banking"] = {
                    "status": "unknown",
                    "message": "Health check not implemented"
                }
        except Exception as e:
            health_status["services"]["core_banking"] = {
                "status": "error",
                "error": str(e)
            }
    
    return health_status
# Integration function to add ROWBOAT routes to existing II-Agent app

def integrate_rowboat_with_ii_agent(app):
    """
    Integrate ROWBOAT/Multi-Agent routes with existing II-Agent FastAPI app
    
    Usage in II-Agent main.py:
    ```python
    from src.ii_agent.api.rowboat_routes import integrate_rowboat_with_ii_agent
    
    # After creating FastAPI app
    app = FastAPI(...)
    
    # Add multi-agent integration
    integrate_rowboat_with_ii_agent(app)
    ```
    """
    
    # Include the router
    app.include_router(rowboat_router)
    
    # Initialize workflow storage in app state
    if not hasattr(app.state, "workflows"):
        app.state.workflows = {}
    
    logger.info("ROWBOAT multi-agent workflow capabilities integrated with II-Agent")
    
    return app