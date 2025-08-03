# src/ii_agent/copilot/client.py
from typing import Dict, List, Any, Optional, Union
import httpx
import asyncio
from datetime import datetime
import json
import uuid

# FastAPI endpoints for server
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# Import necessary components
from src.ii_agent.copilot.workflow_builder import WorkflowCopilot
from src.ii_agent.llm.chutes_openai import ChutesOpenAIClient
from src.ii_agent.workflows.definitions import WorkflowDefinition

# Request/Response models
class CreateWorkflowRequest(BaseModel):
    description: Optional[str] = None
    workflow: Optional[Dict[str, Any]] = None
    model_preferences: Optional[Dict[str, str]] = None

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    workflow_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    workflow_id: Optional[str] = None
    agents_used: List[str] = []
    execution_time_ms: int

# FastAPI app setup
app = FastAPI(title="II-Agent Copilot API")
security = HTTPBearer()

# Client implementation
class IIAgentClient:
    """Client for interacting with II-Agent workflows"""
    
    def __init__(
        self,
        host: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        project_id: str = "default"
    ):
        self.host = host.rstrip('/')
        self.api_key = api_key
        self.project_id = project_id
        self.headers = {}
        
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        workflow_id: Optional[str] = None,
        model_preferences: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Send chat messages to a workflow"""
        
        endpoint = f"{self.host}/api/v1/chat"
        payload = {
            "messages": messages,
            "workflow_id": workflow_id
        }
        
        if model_preferences:
            payload["model_preferences"] = model_preferences
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                endpoint,
                json=payload,
                headers=self.headers,
                timeout=300.0  # 5 minute timeout for long workflows
            )
            response.raise_for_status()
            return response.json()
    
    async def create_workflow(
        self,
        description: Optional[str] = None,
        workflow_config: Optional[Dict[str, Any]] = None,
        model_preferences: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Create a new workflow using copilot or config"""
        
        endpoint = f"{self.host}/api/v1/{self.project_id}/workflows"
        
        if description:
            # Use copilot
            payload = {
                "description": description,
                "model_preferences": model_preferences or {}
            }
        else:
            # Use direct config
            payload = {"workflow": workflow_config}
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                endpoint,
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
    
    def list_available_models(self) -> Dict[str, Any]:
        """List available Chutes models"""
        from src.ii_agent.llm.model_registry import ChutesModelRegistry
        return ChutesModelRegistry.AVAILABLE_MODELS


class StatefulChat:
    """Stateful chat session with conversation history"""
    
    def __init__(self, client: IIAgentClient):
        self.client = client
        self.conversation_id = str(uuid.uuid4())
        self.message_history: List[Dict[str, str]] = []
        self.workflow_id: Optional[str] = None
        self.created_at = datetime.now()
        self.metadata: Dict[str, Any] = {}
    
    async def send_message(self, message: str, role: str = "user") -> Dict[str, Any]:
        """Send a message and maintain conversation history"""
        
        # Add message to history
        self.message_history.append({
            "role": role,
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Send to API
        response = await self.client.chat(
            messages=self.message_history,
            workflow_id=self.workflow_id
        )
        
        # Add response to history
        if "response" in response:
            self.message_history.append({
                "role": "assistant",
                "content": response["response"],
                "timestamp": datetime.now().isoformat()
            })
        
        # Update workflow_id if provided
        if "workflow_id" in response:
            self.workflow_id = response["workflow_id"]
        
        return response
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.message_history.copy()
    
    def clear_history(self):
        """Clear conversation history"""
        self.message_history.clear()
        self.workflow_id = None
    
    def save_session(self, filepath: str):
        """Save session to file"""
        session_data = {
            "conversation_id": self.conversation_id,
            "workflow_id": self.workflow_id,
            "created_at": self.created_at.isoformat(),
            "message_history": self.message_history,
            "metadata": self.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def load_session(self, filepath: str):
        """Load session from file"""
        with open(filepath, 'r') as f:
            session_data = json.load(f)
        
        self.conversation_id = session_data["conversation_id"]
        self.workflow_id = session_data.get("workflow_id")
        self.created_at = datetime.fromisoformat(session_data["created_at"])
        self.message_history = session_data["message_history"]
        self.metadata = session_data.get("metadata", {})


# API Endpoints
@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Handle chat requests"""
    
    # Process chat with workflow
    # This would connect to the actual workflow execution
    result = await process_chat_with_workflow(
        request.messages,
        workflow_id=request.workflow_id
    )
    
    return result

@app.get("/api/v1/models")
async def list_models_endpoint():
    """List available Chutes models"""
    from src.ii_agent.llm.model_registry import ChutesModelRegistry
    
    models = []
    for key, info in ChutesModelRegistry.AVAILABLE_MODELS.items():
        models.append({
            "id": key,
            "model_id": info["model_id"],
            "capabilities": info["capabilities"],
            "context_window": info["context_window"],
            "recommended_for": info["recommended_for"]
        })
    
    return {"models": models}

@app.post("/api/v1/{project_id}/workflows")
async def create_workflow_endpoint(
    project_id: str,
    request: CreateWorkflowRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Create a new workflow with Chutes model selection"""
    
    if request.description:
        # Use copilot with Chutes LLM
        copilot = WorkflowCopilot(
            ChutesOpenAIClient(model_name="deepseek-ai/DeepSeek-V3-0324")
        )
        
        # Pass model preferences to copilot
        workflow = await copilot.build_from_description(
            request.description,
            user_context={"model_preferences": request.model_preferences}
        )
    else:
        workflow = request.workflow
    
    # Save workflow
    workflow_id = await save_workflow(project_id, workflow)
    
    return {"workflow_id": workflow_id, "workflow": workflow}


# Helper functions (would be implemented in actual service)
async def process_chat_with_workflow(
    messages: List[Dict[str, str]],
    workflow_id: Optional[str] = None
) -> ChatResponse:
    """Process chat messages through workflow"""
    # This would connect to actual workflow execution
    # For now, return a mock response
    return ChatResponse(
        response="This is a mock response. Implement actual workflow processing.",
        workflow_id=workflow_id or str(uuid.uuid4()),
        agents_used=["assistant"],
        execution_time_ms=100
    )

async def save_workflow(project_id: str, workflow: Dict[str, Any]) -> str:
    """Save workflow to database"""
    # This would save to actual database
    # For now, return a mock ID
    return str(uuid.uuid4())