# src/ii_agent/sdk/rowboat_client_enhanced.py
"""
Enhanced ROWBOAT SDK Client
Natural language SDK for building and interacting with multi-agent workflows
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Union, AsyncIterator, Callable
from datetime import datetime
import logging
from enum import Enum
from dataclasses import dataclass
import uuid
from contextlib import asynccontextmanager

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logging.warning("httpx not available - using requests fallback")
    import requests

logger = logging.getLogger(__name__)

class WorkflowMode(Enum):
    """Workflow interaction modes"""
    CHAT = "chat"
    EXECUTE = "execute"
    STREAM = "stream"
    BATCH = "batch"
    PLAYGROUND = "playground"

@dataclass
class WorkflowResult:
    """Structured workflow result"""
    success: bool
    workflow_id: str
    execution_id: str
    result: Any
    duration_ms: int
    agents_used: List[str]
    metadata: Dict[str, Any]
    handoff_count: int = 0
    mention_count: int = 0

@dataclass
class TestResult:
    """Playground test result"""
    scenario_id: str
    passed: bool
    execution_trace: List[Dict[str, Any]]
    assertions: List[Dict[str, Any]]
    duration_ms: int
    error: Optional[str] = None

class StatefulChat:
    """Stateful chat session with a workflow"""
    
    def __init__(self, client: 'EnhancedROWBOATClient', workflow_id: str):
        self.client = client
        self.workflow_id = workflow_id
        self.session_id = str(uuid.uuid4())
        self.state = {}
        self.message_history = []
        self.metadata = {
            "created_at": datetime.utcnow().isoformat(),
            "message_count": 0,
            "agent_interactions": {}
        }
    
    async def send(self, message: str, context: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """Send a message and maintain state"""
        
        # Add to history
        self.message_history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Prepare request with state
        request_data = {
            "messages": self.message_history,
            "state": self.state,
            "session_id": self.session_id,
            "context": context or {}
        }
        
        # Execute workflow
        result = await self.client.execute_workflow(
            self.workflow_id,
            request_data,
            mode=WorkflowMode.CHAT
        )
        
        # Update state
        if result.success:
            self.state = result.metadata.get("new_state", self.state)
            self.message_history.append({
                "role": "assistant",
                "content": result.result.get("response", ""),
                "timestamp": datetime.utcnow().isoformat(),
                "agents_used": result.agents_used
            })
            self.metadata["message_count"] += 1
            
            # Track agent interactions
            for agent in result.agents_used:
                self.metadata["agent_interactions"][agent] = \
                    self.metadata["agent_interactions"].get(agent, 0) + 1
        
        return result
    
    async def reset(self):
        """Reset the chat session"""
        self.state = {}
        self.message_history = []
        self.metadata["message_count"] = 0
        self.metadata["agent_interactions"] = {}
        logger.info(f"Chat session {self.session_id} reset")
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.message_history.copy()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get session metrics"""
        return {
            "session_id": self.session_id,
            "workflow_id": self.workflow_id,
            "metadata": self.metadata,
            "state_size": len(json.dumps(self.state)),
            "total_messages": len(self.message_history)
        }

class WorkflowBuilder:
    """Interactive workflow builder"""
    
    def __init__(self, client: 'EnhancedROWBOATClient'):
        self.client = client
        self.current_workflow = {
            "name": "",
            "description": "",
            "agents": [],
            "edges": [],
            "metadata": {}
        }
    
    async def from_description(self, description: str) -> 'WorkflowBuilder':
        """Build workflow from natural language description"""
        
        response = await self.client.create_workflow(
            description=description,
            return_definition_only=True
        )
        
        self.current_workflow = response["definition"]
        return self
    
    def add_agent(
        self,
        name: str,
        role: str = "generic",
        instructions: str = "",
        tools: List[str] = None,
        **kwargs
    ) -> 'WorkflowBuilder':
        """Add an agent to the workflow"""
        
        agent = {
            "name": name,
            "role": role,
            "instructions": instructions,
            "tools": tools or [],
            **kwargs
        }
        
        self.current_workflow["agents"].append(agent)
        return self
    
    def connect(
        self,
        from_agent: str,
        to_agent: str,
        condition: Optional[str] = None
    ) -> 'WorkflowBuilder':
        """Connect two agents"""
        
        edge = {
            "from_agent": from_agent,
            "to_agent": to_agent
        }
        
        if condition:
            edge["condition"] = condition
        
        self.current_workflow["edges"].append(edge)
        return self
    
    def with_template(self, template_name: str) -> 'WorkflowBuilder':
        """Start with a template"""
        
        asyncio.create_task(self._load_template(template_name))
        return self
    
    async def _load_template(self, template_name: str):
        """Load a workflow template"""
        
        templates = await self.client.list_templates()
        template = templates.get(template_name)
        
        if template:
            self.current_workflow = template.copy()
        else:
            raise ValueError(f"Template '{template_name}' not found")
    
    async def build(self) -> str:
        """Build and deploy the workflow"""
        
        response = await self.client._request(
            "POST",
            "/workflows",
            json=self.current_workflow
        )
        
        return response["workflow_id"]
    
    def visualize(self) -> Dict[str, Any]:
        """Get visual representation of workflow"""
        
        return {
            "nodes": [
                {
                    "id": agent["name"],
                    "label": agent["name"],
                    "type": f"agent_{agent.get('role', 'generic')}"
                }
                for agent in self.current_workflow["agents"]
            ],
            "edges": [
                {
                    "source": edge["from_agent"],
                    "target": edge["to_agent"],
                    "label": edge.get("condition", "")
                }
                for edge in self.current_workflow["edges"]
            ]
        }

class EnhancedROWBOATClient:
    """Enhanced ROWBOAT SDK with full feature support"""
    
    def __init__(
        self,
        host: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        project_id: str = "default",
        chutes_api_key: Optional[str] = None,
        timeout: int = 300,
        enable_caching: bool = True
    ):
        self.host = host.rstrip('/')
        self.api_key = api_key or os.getenv("ROWBOAT_API_KEY")
        self.project_id = project_id
        self.timeout = timeout
        self.enable_caching = enable_caching
        
        # Set Chutes API key if provided
        if chutes_api_key:
            os.environ["CHUTES_API_KEY"] = chutes_api_key
        elif not os.getenv("CHUTES_API_KEY"):
            logger.warning("No Chutes API key found - some features may be limited")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json",
            "User-Agent": "ROWBOAT-SDK/2.0",
            "X-Project-ID": self.project_id
        }
        
        # Initialize client
        if HTTPX_AVAILABLE:
            self._client = httpx.AsyncClient(
                base_url=self.host,
                headers=self.headers,
                timeout=self.timeout
            )
        else:
            self._client = None
        
        # Cache for workflows and templates
        self._workflow_cache = {}
        self._template_cache = {}
        
        # WebSocket support for streaming
        self._ws_connections = {}
        
        logger.info(f"Enhanced ROWBOAT Client initialized for {self.host}")
    
    # Core Workflow Management
    
    async def create_workflow(
        self,
        description: str = None,
        definition: Dict[str, Any] = None,
        examples: Optional[List[Dict[str, str]]] = None,
        documents: Optional[List[str]] = None,
        approval_points: Optional[List[str]] = None,
        model_preferences: Optional[Dict[str, str]] = None,
        return_definition_only: bool = False
    ) -> Union[Dict[str, Any], str]:
        """Create a workflow from natural language or definition"""
        
        if description:
            # Use copilot to build from description
            request_data = {
                "description": description,
                "examples": examples or [],
                "documents": documents or [],
                "approval_points": approval_points or [],
                "model_preferences": model_preferences or {},
                "project_id": self.project_id
            }
            
            endpoint = "/copilot/build" if return_definition_only else "/workflows/create-from-description"
        else:
            # Direct definition
            request_data = definition
            endpoint = "/workflows"
        
        response = await self._request("POST", endpoint, json=request_data)
        
        if not return_definition_only:
            workflow_id = response.get("workflow_id")
            self._workflow_cache[workflow_id] = response
            return workflow_id
        
        return response
    
    async def list_workflows(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List all workflows with optional filters"""
        
        params = filters or {}
        params["project_id"] = self.project_id
        
        response = await self._request("GET", "/workflows", params=params)
        return response.get("workflows", [])
    
    async def get_workflow(self, workflow_id: str, use_cache: bool = True) -> Dict[str, Any]:
        """Get workflow details"""
        
        if use_cache and workflow_id in self._workflow_cache:
            return self._workflow_cache[workflow_id]
        
        response = await self._request("GET", f"/workflows/{workflow_id}")
        
        if self.enable_caching:
            self._workflow_cache[workflow_id] = response
        
        return response
    
    async def update_workflow(self, workflow_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update a workflow"""
        
        response = await self._request("PUT", f"/workflows/{workflow_id}", json=updates)
        
        # Invalidate cache
        if workflow_id in self._workflow_cache:
            del self._workflow_cache[workflow_id]
        
        return response
    
    async def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow"""
        
        response = await self._request("DELETE", f"/workflows/{workflow_id}")
        
        # Remove from cache
        if workflow_id in self._workflow_cache:
            del self._workflow_cache[workflow_id]
        
        return response.get("success", False)
    
    # Workflow Execution
    
    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: Union[str, Dict[str, Any]],
        mode: WorkflowMode = WorkflowMode.EXECUTE,
        stream: bool = False,
        timeout: Optional[int] = None
    ) -> Union[WorkflowResult, AsyncIterator[Dict[str, Any]]]:
        """Execute a workflow with various modes"""
        
        # Prepare request
        if isinstance(input_data, str):
            request_data = {"input": input_data}
        else:
            request_data = input_data
        
        request_data["mode"] = mode.value
        request_data["stream"] = stream
        
        if mode == WorkflowMode.CHAT:
            endpoint = f"/chat"
            request_data["workflow_id"] = workflow_id
        else:
            endpoint = f"/workflows/{workflow_id}/execute"
        
        # Handle streaming
        if stream:
            return self._stream_execution(endpoint, request_data, timeout)
        
        # Regular execution
        start_time = datetime.utcnow()
        response = await self._request(
            "POST",
            endpoint,
            json=request_data,
            timeout=timeout or self.timeout
        )
        
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        return WorkflowResult(
            success=response.get("success", False),
            workflow_id=workflow_id,
            execution_id=response.get("execution_id", ""),
            result=response.get("result", {}),
            duration_ms=duration_ms,
            agents_used=response.get("agents_used", []),
            metadata=response.get("metadata", {}),
            handoff_count=response.get("handoff_count", 0),
            mention_count=response.get("mention_count", 0)
        )
    
    async def _stream_execution(
        self,
        endpoint: str,
        request_data: Dict[str, Any],
        timeout: Optional[int] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream workflow execution events"""
        
        if HTTPX_AVAILABLE:
            async with self._client.stream(
                "POST",
                endpoint,
                json=request_data,
                timeout=timeout or self.timeout
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            event = json.loads(line[6:])
                            yield event
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse SSE event: {line}")
        else:
            raise NotImplementedError("Streaming requires httpx")
    
    # Stateful Chat Sessions
    
    def create_chat_session(self, workflow_id: str) -> StatefulChat:
        """Create a stateful chat session"""
        
        return StatefulChat(self, workflow_id)
    
    @asynccontextmanager
    async def chat_session(self, workflow_id: str):
        """Context manager for chat sessions"""
        
        session = self.create_chat_session(workflow_id)
        try:
            yield session
        finally:
            await session.reset()
    
    # Playground Testing
    
    async def test_workflow(
        self,
        workflow_id: str,
        scenario_id: Optional[str] = None,
        custom_input: Optional[str] = None,
        mock_responses: Optional[Dict[str, Any]] = None
    ) -> TestResult:
        """Test workflow in playground"""
        
        request_data = {
            "workflow_id": workflow_id,
            "scenario_id": scenario_id,
            "custom_input": custom_input,
            "mock_responses": mock_responses or {}
        }
        
        start_time = datetime.utcnow()
        
        try:
            response = await self._request(
                "POST",
                f"/workflows/{workflow_id}/test",
                json=request_data
            )
            
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return TestResult(
                scenario_id=scenario_id or "custom",
                passed=response.get("passed", False),
                execution_trace=response.get("trace", []),
                assertions=response.get("assertions", []),
                duration_ms=duration_ms,
                error=response.get("error")
            )
        except Exception as e:
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return TestResult(
                scenario_id=scenario_id or "custom",
                passed=False,
                execution_trace=[],
                assertions=[],
                duration_ms=duration_ms,
                error=str(e)
            )
    
    async def list_test_scenarios(self, workflow_id: str) -> List[Dict[str, Any]]:
        """List available test scenarios for a workflow"""
        
        response = await self._request(
            "GET",
            f"/workflows/{workflow_id}/test-scenarios"
        )
        
        return response.get("scenarios", [])
    
    # Template Management
    
    async def list_templates(self, category: Optional[str] = None) -> Dict[str, Any]:
        """List available workflow templates"""
        
        if self._template_cache and not category:
            return self._template_cache
        
        params = {"category": category} if category else {}
        response = await self._request("GET", "/templates", params=params)
        
        templates = response.get("templates", {})
        
        if not category and self.enable_caching:
            self._template_cache = templates
        
        return templates
    
    async def create_from_template(
        self,
        template_name: str,
        customizations: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create workflow from template"""
        
        request_data = {
            "template_name": template_name,
            "customizations": customizations or {}
        }
        
        response = await self._request(
            "POST",
            "/workflows/from-template",
            json=request_data
        )
        
        return response.get("workflow_id")
    
    # Visual Builder Integration
    
    async def get_visual_builder_config(self) -> Dict[str, Any]:
        """Get visual builder configuration"""
        
        response = await self._request("GET", "/visual-builder/config")
        return response
    
    async def save_visual_workflow(
        self,
        visual_data: Dict[str, Any],
        workflow_id: Optional[str] = None
    ) -> str:
        """Save workflow from visual builder"""
        
        endpoint = f"/workflows/{workflow_id}/visual" if workflow_id else "/workflows/visual"
        method = "PUT" if workflow_id else "POST"
        
        response = await self._request(method, endpoint, json=visual_data)
        return response.get("workflow_id")
    
    # Copilot Enhancement
    
    async def enhance_workflow(
        self,
        workflow_id: str,
        enhancement_type: str = "all"
    ) -> Dict[str, Any]:
        """Use copilot to enhance an existing workflow"""
        
        valid_types = ["tools", "routing", "error_handling", "performance", "security", "all"]
        
        if enhancement_type not in valid_types:
            raise ValueError(f"Invalid enhancement type. Must be one of: {valid_types}")
        
        request_data = {
            "workflow_id": workflow_id,
            "enhancement_type": enhancement_type
        }
        
        response = await self._request(
            "POST",
            f"/workflows/{workflow_id}/enhance",
            json=request_data
        )
        
        return response.get("enhancements", {})
    
    # Workflow Builder Helper
    
    def workflow_builder(self) -> WorkflowBuilder:
        """Get a workflow builder instance"""
        
        return WorkflowBuilder(self)
    
    # Batch Operations
    
    async def batch_execute(
        self,
        workflow_id: str,
        inputs: List[Union[str, Dict[str, Any]]],
        parallel: bool = True,
        max_concurrency: int = 5
    ) -> List[WorkflowResult]:
        """Execute workflow with multiple inputs"""
        
        if parallel:
            # Parallel execution with concurrency limit
            semaphore = asyncio.Semaphore(max_concurrency)
            
            async def execute_with_limit(input_data):
                async with semaphore:
                    return await self.execute_workflow(workflow_id, input_data)
            
            tasks = [execute_with_limit(inp) for inp in inputs]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Sequential execution
            results = []
            for inp in inputs:
                try:
                    result = await self.execute_workflow(workflow_id, inp)
                    results.append(result)
                except Exception as e:
                    results.append(e)
        
        return results
    
    # Monitoring and Analytics
    
    async def get_workflow_metrics(
        self,
        workflow_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get workflow execution metrics"""
        
        params = {
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None
        }
        
        response = await self._request(
            "GET",
            f"/workflows/{workflow_id}/metrics",
            params={k: v for k, v in params.items() if v is not None}
        )
        
        return response
    
    # Internal Request Handler
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to API"""
        
        url = f"{self.host}/api/v1{endpoint}"
        
        if HTTPX_AVAILABLE and self._client:
            response = await self._client.request(
                method,
                endpoint,
                json=json,
                params=params,
                timeout=timeout or self.timeout
            )
            response.raise_for_status()
            return response.json()
        else:
            # Fallback to requests
            response = requests.request(
                method,
                url,
                json=json,
                params=params,
                headers=self.headers,
                timeout=timeout or self.timeout
            )
            response.raise_for_status()
            return response.json()
    
    async def close(self):
        """Close the client connection"""
        
        if self._client and HTTPX_AVAILABLE:
            await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

# Convenience functions
async def quick_chat(workflow_id: str, message: str, **kwargs) -> str:
    """Quick one-off chat with a workflow"""
    
    async with EnhancedROWBOATClient(**kwargs) as client:
        session = client.create_chat_session(workflow_id)
        result = await session.send(message)
        return result.result.get("response", "")

async def build_and_test(description: str, test_input: str, **kwargs) -> Dict[str, Any]:
    """Build a workflow and immediately test it"""
    
    async with EnhancedROWBOATClient(**kwargs) as client:
        # Build workflow
        workflow_id = await client.create_workflow(description=description)
        
        # Test it
        test_result = await client.test_workflow(
            workflow_id,
            custom_input=test_input
        )
        
        return {
            "workflow_id": workflow_id,
            "test_result": test_result
        }