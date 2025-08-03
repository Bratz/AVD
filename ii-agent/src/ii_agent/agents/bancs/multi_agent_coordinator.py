# src/ii_agent/agents/bancs/rowboat_coordinator.py
"""
Enhanced ROWBOAT Multi-Agent Coordinator
AI-powered multi-agent workflow builder with visibility control and streaming
Integrates directly with ii-agent infrastructure without external dependencies
"""

from typing import Dict, Any, List, Optional, Union, AsyncGenerator, Tuple
import logging
import uuid
from datetime import datetime
from enum import Enum
from collections import defaultdict
import asyncio
import json
import re

from src.ii_agent.agents.base import BaseAgent
from src.ii_agent.agents.tcs_bancs_specialist_agent import TCSBancsSpecialistAgent
from src.ii_agent.llm.chutes_openai import ChutesOpenAIClient
from src.ii_agent.llm.model_registry import ChutesModelRegistry
from src.ii_agent.workflows.langgraph_integration import BaNCSLangGraphBridge
from src.ii_agent.workflows.definitions import WorkflowDefinition, AgentConfig, AgentRole, WorkflowEdge, EdgeConditionType
from src.ii_agent.copilot.workflow_builder import WorkflowCopilot

from src.ii_agent.config.rowboat_config import rowboat_config
from src.ii_agent.config.environment import env_config

from src.ii_agent.workflows.rowboat_streaming import StreamingWorkflowCreator


logger = logging.getLogger(__name__)

# ===== Type Definitions (Embedded to avoid import issues) =====

class OutputVisibility(Enum):
    """Agent output visibility control"""
    EXTERNAL = "user_facing"
    INTERNAL = "internal"

class ControlType(Enum):
    """Control flow after agent execution"""
    RETAIN = "retain"
    PARENT_AGENT = "relinquish_to_parent"
    START_AGENT = "start_agent"

class ResponseType(Enum):
    """Message response type"""
    INTERNAL = "internal"
    EXTERNAL = "external"

class StreamEventType(Enum):
    """Types of streaming events"""
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"
    AGENT_TRANSFER = "agent_transfer"
    CONTROL_TRANSITION = "control_transition"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    ERROR = "error"
    DONE = "done"

class WorkflowStatus(Enum):
    """Workflow execution status"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# ===== Main Coordinator Class =====

class ROWBOATCoordinator(TCSBancsSpecialistAgent):
    """Enhanced ROWBOAT coordinator for natural language multi-agent workflows"""
    
    @classmethod
    async def create(
        cls,
        client,
        tools,
        workspace_manager,
        message_queue,
        logger_for_agent_logs,
        context_manager,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Async factory method to create ROWBOATCoordinator with proper initialization.
        
        Args:
            client: LLM client
            tools: Available tools
            workspace_manager: Workspace manager
            message_queue: Message queue
            logger_for_agent_logs: Logger for agent logs
            context_manager: Context manager
            config: Additional configuration options
            **kwargs: Additional arguments for parent class
        
        Returns:
            ROWBOATCoordinator: Initialized coordinator instance
        """
        # Merge config with kwargs
        full_config = {
            **(config or {}),
            **kwargs
        }
        
        # Create instance using parent's create method
        instance = await super().create(
            client=client,
            tools=tools,
            workspace_manager=workspace_manager,
            message_queue=message_queue,
            logger_for_agent_logs=logger_for_agent_logs,
            context_manager=context_manager,
            user_role=full_config.get('user_role', 'coordinator'),
            mcp_wrapper=full_config.get('mcp_wrapper'),
            use_mcp_prompts=full_config.get('use_mcp_prompts', True),
            **kwargs
        )
        
        # Now initialize ROWBOAT-specific components
        await instance._initialize_rowboat_components(full_config)
        
        return instance
    
    def __init__(self, *args, **kwargs):
        """
        Direct initialization - should not be called directly.
        Use create() class method instead.
        """
        # This will be called by parent's __init__
        super().__init__(*args, **kwargs)
        
        # Initialize basic attributes that don't need async
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.turn_tracking: Dict[str, Dict[str, Any]] = {}
        self.workflow_metrics: Dict[str, Dict[str, Any]] = {}
        
    async def _initialize_rowboat_components(self, config: Dict[str, Any]):
        """Initialize ROWBOAT-specific components asynchronously"""
        
        self.config = config
        
        # Initialize ROWBOAT components
        self.workflow_engine = BaNCSLangGraphBridge()
        self.model_registry = ChutesModelRegistry
        
        # Get model configuration from rowboat_config
        model_config = rowboat_config.get_model_config()

        # Initialize Copilot using MODEL REGISTRY
        try:
            copilot_model_id = model_config["copilot_model"]
            logger.info(f"Attempting to initialize copilot with model: {copilot_model_id}")
            
            # Find model key in registry
            model_key = ChutesModelRegistry.get_model_key_by_id(copilot_model_id)
            
            if model_key:
                # Use registry with model key
                model_info = ChutesModelRegistry.AVAILABLE_MODELS[model_key]
                
                # Access ModelInfo attributes directly (not .get())
                supports_json = hasattr(model_info, 'supports_json_mode') and model_info.supports_json_mode
                
                copilot_llm = ChutesModelRegistry.create_llm_client(
                    model_key=model_key,
                    use_native_tools=True,
                    response_format={"type": "json_object"} if supports_json else None,
                    temperature=0.0
                )
                logger.info(f"Created copilot LLM with model key: {model_key}")
            else:
                # Use registry with model ID directly
                logger.info(f"Model key not found for {copilot_model_id}, using model ID directly")
                copilot_llm = ChutesModelRegistry.create_llm_client(
                    model_key=None,
                    model_id=copilot_model_id,
                    use_native_tools=True,
                    temperature=0.0
                )
            
            self.copilot = WorkflowCopilot(
                copilot_llm, 
                default_model=model_config["default_agent_model"]
            )
            logger.info(f"Initialized WorkflowCopilot with model: {copilot_model_id}")
            
        except Exception as e:
            logger.warning(f"Could not initialize Copilot with {model_config['copilot_model']}: {e}")
            
            # Fallback - find a suitable model from registry
            fallback_model_key = None
            fallback_model_id = None
            
            # Look for a model that supports structured output
            for key, model_info in ChutesModelRegistry.AVAILABLE_MODELS.items():
                if hasattr(model_info, 'model_id') and "deepseek" in model_info.model_id.lower() and "v3" in model_info.model_id.lower():
                    fallback_model_key = key
                    fallback_model_id = model_info.model_id
                    break
            
            # If no DeepSeek V3 found, use any model good for coordination
            if not fallback_model_key:
                fallback_model_id = ChutesModelRegistry.get_model_for_role("coordinator")
                fallback_model_key = ChutesModelRegistry.get_model_key_by_id(fallback_model_id)
            
            if fallback_model_key:
                copilot_llm = ChutesModelRegistry.create_llm_client(
                    model_key=fallback_model_key,
                    use_native_tools=True,
                    temperature=0.0
                )
            else:
                # Last resort - create with model ID
                copilot_llm = ChutesModelRegistry.create_llm_client(
                    model_key=None,
                    model_id=fallback_model_id or "gpt-4",
                    use_native_tools=True,
                    temperature=0.0
                )
            
            self.copilot = WorkflowCopilot(
                copilot_llm, 
                default_model=fallback_model_id or model_config["default_agent_model"]
            )
            logger.info(f"Initialized WorkflowCopilot with fallback model: {fallback_model_id}")
        
        # Store model config for agent creation
        self.model_config = model_config
        
        # ROWBOAT-specific configuration
        self.rowboat_config = {
            "enable_natural_language": True,
            "auto_generate_instructions": True,
            "use_best_practices": True,
            "enable_agent_mentions": True,
            "max_concurrent_workflows": 10,
            "workflow_timeout": 3600,
            # Week 1 features
            "enable_visibility_control": True,
            "enable_streaming": True,
            "enable_parent_child_limits": True,
            "enable_control_flow": True,
            "stream_internal_events": False
        }
        self.config.update(self.rowboat_config)
        
        logger.info("Enhanced ROWBOAT Multi-Agent Coordinator initialized")


    
    # ===== Core ROWBOAT Methods =====
    
    async def create_workflow_from_description(
        self,
        description: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create workflow from natural language description (ROWBOAT core feature)"""
        
        logger.info(f"Building workflow from description: {description}")
        
        try:
            # Use copilot to build workflow from description
            workflow_config = await self.copilot.build_from_description(
                description=description,
                user_context=user_context
            )
            
            # Extract @mentions and setup agent connections
            workflow_config = self._process_agent_mentions(workflow_config)
            
            # Add visibility and control flow defaults
            workflow_config = self._add_rowboat_defaults(workflow_config)
            
            # Convert to WorkflowDefinition
            workflow_definition = self._convert_to_workflow_definition(workflow_config)
            
            # Create and register workflow
            return await self.create_workflow(workflow_definition, user_context)
            
        except Exception as e:
            logger.error(f"Error creating workflow from description: {e}")
            raise
    
    async def create_workflow(
        self,
        workflow_definition: Union[WorkflowDefinition, Dict[str, Any]],
        user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create and register a new workflow"""
        
        # Convert dict to WorkflowDefinition if needed
        if isinstance(workflow_definition, dict):
            workflow_definition = WorkflowDefinition(**workflow_definition)
        
        # Generate workflow ID
        workflow_id = str(uuid.uuid4())
        
        # Create LangGraph workflow
        langgraph_workflow = self.workflow_engine.create_workflow(
            workflow_definition.dict()
        )
        
        # Initialize workflow metrics
        self.workflow_metrics[workflow_id] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_duration_ms": 0.0,
            "average_handoffs": 0.0,
            "agent_usage": defaultdict(int)
        }
        
        # Register workflow
        self.active_workflows[workflow_id] = {
            "id": workflow_id,
            "definition": workflow_definition,
            "langgraph": langgraph_workflow,
            "status": WorkflowStatus.INITIALIZING,
            "created_at": datetime.utcnow(),
            "user_context": user_context or {},
            "execution_history": [],
            "rowboat_metadata": {
                "has_mentions": self._has_agent_mentions(workflow_definition),
                "agent_count": len(workflow_definition.agents),
                "complexity": self._estimate_complexity(workflow_definition),
                "visibility_summary": self._get_visibility_summary(workflow_definition)
            }
        }
        
        logger.info(f"Created ROWBOAT workflow {workflow_id} with {len(workflow_definition.agents)} agents")
        
        return workflow_id
    
    # ===== Enhanced Execution Methods =====
    
    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: Dict[str, Any],
        stream_events: bool = True
    ) -> Dict[str, Any]:
        """Execute workflow with optional streaming"""
        # Add logging at the VERY FIRST LINE
        logger.info(f"=== COORDINATOR EXECUTE_WORKFLOW START ===")
        logger.info(f"Workflow ID requested: {workflow_id}")
        logger.info(f"Stream events: {stream_events}")
        
        try:
            # Log current state
            logger.info(f"self.active_workflows exists: {hasattr(self, 'active_workflows')}")
            if hasattr(self, 'active_workflows'):
                logger.info(f"Active workflows count: {len(self.active_workflows)}")
                logger.info(f"Active workflow IDs: {list(self.active_workflows.keys())}")
            else:
                logger.error("self.active_workflows attribute not found!")
                self.active_workflows = {}  # Initialize if missing
            
            # Check if workflow exists
            if workflow_id not in self.active_workflows:
                logger.error(f"Workflow {workflow_id} not found in active_workflows")
                logger.error(f"Available workflows: {list(self.active_workflows.keys())}")
                raise ValueError(f"Workflow {workflow_id} not found")
            
            workflow_info = self.active_workflows[workflow_id]
            logger.info(f"Found workflow in active_workflows")
            logger.info(f"Workflow status: {workflow_info.get('status')}")
            logger.info(f"Workflow has definition: {bool(workflow_info.get('definition'))}")
            logger.info(f"Workflow has langgraph: {bool(workflow_info.get('langgraph'))}")

            self._initialize_workflow_metrics(workflow_id)
            execution_id = str(uuid.uuid4())
            logger.info(f"Generated execution_id: {execution_id}")

            logger.info("About to track execution start...")
            self._track_execution_start(workflow_id, execution_id)
            logger.info("Execution start tracked successfully")

            logger.info(f"Checking execution mode: stream_events={stream_events}, enable_streaming={self.config.get('enable_streaming', True)}")

            if stream_events and self.config.get("enable_streaming", True):
                # Use streaming execution
                # self._track_agent_invocation(workflow_id, execution_id, agent_name)
                return await self._execute_workflow_streamed(workflow_id, input_data)
            else:
                # Use standard execution (original behavior)
                # self._track_agent_invocation(workflow_id, execution_id, agent_name)
                return await self._execute_workflow_standard(workflow_id, input_data)

            # self._track_execution_complete(workflow_id, execution_id, success=True)
        except Exception as e:
            logger.error(f"Error in execute_workflow: {type(e).__name__}: {str(e)}")
            logger.error(f"Traceback:", exc_info=True)
            raise

    async def _execute_workflow_standard(
        self,
        workflow_id: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Standard workflow execution (original behavior)"""
            
        logger.info(f"=== _execute_workflow_standard START ===")
        logger.info(f"Workflow ID: {workflow_id}")
        logger.info(f"Input data keys: {list(input_data.keys())}")
        
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow_info = self.active_workflows[workflow_id]
        logger.info(f"Got workflow_info, status: {workflow_info.get('status')}")
        
        # Check concurrent workflow limit
        running_count = sum(
            1 for w in self.active_workflows.values()
            if w["status"] == WorkflowStatus.RUNNING
        )
        logger.info(f"Running workflows count: {running_count}, max allowed: {self.config['max_concurrent_workflows']}")
        
        if running_count >= self.config["max_concurrent_workflows"]:
            raise RuntimeError(f"Maximum concurrent workflows ({self.config['max_concurrent_workflows']}) reached")
        
        # Update status
        workflow_info["status"] = WorkflowStatus.RUNNING
        workflow_info["started_at"] = datetime.utcnow()
        logger.info("Updated workflow status to RUNNING")
        
        # Log the preparation steps
        logger.info("Preparing execution context...")
        logger.info(f"Workspace manager exists: {self.workspace_manager is not None}")
        logger.info(f"User context exists: {'user_context' in workflow_info}")
        
        # Prepare execution context with ROWBOAT features
        execution_context = {
            "workspace": self.workspace_manager,
            "permissions": workflow_info.get("user_context", {}).get("permissions", {}),  # Add get() for safety
            "session_id": self.session_id,
            "llm_provider": "chutes",
            "workflow_id": workflow_id,
            "rowboat_features": {
                "mention_tracking": True,
                "auto_handoff": True,
                "instruction_context": workflow_info["definition"].metadata,
                "visibility_control": self.config.get("enable_visibility_control", True),
                "parent_child_limits": self.config.get("enable_parent_child_limits", True)
            }
        }
        logger.info("Execution context prepared successfully")
        
        try:
            # Log initial state preparation
            logger.info("Preparing initial state...")
            logger.info(f"Entry point: {workflow_info['definition'].entry_point}")
            
            # Execute workflow through LangGraph
            initial_state = {
                "messages": [{"role": "user", "content": input_data.get("message", "")}],
                "current_agent": workflow_info["definition"].entry_point,
                "agent_outputs": {},
                "workflow_metadata": {
                    **input_data.get("metadata", {}),
                    "rowboat_enabled": True
                },
                "mention_history": [],  # Track @mentions
                "handoff_log": []      # Track agent handoffs
            }
            logger.info(f"Initial state prepared with message: {input_data.get('message', '')[:50]}...")
            
            # Extract thread_id from input metadata
            thread_id = input_data.get("metadata", {}).get("thread_id", f"thread_{uuid.uuid4().hex[:8]}")
            logger.info(f"Thread ID: {thread_id}")
            
            # Check langgraph
            logger.info(f"Langgraph type: {type(workflow_info.get('langgraph'))}")
            logger.info(f"Langgraph has ainvoke: {hasattr(workflow_info.get('langgraph'), 'ainvoke')}")
            
            # Execute workflow through LangGraph with checkpointer config
            logger.info("About to call langgraph.ainvoke...")
            
            result = await workflow_info["langgraph"].ainvoke(
                initial_state,
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": "default"
                    }
                }
            )
            
            logger.info("Langgraph.ainvoke completed successfully")
            logger.info(f"Result type: {type(result)}")
            logger.info(f"Result content: {result}")
            
            # Check if result is AddableValuesDict and convert it
            if hasattr(result, '__class__') and result.__class__.__name__ == 'AddableValuesDict':
                logger.info("Converting AddableValuesDict to regular dict")
                result = dict(result)
                logger.info(f"Converted result keys: {list(result.keys())}")
            
            # Update workflow info
            workflow_info["status"] = WorkflowStatus.COMPLETED
            workflow_info["completed_at"] = datetime.utcnow()
            workflow_info["result"] = result
            
            # Add execution record
            execution_record = {
                "execution_id": str(uuid.uuid4()),
                "input": input_data,
                "output": result,
                "started_at": workflow_info["started_at"],
                "completed_at": workflow_info["completed_at"],
                "duration_ms": int((workflow_info["completed_at"] - workflow_info["started_at"]).total_seconds() * 1000),
                "mention_count": len(result.get("mention_history", [])),
                "handoff_count": len(result.get("handoff_log", []))
            }
            workflow_info["execution_history"].append(execution_record)
            
            # Update metrics
            self._update_workflow_metrics(workflow_id, execution_record)
            
            logger.info(f"ROWBOAT workflow {workflow_id} completed with {execution_record['handoff_count']} handoffs")
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "result": result,
                "execution_metrics": {
                    "duration_ms": execution_record["duration_ms"],
                    "agents_involved": len(set(h["to_agent"] for h in result.get("handoff_log", []))),
                    "mention_based_handoffs": execution_record["mention_count"]
                }
            }
            
        except Exception as e:
            logger.error(f"Error in try block of _execute_workflow_standard")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error(f"Full traceback:", exc_info=True)
            
            workflow_info["status"] = WorkflowStatus.FAILED
            workflow_info["error"] = str(e)
            workflow_info["failed_at"] = datetime.utcnow()
            
            # Still log the full error before returning
            logger.error(f"ROWBOAT workflow {workflow_id} failed: {e}")
            
            return {
                "success": False,
                "workflow_id": workflow_id,
                "error": str(e)
            }
    
    async def _execute_workflow_streamed(
        self,
        workflow_id: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute workflow with streaming support"""
        
        events = []
        final_result = None
        
        # Collect events from streaming execution
        async for event_type, event_data in self.execute_workflow_with_streaming(
            workflow_id, input_data
        ):
            events.append((event_type, event_data))
            
            if event_type == "done":
                final_result = event_data
        
        # Return aggregated result
        return {
            "success": True,
            "workflow_id": workflow_id,
            "result": final_result,
            "events": events,
            "execution_metrics": self._extract_metrics_from_events(events)
        }
    
    async def create_workflow_with_streaming(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Create workflow from description with streaming
        Compatible with Rowboat copilot service
        """
        creator = StreamingWorkflowCreator(self)
        async for event in creator.create_workflow_streaming(description, context):
            yield event

    async def execute_workflow_with_streaming(
        self,
        workflow_id: str,
        input_data: Dict[str, Any]
    ) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
        """Execute workflow with streaming events (returns tuples for compatibility)"""
        
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow_info = self.active_workflows[workflow_id]
        
        # Initialize turn tracking
        turn_id = str(uuid.uuid4())
        turn_metrics = {
            "turn_id": turn_id,
            "workflow_id": workflow_id,
            "start_time": datetime.utcnow(),
            "agents_involved": [],
            "message_count": 0,
            "internal_message_count": 0,
            "external_message_count": 0,
            "handoff_count": 0,
            "error_count": 0
        }
        
        self.turn_tracking[turn_id] = {
            "workflow_id": workflow_id,
            "messages": [],
            "current_agent": workflow_info["definition"].entry_point,
            "parent_stack": [],
            "agent_message_counts": defaultdict(int),
            "child_call_counts": defaultdict(lambda: defaultdict(int)),
            "internal_messages": [],
            "external_message_sent": False,
            "metrics": turn_metrics
        }
        
        try:
            # Yield turn start event
            yield ("turn_start", {
                "workflow_id": workflow_id,
                "turn_id": turn_id,
                "input": input_data
            })
            
            # Execute turn with streaming
            async for event_type, event_data in self._execute_turn_streamed(
                workflow_info,
                self.turn_tracking[turn_id],
                input_data
            ):
                yield (event_type, event_data)
            
            # Update metrics
            turn_metrics["end_time"] = datetime.utcnow()
            turn_metrics["duration_ms"] = int(
                (turn_metrics["end_time"] - turn_metrics["start_time"]).total_seconds() * 1000
            )
            
            # Yield turn end event
            yield ("turn_end", {
                "workflow_id": workflow_id,
                "turn_id": turn_id,
                "metrics": turn_metrics
            })
            
            # Yield done event
            yield ("done", {
                "state": {
                    "last_agent_name": self.turn_tracking[turn_id]["current_agent"],
                    "tokens": {},  # TODO: Add token tracking
                    "turn_messages": self.turn_tracking[turn_id]["messages"]
                }
            })
            
        except Exception as e:
            turn_metrics["error_count"] += 1
            yield ("error", {"error": str(e), "workflow_id": workflow_id})
            raise
        
        finally:
            # Cleanup turn tracking
            if turn_id in self.turn_tracking:
                del self.turn_tracking[turn_id]
    
    async def _execute_turn_streamed(
        self,
        workflow_info: Dict[str, Any],
        turn_state: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
        """Execute a single turn with ROWBOAT visibility and control flow rules"""
        
        # Prepare initial messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_data.get("message", "")}
        ]
        turn_state["messages"] = messages
        
        # Create enhanced agents
        agents = await self._create_enhanced_agents(workflow_info)
        agents_by_name = {agent.name: agent for agent in agents}
        
        # Get initial agent
        current_agent_name = turn_state["current_agent"]
        current_agent = agents_by_name.get(current_agent_name)
        
        if not current_agent:
            raise ValueError(f"Agent {current_agent_name} not found")
        
        # Main execution loop
        iteration = 0
        max_iterations = 50  # Safety limit
        
        while not turn_state["external_message_sent"] and iteration < max_iterations:
            iteration += 1
            
            logger.debug(f"Turn iteration {iteration}: Agent={current_agent_name}")
            
            # Check parent-child limits
            if not self._check_parent_child_limits(turn_state, current_agent):
                logger.info(f"Parent-child limit exceeded for {current_agent_name}")
                
                # Handle control flow back to parent
                if turn_state["parent_stack"]:
                    parent_name = turn_state["parent_stack"].pop()
                    
                    yield ("control_transition", {
                        "from": current_agent_name,
                        "to": parent_name,
                        "reason": "max_calls_exceeded"
                    })
                    
                    current_agent_name = parent_name
                    current_agent = agents_by_name[parent_name]
                    continue
                else:
                    break
            
            # Execute agent (simplified for now)
            agent_response = await self._execute_agent_simplified(
                current_agent, turn_state, workflow_info
            )
            
            # Update metrics
            turn_metrics = turn_state["metrics"]
            turn_metrics["agents_involved"].append(current_agent_name)
            turn_metrics["message_count"] += 1
            
            # Determine visibility
            is_internal = self._is_internal_agent(current_agent)
            if is_internal:
                turn_metrics["internal_message_count"] += 1
            else:
                turn_metrics["external_message_count"] += 1
            
            # Stream message based on visibility
            if self._should_stream_agent_output(current_agent):
                yield ("message", {
                    "content": agent_response["content"],
                    "role": "assistant",
                    "sender": current_agent.name,
                    "response_type": ResponseType.INTERNAL.value if is_internal else ResponseType.EXTERNAL.value
                })
            
            # Track message output
            turn_state["agent_message_counts"][current_agent_name] += 1
            turn_state["messages"].append({
                "role": "assistant",
                "content": agent_response["content"],
                "sender": current_agent.name
            })
            
            # Handle agent transfers (@mentions)
            if "@" in agent_response["content"]:
                transfer_result = self._handle_agent_transfer(
                    agent_response["content"],
                    current_agent,
                    agents_by_name,
                    turn_state
                )
                
                if transfer_result:
                    new_agent_name = transfer_result["target_agent"]
                    turn_metrics["handoff_count"] += 1
                    
                    # Update parent stack for internal agents
                    if self._is_internal_agent(agents_by_name[new_agent_name]):
                        turn_state["parent_stack"].append(current_agent_name)
                    
                    # Track parent-child call
                    turn_state["child_call_counts"][current_agent_name][new_agent_name] += 1
                    
                    yield ("agent_transfer", {
                        "from": current_agent_name,
                        "to": new_agent_name,
                        "reason": "mention"
                    })
                    
                    current_agent_name = new_agent_name
                    current_agent = agents_by_name[new_agent_name]
                    continue
            
            # Check if external message was sent
            if not is_internal:
                turn_state["external_message_sent"] = True
                break
            else:
                # Internal agent - handle control flow
                control_transition = self._handle_control_flow(
                    current_agent, turn_state, workflow_info
                )
                
                if control_transition:
                    yield ("control_transition", control_transition)
                    
                    if control_transition["to"]:
                        current_agent_name = control_transition["to"]
                        current_agent = agents_by_name[current_agent_name]
    
    # ===== Agent Creation and Enhancement =====
    
    async def _create_enhanced_agents(self, workflow_info: Dict[str, Any]) -> List[BaseAgent]:
        """Create agents with ROWBOAT enhancements"""
        
        agents = []
        shared_context = {
            "workspace": self.workspace_manager,
            "session_id": self.session_id,
            "workflow_id": workflow_info["id"],
            "rowboat_features": {
                "visibility_control": self.config.get("enable_visibility_control", True),
                "parent_child_limits": self.config.get("enable_parent_child_limits", True)
            }
        }
        
        for agent_config in workflow_info["definition"].agents:
            # Create specialized agent - NOW AWAIT IT!
            agent = await self._create_specialized_agent(agent_config, shared_context)
            
            # Add ROWBOAT-specific attributes
            agent.output_visibility = OutputVisibility(
                agent_config.metadata.get("output_visibility", OutputVisibility.EXTERNAL.value)
            )
            agent.control_type = ControlType(
                agent_config.metadata.get("control_type", ControlType.RETAIN.value)
            )
            agent.max_calls_per_parent_agent = agent_config.metadata.get(
                "max_calls_per_parent_agent", 5
            )
            agent.connected_agents = agent_config.metadata.get("connected_agents", [])
            
            agents.append(agent)
        
        return agents    
    # def _create_specialized_agent(
    #     self,
    #     agent_config: AgentConfig,
    #     shared_context: Dict[str, Any]
    # ) -> BaseAgent:
    #     """Create agent with ROWBOAT-enhanced instructions"""
        
    #     # Get appropriate Chutes model
    #     llm_client = self._get_llm_client_for_agent(agent_config)
        
    #     # Enhance instructions with ROWBOAT patterns
    #     enhanced_instructions = self._enhance_agent_instructions(agent_config)
        
    #     # Build agent configuration
    #     config = {
    #         "name": agent_config.name,
    #         "role": agent_config.role,
    #         "instructions": enhanced_instructions,
    #         "llm_client": llm_client,
    #         "temperature": agent_config.temperature,
    #         "tools": self._resolve_tools(agent_config.tools, agent_config.mcp_servers),
    #         "shared_context": shared_context,
    #         "workspace_manager": self.workspace_manager,
    #         "message_queue": self.message_queue,
    #         "logger_for_agent_logs": logger,
    #         "context_manager": self.context_manager,
    #         "metadata": {
    #             **agent_config.metadata,
    #             "rowboat_enhanced": True
    #         }
    #     }
        
    #     return BaseAgent(**config)

    async def _create_specialized_agent(
    self,
    agent_config: AgentConfig,
    shared_context: Dict[str, Any]
) -> BaseAgent:
        """Create agent with ROWBOAT-enhanced instructions - properly using TCSBancsSpecialistAgent"""
        
        # Get appropriate Chutes model
        llm_client = self._get_llm_client_for_agent(agent_config)
        
        # Enhance instructions with ROWBOAT patterns
        enhanced_instructions = self._enhance_agent_instructions(agent_config)
        
        # Create agent using TCSBancsSpecialistAgent.create() factory method
        agent = await TCSBancsSpecialistAgent.create(
            client=llm_client,
            # tools=[],  # Tools will be resolved and added later
            tools=self._resolve_tools(agent_config.tools, agent_config.mcp_servers),
            workspace_manager=self.workspace_manager,
            message_queue=self.message_queue,
            logger_for_agent_logs=logger,
            context_manager=self.context_manager,
            user_role=agent_config.role.value if hasattr(agent_config.role, 'value') else str(agent_config.role),
            mcp_wrapper=self.config.get("mcp_wrapper"),
            use_mcp_prompts=False,  # We're using custom instructions
            instructions=enhanced_instructions  # Use our enhanced instructions as the prompt
        )
        
        # Set agent-specific attributes after creation
        agent.name = agent_config.name
        agent.description = agent_config.description
        agent.instructions = enhanced_instructions
        agent.temperature = agent_config.temperature
        
        # Add resolved tools
        if agent_config.tools:
            resolved_tools = self._resolve_tools(agent_config.tools, agent_config.mcp_servers)
            agent.tools.extend(resolved_tools)
        
        # Add ROWBOAT metadata
        agent.metadata = {
            **agent_config.metadata,
            "rowboat_enhanced": True
        }
        
        return agent    

    def _enhance_agent_instructions(self, agent_config: AgentConfig) -> str:
        """Enhance instructions with ROWBOAT best practices"""
        
        base_instructions = agent_config.instructions
        
        # Add @mention handling
        if agent_config.metadata.get("connected_agents"):
            mention_instructions = "\n\nWhen you need assistance or handoff:"
            for connected in agent_config.metadata["connected_agents"]:
                mention_instructions += f"\n- Use @{connected} for handoff to {connected}"
            base_instructions += mention_instructions
        
        # Add role-specific best practices
        if agent_config.role == AgentRole.RESEARCHER:
            base_instructions += "\n\nBest practices:\n- Cite sources\n- Verify facts\n- Be thorough"
        elif agent_config.role == AgentRole.WRITER:
            base_instructions += "\n\nBest practices:\n- Clear structure\n- Engaging tone\n- Proper formatting"
        elif agent_config.role == AgentRole.ANALYZER:
            base_instructions += "\n\nBest practices:\n- Be systematic\n- Provide evidence\n- Clear conclusions"
        
        return base_instructions
    
    def _get_llm_client_for_agent(self, agent_config: AgentConfig) -> Any:
        """Get appropriate Chutes LLM client for agent using MODEL REGISTRY"""
        
        model_id = agent_config.model or rowboat_config.get_model_for_role(agent_config.role.value)
        
        # Check if it's a model key
        if model_id in ChutesModelRegistry.AVAILABLE_MODELS:
            return ChutesModelRegistry.create_llm_client(
                model_key=model_id,
                use_native_tools=True,
                temperature=agent_config.temperature
            )
        
        # Try to find by model ID
        model_key = ChutesModelRegistry.get_model_key_by_id(model_id)
        if model_key:
            return ChutesModelRegistry.create_llm_client(
                model_key=model_key,
                use_native_tools=True,
                temperature=agent_config.temperature
            )
        
        # Fallback: create with model ID directly
        logger.warning(f"Model {model_id} not found in registry, creating directly")
        return ChutesModelRegistry.create_llm_client(
            model_key=None,
            model_id=model_id,
            use_native_tools=True,
            temperature=agent_config.temperature
        )    
    def _resolve_tools(self, native_tools: List[str], mcp_servers: List[str]) -> List[Any]:
        """Resolve tools (simplified without MCP for now)"""
        resolved_tools = []
        
        # Resolve native tools from tool manager if available
        if hasattr(self, 'tool_manager'):
            for tool_name in native_tools:
                tool = self.tool_manager.get_tool(tool_name)
                if tool:
                    resolved_tools.append(tool)
        
        return resolved_tools
    
    # ===== Simplified Agent Execution =====
    
    async def _execute_agent_simplified(
        self,
        agent: BaseAgent,
        turn_state: Dict[str, Any],
        workflow_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simplified agent execution for demonstration"""
        
        # This is a simplified version - in production, this would call
        # the actual agent execution through LangGraph
        
        # Simulate processing delay
        await asyncio.sleep(0.1)
        
        # Generate response based on agent type and visibility
        if self._is_internal_agent(agent):
            content = f"[Internal processing by {agent.name}] Processing request..."
        else:
            content = f"I understand your request. Let me help you with that."
        
        # Check for @mentions in connected agents
        if agent.connected_agents and len(turn_state["messages"]) > 2:
            # Simulate handoff decision
            for connected in agent.connected_agents[:1]:  # Just use first connected agent
                if turn_state["agent_message_counts"][agent.name] == 0:
                    content += f" @{connected}"
                    break
        
        return {
            "content": content,
            "agent": agent.name
        }
    
    # ===== Control Flow and Visibility Methods =====
    
    def _check_parent_child_limits(
        self,
        turn_state: Dict[str, Any],
        child_agent: BaseAgent
    ) -> bool:
        """Check if parent-child call limit is exceeded"""
        
        if not self.config.get("enable_parent_child_limits", True):
            return True
        
        if not turn_state["parent_stack"]:
            return True
        
        parent_name = turn_state["parent_stack"][-1]
        child_name = child_agent.name
        
        current_calls = turn_state["child_call_counts"][parent_name][child_name]
        max_calls = child_agent.max_calls_per_parent_agent
        
        return current_calls < max_calls
    
    def _handle_agent_transfer(
        self,
        content: str,
        current_agent: BaseAgent,
        agents_by_name: Dict[str, BaseAgent],
        turn_state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Handle @mention-based agent transfers"""
        
        mentions = re.findall(r'@(\w+)', content)
        
        for mention in mentions:
            if mention in agents_by_name and mention in current_agent.connected_agents:
                return {"target_agent": mention}
        
        return None
    
    def _handle_control_flow(
        self,
        agent: BaseAgent,
        turn_state: Dict[str, Any],
        workflow_info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Handle control flow based on agent's control type"""
        
        if not self.config.get("enable_control_flow", True):
            return None
        
        control_type = agent.control_type
        
        if control_type == ControlType.PARENT_AGENT:
            # Return to parent
            if turn_state["parent_stack"]:
                parent_name = turn_state["parent_stack"].pop()
                return {
                    "from": agent.name,
                    "to": parent_name,
                    "control_type": "relinquish_to_parent"
                }
        
        elif control_type == ControlType.START_AGENT:
            # Return to start agent
            start_agent = workflow_info["definition"].entry_point
            turn_state["parent_stack"].clear()
            return {
                "from": agent.name,
                "to": start_agent,
                "control_type": "start_agent"
            }
        
        return None
    
    def _is_internal_agent(self, agent: BaseAgent) -> bool:
        """Check if agent has internal visibility"""
        return getattr(agent, "output_visibility", OutputVisibility.EXTERNAL) == OutputVisibility.INTERNAL
    
    def _should_stream_agent_output(self, agent: BaseAgent) -> bool:
        """Determine if agent output should be streamed"""
        
        # Always stream external agent outputs
        if not self._is_internal_agent(agent):
            return True
        
        # Stream internal outputs only if configured
        return self.config.get("stream_internal_events", False)
    
    # ===== Helper Methods =====
    
    # In multi_agent_coordinator.py, update the _process_agent_mentions method:

    def _process_agent_mentions(self, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process @mentions in agent instructions for handoffs"""
        
        agent_names = {agent["name"] for agent in workflow_config.get("agents", [])}
        
        for agent in workflow_config.get("agents", []):
            instructions = agent.get("instructions", "")
            agent_name = agent.get("name", "")
            
            # Skip if no name
            if not agent_name:
                logger.warning("Agent without name found, skipping mention processing")
                continue
            
            # Find @mentions in instructions
            mentions = re.findall(r'@(\w+)', instructions)
            
            # Validate mentions
            valid_mentions = []
            for mention in mentions:
                if mention in agent_names:
                    valid_mentions.append(mention)
                else:
                    logger.warning(f"Invalid @mention '{mention}' in agent '{agent_name}'")
            
            # Update connected_agents if not already set
            if "connected_agents" not in agent:
                agent["connected_agents"] = valid_mentions
            else:
                # Merge with existing connected_agents
                existing = set(agent.get("connected_agents", []))
                agent["connected_agents"] = list(existing.union(set(valid_mentions)))
            
            # Add handoff logic to edges with CORRECT key names
            for mention in valid_mentions:
                # Check if edge already exists (check both old and new format)
                edge_exists = any(
                    (e.get("from_agent") == agent_name and e.get("to_agent") == mention) or
                    (e.get("from") == agent_name and e.get("to") == mention)
                    for e in workflow_config.get("edges", [])
                )
                
                if not edge_exists:
                    # Use CORRECT key names: from_agent and to_agent
                    workflow_config.setdefault("edges", []).append({
                        "from_agent": agent_name,      # Correct key name
                        "to_agent": mention,           # Correct key name
                        "condition_type": "mention_based",
                        "condition": f"@{mention} mentioned"
                    })
        
        return workflow_config
    
    def _add_rowboat_defaults(self, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """Add ROWBOAT-specific defaults to workflow configuration"""
        
        for agent in workflow_config.get("agents", []):
            # Add visibility defaults
            if "outputVisibility" not in agent:
                # Determine based on role or name
                if any(keyword in agent.get("name", "").lower() 
                      for keyword in ["internal", "process", "analyze"]):
                    agent["outputVisibility"] = OutputVisibility.INTERNAL.value
                else:
                    agent["outputVisibility"] = OutputVisibility.EXTERNAL.value
            
            # Add control type defaults
            if "controlType" not in agent:
                if agent.get("connected_agents"):
                    agent["controlType"] = ControlType.PARENT_AGENT.value
                else:
                    agent["controlType"] = ControlType.RETAIN.value
            
            # Add parent-child limits
            if "maxCallsPerParentAgent" not in agent:
                agent["maxCallsPerParentAgent"] = 5
        
        return workflow_config
    
    def _convert_to_workflow_definition(self, workflow_config: Dict[str, Any]) -> WorkflowDefinition:
        """Convert copilot output to WorkflowDefinition"""
        
        # Process agent mentions to extract edges if needed
        workflow_config = self._process_agent_mentions(workflow_config)
        
        # Add ROWBOAT defaults
        workflow_config = self._add_rowboat_defaults(workflow_config)
        
        # Create AgentConfig objects
        agents = []
        for agent_data in workflow_config.get("agents", []):
            # Determine role from agent data
            role = self._determine_agent_role(agent_data)
            
            # Extract metadata from agent data
            metadata = {
                "output_visibility": agent_data.get("outputVisibility", OutputVisibility.EXTERNAL.value),
                "control_type": agent_data.get("controlType", ControlType.RETAIN.value),
                "connected_agents": agent_data.get("connected_agents", []),
                "examples": agent_data.get("examples", []),
                "max_calls_per_parent_agent": agent_data.get("maxCallsPerParentAgent", 5)
            }
            
            agent_config = AgentConfig(
                name=agent_data["name"],
                role=role,
                description=agent_data.get("description", ""),
                instructions=agent_data.get("instructions", ""),
                model=agent_data.get("model"),
                temperature=agent_data.get("temperature", 0.7),
                tools=agent_data.get("tools", []),
                mcp_servers=agent_data.get("mcp_servers", []),
                custom_prompts=agent_data.get("custom_prompts", {}),
                metadata=metadata,
                # Add ROWBOAT fields directly to AgentConfig
                output_visibility=OutputVisibility(agent_data.get("outputVisibility", OutputVisibility.EXTERNAL.value)),
                control_type=ControlType(agent_data.get("controlType", ControlType.RETAIN.value)),
                connected_agents=agent_data.get("connected_agents", []),
                examples=agent_data.get("examples", []),
                max_calls_per_parent_agent=agent_data.get("maxCallsPerParentAgent", 5)
            )
            agents.append(agent_config)
        
        # Create WorkflowEdge objects
        edges = []
        for edge_data in workflow_config.get("edges", []):
            edge = WorkflowEdge(
                from_agent=edge_data.get("from_agent"),
                to_agent=edge_data.get("to_agent"),
                condition_type=EdgeConditionType(edge_data.get("condition_type", EdgeConditionType.ALWAYS.value)),
                condition=edge_data.get("condition"),
                to_agents=edge_data.get("to_agents"),
                metadata=edge_data.get("metadata", {})
            )
            edges.append(edge)
        
        # Handle startAgent vs entry_point
        entry_point = workflow_config.get("entry_point") or workflow_config.get("startAgent")
        if not entry_point and agents:
            entry_point = agents[0].name
        
        # Create WorkflowDefinition
        return WorkflowDefinition(
            name=workflow_config.get("name", "ROWBOAT Workflow"),
            description=workflow_config.get("description", f"AI-generated workflow from: {workflow_config.get('original_description', 'user request')}"),
            version=workflow_config.get("version", "1.0.0"),
            agents=agents,
            edges=edges,
            entry_point=entry_point,
            metadata={
                "generated_by": "ROWBOAT",
                "original_description": workflow_config.get("original_description", ""),
                "version": "2.0"
            }
        )
    
    def _determine_agent_role(self, agent_data: Dict[str, Any]) -> AgentRole:
        """Determine agent role from description/name"""
        
        name_lower = agent_data.get("name", "").lower()
        role_str = agent_data.get("role", "").lower()
        description_lower = agent_data.get("description", "").lower()
        
        # Check all text for role patterns
        combined_text = f"{name_lower} {role_str} {description_lower}"
        
        # Map patterns to roles
        role_mapping = {
            "research": AgentRole.RESEARCHER,
            "analyz": AgentRole.ANALYZER,  # catches analyze, analyzer, analysis
            "writ": AgentRole.WRITER,      # catches write, writer, writing
            "review": AgentRole.REVIEWER,
            "code": AgentRole.CODER,
            "test": AgentRole.TESTER,
            "coordinat": AgentRole.COORDINATOR,  # catches coordinate, coordinator
            "support": AgentRole.CUSTOMER_SUPPORT,
            "custom": AgentRole.CUSTOM
        }
        
        # Check patterns in order
        for pattern, role in role_mapping.items():
            if pattern in combined_text:
                return role
        
        return AgentRole.CUSTOM
    
    def _has_agent_mentions(self, workflow_def: WorkflowDefinition) -> bool:
        """Check if workflow uses @mention system"""
        for agent in workflow_def.agents:
            if "@" in agent.instructions:
                return True
        return False
    
    def _estimate_complexity(self, workflow_def: WorkflowDefinition) -> str:
        """Estimate workflow complexity"""
        agent_count = len(workflow_def.agents)
        edge_count = len(workflow_def.edges)
        
        if agent_count <= 2 and edge_count <= 2:
            return "simple"
        elif agent_count <= 5 and edge_count <= 8:
            return "moderate"
        else:
            return "complex"
    
    def _get_visibility_summary(self, workflow_def: WorkflowDefinition) -> Dict[str, int]:
        """Get summary of agent visibility in workflow"""
        
        internal_count = 0
        external_count = 0
        
        for agent in workflow_def.agents:
            visibility = agent.metadata.get("output_visibility", OutputVisibility.EXTERNAL.value)
            if visibility == OutputVisibility.INTERNAL.value:
                internal_count += 1
            else:
                external_count += 1
        
        return {
            "internal_agents": internal_count,
            "external_agents": external_count,
            "total_agents": len(workflow_def.agents)
        }
    
    def _update_workflow_metrics(self, workflow_id: str, execution_record: Dict[str, Any]):
        """Update workflow metrics with execution data"""
        
        if workflow_id not in self.workflow_metrics:
            return
        
        metrics = self.workflow_metrics[workflow_id]
        metrics["total_executions"] += 1
        
        if execution_record.get("error"):
            metrics["failed_executions"] += 1
        else:
            metrics["successful_executions"] += 1
        
        # Update averages
        duration = execution_record.get("duration_ms", 0)
        if duration:
            metrics["average_duration_ms"] = (
                (metrics["average_duration_ms"] * (metrics["total_executions"] - 1) + duration)
                / metrics["total_executions"]
            )
        
        handoffs = execution_record.get("handoff_count", 0)
        metrics["average_handoffs"] = (
            (metrics["average_handoffs"] * (metrics["total_executions"] - 1) + handoffs)
            / metrics["total_executions"]
        )
    
    def _extract_metrics_from_events(self, events: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Extract execution metrics from streaming events"""
        
        metrics = {
            "message_count": 0,
            "agent_transfers": 0,
            "control_transitions": 0,
            "internal_messages": 0,
            "external_messages": 0
        }
        
        for event_type, event_data in events:
            if event_type == "message":
                metrics["message_count"] += 1
                if event_data.get("response_type") == ResponseType.INTERNAL.value:
                    metrics["internal_messages"] += 1
                else:
                    metrics["external_messages"] += 1
            elif event_type == "agent_transfer":
                metrics["agent_transfers"] += 1
            elif event_type == "control_transition":
                metrics["control_transitions"] += 1
        
        return metrics
    
    def _initialize_workflow_metrics(self, workflow_id: str):
        """Initialize metrics for a workflow"""
        
        if workflow_id not in self.active_workflows:
            return
        
        self.active_workflows[workflow_id]["metrics"] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_duration_ms": 0,
            "average_handoffs": 0,
            "execution_history": [],
            "agent_invocations": {},
            "error_log": []
        }

    def _track_execution_start(self, workflow_id: str, execution_id: str):
        """Track the start of a workflow execution"""
        
        if workflow_id not in self.active_workflows:
            return
        
        metrics = self.active_workflows[workflow_id].get("metrics", {})
        metrics["total_executions"] += 1
        
        # Add to active executions
        if "active_executions" not in metrics:
            metrics["active_executions"] = {}
        
        metrics["active_executions"][execution_id] = {
            "started_at": datetime.utcnow().isoformat(),
            "agent_calls": {},
            "handoffs": []
        }

    def _track_agent_invocation(self, workflow_id: str, execution_id: str, agent_name: str):
        """Track when an agent is invoked"""
        
        if workflow_id not in self.active_workflows:
            return
        
        metrics = self.active_workflows[workflow_id].get("metrics", {})
        
        # Update agent invocation count
        if agent_name not in metrics.get("agent_invocations", {}):
            metrics["agent_invocations"][agent_name] = 0
        metrics["agent_invocations"][agent_name] += 1
        
        # Track in active execution
        if execution_id in metrics.get("active_executions", {}):
            exec_data = metrics["active_executions"][execution_id]
            if agent_name not in exec_data["agent_calls"]:
                exec_data["agent_calls"][agent_name] = 0
            exec_data["agent_calls"][agent_name] += 1

    def _track_execution_complete(self, workflow_id: str, execution_id: str, success: bool, error: str = None):
        """Track workflow execution completion"""
        
        if workflow_id not in self.active_workflows:
            return
        
        metrics = self.active_workflows[workflow_id].get("metrics", {})
        
        # Update success/failure counts
        if success:
            metrics["successful_executions"] += 1
        else:
            metrics["failed_executions"] += 1
            if error:
                metrics["error_log"].append({
                    "execution_id": execution_id,
                    "error": error,
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        # Calculate duration and update average
        if execution_id in metrics.get("active_executions", {}):
            exec_data = metrics["active_executions"][execution_id]
            started_at = datetime.fromisoformat(exec_data["started_at"])
            duration_ms = int((datetime.utcnow() - started_at).total_seconds() * 1000)
            
            # Update average duration
            total_execs = metrics["total_executions"]
            current_avg = metrics["average_duration_ms"]
            metrics["average_duration_ms"] = ((current_avg * (total_execs - 1)) + duration_ms) / total_execs
            
            # Add to execution history
            metrics["execution_history"].append({
                "execution_id": execution_id,
                "started_at": exec_data["started_at"],
                "completed_at": datetime.utcnow().isoformat(),
                "duration_ms": duration_ms,
                "success": success,
                "error": error,
                "agent_calls": exec_data["agent_calls"],
                "handoff_count": len(exec_data.get("handoffs", []))
            })
            
            # Keep only last 100 executions in history
            if len(metrics["execution_history"]) > 100:
                metrics["execution_history"] = metrics["execution_history"][-100:]
            
            # Remove from active executions
            del metrics["active_executions"][execution_id]


    # ===== Public API Methods =====
    
    async def get_workflow_templates(self) -> Dict[str, Any]:
        """Get pre-built workflow templates with visibility examples"""
        return {
            "customer_support": {
                "name": "Customer Support System",
                "description": "Multi-tier support with triage and escalation",
                "agents": [
                    {
                        "name": "TriageAgent",
                        "role": "customer_support",
                        "outputVisibility": "user_facing",
                        "instructions": "Greet customers and identify their needs. Use @TechnicalSupport or @BillingSupport as needed.",
                        "connectedAgents": ["TechnicalSupport", "BillingSupport"]
                    },
                    {
                        "name": "TechnicalSupport",
                        "role": "analyzer",
                        "outputVisibility": "internal",
                        "controlType": "relinquish_to_parent",
                        "maxCallsPerParentAgent": 3,
                        "instructions": "Resolve technical issues. Analyze problems and provide solutions."
                    },
                    {
                        "name": "BillingSupport",
                        "role": "customer_support",
                        "outputVisibility": "internal",
                        "controlType": "relinquish_to_parent",
                        "instructions": "Handle billing inquiries and payment issues."
                    }
                ]
            },
            "research_assistant": {
                "name": "Research Assistant",
                "description": "AI research team with specialized roles",
                "agents": [
                    {
                        "name": "ResearchCoordinator",
                        "role": "coordinator",
                        "outputVisibility": "user_facing",
                        "instructions": "Coordinate research tasks. Use @DataAnalyst for data analysis and @FactChecker for verification.",
                        "connectedAgents": ["DataAnalyst", "FactChecker"]
                    },
                    {
                        "name": "DataAnalyst",
                        "role": "analyzer",
                        "outputVisibility": "internal",
                        "controlType": "relinquish_to_parent",
                        "instructions": "Analyze data and provide insights."
                    },
                    {
                        "name": "FactChecker",
                        "role": "researcher",
                        "outputVisibility": "internal",
                        "controlType": "relinquish_to_parent",
                        "instructions": "Verify facts and check sources."
                    }
                ]
            }
        }
    
    async def test_workflow(self, workflow_id: str, test_input: str) -> Dict[str, Any]:
        """Test workflow with visibility tracking"""
        
        test_results = {
            "workflow_id": workflow_id,
            "test_input": test_input,
            "events": [],
            "visibility_summary": {
                "internal_events": 0,
                "external_events": 0,
                "control_transitions": 0,
                "agent_transfers": 0
            }
        }
        
        try:
            async for event_type, event_data in self.execute_workflow_with_streaming(
                workflow_id,
                {"message": test_input}
            ):
                test_results["events"].append({
                    "type": event_type,
                    "data": event_data
                })
                
                # Track visibility
                if event_type == "message":
                    if event_data.get("response_type") == ResponseType.INTERNAL.value:
                        test_results["visibility_summary"]["internal_events"] += 1
                    else:
                        test_results["visibility_summary"]["external_events"] += 1
                elif event_type == "control_transition":
                    test_results["visibility_summary"]["control_transitions"] += 1
                elif event_type == "agent_transfer":
                    test_results["visibility_summary"]["agent_transfers"] += 1
            
            test_results["status"] = "completed"
            
        except Exception as e:
            test_results["status"] = "failed"
            test_results["error"] = str(e)
        
        return test_results
    
    async def suggest_workflow_improvements(self, workflow_id: str) -> Dict[str, Any]:
        """Use AI to suggest workflow improvements"""
        
        if workflow_id not in self.active_workflows:
            return {"error": "Workflow not found"}
        
        workflow_info = self.active_workflows[workflow_id]
        metrics = self.workflow_metrics.get(workflow_id, {})
        
        if not metrics.get("total_executions", 0):
            return {"suggestions": ["Run the workflow at least once to get improvement suggestions"]}
        
        suggestions = []
        
        # Analyze metrics
        if metrics.get("average_duration_ms", 0) > 60000:  # More than 1 minute
            suggestions.append("Consider parallelizing some agent tasks to reduce execution time")
        
        if metrics.get("average_handoffs", 0) > 5:
            suggestions.append("High number of handoffs detected. Consider consolidating some agent responsibilities")
        
        # Analyze visibility
        visibility = workflow_info["rowboat_metadata"]["visibility_summary"]
        if visibility["internal_agents"] > visibility["external_agents"] * 2:
            suggestions.append("Many internal agents detected. Consider if all processing needs to be hidden from users")
        
        return {
            "suggestions": suggestions,
            "metrics": metrics,
            "visibility_analysis": visibility
        }
    
    def get_workflow_visualization_data(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow data formatted for visualization"""
        
        if workflow_id not in self.active_workflows:
            return {"error": "Workflow not found"}
        
        workflow_info = self.active_workflows[workflow_id]
        workflow_def = workflow_info["definition"]
        
        # Format for visualization
        nodes = []
        edges = []
        
        for i, agent in enumerate(workflow_def.agents):
            # Determine node color based on visibility
            visibility = agent.metadata.get("output_visibility", OutputVisibility.EXTERNAL.value)
            node_color = "#90EE90" if visibility == OutputVisibility.EXTERNAL.value else "#FFB6C1"
            
            nodes.append({
                "id": agent.name,
                "type": "default",
                "data": {
                    "label": agent.name,
                    "role": agent.role.value if hasattr(agent.role, 'value') else str(agent.role),
                    "hasInstructions": bool(agent.instructions),
                    "connectedAgents": agent.metadata.get("connected_agents", []),
                    "visibility": visibility,
                    "controlType": agent.metadata.get("control_type", ControlType.RETAIN.value)
                },
                "position": {"x": (i % 3) * 200, "y": (i // 3) * 150},
                "style": {"backgroundColor": node_color}
            })
        
        for edge in workflow_def.edges:
            edges.append({
                "id": f"{edge.from_agent}-{edge.to_agent}",
                "source": edge.from_agent,
                "target": edge.to_agent,
                "type": "smoothstep",
                "animated": edge.condition_type == EdgeConditionType.MENTION_BASED,
                "label": edge.condition or ""
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": workflow_info["rowboat_metadata"]
        }
