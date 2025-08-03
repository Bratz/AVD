"""
Enhanced II-Agent base classes incorporating patterns from the original II-Agent repository
"""
from abc import ABC, abstractmethod
import json
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel
import uuid
import asyncio
from datetime import datetime

# Import our custom memory and planning components
from .memory import AgentMemory, ConversationMemory
from .planning import PlanningEngine, Plan, PlanStep, PlanStepStatus
from .event import EventType, RealtimeEvent
# Add this import at the top
from src.ii_agent.llm.base import TextPrompt, TextResult


class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    ERROR = "error"

class ThoughtStep(BaseModel):
    id: str
    timestamp: datetime
    type: str  # "observation", "thought", "action", "reflection", "planning"
    content: str
    metadata: Dict[str, Any] = {}

class AgentContext(BaseModel):
    """Enhanced agent context following II-Agent patterns."""
    session_id: str
    conversation_history: List[Dict[str, Any]] = []
    working_memory: Dict[str, Any] = {}
    long_term_memory: Dict[str, Any] = {}
    thought_trail: List[ThoughtStep] = []
    current_goal: Optional[str] = None
    sub_goals: List[str] = []
    current_plan: Optional[str] = None  # Plan ID
    
    # Enhanced II-Agent features
    agent_memory: Optional[AgentMemory] = None
    conversation_memory: Optional[ConversationMemory] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.agent_memory:
            self.agent_memory = AgentMemory(session_id=self.session_id)
        if not self.conversation_memory:
            self.conversation_memory = ConversationMemory(session_id=self.session_id)

class IIAgent(ABC):
    """Enhanced II-Agent base class incorporating patterns from the original repository."""
    
    def __init__(self, agent_id: str, name: str, description: str):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.state = AgentState.IDLE
        self.context: Optional[AgentContext] = None
        self.mcp_wrapper = None
        self.ollama_wrapper = None
        
        # II-Agent specific components
        self.planning_engine = PlanningEngine(agent_id)
        self.capabilities = []
        self.constraints = []


        #New enhanced features
        from .event_stream import EventStream
        self.event_stream = EventStream()
        self._event_processing_task = None
        
    async def initialize(self, context: AgentContext, mcp_wrapper, ollama_wrapper=None):
        """Initialize agent with enhanced context and wrappers."""
        self.context = context
        self.mcp_wrapper = mcp_wrapper
        self.ollama_wrapper = ollama_wrapper
        
        # Initialize agent-specific setup
        await self._setup_capabilities()
        await self._setup_constraints()
        
        await self.add_thought("observation", f"II-Agent {self.name} initialized with enhanced capabilities")
        #Enhanced event stream processing
        self._event_processing_task = asyncio.create_task(self._process_event_stream())

        # Emit initialization event
        await self.event_stream.emit_event(
            EventType.AGENT_INITIALIZED,
            {
                "agent_id": self.agent_id,
                "agent_name": self.name,
                "capabilities": self.capabilities
            }
        )


    async def _setup_capabilities(self):
        """Setup agent capabilities (to be overridden by subclasses)."""
        self.capabilities = ["basic_reasoning", "mcp_tool_execution"]
        if self.ollama_wrapper:
            self.capabilities.extend(["llm_planning", "llm_reasoning", "llm_reflection"])
    
    async def _setup_constraints(self):
        """Setup agent constraints (to be overridden by subclasses)."""
        self.constraints = ["banking_domain_only", "ethical_operations"]
        
    async def add_thought(self, thought_type: str, content: str, metadata: Dict[str, Any] = None):
        """Add a thought step to the trail with enhanced metadata."""
        if self.context:
            thought = ThoughtStep(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                type=thought_type,
                content=content,
                metadata={
                    **(metadata or {}),
                    "agent_id": self.agent_id,
                    "agent_state": self.state.value,
                    "capabilities": self.capabilities
                }
            )
            self.context.thought_trail.append(thought)
            
            # Also store in conversation memory
            if self.context.conversation_memory:
                self.context.conversation_memory.add_message(
                    role="agent",
                    content=f"[{thought_type.upper()}] {content}",
                    metadata=thought.metadata
                )
            # Emit thought as event
            await self.event_stream.emit_thought_as_event(thought)

    async def create_plan(self, goal: str, planning_context: Dict[str, Any] = None) -> Plan:
        """Create a detailed plan using II-Agent planning patterns."""
        self.state = AgentState.PLANNING
        await self.add_thought("planning", f"Creating plan for goal: {goal}")
        
        # Use LLM for intelligent planning if available
        if self.ollama_wrapper:
            try:
                step_descriptions = await self.ollama_wrapper.generate_plan(goal, planning_context or {})
                steps_data = [
                    {
                        "description": desc,
                        "action": self._convert_description_to_action(desc),
                        "parameters": {}
                    }
                    for desc in step_descriptions
                ]
            except Exception as e:
                await self.add_thought("planning", f"LLM planning failed: {e}, using fallback")
                steps_data = await self._create_fallback_plan_steps(goal)
        else:
            steps_data = await self._create_fallback_plan_steps(goal)
        
        # Create plan using planning engine
        plan = self.planning_engine.create_plan(goal, steps_data)
        self.context.current_plan = plan.id
        
        await self.add_thought("planning", f"Created plan with {len(plan.steps)} steps", 
                              {"plan_id": plan.id, "steps": len(plan.steps)})
        return plan
        
    def _convert_description_to_action(self, description: str) -> str:
        """Convert a plan description to an actionable method name."""
        desc_lower = description.lower()
        
        # FIRST: Check if the description explicitly mentions a tool name
        # This is the most important check
        if "`list_api_endpoints`" in description or "list_api_endpoints" in desc_lower:
            return "list_api_endpoints"
        elif "`get_api_endpoint_schema`" in description or "get_api_endpoint_schema" in desc_lower:
            return "get_api_endpoint_schema"
        elif "`invoke_api_endpoint`" in description or "invoke_api_endpoint" in desc_lower:
            return "invoke_api_endpoint"
        elif "`analyze_goal`" in description:
            return "analyze_goal"
        elif "`format_results`" in description:
            return "format_results"
        
        # Then check for banking wrapper names
        if "list_banking_apis" in desc_lower:
            return "list_banking_apis"
        elif "get_api_structure" in desc_lower:
            return "get_api_structure"
        elif "invoke_banking_api" in desc_lower:
            return "invoke_banking_api"
        
        # Generic keyword mappings (only as fallback)
        if "analyze" in desc_lower and "goal" in desc_lower:
            return "analyze_goal"
        elif "format" in desc_lower and "result" in desc_lower:
            return "format_results"
        elif "present" in desc_lower and "result" in desc_lower:
            return "format_results"
        
        # These generic mappings should be last resort
        if "extract" in desc_lower or "identify" in desc_lower:
            return "extract_data"
        elif "execute" in desc_lower and "step" in desc_lower:
            return "execute_step"
        elif "iterate" in desc_lower:
            return "execute_step"
        else:
            return "execute_task"
            
    async def _create_fallback_plan_steps(self, goal: str) -> List[Dict[str, Any]]:
        # """Create fallback plan steps when LLM is unavailable."""
        # # This will be overridden by subclasses
        # return [
        #     {"description": "Analyze goal", "action": "analyze_goal", "parameters": {}},
        #     {"description": "Execute main task", "action": "execute_task", "parameters": {}},
        #     {"description": "Format results", "action": "format_results", "parameters": {}}
        # ]
        """Create fallback plan steps using MCP tool names"""
        goal_lower = goal.lower()
        
        # For API listing requests
        if any(word in goal_lower for word in ["list", "show", "get", "what"]) and "api" in goal_lower:
            return [{
                "description": "List all available API endpoints",
                "action": "list_api_endpoints",
                "parameters": {}  # Empty parameters - let the tool use its defaults
            }]

    async def execute(self, goal: str) -> Dict[str, Any]:
        """Execute the agent's main task with enhanced II-Agent patterns."""
        self.state = AgentState.EXECUTING
        await self.add_thought("observation", f"Starting execution for goal: {goal}")
        
        try:
            # Create plan
            plan = await self.create_plan(goal)
            
            # Execute plan steps
            results = []
            for step in plan.steps:
                await self.add_thought("action", f"Executing step: {step.description}")
                
                try:
                    step_result = await self._execute_plan_step(step)
                    plan.mark_step_completed(step.id, step_result)
                    results.append(step_result)
                    await self.add_thought("observation", f"Step completed: {step.description}")
                except Exception as e:
                    plan.mark_step_failed(step.id, str(e))
                    await self.add_thought("observation", f"Step failed: {step.description} - {e}")
                    results.append({"error": str(e)})
            
            # Reflect on results
            final_result = await self.reflect(results, plan)
            self.state = AgentState.COMPLETED
            
            return final_result
            
        except Exception as e:
            self.state = AgentState.ERROR
            await self.add_thought("observation", f"Execution failed: {str(e)}")
            raise
            
    async def _execute_plan_step(self, step: PlanStep) -> Any:
        """Execute an individual plan step."""
        # This maps to the existing _execute_step method
        return await self._execute_step(step.action, step.parameters)
            
    async def reflect(self, results: List[Any], plan: Plan = None) -> Dict[str, Any]:
        """Enhanced reflection incorporating II-Agent patterns."""
        self.state = AgentState.REFLECTING
        await self.add_thought("reflection", "Starting enhanced reflection on execution results")
        
        # Use LLM for intelligent reflection if available
        if self.ollama_wrapper:
            try:
                reflection_data = await self.ollama_wrapper.generate_reflection(
                    results, 
                    self.context.current_goal or "task execution"
                )
                await self.add_thought("reflection", f"LLM reflection: {reflection_data}")
                
                # Enhanced reflection with plan analysis
                if plan:
                    reflection_data["plan_analysis"] = {
                        "total_steps": len(plan.steps),
                        "completed_steps": len([s for s in plan.steps if s.status == PlanStepStatus.COMPLETED]),
                        "failed_steps": len([s for s in plan.steps if s.status == PlanStepStatus.FAILED]),
                        "plan_effectiveness": "high" if plan.is_complete() else "medium"
                    }
                
                return reflection_data
            except Exception as e:
                await self.add_thought("reflection", f"LLM reflection failed: {e}")
        
        # Fallback reflection
        return await self._perform_fallback_reflection(results, plan)
        
    async def _perform_fallback_reflection(self, results: List[Any], plan: Plan = None) -> Dict[str, Any]:
        """Fallback reflection when LLM is unavailable."""
        successful_results = [r for r in results if r is not None and not isinstance(r, dict) or not r.get("error")]
        
        reflection = {
            "goal_achievement": "complete" if len(successful_results) == len(results) else "partial",
            "result_quality": "high" if len(successful_results) > len(results) * 0.8 else "medium",
            "execution_effectiveness": "high" if plan and plan.is_complete() else "medium",
            "success_rating": min(10, int((len(successful_results) / len(results)) * 10)) if results else 5,
            "lessons_learned": ["Execution completed with II-Agent framework"],
            "recommendations": ["Continue using structured approach"],
            "summary": f"Completed {len(successful_results)}/{len(results)} steps successfully"
        }
        
        if plan:
            reflection["plan_analysis"] = {
                "total_steps": len(plan.steps),
                "completed_steps": len([s for s in plan.steps if s.status == PlanStepStatus.COMPLETED]),
                "failed_steps": len([s for s in plan.steps if s.status == PlanStepStatus.FAILED])
            }
        
        return reflection

    async def _process_event_stream(self):
        """Main event processing loop following II-Agent pattern"""
        while True:
            try:
                event = await self.event_stream.get_next()
                await self._handle_event(event)
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.event_stream.emit_event(
                    EventType.ERROR,
                    {"error": str(e), "context": "event_processing"}
                )
                
    async def _handle_event(self, event: RealtimeEvent):
        """Handle events - to be overridden by subclasses"""
        # Default handling
        if event.type == EventType.USER_MESSAGE:
            await self._handle_user_message(event.content)
        elif event.type == EventType.TOOL_RESULT:
            await self._handle_tool_result(event.content)
            
    async def _handle_user_message(self, content: Dict[str, Any]):
        """Handle user message events"""
        message = content.get("message", "")
        await self.add_thought("observation", f"Received user message: {message}")
        
    async def _handle_tool_result(self, content: Dict[str, Any]):
        """Handle tool result events"""
        tool_name = content.get("tool_name", "")
        result = content.get("result", {})
        await self.add_thought("observation", f"Tool {tool_name} completed with result")


    async def create_plan(self, goal: str, planning_context: Dict[str, Any] = None) -> Plan:
        """Create a detailed plan using II-Agent planning patterns."""
        self.state = AgentState.PLANNING
        await self.add_thought("planning", f"Creating plan for goal: {goal}")
        
        # Use LLM for intelligent planning if available
        if self.ollama_wrapper:
            try:
                # Create planning prompt
                planning_prompt = f"""Create a step-by-step plan to accomplish this goal: {goal}

Context: {json.dumps(planning_context or {}, indent=2) if planning_context else "No additional context"}

Available tools:
For banking operations, use these MCP tools:
- list_api_endpoints: To discover banking APIs (always use with compact=true)
- get_api_endpoint_schema: To understand API parameters
- invoke_api_endpoint: To execute banking operations

For general tasks, actions like:
- analyze_goal: To understand the request
- execute_task: To perform general operations
- format_results: To format outputs

Create numbered steps mentioning which tool/action to use. Be specific."""

                # Create message in the correct format for Ollama
                messages = [[TextPrompt(text=planning_prompt)]]
                
                # Use the generate method with correct format
                response_blocks, _ = await self.ollama_wrapper.generate(
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7
                )
                
                # Extract text from response blocks
                response_text = ""
                for block in response_blocks:
                    if isinstance(block, TextResult):
                        response_text += block.text
                    elif hasattr(block, 'text'):
                        response_text += block.text
                    else:
                        response_text += str(block)
                
                # Parse numbered steps from response
                step_descriptions = []
                lines = response_text.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    # Match lines that start with numbers (1. 2. etc) or bullets
                    if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                        # Remove numbering/bullets
                        import re
                        step_text = re.sub(r'^[\d\.\-\*\s]+', '', line).strip()
                        if step_text:
                            step_descriptions.append(step_text)
                
                if not step_descriptions:
                    # Try to extract any reasonable steps even without numbering
                    for line in lines:
                        line = line.strip()
                        if line and len(line) > 10:  # Reasonable length for a step
                            step_descriptions.append(line)
                
                if not step_descriptions:
                    raise ValueError("No valid steps extracted from LLM response")
                    
                steps_data = [
                    {
                        "description": desc,
                        "action": self._convert_description_to_action(desc),
                        "parameters": {}
                    }
                    for desc in step_descriptions[:10]  # Limit to 10 steps max
                ]
                
            except Exception as e:
                await self.add_thought("planning", f"LLM planning failed: {e}, using fallback")
                steps_data = await self._create_fallback_plan_steps(goal)
        else:
            steps_data = await self._create_fallback_plan_steps(goal)
        
        # Create plan using planning engine
        plan = self.planning_engine.create_plan(goal, steps_data)
        self.context.current_plan = plan.id
        
        await self.add_thought("planning", f"Created plan with {len(plan.steps)} steps", 
                            {"plan_id": plan.id, "steps": len(plan.steps)})
        return plan


    async def reflect(self, results: List[Any], plan: Plan = None) -> Dict[str, Any]:
        """Enhanced reflection incorporating II-Agent patterns."""
        self.state = AgentState.REFLECTING
        await self.add_thought("reflection", "Starting enhanced reflection on execution results")
        
        # Use LLM for intelligent reflection if available
        if self.ollama_wrapper:
            try:
                # Limit results to avoid token overflow
                results_summary = []
                for i, result in enumerate(results[:5]):
                    if isinstance(result, dict):
                        results_summary.append(f"Step {i+1}: {result.get('status', 'completed')}")
                    else:
                        results_summary.append(f"Step {i+1}: success")
                
                reflection_prompt = f"""Reflect on the execution results and provide insights.

    Goal: {self.context.current_goal or "task execution"}
    Results Summary: {', '.join(results_summary)}
    Total Steps: {len(results)}

    Provide:
    1. Brief summary of what was accomplished
    2. Success rate assessment
    3. Key insights or observations
    4. Recommendations for improvement

    Be concise and actionable."""

                # Create message in the correct format
                messages = [[TextPrompt(text=reflection_prompt)]]
                
                # Use the generate method
                response_blocks, _ = await self.ollama_wrapper.generate(
                    messages=messages,
                    max_tokens=300,
                    temperature=0.5
                )
                
                # Extract reflection text
                reflection_text = ""
                for block in response_blocks:
                    if isinstance(block, TextResult):
                        reflection_text += block.text
                    elif hasattr(block, 'text'):
                        reflection_text += block.text
                    else:
                        reflection_text += str(block)
                
                # Parse reflection into structured format
                reflection_data = {
                    "summary": reflection_text[:200] if reflection_text else "Reflection completed",
                    "full_reflection": reflection_text,
                    "goal_achievement": "complete" if len([r for r in results if not isinstance(r, dict) or "error" not in r]) == len(results) else "partial",
                    "success_count": len([r for r in results if not isinstance(r, dict) or "error" not in r]),
                    "total_count": len(results)
                }
                
                # Enhanced reflection with plan analysis
                if plan:
                    reflection_data["plan_analysis"] = {
                        "total_steps": len(plan.steps),
                        "completed_steps": len([s for s in plan.steps if s.status == PlanStepStatus.COMPLETED]),
                        "failed_steps": len([s for s in plan.steps if s.status == PlanStepStatus.FAILED]),
                        "plan_effectiveness": "high" if plan.is_complete() else "medium"
                    }
                
                await self.add_thought("reflection", f"LLM reflection completed")
                return reflection_data
                
            except Exception as e:
                await self.add_thought("reflection", f"LLM reflection failed: {e}")
                # Fall through to fallback
        
        # Fallback reflection
        return await self._perform_fallback_reflection(results, plan)

    # Abstract methods that subclasses must implement
    @abstractmethod
    async def _execute_step(self, action: str, parameters: Dict[str, Any] = None) -> Any:
        """Execute a single step - to be implemented by subclasses."""
        pass