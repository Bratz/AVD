# src/ii_agent/workflows/langgraph_integration.py
"""
BaNCS LangGraph Integration Module - Enhanced with ROWBOAT principles
Bridges BaNCS with LangGraph for multi-agent workflows using Chutes LLM
Fixed to use LangGraph's built-in message handling and proper ChutesOpenAIClient integration
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated, AsyncGenerator, Tuple
import json
import logging
from pathlib import Path
import asyncio
from datetime import datetime
import pickle
import hashlib
from collections import deque

# Updated imports for latest LangGraph with proper message handling
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint
from typing_extensions import TypedDict

# For messages - using LangChain's message types
from langchain_core.messages import (
    BaseMessage, 
    HumanMessage, 
    AIMessage, 
    SystemMessage,
    ToolMessage,
    AnyMessage
)

from src.ii_agent.llm.chutes_openai import ChutesOpenAIClient
from src.ii_agent.agents.base import BaseAgent
from mcp.integration_manager import MCPIntegrationManager

logger = logging.getLogger(__name__)

class WorkflowState(MessagesState):
    """
    State schema for multi-agent workflows.
    Inherits from MessagesState which provides built-in message handling.
    """
    # MessagesState already provides:
    # messages: Annotated[List[AnyMessage], add_messages]
    
    # Additional workflow-specific fields
    current_agent: str = ""
    agent_outputs: Dict[str, Any] = {}
    workflow_metadata: Dict[str, Any] = {}
    next_agent: Optional[str] = None
    final_output: Optional[Any] = None
    error_count: int = 0
    save_point_id: Optional[str] = None
    handoff_log: List[Dict[str, Any]] = []
    mention_history: List[Dict[str, Any]] = []
    last_external_message: Optional[str] = None
    workflow_complete: bool = False
    errors: List[Dict[str, Any]] = []

class StreamEvent(TypedDict):
    """Event structure for streaming"""
    type: str  # "token", "tool_call", "agent_transfer", "error", "checkpoint"
    agent: str
    content: Any
    timestamp: datetime

class BaNCSLangGraphBridge:
    """Enhanced BaNCS-LangGraph bridge with ROWBOAT features using built-in message handling"""
    
    def __init__(self, bancs_instance=None, checkpoint_path: str = None):
        self.bancs = bancs_instance
        
        # Use ROWBOAT checkpointer
        self.checkpointer = MemorySaver()  # Default to in-memory storage

        # Initialize components
        self.mcp_manager = MCPIntegrationManager()
        self.tool_executor = ToolExecutor(self.mcp_manager)
        self.context_manager = ContextManager()
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        
        # Initialize dependency placeholders
        self.workspace_manager = None
        self.message_queue = None
        self.context_manager_external = None
        self.mcp_wrapper = None
        
        # Initialize Chutes LLM clients with different models for different agents
        self.llm_registry = {
            "default": ChutesOpenAIClient(
                model_name="deepseek-ai/DeepSeek-V3-0324",
                use_native_tool_calling=True
            ),
            "researcher": ChutesOpenAIClient(
                model_name="NVIDIA/Nemotron-4-340B-Chat",
                use_native_tool_calling=True
            ),
            "analyzer": ChutesOpenAIClient(
                model_name="deepseek-ai/DeepSeek-V3-Chat",
                use_native_tool_calling=True
            ),
            "writer": ChutesOpenAIClient(
                model_name="Qwen/Qwen3-72B-Instruct",
                use_native_tool_calling=False  # JSON workaround mode
            ),
            "vision": ChutesOpenAIClient(
                model_name="chutesai/Llama-4-Maverick-17B-128E-Instruct-FP8",
                use_native_tool_calling=True
            )
        }
        
    def create_workflow(self, workflow_definition: Dict[str, Any]) -> Any:
        """Create a LangGraph workflow from definition with proper routing"""
        
        # Store definition for reference
        self.current_workflow_def = workflow_definition
        
        # Initialize the graph with state schema
        workflow = StateGraph(WorkflowState)
        
        # Add nodes for each agent
        for agent_config in workflow_definition.get("agents", []):
            agent_name = agent_config["name"]
            workflow.add_node(
                agent_name,
                self._create_agent_node(agent_config)
            )
        
        # Add routing logic
        def route_next_agent(state: WorkflowState) -> str:
            """Determine next agent based on state"""
            
            # Check if workflow is complete
            if state.get("workflow_complete"):
                return END
            
            # Check for errors
            if state.get("errors") and len(state.get("errors", [])) > self.max_retries:
                return END
            
            # Use next_agent if set
            if state.get("next_agent"):
                return state["next_agent"]
            
            # Get the current agent from state
            current = state.get("current_agent")
            
            # If no current agent, end workflow
            if not current:
                return END
            
            # Default to END
            return END
        
        # Set up routing for each agent
        for agent_config in workflow_definition.get("agents", []):
            agent_name = agent_config["name"]
            connected = agent_config.get("connected_agents", [])
            
            if connected:
                # Create routing map including all connected agents plus END
                route_map = {agent: agent for agent in connected}
                route_map[END] = END
                
                # Add conditional edges with routing function
                workflow.add_conditional_edges(
                    agent_name,
                    route_next_agent,
                    route_map
                )
            else:
                # No connections, end after this agent
                workflow.add_edge(agent_name, END)
        
        # Set entry point
        entry_point = workflow_definition.get("startAgent")
        if not entry_point and workflow_definition.get("agents"):
            entry_point = workflow_definition["agents"][0]["name"]
        
        if entry_point:
            workflow.set_entry_point(entry_point)
        
        # Compile workflow with checkpointer if available
        compiled = workflow.compile(checkpointer=self.checkpointer)
        
        return compiled
    
    def _create_agent_node(self, agent_config: Dict[str, Any]):
        """Create a LangGraph node for a BaNCS agent using built-in message handling"""
        
        async def agent_node(state: WorkflowState) -> Dict[str, Any]:
            try:
                # Extract agent details
                agent_name = agent_config["name"]
                agent_instructions = agent_config.get("instructions", "")
                agent_tools = agent_config.get("tools", [])
                output_visibility = agent_config.get("outputVisibility", "user_facing")
                
                # Get LLM client for this agent
                llm_client = self._get_llm_for_agent(agent_config)
                
                # Build conversation with proper LangChain messages
                # No need for _prepare_agent_messages - use state messages directly
                conversation_messages = self._build_conversation_for_agent(
                    state.get("messages", []), 
                    agent_config
                )
                
                # Execute agent with LLM
                logger.info(f"Executing agent: {agent_name}")
                
                # Make LLM call using the helper method
                tools = self._resolve_tools(agent_tools) if agent_tools else None
                
                try:
                    response_content = await self._call_llm(
                        llm_client,
                        conversation_messages,
                        tools=tools,
                        temperature=agent_config.get("temperature", 0.7),
                        max_tokens=1000
                    )
                except Exception as e:
                    logger.error(f"LLM call failed: {str(e)}")
                    raise
                
                # Check for @mentions to determine handoffs
                mentioned_agents = self._extract_agent_mentions(response_content)
                next_agent = None
                
                # Validate mentioned agents against connected agents
                if mentioned_agents:
                    connected = agent_config.get('connected_agents', [])
                    for mentioned in mentioned_agents:
                        if mentioned in connected:
                            next_agent = mentioned
                            break
                
                # Initialize handoff_log if not exists
                if "handoff_log" not in state:
                    state["handoff_log"] = []
                
                # Update handoff log if agent is handing off
                if next_agent:
                    logger.info(f"Agent {agent_name} handing off to {next_agent}")
                    state["handoff_log"].append({
                        "from_agent": agent_name,
                        "to_agent": next_agent,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                # Create agent output
                agent_output = {
                    "agent": agent_name,
                    "output": response_content,
                    "visibility": output_visibility,
                    "timestamp": datetime.utcnow().isoformat(),
                    "mentioned_agents": mentioned_agents
                }
                
                # Initialize agent_outputs if not exists
                if "agent_outputs" not in state:
                    state["agent_outputs"] = {}
                
                # Update agent_outputs
                state["agent_outputs"][agent_name] = agent_output
                
                # Create the AI message using LangChain's message type
                ai_message = AIMessage(
                    content=response_content,
                    name=agent_name
                )
                
                # Initialize mention_history if not exists
                if "mention_history" not in state:
                    state["mention_history"] = []
                
                # Track mentions in history
                if mentioned_agents:
                    state["mention_history"].extend([
                        {
                            "from": agent_name, 
                            "to": agent, 
                            "message_snippet": response_content[:100]
                        }
                        for agent in mentioned_agents
                    ])
                
                # Mark if external message was sent
                if output_visibility == "user_facing":
                    state["last_external_message"] = response_content
                    # Don't continue if user-facing agent responded
                    if not next_agent:
                        state["workflow_complete"] = True
                
                # Return state update with proper message handling
                return {
                    "messages": [ai_message],  # Will be added via add_messages reducer
                    "current_agent": next_agent if next_agent else agent_name,
                    "next_agent": next_agent,
                    "agent_outputs": state["agent_outputs"],
                    "handoff_log": state.get("handoff_log", []),
                    "mention_history": state.get("mention_history", []),
                    "last_external_message": state.get("last_external_message"),
                    "workflow_complete": state.get("workflow_complete", False)
                }
                
            except Exception as e:
                logger.error(f"Error executing agent {agent_config['name']}: {e}")
                
                # Initialize errors list if not exists
                if "errors" not in state:
                    state["errors"] = []
                
                # Add error to state
                state["errors"].append({
                    "agent": agent_config['name'],
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Return error message
                error_message = AIMessage(
                    content=f"I apologize, but I encountered an error: {str(e)}",
                    name=agent_config['name']
                )
                
                return {
                    "messages": [error_message],
                    "errors": state["errors"],
                    "error_count": state.get("error_count", 0) + 1
                }
        
        return agent_node
    
    def _build_conversation_for_agent(self, messages: List[AnyMessage], agent_config: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Build conversation for agent from LangGraph messages.
        This replaces _prepare_agent_messages and works with LangChain message types.
        """
        conversation = []
        
        # Add system message with agent instructions
        system_prompt = self._build_system_prompt(agent_config)
        conversation.append({"role": "system", "content": system_prompt})
        
        # Convert LangChain messages to dict format for LLM
        for msg in messages:
            if isinstance(msg, HumanMessage):
                conversation.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                # Include agent name if available
                if hasattr(msg, 'name') and msg.name:
                    if msg.name != agent_config["name"]:
                        # Messages from other agents
                        conversation.append({
                            "role": "assistant", 
                            "content": f"[{msg.name}]: {msg.content}"
                        })
                    else:
                        # This agent's previous messages
                        conversation.append({"role": "assistant", "content": msg.content})
                else:
                    conversation.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                # Additional system messages
                conversation.append({"role": "system", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                # Tool results
                conversation.append({
                    "role": "function",
                    "name": msg.name if hasattr(msg, 'name') else "tool",
                    "content": msg.content
                })
            elif isinstance(msg, dict):
                # Handle dict-format messages (backwards compatibility)
                role = msg.get("role", "user")
                content = msg.get("content", "")
                conversation.append({"role": role, "content": content})
        
        # Use context manager to limit conversation size
        managed_conversation = self.context_manager.manage_context(conversation)
        
        return managed_conversation
    
    def _build_system_prompt(self, agent_config: Dict[str, Any]) -> str:
        """Build comprehensive system prompt for agent"""
        agent_name = agent_config["name"]
        agent_instructions = agent_config.get("instructions", "")
        
        prompt_parts = [
            f"You are {agent_name}, part of a multi-agent workflow.",
            "",
            "Your Instructions:",
            agent_instructions
        ]
        
        # Add role-specific guidance
        if agent_config.get("outputVisibility") == "internal":
            prompt_parts.extend([
                "",
                "Important: You are an internal processing agent. Your output will not be shown directly to users.",
                "Focus on analysis and preparation for other agents."
            ])
        
        # Add handoff instructions if agent has connections
        connected_agents = agent_config.get('connected_agents', [])
        if connected_agents:
            prompt_parts.extend([
                "",
                "Agent Handoff Instructions:",
                f"- You can hand off to these agents: {', '.join(['@' + a for a in connected_agents])}",
                "- To hand off, mention the agent with @ (e.g., @AgentName)",
                "- Only hand off when the task requires their specific expertise"
            ])
        
        # Add tool usage instructions if agent has tools
        if agent_config.get("tools"):
            prompt_parts.extend([
                "",
                "Tool Usage:",
                "- Use the provided tools when necessary to complete tasks",
                "- Always explain what you're doing when using tools"
            ])
        
        prompt_parts.extend([
            "",
            "General Guidelines:",
            "- Be clear and concise in your responses",
            "- Follow your specific role and instructions carefully",
            "- Provide actionable outputs based on your expertise"
        ])
        
        return "\n".join(prompt_parts)
    
    def _resolve_tools(self, tool_names: List[str]) -> List[Dict[str, Any]]:
        """Resolve tool names to tool configurations"""
        if not tool_names:
            return []
            
        return self._format_tools_for_chutes(tool_names)
    
    async def _call_llm(self, llm_client, messages: List[Dict[str, str]], 
                       tools: Optional[List[Dict[str, Any]]] = None,
                       temperature: float = 0.7,
                       max_tokens: int = 1000) -> str:
        """
        Helper method to call different types of LLM clients.
        Returns the response content as a string.
        """
        # Check if this is a ChutesOpenAIClient - use its generate method
        if type(llm_client).__name__ == 'ChutesOpenAIClient':
            # Use the sync generate method with proper conversion
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._call_chutes_client(llm_client, messages, tools, temperature, max_tokens)
            )
            return result
        
        # For OpenAI-style clients with chat.completions.create
        elif hasattr(llm_client, 'chat') and hasattr(llm_client.chat, 'completions'):
            create_params = {
                "model": getattr(llm_client, 'model_name', 'default'),
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            if tools:
                create_params["tools"] = tools
            
            # Handle both sync and async create methods
            create_method = llm_client.chat.completions.create
            if asyncio.iscoroutinefunction(create_method):
                response = await create_method(**create_params)
            else:
                # If it's sync, we need to run it in an executor
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, lambda: create_method(**create_params))
            
            # Extract content from OpenAI-style response
            return response.choices[0].message.content
        
        # For LangChain-style clients  
        elif hasattr(llm_client, 'agenerate'):
            response = await llm_client.agenerate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.generations[0][0].text
            
        elif hasattr(llm_client, 'generate'):
            # Sync generate - run in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: llm_client.generate(messages=messages, temperature=temperature, max_tokens=max_tokens)
            )
            return response.generations[0][0].text
            
        # For clients with invoke method
        elif hasattr(llm_client, 'ainvoke'):
            response = await llm_client.ainvoke(messages[-1]['content'])  # Use last message
            return response.content if hasattr(response, 'content') else str(response)
            
        elif hasattr(llm_client, 'invoke'):
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: llm_client.invoke(messages[-1]['content'])
            )
            return response.content if hasattr(response, 'content') else str(response)
            
        else:
            raise AttributeError(
                f"LLM client {type(llm_client).__name__} doesn't have any recognized completion methods. "
                f"Available methods: {[m for m in dir(llm_client) if not m.startswith('_')]}"
            )
    
    def _call_chutes_client(self, llm_client, messages: List[Dict[str, str]], 
                           tools: Optional[List[Dict[str, Any]]], 
                           temperature: float, 
                           max_tokens: int) -> str:
        """Synchronous helper to call ChutesOpenAIClient through its generate method"""
        from src.ii_agent.llm.base import TextPrompt, TextResult, ToolParam
        
        # Convert messages to format expected by ChutesOpenAIClient.generate()
        llm_messages = []
        system_prompt = None
        
        # Extract system prompt first
        for msg in messages:
            if msg.get('role') == 'system':
                system_prompt = msg['content']
                break
        
        # Group messages by alternating user/assistant pattern
        current_user_messages = []
        current_assistant_messages = []
        
        for msg in messages:
            if msg.get('role') == 'system':
                continue  # Already extracted
                
            if msg.get('role') == 'user':
                # If we have pending assistant messages, add them first
                if current_assistant_messages:
                    llm_messages.append(current_assistant_messages)
                    current_assistant_messages = []
                
                # If we already have user messages, add them and start new group
                if current_user_messages:
                    llm_messages.append(current_user_messages)
                    current_user_messages = []
                    
                current_user_messages.append(TextPrompt(text=msg['content']))
                
            elif msg.get('role') == 'assistant':
                # If we have pending user messages, add them first
                if current_user_messages:
                    llm_messages.append(current_user_messages)
                    current_user_messages = []
                
                current_assistant_messages.append(TextResult(text=msg['content']))
        
        # Add any remaining messages
        if current_user_messages:
            llm_messages.append(current_user_messages)
        if current_assistant_messages:
            llm_messages.append(current_assistant_messages)
        
        # Convert tools to ToolParam format if needed
        tool_params = []
        if tools:
            for tool in tools:
                if isinstance(tool, dict) and 'function' in tool:
                    func = tool['function']
                    tool_params.append(ToolParam(
                        name=func.get('name', ''),
                        description=func.get('description', ''),
                        input_schema=func.get('parameters', {})
                    ))
        
        # Call the generate method
        try:
            internal_messages, metadata = llm_client.generate(
                messages=llm_messages,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
                tools=tool_params
            )
            
            # Extract response text from internal messages
            for msg in internal_messages:
                if hasattr(msg, 'text'):
                    return msg.text
            
            # If no text found, return empty string
            return ""
            
        except Exception as e:
            logger.error(f"ChutesOpenAIClient.generate() failed: {str(e)}")
            raise
    
    def _extract_agent_mentions(self, content: str) -> List[str]:
        """Extract @mentions of agents from content"""
        import re
        
        # Find all @mentions
        mentions = re.findall(r'@(\w+)', content)
        
        # Return unique mentions
        return list(set(mentions))
    
    def _get_llm_for_agent(self, agent_config: Dict[str, Any]) -> ChutesOpenAIClient:
        """Get LLM client for agent based on configuration"""
        
        role = agent_config.get("role", "generic")
        model = agent_config.get("model")
        
        if model:
            # Create specific model client
            return ChutesOpenAIClient(
                model_name=model,
                use_native_tool_calling=agent_config.get("use_native_tools", True)
            )
        elif role in self.llm_registry:
            return self.llm_registry[role]
        else:
            return self.llm_registry["default"]
    
    def _format_tools_for_chutes(self, tool_names: List[str]) -> List[Dict[str, Any]]:
        """Format tools for Chutes OpenAI client"""
        tools = []
        
        for tool_name in tool_names:
            # Get tool definition from registry or MCP
            if tool_name.startswith("mcp:"):
                # MCP tool format
                parts = tool_name.split(":")
                if len(parts) >= 3:
                    server = parts[1]
                    tool = ":".join(parts[2:])
                    
                    # Get tool schema from MCP
                    tool_schema = self.mcp_manager.get_tool_schema(server, tool)
                    if tool_schema:
                        tools.append({
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "description": tool_schema.get("description", f"MCP tool: {tool}"),
                                "parameters": tool_schema.get("inputSchema", {
                                    "type": "object",
                                    "properties": {},
                                    "required": []
                                })
                            }
                        })
            else:
                # Native tool - basic schema
                tools.append({
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": f"Tool: {tool_name}",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                })
        
        return tools
    
    # Simplified invoke method that works with built-in message handling
    async def ainvoke(self, workflow, input_data: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Async invoke workflow with proper message handling.
        Input should include 'messages' key with LangChain message objects.
        """
        # Ensure input has proper format
        if "messages" not in input_data:
            # Convert to proper format if needed
            if "message" in input_data:
                input_data["messages"] = [HumanMessage(content=input_data["message"])]
            elif "content" in input_data:
                input_data["messages"] = [HumanMessage(content=input_data["content"])]
            else:
                input_data["messages"] = [HumanMessage(content=str(input_data))]
        
        # Ensure messages are LangChain message objects
        formatted_messages = []
        for msg in input_data["messages"]:
            if isinstance(msg, BaseMessage):
                formatted_messages.append(msg)
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    formatted_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    formatted_messages.append(AIMessage(content=content))
                elif role == "system":
                    formatted_messages.append(SystemMessage(content=content))
            else:
                # Fallback - treat as user message
                formatted_messages.append(HumanMessage(content=str(msg)))
        
        input_data["messages"] = formatted_messages
        
        # Set initial agent if specified
        if "current_agent" not in input_data and self.current_workflow_def:
            entry_point = self.current_workflow_def.get("startAgent")
            if entry_point:
                input_data["current_agent"] = entry_point
        
        # Invoke workflow
        return await workflow.ainvoke(input_data, config)


# Keep the existing helper classes (ContextManager, ToolExecutor, ROWBOATCheckpointer)
# from the original file as they don't need changes

class ContextManager:
    """ROWBOAT-style context management with token limits"""
    
    def __init__(self, max_tokens: int = 4000, model_context_window: int = 8192):
        self.max_tokens = max_tokens
        self.model_context_window = model_context_window
        self.token_estimate_ratio = 4  # Rough estimate: 1 token ≈ 4 characters
        
    def manage_context(self, messages: List[Dict[str, str]], preserve_system: bool = True) -> List[Dict[str, str]]:
        """Manage context size to fit within token limits"""
        if not messages:
            return messages
        
        # Always preserve system message
        system_messages = [m for m in messages if m.get("role") == "system"]
        other_messages = [m for m in messages if m.get("role") != "system"]
        
        # Estimate tokens
        total_chars = sum(len(m.get("content", "")) for m in messages)
        estimated_tokens = total_chars // self.token_estimate_ratio
        
        if estimated_tokens <= self.max_tokens:
            return messages
        
        # Implement sliding window with summarization for older messages
        result = system_messages.copy()
        
        # Keep recent messages
        recent_messages = other_messages[-10:]  # Keep last 10 messages
        older_messages = other_messages[:-10]
        
        # Summarize older messages if any
        if older_messages:
            summary = self._summarize_messages(older_messages)
            result.append({"role": "assistant", "content": f"[Previous conversation summary]: {summary}"})
        
        result.extend(recent_messages)
        return result
    
    def _summarize_messages(self, messages: List[Dict[str, str]]) -> str:
        """Simple summarization - in production, use LLM for this"""
        # Count message types
        user_count = sum(1 for m in messages if m.get("role") == "user")
        assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
        
        # Extract key topics (simplified)
        all_content = " ".join(m.get("content", "")[:100] for m in messages)
        
        return f"Previous {len(messages)} messages ({user_count} user, {assistant_count} assistant) discussed: {all_content[:200]}..."


class ToolExecutor:
    """ROWBOAT-style tool executor for MCP and native tools"""
    
    def __init__(self, mcp_manager: MCPIntegrationManager = None, native_tools: Dict[str, Any] = None):
        self.mcp_manager = mcp_manager or MCPIntegrationManager()
        self.native_tools = native_tools or {}
        self.execution_history = deque(maxlen=100)  # Keep last 100 executions
        
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute tool with proper error handling and logging"""
        execution_id = f"exec_{datetime.utcnow().timestamp()}"
        
        try:
            # Log execution start
            self.execution_history.append({
                "id": execution_id,
                "tool": tool_name,
                "arguments": arguments,
                "timestamp": datetime.utcnow(),
                "status": "started"
            })
            
            # Determine tool type and execute
            if tool_name.startswith("mcp:"):
                result = await self._execute_mcp_tool(tool_name, arguments)
            else:
                result = await self._execute_native_tool(tool_name, arguments)
            
            # Log success
            self.execution_history[-1]["status"] = "completed"
            self.execution_history[-1]["result"] = result
            
            return result
            
        except Exception as e:
            # Log failure
            self.execution_history[-1]["status"] = "failed"
            self.execution_history[-1]["error"] = str(e)
            
            logger.error(f"Tool execution failed: {tool_name} - {str(e)}")
            return {
                "error": True,
                "message": f"Tool execution failed: {str(e)}",
                "tool": tool_name
            }
    
    async def _execute_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute MCP tool"""
        # Parse MCP tool format: mcp:server:tool
        parts = tool_name.split(":")
        if len(parts) < 3:
            raise ValueError(f"Invalid MCP tool format: {tool_name}")
        
        server_name = parts[1]
        actual_tool = ":".join(parts[2:])  # Handle tools with colons in name
        
        # Execute through MCP manager
        result = await self.mcp_manager.execute_tool(
            server_name=server_name,
            tool_name=actual_tool,
            arguments=arguments
        )
        
        return result
    
    async def _execute_native_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute native tool"""
        if tool_name not in self.native_tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        tool = self.native_tools[tool_name]
        
        # Execute tool (assuming async interface)
        if asyncio.iscoroutinefunction(tool):
            result = await tool(**arguments)
        else:
            result = tool(**arguments)
        
        return result