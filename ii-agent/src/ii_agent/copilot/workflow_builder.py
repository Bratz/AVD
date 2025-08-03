# src/ii_agent/copilot/workflow_builder.py
"""
ROWBOAT-aligned WorkflowCopilot for ii-agent integration
Follows the pattern from https://github.com/rowboatlabs/rowboat
"""

from typing import Dict, List, Any, Optional, Union, Literal, AsyncGenerator
import json
import re
import logging
import asyncio
from dataclasses import dataclass
from enum import Enum
from src.ii_agent.config.rowboat_config import rowboat_config
from src.ii_agent.workflows.definitions import WorkflowDefinition

logger = logging.getLogger(__name__)

# Align with ROWBOAT types
class OutputVisibility(Enum):
    EXTERNAL = "user_facing"
    INTERNAL = "internal"

class ControlType(Enum):
    RETAIN = "retain"
    PARENT_AGENT = "relinquish_to_parent"
    START_AGENT = "start_agent"

# class AgentRole(Enum):
#     ESCALATION = "escalation"
#     POST_PROCESSING = "post_process"
#     GUARDRAILS = "guardrails"

# Add after the existing enums
class WorkflowStreamEvent(Enum):
    """Types of events during workflow creation"""
    ANALYSIS_START = "analysis_start"
    AGENT_CREATED = "agent_created"
    TOOL_ADDED = "tool_added"
    EDGE_CREATED = "edge_created"
    PROMPT_CREATED = "prompt_created"
    PROGRESS = "progress"
    WORKFLOW_COMPLETE = "workflow_complete"
    ERROR = "error"

@dataclass
class AgentConfig:
    """ROWBOAT-compatible agent configuration"""
    name: str
    instructions: str
    tools: List[str] = None
    prompts: List[str] = None
    examples: List[Dict[str, str]] = None
    outputVisibility: str = "user_facing"
    controlType: str = "retain"
    connected_agents: List[str] = None
    ragDataSources: List[str] = None
    model: str = None

    def to_dict(self):
        """Convert to ROWBOAT-compatible dict format"""
        config = {
            "name": self.name,
            "instructions": self.instructions,
            "outputVisibility": self.outputVisibility,
            "controlType": self.controlType
        }
        if self.tools:
            config["tools"] = self.tools
        if self.prompts:
            config["prompts"] = self.prompts
        if self.examples:
            config["examples"] = self.examples
        if self.connected_agents:
            config["connected_agents"] = self.connected_agents
        if self.ragDataSources:
            config["ragDataSources"] = self.ragDataSources
        if self.model:
            config["model"] = self.model
        return config


class WorkflowCopilot:
    """ROWBOAT-aligned copilot for building multi-agent workflows"""
    
    def __init__(self, llm_client, default_model: str = None):
        """
        Initialize with a ChutesOpenAIClient instance
        
        Args:
            llm_client: ChutesOpenAIClient instance
            default_model: Default model for new agents
        """
        self.llm = llm_client
        self.default_model = default_model or rowboat_config.default_agent_model

        # Store rowboat config reference
        self.rowboat_config = rowboat_config
        
        self._current_workflow_spec = None  # Add this line

        # Load copilot instructions (simplified version of copilot_multi_agent.md)
        self.copilot_instructions = self._load_copilot_instructions()
    
    async def build_workflow_stream(
        self,
        description: str,
        examples: Optional[List[Dict[str, str]]] = None,
        documents: Optional[List[str]] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Build workflow configuration from natural language with streaming events."""
        
        try:
            # Emit start
            yield {
                "type": WorkflowStreamEvent.ANALYSIS_START.value,
                "data": {"message": "Starting workflow analysis...", "description": description[:100]}
            }
            
            # Call existing build method
            workflow_config = await self.build_from_description(description, user_context)
            self._current_workflow_spec = workflow_config
            
            # Emit agent events
            for i, agent in enumerate(workflow_config.get("agents", [])):
                yield {
                    "type": WorkflowStreamEvent.AGENT_CREATED.value,
                    "data": {
                        "name": agent["name"],
                        "role": agent.get("role", "custom"),
                        "instructions": agent.get("instructions", "")[:200] + "...",
                        "index": i
                    }
                }
                
                # Emit tool events
                for tool in agent.get("tools", []):
                    yield {
                        "type": WorkflowStreamEvent.TOOL_ADDED.value,
                        "data": {"agent": agent["name"], "tool": tool}
                    }
                
                await asyncio.sleep(0.1)  # Small delay for UI
            
            # Emit edge events
            edges = workflow_config.get("edges", [])
            if not edges and workflow_config.get("agents"):
                # Generate from connected_agents
                for agent in workflow_config["agents"]:
                    for target in agent.get("connected_agents", []):
                        edges.append({"from_agent": agent["name"], "to_agent": target})
            
            for edge in edges:
                yield {
                    "type": WorkflowStreamEvent.EDGE_CREATED.value,
                    "data": edge
                }
            
            # Complete
            yield {
                "type": WorkflowStreamEvent.WORKFLOW_COMPLETE.value,
                "data": {
                    "agent_count": len(workflow_config.get("agents", [])),
                    "tool_count": sum(len(a.get("tools", [])) for a in workflow_config.get("agents", [])),
                    "edge_count": len(edges)
                }
            }
            
        except Exception as e:
            yield {
                "type": WorkflowStreamEvent.ERROR.value,
                "data": {"error": str(e)}
            }

    async def get_workflow_definition(self) -> WorkflowDefinition:
        """Get the last built workflow definition"""
        if not self._current_workflow_spec:
            raise ValueError("No workflow has been built yet")
        return self._create_workflow_definition(self._current_workflow_spec)

    def _select_model_for_agent(self, agent_config: Dict[str, Any]) -> str:
        """Select appropriate model based on agent type and configuration"""
        
        # Check if model explicitly specified
        if agent_config.get("model"):
            return agent_config["model"]
        
        # Determine agent type from name/role/description
        agent_name = agent_config.get("name", "").lower()
        agent_role = agent_config.get("role", "").lower()
        instructions = agent_config.get("instructions", "").lower()
        
        # Determine agent type
        if any(keyword in agent_name + agent_role for keyword in ["hub", "main", "orchestrat", "coordinat"]):
            agent_type = "hub"
        elif any(keyword in agent_name + agent_role + instructions for keyword in ["info", "question", "answer", "search", "knowledge"]):
            agent_type = "info"
        elif any(keyword in agent_name + agent_role + instructions for keyword in ["code", "develop", "program", "script"]):
            agent_type = "code"
        elif agent_config.get("outputVisibility") == "internal":
            agent_type = "internal"
        elif any(keyword in agent_name + agent_role for keyword in ["process", "procedure", "workflow", "step"]):
            agent_type = "procedural"
        else:
            agent_type = "default"
        
        # Get model from config
        return self.rowboat_config.get_model_for_agent_type(agent_type)

    def _load_copilot_instructions(self) -> str:
        """Load ROWBOAT copilot instructions"""
        return """
    You are a helpful co-pilot for building multi-agent systems. 

    When creating agents:
    1. Decompose the problem into multiple smaller agents when needed
    2. Create hub agents for orchestration, info agents for Q&A, and procedural agents for workflows
    3. Set outputVisibility to "user_facing" for agents that talk to users, "internal" for background agents
    4. Use @agent mentions for handoffs and @tool mentions for tool calls
    5. Add examples in the format:
    - **User**: <message>
    - **Agent actions**: <actions if applicable>
    - **Agent response**: <response>

    Agent structure must include:
    - name: Agent name (no special characters)
    - instructions: Detailed step-by-step instructions
    - outputVisibility: "user_facing" or "internal"
    - controlType: "retain", "relinquish_to_parent", or "start_agent"
    - tools: List of tool IDs
    - prompts: List of prompt IDs
    - examples: List of example interactions
    - connected_agents: List of connected agent names
    - model: Model to use for the agent

    IMPORTANT: You must also create an "edges" array that defines the workflow connections between agents.
    The edges array should be at the root level of your JSON response, alongside "agents", "tools", and "prompts".

    Each edge in the edges array must have:
    - from_agent: The name of the source agent
    - to_agent: The name of the target agent

    For every agent that has connected_agents, create corresponding edges.
    For example, if "AgentA" has connected_agents: ["AgentB", "AgentC"], then create:
    - {"from_agent": "AgentA", "to_agent": "AgentB"}
    - {"from_agent": "AgentA", "to_agent": "AgentC"}

    Return a JSON object with this exact structure:
    {
        "agents": [
            {
                "name": "...",
                "instructions": "...",
                "outputVisibility": "...",
                "controlType": "...",
                "tools": [...],
                "prompts": [...],
                "examples": [...],
                "connected_agents": [...],
                "model": "..."
            }
        ],
        "tools": [...],
        "prompts": [...],
        "edges": [
            {"from_agent": "agent1", "to_agent": "agent2"},
            {"from_agent": "agent2", "to_agent": "agent3"}
        ],
        "startAgent": "name_of_start_agent"
    }

    Use model: {model} for all agents.

    Important:
    - The start agent should be user_facing and act as a hub
    - Break down complex tasks into multiple agents
    - Use internal agents for sub-tasks
    - Include clear handoff instructions with @agent mentions
    - ALWAYS include the edges array with proper connections
    """
    
    async def build_from_description(
        self,
        description: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build complete workflow from natural language description"""
        
        # Create the planning prompt
        planning_prompt = f"""Based on this request, create a multi-agent system:

Request: {description}

Context: {json.dumps(user_context) if user_context else "No additional context"}

{self.copilot_instructions}

Use model: {self.default_model} for all agents.

Important:
- The start agent should be user_facing and act as a hub
- Break down complex tasks into multiple agents
- Use internal agents for sub-tasks
- Include clear handoff instructions with @agent mentions
"""

        # Use ChutesOpenAIClient's generate method
        from src.ii_agent.llm.base import TextPrompt, TextResult
        
        messages = [[TextPrompt(text=planning_prompt)]]
        
        try:
            response_blocks, _ = self.llm.generate(
                messages=messages,
                max_tokens=4096,
                temperature=0.0,
                system_prompt="You are an expert at designing ROWBOAT multi-agent workflows. Always respond with valid JSON."
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
            
            # Parse JSON response
            try:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    workflow_config = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found in response")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}\nResponse: {response_text}")
                # Return minimal valid structure
                workflow_config = self._create_default_workflow(description)
            
            # Ensure required fields exist
            workflow_config = self._validate_workflow_config(workflow_config)
            
            return workflow_config
            
        except Exception as e:
            logger.error(f"Error in build_from_description: {e}")
            return self._create_default_workflow(description)
    
    def _create_default_workflow(self, description: str) -> Dict[str, Any]:
        """Create a default workflow structure"""
        return {
            "agents": [
                {
                    "name": "main_agent",
                    "instructions": f"You are a helpful assistant. Help the user with: {description}",
                    "outputVisibility": "user_facing",
                    "controlType": "retain",
                    "tools": [],
                    "prompts": [],
                    "examples": [
                        {
                            "input": "Hello",
                            "output": "Hello! I'm here to help. How can I assist you today?"
                        }
                    ],
                    "model": self.default_model
                }
            ],
            "tools": [],
            "prompts": [],
            "startAgent": "main_agent"
        }
    
    def _validate_workflow_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix workflow configuration to match ROWBOAT format"""
        
        # Ensure required top-level fields
        if "agents" not in config:
            config["agents"] = []
        if "tools" not in config:
            config["tools"] = []
        if "prompts" not in config:
            config["prompts"] = []
        
        # Ensure each agent has required fields
        for i, agent in enumerate(config.get("agents", [])):
            # Required fields
            if "name" not in agent:
                agent["name"] = f"agent_{i + 1}"
            if "instructions" not in agent:
                agent["instructions"] = "Follow user instructions"
            if "outputVisibility" not in agent:
                agent["outputVisibility"] = "user_facing"
            if "controlType" not in agent:
                agent["controlType"] = "retain"
            
            # Add role if missing (required by AgentConfig)
            if "role" not in agent:
                # Try to determine role from name or default to custom
                name_lower = agent["name"].lower()
                if "coordinat" in name_lower:
                    agent["role"] = "coordinator"
                elif "research" in name_lower:
                    agent["role"] = "researcher"
                elif "analyz" in name_lower or "analyst" in name_lower:
                    agent["role"] = "analyzer"
                elif "writ" in name_lower:
                    agent["role"] = "writer"
                else:
                    agent["role"] = "custom"
            
            # Add description if missing (required by AgentConfig)
            if "description" not in agent:
                agent["description"] = f"{agent['name']} - {agent.get('role', 'custom')} agent"
            
            # Optional fields with defaults
            if "tools" not in agent:
                agent["tools"] = []
            if "prompts" not in agent:
                agent["prompts"] = []
            if "examples" not in agent:
                agent["examples"] = []
            if "model" not in agent:
                agent["model"] = self._select_model_for_agent(agent)

            # Clean agent name (no special characters)
            agent["name"] = re.sub(r'[^a-zA-Z0-9_]', '_', agent["name"])
        
        # Generate edges from connected_agents if edges not provided
        if "edges" not in config:
            config["edges"] = []
            for agent in config["agents"]:
                if agent.get("connected_agents"):
                    for target in agent["connected_agents"]:
                        config["edges"].append({
                            "from_agent": agent["name"],
                            "to_agent": target
                        })
        
        # Set start agent if not specified
        if "startAgent" not in config and config["agents"]:
            # Find first user_facing agent or use first agent
            start_agent = None
            for agent in config["agents"]:
                if agent.get("outputVisibility") == "user_facing":
                    start_agent = agent["name"]
                    break
            if not start_agent:
                start_agent = config["agents"][0]["name"]
            config["startAgent"] = start_agent
        
        return config
    
    async def enhance_agent(
        self,
        agent_config: Dict[str, Any],
        enhancement_type: str = "improve"
    ) -> Dict[str, Any]:
        """Enhance an existing agent configuration"""
        
        enhancement_prompt = f"""Enhance this agent configuration:

Current config:
{json.dumps(agent_config, indent=2)}

Enhancement type: {enhancement_type}

Please improve the agent by:
1. Making instructions more detailed and clear
2. Adding relevant examples (5 examples)
3. Ensuring proper @agent and @tool mentions
4. Following ROWBOAT best practices

Return the enhanced agent configuration as JSON.
"""

        from src.ii_agent.llm.base import TextPrompt, TextResult
        messages = [[TextPrompt(text=enhancement_prompt)]]
        
        response_blocks, _ = self.llm.generate(
            messages=messages,
            max_tokens=2048,
            temperature=0.0,
            system_prompt="You are an expert at improving ROWBOAT agent configurations. Return valid JSON."
        )
        
        # Extract and parse response
        response_text = ""
        for block in response_blocks:
            if isinstance(block, TextResult):
                response_text += block.text
        
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                enhanced_config = json.loads(json_match.group())
                return enhanced_config
        except:
            logger.error("Failed to parse enhanced agent config")
            return agent_config
    
    def create_mock_tool(self, tool_name: str, description: str) -> Dict[str, Any]:
        """Create a mock tool configuration"""
        return {
            "id": f"mock_{tool_name}",
            "name": tool_name,
            "description": description,
            "type": "mock",
            "parameters": {}
        }
    
    def create_style_prompt(self, name: str, content: str) -> Dict[str, Any]:
        """Create a style prompt configuration"""
        return {
            "id": f"prompt_{name}",
            "name": name,
            "type": "style_prompt",
            "content": content
        }

    def _create_workflow_definition(self, workflow_spec: Dict[str, Any]) -> WorkflowDefinition:
        """Convert workflow spec to WorkflowDefinition object"""
        
        from src.ii_agent.workflows.definitions import (
            WorkflowDefinition, 
            AgentConfig, 
            AgentRole, 
            WorkflowEdge, 
            EdgeConditionType
        )
        from datetime import datetime
        
        # Extract agents and edges
        agents = workflow_spec.get("agents", [])
        edges_data = workflow_spec.get("edges", [])
        
        # If edges weren't provided, generate from connected_agents
        if not edges_data:
            for agent in agents:
                for target in agent.get("connected_agents", []):
                    edges_data.append({
                        "from_agent": agent["name"],
                        "to_agent": target,
                        "isMentionBased": True
                    })
        
        # Convert to AgentConfig objects
        agent_configs = []
        for agent in agents:
            # Determine role - map to valid AgentRole values
            role_str = agent.get("role", "custom").lower()
            
            # Map common role strings to valid enum values
            role_mapping = {
                "coordinator": AgentRole.COORDINATOR,
                "researcher": AgentRole.RESEARCHER,
                "analyzer": AgentRole.ANALYZER,
                "writer": AgentRole.WRITER,
                "reviewer": AgentRole.REVIEWER,
                "specialist": AgentRole.CUSTOM,
                "coder": AgentRole.CODER,
                "tester": AgentRole.TESTER,
                "vision_analyst": AgentRole.VISION_ANALYST,
                "customer_support": AgentRole.CUSTOMER_SUPPORT,
                "custom": AgentRole.CUSTOM
            }
            
            role = role_mapping.get(role_str, AgentRole.CUSTOM)
            
            # Create AgentConfig with all required fields
            agent_config = AgentConfig(
                name=agent["name"],
                role=role,
                description=agent.get("description", f"{agent['name']} - {role.value} agent"),
                instructions=agent.get("instructions", ""),
                tools=agent.get("tools", []),
                temperature=agent.get("temperature", 0.7),
                model=agent.get("model", self.default_model),
                mcp_servers=agent.get("mcp_servers", []),
                custom_prompts=agent.get("custom_prompts", {}),
                metadata={
                    "output_visibility": agent.get("outputVisibility", "user_facing"),
                    "control_type": agent.get("controlType", "retain"),
                    "connected_agents": agent.get("connected_agents", []),
                    "examples": agent.get("examples", []),
                    "max_calls_per_parent_agent": agent.get("maxCallsPerParentAgent", 5)
                }
            )
            agent_configs.append(agent_config)
        
        # Convert edges
        workflow_edges = []
        for edge_data in edges_data:
            # Determine edge type
            condition_type = EdgeConditionType.ALWAYS
            if edge_data.get("isMentionBased"):
                condition_type = EdgeConditionType.MENTION_BASED
            elif edge_data.get("condition"):
                condition_type = EdgeConditionType.CONDITIONAL
            
            edge = WorkflowEdge(
                from_agent=edge_data["from_agent"],
                to_agent=edge_data.get("to_agent"),
                to_agents=edge_data.get("to_agents"),  # For conditional edges
                condition_type=condition_type,
                condition=edge_data.get("condition"),
                data_transform=edge_data.get("data_transform"),
                metadata=edge_data.get("metadata", {})
            )
            workflow_edges.append(edge)
        
        # Create WorkflowDefinition with all required fields
        return WorkflowDefinition(
            name=workflow_spec.get("name", f"workflow_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"),
            description=workflow_spec.get("description", "Workflow created by ROWBOAT Copilot"),
            version=workflow_spec.get("version", "1.0.0"),
            agents=agent_configs,
            edges=workflow_edges,
            entry_point=workflow_spec.get("startAgent", agent_configs[0].name if agent_configs else "main"),
            end_conditions=workflow_spec.get("end_conditions"),
            global_state_schema=workflow_spec.get("global_state_schema", {}),
            timeout_seconds=workflow_spec.get("timeout_seconds", 3600),
            max_iterations=workflow_spec.get("max_iterations", 100),
            checkpointing=workflow_spec.get("checkpointing", False),
            metadata={
                "created_by": "rowboat_copilot",
                "created_at": datetime.utcnow().isoformat(),
                **workflow_spec.get("metadata", {})
            }
        )


# UI Component for Copilot (keeping the same UI)
class CopilotUI:
    """React component for copilot interface"""
    
    template = """
    import React, { useState } from 'react';
    import { Card, Button, Textarea, Progress } from '@/components/ui';
    
    // Define WorkflowPreview component
    const WorkflowPreview = ({ workflow, onEdit, onDeploy }) => {
        return (
            <Card className="workflow-preview">
                <h3>Workflow: {workflow.startAgent || 'Untitled'}</h3>
                <div className="agents-list">
                    <h4>Agents ({workflow.agents?.length || 0}):</h4>
                    {workflow.agents && workflow.agents.map((agent, index) => (
                        <div key={index} className="agent-item">
                            <strong>{agent.name}</strong>
                            <span className="visibility-badge">
                                {agent.outputVisibility}
                            </span>
                            <p>{agent.instructions?.substring(0, 100)}...</p>
                        </div>
                    ))}
                </div>
                {workflow.tools && workflow.tools.length > 0 && (
                    <div className="tools-list">
                        <h4>Tools ({workflow.tools.length}):</h4>
                        {workflow.tools.map((tool, index) => (
                            <span key={index} className="tool-chip">{tool.name}</span>
                        ))}
                    </div>
                )}
                <div className="actions">
                    <Button onClick={() => onEdit(workflow)}>Edit</Button>
                    <Button onClick={() => onDeploy(workflow)} variant="primary">Deploy</Button>
                </div>
            </Card>
        );
    };
    
    const WorkflowCopilot = () => {
        const [description, setDescription] = useState('');
        const [workflow, setWorkflow] = useState(null);
        const [loading, setLoading] = useState(false);
        const [stage, setStage] = useState('');
        
        const buildWorkflow = async () => {
            setLoading(true);
            
            // Show progress stages
            const stages = [
                'Analyzing requirements...',
                'Designing agent architecture...',
                'Generating instructions...',
                'Setting up connections...',
                'Configuring tools...'
            ];
            
            for (const stage of stages) {
                setStage(stage);
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
            
            try {
                const response = await window.api.copilot.buildFromDescription(description);
                setWorkflow(response);
            } catch (error) {
                console.error('Failed to build workflow:', error);
                alert('Failed to build workflow. Please try again.');
            } finally {
                setLoading(false);
            }
        };
        
        const deployWorkflow = async (workflow) => {
            try {
                const result = await window.api.workflows.deploy(workflow);
                alert(`Workflow deployed successfully! ID: ${result.workflow_id}`);
            } catch (error) {
                console.error('Failed to deploy workflow:', error);
                alert('Failed to deploy workflow. Please try again.');
            }
        };
        
        return (
            <div className="copilot-container">
                <Card className="copilot-input">
                    <h2>Describe Your Multi-Agent System</h2>
                    <Textarea
                        placeholder="E.g., Build me an assistant for a telecom company to handle data plan upgrades and billing inquiries"
                        value={description}
                        onChange={(e) => setDescription(e.target.value)}
                        rows={4}
                    />
                    <Button onClick={buildWorkflow} disabled={loading || !description}>
                        {loading ? 'Building...' : 'Build Workflow'}
                    </Button>
                    {loading && (
                        <div className="progress-section">
                            <Progress value={33} />
                            <p>{stage}</p>
                        </div>
                    )}
                </Card>
                
                {workflow && (
                    <WorkflowPreview 
                        workflow={workflow}
                        onEdit={(updatedWorkflow) => setWorkflow(updatedWorkflow)}
                        onDeploy={deployWorkflow}
                    />
                )}
            </div>
        );
    };
    
    export default WorkflowCopilot;
    """