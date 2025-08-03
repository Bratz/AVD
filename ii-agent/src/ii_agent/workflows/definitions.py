# src/ii_agent/workflows/definitions.py
"""
BaNCS Workflow Definitions
Models and schemas for multi-agent workflows
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
from datetime import datetime

from src.ii_agent.workflows.rowboat_types import OutputVisibility, ControlType


class AgentRole(str, Enum):
    """Predefined agent roles"""
    RESEARCHER = "researcher"
    ANALYZER = "analyzer"
    WRITER = "writer"
    REVIEWER = "reviewer"
    COORDINATOR = "coordinator"
    CODER = "coder"
    TESTER = "tester"
    VISION_ANALYST = "vision_analyst"
    CUSTOMER_SUPPORT = "customer_support"
    CUSTOM = "custom"

class ToolType(str, Enum):
    """Tool types"""
    NATIVE = "native"
    MCP = "mcp"
    CUSTOM = "custom"

class EdgeConditionType(str, Enum):
    """Types of edge conditions"""
    ALWAYS = "always"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    CONFIDENCE_BASED = "confidence_based"
    ERROR = "error"
    MENTION_BASED = "mention_based"  # Add this

class AgentConfig(BaseModel):
    """Configuration for individual agent in workflow"""
    name: str = Field(..., description="Unique agent identifier")
    role: AgentRole = Field(..., description="Agent's role in the workflow")
    description: str = Field(..., description="Agent's purpose and responsibilities")
    instructions: str = Field(..., description="Detailed instructions for the agent")
    model: Optional[str] = Field(None, description="Specific Chutes model to use")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int = Field(4096, gt=0, description="Maximum tokens for response")
    tools: List[str] = Field(default_factory=list, description="Native tools available to agent")
    mcp_servers: List[str] = Field(default_factory=list, description="MCP servers accessible by agent")
    custom_prompts: Dict[str, str] = Field(default_factory=dict, description="Custom prompt templates")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional agent metadata")
    
    # ROWBOAT enhancements - add these new fields
    output_visibility: Union[OutputVisibility, str] = Field(
        default=OutputVisibility.EXTERNAL,
        description="Agent visibility - internal agents don't show output to users"
    )
    control_type: Union[ControlType, str] = Field(
        default=ControlType.RETAIN,
        description="Control flow after agent execution"
    )
    max_calls_per_parent_agent: Optional[int] = Field(
        default=3,
        description="Maximum times this agent can be called by a parent in one turn"
    )
    connected_agents: List[str] = Field(
        default_factory=list,
        description="Agents this agent can hand off to via @mentions"
    )
    examples: List[Union[str, Dict[str, Any]]] = Field(
        default_factory=list,
        description="Example interactions for this agent"
    )
    # Add parent tracking
    parent_aware: bool = Field(
        default=True,
        description="Whether child agents are aware of their parent"
    )

    model_config = {
        "use_enum_values": True,  # This ensures enums are serialized as values
        "validate_assignment": True
    }


    @field_validator('examples', mode='before')
    @classmethod
    def validate_examples(cls, v):
        """Convert string examples to dict format if needed"""
        if not isinstance(v, list):
            return v
        
        normalized = []
        for example in v:
            if isinstance(example, str):
                # Parse string format into dict
                # This is a simple parser - adjust based on your needs
                parts = example.split('\n')
                example_dict = {
                    "input": "",
                    "actions": "",
                    "output": ""
                }
                
                for part in parts:
                    if "User" in part:
                        example_dict["input"] = part.split(": ", 1)[1].strip('"')
                    elif "Agent actions" in part:
                        example_dict["actions"] = part.split(": ", 1)[1].strip()
                    elif "Agent response" in part:
                        example_dict["output"] = part.split(": ", 1)[1].strip('"')
                
                normalized.append(example_dict)
            else:
                normalized.append(example)
        
        return normalized


    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure agent name is valid"""
        if not v or not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Agent name must be alphanumeric with underscores or hyphens")
        return v
    
    @field_validator('output_visibility', mode='before')
    @classmethod
    def validate_output_visibility(cls, v):
        """Handle both enum and string inputs"""
        if isinstance(v, OutputVisibility):
            return v.value
        return v
    
    @field_validator('control_type', mode='before')
    @classmethod
    def validate_control_type(cls, v):
        """Handle both enum and string inputs"""
        if isinstance(v, ControlType):
            return v.value
        return v

class WorkflowEdge(BaseModel):
    """Defines connection between agents"""
    from_agent: str = Field(..., description="Source agent name")
    to_agent: Optional[str] = Field(None, description="Target agent name (for simple edges)")
    to_agents: Optional[Dict[str, str]] = Field(None, description="Conditional targets")
    condition_type: EdgeConditionType = Field(EdgeConditionType.ALWAYS, description="Type of edge condition")
    condition: Optional[str] = Field(None, description="Python expression for conditional routing")
    data_transform: Optional[str] = Field(None, description="Function to transform data between agents")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional edge metadata")

    @model_validator(mode='after')
    def validate_edge_targets(self):
        """Ensure either to_agent or to_agents is specified"""
        if not self.to_agent and not self.to_agents:
            raise ValueError("Either to_agent or to_agents must be specified")
        return self

class WorkflowDefinition(BaseModel):
    """Complete workflow definition"""
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow purpose and overview")
    version: str = Field("1.0.0", description="Workflow version")
    agents: List[AgentConfig] = Field(..., min_items=1, description="Agents in the workflow")
    edges: List[WorkflowEdge] = Field(..., description="Connections between agents")
    entry_point: str = Field(..., description="Starting agent name")
    end_conditions: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="Workflow termination conditions")
    global_state_schema: Dict[str, Any] = Field(default_factory=dict, description="Schema for shared state")
    timeout_seconds: int = Field(3600, gt=0, description="Maximum workflow execution time")
    max_iterations: int = Field(100, gt=0, description="Maximum agent iterations")
    checkpointing: bool = Field(True, description="Enable workflow checkpointing")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional workflow metadata")
    
    @model_validator(mode='after')
    def validate_workflow(self):
        """Validate the entire workflow configuration"""
        agent_names = {agent.name for agent in self.agents}
        
        # Validate entry point
        if self.entry_point not in agent_names:
            raise ValueError(f"Entry point '{self.entry_point}' not found in agents")
        
        # Validate edges
        for edge in self.edges:
            if edge.from_agent not in agent_names:
                raise ValueError(f"Edge source '{edge.from_agent}' not found in agents")
            if edge.to_agent and edge.to_agent not in agent_names:
                raise ValueError(f"Edge target '{edge.to_agent}' not found in agents")
            if edge.to_agents:
                for target in edge.to_agents.values():
                    if target != "__end__" and target not in agent_names:
                        raise ValueError(f"Edge target '{target}' not found in agents")
        
        return self
    
# Predefined workflow templates
RESEARCH_WORKFLOW = WorkflowDefinition(
    name="research_pipeline",
    description="Multi-stage research and analysis workflow",
    version="1.0.0",
    agents=[
        AgentConfig(
            name="researcher",
            role=AgentRole.RESEARCHER,
            description="Gathers information from various sources",
            instructions="""Research the given topic thoroughly. Use web search, document analysis,
            and other available tools to gather comprehensive information. Focus on credible sources
            and recent data. Summarize findings clearly.""",
            tools=["web_search", "document_analyzer"],
            model="claude-3-sonnet-20240229"
        ),
        AgentConfig(
            name="analyzer",
            role=AgentRole.ANALYZER,
            description="Analyzes gathered information",
            instructions="""Analyze the research findings for patterns, insights, and key takeaways.
            Identify trends, contradictions, and gaps. Provide data-driven conclusions.""",
            tools=["data_analyzer", "chart_generator"],
            model="gpt-4-turbo-preview"
        ),
        AgentConfig(
            name="writer",
            role=AgentRole.WRITER,
            description="Creates comprehensive report",
            instructions="""Write a well-structured report based on the research and analysis.
            Include executive summary, detailed findings, conclusions, and recommendations.
            Ensure clarity and professional tone.""",
            tools=["document_generator"],
            model="claude-3-opus-20240229"
        ),
        AgentConfig(
            name="reviewer",
            role=AgentRole.REVIEWER,
            description="Reviews and improves the report",
            instructions="""Review the report for accuracy, completeness, and clarity.
            Check facts, improve structure, and ensure professional quality. Suggest improvements.""",
            model="gpt-4-turbo-preview"
        )
    ],
    edges=[
        WorkflowEdge(from_agent="researcher", to_agent="analyzer"),
        WorkflowEdge(from_agent="analyzer", to_agent="writer"),
        WorkflowEdge(from_agent="writer", to_agent="reviewer"),
        WorkflowEdge(
            from_agent="reviewer",
            condition_type=EdgeConditionType.CONDITIONAL,
            condition="needs_revision",
            to_agents={
                "True": "writer",
                "False": "__end__"
            }
        )
    ],
    entry_point="researcher"
)

VISION_ANALYSIS_WORKFLOW = WorkflowDefinition(
    name="vision_analysis_pipeline",
    description="Analyze images and generate comprehensive reports",
    version="1.0.0",
    agents=[
        AgentConfig(
            name="image_analyzer",
            role=AgentRole.VISION_ANALYST,
            description="Analyzes images using vision models",
            instructions="""Analyze the provided images in detail. Identify objects, scenes, text, and any notable features.
            Provide comprehensive descriptions and technical analysis.""",
            tools=["image_processing"],
            model="gpt-4-vision-preview"
        ),
        AgentConfig(
            name="caption_writer",
            role=AgentRole.WRITER,
            description="Writes descriptive captions",
            instructions="""Based on the image analysis, write clear, engaging captions that capture the essence of the images.
            Include alt-text for accessibility.""",
            model="claude-3-sonnet-20240229"
        ),
        AgentConfig(
            name="report_generator",
            role=AgentRole.WRITER,
            description="Generates comprehensive report",
            instructions="""Create a detailed report combining image analysis and captions.
            Structure the report with sections for each image and overall findings.""",
            tools=["document_generator"],
            model="claude-3-opus-20240229"
        )
    ],
    edges=[
        WorkflowEdge(from_agent="image_analyzer", to_agent="caption_writer"),
        WorkflowEdge(from_agent="caption_writer", to_agent="report_generator")
    ],
    entry_point="image_analyzer"
)

CODE_GENERATION_WORKFLOW = WorkflowDefinition(
    name="code_generation_pipeline",
    description="Design, implement, test, and document code",
    version="1.0.0",
    agents=[
        AgentConfig(
            name="architect",
            role=AgentRole.CODER,
            description="Designs system architecture",
            instructions="""Design the system architecture based on requirements. Create high-level design,
            identify components, define interfaces, and plan implementation approach.""",
            tools=["sequential_thinking"],
            model="claude-3-opus-20240229"
        ),
        AgentConfig(
            name="coder",
            role=AgentRole.CODER,
            description="Implements the code",
            instructions="""Implement the code following the architecture design. Write clean, efficient,
            and well-commented code. Follow best practices and coding standards.""",
            tools=["code_editor", "terminal"],
            model="gpt-4-turbo-preview"
        ),
        AgentConfig(
            name="tester",
            role=AgentRole.TESTER,
            description="Tests the implementation",
            instructions="""Write and execute tests for the implemented code. Include unit tests,
            integration tests, and edge cases. Report any issues found.""",
            tools=["code_editor", "terminal", "test_runner"],
            model="claude-3-sonnet-20240229"
        ),
        AgentConfig(
            name="documenter",
            role=AgentRole.WRITER,
            description="Creates documentation",
            instructions="""Create comprehensive documentation including API docs, usage examples,
            and deployment instructions. Ensure clarity and completeness.""",
            tools=["document_generator"],
            model="claude-3-sonnet-20240229"
        )
    ],
    edges=[
        WorkflowEdge(from_agent="architect", to_agent="coder"),
        WorkflowEdge(from_agent="coder", to_agent="tester"),
        WorkflowEdge(
            from_agent="tester",
            condition_type=EdgeConditionType.CONDITIONAL,
            condition="tests_passed",
            to_agents={
                "True": "documenter",
                "False": "coder"
            }
        )
    ],
    entry_point="architect",
    max_iterations=50
)

# Workflow templates registry
WORKFLOW_TEMPLATES = {
    "research": RESEARCH_WORKFLOW,
    "vision_analysis": VISION_ANALYSIS_WORKFLOW,
    "code_generation": CODE_GENERATION_WORKFLOW
}

def get_workflow_template(template_name: str) -> Optional[WorkflowDefinition]:
    """Get a predefined workflow template"""
    return WORKFLOW_TEMPLATES.get(template_name)

def list_workflow_templates() -> List[Dict[str, str]]:
    """List available workflow templates"""
    return [
        {
            "name": name,
            "description": workflow.description,
            "agent_count": len(workflow.agents)
        }
        for name, workflow in WORKFLOW_TEMPLATES.items()
    ]