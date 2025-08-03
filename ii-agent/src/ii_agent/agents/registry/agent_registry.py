# src/ii_agent/agents/registry/agent_registry.py
"""
Agent Registry for ROWBOAT Multi-Agent Framework
Manages agent templates, configurations, and dynamic agent creation
"""

import logging
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass
import json
from datetime import datetime

from src.ii_agent.agents.base import BaseAgent
from src.ii_agent.workflows.definitions import AgentConfig, AgentRole
from src.ii_agent.workflows.rowboat_types import OutputVisibility, ControlType
from src.ii_agent.llm.chutes_openai import ChutesOpenAIClient
from src.ii_agent.llm.model_registry import ChutesModelRegistry

logger = logging.getLogger(__name__)

@dataclass
class AgentTemplate:
    """Template for creating agents"""
    name: str
    role: AgentRole
    description: str
    default_instructions: str
    default_model: Optional[str] = None
    default_tools: List[str] = None
    default_visibility: OutputVisibility = OutputVisibility.EXTERNAL
    default_control_type: ControlType = ControlType.RETAIN
    max_calls_per_parent: int = 5
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.default_tools is None:
            self.default_tools = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_agent_config(self, **overrides) -> AgentConfig:
        """Convert template to AgentConfig with overrides"""
        
        config_data = {
            "name": overrides.get("name", self.name),
            "role": self.role,
            "description": overrides.get("description", self.description),
            "instructions": overrides.get("instructions", self.default_instructions),
            "model": overrides.get("model", self.default_model),
            "temperature": overrides.get("temperature", 0.7),
            "tools": overrides.get("tools", self.default_tools.copy()),
            "mcp_servers": overrides.get("mcp_servers", []),
            "custom_prompts": overrides.get("custom_prompts", {}),
            "metadata": {
                **self.metadata,
                "output_visibility": overrides.get("output_visibility", self.default_visibility.value),
                "control_type": overrides.get("control_type", self.default_control_type.value),
                "max_calls_per_parent_agent": overrides.get("max_calls_per_parent", self.max_calls_per_parent),
                "connected_agents": overrides.get("connected_agents", []),
                "template_name": self.name
            }
        }
        
        return AgentConfig(**config_data)

class AgentRegistry:
    """Registry for managing agent templates and configurations"""
    
    def __init__(self):
        self.templates: Dict[str, AgentTemplate] = {}
        self.agent_cache: Dict[str, BaseAgent] = {}
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize default agent templates"""
        
        # Customer Support Templates
        self.register_template(AgentTemplate(
            name="triage_agent",
            role=AgentRole.CUSTOMER_SUPPORT,
            description="Initial customer contact and issue categorization",
            default_instructions="""You are a friendly customer support triage agent.
Your role is to:
1. Greet customers warmly
2. Understand their needs
3. Route to appropriate specialists using @mentions
4. Keep customers informed about the process""",
            default_tools=["customer_lookup", "ticket_create"],
            default_visibility=OutputVisibility.EXTERNAL,
            default_control_type=ControlType.RETAIN
        ))
        
        self.register_template(AgentTemplate(
            name="technical_support",
            role=AgentRole.ANALYZER,
            description="Technical issue analysis and resolution",
            default_instructions="""You are a technical support specialist.
Your role is to:
1. Analyze technical issues
2. Provide step-by-step solutions
3. Escalate complex issues if needed
4. Document solutions for future reference""",
            default_tools=["knowledge_search", "system_diagnostics"],
            default_visibility=OutputVisibility.INTERNAL,
            default_control_type=ControlType.PARENT_AGENT,
            max_calls_per_parent=3
        ))
        
        # Research Templates
        self.register_template(AgentTemplate(
            name="research_coordinator",
            role=AgentRole.COORDINATOR,
            description="Coordinate research tasks across specialists",
            default_instructions="""You are a research coordinator.
Your role is to:
1. Break down research requests into tasks
2. Delegate to appropriate specialists using @mentions
3. Synthesize findings into coherent responses
4. Ensure accuracy and completeness""",
            default_tools=["web_search", "rag_search"],
            default_visibility=OutputVisibility.EXTERNAL,
            default_control_type=ControlType.RETAIN
        ))
        
        self.register_template(AgentTemplate(
            name="data_analyst",
            role=AgentRole.ANALYZER,
            description="Analyze data and provide insights",
            default_instructions="""You are a data analyst.
Your role is to:
1. Analyze quantitative data
2. Identify patterns and trends
3. Create clear visualizations
4. Provide actionable insights""",
            default_tools=["data_query", "chart_generator"],
            default_visibility=OutputVisibility.INTERNAL,
            default_control_type=ControlType.PARENT_AGENT
        ))
        
        self.register_template(AgentTemplate(
            name="fact_checker",
            role=AgentRole.RESEARCHER,
            description="Verify facts and check sources",
            default_instructions="""You are a fact checker.
Your role is to:
1. Verify claims with reliable sources
2. Cross-reference information
3. Rate confidence levels
4. Provide source citations""",
            default_tools=["web_search", "fact_database"],
            default_visibility=OutputVisibility.INTERNAL,
            default_control_type=ControlType.PARENT_AGENT
        ))
        
        # Writing Templates
        self.register_template(AgentTemplate(
            name="content_writer",
            role=AgentRole.WRITER,
            description="Create engaging written content",
            default_instructions="""You are a content writer.
Your role is to:
1. Write clear, engaging content
2. Adapt tone to audience
3. Structure content logically
4. Ensure grammatical correctness""",
            default_tools=["grammar_check", "style_guide"],
            default_visibility=OutputVisibility.EXTERNAL,
            default_control_type=ControlType.RETAIN
        ))
        
        # Code Review Templates
        self.register_template(AgentTemplate(
            name="code_reviewer",
            role=AgentRole.REVIEWER,
            description="Review code for quality and standards",
            default_instructions="""You are a code reviewer.
Your role is to:
1. Review code for bugs and issues
2. Check adherence to coding standards
3. Suggest improvements
4. Delegate specialized checks using @mentions""",
            default_tools=["linter", "code_analyzer"],
            default_visibility=OutputVisibility.EXTERNAL,
            default_control_type=ControlType.RETAIN
        ))
        
        self.register_template(AgentTemplate(
            name="security_checker",
            role=AgentRole.ANALYZER,
            description="Check for security vulnerabilities",
            default_instructions="""You are a security specialist.
Your role is to:
1. Identify security vulnerabilities
2. Check for common attack vectors
3. Verify secure coding practices
4. Recommend security improvements""",
            default_tools=["security_scanner", "vulnerability_db"],
            default_visibility=OutputVisibility.INTERNAL,
            default_control_type=ControlType.PARENT_AGENT
        ))
        
        # Escalation Templates
        self.register_template(AgentTemplate(
            name="escalation_handler",
            role=AgentRole.ESCALATION,
            description="Handle escalated issues requiring human intervention",
            default_instructions="""You are an escalation handler.
Your role is to:
1. Review escalated issues
2. Determine if human intervention is needed
3. Prepare detailed context for handoff
4. Track escalation outcomes""",
            default_tools=["ticket_system", "notification_service"],
            default_visibility=OutputVisibility.EXTERNAL,
            default_control_type=ControlType.START_AGENT
        ))
        
        logger.info(f"Initialized {len(self.templates)} default agent templates")
    
    def register_template(self, template: AgentTemplate):
        """Register an agent template"""
        
        self.templates[template.name] = template
        logger.info(f"Registered agent template: {template.name}")
    
    def get_template(self, name: str) -> Optional[AgentTemplate]:
        """Get an agent template by name"""
        
        return self.templates.get(name)
    
    def list_templates(self) -> List[str]:
        """List all available template names"""
        
        return list(self.templates.keys())
    
    def get_templates_by_role(self, role: AgentRole) -> List[AgentTemplate]:
        """Get all templates for a specific role"""
        
        return [t for t in self.templates.values() if t.role == role]
    
    def create_agent_config(
        self,
        template_name: str,
        instance_name: Optional[str] = None,
        **overrides
    ) -> AgentConfig:
        """Create an agent configuration from a template"""
        
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Unknown template: {template_name}")
        
        # Generate unique instance name if not provided
        if not instance_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            instance_name = f"{template.name}_{timestamp}"
        
        overrides["name"] = instance_name
        
        return template.to_agent_config(**overrides)
    
    def create_agent(
        self,
        config: AgentConfig,
        llm_client: Optional[ChutesOpenAIClient] = None,
        shared_context: Optional[Dict[str, Any]] = None
    ) -> BaseAgent:
        """Create an agent instance from configuration"""
        
        # Get LLM client
        if not llm_client:
            if config.model:
                llm_client = ChutesModelRegistry.create_llm_client(
                    model_key=config.model,
                    use_native_tools=True
                )
            else:
                # Get default model for role
                model_id = ChutesModelRegistry.get_model_for_role(config.role.value)
                llm_client = ChutesOpenAIClient(
                    model_name=model_id,
                    use_native_tool_calling=True
                )
        
        # Build agent configuration
        agent_config = {
            "name": config.name,
            "role": config.role,
            "instructions": config.instructions,
            "llm_client": llm_client,
            "temperature": config.temperature,
            "tools": config.tools,  # Tools should be resolved by coordinator
            "shared_context": shared_context or {},
            "metadata": config.metadata
        }
        
        # Create agent
        agent = BaseAgent(**agent_config)
        
        # Add ROWBOAT attributes
        agent.output_visibility = config.metadata.get(
            "output_visibility",
            OutputVisibility.EXTERNAL
        )
        agent.control_type = config.metadata.get(
            "control_type",
            ControlType.RETAIN
        )
        agent.max_calls_per_parent_agent = config.metadata.get(
            "max_calls_per_parent_agent",
            5
        )
        agent.connected_agents = config.metadata.get("connected_agents", [])
        
        # Cache agent
        self.agent_cache[config.name] = agent
        
        return agent
    
    def get_cached_agent(self, name: str) -> Optional[BaseAgent]:
        """Get a cached agent instance"""
        
        return self.agent_cache.get(name)
    
    def clear_cache(self):
        """Clear the agent cache"""
        
        self.agent_cache.clear()
    
    def export_templates(self) -> Dict[str, Any]:
        """Export all templates as JSON-serializable dict"""
        
        return {
            name: {
                "name": template.name,
                "role": template.role.value,
                "description": template.description,
                "default_instructions": template.default_instructions,
                "default_model": template.default_model,
                "default_tools": template.default_tools,
                "default_visibility": template.default_visibility.value,
                "default_control_type": template.default_control_type.value,
                "max_calls_per_parent": template.max_calls_per_parent,
                "metadata": template.metadata
            }
            for name, template in self.templates.items()
        }
    
    def import_templates(self, templates_data: Dict[str, Any]):
        """Import templates from JSON data"""
        
        for name, data in templates_data.items():
            template = AgentTemplate(
                name=data["name"],
                role=AgentRole(data["role"]),
                description=data["description"],
                default_instructions=data["default_instructions"],
                default_model=data.get("default_model"),
                default_tools=data.get("default_tools", []),
                default_visibility=OutputVisibility(data.get("default_visibility", OutputVisibility.EXTERNAL.value)),
                default_control_type=ControlType(data.get("default_control_type", ControlType.RETAIN.value)),
                max_calls_per_parent=data.get("max_calls_per_parent", 5),
                metadata=data.get("metadata", {})
            )
            self.register_template(template)
    
    def create_workflow_agents(
        self,
        workflow_config: Dict[str, Any],
        shared_context: Optional[Dict[str, Any]] = None
    ) -> List[BaseAgent]:
        """Create all agents for a workflow"""
        
        agents = []
        
        for agent_data in workflow_config.get("agents", []):
            # Check if using a template
            if "template" in agent_data:
                config = self.create_agent_config(
                    template_name=agent_data["template"],
                    instance_name=agent_data.get("name"),
                    **agent_data.get("overrides", {})
                )
            else:
                # Create config directly
                config = AgentConfig(**agent_data)
            
            agent = self.create_agent(config, shared_context=shared_context)
            agents.append(agent)
        
        return agents

# Global registry instance
agent_registry = AgentRegistry()

# Convenience functions
def get_agent_template(name: str) -> Optional[AgentTemplate]:
    """Get an agent template from the global registry"""
    return agent_registry.get_template(name)

def create_agent_from_template(
    template_name: str,
    instance_name: Optional[str] = None,
    **overrides
) -> BaseAgent:
    """Create an agent from a template in the global registry"""
    
    config = agent_registry.create_agent_config(
        template_name,
        instance_name,
        **overrides
    )
    return agent_registry.create_agent(config)

def list_available_templates() -> List[str]:
    """List all available agent templates"""
    return agent_registry.list_templates()
