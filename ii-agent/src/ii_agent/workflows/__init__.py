"""
BaNCS Workflow Management System
"""

from src.ii_agent.workflows.definitions import (
    AgentConfig,
    AgentRole,
    WorkflowEdge,
    WorkflowDefinition,
    EdgeConditionType,
    ToolType,
    WORKFLOW_TEMPLATES,
    get_workflow_template,
    list_workflow_templates
)

from src.ii_agent.workflows.langgraph_integration import (
    BaNCSLangGraphBridge,
    WorkflowState
)

__all__ = [
    # Definitions
    "AgentConfig",
    "AgentRole",
    "WorkflowEdge",
    "WorkflowDefinition",
    "EdgeConditionType",
    "ToolType",
    "WORKFLOW_TEMPLATES",
    "get_workflow_template",
    "list_workflow_templates",
    
    # LangGraph Integration
    "BaNCSLangGraphBridge",
    "WorkflowState",
]