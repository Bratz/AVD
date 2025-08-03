# src/ii_agent/copilot/__init__.py
"""
ROWBOAT Framework - AI-powered multi-agent workflow builder
Part of the II-Agent ecosystem
"""

from src.ii_agent.copilot.workflow_builder import WorkflowCopilot, CopilotUI
from src.ii_agent.copilot.client import IIAgentClient, StatefulChat

__all__ = [
    "WorkflowCopilot",
    "CopilotUI",
    "IIAgentClient", 
    "StatefulChat"
]

# Version info
__version__ = "1.0.0"
__author__ = "II-Agent Team"