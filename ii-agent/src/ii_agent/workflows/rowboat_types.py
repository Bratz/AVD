# ============================================================
# FILE 1: src/ii_agent/workflows/rowboat_types.py
# ============================================================
"""
ROWBOAT Type Definitions
Enums and data structures for multi-agent visibility and control flow
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional

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

class PromptType(Enum):
    """Prompt template types"""
    STYLE = "style_prompt"
    GREETING = "greeting"

class ErrorType(Enum):
    """Error handling types"""
    FATAL = "fatal"
    ESCALATE = "escalate"

class AgentRole(Enum):
    """Extended agent roles matching ROWBOAT"""
    # ROWBOAT-specific roles
    ESCALATION = "escalation"
    POST_PROCESSING = "post_process"
    GUARDRAILS = "guardrails"
    
    # ii-agent existing roles
    RESEARCHER = "researcher"
    ANALYZER = "analyzer"
    WRITER = "writer"
    REVIEWER = "reviewer"
    CODER = "coder"
    TESTER = "tester"
    COORDINATOR = "coordinator"
    CUSTOMER_SUPPORT = "customer_support"
    CUSTOM = "custom"

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
    
    # Additional ROWBOAT events
    WEB_SEARCH = "web_search"
    INTERNAL_MESSAGE = "internal_message"
    HANDOFF = "handoff"

@dataclass
class StreamEvent:
    """Streaming event structure"""
    type: str
    data: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)
    agent_name: Optional[str] = None
    visibility: Optional[OutputVisibility] = None
    response_type: Optional[ResponseType] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat()
        }
        
        if self.agent_name:
            result["agent_name"] = self.agent_name
        
        if self.visibility:
            result["visibility"] = self.visibility.value
            
        if self.response_type:
            result["response_type"] = self.response_type.value
            
        return result

@dataclass
class AgentTransfer:
    """Agent transfer details"""
    from_agent: str
    to_agent: str
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: Optional[Dict[str, Any]] = None

@dataclass
class TurnMetrics:
    """Metrics for a single turn execution"""
    turn_id: str
    workflow_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    agents_involved: List[str] = field(default_factory=list)
    message_count: int = 0
    internal_message_count: int = 0
    external_message_count: int = 0
    handoff_count: int = 0
    error_count: int = 0
    
    @property
    def duration_ms(self) -> Optional[int]:
        """Calculate duration in milliseconds"""
        if self.end_time:
            return int((self.end_time - self.start_time).total_seconds() * 1000)
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "turn_id": self.turn_id,
            "workflow_id": self.workflow_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "agents_involved": self.agents_involved,
            "message_count": self.message_count,
            "internal_message_count": self.internal_message_count,
            "external_message_count": self.external_message_count,
            "handoff_count": self.handoff_count,
            "error_count": self.error_count
        }

@dataclass
class WorkflowMetrics:
    """Aggregated workflow execution metrics"""
    workflow_id: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_duration_ms: float = 0.0
    average_handoffs: float = 0.0
    agent_usage: Dict[str, int] = field(default_factory=dict)
    
    def update_from_turn(self, turn_metrics: TurnMetrics):
        """Update workflow metrics from turn metrics"""
        self.total_executions += 1
        
        if turn_metrics.error_count == 0:
            self.successful_executions += 1
        else:
            self.failed_executions += 1
        
        # Update averages
        if turn_metrics.duration_ms:
            self.average_duration_ms = (
                (self.average_duration_ms * (self.total_executions - 1) + turn_metrics.duration_ms) 
                / self.total_executions
            )
        
        self.average_handoffs = (
            (self.average_handoffs * (self.total_executions - 1) + turn_metrics.handoff_count)
            / self.total_executions
        )
        
        # Update agent usage
        for agent in turn_metrics.agents_involved:
            self.agent_usage[agent] = self.agent_usage.get(agent, 0) + 1