"""
A2A Protocol Models and Data Structures
Implements data models for Agent-to-Agent communication
"""
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class A2AProtocolVersion(str, Enum):
    """Supported A2A protocol versions"""
    V1_0 = "1.0"
    V2_0 = "2.0"


class A2AMessagePriority(str, Enum):
    """Message priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class A2ACapability(str, Enum):
    """Standard agent capabilities"""
    TASK_EXECUTION = "task_execution"
    INFORMATION_QUERY = "information_query"
    EVENT_STREAMING = "event_streaming"
    BATCH_PROCESSING = "batch_processing"
    REAL_TIME_PROCESSING = "real_time_processing"
    DATA_TRANSFORMATION = "data_transformation"
    DECISION_MAKING = "decision_making"
    LEARNING = "learning"
    PLANNING = "planning"
    MONITORING = "monitoring"


class A2AErrorCode(int, Enum):
    """Standard A2A error codes"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    SERVER_ERROR = -32000
    AGENT_UNAVAILABLE = -32001
    CAPABILITY_NOT_SUPPORTED = -32002
    AUTHENTICATION_FAILED = -32003
    AUTHORIZATION_FAILED = -32004
    TIMEOUT = -32005
    RESOURCE_LIMIT_EXCEEDED = -32006


class A2AHeader(BaseModel):
    """Standard A2A message header"""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    source_agent: str
    target_agent: Optional[str] = None
    priority: A2AMessagePriority = A2AMessagePriority.NORMAL
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    expires_at: Optional[datetime] = None
    protocol_version: A2AProtocolVersion = A2AProtocolVersion.V2_0
    
    @validator('expires_at')
    def validate_expiry(cls, v, values):
        if v and v <= values.get('timestamp', datetime.now()):
            raise ValueError('expires_at must be in the future')
        return v


class A2ARequest(BaseModel):
    """A2A request message structure"""
    jsonrpc: str = "2.0"
    id: Union[str, int]
    method: str
    params: Dict[str, Any] = Field(default_factory=dict)
    header: Optional[A2AHeader] = None
    
    class Config:
        schema_extra = {
            "example": {
                "jsonrpc": "2.0",
                "id": "req_123",
                "method": "execute",
                "params": {
                    "task": "analyze_data",
                    "data": {"values": [1, 2, 3]}
                }
            }
        }


class A2AResponse(BaseModel):
    """A2A response message structure"""
    jsonrpc: str = "2.0"
    id: Union[str, int]
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    header: Optional[A2AHeader] = None
    
    @validator('error')
    def validate_error(cls, v):
        if v:
            required_fields = {'code', 'message'}
            if not required_fields.issubset(v.keys()):
                raise ValueError('Error must contain code and message fields')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "jsonrpc": "2.0",
                "id": "req_123",
                "result": {
                    "status": "success",
                    "data": {"analysis": "completed"}
                }
            }
        }


class A2ANotification(BaseModel):
    """A2A notification message (no response expected)"""
    jsonrpc: str = "2.0"
    method: str
    params: Dict[str, Any] = Field(default_factory=dict)
    header: Optional[A2AHeader] = None


class A2AEvent(BaseModel):
    """Event structure for A2A event streaming"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    agent_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class A2ASubscription(BaseModel):
    """Subscription request structure"""
    subscription_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subscriber_agent: str
    publisher_agent: str
    event_types: List[str]
    filter: Optional[Dict[str, Any]] = None
    callback_url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    active: bool = True


class AgentCard(BaseModel):
    """Agent discovery card with enhanced metadata"""
    agent_id: str
    name: str
    description: str
    version: str = "1.0.0"
    
    # Capabilities
    capabilities: List[str] = Field(default_factory=list)
    supported_methods: List[str] = Field(default_factory=list)
    
    # Endpoints
    endpoints: Dict[str, str] = Field(default_factory=dict)
    
    # Authentication
    auth_required: bool = False
    auth_methods: List[str] = Field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    
    # Status
    status: str = "active"
    last_seen: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)
    
    # Performance hints
    performance: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "agent_id": "agent_123",
                "name": "DataAnalyzer",
                "description": "Agent for data analysis tasks",
                "version": "1.0.0",
                "capabilities": ["data_analysis", "visualization"],
                "supported_methods": ["execute", "query", "subscribe"],
                "endpoints": {
                    "rpc": "http://localhost:8001/a2a/rpc",
                    "websocket": "ws://localhost:8001/a2a/ws"
                },
                "auth_required": True,
                "auth_methods": ["bearer", "api_key"],
                "tags": ["analytics", "reporting"],
                "performance": {
                    "avg_response_time_ms": 150,
                    "max_concurrent_requests": 100
                }
            }
        }


class A2AAuthToken(BaseModel):
    """Authentication token for A2A communication"""
    token: str
    token_type: str = "Bearer"
    expires_at: datetime
    agent_id: str
    permissions: List[str] = Field(default_factory=list)
    
    @property
    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at


class A2ABatch(BaseModel):
    """Batch request structure for multiple operations"""
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    requests: List[A2ARequest]
    parallel: bool = True
    stop_on_error: bool = False
    timeout_seconds: Optional[int] = 300


class A2AContract(BaseModel):
    """Service contract between agents"""
    contract_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    provider_agent: str
    consumer_agent: str
    capabilities: List[str]
    sla: Dict[str, Any] = Field(default_factory=dict)
    valid_from: datetime = Field(default_factory=datetime.now)
    valid_until: Optional[datetime] = None
    terms: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "contract_id": "contract_456",
                "provider_agent": "data_processor",
                "consumer_agent": "analytics_engine",
                "capabilities": ["data_transformation", "batch_processing"],
                "sla": {
                    "response_time_ms": 1000,
                    "availability": 0.99,
                    "throughput_rps": 100
                },
                "terms": {
                    "rate_limit": "1000 requests/hour",
                    "data_retention": "7 days"
                }
            }
        }


class A2AMetrics(BaseModel):
    """Performance metrics for A2A communication"""
    agent_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Performance metrics
    avg_response_time_ms: float = 0.0
    max_response_time_ms: float = 0.0
    min_response_time_ms: float = 0.0
    
    # Resource metrics
    active_connections: int = 0
    queued_requests: int = 0
    
    # Error metrics
    error_rate: float = 0.0
    timeout_count: int = 0
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests


class A2AWorkflow(BaseModel):
    """Multi-agent workflow definition"""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    
    # Workflow steps
    steps: List[Dict[str, Any]]
    
    # Agent assignments
    agent_roles: Dict[str, str]
    
    # Execution settings
    timeout_seconds: Optional[int] = 3600
    retry_policy: Dict[str, Any] = Field(default_factory=dict)
    
    # Status tracking
    status: str = "draft"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        schema_extra = {
            "example": {
                "workflow_id": "wf_789",
                "name": "Customer Onboarding",
                "description": "Complete customer onboarding workflow",
                "steps": [
                    {
                        "step_id": "1",
                        "name": "KYC Verification",
                        "agent_role": "kyc_agent",
                        "inputs": ["customer_data"],
                        "outputs": ["verification_result"]
                    },
                    {
                        "step_id": "2",
                        "name": "Account Creation",
                        "agent_role": "account_agent",
                        "inputs": ["verification_result"],
                        "outputs": ["account_details"]
                    }
                ],
                "agent_roles": {
                    "kyc_agent": "agent_kyc_001",
                    "account_agent": "agent_account_002"
                },
                "retry_policy": {
                    "max_attempts": 3,
                    "backoff_seconds": 60
                }
            }
        }


# Utility functions for A2A protocol

def create_error_response(
    request_id: Union[str, int],
    code: A2AErrorCode,
    message: str,
    data: Optional[Dict[str, Any]] = None
) -> A2AResponse:
    """Create a standard error response"""
    error = {
        "code": code.value,
        "message": message
    }
    if data:
        error["data"] = data
    
    return A2AResponse(
        id=request_id,
        error=error
    )


def create_success_response(
    request_id: Union[str, int],
    result: Dict[str, Any]
) -> A2AResponse:
    """Create a standard success response"""
    return A2AResponse(
        id=request_id,
        result=result
    )


def validate_agent_card(card: Dict[str, Any]) -> bool:
    """Validate agent card structure"""
    try:
        AgentCard(**card)
        return True
    except Exception:
        return False