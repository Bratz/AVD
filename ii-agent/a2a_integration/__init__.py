"""
Agent-to-Agent (A2A) Communication Integration
"""
from .client import (
    A2AClient,
    A2AConnectionPool,
    A2AMessageType,
    A2AMethod,
    A2AMessage,
    A2ARequest,
    A2AResponse,
    A2ANotification,
    AgentCard
)
from .server import A2AServer, A2ARegistry
from .models import (
    A2AProtocolVersion,
    A2AMessagePriority,
    A2ACapability,
    A2AErrorCode,
    A2AHeader,
    A2AEvent,
    A2ASubscription,
    A2AAuthToken,
    A2ABatch,
    A2AContract,
    A2AMetrics,
    A2AWorkflow,
    create_error_response,
    create_success_response,
    validate_agent_card
)
from .discovery import (
    A2ADiscoveryService,
    A2ADiscoveryClient,
    DiscoveryConfig
)

__all__ = [
    # Client
    'A2AClient',
    'A2AConnectionPool',
    'A2AMessageType',
    'A2AMethod',
    'A2AMessage',
    'A2ARequest',
    'A2AResponse',
    'A2ANotification',
    'AgentCard',
    # Server
    'A2AServer',
    'A2ARegistry',
    # Models
    'A2AProtocolVersion',
    'A2AMessagePriority',
    'A2ACapability',
    'A2AErrorCode',
    'A2AHeader',
    'A2AEvent',
    'A2ASubscription',
    'A2AAuthToken',
    'A2ABatch',
    'A2AContract',
    'A2AMetrics',
    'A2AWorkflow',
    'create_error_response',
    'create_success_response',
    'validate_agent_card',
    # Discovery
    'A2ADiscoveryService',
    'A2ADiscoveryClient',
    'DiscoveryConfig'
]

__version__ = '1.0.0'