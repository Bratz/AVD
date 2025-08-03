# src/ii_agent/workflows/rowboat_human_in_loop.py
"""
ROWBOAT Human-in-the-Loop System
Natural language configuration of human approval points with enhanced UX
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime
from enum import Enum
import logging
from dataclasses import dataclass

from src.ii_agent.workflows.definitions import WorkflowDefinition, WorkflowEdge, EdgeConditionType
from src.ii_agent.workflows.langgraph_integration import WorkflowState

logger = logging.getLogger(__name__)

class ApprovalStatus(Enum):
    """Status of approval requests"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    EXPIRED = "expired"
    DELEGATED = "delegated"

class ApprovalUrgency(Enum):
    """Urgency levels for approvals"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ApprovalRequest:
    """Enhanced approval request with ROWBOAT features"""
    id: str
    workflow_id: str
    agent_name: str
    agent_output: Dict[str, Any]
    context: Dict[str, Any]
    created_at: datetime
    expires_at: Optional[datetime]
    urgency: ApprovalUrgency
    natural_language_summary: str
    suggested_actions: List[Dict[str, Any]]
    approval_criteria: List[str]
    status: ApprovalStatus = ApprovalStatus.PENDING
    approver_id: Optional[str] = None
    approval_time: Optional[datetime] = None
    modifications: Optional[Dict[str, Any]] = None
    rejection_reason: Optional[str] = None
    delegation_target: Optional[str] = None

class ROWBOATHumanInLoopWorkflow:
    """ROWBOAT-enhanced workflow with natural language approval configuration"""
    
    def __init__(self, langgraph_bridge, llm_client=None):
        self.bridge = langgraph_bridge
        self.llm_client = llm_client
        self.pending_approvals: Dict[str, ApprovalRequest] = {}
        self.approval_history: List[ApprovalRequest] = []
        self.approval_callbacks: Dict[str, List[Callable]] = {}
        
        # ROWBOAT configuration
        self.config = {
            "auto_approve_threshold": 0.95,  # Confidence threshold for auto-approval
            "default_timeout_minutes": 60,
            "enable_delegation": True,
            "enable_batch_approvals": True,
            "natural_language_summaries": True
        }
        
        logger.info("ROWBOAT Human-in-the-Loop system initialized")
    
    async def configure_approvals_from_description(
        self,
        workflow_def: Union[WorkflowDefinition, Dict[str, Any]],
        description: str
    ) -> WorkflowDefinition:
        """Configure approval points from natural language description"""
        
        logger.info(f"Configuring approvals from: {description}")
        
        # Parse natural language to identify approval points
        approval_config = await self._parse_approval_requirements(description)
        
        # Apply approval configuration to workflow
        enhanced_workflow = self._apply_approval_configuration(
            workflow_def,
            approval_config
        )
        
        return enhanced_workflow
    
    async def _parse_approval_requirements(self, description: str) -> Dict[str, Any]:
        """Parse natural language to identify where approvals are needed"""
        
        if not self.llm_client:
            # Fallback to rule-based parsing
            return self._rule_based_approval_parsing(description)
        
        prompt = f"""Analyze this workflow description and identify where human approvals are needed.
        
        Description: {description}
        
        Identify:
        1. Which agents need approval before proceeding
        2. What criteria should trigger approval requirements
        3. Urgency level for each approval
        4. Any delegation rules
        
        Common patterns:
        - "require approval before..." -> approval needed
        - "critical decision" -> high urgency approval
        - "manager must approve" -> specific approver required
        - "can be delegated to..." -> delegation allowed
        
        Return as JSON with structure:
        {{
            "approval_points": [
                {{
                    "agent": "agent_name",
                    "condition": "always|conditional",
                    "criteria": ["list of criteria"],
                    "urgency": "normal|high|critical",
                    "approvers": ["list of approver roles"],
                    "delegation_allowed": true|false
                }}
            ]
        }}"""
        
        try:
            response = await self.llm_client.agenerate([prompt])
            result = response.generations[0][0].text.strip()
            # Parse JSON from response
            return json.loads(result)
        except Exception as e:
            logger.error(f"Failed to parse approval requirements: {e}")
            return self._rule_based_approval_parsing(description)
    
    def _rule_based_approval_parsing(self, description: str) -> Dict[str, Any]:
        """Fallback rule-based parsing for approval requirements"""
        
        description_lower = description.lower()
        approval_points = []
        
        # Common approval patterns
        patterns = {
            "approval": ApprovalUrgency.NORMAL,
            "review": ApprovalUrgency.NORMAL,
            "authorize": ApprovalUrgency.HIGH,
            "critical": ApprovalUrgency.CRITICAL,
            "validate": ApprovalUrgency.NORMAL,
            "confirm": ApprovalUrgency.NORMAL
        }
        
        for pattern, urgency in patterns.items():
            if pattern in description_lower:
                # Try to identify associated agent
                words = description_lower.split()
                pattern_index = description_lower.index(pattern)
                
                # Look for agent indicators before the pattern
                before_text = description_lower[:pattern_index].split()[-3:]
                agent_name = self._extract_agent_name(before_text)
                
                if agent_name:
                    approval_points.append({
                        "agent": agent_name,
                        "condition": "always",
                        "criteria": [f"Contains {pattern}"],
                        "urgency": urgency.value,
                        "approvers": ["user"],
                        "delegation_allowed": urgency != ApprovalUrgency.CRITICAL
                    })
        
        return {"approval_points": approval_points}
    
    def _extract_agent_name(self, words: List[str]) -> Optional[str]:
        """Extract potential agent name from words"""
        # Common agent role indicators
        role_keywords = ["agent", "analyzer", "writer", "researcher", "reviewer"]
        
        for word in words:
            for keyword in role_keywords:
                if keyword in word:
                    return word
        
        return None
    
    def _apply_approval_configuration(
        self,
        workflow_def: Union[WorkflowDefinition, Dict[str, Any]],
        approval_config: Dict[str, Any]
    ) -> WorkflowDefinition:
        """Apply approval configuration to workflow definition"""
        
        if isinstance(workflow_def, dict):
            workflow_def = WorkflowDefinition(**workflow_def)
        
        # Add approval metadata to agents
        for approval_point in approval_config.get("approval_points", []):
            agent_name = approval_point["agent"]
            
            # Find the agent in workflow
            for agent in workflow_def.agents:
                if agent.name == agent_name:
                    # Add approval metadata
                    agent.metadata["requires_approval"] = True
                    agent.metadata["approval_config"] = approval_point
                    
                    # Update instructions to mention approval
                    agent.instructions += f"\n\nNOTE: Output requires human approval before proceeding."
                    
                    # Add approval edge if not exists
                    self._add_approval_edge(workflow_def, agent_name)
        
        return workflow_def
    
    def _add_approval_edge(self, workflow_def: WorkflowDefinition, agent_name: str):
        """Add approval routing edge"""
        
        # Find edges from this agent
        for edge in workflow_def.edges:
            if edge.from_agent == agent_name:
                # Convert to conditional edge for approval
                edge.condition_type = EdgeConditionType.CONDITIONAL
                edge.condition = "approval_status"
                edge.to_agents = {
                    "approved": edge.to_agent or "__end__",
                    "rejected": f"{agent_name}_retry",
                    "modified": edge.to_agent or "__end__"
                }
    
    def create_workflow_with_approvals(
        self,
        workflow_def: Dict[str, Any],
        approval_points: Optional[List[str]] = None
    ) -> Any:  # Returns compiled workflow
        """Create workflow with human approval points"""
        
        # Auto-detect approval points if not specified
        if not approval_points:
            approval_points = self._detect_approval_points(workflow_def)
        
        # Create base workflow
        workflow = self.bridge.create_workflow(workflow_def)
        
        # Inject approval nodes with ROWBOAT enhancements
        for agent_name in approval_points:
            self._inject_enhanced_approval_node(workflow, agent_name)
        
        # Add batch approval handler
        if self.config["enable_batch_approvals"]:
            self._add_batch_approval_handler(workflow)
        
        return workflow
    
    def _detect_approval_points(self, workflow_def: Dict[str, Any]) -> List[str]:
        """Auto-detect which agents need approval"""
        approval_points = []
        
        for agent in workflow_def.get("agents", []):
            # Check agent metadata
            if agent.get("metadata", {}).get("requires_approval"):
                approval_points.append(agent["name"])
            # Check for high-impact roles
            elif agent.get("role") in ["reviewer", "approver", "validator"]:
                approval_points.append(agent["name"])
            # Check instructions for approval keywords
            elif any(word in agent.get("instructions", "").lower() 
                    for word in ["approve", "review", "validate", "authorize"]):
                approval_points.append(agent["name"])
        
        return approval_points
    
    def _inject_enhanced_approval_node(self, workflow: Any, after_agent: str):
        """Inject ROWBOAT-enhanced approval node"""
        
        approval_node_name = f"{after_agent}_approval"
        
        async def enhanced_approval_node(state: WorkflowState) -> WorkflowState:
            # Get agent output
            agent_output = state["agent_outputs"].get(after_agent, {})
            
            # Create enhanced approval request
            approval_request = await self._create_enhanced_approval_request(
                state, after_agent, agent_output
            )
            
            # Check for auto-approval conditions
            if await self._check_auto_approval(approval_request):
                approval_request.status = ApprovalStatus.APPROVED
                approval_request.approver_id = "auto_approver"
                approval_request.approval_time = datetime.utcnow()
                logger.info(f"Auto-approved request {approval_request.id}")
            else:
                # Wait for human approval
                approval_result = await self._wait_for_approval(approval_request)
                
                if approval_result["status"] == ApprovalStatus.APPROVED:
                    # Apply any modifications
                    if approval_result.get("modifications"):
                        state["agent_outputs"][after_agent].update(
                            approval_result["modifications"]
                        )
                elif approval_result["status"] == ApprovalStatus.REJECTED:
                    # Route to rejection handler or retry
                    state["next_agent"] = f"{after_agent}_rejection_handler"
                    state["rejection_reason"] = approval_result.get("reason", "")
            
            # Update state with approval info
            state["approval_history"] = state.get("approval_history", [])
            state["approval_history"].append({
                "agent": after_agent,
                "status": approval_request.status.value,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return state
        
        workflow.add_node(approval_node_name, enhanced_approval_node)
    
    async def _create_enhanced_approval_request(
        self,
        state: WorkflowState,
        agent_name: str,
        agent_output: Dict[str, Any]
    ) -> ApprovalRequest:
        """Create ROWBOAT-enhanced approval request"""
        
        # Generate natural language summary
        summary = await self._generate_approval_summary(agent_name, agent_output)
        
        # Generate suggested actions
        suggested_actions = await self._generate_suggested_actions(
            agent_name, agent_output, state
        )
        
        # Determine urgency
        urgency = self._determine_urgency(agent_name, agent_output, state)
        
        # Create approval request
        request = ApprovalRequest(
            id=f"approval_{agent_name}_{datetime.utcnow().timestamp()}",
            workflow_id=state.get("workflow_metadata", {}).get("workflow_id", ""),
            agent_name=agent_name,
            agent_output=agent_output,
            context={
                "previous_agents": list(state.get("agent_outputs", {}).keys()),
                "message_count": len(state.get("messages", [])),
                "workflow_metadata": state.get("workflow_metadata", {})
            },
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(
                minutes=self.config["default_timeout_minutes"]
            ),
            urgency=urgency,
            natural_language_summary=summary,
            suggested_actions=suggested_actions,
            approval_criteria=self._get_approval_criteria(agent_name)
        )
        
        # Store request
        self.pending_approvals[request.id] = request
        
        # Trigger callbacks
        await self._trigger_approval_callbacks(request)
        
        return request
    
    async def _generate_approval_summary(
        self,
        agent_name: str,
        agent_output: Dict[str, Any]
    ) -> str:
        """Generate natural language summary for approval"""
        
        if not self.config["natural_language_summaries"] or not self.llm_client:
            # Fallback to template-based summary
            return f"{agent_name} has completed its task and produced output requiring approval."
        
        prompt = f"""Summarize this agent output for human approval in 2-3 sentences.
        Focus on what decision needs to be made.
        
        Agent: {agent_name}
        Output: {json.dumps(agent_output, indent=2)[:500]}
        
        Summary:"""
        
        try:
            response = await self.llm_client.agenerate([prompt])
            return response.generations[0][0].text.strip()
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return f"{agent_name} has completed its task and produced output requiring approval."
    
    async def _generate_suggested_actions(
        self,
        agent_name: str,
        agent_output: Dict[str, Any],
        state: WorkflowState
    ) -> List[Dict[str, Any]]:
        """Generate suggested approval actions"""
        
        actions = [
            {
                "action": "approve",
                "label": "Approve as is",
                "description": "Accept the agent's output without changes"
            },
            {
                "action": "reject",
                "label": "Reject and retry",
                "description": "Send back to agent for revision"
            }
        ]
        
        # Add modification option if output is editable
        if isinstance(agent_output.get("output"), (str, dict)):
            actions.append({
                "action": "modify",
                "label": "Approve with changes",
                "description": "Make modifications before approving"
            })
        
        # Add delegation option if enabled
        if self.config["enable_delegation"]:
            actions.append({
                "action": "delegate",
                "label": "Delegate decision",
                "description": "Forward to another approver"
            })
        
        return actions
    
    def _determine_urgency(
        self,
        agent_name: str,
        agent_output: Dict[str, Any],
        state: WorkflowState
    ) -> ApprovalUrgency:
        """Determine approval urgency based on context"""
        
        # Check agent metadata for urgency
        agent_metadata = None
        for agent in state.get("workflow_metadata", {}).get("agents", []):
            if agent.get("name") == agent_name:
                agent_metadata = agent.get("metadata", {})
                break
        
        if agent_metadata:
            approval_config = agent_metadata.get("approval_config", {})
            urgency_str = approval_config.get("urgency", "normal")
            return ApprovalUrgency(urgency_str)
        
        # Default urgency rules
        if "critical" in agent_name.lower() or "urgent" in agent_name.lower():
            return ApprovalUrgency.CRITICAL
        elif "review" in agent_name.lower():
            return ApprovalUrgency.HIGH
        else:
            return ApprovalUrgency.NORMAL
    
    def _get_approval_criteria(self, agent_name: str) -> List[str]:
        """Get approval criteria for agent"""
        
        # Default criteria
        criteria = [
            "Output is accurate and complete",
            "Follows business rules and policies",
            "No sensitive information exposed"
        ]
        
        # Role-specific criteria
        if "financial" in agent_name.lower():
            criteria.append("Financial calculations are correct")
        elif "customer" in agent_name.lower():
            criteria.append("Customer communication is appropriate")
        elif "legal" in agent_name.lower():
            criteria.append("Complies with legal requirements")
        
        return criteria
    
    async def _check_auto_approval(self, request: ApprovalRequest) -> bool:
        """Check if request can be auto-approved"""
        
        # Don't auto-approve critical requests
        if request.urgency == ApprovalUrgency.CRITICAL:
            return False
        
        # Check confidence score if available
        confidence = request.agent_output.get("confidence_score", 0)
        if confidence >= self.config["auto_approve_threshold"]:
            logger.info(f"Auto-approving high confidence request: {confidence}")
            return True
        
        # Check for pre-approved patterns
        # This could be extended with more sophisticated rules
        
        return False
    
    async def _wait_for_approval(
        self,
        request: ApprovalRequest,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Wait for approval with timeout"""
        
        timeout = timeout or (self.config["default_timeout_minutes"] * 60)
        start_time = asyncio.get_event_loop().time()
        
        while True:
            # Check if approval received
            if request.status != ApprovalStatus.PENDING:
                return {
                    "status": request.status,
                    "modifications": request.modifications,
                    "reason": request.rejection_reason,
                    "approver": request.approver_id
                }
            
            # Check timeout
            if asyncio.get_event_loop().time() - start_time > timeout:
                request.status = ApprovalStatus.EXPIRED
                logger.warning(f"Approval request {request.id} expired")
                return {
                    "status": ApprovalStatus.EXPIRED,
                    "reason": "Approval request timed out"
                }
            
            # Wait a bit before checking again
            await asyncio.sleep(1)
    
    async def _trigger_approval_callbacks(self, request: ApprovalRequest):
        """Trigger registered approval callbacks"""
        
        callbacks = self.approval_callbacks.get(request.workflow_id, [])
        for callback in callbacks:
            try:
                await callback(request)
            except Exception as e:
                logger.error(f"Approval callback error: {e}")
    
    # Public API methods
    
    async def handle_approval(
        self,
        approval_id: str,
        approved: bool,
        approver_id: str,
        modifications: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None
    ):
        """Handle human approval response"""
        
        if approval_id not in self.pending_approvals:
            raise ValueError(f"Approval request {approval_id} not found")
        
        request = self.pending_approvals[approval_id]
        
        # Update request
        request.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        request.approver_id = approver_id
        request.approval_time = datetime.utcnow()
        request.modifications = modifications
        request.rejection_reason = reason
        
        # Move to history
        self.approval_history.append(request)
        
        logger.info(f"Approval {approval_id} handled: {request.status.value}")
    
    async def delegate_approval(
        self,
        approval_id: str,
        delegate_to: str,
        delegator_id: str,
        reason: Optional[str] = None
    ):
        """Delegate approval to another user"""
        
        if not self.config["enable_delegation"]:
            raise ValueError("Delegation is not enabled")
        
        if approval_id not in self.pending_approvals:
            raise ValueError(f"Approval request {approval_id} not found")
        
        request = self.pending_approvals[approval_id]
        request.status = ApprovalStatus.DELEGATED
        request.delegation_target = delegate_to
        
        # Create new approval request for delegate
        new_request = ApprovalRequest(
            **{
                **request.__dict__,
                "id": f"{request.id}_delegated",
                "status": ApprovalStatus.PENDING,
                "natural_language_summary": f"DELEGATED FROM {delegator_id}: {request.natural_language_summary}"
            }
        )
        
        self.pending_approvals[new_request.id] = new_request
        
        logger.info(f"Approval {approval_id} delegated to {delegate_to}")
    
    def get_pending_approvals(
        self,
        workflow_id: Optional[str] = None,
        approver_id: Optional[str] = None
    ) -> List[ApprovalRequest]:
        """Get pending approval requests"""
        
        requests = list(self.pending_approvals.values())
        
        if workflow_id:
            requests = [r for r in requests if r.workflow_id == workflow_id]
        
        if approver_id:
            # Filter by approver (would need approver assignment logic)
            pass
        
        # Sort by urgency and creation time
        urgency_order = {
            ApprovalUrgency.CRITICAL: 0,
            ApprovalUrgency.HIGH: 1,
            ApprovalUrgency.NORMAL: 2,
            ApprovalUrgency.LOW: 3
        }
        
        requests.sort(key=lambda r: (urgency_order[r.urgency], r.created_at))
        
        return requests
    
    async def batch_approve(
        self,
        approval_ids: List[str],
        approver_id: str,
        approved: bool = True
    ):
        """Handle multiple approvals at once"""
        
        if not self.config["enable_batch_approvals"]:
            raise ValueError("Batch approvals are not enabled")
        
        results = []
        for approval_id in approval_ids:
            try:
                await self.handle_approval(
                    approval_id=approval_id,
                    approved=approved,
                    approver_id=approver_id
                )
                results.append({"id": approval_id, "success": True})
            except Exception as e:
                results.append({"id": approval_id, "success": False, "error": str(e)})
        
        return results


# UI Components for approval interface
class ROWBOATApprovalUI:
    """React component templates for approval UI"""
    
    approval_widget_template = """
    import React, { useState, useEffect } from 'react';
    import { Card, Button, Alert, Badge, Textarea } from '@/components/ui';
    
    const ApprovalWidget = ({ request, onApprove, onReject, onDelegate }) => {
        const [showModification, setShowModification] = useState(false);
        const [modifications, setModifications] = useState('');
        const [delegateTarget, setDelegateTarget] = useState('');
        
        const urgencyColors = {
            critical: 'red',
            high: 'orange',
            normal: 'blue',
            low: 'gray'
        };
        
        return (
            <Card className="approval-request">
                <div className="approval-header">
                    <h3>{request.agent_name} - Approval Required</h3>
                    <Badge color={urgencyColors[request.urgency]}>
                        {request.urgency.toUpperCase()}
                    </Badge>
                </div>
                
                <div className="approval-summary">
                    <p>{request.natural_language_summary}</p>
                </div>
                
                <div className="approval-criteria">
                    <h4>Approval Criteria:</h4>
                    <ul>
                        {request.approval_criteria.map((criterion, i) => (
                            <li key={i}>
                                <input type="checkbox" id={`criterion-${i}`} />
                                <label htmlFor={`criterion-${i}`}>{criterion}</label>
                            </li>
                        ))}
                    </ul>
                </div>
                
                <div className="approval-output">
                    <h4>Agent Output:</h4>
                    <pre>{JSON.stringify(request.agent_output, null, 2)}</pre>
                </div>
                
                <div className="approval-actions">
                    {request.suggested_actions.map((action) => (
                        <Button
                            key={action.action}
                            onClick={() => handleAction(action.action)}
                            variant={action.action === 'approve' ? 'primary' : 'secondary'}
                        >
                            {action.label}
                        </Button>
                    ))}
                </div>
                
                {showModification && (
                    <div className="modification-panel">
                        <Textarea
                            value={modifications}
                            onChange={(e) => setModifications(e.target.value)}
                            placeholder="Enter modifications..."
                        />
                        <Button onClick={() => onApprove(request.id, modifications)}>
                            Approve with Changes
                        </Button>
                    </div>
                )}
            </Card>
        );
    };
    """


# WebSocket handler remains similar but with ROWBOAT enhancements
class ROWBOATApprovalWebSocketHandler:
    """Enhanced WebSocket handler for ROWBOAT approvals"""
    
    def __init__(self, approval_system: ROWBOATHumanInLoopWorkflow):
        self.approval_system = approval_system
        self.connections = {}
        self.user_subscriptions = {}  # User -> List[workflow_ids]
    
    async def handle_connection(self, websocket, path):
        """Handle WebSocket connection for approvals"""
        
        user_id = await self._authenticate(websocket)
        self.connections[user_id] = websocket
        
        try:
            # Send pending approvals on connect
            await self._send_pending_approvals(user_id)
            
            async for message in websocket:
                data = json.loads(message)
                await self._handle_message(user_id, data)
                
        finally:
            del self.connections[user_id]
            if user_id in self.user_subscriptions:
                del self.user_subscriptions[user_id]
    
    async def _handle_message(self, user_id: str, data: Dict[str, Any]):
        """Handle incoming WebSocket message"""
        
        msg_type = data.get("type")
        
        if msg_type == "approval_response":
            await self.approval_system.handle_approval(
                approval_id=data["approval_id"],
                approved=data["approved"],
                approver_id=user_id,
                modifications=data.get("modifications"),
                reason=data.get("reason")
            )
            
        elif msg_type == "delegate_approval":
            await self.approval_system.delegate_approval(
                approval_id=data["approval_id"],
                delegate_to=data["delegate_to"],
                delegator_id=user_id,
                reason=data.get("reason")
            )
            
        elif msg_type == "batch_approve":
            results = await self.approval_system.batch_approve(
                approval_ids=data["approval_ids"],
                approver_id=user_id,
                approved=data.get("approved", True)
            )
            await self._send_batch_results(user_id, results)
            
        elif msg_type == "subscribe_workflows":
            self.user_subscriptions[user_id] = data["workflow_ids"]
            await self._send_pending_approvals(user_id)
    
    async def _send_pending_approvals(self, user_id: str):
        """Send all pending approvals for user"""
        
        workflow_ids = self.user_subscriptions.get(user_id, [])
        
        for workflow_id in workflow_ids:
            approvals = self.approval_system.get_pending_approvals(
                workflow_id=workflow_id
            )
            
            for approval in approvals:
                await self._send_approval_request(user_id, approval)
    
    async def _send_approval_request(self, user_id: str, approval: ApprovalRequest):
        """Send approval request to user"""
        
        if user_id in self.connections:
            await self.connections[user_id].send(json.dumps({
                "type": "approval_request",
                "request": {
                    "id": approval.id,
                    "workflow_id": approval.workflow_id,
                    "agent_name": approval.agent_name,
                    "agent_output": approval.agent_output,
                    "urgency": approval.urgency.value,
                    "natural_language_summary": approval.natural_language_summary,
                    "suggested_actions": approval.suggested_actions,
                    "approval_criteria": approval.approval_criteria,
                    "created_at": approval.created_at.isoformat(),
                    "expires_at": approval.expires_at.isoformat() if approval.expires_at else None
                }
            }))
    
    async def _authenticate(self, websocket) -> str:
        """Authenticate WebSocket connection"""
        # Simplified authentication - would be more robust in production
        auth_message = await websocket.recv()
        auth_data = json.loads(auth_message)
        return auth_data.get("user_id", "anonymous")


# Import required for datetime
from datetime import timedelta