# src/ii_agent/learning/feedback_loop.py
"""
Continuous Learning System for II-Agent
Collects feedback and optimizes workflows based on user interactions
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
import json
import asyncio
from abc import ABC, abstractmethod

class OptimizationType(str, Enum):
    """Types of optimizations that can be applied"""
    PROMPT_REFINEMENT = "prompt_refinement"
    TOOL_ADDITION = "tool_addition"
    WORKFLOW_RESTRUCTURE = "workflow_restructure"
    PARAMETER_TUNING = "parameter_tuning"
    AGENT_REPLACEMENT = "agent_replacement"

class RiskLevel(str, Enum):
    """Risk levels for optimizations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class FeedbackStore:
    """Store for user feedback data"""
    
    def __init__(self, storage_backend: Optional[str] = "memory"):
        self.storage_backend = storage_backend
        self.feedback_data: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
    
    async def save(self, feedback_entry: Dict[str, Any]) -> str:
        """Save feedback entry and return ID"""
        async with self._lock:
            feedback_id = f"feedback_{len(self.feedback_data)}_{datetime.utcnow().timestamp()}"
            feedback_entry["id"] = feedback_id
            self.feedback_data.append(feedback_entry)
            return feedback_id
    
    async def count(self, workflow_id: str) -> int:
        """Count feedback entries for a workflow"""
        async with self._lock:
            return sum(1 for entry in self.feedback_data 
                      if entry.get("workflow_id") == workflow_id)
    
    async def get_recent(self, workflow_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent feedback entries for a workflow"""
        async with self._lock:
            workflow_feedback = [
                entry for entry in self.feedback_data
                if entry.get("workflow_id") == workflow_id
            ]
            # Sort by timestamp (most recent first)
            workflow_feedback.sort(
                key=lambda x: x.get("timestamp", datetime.min),
                reverse=True
            )
            return workflow_feedback[:limit]

class FeedbackAnalyzer:
    """Analyze feedback patterns and generate insights"""
    
    def __init__(self):
        self.analysis_patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[str, Any]:
        """Initialize analysis patterns"""
        return {
            "low_ratings": {"threshold": 3, "action": "investigate"},
            "repeated_issues": {"threshold": 3, "action": "optimize"},
            "tool_requests": {"threshold": 5, "action": "add_tool"},
            "performance_complaints": {"threshold": 3, "action": "tune_parameters"}
        }
    
    async def analyze_feedback(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze feedback data and identify patterns"""
        analysis = {
            "total_feedback": len(feedback_data),
            "average_rating": 0.0,
            "common_issues": [],
            "improvement_areas": [],
            "positive_aspects": [],
            "patterns_detected": []
        }
        
        if not feedback_data:
            return analysis
        
        # Calculate average rating
        ratings = [
            entry["feedback"].get("rating", 0) 
            for entry in feedback_data 
            if entry.get("feedback", {}).get("rating")
        ]
        if ratings:
            analysis["average_rating"] = sum(ratings) / len(ratings)
        
        # Analyze text feedback
        all_feedback_text = " ".join([
            entry.get("feedback", {}).get("feedback", "") 
            for entry in feedback_data
        ])
        
        # Simple keyword analysis (in production, use NLP)
        issue_keywords = ["problem", "issue", "error", "slow", "failed", "confused"]
        positive_keywords = ["great", "excellent", "fast", "helpful", "perfect", "good"]
        
        for keyword in issue_keywords:
            count = all_feedback_text.lower().count(keyword)
            if count > 2:
                analysis["common_issues"].append({
                    "keyword": keyword,
                    "frequency": count
                })
        
        for keyword in positive_keywords:
            count = all_feedback_text.lower().count(keyword)
            if count > 2:
                analysis["positive_aspects"].append({
                    "keyword": keyword,
                    "frequency": count
                })
        
        # Detect patterns
        if analysis["average_rating"] < 3:
            analysis["patterns_detected"].append("low_satisfaction")
        
        if len(analysis["common_issues"]) > 3:
            analysis["patterns_detected"].append("multiple_issues")
        
        return analysis

class WorkflowOptimizer:
    """Generate optimization suggestions based on analysis"""
    
    async def generate_suggestions(
        self,
        workflow_id: str,
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate optimization suggestions"""
        suggestions = []
        
        # Low satisfaction pattern
        if "low_satisfaction" in analysis.get("patterns_detected", []):
            suggestions.append({
                "type": OptimizationType.PROMPT_REFINEMENT,
                "description": "Refine agent prompts for better clarity",
                "confidence": 0.85,
                "risk": RiskLevel.LOW,
                "expected_improvement": 0.2,
                "details": {
                    "action": "update_prompts",
                    "target_agents": ["all"]
                }
            })
        
        # Performance issues
        performance_issues = [
            issue for issue in analysis.get("common_issues", [])
            if issue["keyword"] in ["slow", "timeout", "performance"]
        ]
        if performance_issues:
            suggestions.append({
                "type": OptimizationType.PARAMETER_TUNING,
                "description": "Optimize performance parameters",
                "confidence": 0.9,
                "risk": RiskLevel.LOW,
                "expected_improvement": 0.3,
                "details": {
                    "action": "tune_parameters",
                    "parameters": ["timeout", "max_tokens", "temperature"]
                }
            })
        
        # Tool-related issues
        tool_issues = [
            issue for issue in analysis.get("common_issues", [])
            if "tool" in issue["keyword"] or "feature" in issue["keyword"]
        ]
        if tool_issues:
            suggestions.append({
                "type": OptimizationType.TOOL_ADDITION,
                "description": "Add additional tools to agents",
                "confidence": 0.75,
                "risk": RiskLevel.MEDIUM,
                "expected_improvement": 0.25,
                "details": {
                    "action": "add_tools",
                    "recommended_tools": ["web_search", "calculator", "file_manager"]
                }
            })
        
        return suggestions

class ContinuousLearningSystem:
    """Learn and improve from user interactions"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.feedback_store = FeedbackStore()
        self.analyzer = FeedbackAnalyzer()
        self.optimizer = WorkflowOptimizer()
        self.optimization_queue: List[Dict[str, Any]] = []
        self._execution_contexts: Dict[str, Dict[str, Any]] = {}
    
    async def collect_feedback(
        self,
        workflow_id: str,
        execution_id: str,
        feedback: Dict[str, Any]
    ):
        """Collect user feedback"""
        
        await self.feedback_store.save({
            "workflow_id": workflow_id,
            "execution_id": execution_id,
            "feedback": feedback,
            "timestamp": datetime.utcnow(),
            "context": await self._get_execution_context(execution_id)
        })
        
        # Trigger analysis if enough feedback collected
        feedback_count = await self.feedback_store.count(workflow_id)
        if feedback_count % 10 == 0:  # Every 10 feedbacks
            await self.analyze_and_optimize(workflow_id)
    
    async def analyze_and_optimize(self, workflow_id: str):
        """Analyze feedback and optimize workflow"""
        
        # Get recent feedback
        feedback_data = await self.feedback_store.get_recent(
            workflow_id,
            limit=100
        )
        
        # Analyze patterns
        analysis = await self.analyzer.analyze_feedback(feedback_data)
        
        # Generate optimization suggestions
        suggestions = await self.optimizer.generate_suggestions(
            workflow_id,
            analysis
        )
        
        # Apply automatic optimizations
        for suggestion in suggestions:
            if suggestion["confidence"] > 0.8 and suggestion["risk"] == RiskLevel.LOW:
                await self._apply_optimization(workflow_id, suggestion)
            else:
                # Queue for human review
                await self._queue_for_review(workflow_id, suggestion)
    
    async def _get_execution_context(self, execution_id: str) -> Dict[str, Any]:
        """Get execution context for feedback"""
        return self._execution_contexts.get(execution_id, {
            "execution_id": execution_id,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def _apply_optimization(
        self,
        workflow_id: str,
        suggestion: Dict[str, Any]
    ):
        """Apply optimization to workflow"""
        
        optimization_type = suggestion["type"]
        
        if optimization_type == OptimizationType.PROMPT_REFINEMENT:
            # Update agent prompts
            await self.update_agent_prompt(
                workflow_id,
                suggestion["details"]["target_agents"],
                suggestion.get("new_prompt", "")
            )
        elif optimization_type == OptimizationType.TOOL_ADDITION:
            # Add recommended tools
            for tool in suggestion["details"]["recommended_tools"]:
                await self.add_tool_to_agent(
                    workflow_id,
                    "all",  # Add to all agents initially
                    tool
                )
        elif optimization_type == OptimizationType.WORKFLOW_RESTRUCTURE:
            # Restructure agent connections
            await self.restructure_workflow(
                workflow_id,
                suggestion["details"].get("new_structure", {})
            )
        elif optimization_type == OptimizationType.PARAMETER_TUNING:
            # Tune parameters
            await self.tune_parameters(
                workflow_id,
                suggestion["details"]["parameters"]
            )
    
    async def _queue_for_review(
        self,
        workflow_id: str,
        suggestion: Dict[str, Any]
    ):
        """Queue optimization for human review"""
        self.optimization_queue.append({
            "workflow_id": workflow_id,
            "suggestion": suggestion,
            "queued_at": datetime.utcnow(),
            "status": "pending_review"
        })
    
    async def update_agent_prompt(
        self,
        workflow_id: str,
        agent_names: List[str],
        new_prompt: str
    ):
        """Update agent prompts (to be implemented with actual workflow management)"""
        # This would integrate with the workflow management system
        print(f"Updating prompts for agents {agent_names} in workflow {workflow_id}")
    
    async def add_tool_to_agent(
        self,
        workflow_id: str,
        agent_name: str,
        tool_name: str
    ):
        """Add tool to agent (to be implemented with actual workflow management)"""
        # This would integrate with the workflow management system
        print(f"Adding tool {tool_name} to agent {agent_name} in workflow {workflow_id}")
    
    async def restructure_workflow(
        self,
        workflow_id: str,
        new_structure: Dict[str, Any]
    ):
        """Restructure workflow (to be implemented with actual workflow management)"""
        # This would integrate with the workflow management system
        print(f"Restructuring workflow {workflow_id}")
    
    async def tune_parameters(
        self,
        workflow_id: str,
        parameters: List[str]
    ):
        """Tune workflow parameters (to be implemented with actual workflow management)"""
        # This would integrate with the workflow management system
        print(f"Tuning parameters {parameters} for workflow {workflow_id}")
    
    def get_optimization_queue(self) -> List[Dict[str, Any]]:
        """Get pending optimizations for review"""
        return [opt for opt in self.optimization_queue if opt["status"] == "pending_review"]