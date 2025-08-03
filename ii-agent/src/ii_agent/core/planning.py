"""
Planning engine for II-Agent framework  
Based on II-Agent repository patterns
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from enum import Enum
import uuid

class PlanStepStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class PlanStep(BaseModel):
    """Individual step in a plan."""
    
    id: str = ""
    description: str
    action: str
    parameters: Dict[str, Any] = {}
    dependencies: List[str] = []
    status: PlanStepStatus = PlanStepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.id:
            self.id = str(uuid.uuid4())

class Plan(BaseModel):
    """A complete plan with multiple steps."""
    
    id: str = ""
    goal: str
    steps: List[PlanStep] = []
    status: str = "created"
    created_by: str = ""
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def add_step(self, description: str, action: str, parameters: Dict[str, Any] = None) -> PlanStep:
        """Add a step to the plan."""
        step = PlanStep(
            description=description,
            action=action,
            parameters=parameters or {}
        )
        self.steps.append(step)
        return step
    
    def get_next_step(self) -> Optional[PlanStep]:
        """Get the next pending step."""
        for step in self.steps:
            if step.status == PlanStepStatus.PENDING:
                # Check if dependencies are satisfied
                dependencies_satisfied = all(
                    any(s.id == dep_id and s.status == PlanStepStatus.COMPLETED 
                        for s in self.steps) 
                    for dep_id in step.dependencies
                ) if step.dependencies else True
                
                if dependencies_satisfied:
                    return step
        return None
    
    def mark_step_completed(self, step_id: str, result: Any = None):
        """Mark a step as completed."""
        for step in self.steps:
            if step.id == step_id:
                step.status = PlanStepStatus.COMPLETED
                step.result = result
                break
    
    def mark_step_failed(self, step_id: str, error: str):
        """Mark a step as failed."""
        for step in self.steps:
            if step.id == step_id:
                step.status = PlanStepStatus.FAILED
                step.error = error
                break
    
    def is_complete(self) -> bool:
        """Check if all steps are completed."""
        return all(step.status == PlanStepStatus.COMPLETED for step in self.steps)
    
    def has_failed_steps(self) -> bool:
        """Check if any steps have failed."""
        return any(step.status == PlanStepStatus.FAILED for step in self.steps)

class PlanningEngine:
    """Engine for creating and managing plans."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.active_plans: Dict[str, Plan] = {}
    
    def create_plan(self, goal: str, steps: List[Dict[str, Any]] = None) -> Plan:
        """Create a new plan."""
        plan = Plan(goal=goal, created_by=self.agent_id)
        
        if steps:
            for step_data in steps:
                plan.add_step(
                    description=step_data.get("description", ""),
                    action=step_data.get("action", ""),
                    parameters=step_data.get("parameters", {})
                )
        
        self.active_plans[plan.id] = plan
        return plan
    
    def get_plan(self, plan_id: str) -> Optional[Plan]:
        """Get a plan by ID."""
        return self.active_plans.get(plan_id)
    
    def execute_next_step(self, plan_id: str) -> Optional[PlanStep]:
        """Get the next step to execute for a plan."""
        plan = self.get_plan(plan_id)
        if plan:
            return plan.get_next_step()
        return None
    
    def update_step_status(self, plan_id: str, step_id: str, status: PlanStepStatus, result: Any = None, error: str = None):
        """Update the status of a plan step."""
        plan = self.get_plan(plan_id)
        if plan:
            for step in plan.steps:
                if step.id == step_id:
                    step.status = status
                    if result is not None:
                        step.result = result
                    if error:
                        step.error = error
                    break