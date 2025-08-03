"""
Banking Specialist Agent using II-Agent framework
"""
from typing import Any, Dict, List
from src.ii_agent.core.agent import IIAgent
from src.ii_agent.tools.banking_tool_registry import BankingToolRegistry

class BankingSpecialistAgent(IIAgent):
    """Specialized banking agent using II-Agent framework."""
    
    def __init__(self, tool_registry: BankingToolRegistry):
        super().__init__(
            name="Banking Specialist",
            description="Specialized agent for banking operations using II-Agent framework"
        )
        self.tool_registry = tool_registry
        
    async def plan(self, goal: str, context: Dict[str, Any]) -> List[str]:
        """Plan banking-specific tasks."""
        # Use II-Agent planning with banking context
        banking_context = {
            **context,
            "available_tools": list(self.tool_registry.tools.keys()),
            "domain": "banking",
            "compliance_required": True
        }
        
        return await super().plan(goal, banking_context)
        
    async def execute_step(self, step: str, parameters: Dict[str, Any]) -> Any:
        """Execute banking-specific steps."""
        # Map step to appropriate banking tool
        if step in self.tool_registry.tools:
            tool = self.tool_registry.tools[step]
            return await tool.execute(parameters)
        
        return await super().execute_step(step, parameters)