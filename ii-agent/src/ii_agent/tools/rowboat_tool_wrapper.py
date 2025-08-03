# ============================================================
# FILE 2: src/ii_agent/tools/rowboat_tool_wrapper.py
# ============================================================
"""
Wrapper to integrate ii-agent tools with ROWBOAT
"""

from typing import Dict, Any
from src.ii_agent.tools.base import BaseTool as IIAgentBaseTool
from src.ii_agent.tools.rowboat_tools import BaseTool as ROWBOATBaseTool

class IIAgentToolWrapper(ROWBOATBaseTool):
    """Wrapper to make ii-agent tools compatible with ROWBOAT"""
    
    def __init__(self, ii_agent_tool: IIAgentBaseTool):
        self.wrapped_tool = ii_agent_tool
        super().__init__(
            name=ii_agent_tool.name,
            description=ii_agent_tool.description
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute wrapped tool"""
        
        # ii-agent tools may be sync, so handle both cases
        if hasattr(self.wrapped_tool, 'arun'):
            # Async tool
            return await self.wrapped_tool.arun(**kwargs)
        else:
            # Sync tool - run in executor
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.wrapped_tool.run,
                kwargs
            )
