"""
Integration utilities for ROWBOAT with ii-agent infrastructure
"""

from typing import Dict, Any, Optional
import logging
from src.ii_agent.agents.bancs.multi_agent_coordinator import ROWBOATCoordinator
from src.ii_agent.agents.registry import AgentRegistry

logger = logging.getLogger(__name__)

class ROWBOATIntegration:
    """Integration helper for ROWBOAT with ii-agent"""
    
    @staticmethod
    async def register_rowboat_coordinator(
        agent_registry: AgentRegistry,
        workspace_manager,
        message_queue,
        context_manager,
        config: Optional[Dict[str, Any]] = None,
        client=None,
        tools=None
    ) -> ROWBOATCoordinator:
        """Register ROWBOAT coordinator with ii-agent registry"""
        
        # Create coordinator using async factory
        coordinator = await ROWBOATCoordinator.create(
            client=client,
            tools=tools or [],
            workspace_manager=workspace_manager,
            message_queue=message_queue,
            logger_for_agent_logs=logging.getLogger(__name__),
            context_manager=context_manager,
            config=config
        )
        
        # Register as a specialized agent
        agent_registry.register_agent(
            agent_type="rowboat_coordinator",
            agent_instance=coordinator
        )
        
        logger.info("ROWBOAT coordinator registered with ii-agent")
        
        return coordinator
    
    @staticmethod
    def create_rowboat_enabled_agent(
        base_agent_class,
        **kwargs
    ):
        """Create an agent with ROWBOAT multi-agent capabilities"""
        
        class ROWBOATEnabledAgent(base_agent_class):
            """Agent enhanced with ROWBOAT capabilities"""
            
            def __init__(self, **agent_kwargs):
                super().__init__(**agent_kwargs)
                self.rowboat = None  # Will be initialized asynchronously
            
            async def initialize_rowboat(self):
                """Initialize ROWBOAT coordinator asynchronously"""
                if not self.rowboat:
                    self.rowboat = await ROWBOATCoordinator.create(
                        client=self.client,
                        tools=self.tools,
                        workspace_manager=self.workspace_manager,
                        message_queue=self.message_queue,
                        logger_for_agent_logs=self.logger_for_agent_logs,
                        context_manager=self.context_manager
                    )
            
            async def execute_with_rowboat(
                self,
                input_data: Dict[str, Any]
            ) -> Dict[str, Any]:
                """Execute using ROWBOAT multi-agent workflow"""
                
                # Ensure ROWBOAT is initialized
                if not self.rowboat:
                    await self.initialize_rowboat()
                
                # Create workflow from agent's knowledge
                workflow_id = await self.rowboat.create_workflow_from_description(
                    description=input_data.get("workflow_description", ""),
                    user_context={"agent": self.name}
                )
                
                # Execute workflow
                return await self.rowboat.execute_workflow(
                    workflow_id,
                    input_data
                )
        
        return ROWBOATEnabledAgent(**kwargs)