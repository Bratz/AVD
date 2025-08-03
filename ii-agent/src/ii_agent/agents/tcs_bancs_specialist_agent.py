"""
TCS BaNCS Specialist Agent using II-Agent Framework
Enhanced with MCP prompt support
"""
from typing import Optional, List, Any, Dict
import logging
import asyncio

from src.ii_agent.agents.anthropic_fc import AnthropicFC
from src.ii_agent.prompts.system_prompt import get_banking_prompt
from src.ii_agent.utils.logging_config import get_logger


class TCSBancsSpecialistAgent(AnthropicFC):
    """Banking specialist extending AnthropicFC with MCP prompt support."""
    
    @classmethod
    async def create(
        cls,
        client,
        tools,
        workspace_manager,
        message_queue,
        logger_for_agent_logs,
        context_manager,
        user_role: str = "customer",
        mcp_wrapper=None,
        use_mcp_prompts: bool = True,
        **kwargs
    ):
        """
        Async factory method to create agent with MCP prompts.
        
        Args:
            client: LLM client
            tools: Available tools
            workspace_manager: Workspace manager
            message_queue: Message queue
            logger_for_agent_logs: Logger for agent logs
            context_manager: Context manager
            user_role: User role (customer, admin, etc.)
            mcp_wrapper: MCP client wrapper for prompts
            use_mcp_prompts: Whether to use MCP prompts (default: True)
            **kwargs: Additional arguments for parent class
        
        Returns:
            TCSBancsSpecialistAgent: Initialized agent instance
        """
        logger = get_logger("TCSBancsAgent")
        banking_prompt = None
        mcp_prompts_metadata = {}
        
        # Try to load MCP prompts if wrapper is available
        if mcp_wrapper and use_mcp_prompts:
            try:
                # Discover available prompts
                logger.info("Discovering MCP prompts...")
                available_prompts = await mcp_wrapper.list_prompts()
                logger.info(f"Found {len(available_prompts)} MCP prompts: {[p['name'] for p in available_prompts]}")
                
                # Store prompt metadata for later use
                mcp_prompts_metadata = {
                    prompt['name']: prompt 
                    for prompt in available_prompts
                }
                
                # Get the banking setup prompt with role-specific arguments
                prompt_name = "banking_specialist"
                if prompt_name in mcp_prompts_metadata:
                    logger.info(f"Loading MCP prompt: {prompt_name} for role: {user_role}")
                    prompt_data = await mcp_wrapper.get_prompt(
                        prompt_name,
                        {"role": user_role}  # Pass role as argument
                    )
                    
                    # Extract prompt content from MCP response
                    if 'messages' in prompt_data and prompt_data['messages']:
                        banking_prompt = prompt_data['messages'][0]['content']
                        logger.info("Successfully loaded MCP prompt")
                    else:
                        logger.warning("MCP prompt response missing messages")
                        
            except Exception as e:
                logger.error(f"Failed to load MCP prompts: {e}")
                logger.info("Falling back to local prompts")
        
        # Fallback to local prompt if MCP fails or is disabled
        if not banking_prompt:
            llm_type = "ollama" if "ollama" in str(type(client)).lower() else "cloud"
            banking_prompt = get_banking_prompt(user_role, llm_type)
            logger.info(f"Using local prompt for role: {user_role}, llm_type: {llm_type}")
        
        # Create instance with the resolved prompt
        instance = cls(
            client=client,
            tools=tools,
            workspace_manager=workspace_manager,
            message_queue=message_queue,
            logger_for_agent_logs=logger_for_agent_logs,
            context_manager=context_manager,
            user_role=user_role,
            mcp_wrapper=mcp_wrapper,
            banking_prompt=banking_prompt,
            mcp_prompts_metadata=mcp_prompts_metadata,
            **kwargs
        )
        
        return instance
    
    def __init__(
        self,
        client,
        tools,
        workspace_manager,
        message_queue,
        logger_for_agent_logs,
        context_manager,
        user_role: str = "customer",
        mcp_wrapper=None,
        banking_prompt: str = None,
        mcp_prompts_metadata: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Initialize with banking-specific system prompt.
        
        Note: Use the create() class method for async initialization with MCP prompts.
        """
        # Use provided prompt or generate default
        if not banking_prompt:
            llm_type = "ollama" if "ollama" in str(type(client)).lower() else "cloud"
            banking_prompt = get_banking_prompt(user_role, llm_type)
        
        # Initialize parent with banking prompt
        super().__init__(
            system_prompt=banking_prompt,
            client=client,
            tools=tools,
            workspace_manager=workspace_manager,
            message_queue=message_queue,
            logger_for_agent_logs=logger_for_agent_logs,
            context_manager=context_manager,
            **kwargs
        )
        
        # Store banking-specific attributes
        self.mcp_wrapper = mcp_wrapper
        self.user_role = user_role
        self.banking_logger = get_logger("TCSBancsAgent")
        self.mcp_prompts_metadata = mcp_prompts_metadata or {}
        self._mcp_enabled = bool(mcp_wrapper and mcp_prompts_metadata)
        
        # Log initialization details
        self.banking_logger.info(
            f"TCSBancsAgent initialized - "
            f"Role: {user_role}, "
            f"MCP Enabled: {self._mcp_enabled}, "
            f"Available MCP Prompts: {list(self.mcp_prompts_metadata.keys())}"
        )
    
    async def get_contextual_prompt(self, context: str, **kwargs) -> Optional[str]:
        """
        Get appropriate MCP prompt based on context.
        
        Args:
            context: Context string to determine which prompt to use
            **kwargs: Additional arguments for the prompt
            
        Returns:
            Optional[str]: Prompt content or None if not available
        """
        if not self._mcp_enabled:
            self.banking_logger.debug("MCP prompts not enabled")
            return None
        
        # Determine which prompt to use based on context
        prompt_name = None
        
        if "error" in context.lower() or "failed" in context.lower():
            prompt_name = "banking_error_handling"
        elif "discover" in context.lower() or "find" in context.lower():
            prompt_name = "discover_banking_apis_by_domain"
        elif "example" in context.lower() or "request" in context.lower():
            prompt_name = "generate_banking_request"
        elif "explain" in context.lower():
            prompt_name = "explain_banking_endpoint"
        else:
            prompt_name = "banking_api_quick_start"
        
        # Check if prompt is available
        if prompt_name not in self.mcp_prompts_metadata:
            self.banking_logger.warning(f"MCP prompt '{prompt_name}' not available")
            return None
        
        try:
            # Get the prompt with arguments
            self.banking_logger.info(f"Fetching MCP prompt: {prompt_name}")
            prompt_data = await self.mcp_wrapper.get_prompt(prompt_name, kwargs)
            
            if 'messages' in prompt_data and prompt_data['messages']:
                return prompt_data['messages'][0]['content']
            else:
                self.banking_logger.warning(f"Invalid prompt data for '{prompt_name}'")
                return None
                
        except Exception as e:
            self.banking_logger.error(f"Failed to get MCP prompt '{prompt_name}': {e}")
            return None
    
    def run_agent(
        self,
        instruction: str,
        files: list[str] | None = None,
        resume: bool = False,
        orientation_instruction: str | None = None,
    ) -> str:
        """
        Override to add banking context logging and dynamic prompt loading.
        """
        self.banking_logger.info(f"Banking request: {instruction[:100]}...")
        
        # Optionally load contextual prompts for specific requests
        if self._mcp_enabled and "help" in instruction.lower():
            # This is synchronous, so we'd need to handle async differently
            # For now, just log that contextual prompts are available
            self.banking_logger.info(
                f"Contextual MCP prompts available: {list(self.mcp_prompts_metadata.keys())}"
            )
        
        # Call parent implementation
        result = super().run_agent(instruction, files, resume, orientation_instruction)
        
        self.banking_logger.info("Banking request completed")
        return result
    
    async def get_available_prompts(self) -> List[str]:
        """Get list of available MCP prompt names."""
        return list(self.mcp_prompts_metadata.keys())
    
    def is_mcp_enabled(self) -> bool:
        """Check if MCP prompts are enabled."""
        return self._mcp_enabled


# Convenience function for backward compatibility
def create_banking_agent(
    client,
    tools,
    workspace_manager,
    message_queue,
    logger_for_agent_logs,
    context_manager,
    user_role: str = "customer",
    mcp_wrapper=None,
    **kwargs
) -> TCSBancsSpecialistAgent:
    """
    Create a banking agent synchronously (without MCP prompts).
    
    For MCP prompt support, use TCSBancsSpecialistAgent.create() instead.
    """
    return TCSBancsSpecialistAgent(
        client=client,
        tools=tools,
        workspace_manager=workspace_manager,
        message_queue=message_queue,
        logger_for_agent_logs=logger_for_agent_logs,
        context_manager=context_manager,
        user_role=user_role,
        mcp_wrapper=mcp_wrapper,
        **kwargs
    )