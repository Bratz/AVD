"""
Base LLM Client interface for ii-agent
All LLM providers should inherit from this base class
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, AsyncGenerator
import logging


class BaseLLMClient(ABC):
    """Base class for all LLM clients in ii-agent"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Generate a response from the LLM
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            stream: Whether to stream the response
            **kwargs: Additional provider-specific arguments
            
        Returns:
            The generated response as a string
        """
        pass
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from the LLM
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            **kwargs: Additional provider-specific arguments
            
        Yields:
            Response chunks as they are generated
        """
        # Default implementation - providers can override
        response = await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            **kwargs
        )
        yield response
    
    async def close(self):
        """Clean up any resources"""
        pass
    
    def __str__(self):
        return f"{self.__class__.__name__}"
    
    def __repr__(self):
        return f"<{self.__class__.__name__}>"