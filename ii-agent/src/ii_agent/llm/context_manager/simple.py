import logging
from src.ii_agent.llm.base import GeneralContentBlock, TextPrompt, TextResult
from src.ii_agent.llm.context_manager.base import ContextManager
from src.ii_agent.llm.token_counter import TokenCounter
from src.ii_agent.utils.constants import TOKEN_BUDGET


class SimpleContextManager(ContextManager):
    """A simple context manager that truncates old messages without summarization.
    
    This is more suitable for local LLMs like Ollama that may struggle with
    large summarization tasks.
    """

    def __init__(
        self,
        token_counter: TokenCounter,
        logger: logging.Logger,
        token_budget: int = TOKEN_BUDGET,
        keep_recent: int = 10,
    ):
        """
        Initialize the simple context manager.
        
        Args:
            token_counter: Token counter instance
            logger: Logger instance
            token_budget: Maximum tokens allowed
            keep_recent: Number of recent message lists to always keep
        """
        super().__init__(token_counter, logger, token_budget)
        self.keep_recent = keep_recent

    def apply_truncation(
        self, message_lists: list[list[GeneralContentBlock]]
    ) -> list[list[GeneralContentBlock]]:
        """Apply simple truncation by keeping only recent messages."""
        
        # If we're under the limit, return as is
        if not self.should_truncate(message_lists):
            return message_lists
        
        # Always keep the first message (usually system prompt or initial context)
        first_message = message_lists[:1] if message_lists else []
        
        # Keep the most recent messages
        if len(message_lists) > self.keep_recent + 1:
            # Add a truncation notice
            truncation_notice = [[
                TextResult(text=f"[Previous {len(message_lists) - self.keep_recent - 1} messages truncated to fit context]")
            ]]
            
            recent_messages = message_lists[-self.keep_recent:]
            
            truncated_messages = first_message + truncation_notice + recent_messages
            
            self.logger.info(
                f"Truncated {len(message_lists)} messages to {len(truncated_messages)} "
                f"(kept 1 first + 1 notice + {self.keep_recent} recent)"
            )
            
            return truncated_messages
        
        # If we have fewer messages than keep_recent, just return them
        return message_lists


class OllamaOptimizedContextManager(ContextManager):
    """An optimized context manager for Ollama that uses selective truncation."""

    def __init__(
        self,
        token_counter: TokenCounter,
        logger: logging.Logger,
        token_budget: int = 8000,  # Smaller default for Ollama
        max_message_length: int = 1000,  # Truncate individual long messages
    ):
        """
        Initialize the Ollama-optimized context manager.
        
        Args:
            token_counter: Token counter instance
            logger: Logger instance
            token_budget: Maximum tokens allowed (default lower for Ollama)
            max_message_length: Maximum length for individual messages
        """
        super().__init__(token_counter, logger, token_budget)
        self.max_message_length = max_message_length

    def _truncate_message_content(self, content: str) -> str:
        """Truncate individual message content if too long."""
        if len(content) <= self.max_message_length:
            return content
        
        # Keep beginning and end
        keep_start = self.max_message_length // 2
        keep_end = self.max_message_length // 4
        
        return (
            content[:keep_start] + 
            f"\n[... {len(content) - keep_start - keep_end} characters truncated ...]\n" + 
            content[-keep_end:]
        )

    def apply_truncation(
        self, message_lists: list[list[GeneralContentBlock]]
    ) -> list[list[GeneralContentBlock]]:
        """Apply intelligent truncation suitable for Ollama."""
        
        # First, truncate individual messages that are too long
        truncated_lists = []
        
        for message_list in message_lists:
            truncated_list = []
            for message in message_list:
                if isinstance(message, (TextPrompt, TextResult)):
                    # Truncate long messages
                    truncated_text = self._truncate_message_content(message.text)
                    if isinstance(message, TextPrompt):
                        truncated_list.append(TextPrompt(text=truncated_text))
                    else:
                        truncated_list.append(TextResult(text=truncated_text))
                else:
                    truncated_list.append(message)
            truncated_lists.append(truncated_list)
        
        # If still over budget, remove middle messages
        if self.count_tokens(truncated_lists) > self._token_budget:
            # Keep first (system/context) and recent messages
            keep_first = 1
            keep_recent = 5
            
            if len(truncated_lists) > keep_first + keep_recent:
                first_messages = truncated_lists[:keep_first]
                recent_messages = truncated_lists[-(keep_recent):]
                
                # Add a summary of what was removed
                removed_count = len(truncated_lists) - keep_first - keep_recent
                truncation_notice = [[
                    TextResult(text=f"[{removed_count} previous exchanges removed to fit context window]")
                ]]
                
                truncated_lists = first_messages + truncation_notice + recent_messages
                
                self.logger.info(
                    f"Removed {removed_count} message lists to fit token budget"
                )
        
        return truncated_lists