"""
Memory components for II-Agent framework
Based on II-Agent repository patterns
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime
import json

class ConversationMemory(BaseModel):
    """Manages conversation history and context."""
    
    messages: List[Dict[str, Any]] = []
    context: Dict[str, Any] = {}
    session_id: str
    max_messages: int = 50
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add a message to conversation history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.messages.append(message)
        
        # Keep only recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_recent_messages(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages."""
        return self.messages[-count:]
    
    def update_context(self, new_context: Dict[str, Any]):
        """Update conversation context."""
        self.context.update(new_context)
    
    def get_context(self) -> Dict[str, Any]:
        """Get current conversation context."""
        return self.context

class AgentMemory(BaseModel):
    """Manages agent memory including working and long-term memory."""
    
    working_memory: Dict[str, Any] = {}
    long_term_memory: Dict[str, Any] = {}
    episodic_memory: List[Dict[str, Any]] = []
    session_id: str
    
    def store_working(self, key: str, value: Any):
        """Store data in working memory."""
        self.working_memory[key] = value
    
    def retrieve_working(self, key: str, default: Any = None) -> Any:
        """Retrieve data from working memory."""
        return self.working_memory.get(key, default)
    
    def store_long_term(self, key: str, value: Any):
        """Store data in long-term memory."""
        self.long_term_memory[key] = value
    
    def retrieve_long_term(self, key: str, default: Any = None) -> Any:
        """Retrieve data from long-term memory."""
        return self.long_term_memory.get(key, default)
    
    def add_episode(self, episode: Dict[str, Any]):
        """Add an episode to episodic memory."""
        episode_with_timestamp = {
            **episode,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id
        }
        self.episodic_memory.append(episode_with_timestamp)
    
    def clear_working_memory(self):
        """Clear working memory."""
        self.working_memory.clear()
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of all memory contents."""
        return {
            "working_memory_keys": list(self.working_memory.keys()),
            "long_term_memory_keys": list(self.long_term_memory.keys()),
            "episodic_memory_count": len(self.episodic_memory),
            "session_id": self.session_id
        }