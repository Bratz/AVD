"""
Event stream management for II-Agent
Bridges existing event types with agent execution
"""
import asyncio
from typing import List, Optional, Callable, Dict, Any, TYPE_CHECKING
from collections import deque
from datetime import datetime

from .event import EventType, RealtimeEvent

# Use TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    from .agent import ThoughtStep

class EventStream:
    """Manages the event stream for agent execution following II-Agent patterns"""
    
    def __init__(self, max_history: int = 1000):
        self.queue = asyncio.Queue()
        self.history = deque(maxlen=max_history)
        self.subscribers: List[Callable] = []
        
        # Map ThoughtStep types to EventTypes
        self.thought_to_event_mapping = {
            "observation": EventType.AGENT_THINKING,
            "thought": EventType.AGENT_THINKING,
            "action": EventType.TOOL_CALL,
            "reflection": EventType.AGENT_RESPONSE,
            "planning": EventType.PROCESSING
        }
        
    async def emit_thought_as_event(self, thought: 'ThoughtStep'):
        """Convert ThoughtStep to Event and emit"""
        event_type = self.thought_to_event_mapping.get(
            thought.type, 
            EventType.AGENT_THINKING
        )
        
        event = RealtimeEvent(
            type=event_type,
            content={
                "thought_id": thought.id,
                "thought_type": thought.type,
                "content": thought.content,
                "timestamp": thought.timestamp.isoformat(),
                "metadata": thought.metadata
            }
        )
        
        await self.emit(event)
        
    async def emit(self, event: RealtimeEvent):
        """Emit an event to the stream"""
        self.history.append(event)
        await self.queue.put(event)
        
        # Notify subscribers
        for subscriber in self.subscribers:
            try:
                await subscriber(event)
            except Exception as e:
                # Log but don't fail
                pass
                
    async def emit_event(self, event_type: EventType, content: Dict[str, Any]):
        """Helper to emit events directly"""
        event = RealtimeEvent(type=event_type, content=content)
        await self.emit(event)
        
    async def get_next(self) -> RealtimeEvent:
        """Get the next event from the stream"""
        return await self.queue.get()
        
    def get_history(self, event_type: Optional[EventType] = None) -> List[RealtimeEvent]:
        """Get event history, optionally filtered by type"""
        if event_type:
            return [e for e in self.history if e.type == event_type]
        return list(self.history)
        
    def subscribe(self, callback: Callable):
        """Subscribe to events"""
        self.subscribers.append(callback)
        
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from events"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)