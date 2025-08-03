# src/ii_agent/workflows/studio_integration.py
import asyncio
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph
import os

class LangGraphStudioIntegration:
    """Integration with LangGraph Studio for visualization and debugging"""
    
    def __init__(self):
        self.studio_api_key = os.getenv("LANGGRAPH_STUDIO_API_KEY")
        self.studio_url = os.getenv("LANGGRAPH_STUDIO_URL", "http://localhost:8123")
    
    def export_for_studio(self, workflow: StateGraph) -> Dict[str, Any]:
        """Export workflow in LangGraph Studio format"""
        
        return {
            "nodes": self._extract_nodes(workflow),
            "edges": self._extract_edges(workflow),
            "metadata": {
                "name": workflow.name,
                "description": workflow.description,
                "version": "1.0.0"
            }
        }
    
    def enable_studio_debugging(self, workflow: StateGraph):
        """Enable LangGraph Studio debugging for workflow"""
        
        if not self.studio_api_key:
            raise ValueError("LANGGRAPH_STUDIO_API_KEY not set")
        
        # Add studio tracking
        workflow._graph.graph["studio_enabled"] = True
        workflow._graph.graph["studio_api_key"] = self.studio_api_key
        
        # Add debug callbacks
        for node in workflow.nodes:
            self._add_studio_callbacks(node)
    
    def _add_studio_callbacks(self, node):
        """Add LangGraph Studio callbacks to node"""
        
        original_func = node.func
        
        def wrapped_func(state):
            # Send pre-execution event to Studio
            self._send_studio_event({
                "type": "node_start",
                "node": node.name,
                "state": state,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            # Execute original function
            result = original_func(state)
            
            # Send post-execution event to Studio
            self._send_studio_event({
                "type": "node_end",
                "node": node.name,
                "result": result,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            return result
        
        node.func = wrapped_func