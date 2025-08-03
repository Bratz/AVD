from typing import Dict, Any, List

class Capabilities:
    """MCP capabilities for client and server."""

    def __init__(self, tools: List[Dict[str, Any]] = None):
        self.tools = tools or []

    def to_dict(self) -> Dict[str, Any]:
        return {"tools": self.tools}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Capabilities':
        return cls(tools=data.get("tools", []))