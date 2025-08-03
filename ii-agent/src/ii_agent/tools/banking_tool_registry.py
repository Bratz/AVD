"""
Global Banking Tool Registry
Provides a singleton registry for all MCP tools used across the banking system
"""

import logging
import jsonschema
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import threading


@dataclass
class ToolMetadata:
    """Enhanced tool metadata with full OpenAPI-style information"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    category: str = "general"
    permissions: List[str] = field(default_factory=list)
    last_used: Optional[datetime] = None
    usage_count: int = 0
    # Enhanced fields for OpenAPI compatibility
    method: str = "POST"
    path: str = ""
    tags: List[str] = field(default_factory=list)
    deprecated: bool = False
    auth_required: bool = True
    operation_id: Optional[str] = None
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    examples: Dict[str, Any] = field(default_factory=dict)
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate input parameters against the tool's input schema
        
        Args:
            parameters: Dictionary of input parameters
        
        Returns:
            bool: Whether parameters are valid
        
        Raises:
            jsonschema.ValidationError if parameters are invalid
        """
        try:
            jsonschema.validate(instance=parameters, schema=self.input_schema)
            return True
        except jsonschema.ValidationError as e:
            logging.error(f"Parameter validation failed for tool {self.name}: {e}")
            raise


class MCPToolRegistry:
    """
    Enhanced singleton registry for MCP tools with comprehensive categorization
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._tools: Dict[str, ToolMetadata] = {}
        self._logger = logging.getLogger("MCPToolRegistry")
        
        # Enhanced indexes for better tool discovery
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)
        self._category_index: Dict[str, Set[str]] = defaultdict(set)
        self._method_index: Dict[str, Set[str]] = defaultdict(set)
        self._operation_id_index: Dict[str, str] = {}
        self._deprecated_tools: Set[str] = set()
        self._permission_index: Dict[str, Set[str]] = defaultdict(set)
        
        self._initialized = True
        self._logger.info("Global MCP Tool Registry initialized")

    def register_tools(self, tools: List[Dict[str, Any]]):
        """
        Register multiple tools from MCP server discovery
        
        Args:
            tools: List of tool configurations from MCP server
        """
        registered_count = 0
        for tool_config in tools:
            try:
                # Handle both input_schema and inputSchema
                schema = tool_config.get('input_schema') or tool_config.get('inputSchema', {})
                
                # Ensure schema has basic structure
                if not isinstance(schema, dict):
                    schema = {"type": "object", "properties": {}, "required": []}
                
                tool_metadata = ToolMetadata(
                    name=tool_config['name'],
                    description=tool_config.get('description', ''),
                    input_schema=schema,
                    category=tool_config.get('category', 'general'),
                    permissions=tool_config.get('permissions', []),
                    method=tool_config.get('method', 'POST'),
                    path=tool_config.get('path', ''),
                    tags=tool_config.get('tags', []),
                    deprecated=tool_config.get('deprecated', False),
                    auth_required=tool_config.get('auth_required', True),
                    operation_id=tool_config.get('operation_id'),
                    parameters=tool_config.get('parameters', []),
                    examples=tool_config.get('examples', {})
                )
                self.register_tool(tool_metadata)
                registered_count += 1
            except Exception as e:
                self._logger.error(f"Failed to register tool {tool_config.get('name')}: {e}")
                
        self._logger.info(f"Registered {registered_count}/{len(tools)} tools")

    def register_tool(self, tool_metadata: ToolMetadata):
        """Register a single tool with validation and indexing"""
        try:
            self._logger.info(f"Attempting to register tool: {tool_metadata.name}")
            self._logger.info(f"Tool details: category={tool_metadata.category}, tags={tool_metadata.tags}")
            # Validate input schema
            jsonschema.validators.Draft7Validator.check_schema(tool_metadata.input_schema)
            
            # Check for name conflicts
            if tool_metadata.name in self._tools:
                self._logger.info(f"Overwriting existing tool: {tool_metadata.name}")
            
            # Register the tool
            self._tools[tool_metadata.name] = tool_metadata
            
            # Update indexes
            self._update_indexes(tool_metadata)
            
            self._logger.info(f"Registered tool in register_tool: {tool_metadata.name} [{tool_metadata.category}]")
            
        except jsonschema.exceptions.SchemaError as e:
            self._logger.error(f"Invalid tool schema for {tool_metadata.name}: {e}")
            raise ValueError(f"Invalid tool schema: {e}")
    
    def _update_indexes(self, tool_metadata: ToolMetadata):
        """Update all registry indexes"""
        name = tool_metadata.name
        
        # Category index
        self._category_index[tool_metadata.category].add(name)
        
        # Tag index
        for tag in tool_metadata.tags:
            self._tag_index[tag.lower()].add(name)
        
        # Method index
        self._method_index[tool_metadata.method].add(name)
        
        # Operation ID index
        if tool_metadata.operation_id:
            self._operation_id_index[tool_metadata.operation_id] = name
        
        # Permission index
        for permission in tool_metadata.permissions:
            self._permission_index[permission].add(name)
        
        # Deprecated tools
        if tool_metadata.deprecated:
            self._deprecated_tools.add(name)

    def get_tool(self, name: str) -> Optional[ToolMetadata]:
        """Retrieve tool metadata by name"""
        return self._tools.get(name)
    
    def get_tool_by_operation_id(self, operation_id: str) -> Optional[ToolMetadata]:
        """Get tool by operation ID"""
        tool_name = self._operation_id_index.get(operation_id)
        return self._tools.get(tool_name) if tool_name else None

    def list_tools(self, 
                   category: Optional[str] = None,
                   tag: Optional[str] = None,
                   method: Optional[str] = None,
                   permission: Optional[str] = None,
                   include_deprecated: bool = False) -> List[ToolMetadata]:
        """
        List registered tools with multiple filter options
        
        Args:
            category: Filter by category
            tag: Filter by tag
            method: Filter by HTTP method
            permission: Filter by required permission
            include_deprecated: Include deprecated tools
        
        Returns:
            List of ToolMetadata matching filters
        """
        # Start with filtered set based on most specific filter
        if tag:
            tool_names = self._tag_index.get(tag.lower(), set())
        elif category:
            tool_names = self._category_index.get(category, set())
        elif method:
            tool_names = self._method_index.get(method.upper(), set())
        elif permission:
            tool_names = self._permission_index.get(permission, set())
        else:
            tool_names = set(self._tools.keys())
        
        # Apply additional filters
        results = []
        for name in tool_names:
            tool = self._tools[name]
            
            if category and tool.category != category:
                continue
            if method and tool.method != method.upper():
                continue
            if permission and permission not in tool.permissions:
                continue
            if not include_deprecated and tool.deprecated:
                continue
                
            results.append(tool)
        
        return sorted(results, key=lambda t: (t.deprecated, t.category, t.name))
    
    def search_tools(self, 
                     query: str,
                     limit: Optional[int] = None) -> List[ToolMetadata]:
        """Search tools by text query across multiple fields"""
        query_lower = query.lower()
        matches = []
        
        for tool in self._tools.values():
            # Search in multiple fields
            searchable_text = [
                tool.name.lower(),
                tool.description.lower(),
                tool.path.lower(),
                ' '.join(tool.tags).lower(),
                tool.category.lower(),
                tool.operation_id.lower() if tool.operation_id else '',
                ' '.join(tool.permissions).lower()
            ]
            
            if any(query_lower in text for text in searchable_text):
                matches.append(tool)
        
        # Sort by relevance
        matches.sort(key=lambda t: (
            query_lower not in t.name.lower(),
            t.deprecated,
            t.name
        ))
        
        if limit:
            matches = matches[:limit]
            
        return matches
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics"""
        return {
            "total_tools": len(self._tools),
            "categories": dict(sorted(
                [(cat, len(tools)) for cat, tools in self._category_index.items()]
            )),
            "tags": dict(sorted(
                [(tag, len(tools)) for tag, tools in self._tag_index.items()]
            )),
            "methods": dict(sorted(
                [(method, len(tools)) for method, tools in self._method_index.items()]
            )),
            "deprecated_count": len(self._deprecated_tools),
            "permissions": dict(sorted(
                [(perm, len(tools)) for perm, tools in self._permission_index.items()]
            )),
            "most_used": sorted(
                [(t.name, t.usage_count) for t in self._tools.values() if t.usage_count > 0],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
    
    def clear(self):
        """Clear all registered tools (use with caution)"""
        self._tools.clear()
        self._tag_index.clear()
        self._category_index.clear()
        self._method_index.clear()
        self._operation_id_index.clear()
        self._deprecated_tools.clear()
        self._permission_index.clear()
        self._logger.info("Registry cleared")
    
    def export_catalog(self) -> Dict[str, Any]:
        """Export tool catalog in OpenAPI-style format"""
        catalog = {}
        for name, tool in self._tools.items():
            catalog[name] = {
                "method": tool.method,
                "path": tool.path,
                "description": tool.description,
                "tags": tool.tags,
                "deprecated": tool.deprecated,
                "auth_required": tool.auth_required,
                "operation_id": tool.operation_id,
                "parameters": tool.parameters,
                "input_schema": tool.input_schema,
                "examples": tool.examples
            }
        return catalog


# Global singleton instance
mcp_tool_registry = MCPToolRegistry()