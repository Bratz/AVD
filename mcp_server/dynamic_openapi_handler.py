"""
Simplified Dynamic API Handler for FastMCP
Generates proper function signatures without **kwargs
"""

import os
import json
import yaml
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from collections import defaultdict
import re
from fastmcp import FastMCP
import httpx
from functools import wraps

logger = logging.getLogger(__name__)

class SimplifiedDynamicAPIHandler:
    """
    Simplified handler that generates proper FastMCP tools from OpenAPI/Postman specs.
    Avoids **kwargs by creating functions with explicit parameters.
    """
    
    def __init__(self, spec_path: str, base_url: Optional[str] = None):
        self.spec_path = spec_path
        self.base_url = base_url
        self.registered_tools = {}
        
        # Load spec
        with open(spec_path, 'r') as f:
            self.spec = json.load(f) if spec_path.endswith('.json') else yaml.safe_load(f)
        
        # Detect if it's Postman or OpenAPI
        self.is_postman = 'info' in self.spec and '_postman_id' in self.spec.get('info', {})
        
        if not self.base_url:
            self.base_url = self._extract_base_url()
    
    def _extract_base_url(self) -> str:
        """Extract base URL from spec."""
        if self.is_postman:
            # Try to extract from first request
            for item in self.spec.get('item', []):
                if 'request' in item:
                    url = item['request'].get('url', {})
                    if isinstance(url, dict):
                        protocol = url.get('protocol', 'https')
                        host = '.'.join(url.get('host', []))
                        return f"{protocol}://{host}"
                    elif isinstance(url, str):
                        from urllib.parse import urlparse
                        parsed = urlparse(url)
                        return f"{parsed.scheme}://{parsed.netloc}"
        else:
            # OpenAPI
            if 'servers' in self.spec:
                return self.spec['servers'][0]['url']
        
        return "https://api.example.com"
    
    def register_all_tools(self, mcp: FastMCP, max_tools: int = 100):
        """Register all tools from the spec (up to max_tools)."""
        if self.is_postman:
            self._register_postman_tools(mcp, max_tools)
        else:
            self._register_openapi_tools(mcp, max_tools)
    
    def register_tools_by_category(self, mcp: FastMCP, categories: List[str], max_per_category: int = 20):
        """Register tools filtered by category."""
        if self.is_postman:
            self._register_postman_tools_by_category(mcp, categories, max_per_category)
        else:
            self._register_openapi_tools_by_category(mcp, categories, max_per_category)
    
    def _register_postman_tools(self, mcp: FastMCP, max_tools: int):
        """Register tools from Postman collection."""
        count = 0
        
        for item in self.spec.get('item', []):
            if count >= max_tools:
                break
                
            if 'request' in item:
                self._create_tool_from_postman_request(mcp, item)
                count += 1
    
    def _register_postman_tools_by_category(self, mcp: FastMCP, categories: List[str], max_per_category: int):
        """Register Postman tools filtered by category."""
        category_counts = defaultdict(int)
        
        for item in self.spec.get('item', []):
            if 'request' in item:
                # Determine category from path
                request = item['request']
                url = request.get('url', {})
                path = self._get_path_from_postman_url(url)
                
                category = self._categorize_path(path)
                
                if category in categories and category_counts[category] < max_per_category:
                    self._create_tool_from_postman_request(mcp, item)
                    category_counts[category] += 1
    
    def _create_tool_from_postman_request(self, mcp: FastMCP, item: Dict[str, Any]):
        """Create a tool from a Postman request."""
        request = item['request']
        name = self._sanitize_name(item.get('name', 'unknown'))
        method = request.get('method', 'GET')
        
        # Parse URL
        url = request.get('url', {})
        path = self._get_path_from_postman_url(url)
        
        # Parse parameters
        headers = request.get('header', [])
        query_params = url.get('query', []) if isinstance(url, dict) else []
        
        # Parse body
        body_params = []
        body = request.get('body', {})
        if body.get('mode') == 'raw':
            try:
                body_data = json.loads(body.get('raw', '{}'))
                if isinstance(body_data, dict):
                    body_params = list(body_data.keys())
            except:
                body_params = ['request_body']
        
        # Create the tool function
        self._create_and_register_tool(
            mcp=mcp,
            name=name,
            method=method,
            path=path,
            query_params=[p.get('key') for p in query_params if isinstance(p, dict)],
            header_params=[h.get('key') for h in headers if isinstance(h, dict) and h.get('key') not in ['Content-Type', 'Accept']],
            body_params=body_params,
            description=item.get('name', '')
        )
    
    def _create_and_register_tool(
        self,
        mcp: FastMCP,
        name: str,
        method: str,
        path: str,
        query_params: List[str],
        header_params: List[str],
        body_params: List[str],
        description: str
    ):
        """Create and register a tool with explicit parameters."""
        
        # Extract path parameters
        path_params = re.findall(r'\{([^}]+)\}', path)
        
        # For Postman collections with literal IDs in paths, convert them to parameters
        if not path_params and method == 'GET':
            # Look for numeric IDs in path
            path_parts = path.split('/')
            for i, part in enumerate(path_parts):
                if part.isdigit() or (len(part) > 10 and part.isalnum()):
                    param_name = self._infer_param_name(path_parts, i)
                    path_params.append(param_name)
                    path_parts[i] = f'{{{param_name}}}'
            path = '/'.join(path_parts)
        
        # Build parameter list
        all_params = []
        param_defaults = {}
        
        # Add path parameters (always required)
        for param in path_params:
            param_name = self._sanitize_param_name(param)
            all_params.append((param_name, 'str', True))
        
        # Add query parameters (optional)
        for param in query_params:
            param_name = self._sanitize_param_name(param)
            all_params.append((param_name, 'Optional[str]', False))
            param_defaults[param_name] = None
        
        # Add header parameters (optional)
        for param in header_params:
            param_name = self._sanitize_param_name(param)
            all_params.append((param_name, 'Optional[str]', False))
            param_defaults[param_name] = None
        
        # Add body parameters
        if body_params:
            if len(body_params) == 1 and body_params[0] == 'request_body':
                all_params.append(('request_body', 'Dict[str, Any]', True))
            else:
                for param in body_params:
                    param_name = self._sanitize_param_name(param)
                    all_params.append((param_name, 'Any', False))
                    param_defaults[param_name] = None
        
        # Create function
        def create_tool_function():
            # Create a closure that captures the necessary variables
            tool_method = method
            tool_path = path
            tool_base_url = self.base_url
            
            # The actual tool function
            async def tool_impl(**kwargs):
                url = tool_base_url.rstrip('/') + tool_path
                
                # Process path parameters
                for param in path_params:
                    param_name = self._sanitize_param_name(param)
                    if param_name in kwargs:
                        url = url.replace(f'{{{param}}}', str(kwargs[param_name]))
                
                # Process query parameters
                params = {}
                for param in query_params:
                    param_name = self._sanitize_param_name(param)
                    if param_name in kwargs and kwargs[param_name] is not None:
                        params[param] = kwargs[param_name]
                
                # Process headers
                headers = {"Content-Type": "application/json"}
                headers.update(self._get_default_headers())
                
                for param in header_params:
                    param_name = self._sanitize_param_name(param)
                    if param_name in kwargs and kwargs[param_name] is not None:
                        headers[param] = kwargs[param_name]
                
                # Process body
                body = None
                if body_params:
                    if 'request_body' in kwargs:
                        body = kwargs['request_body']
                    else:
                        body = {}
                        for param in body_params:
                            param_name = self._sanitize_param_name(param)
                            if param_name in kwargs and kwargs[param_name] is not None:
                                body[param] = kwargs[param_name]
                
                # Make request
                async with httpx.AsyncClient() as client:
                    try:
                        response = await client.request(
                            method=tool_method,
                            url=url,
                            params=params,
                            json=body,
                            headers=headers,
                            timeout=30.0
                        )
                        response.raise_for_status()
                        
                        if response.headers.get('content-type', '').startswith('application/json'):
                            return response.json()
                        else:
                            return {"data": response.text}
                    except Exception as e:
                        return {"error": str(e)}
            
            # Set function metadata
            tool_impl.__name__ = name
            tool_impl.__doc__ = f"""{description}
            
Method: {method}
Path: {path}
"""
            
            # Create wrapper with explicit parameters
            if all_params:
                # Build parameter string
                param_strs = []
                for param_name, param_type, required in all_params:
                    if required:
                        param_strs.append(f"{param_name}: {param_type}")
                    else:
                        param_strs.append(f"{param_name}: {param_type} = {param_defaults.get(param_name, None)}")
                
                param_str = ", ".join(param_strs)
                
                # Create wrapper function with exec
                func_def = f"""
async def {name}({param_str}) -> Dict[str, Any]:
    kwargs = {{}}
"""
                
                # Add parameter assignments
                for param_name, _, _ in all_params:
                    func_def += f"    kwargs['{param_name}'] = {param_name}\n"
                
                func_def += "    return await tool_impl(**kwargs)\n"
                
                # Execute to create function
                namespace = {
                    'Dict': Dict,
                    'Any': Any,
                    'Optional': Optional,
                    'tool_impl': tool_impl
                }
                exec(func_def, namespace)
                return namespace[name]
            else:
                # No parameters, return as-is
                return tool_impl
        
        # Create and register the tool
        tool_func = create_tool_function()
        tool_func.__doc__ = description
        
        # Register with MCP
        mcp.tool()(tool_func)
        self.registered_tools[name] = tool_func
    
    def _get_path_from_postman_url(self, url: Any) -> str:
        """Extract path from Postman URL."""
        if isinstance(url, dict):
            return '/' + '/'.join(str(p) for p in url.get('path', []))
        elif isinstance(url, str):
            from urllib.parse import urlparse
            return urlparse(url).path
        return '/'
    
    def _categorize_path(self, path: str) -> str:
        """Categorize path based on keywords."""
        path_lower = path.lower()
        if 'account' in path_lower:
            return 'Account Management'
        elif 'customer' in path_lower:
            return 'Customer Management'
        elif 'loan' in path_lower:
            return 'Loan Management'
        elif 'financial' in path_lower or 'booking' in path_lower:
            return 'Financial Accounting'
        return 'General'
    
    def _infer_param_name(self, path_parts: List[str], index: int) -> str:
        """Infer parameter name from context."""
        if index > 0:
            prev_part = path_parts[index - 1].lower()
            if 'account' in prev_part:
                return 'accountId'
            elif 'customer' in prev_part:
                return 'customerId'
            elif 'loan' in prev_part:
                return 'loanId'
            elif 'collateral' in prev_part:
                return 'collateralId'
            elif 'cash' in prev_part and 'block' in prev_part:
                return 'cashBlockId'
        return 'id'
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for function."""
        name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name).lower()
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        name = re.sub(r'_+', '_', name).strip('_')
        if name and name[0].isdigit():
            name = 'api_' + name
        return name
    
    def _sanitize_param_name(self, name: str) -> str:
        """Sanitize parameter name."""
        name = name.replace('-', '_').replace('.', '_').replace(' ', '_')
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        if name and name[0].isdigit():
            name = 'param_' + name
        return name.lower()
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers from environment."""
        headers = {}
        
        # TCS BaNCS specific headers
        headers['entity'] = os.getenv('API_ENTITY', 'GPRDTTSTOU')
        headers['userId'] = os.getenv('API_USER_ID', '1')
        headers['languageCode'] = os.getenv('API_LANGUAGE_CODE', '1')
        
        # Auth
        api_key = os.getenv('API_KEY')
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        
        return headers

# Example usage
if __name__ == "__main__":
    # Create handler
    handler = SimplifiedDynamicAPIHandler(
        "AWSBancs Collection.postman_collection (1).json",
        base_url="https://demoapps.tcsbancs.com"
    )
    
    # Create MCP server
    mcp = FastMCP("TCS BaNCS Dynamic API")
    
    # Register tools by category
    handler.register_tools_by_category(
        mcp,
        categories=["Account Management", "Customer Management"],
        max_per_category=10
    )
    
    # Or register all tools
    # handler.register_all_tools(mcp, max_tools=20)
    
    print(f"Registered {len(handler.registered_tools)} tools")
    
    # Run server
    mcp.run()