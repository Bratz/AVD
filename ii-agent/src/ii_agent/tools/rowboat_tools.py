# ============================================================
# FILE 3: src/ii_agent/tools/rowboat_tools.py
# ============================================================
"""
Enhanced ROWBOAT Tool System
Week 2 implementation: Mock tools, Webhook tools, and MCP integration
"""

import json
import asyncio
import aiohttp
import jwt
import logging
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from abc import ABC, abstractmethod

from src.ii_agent.tools.base import BaseTool, ToolExecutionError

logger = logging.getLogger(__name__)



class MockTool(BaseTool):
    """AI-powered mock responses for testing workflows"""
    
    def __init__(self, llm_client, tool_configs: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="mock_tool",
            description="Generate realistic mock responses for any tool"
        )
        self.llm_client = llm_client
        self.tool_configs = tool_configs or {}
        self.mock_history = []  # Track mock calls for consistency
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Generate mock response using LLM"""
        
        tool_name = kwargs.get("tool_name", "unknown_tool")
        tool_args = kwargs.get("arguments", {})
        expected_format = kwargs.get("format", None)
        
        # Check if we have a specific config for this tool
        tool_config = self.tool_configs.get(tool_name, {})
        
        # Build prompt for LLM
        prompt = self._build_mock_prompt(tool_name, tool_args, expected_format, tool_config)
        
        try:
            # Generate mock response
            if hasattr(self.llm_client, 'generate'):
                response = await self.llm_client.generate(prompt)
            else:
                # Fallback for sync clients
                response = self.llm_client.complete(prompt)
            
            # Parse response
            mock_data = self._parse_mock_response(response, tool_name)
            
            # Track mock call
            self.mock_history.append({
                "tool_name": tool_name,
                "arguments": tool_args,
                "response": mock_data,
                "timestamp": datetime.utcnow()
            })
            
            return mock_data
            
        except Exception as e:
            logger.error(f"Error generating mock for {tool_name}: {e}")
            return self._get_fallback_mock(tool_name, tool_args)
    
    def _build_mock_prompt(
        self, 
        tool_name: str, 
        args: Dict[str, Any], 
        format: Optional[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> str:
        """Build prompt for mock generation"""
        
        prompt = f"""Generate a realistic mock response for the following tool call:

Tool Name: {tool_name}
Arguments: {json.dumps(args, indent=2)}
"""
        
        if format:
            prompt += f"\nExpected Response Format: {json.dumps(format, indent=2)}"
        
        if config.get("description"):
            prompt += f"\nTool Description: {config['description']}"
        
        if config.get("examples"):
            prompt += "\n\nExample Responses:"
            for example in config["examples"][:3]:  # Limit to 3 examples
                prompt += f"\n{json.dumps(example, indent=2)}"
        
        prompt += """

Generate a realistic response that:
1. Matches the expected format exactly
2. Contains plausible data based on the tool name and arguments
3. Is returned as valid JSON

Response:"""
        
        return prompt
    
    def _parse_mock_response(self, response: str, tool_name: str) -> Dict[str, Any]:
        """Parse LLM response into mock data"""
        
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            
            return json.loads(response)
            
        except json.JSONDecodeError:
            # If parsing fails, create a simple mock
            return {
                "status": "success",
                "tool": tool_name,
                "data": response.strip(),
                "mock": True,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _get_fallback_mock(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback mock when LLM fails"""
        
        # Tool-specific fallbacks
        if "weather" in tool_name.lower():
            return {
                "temperature": 72,
                "condition": "partly cloudy",
                "humidity": 65,
                "location": args.get("location", "Unknown")
            }
        elif "search" in tool_name.lower():
            return {
                "results": [
                    {"title": "Mock Result 1", "url": "https://example.com/1"},
                    {"title": "Mock Result 2", "url": "https://example.com/2"}
                ],
                "query": args.get("query", "")
            }
        else:
            return {
                "status": "success",
                "message": f"Mock response for {tool_name}",
                "data": args,
                "timestamp": datetime.utcnow().isoformat()
            }



@dataclass
class WebhookConfig:
    """Configuration for webhook tool"""
    url: str
    method: str = "POST"
    headers: Dict[str, str] = None
    auth_type: str = "jwt"  # jwt, bearer, basic, none
    secret: Optional[str] = None
    timeout: int = 30
    retry_count: int = 3
    retry_delay: int = 1

class WebhookTool(BaseTool):
    """External webhook integration with authentication"""
    
    def __init__(self, config: WebhookConfig):
        super().__init__(
            name=f"webhook_{config.url.split('/')[-1]}",
            description=f"Call webhook at {config.url}"
        )
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute webhook with authentication and retries"""
        
        # Prepare request
        headers = self._prepare_headers(kwargs)
        payload = self._prepare_payload(kwargs)
        
        # Execute with retries
        last_error = None
        for attempt in range(self.config.retry_count):
            try:
                response = await self._make_request(headers, payload)
                return self._process_response(response)
                
            except Exception as e:
                last_error = e
                if attempt < self.config.retry_count - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    continue
        
        raise ToolExecutionError(f"Webhook failed after {self.config.retry_count} attempts: {last_error}")
    
    def _prepare_headers(self, kwargs: Dict[str, Any]) -> Dict[str, str]:
        """Prepare request headers with authentication"""
        
        headers = {
            "Content-Type": "application/json",
            "X-Request-ID": str(kwargs.get("request_id", datetime.utcnow().timestamp())),
            **(self.config.headers or {})
        }
        
        # Add authentication
        if self.config.auth_type == "jwt" and self.config.secret:
            token = self._generate_jwt(kwargs)
            headers["Authorization"] = f"Bearer {token}"
        elif self.config.auth_type == "bearer" and self.config.secret:
            headers["Authorization"] = f"Bearer {self.config.secret}"
        elif self.config.auth_type == "basic" and self.config.secret:
            import base64
            encoded = base64.b64encode(self.config.secret.encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"
        
        return headers
    
    def _generate_jwt(self, kwargs: Dict[str, Any]) -> str:
        """Generate JWT token for authentication"""
        
        payload = {
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(minutes=5),
            "data": kwargs,
            "tool": self.name
        }
        
        return jwt.encode(payload, self.config.secret, algorithm="HS256")
    
    def _prepare_payload(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare request payload"""
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "tool": self.name,
            "arguments": kwargs,
            "metadata": {
                "retry_count": self.config.retry_count,
                "timeout": self.config.timeout
            }
        }
    
    async def _make_request(
        self, 
        headers: Dict[str, str], 
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make HTTP request to webhook"""
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        
        async with self.session.request(
            method=self.config.method,
            url=self.config.url,
            json=payload,
            headers=headers,
            timeout=timeout
        ) as response:
            response.raise_for_status()
            return await response.json()
    
    def _process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Process webhook response"""
        
        return {
            "status": "success",
            "webhook_response": response,
            "timestamp": datetime.utcnow().isoformat(),
            "tool": self.name
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()



class RAGSearchTool(BaseTool):
    """Enhanced RAG search with reranking and filtering"""
    
    def __init__(
        self, 
        vector_store,
        reranker=None,
        metadata_filters: Optional[List[str]] = None
    ):
        super().__init__(
            name="rag_search",
            description="Search knowledge base with semantic understanding"
        )
        self.vector_store = vector_store
        self.reranker = reranker
        self.metadata_filters = metadata_filters or []
        self.search_history = []
    
    async def execute(
        self, 
        query: str, 
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = True,
        return_metadata: bool = True
    ) -> Dict[str, Any]:
        """Execute semantic search with optional reranking"""
        
        start_time = datetime.utcnow()
        
        # Prepare filters
        search_filters = self._prepare_filters(filters)
        
        # Search vector store (get more results if reranking)
        initial_k = k * 3 if rerank and self.reranker else k
        
        try:
            results = await self._vector_search(query, initial_k, search_filters)
            
            # Rerank if available and requested
            if rerank and self.reranker and len(results) > k:
                results = await self._rerank_results(query, results, k)
            
            # Process results
            processed_results = self._process_results(results, return_metadata)
            
            # Track search
            search_record = {
                "query": query,
                "filters": filters,
                "result_count": len(processed_results),
                "timestamp": start_time,
                "duration_ms": int((datetime.utcnow() - start_time).total_seconds() * 1000)
            }
            self.search_history.append(search_record)
            
            return {
                "query": query,
                "results": processed_results,
                "metadata": {
                    "total_results": len(processed_results),
                    "search_time_ms": search_record["duration_ms"],
                    "reranked": rerank and self.reranker is not None,
                    "filters_applied": bool(filters)
                }
            }
            
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            raise ToolExecutionError(f"Search failed: {str(e)}")
    
    async def _vector_search(
        self, 
        query: str, 
        k: int, 
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Perform vector search"""
        
        if hasattr(self.vector_store, 'asearch'):
            # Async search
            return await self.vector_store.asearch(
                query=query,
                k=k,
                filter=filters
            )
        else:
            # Sync search in thread pool
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.vector_store.search,
                query,
                k,
                filters
            )
    
    async def _rerank_results(
        self, 
        query: str, 
        results: List[Dict[str, Any]], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Rerank search results"""
        
        if hasattr(self.reranker, 'arerank'):
            reranked = await self.reranker.arerank(
                query=query,
                documents=[r.get('content', r.get('text', '')) for r in results],
                top_k=top_k
            )
        else:
            # Sync reranking
            import asyncio
            loop = asyncio.get_event_loop()
            reranked = await loop.run_in_executor(
                None,
                self.reranker.rerank,
                query,
                [r.get('content', r.get('text', '')) for r in results],
                top_k
            )
        
        # Map reranked indices back to original results
        reranked_results = []
        for idx, score in reranked:
            result = results[idx].copy()
            result['rerank_score'] = score
            reranked_results.append(result)
        
        return reranked_results
    
    def _prepare_filters(self, user_filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare search filters"""
        
        filters = {}
        
        # Apply metadata filters
        for filter_key in self.metadata_filters:
            if user_filters and filter_key in user_filters:
                filters[filter_key] = user_filters[filter_key]
        
        return filters
    
    def _process_results(
        self, 
        results: List[Dict[str, Any]], 
        include_metadata: bool
    ) -> List[Dict[str, Any]]:
        """Process and format search results"""
        
        processed = []
        
        for result in results:
            item = {
                "content": result.get('content', result.get('text', '')),
                "score": result.get('score', result.get('rerank_score', 0))
            }
            
            if include_metadata:
                item["metadata"] = {
                    k: v for k, v in result.items() 
                    if k not in ['content', 'text', 'score', 'rerank_score']
                }
            
            processed.append(item)
        
        return processed



class MCPTransport(ABC):
    """Abstract transport for MCP communication"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to MCP server"""
        pass
    
    @abstractmethod
    async def send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to MCP server"""
        pass
    
    @abstractmethod
    async def close(self):
        """Close connection"""
        pass

class HTTPMCPTransport(MCPTransport):
    """HTTP/SSE transport for MCP"""
    
    def __init__(self, base_url: str, auth_token: Optional[str] = None):
        self.base_url = base_url
        self.auth_token = auth_token
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def connect(self) -> bool:
        """Initialize HTTP session"""
        
        if not self.session:
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            
            self.session = aiohttp.ClientSession(headers=headers)
        
        return True
    
    async def send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send HTTP request to MCP server"""
        
        if not self.session:
            await self.connect()
        
        url = f"{self.base_url}/mcp/v1/{method}"
        
        async with self.session.post(url, json=params) as response:
            response.raise_for_status()
            return await response.json()
    
    async def close(self):
        """Close HTTP session"""
        
        if self.session:
            await self.session.close()
            self.session = None

class MCPToolClient:
    """Client for MCP (Model Context Protocol) servers"""
    
    def __init__(self, server_configs: List[Dict[str, Any]]):
        self.servers: Dict[str, MCPTransport] = {}
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.tool_schemas: Dict[str, Any] = {}
        self._initialize_servers(server_configs)
    
    def _initialize_servers(self, configs: List[Dict[str, Any]]):
        """Initialize MCP server connections"""
        
        for config in configs:
            server_name = config["name"]
            
            # Try to use native MCP client if available
            try:
                from mcp import MCPClient, StreamableHTTPTransport
                
                transport = StreamableHTTPTransport(
                    base_url=config["url"],
                    auth_token=config.get("auth_token")
                )
                
                client = MCPClient(transport=transport)
                self.servers[server_name] = client
                
            except ImportError:
                # Fallback to HTTP transport
                logger.info(f"MCP package not available, using HTTP transport for {server_name}")
                self.servers[server_name] = HTTPMCPTransport(
                    base_url=config["url"],
                    auth_token=config.get("auth_token")
                )
    
    async def discover_tools(self) -> Dict[str, Any]:
        """Discover all available tools from MCP servers"""
        
        all_tools = {}
        
        for server_name, transport in self.servers.items():
            try:
                # Connect to server
                await transport.connect()
                
                # List tools
                if hasattr(transport, 'list_tools'):
                    # Native MCP client
                    tools_response = await transport.list_tools()
                else:
                    # HTTP transport
                    tools_response = await transport.send_request("tools/list", {})
                
                # Process tools
                tools = tools_response.get("tools", [])
                for tool in tools:
                    tool_id = f"{server_name}:{tool['name']}"
                    all_tools[tool_id] = {
                        **tool,
                        "server": server_name
                    }
                    
                    # Store schema if available
                    if "inputSchema" in tool:
                        self.tool_schemas[tool_id] = tool["inputSchema"]
                
                logger.info(f"Discovered {len(tools)} tools from {server_name}")
                
            except Exception as e:
                logger.error(f"Error discovering tools from {server_name}: {e}")
        
        self.tools = all_tools
        return all_tools
    
    async def execute_tool(
        self, 
        tool_id: str, 
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool via MCP"""
        
        if tool_id not in self.tools:
            raise ValueError(f"Unknown tool: {tool_id}")
        
        tool_info = self.tools[tool_id]
        server_name = tool_info["server"]
        tool_name = tool_info["name"]
        
        transport = self.servers[server_name]
        
        try:
            # Validate arguments against schema if available
            if tool_id in self.tool_schemas:
                self._validate_arguments(arguments, self.tool_schemas[tool_id])
            
            # Execute tool
            if hasattr(transport, 'execute_tool'):
                # Native MCP client
                result = await transport.execute_tool(tool_name, arguments)
            else:
                # HTTP transport
                result = await transport.send_request(
                    "tools/execute",
                    {
                        "tool": tool_name,
                        "arguments": arguments
                    }
                )
            
            return {
                "tool_id": tool_id,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_id}: {e}")
            raise ToolExecutionError(f"MCP tool execution failed: {str(e)}")
    
    def _validate_arguments(self, arguments: Dict[str, Any], schema: Dict[str, Any]):
        """Validate tool arguments against schema"""
        
        # Basic validation - can be enhanced with jsonschema
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        
        # Check required fields
        for field in required:
            if field not in arguments:
                raise ValueError(f"Missing required field: {field}")
        
        # Check field types (basic)
        for field, value in arguments.items():
            if field in properties:
                expected_type = properties[field].get("type")
                if expected_type:
                    # Basic type checking
                    if expected_type == "string" and not isinstance(value, str):
                        raise ValueError(f"Field {field} must be a string")
                    elif expected_type == "number" and not isinstance(value, (int, float)):
                        raise ValueError(f"Field {field} must be a number")
                    elif expected_type == "boolean" and not isinstance(value, bool):
                        raise ValueError(f"Field {field} must be a boolean")
    
    async def cleanup(self):
        """Cleanup all server connections"""
        
        for transport in self.servers.values():
            try:
                await transport.close()
            except Exception as e:
                logger.error(f"Error closing transport: {e}")

class MCPTool(BaseTool):
    """Wrapper for MCP tools in ROWBOAT"""
    
    def __init__(self, mcp_client: MCPToolClient, tool_id: str):
        tool_info = mcp_client.tools.get(tool_id, {})
        
        super().__init__(
            name=tool_id,
            description=tool_info.get("description", "MCP tool")
        )
        
        self.mcp_client = mcp_client
        self.tool_id = tool_id
        self.schema = mcp_client.tool_schemas.get(tool_id, {})
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute MCP tool"""
        
        return await self.mcp_client.execute_tool(self.tool_id, kwargs)



class ROWBOATToolRegistry:
    """Registry for managing ROWBOAT tools"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.mock_tool: Optional[MockTool] = None
        self.mcp_client: Optional[MCPToolClient] = None
    
    def register_tool(self, tool: BaseTool):
        """Register a tool"""
        
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def register_webhook(self, name: str, config: WebhookConfig):
        """Register a webhook tool"""
        
        tool = WebhookTool(config)
        tool.name = name  # Override name
        self.register_tool(tool)
    
    async def register_mcp_servers(self, server_configs: List[Dict[str, Any]]):
        """Register MCP servers and discover tools"""
        
        self.mcp_client = MCPToolClient(server_configs)
        tools = await self.mcp_client.discover_tools()
        
        # Create tool wrappers
        for tool_id in tools:
            mcp_tool = MCPTool(self.mcp_client, tool_id)
            self.register_tool(mcp_tool)
        
        logger.info(f"Registered {len(tools)} MCP tools")
    
    def enable_mocking(self, llm_client, tool_configs: Optional[Dict[str, Any]] = None):
        """Enable mock tool for testing"""
        
        self.mock_tool = MockTool(llm_client, tool_configs)
        logger.info("Mock tool enabled")
    
    async def execute_tool(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any],
        use_mock: bool = False
    ) -> Dict[str, Any]:
        """Execute a tool by name"""
        
        # Use mock if requested and available
        if use_mock and self.mock_tool:
            return await self.mock_tool.execute(
                tool_name=tool_name,
                arguments=arguments
            )
        
        # Find and execute real tool
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        tool = self.tools[tool_name]
        return await tool.execute(**arguments)
    
    def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """Get tool schema/description"""
        
        if tool_name not in self.tools:
            return {}
        
        tool = self.tools[tool_name]
        
        schema = {
            "name": tool.name,
            "description": tool.description
        }
        
        # Add MCP schema if available
        if isinstance(tool, MCPTool):
            schema["inputSchema"] = tool.schema
        
        return schema
    
    async def cleanup(self):
        """Cleanup all tools"""
        
        # Cleanup webhook tools
        for tool in self.tools.values():
            if hasattr(tool, 'cleanup'):
                await tool.cleanup()
        
        # Cleanup MCP client
        if self.mcp_client:
            await self.mcp_client.cleanup()
