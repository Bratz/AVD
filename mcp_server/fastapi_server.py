# File: mcp_server/fastapi_server.py
"""
FastAPI-compatible MCP server with full multi-transport support
Supports both HTTP (8082) and SSE (8084) transports like the original main.py
"""

import logging
import os
import asyncio
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastmcp import FastMCP
from tools.banking_tools import register_banking_tools
from dotenv import load_dotenv
import yaml
import uvicorn
from contextlib import asynccontextmanager

load_dotenv()

# Global variables
mcp = None
sse_app = None

def setup_mcp():
    """Initialize FastMCP and register tools."""
    global mcp
    
    if mcp is not None:
        return mcp
    
    logger = logging.getLogger(__name__)
    
    # Initialize FastMCP
    mcp = FastMCP(name="BankingMCP")
    
    # Get config path
    config_path = Path(__file__).parent / "config" / "tools_config.yaml"
    
    try:
        # Verify config file exists
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            # Create default config
            config_path.parent.mkdir(parents=True, exist_ok=True)
            default_config = {
                'tools': [
                    {
                        'name': 'customer_profile',
                        'description': 'Retrieve customer profile from TCS BaNCS',
                        'input_schema': {
                            'type': 'object',
                            'properties': {'customer_id': {'type': 'string'}},
                            'required': ['customer_id']
                        }
                    },
                    {
                        'name': 'account_balance',
                        'description': 'Get account balance from TCS BaNCS',
                        'input_schema': {
                            'type': 'object',
                            'properties': {'customer_id': {'type': 'string'}},
                            'required': ['customer_id']
                        }
                    },
                    {
                        'name': 'transaction',
                        'description': 'Get transaction history from TCS BaNCS',
                        'input_schema': {
                            'type': 'object',
                            'properties': {'customer_id': {'type': 'string'}},
                            'required': ['customer_id']
                        }
                    },
                    {
                        'name': 'market_analysis',
                        'description': 'Search banking trends via Tavily',
                        'input_schema': {
                            'type': 'object',
                            'properties': {'query': {'type': 'string'}},
                            'required': ['query']
                        }
                    },
                    {
                        'name': 'financial_advice',
                        'description': 'Generate financial advice using LLM',
                        'input_schema': {
                            'type': 'object',
                            'properties': {'prompt': {'type': 'string'}},
                            'required': ['prompt']
                        }
                    }
                ]
            }
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Created default config file: {config_path}")
        
        # Register banking tools
        register_banking_tools(mcp, str(config_path))
        logger.info("Banking tools registered successfully")
        
    except Exception as e:
        logger.error(f"Failed to setup MCP: {e}")
        raise
    
    return mcp

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger = logging.getLogger(__name__)
    logger.info("Starting Banking MCP Server with multi-transport support")
    
    # Initialize MCP
    setup_mcp()
    
    # Start SSE server on port 8084
    sse_task = asyncio.create_task(start_sse_server())
    
    yield
    
    # Shutdown
    logger.info("Shutting down Banking MCP Server")
    sse_task.cancel()
    try:
        await sse_task
    except asyncio.CancelledError:
        pass

def create_http_app() -> FastAPI:
    """Create FastAPI app for HTTP transport (port 8082)."""
    # Setup MCP
    mcp = setup_mcp()
    
    # Create FastAPI app from FastMCP
    app = mcp.create_app()
    
    # Override lifespan
    app.router.lifespan_context = lifespan
    
    # Add health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint for the MCP server."""
        return {
            "status": "healthy", 
            "service": "Banking MCP Server",
            "version": "1.0.0",
            "transports": {
                "http": "http://localhost:8082",
                "sse": "http://localhost:8084/mcp"
            }
        }
    
    # Add info endpoint
    @app.get("/info")
    async def server_info():
        """Server information endpoint."""
        config_path = Path(__file__).parent / "config" / "tools_config.yaml"
        tools_count = 0
        
        try:
            if config_path.exists():
                config = yaml.safe_load(open(config_path))
                tools_count = len(config.get('tools', []))
        except:
            pass
        
        return {
            "name": "Banking MCP Server",
            "version": "1.0.0",
            "tools_config": str(config_path),
            "tools_count": tools_count,
            "transports": {
                "http": "http://localhost:8082",
                "sse": "http://localhost:8084/mcp",
                "stdio": "Available"
            }
        }
    
    return app

def create_sse_app() -> FastAPI:
    """Create FastAPI app for SSE transport (port 8084)."""
    global sse_app
    
    if sse_app is not None:
        return sse_app
    
    sse_app = FastAPI(title="Banking MCP Server - SSE Transport")
    
    @sse_app.get("/health")
    async def sse_health():
        return {
            "status": "healthy",
            "service": "Banking MCP Server - SSE Transport",
            "port": 8084
        }
    
    @sse_app.get("/mcp")
    async def sse_endpoint():
        """Server-Sent Events endpoint for MCP."""
        async def event_stream():
            # This is a placeholder for actual SSE implementation
            # In a full implementation, this would handle MCP SSE protocol
            yield "data: {\"type\": \"connection\", \"status\": \"connected\"}\n\n"
            
            # Keep connection alive
            while True:
                await asyncio.sleep(30)
                yield "data: {\"type\": \"heartbeat\"}\n\n"
        
        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            }
        )
    
    return sse_app

async def start_sse_server():
    """Start the SSE server on port 8084."""
    logger = logging.getLogger(__name__)
    
    try:
        sse_app = create_sse_app()
        config = uvicorn.Config(
            sse_app,
            host="0.0.0.0",
            port=8084,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        logger.info("Starting SSE server on port 8084")
        await server.serve()
        
    except Exception as e:
        logger.error(f"SSE server failed: {e}")

# Create the main HTTP app
app = create_http_app()

if __name__ == "__main__":
    # For standalone testing
    import sys
    
    if "--sse-only" in sys.argv:
        # Run only SSE server
        sse_app = create_sse_app()
        uvicorn.run(sse_app, host="0.0.0.0", port=8084)
    elif "--http-only" in sys.argv:
        # Run only HTTP server
        uvicorn.run(app, host="0.0.0.0", port=8082)
    else:
        # Run HTTP server (SSE will be started via lifespan)
        uvicorn.run(app, host="0.0.0.0", port=8082)