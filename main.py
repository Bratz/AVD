"""
Updated main.py with proper OAuth integration
OAuth is integrated into the MCP server, not run as a separate proxy
"""
import asyncio
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

async def main():
    """Entry point with architecture selection."""
    
    # Configure logging
    logger = logging.getLogger('banking_query_system')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = RotatingFileHandler(
        'banking_system.log', maxBytes=5*1024*1024, backupCount=3
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Check architecture mode
    use_clean_architecture = os.getenv("USE_CLEAN_ARCHITECTURE", "false").lower() == "true"
    use_oauth = os.getenv("ENABLE_OAUTH", "false").lower() == "true"
    
    try:
        if use_clean_architecture:
            logger.info("Starting with Clean Architecture")
            
            if use_oauth:
                logger.info("OAuth is ENABLED - Using OAuth MCP Wrapper")
                # Use the OAuth wrapper instead of regular server
                from mcp_server.oauth_mcp_wrapper import app as mcp_app
                
                # Run OAuth-enabled MCP server
                mcp_config = uvicorn.Config(
                    mcp_app,
                    host="0.0.0.0",
                    port=8082,
                    log_level="info"
                )
                mcp_server = uvicorn.Server(mcp_config)
                mcp_task = asyncio.create_task(mcp_server.serve())
                
                logger.info("OAuth-enabled MCP Server started on port 8082")
                logger.info("OAuth metadata: http://localhost:8082/.well-known/oauth-protected-resource")
                
            else:
                logger.info("OAuth is DISABLED - Using standard MCP server")
                # Start standard MCP Server
                from mcp_server.server import BankingMCPServer
                mcp_server = BankingMCPServer(
                    config_path=os.path.join("mcp_server", "config", "tools_config.yaml")
                )
                await mcp_server.initialize()
                mcp_task = asyncio.create_task(mcp_server.run())
            
            # # Start Enhanced Web Server (II-Agent)
            # from app.api.II_web_server import app
            # web_config = uvicorn.Config(
            #     app, 
            #     host="0.0.0.0", 
            #     port=8000,
            #     log_level="debug"
            # )
            # web_server = uvicorn.Server(web_config)
            # web_task = asyncio.create_task(web_server.serve())
            
            # logger.info("Enhanced Web Server (II-Agent) started on port 8000")
            
            # Wait for both servers
            # await asyncio.gather(mcp_task, web_task)
            await asyncio.gather(mcp_task)
        else:
            logger.info("Starting with Legacy Architecture")
            
            if use_oauth:
                logger.info("OAuth is ENABLED - Using OAuth MCP Wrapper")
                # Import and run the OAuth wrapper as the main server
                from mcp_server.oauth_mcp_wrapper import app as mcp_app
                
                # Configure server
                port = int(os.getenv("MCP_PORT", "8082"))
                
                logger.info(f"🚀 Banking MCP Server with OAuth")
                logger.info(f"=" * 50)
                logger.info(f"📍 Server URL: http://localhost:{port}")
                logger.info(f"🔐 OAuth: ENABLED")
                logger.info(f"🔑 Keycloak: {os.getenv('KEYCLOAK_URL')}/realms/banking-mcp")
                logger.info(f"📡 Endpoints:")
                logger.info(f"   - MCP: http://localhost:{port}/mcp")
                logger.info(f"   - Health: http://localhost:{port}/health")
                logger.info(f"   - OAuth Metadata: http://localhost:{port}/.well-known/oauth-protected-resource")
                logger.info(f"\n📝 For MCP Inspector: Use http://localhost:{port}")
                
                # Run the server
                await uvicorn.Server(
                    uvicorn.Config(
                        mcp_app,
                        host="0.0.0.0",
                        port=port,
                        log_level="info"
                    )
                ).serve()
                
            else:
                logger.info("OAuth is DISABLED - Using standard MCP server")
                # Start original MCP Server
                from mcp_server.server import BankingMCPServer
                server = BankingMCPServer(
                    config_path=os.path.join("mcp_server", "config", "tools_config.yaml")
                )
                await server.initialize()
                await server.run()
            
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        # Print startup banner
        print("\n" + "="*60)
        print("🏦 Banking MCP System")
        print("="*60)
        print(f"Architecture: {'Clean' if os.getenv('USE_CLEAN_ARCHITECTURE', 'false').lower() == 'true' else 'Legacy'}")
        print(f"OAuth: {'ENABLED' if os.getenv('ENABLE_OAUTH', 'false').lower() == 'true' else 'DISABLED'}")
        print("="*60 + "\n")
        
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)