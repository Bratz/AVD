"""
CLI interface for the Agent with OAuth support.

This script provides a command-line interface for interacting with the Agent.
It instantiates an Agent and prompts the user for input, which is then passed to the Agent.
"""

import os
import argparse
import logging
import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from src.ii_agent.core.event import RealtimeEvent, EventType
from src.ii_agent.utils.constants import TOKEN_BUDGET
from utils import parse_common_args, create_workspace_manager_for_connection
from rich.console import Console
from rich.panel import Panel

from src.ii_agent.tools import get_system_tools
from src.ii_agent.tools.message_tool import MessageTool
from src.ii_agent.tools.complete_tool import ReturnControlToUserTool, CompleteTool
from src.ii_agent.prompts.system_prompt import (
    SYSTEM_PROMPT, 
    SYSTEM_PROMPT_WITH_SEQ_THINKING,
    get_banking_prompt
)
from src.ii_agent.agents.anthropic_fc import AnthropicFC
from src.ii_agent.utils import WorkspaceManager
from src.ii_agent.llm import get_client
from src.ii_agent.llm.context_manager.llm_summarizing import LLMSummarizingContextManager
from src.ii_agent.llm.token_counter import TokenCounter
from src.ii_agent.db.manager import DatabaseManager
from src.ii_agent.llm.ollama import OllamaContextManager, OllamaAdaptiveContextManager

# MCP/Banking imports
from wrappers.mcp_client_wrapper import MCPClientWrapper
from src.ii_agent.tools.banking_tool_registry import mcp_tool_registry
from src.ii_agent.tools.mcp_tool_adapter import create_mcp_tool_adapters

# OAuth imports
try:
    from src.ii_agent.utils.oauth_utils import OAuthTokenManager
    OAUTH_AVAILABLE = True
except ImportError:
    OAUTH_AVAILABLE = False
    logging.warning("OAuth utilities not available. OAuth features will be disabled.")

MCP_AVAILABLE = True

from src.ii_agent.tools.base import LLMTool, ToolImplOutput

from enhanced_system.enhanced_agent_system import (
    IntelligentAgentProposalSystem,
    AgentProposal,
    WorkflowComplexity
)
from enhanced_system.complexity_analyzer import ComplexityAnalyzer, QuickComplexityCheck
from a2a_integration.client import A2AClient
from rich.table import Table
from rich.prompt import Confirm, Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn

"""
Environment Variables for Agent Proposals with Chutes:
- ENABLE_AGENT_PROPOSALS: Enable intelligent agent proposal system (true/false)
- AGENT_STORAGE_PATH: Path to store saved agents (default: ./agents/saved)
- AGENT_CONFIG_PATH: Path to agent proposal configuration (default: ./config/agent_proposals.yaml)
- AGENT_PROPOSAL_THRESHOLD: Complexity threshold (simple/medium/complex)
- ENABLE_A2A: Enable A2A communication (true/false)
- A2A_BASE_URL: Base URL for A2A server (default: http://localhost:8001)
- CHUTES_API_KEY: Your Chutes API key (required)
- CHUTES_MODEL: Model to use for proposals (default: uses main model)

OAuth Environment Variables:
- ENABLE_OAUTH: Enable OAuth authentication (true/false)
- KEYCLOAK_URL: Keycloak server URL
- KEYCLOAK_REALM: Keycloak realm name
- OAUTH_TOKEN: Direct OAuth token (optional)
- OAUTH_REFRESH_TOKEN: Refresh token for automatic renewal (optional)
- OAUTH_CLIENT_ID: OAuth client ID
- OAUTH_CLIENT_SECRET: OAuth client secret (if using confidential client)
"""


MAX_OUTPUT_TOKENS_PER_TURN = 32768
MAX_TURNS = 200

# Banking-specific configuration
BANKING_MCP_SERVICES = {
    "core_banking": os.getenv("CORE_BANKING_URL", "http://localhost:8082"),
}

# Add these configuration constants after existing constants
AGENT_PROPOSAL_COMMANDS = {
    "/agents": "List all available agents",
    "/create": "Create a new specialized agent",
    "/a2a": "Show A2A communication status",
    "/proposals": "Show active agent proposals",
    "/help": "Show this help message"
}

def setup_comprehensive_logging(log_path: str, minimize_stdout: bool = False) -> logging.Logger:
    """Set up comprehensive logging that captures all loggers"""
    
    # Ensure log directory exists
    log_dir = os.path.dirname(log_path) if os.path.dirname(log_path) else '.'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        print(f"Created log directory: {log_dir}")
    
    # Remove existing log file
    if os.path.exists(log_path):
        os.remove(log_path)
        print(f"Removed existing log file: {log_path}")
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Create file handler
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Configure root logger to capture everything
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    # Remove any existing handlers
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    
    # Add console handler if not minimizing logs
    if not minimize_stdout:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # Configure specific loggers to ensure they're captured
    loggers_to_configure = [
        'agent_logs',
        'ii_agent',
        'src.ii_agent',
        'src.ii_agent.llm',
        'src.ii_agent.llm.ollama',
        'src.ii_agent.agents',
        'src.ii_agent.tools',
        'httpx',
        'asyncio',
    ]
    
    for logger_name in loggers_to_configure:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = True
    
    # Create and configure the main agent logger
    agent_logger = logging.getLogger('agent_logs')
    
    # Write initial log entries
    agent_logger.info("="*60)
    agent_logger.info("II-Agent CLI Starting")
    agent_logger.info(f"Log file: {os.path.abspath(log_path)}")
    agent_logger.info(f"Python version: {os.sys.version}")
    agent_logger.info(f"Current directory: {os.getcwd()}")
    agent_logger.info(f"OAuth support: {'Available' if OAUTH_AVAILABLE else 'Not available'}")
    agent_logger.info("="*60)
    
    # Flush to ensure logs are written
    for handler in root_logger.handlers:
        handler.flush()
    
    # Verify log file was created
    if os.path.exists(log_path):
        size = os.path.getsize(log_path)
        print(f"Log file created successfully: {log_path} ({size} bytes)")
    else:
        print(f"WARNING: Log file was not created at {log_path}")
    
    return agent_logger

async def get_oauth_token_interactive(console: Console) -> Optional[str]:
    """Get OAuth token interactively if not provided"""
    console.print("\n[bold yellow]OAuth Authentication Required[/bold yellow]")
    console.print("Choose an option:")
    console.print("1. Enter OAuth token manually")
    console.print("2. Get new token (opens browser)")
    console.print("3. Continue without OAuth (may fail)")
    
    choice = Prompt.ask("Select option", choices=["1", "2", "3"], default="3")
    
    if choice == "1":
        token = Prompt.ask("Enter OAuth token", password=True)
        return token
    elif choice == "2":
        console.print("[yellow]Opening browser for OAuth login...[/yellow]")
        # Import and run the OAuth flow
        try:
            from src.ii_agent.utils.test_oauth_flow import BankingOAuthTester
            tester = BankingOAuthTester()
            
            # Get authorization code
            auth_code = await tester.get_authorization_code_manual()
            if auth_code:
                # Exchange for token
                token_data = await tester.exchange_code_for_token(auth_code)
                if token_data and 'access_token' in token_data:
                    console.print("[green]✓ OAuth token obtained successfully[/green]")
                    return token_data['access_token']
        except Exception as e:
            console.print(f"[red]Failed to get OAuth token: {e}[/red]")
    
    return None

async def handle_special_command(
    command: str,
    proposal_system: Optional[IntelligentAgentProposalSystem],
    console: Console
) -> Optional[str]:
    """Handle special CLI commands for agent management"""
    
    if command == "/help":
        table = Table(title="Agent Proposal Commands", show_header=True)
        table.add_column("Command", style="cyan")
        table.add_column("Description", style="white")
        
        for cmd, desc in AGENT_PROPOSAL_COMMANDS.items():
            table.add_row(cmd, desc)
        
        console.print(table)
        return None
    
    elif command == "/agents" and proposal_system:
        # List all available agents
        agents = proposal_system.get_registry()
        
        if not agents:
            console.print("[yellow]No agents available[/yellow]")
            return None
        
        table = Table(title="Available Agents", show_header=True)
        table.add_column("ID", style="cyan", width=12)
        table.add_column("Name", style="green")
        table.add_column("Type", style="yellow")
        table.add_column("Usage", style="white")
        table.add_column("Success Rate", style="magenta")
        
        for agent in agents:
            usage = f"{agent.get('usage_count', 0)}"
            success = f"{agent.get('success_rate', 0):.1%}"
            table.add_row(
                agent['agent_id'][:12],
                agent['name'],
                agent['type'],
                usage,
                success
            )
        
        console.print(table)
        return None
    
    elif command == "/proposals" and proposal_system:
        # Show active proposals
        proposals = proposal_system.get_active_proposals()
        
        if not proposals:
            console.print("[yellow]No active proposals[/yellow]")
            return None
        
        table = Table(title="Active Agent Proposals", show_header=True)
        table.add_column("ID", style="cyan", width=12)
        table.add_column("Agent Name", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Complexity", style="magenta")
        
        for proposal in proposals:
            table.add_row(
                proposal.proposal_id[:12],
                proposal.agent_name,
                proposal.status.value,
                proposal.complexity_analysis.complexity_level.value
            )
        
        console.print(table)
        return None
    
    elif command.startswith("/create") and proposal_system:
        # Create a new agent manually
        task = Prompt.ask("[bold cyan]What task should this agent handle?[/bold cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task_id = progress.add_task("Analyzing task complexity...", total=None)
            
            # Analyze and propose
            proposal = await proposal_system.analyze_and_propose(task)
            
            progress.update(task_id, completed=True)
        
        if proposal:
            await display_and_confirm_proposal(proposal, proposal_system, console)
        else:
            console.print("[yellow]Task is simple enough for existing agents[/yellow]")
        
        return None
    
    elif command == "/a2a" and proposal_system:
        # Show A2A communication status
        active_agents = len(proposal_system.active_agents)
        console.print(f"[green]A2A Status:[/green] {active_agents} active agents")
        
        # In full implementation, would show active connections
        return None
    
    return command  # Not a special command, treat as regular input


# Add this function to display and confirm proposals
async def display_and_confirm_proposal(
    proposal: AgentProposal,
    proposal_system: IntelligentAgentProposalSystem,
    console: Console
) -> Optional[str]:
    """Display agent proposal and get user confirmation"""
    
    # Create proposal display panel
    console.print("\n[bold yellow]🤖 Agent Proposal[/bold yellow]")
    console.print(f"[cyan]Name:[/cyan] {proposal.agent_name}")
    console.print(f"[cyan]Description:[/cyan] {proposal.agent_description}")
    console.print(f"[cyan]Complexity:[/cyan] {proposal.complexity_analysis.complexity_level.value}")
    console.print(f"[cyan]Confidence:[/cyan] {proposal.complexity_analysis.confidence_score:.1%}")
    console.print(f"[cyan]Justification:[/cyan] {proposal.justification}")
    
    # Show capabilities
    console.print("\n[bold]Capabilities:[/bold]")
    for cap in proposal.capabilities:
        console.print(f"  • {cap}")
    
    # Show MCP tools
    if proposal.mcp_tools:
        console.print("\n[bold]MCP Tools:[/bold]")
        for tool in proposal.mcp_tools:
            console.print(f"  • {tool}")
    
    # Get confirmation
    if Confirm.ask("\n[bold yellow]Create this specialized agent?[/bold yellow]"):
        console.print("[green]✓ Creating agent...[/green]")
        
        # Confirm proposal
        await proposal_system.confirm_proposal(proposal.proposal_id)
        
        # Create agent
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task_id = progress.add_task("Creating specialized agent...", total=None)
            
            agent = await proposal_system.create_agent(proposal.proposal_id)
            
            progress.update(task_id, completed=True)
        
        if agent:
            console.print(f"[green]✓ Agent '{agent.name}' created successfully![/green]")
            return agent.agent_id
        else:
            console.print("[red]✗ Failed to create agent[/red]")
    else:
        await proposal_system.reject_proposal(proposal.proposal_id)
        console.print("[yellow]Agent creation cancelled[/yellow]")
    
    return None

async def initialize_proposal_system(
    llm_client,  # Changed from ollama_wrapper
    mcp_wrapper,
    args,
    logger_for_agent_logs
) -> Optional[IntelligentAgentProposalSystem]:
    """Initialize the intelligent agent proposal system with Chutes"""
    
    logger_for_agent_logs.info("Initializing Intelligent Agent Proposal System with Chutes...")
    
    try:
        # Create a wrapper to adapt Chutes client for the proposal system
        from enhanced_system.chutes_adapter import ChutesLLMAdapter
        
        chutes_adapter = ChutesLLMAdapter(llm_client)
        
        proposal_system = IntelligentAgentProposalSystem(
            ollama_wrapper=chutes_adapter,  # The system still expects this interface
            mcp_wrapper=mcp_wrapper,
            storage_path=args.agent_storage_path,
            config_path=args.agent_config_path
        )
        
        logger_for_agent_logs.info(
            f"Proposal system initialized with {len(proposal_system.agent_registry)} saved agents"
        )
        
        return proposal_system
        
    except Exception as e:
        logger_for_agent_logs.error(f"Failed to initialize proposal system: {e}")
        return None

def add_a2a_arguments(parser):
    """Add A2A communication related arguments"""
    
    a2a_group = parser.add_argument_group("A2A Communication Options")
    
    a2a_group.add_argument(
        "--enable-a2a",
        action="store_true",
        default=os.getenv("ENABLE_A2A", "false").lower() == "true",
        help="Enable Agent-to-Agent (A2A) communication"
    )
    
    a2a_group.add_argument(
        "--a2a-base-url",
        type=str,
        default=os.getenv("A2A_BASE_URL", "http://localhost:8001"),
        help="Base URL for A2A server"
    )
    
    a2a_group.add_argument(
        "--a2a-discovery-url",
        type=str,
        default=os.getenv("A2A_DISCOVERY_URL", "http://localhost:8000"),
        help="URL for A2A discovery service"
    )
    
    a2a_group.add_argument(
        "--a2a-base-port",
        type=int,
        default=int(os.getenv("A2A_BASE_PORT", "8001")),
        help="Starting port for A2A agent servers"
    )

def add_proposal_arguments(parser):
    """Add agent proposal related arguments"""
    
    proposal_group = parser.add_argument_group("Agent Proposal Options")
    
    proposal_group.add_argument(
        "--enable-agent-proposals",
        action="store_true",
        default=os.getenv("ENABLE_AGENT_PROPOSALS", "false").lower() == "true",
        help="Enable intelligent agent proposal system"
    )
    
    proposal_group.add_argument(
        "--agent-storage-path",
        type=str,
        default=os.getenv("AGENT_STORAGE_PATH", "./agents/saved"),
        help="Path to store saved agents"
    )
    
    proposal_group.add_argument(
        "--agent-config-path",
        type=str,
        default=os.getenv("AGENT_CONFIG_PATH", "./configs/agent_proposals.yaml"),
        help="Path to agent proposal configuration"
    )
    
    proposal_group.add_argument(
        "--proposal-threshold",
        type=str,
        choices=["simple", "medium", "complex"],
        default=os.getenv("AGENT_PROPOSAL_THRESHOLD", "medium"),
        help="Complexity threshold for proposing new agents"
    )

async def async_main():
    """Async main entry point"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="CLI for interacting with the Agent")
    parser = parse_common_args(parser)
    
    # Add MCP-inclusion arguments
    parser.add_argument(
        "--enable-mcp",
        action="store_true",
        default=bool(os.getenv("ENABLE_MCP", "false").lower() == "true"),
        help="Enable MCP tool integration"
    )
    
    # Add banking mode flag
    parser.add_argument(
        "--enable-banking",
        action="store_true",
        default=bool(os.getenv("ENABLE_BANKING", "false").lower() == "true"),
        help="Enable banking mode with MCP services"
    )
    
    parser.add_argument(
        "--user-role",
        type=str,
        choices=["admin", "customer", "guest"],
        default="customer",
        help="User role for banking operations"
    )

    # Add MCP-tools-only flag
    parser.add_argument(
        "--mcp-tools-only",
        action="store_true",
        default=bool(os.getenv("MCP_TOOLS_ONLY", "false").lower() == "true"),
        help="Load only MCP tools, exclude system tools (useful for banking)"
    )

    # Add Ollama-specific arguments
    parser.add_argument(
        "--ollama-base-url",
        type=str,
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Ollama server base URL"
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        help="Ollama model to use"
    )
    parser.add_argument(
    "--chutes-api-key",
    type=str,
    default=os.getenv("CHUTES_API_KEY"),
    help="Chutes.ai API key"
    )

    parser.add_argument(
        "--chutes-model",
        type=str,
        default=os.getenv("CHUTES_MODEL", "deepseek-ai/DeepSeek-V3-0324"),
        help="Chutes.ai model to use"
    )

    parser.add_argument(
        "--chutes-no-fallback",
        action="store_true",
        help="Disable fallback models for Chutes"
    )

    parser.add_argument(
        "--chutes-native-tools",
        action="store_true",
        help="Use native tool calling instead of JSON workaround"
    )

    add_proposal_arguments(parser)
    add_a2a_arguments(parser)
    args = parser.parse_args()



    # Set up comprehensive logging
    logger_for_agent_logs = setup_comprehensive_logging(
        args.logs_path, 
        args.minimize_stdout_logs
    )
    
    # Log parsed arguments
    logger_for_agent_logs.info(f"Parsed arguments: {vars(args)}")

    # Initialize console
    console = Console()

    # Initialize database manager
    db_manager = DatabaseManager()

    # Create a new workspace manager for the CLI session
    workspace_manager, session_id = create_workspace_manager_for_connection(
        args.workspace, args.use_container_workspace
    )
    workspace_path = workspace_manager.root

    # Create a new session
    db_manager.create_session(
        session_uuid=session_id, workspace_path=workspace_manager.root
    )
    logger_for_agent_logs.info(
        f"Created new session {session_id} with workspace at {workspace_manager.root}"
    )

    proposal_system = None
    
    # Initialize MCP wrapper(s)
    mcp_wrappers = {}
    available_tools = []
    token_manager = None

    # Define oauth_enabled here, before the MCP check
    oauth_enabled = os.getenv("ENABLE_OAUTH", "false").lower() == "true"
    oauth_token = None
    
    if args.enable_mcp and MCP_AVAILABLE:
        # Check if OAuth is enabled
        oauth_enabled = os.getenv("ENABLE_OAUTH", "false").lower() == "true"
        oauth_token = None
        
        if oauth_enabled and OAUTH_AVAILABLE:
            # Try to create token manager from environment
            token_manager = OAuthTokenManager.from_env()
            
            if token_manager:
                logger_for_agent_logs.info("OAuth token manager initialized")
                try:
                    # Try to get a valid token
                    oauth_token = await token_manager.get_valid_token()
                    if oauth_token:
                        logger_for_agent_logs.info("OAuth token obtained from token manager")
                except Exception as e:
                    logger_for_agent_logs.warning(f"Failed to get token from manager: {e}")
                    oauth_token = None
            
            # If no token from manager, try environment variable
            if not oauth_token:
                oauth_token = os.getenv("OAUTH_TOKEN")
            
            # If still no token and not minimizing output, prompt user
            if not oauth_token and not args.minimize_stdout_logs:
                oauth_token = await get_oauth_token_interactive(console)
                
                if not oauth_token:
                    if not Confirm.ask("Continue without OAuth token?"):
                        console.print("[yellow]Exiting...[/yellow]")
                        return
                    oauth_enabled = False

        if args.enable_banking:
            # Banking mode: Initialize MCP services
            logger_for_agent_logs.info("Initializing TCS BaNCS MCP services...")
            
            for service_name, service_url in BANKING_MCP_SERVICES.items():
                try:
                    # Create MCP wrapper with OAuth support
                    mcp_wrapper = MCPClientWrapper(
                        base_url=service_url,
                        sse_endpoint=os.getenv(f"{service_name.upper()}_SSE_URL", f"{service_url}/mcp/sse"),
                        api_key=os.getenv(f"{service_name.upper()}_API_KEY", "test-api-key-123"),
                        oauth_token=oauth_token  # Pass OAuth token
                    )
                    
                    # Set token manager if available
                    if token_manager:
                        mcp_wrapper.set_oauth_token_manager(token_manager)
                    
                    await mcp_wrapper.initialize()
                    mcp_wrappers[service_name] = mcp_wrapper
                    
                    # Log discovered tools
                    logger_for_agent_logs.info(f"{service_name}: {len(mcp_tool_registry._tools)} tools in registry")
                    
                    if not args.minimize_stdout_logs:
                        console.print(f"[green]✓[/green] {service_name}: Connected")
                        if oauth_enabled:
                            console.print(f"  [dim]OAuth: Enabled[/dim]")
                        
                except Exception as e:
                    error_msg = str(e)
                    if "401" in error_msg or "unauthorized" in error_msg.lower():
                        logger_for_agent_logs.error(f"Authentication failed for {service_name}: {e}")
                        if not args.minimize_stdout_logs:
                            console.print(f"[red]✗[/red] {service_name}: Authentication failed")
                            console.print("  [dim]Check OAuth token validity[/dim]")
                    else:
                        logger_for_agent_logs.warning(f"Failed to initialize {service_name}: {e}")
                        if not args.minimize_stdout_logs:
                            console.print(f"[yellow]⚠[/yellow] {service_name}: Failed - {str(e)}")
            
            # Display banking tools summary
            if not args.minimize_stdout_logs and mcp_wrappers:
                stats = mcp_tool_registry.get_statistics()
                console.print(f"\n[bold]TCS BaNCS Tools Summary:[/bold]")
                for category, count in stats["categories"].items():
                    console.print(f"  {category}: {count} tools")
        else:
            # Standard MCP mode: Single MCP service
            try:
                mcp_wrapper = MCPClientWrapper(
                    base_url=os.getenv("MCP_BASE_URL", "http://localhost:8082"),
                    sse_endpoint=os.getenv("MCP_SSE_URL", "http://localhost:8084/mcp"),
                    api_key=os.getenv("MCP_API_KEY", "test-api-key-123"),
                    oauth_token=oauth_token  # Pass OAuth token
                )
                
                # Set token manager if available
                if token_manager:
                    mcp_wrapper.set_oauth_token_manager(token_manager)
                    
                await mcp_wrapper.initialize()
                mcp_wrappers["default"] = mcp_wrapper

                # Log discovered tools
                available_tools = await mcp_wrapper.list_available_tools()
                logger_for_agent_logs.info(f"Available MCP Tools: {len(available_tools)}")
                
                if not args.minimize_stdout_logs:
                    console.print(f"[bold]Available MCP Tools:[/bold]")
                    for tool in available_tools[:10]:
                        console.print(f"- {tool['name']}: {tool.get('description', 'No description')}")
                    if len(available_tools) > 10:
                        console.print(f"... and {len(available_tools) - 10} more")
                        
            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg or "unauthorized" in error_msg.lower():
                    logger_for_agent_logs.error(f"MCP authentication failed: {e}")
                    if not args.minimize_stdout_logs:
                        console.print("[red]MCP authentication failed. Check OAuth token.[/red]")
                else:
                    logger_for_agent_logs.warning(f"Failed to initialize MCP wrapper: {e}")
                available_tools = []
    elif args.enable_mcp and not MCP_AVAILABLE:
        logger_for_agent_logs.warning("MCP tools requested but not available. Check imports.")
        if not args.minimize_stdout_logs:
            console.print("[yellow]Warning: MCP tools not available. Check installation.[/yellow]")

    # Print welcome message
    if not args.minimize_stdout_logs:
        # Customize welcome message based on LLM client
        llm_info = f"LLM: {args.llm_client}"
        if args.llm_client == "ollama":
            llm_info = f"LLM: Ollama ({args.ollama_model}) at {args.ollama_base_url}"
        elif args.llm_client == "chutes":
            llm_info = f"LLM: Chutes ({args.chutes_model})"
        
        # Add mode info
        mode_info = "Standard mode"
        if args.enable_banking:
            mode_info = f"Banking mode (Role: {args.user_role})"
        elif args.enable_mcp:
            mode_info = "MCP-enabled mode"
        
        # Add OAuth info
        oauth_info = ""
        if oauth_enabled and OAUTH_AVAILABLE:
            oauth_info = f"OAuth: Enabled\n"
            if token_manager:
                oauth_info += "Token Manager: Active (auto-refresh)\n"
        
        # Add proposal system info
        proposal_info = ""
        if args.enable_agent_proposals:
            proposal_info = "Agent Proposals: Enabled\n"

        console.print(
            Panel(
                "[bold]TCS BANCS -Agent CLI[/bold]\n\n"
                + f"Session ID: {session_id}\n"
                + f"Workspace: {workspace_path}\n"
                + f"{llm_info}\n"
                + f"Mode: {mode_info}\n"
                + oauth_info
                + proposal_info
                + f"Log file: {os.path.abspath(args.logs_path)}\n\n"
                + "Type your instructions to the agent. Press Ctrl+C to exit.\n"
                + "Type 'exit' or 'quit' to end the session."
                + ("\nType '/help' for agent proposal commands." if args.enable_agent_proposals else ""),
                title="[bold blue]TCS BANCS-Agent CLI[/bold blue]",
                border_style="blue",
                padding=(1, 2),
            )
        )
    else:
        logger_for_agent_logs.info(
            f"II-Agent CLI started with session {session_id}. Waiting for user input."
        )

    # Initialize LLM client
    client_kwargs = {
        "model_name": args.model_name,
    }
    
    logger_for_agent_logs.info(f"Initializing LLM client: {args.llm_client}")
    
    if args.llm_client == "anthropic-direct":
        client_kwargs["use_caching"] = False
        client_kwargs["project_id"] = args.project_id
        client_kwargs["region"] = args.region
    elif args.llm_client == "openai-direct":
        client_kwargs["azure_model"] = args.azure_model
        client_kwargs["cot_model"] = args.cot_model
    elif args.llm_client == "ollama":
        # Add Ollama-specific configuration
        client_kwargs["base_url"] = args.ollama_base_url
        client_kwargs["model_name"] = args.ollama_model
        client_kwargs["timeout"] = 300
    elif args.llm_client == "chutes":
        client_kwargs["model_name"] = args.chutes_model
        client_kwargs["no_fallback"] = args.chutes_no_fallback
        client_kwargs["use_native_tool_calling"] = args.chutes_native_tools
        
        logger_for_agent_logs.info(f"Chutes configuration: {client_kwargs}")
    
    # Create LLM client
    try:
        client = get_client(
            args.llm_client,
            **client_kwargs
        )
        logger_for_agent_logs.info(f"Successfully created LLM client: {type(client).__name__}")
    except Exception as e:
        logger_for_agent_logs.error(f"Failed to create LLM client: {e}", exc_info=True)
        console.print(f"[red]Failed to create LLM client: {e}[/red]")
        return

    # Initialize workspace manager with the session-specific workspace
    workspace_manager = WorkspaceManager(
        root=workspace_path, container_workspace=args.use_container_workspace
    )

    # Initialize token counter
    token_counter = TokenCounter()

    # Create context manager based on LLM client type
    if args.llm_client == "ollama":
        # Determine context strategy based on model
        ollama_context_strategy = os.getenv("OLLAMA_CONTEXT_STRATEGY", "auto")
        
        if ollama_context_strategy == "auto":
            # Use adaptive manager that chooses based on model
            context_manager = OllamaAdaptiveContextManager(
                token_counter=token_counter,
                logger=logger_for_agent_logs,
                token_budget=4096,
                model_name=args.ollama_model,
                client=client
            )
            logger_for_agent_logs.info(f"Using adaptive context strategy for Ollama model: {args.ollama_model}")
        else:
            # Use specific strategy
            context_manager = OllamaContextManager.create(
                token_counter=token_counter,
                logger=logger_for_agent_logs,
                token_budget=4096,
                strategy=ollama_context_strategy,
                client=client if ollama_context_strategy == "summarizing" else None
            )
            logger_for_agent_logs.info(f"Using {ollama_context_strategy} context strategy for Ollama")
    else:
        # Use full summarizing context manager for cloud LLMs
        context_manager = LLMSummarizingContextManager(
            client=client,
            token_counter=token_counter,
            logger=logger_for_agent_logs,
            token_budget=TOKEN_BUDGET
        )
        logger_for_agent_logs.info("Using LLM summarizing context manager")

    queue = asyncio.Queue()

    # Configure tools based on mode and LLM client
    if args.mcp_tools_only and args.enable_mcp:
        # MCP-only mode with essential communication tools
        tools = [
            MessageTool(),
            ReturnControlToUserTool() if True else CompleteTool(),  # TODO: detect interactive mode
        ]
        logger_for_agent_logs.info(f"MCP-only mode: Loaded {len(tools)} essential communication tools")
        system_tools_count = len(tools)
    else:
        # Standard mode - load system tools
        if args.llm_client == "ollama":
            # Reduced tools for Ollama
            tool_args = {
                "deep_research": False,
                "pdf": False,
                "media_generation": False,
                "audio_generation": False,
                "browser": False,
                "memory_tool": "none",
            }
        else:
            tool_args = {
                "deep_research": False,
                "pdf": True,
                "media_generation": False,
                "audio_generation": False,
                "browser": True,
                "memory_tool": args.memory_tool,
            }
        
        # Add banking-specific flags if needed
        if args.enable_banking and mcp_wrappers:
            tool_args["sequential_thinking"] = True
        
        logger_for_agent_logs.info(f"Tool configuration: {tool_args}")
        
        tools = get_system_tools(
            client=client,
            workspace_manager=workspace_manager,
            message_queue=queue,
            container_id=args.docker_container_id,
            ask_user_permission=args.needs_permission,
            tool_args=tool_args,
        )
        
        system_tools_count = len(tools)
        logger_for_agent_logs.info(f"Loaded {system_tools_count} system tools")
    
    # Select appropriate system prompt
    if args.enable_banking:
        # Use role-specific banking prompt
        llm_type = "ollama" if args.llm_client == "ollama" else "cloud"
        system_prompt = get_banking_prompt(args.user_role, llm_type)
        logger_for_agent_logs.info(f"Using banking prompt for role: {args.user_role} ({llm_type})")
    else:
        # Standard mode - use original prompts
        if "sequential_thinking" in tool_args and tool_args["sequential_thinking"]:
            system_prompt = SYSTEM_PROMPT_WITH_SEQ_THINKING
        else:
            system_prompt = SYSTEM_PROMPT
        logger_for_agent_logs.info("Using standard system prompt")
    
    # Add MCP tools if enabled
    if args.enable_mcp and mcp_wrappers and MCP_AVAILABLE:
        logger_for_agent_logs.info("Adding MCP tools to agent...")
        
        # Debug: Check registry state before
        logger_for_agent_logs.debug(f"Registry has {len(mcp_tool_registry._tools)} tools before adapter creation")
        
        if args.enable_banking:
            # Use banking mode to get only the 3 core tools
            mcp_adapters = create_mcp_tool_adapters(
                mcp_wrapper=next(iter(mcp_wrappers.values())),
                banking_mode=True
            )
            logger_for_agent_logs.info(f"Created {len(mcp_adapters)} banking tool adapters")
            for adapter in mcp_adapters:
                logger_for_agent_logs.info(f"  - Banking tool: {adapter.name}")
        else:
            # Standard mode - get all tools from registry
            mcp_adapters = create_mcp_tool_adapters(
                registry=mcp_tool_registry,
                include_deprecated=False,
                categories=None if args.user_role == "admin" else ["accounts", "transfers"],
                tags=None
            )
        
        tools.extend(mcp_adapters)
        
        logger_for_agent_logs.info(
            f"Total tools available: {len(tools)} "
            f"(System: {system_tools_count}, MCP: {len(mcp_adapters)})"
        )

    # Create agent based on mode - SIMPLIFIED VERSION
    if args.enable_banking and mcp_wrappers:
        # Banking mode: Use TCSBancsSpecialistAgent (extends AnthropicFC)
        from src.ii_agent.agents.tcs_bancs_specialist_agent import TCSBancsSpecialistAgent
        
        logger_for_agent_logs.info("Creating TCS BaNCS Specialist Agent...")
        
        agent = await TCSBancsSpecialistAgent.create(
            client=client,
            tools=tools,
            workspace_manager=workspace_manager,
            message_queue=queue,
            logger_for_agent_logs=logger_for_agent_logs,
            context_manager=context_manager,
            user_role=args.user_role,
            mcp_wrapper=next(iter(mcp_wrappers.values())) if mcp_wrappers else None,
            use_mcp_prompts=True,  # Enable MCP prompts
            max_output_tokens_per_turn=MAX_OUTPUT_TOKENS_PER_TURN,
            max_turns=MAX_TURNS,
            session_id=session_id,
            interactive_mode=True
        )
        
        # Log MCP prompt status
        if agent.is_mcp_enabled():
            available_prompts = await agent.get_available_prompts()
            logger_for_agent_logs.info(f"MCP prompts loaded: {available_prompts}")
        else:
            logger_for_agent_logs.info("Using local prompts (MCP prompts not available)")

        
    else:
        # Standard mode: Use AnthropicFC
        logger_for_agent_logs.info("Creating standard AnthropicFC agent...")
        
        agent = AnthropicFC(
            system_prompt=system_prompt,
            client=client,
            workspace_manager=workspace_manager,
            tools=tools,
            message_queue=queue,
            logger_for_agent_logs=logger_for_agent_logs,
            context_manager=context_manager,
            max_output_tokens_per_turn=MAX_OUTPUT_TOKENS_PER_TURN,
            max_turns=MAX_TURNS,
            session_id=session_id,
            interactive_mode=True
        )
        
        logger_for_agent_logs.info("AnthropicFC agent created")
        # Initialize proposal system (after line 668 where agent is created)
        

    # Debug: Verify tool registration
    logger_for_agent_logs.info("=== Agent Tool Verification ===")
    agent_tools = agent.tool_manager.get_tools() if hasattr(agent, 'tool_manager') else []
    logger_for_agent_logs.info(f"Agent has {len(agent_tools)} tools total:")
    
    # Group by type
    tool_types = {}
    for tool in agent_tools:
        tool_type = type(tool).__name__
        if tool_type not in tool_types:
            tool_types[tool_type] = []
        tool_types[tool_type].append(tool.name)
    
    for tool_type, tool_names in tool_types.items():
        logger_for_agent_logs.info(f"  {tool_type}: {tool_names}")
    

    # Specifically check for banking tools
    if args.enable_banking:
        banking_tool_names = ['list_banking_apis', 'get_api_structure', 'invoke_banking_api']
        for tool_name in banking_tool_names:
            try:
                tool = agent.tool_manager.get_tool(tool_name)
                logger_for_agent_logs.info(f"  ✓ {tool_name}: Available")
            except ValueError:
                logger_for_agent_logs.error(f"  ✗ {tool_name}: NOT FOUND")

        
    if args.enable_agent_proposals and args.llm_client == "chutes":
        try:
            # Use the existing Chutes client for proposal system
            proposal_system = await initialize_proposal_system(
                client,  # Use the already initialized Chutes client
                next(iter(mcp_wrappers.values())) if mcp_wrappers else None,
                args, 
                logger_for_agent_logs
            )
            
            if proposal_system:
                console.print(
                    f"[green]✓[/green] Agent Proposal System enabled "
                    f"({len(proposal_system.agent_registry)} saved agents)"
                )
        except Exception as e:
            logger_for_agent_logs.error(f"Failed to initialize proposal system: {e}")
            console.print(f"[yellow]⚠ Agent proposals disabled: {e}[/yellow]")

    # Create background task for message processing
    message_task = agent.start_message_processing()

    # Main interaction loop - UNIFIED FOR BOTH AGENT TYPES
    # Main interaction loop - Enhanced with proposal support
    try:
        # Quick complexity checker for real-time response
        quick_check = QuickComplexityCheck() if proposal_system else None
        
        # A2A client for agent communication (if enabled)
        a2a_client = None
        if proposal_system and os.getenv("ENABLE_A2A", "false").lower() == "true":
            a2a_client = A2AClient(
                agent_id="main_cli_agent",
                agent_name="CLI Interface Agent",
                base_url=os.getenv("A2A_BASE_URL", "http://localhost:8001")
            )
        
        # Only show tip if using Chutes without proposals
        if args.llm_client == "chutes" and not args.enable_agent_proposals:
            console.print(
                "[yellow]Tip: You're using Chutes. Enable agent proposals with "
                "--enable-agent-proposals for intelligent agent creation![/yellow]"
            )
        
        while True:
            # Get user input
            if args.prompt is None:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: Prompt.ask("\n[bold cyan]You[/bold cyan]", default="")
                )
            else:
                user_input = args.prompt
                args.prompt = None  # Only use prompt once

            if not user_input:
                continue

            logger_for_agent_logs.info(f"User input: {user_input}")

            if user_input.lower() in ["exit", "quit", "bye"]:
                console.print("[yellow]Goodbye![/yellow]")
                logger_for_agent_logs.info("User requested exit")
                break

            # Check for special commands
            if user_input.startswith("/"):
                result = await handle_special_command(user_input, proposal_system, console)
                if result is None:
                    continue  # Command was handled
                # Otherwise, treat as regular input

            # Quick complexity check for real-time decision
            if proposal_system and quick_check and not quick_check.is_simple(user_input):
                # Perform full analysis
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True
                ) as progress:
                    task_id = progress.add_task("Analyzing request...", total=None)
                    
                    proposal = await proposal_system.analyze_and_propose(user_input, {
                        "user_role": args.user_role,
                        "session_id": str(session_id)
                    })
                    
                    progress.update(task_id, completed=True)
                
                if proposal:
                    # Display proposal
                    agent_id = await display_and_confirm_proposal(
                        proposal, proposal_system, console
                    )
                    
                    if agent_id:
                        # Execute with new agent
                        console.print("\n[bold green]Specialized Agent:[/bold green]")
                        
                        with Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            console=console
                        ) as progress:
                            task_id = progress.add_task("Executing with specialized agent...", total=None)
                            
                            result = await proposal_system.execute_with_agent(
                                agent_id, user_input
                            )
                            
                            progress.update(task_id, completed=True)
                        
                        # Display result
                        if "error" in result:
                            console.print(f"[red]Error: {result['error']}[/red]")
                        else:
                            output = result.get("output", result)
                            if isinstance(output, dict):
                                # Format dict output nicely
                                import json
                                console.print(json.dumps(output, indent=2))
                            else:
                                console.print(str(output))
                        
                        logger_for_agent_logs.info(f"Specialized agent response: {result}")
                        logger_for_agent_logs.info("\n" + "-" * 40 + "\n")
                        continue

            # Use existing agent for simple tasks or if proposal rejected
            if not args.minimize_stdout_logs:
                console.print("\n[bold green]Agent:[/bold green]")

            # Add to message queue for existing agent
            agent.message_queue.put_nowait(
                RealtimeEvent(type=EventType.USER_MESSAGE, content={"text": user_input})
            )

            logger_for_agent_logs.info("Agent is processing...")
            
            try:
                # Use existing agent
                result = agent.run_agent(user_input, resume=True)
                logger_for_agent_logs.info(f"Agent response: {result}")
                
                # Display result if not minimizing output
                if not args.minimize_stdout_logs and result:
                    console.print(result)
                    
            except KeyboardInterrupt:
                agent.cancel()
                logger_for_agent_logs.info("Agent cancelled")
            except Exception as e:
                logger_for_agent_logs.error(f"Error during agent execution: {str(e)}", exc_info=True)
                console.print(f"[red]Error: {str(e)}[/red]")

            logger_for_agent_logs.info("\n" + "-" * 40 + "\n")

    except KeyboardInterrupt:
        console.print("\n[bold]Session interrupted. Exiting...[/bold]")
        logger_for_agent_logs.info("Session interrupted by user")
    except Exception as e:
        logger_for_agent_logs.error(f"Unexpected error in main loop: {e}", exc_info=True)
        console.print(f"[red]Unexpected error: {e}[/red]")
    finally:
        # Cleanup tasks
        message_task.cancel()
        
        # Close A2A client if initialized
        if 'a2a_client' in locals() and a2a_client:
            try:
                await a2a_client.close()
                logger_for_agent_logs.info("Closed A2A client")
            except Exception as e:
                logger_for_agent_logs.error(f"Error closing A2A client: {e}")
        
        # Close all MCP connections if any
        if mcp_wrappers:
            for service_name, wrapper in mcp_wrappers.items():
                try:
                    await wrapper.close()
                    logger_for_agent_logs.info(f"Closed {service_name} MCP connection")
                except Exception as e:
                    logger_for_agent_logs.error(f"Error closing {service_name}: {e}")
        
        logger_for_agent_logs.info("Cleanup completed")

    console.print("[bold]Goodbye![/bold]")
    logger_for_agent_logs.info("TCS BANCS Agent CLI ended")



if __name__ == "__main__":
    asyncio.run(async_main())