import os
import asyncio
import logging
import sys
from copy import deepcopy
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from src.ii_agent.llm.base import LLMClient
from src.ii_agent.llm.context_manager.llm_summarizing import LLMSummarizingContextManager
from src.ii_agent.llm.token_counter import TokenCounter
from src.ii_agent.tools.advanced_tools.image_search_tool import ImageSearchTool
from src.ii_agent.tools.base import LLMTool
from src.ii_agent.llm.message_history import ToolCallParameters
from src.ii_agent.tools.memory.compactify_memory import CompactifyMemoryTool
from src.ii_agent.tools.memory.simple_memory import SimpleMemoryTool
from src.ii_agent.tools.slide_deck_tool import SlideDeckInitTool, SlideDeckCompleteTool
from src.ii_agent.tools.web_search_tool import WebSearchTool
from src.ii_agent.tools.visit_webpage_tool import VisitWebpageTool
from src.ii_agent.tools.str_replace_tool_relative import StrReplaceEditorTool
from src.ii_agent.tools.static_deploy_tool import StaticDeployTool
from src.ii_agent.tools.sequential_thinking_tool import SequentialThinkingTool
from src.ii_agent.tools.message_tool import MessageTool
from src.ii_agent.tools.complete_tool import CompleteTool, ReturnControlToUserTool
from src.ii_agent.tools.bash_tool import create_bash_tool, create_docker_bash_tool
from src.ii_agent.browser.browser import Browser
from src.ii_agent.utils import WorkspaceManager
from src.ii_agent.llm.message_history import MessageHistory
from src.ii_agent.tools.browser_tools import (
    BrowserNavigationTool,
    BrowserRestartTool,
    BrowserScrollDownTool,
    BrowserScrollUpTool,
    BrowserViewTool,
    BrowserWaitTool,
    BrowserSwitchTabTool,
    BrowserOpenNewTabTool,
    BrowserClickTool,
    BrowserEnterTextTool,
    BrowserPressKeyTool,
    BrowserGetSelectOptionsTool,
    BrowserSelectDropdownOptionTool,
)
from src.ii_agent.tools.visualizer import DisplayImageTool
from src.ii_agent.tools.advanced_tools.audio_tool import (
    AudioTranscribeTool,
    AudioGenerateTool,
)
from src.ii_agent.tools.advanced_tools.video_gen_tool import VideoGenerateFromTextTool
from src.ii_agent.tools.advanced_tools.image_gen_tool import ImageGenerateTool
from src.ii_agent.tools.advanced_tools.pdf_tool import PdfTextExtractTool
# from src.ii_agent.tools.deep_research_tool import DeepResearchTool
from src.ii_agent.tools.list_html_links_tool import ListHtmlLinksTool
from datetime import datetime
from src.ii_agent.core.event import EventType, RealtimeEvent
# from src.ii_agent.core.event_stream import EventStream  # Will be created
from src.ii_agent.tools.chart_tool import ChartTool

if TYPE_CHECKING:
    from src.ii_agent.core.event_stream import EventStream


def setup_utf8_logging():
    """Set up UTF-8 encoding for logging to handle Unicode characters"""
    # For Windows console
    if sys.platform == 'win32':
        import locale
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8')
        if sys.stderr.encoding != 'utf-8':
            sys.stderr.reconfigure(encoding='utf-8')


def get_system_tools(
    client: LLMClient,
    workspace_manager: WorkspaceManager,
    message_queue: asyncio.Queue,
    container_id: Optional[str] = None,
    ask_user_permission: bool = False,
    tool_args: Dict[str, Any] = None,
) -> list[LLMTool]:
    """
    Retrieves a list of all system tools.

    Returns:
        list[LLMTool]: A list of all system tools.
    """
    # if container_id is not None:
    #     bash_tool = create_docker_bash_tool(
    #         container=container_id, ask_user_permission=ask_user_permission
    #     )
    # else:
    #     bash_tool = create_bash_tool(
    #         ask_user_permission=ask_user_permission, cwd=workspace_manager.root
    #     )

    logger = logging.getLogger("presentation_context_manager")
    context_manager = LLMSummarizingContextManager(
        client=client,
        token_counter=TokenCounter(),
        logger=logger,
        token_budget=120_000,
    )

    tools = [
        MessageTool(),
        SequentialThinkingTool(),
        ChartTool(),
    #     WebSearchTool(),
    #     VisitWebpageTool(),
    #     StaticDeployTool(workspace_manager=workspace_manager),
    #     StrReplaceEditorTool(
    #         workspace_manager=workspace_manager, message_queue=message_queue
    #     ),
    #     # bash_tool,
    #     ListHtmlLinksTool(workspace_manager=workspace_manager),
    #     SlideDeckInitTool(
    #         workspace_manager=workspace_manager,
    #     ),
    #     SlideDeckCompleteTool(
    #         workspace_manager=workspace_manager,
    #     ),
    #     DisplayImageTool(workspace_manager=workspace_manager),
    ]
    # image_search_tool = ImageSearchTool()
    # if image_search_tool.is_available():
    #     tools.append(image_search_tool)

    # Conditionally add tools based on tool_args
    if tool_args:
        # if tool_args.get("sequential_thinking", False):
        #     tools.append(SequentialThinkingTool())
        # if tool_args.get("deep_research", False):
            # tools.append(DeepResearchTool())
        # if tool_args.get("pdf", False):
        #     tools.append(PdfTextExtractTool(workspace_manager=workspace_manager))
        # if tool_args.get("media_generation", False) and (
        #     os.environ.get("GOOGLE_CLOUD_PROJECT")
        #     and os.environ.get("GOOGLE_CLOUD_REGION")
        # ):
        #     tools.append(ImageGenerateTool(workspace_manager=workspace_manager))
        #     if tool_args.get("video_generation", False):
        #         tools.append(VideoGenerateFromTextTool(workspace_manager=workspace_manager))
        # if tool_args.get("audio_generation", False) and (
        #     os.environ.get("OPEN_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT")
        # ):
        #     tools.extend(
        #         [
        #             AudioTranscribeTool(workspace_manager=workspace_manager),
        #             AudioGenerateTool(workspace_manager=workspace_manager),
        #         ]
        #     )
            
        # # Browser tools
        # if tool_args.get("browser", False):
        #     browser = Browser()
        #     tools.extend(
        #         [
        #             BrowserNavigationTool(browser=browser),
        #             BrowserRestartTool(browser=browser),
        #             BrowserScrollDownTool(browser=browser),
        #             BrowserScrollUpTool(browser=browser),
        #             BrowserViewTool(browser=browser),
        #             BrowserWaitTool(browser=browser),
        #             BrowserSwitchTabTool(browser=browser),
        #             BrowserOpenNewTabTool(browser=browser),
        #             BrowserClickTool(browser=browser),
        #             BrowserEnterTextTool(browser=browser),
        #             BrowserPressKeyTool(browser=browser),
        #             BrowserGetSelectOptionsTool(browser=browser),
        #             BrowserSelectDropdownOptionTool(browser=browser),
        #         ]
        #     )

        memory_tool = tool_args.get("memory_tool")
        if memory_tool == "compactify-memory":
            tools.append(CompactifyMemoryTool(context_manager=context_manager))
        elif memory_tool == "none":
            pass
        elif memory_tool == "simple":
            tools.append(SimpleMemoryTool())

        if tool_args and tool_args.get("banking", False):
        # Import here to avoid circular dependencies
            from src.ii_agent.tools.banking_core_tools import get_banking_core_tools
        
              # Assume mcp_wrapper is passed in tool_args
            mcp_wrapper = tool_args.get("mcp_wrapper")
            if mcp_wrapper:
                banking_tools = get_banking_core_tools(mcp_wrapper)
                tools.extend(banking_tools)
        return tools


class AgentToolManager:
    """
    Manages the creation and execution of tools for the agent.

    This class is responsible for:
    - Initializing and managing all available tools
    - Providing access to tools by name
    - Executing tools with appropriate inputs
    - Logging tool execution details

    Tools include bash commands, browser interactions, file operations,
    search capabilities, and task completion functionality.
    """

    def __init__(self, tools: List[LLMTool], logger_for_agent_logs: logging.Logger, interactive_mode: bool = True,event_stream: Optional['EventStream'] = None):
        self.logger_for_agent_logs = logger_for_agent_logs
        self.complete_tool = ReturnControlToUserTool() if interactive_mode else CompleteTool()
        self.tools = tools
        self.event_stream = event_stream
        
        # Set up UTF-8 logging support
        setup_utf8_logging()

    def get_tool(self, tool_name: str) -> LLMTool:
        """
        Retrieves a tool by its name.

        Args:
            tool_name (str): The name of the tool to retrieve.

        Returns:
            LLMTool: The tool object corresponding to the given name.

        Raises:
            ValueError: If the tool with the specified name is not found.
        """
        try:
            tool: LLMTool = next(t for t in self.get_tools() if t.name == tool_name)
            return tool
        except StopIteration:
            raise ValueError(f"Tool with name {tool_name} not found")

    def _safe_log_message(self, message: str):
        """
        Safely log a message, handling potential Unicode encoding issues.
        
        Args:
            message (str): The message to log
        """
        try:
            self.logger_for_agent_logs.info(message)
        except UnicodeEncodeError:
            # Fallback: Remove problematic Unicode characters
            safe_message = message.encode('ascii', 'replace').decode('ascii')
            self.logger_for_agent_logs.info(safe_message)

    def run_tool(self, tool_params: ToolCallParameters, history: MessageHistory):
        """
        Executes a llm tool.

        Args:
            tool (LLMTool): The tool to execute.
            history (MessageHistory): The history of the conversation.
        Returns:
            ToolResult: The result of the tool execution.
        """
        llm_tool = self.get_tool(tool_params.tool_name)
        tool_name = tool_params.tool_name
        tool_input = tool_params.tool_input
                    
        self._safe_log_message(f"Running tool: {tool_name}")
        self._safe_log_message(f"Tool input: {tool_input}")
        result = llm_tool.run(tool_input, history)

        tool_input_str = "\n".join([f" - {k}: {v}" for k, v in tool_input.items()])

        log_message = f"Calling tool {tool_name} with input:\n{tool_input_str}"
        if isinstance(result, str):
            log_message += f"\nTool output: \n{result}\n\n"
        else:
            result_to_log = deepcopy(result)
            for i in range(len(result_to_log)):
                if result_to_log[i].get("type") == "image":
                    result_to_log[i]["source"]["data"] = "[REDACTED]"
            log_message += f"\nTool output: \n{result_to_log}\n\n"

        self._safe_log_message(log_message)

        # Handle both ToolResult objects and tuples
        if isinstance(result, tuple):
            tool_result, _ = result
        else:
            tool_result = result

        return tool_result
    
    async def run_tool_async(self, tool_params: ToolCallParameters, history: MessageHistory):
        """
        Executes a tool asynchronously with event emission.
        This is the II-Agent pattern compliant version.
        """
        tool_name = tool_params.tool_name
        tool_input = tool_params.tool_input
        
        # Emit tool call event
        if self.event_stream:
            await self.event_stream.emit_event(
                EventType.TOOL_CALL,
                {
                    "tool_name": tool_name,
                    "parameters": tool_input,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        try:
            # Get and execute tool
            llm_tool = self.get_tool(tool_name)
            self._safe_log_message(f"Running tool: {tool_name}")
            self._safe_log_message(f"Tool input: {tool_input}")
            
            # Execute tool
            result = llm_tool.run(tool_input, history)
            
            # Log results
            tool_input_str = "\n".join([f" - {k}: {v}" for k, v in tool_input.items()])
            log_message = f"Calling tool {tool_name} with input:\n{tool_input_str}"
            
            if isinstance(result, str):
                log_message += f"\nTool output: \n{result}\n\n"
                result_preview = result[:500] if len(result) > 500 else result
            else:
                result_to_log = deepcopy(result)
                if isinstance(result_to_log, list):
                    for i in range(len(result_to_log)):
                        if isinstance(result_to_log[i], dict) and result_to_log[i].get("type") == "image":
                            result_to_log[i]["source"]["data"] = "[REDACTED]"
                log_message += f"\nTool output: \n{result_to_log}\n\n"
                result_preview = str(result_to_log)[:500]
            
            self._safe_log_message(log_message)
            
            # Emit tool result event
            if self.event_stream:
                await self.event_stream.emit_event(
                    EventType.TOOL_RESULT,
                    {
                        "tool_name": tool_name,
                        "result": result_preview,  # Truncated for events
                        "success": True,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            # Handle both ToolResult objects and tuples
            if isinstance(result, tuple):
                tool_result, _ = result
            else:
                tool_result = result
                
            return tool_result
            
        except Exception as e:
            # Emit error event
            if self.event_stream:
                await self.event_stream.emit_event(
                    EventType.TOOL_RESULT,
                    {
                        "tool_name": tool_name,
                        "error": str(e),
                        "success": False,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            self.logger_for_agent_logs.error(f"Tool execution failed: {e}")
            raise
    def should_stop(self):
        """
        Checks if the agent should stop based on the completion tool.

        Returns:
            bool: True if the agent should stop, False otherwise.
        """
        return self.complete_tool.should_stop

    def get_final_answer(self):
        """
        Retrieves the final answer from the completion tool.

        Returns:
            str: The final answer from the completion tool.
        """
        return self.complete_tool.answer

    def reset(self):
        """
        Resets the completion tool.
        """
        self.complete_tool.reset()

    def get_tools(self) -> list[LLMTool]:
        """
        Retrieves a list of all available tools.

        Returns:
            list[LLMTool]: A list of all available tools.
        """
        return self.tools + [self.complete_tool]