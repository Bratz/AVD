import logging
import yaml
import os
import httpx
from tavily import TavilyClient
from litellm import completion
from typing import Dict, Any
from importlib import import_module
import uuid

class ToolManager:
    """Manages execution of tools defined in commands.yaml."""

    def __init__(self, config_path: str = "app/tools/config/II_commands.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.commands = self.load_commands()
        self.api_key = os.getenv("TCS_BANCS_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")

    def load_commands(self) -> Dict[str, Any]:
        """Load command definitions from YAML."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return {cmd['name']: cmd for cmd in config.get('commands', [])}

    async def execute(self, command_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute a command by invoking its handler."""
        self.logger.debug(f"Executing command: {command_name} with parameters: {parameters}")
        if command_name not in self.commands:
            self.logger.error(f"Command not found: {command_name}")
            raise ValueError(f"Command not found: {command_name}")

        command = self.commands[command_name]
        handler_path = command.get('handler')
        if not handler_path:
            self.logger.error(f"No handler defined for command: {command_name}")
            raise ValueError(f"No handler defined for command: {command_name}")

        module_name, class_name = handler_path.rsplit(':', 1)
        module = import_module(module_name)
        handler_class = getattr(module, class_name)

        handler = handler_class(session_id=str(uuid.uuid4()))
        if hasattr(handler, 'initialize'):
            await handler.initialize()

        result = await handler.execute(parameters)
        self.logger.info(f"Command {command_name} executed successfully")
        return result