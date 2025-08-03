import os
import json
import logging
import random
import httpx
import uuid
import time
import re
import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple, Union, AsyncGenerator, Callable


from src.ii_agent.llm.base import (
    LLMClient,
    LLMMessages,
    ToolParam,
    TextPrompt,
    ToolCall,
    TextResult,
    AssistantContentBlock,
    ToolFormattedResult,
)

class ModelType(Enum):
    INTENT = "intent"
    RESPONSE = "response" 
    EMBEDDING = "embedding"
    REASONING = "reasoning"

@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    timeout: int = 300
    models: Dict[ModelType, str] = None

    def __post_init__(self):
        if self.models is None:
            self.models = {
                ModelType.INTENT: "llama3.1:8b",
                ModelType.RESPONSE: "llama3.1:8b",
                ModelType.EMBEDDING: "nomic-embed-text:latest",
                ModelType.REASONING: "llama3.1:8b"
            }

class OllamaDirectClient(LLMClient):
    """Ollama Direct Client with native tool calling support (latest Ollama version)"""

    def __init__(
        self, 
        model_type: ModelType = ModelType.RESPONSE,
        **kwargs
    ):
        """Initialize Ollama client with flexible model configuration"""
        self.logger = logging.getLogger(__name__)
        
        # Configuration management
        config = kwargs.get('config', OllamaConfig())
        
        # CRITICAL FIX: Always respect the model_name from kwargs if provided
        # This ensures CLI arguments take precedence
        if 'model_name' in kwargs and kwargs['model_name']:
            self.model_name = kwargs['model_name']
            self.logger.info(f"Using model from kwargs: {self.model_name}")
        else:
            # Only use model_type mapping if no explicit model_name provided
            self.model_name = config.models.get(model_type, config.models[ModelType.RESPONSE])
            self.logger.info(f"Using model from type {model_type}: {self.model_name}")
        
        # Base URL and timeout
        self.base_url = kwargs.get('base_url', config.base_url)
        self.timeout = kwargs.get('timeout', config.timeout)
        
        # HTTP Client setup
        self.client = httpx.Client(
            timeout=httpx.Timeout(self.timeout)
        )
        
        # Async client for streaming
        self.async_client = None
        
        # Feature detection
        self.supports_native_tools = self._check_native_tool_support()
        
        # Logging configuration
        self.logger.info(f"Initialized Ollama Client: Model={self.model_name}, Base URL={self.base_url}")
        self.logger.info(f"Native tool support: {self.supports_native_tools}")
        
        # Validate model availability
        self._validate_model_availability()

    def _check_native_tool_support(self) -> bool:
        """Check if Ollama version supports native tool calling"""
        try:
            # Try to get version info
            response = self.client.get(f"{self.base_url}/api/version")
            if response.status_code == 200:
                version_info = response.json()
                # Check version - native tools added in 0.3.0+
                version = version_info.get('version', '0.0.0')
                major, minor = map(int, version.split('.')[:2])
                return major > 0 or (major == 0 and minor >= 3)
        except:
            pass
        
        # Default to True for latest versions
        return True

    def _validate_model_availability(self):
        """Check if the selected model is available on the Ollama server"""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            data = response.json()
            available_models = [model['name'] for model in data.get('models', [])]
            
            self.logger.info(f"Available Ollama models: {available_models}")
            self.logger.info(f"Requested model: {self.model_name}")
            
            # Check if the model is an embedding model (which can't be used for chat)
            embedding_models = ['nomic-embed-text', 'mxbai-embed-large', 'all-minilm']
            if any(embed in self.model_name for embed in embedding_models):
                self.logger.error(f"Model {self.model_name} is an embedding model and cannot be used for chat")
                # Try to find a chat model
                chat_models = [m for m in available_models if not any(embed in m for embed in embedding_models)]
                if chat_models:
                    self.model_name = chat_models[0]
                    self.logger.warning(f"Switching to chat model: {self.model_name}")
                else:
                    raise RuntimeError(f"Model {self.model_name} is an embedding model and cannot be used for chat. No chat models available.")
            
            if self.model_name not in available_models:
                self.logger.warning(f"Model {self.model_name} not found. Available models: {available_models}")
                # Find a suitable chat model
                chat_models = [m for m in available_models if not any(embed in m for embed in embedding_models)]
                if chat_models:
                    self.model_name = chat_models[0]
                    self.logger.info(f"Falling back to chat model: {self.model_name}")
                else:
                    raise RuntimeError("No chat models available on Ollama server")
                
        except httpx.RequestError as e:
            self.logger.error(f"Cannot connect to Ollama: {e}")
            raise RuntimeError(f"Cannot connect to Ollama server at {self.base_url}. Is Ollama running?")
        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            raise

    def _prepare_messages(self, messages: LLMMessages, system_prompt: Optional[str]) -> List[Dict[str, str]]:
        """Convert messages to Ollama format with proper role detection and length limiting"""
        ollama_messages = []
        
        # Add system prompt if provided
        if system_prompt:
            ollama_messages.append({"role": "system", "content": system_prompt})
            self.logger.info(f"Added system prompt: {system_prompt[:200]}..." if len(system_prompt) > 200 else f"Added system prompt: {system_prompt}")
        
        # Process messages
        self.logger.info(f"Processing {len(messages)} message groups")
        self.logger.debug(f"Raw messages structure: {type(messages)}")
        
        for i, message_group in enumerate(messages):
            self.logger.debug(f"\n--- Processing message group {i} ---")
            self.logger.debug(f"Message group type: {type(message_group)}")
            
            # Handle different message structures
            if isinstance(message_group, list):
                self.logger.debug(f"Message group is a list with {len(message_group)} items")
                for j, msg in enumerate(message_group):
                    self.logger.debug(f"  Item {j}: {type(msg).__name__}")
                    self._add_message_to_ollama(msg, ollama_messages)
            else:
                self.logger.debug(f"Message group is single item: {type(message_group).__name__}")
                self._add_message_to_ollama(message_group, ollama_messages)
        
        # Log the prepared messages before validation
        self.logger.info(f"\n=== Prepared {len(ollama_messages)} messages for Ollama ===")
        for i, msg in enumerate(ollama_messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            self.logger.info(f"Message {i} [{role}]: {content[:100]}..." if len(content) > 100 else f"Message {i} [{role}]: {content}")
        
        # Validate message structure
        self._validate_message_structure(ollama_messages)
        
        # Limit context length to prevent 500 errors
        original_count = len(ollama_messages)
        ollama_messages = self._limit_context_length(ollama_messages)
        if len(ollama_messages) < original_count:
            self.logger.warning(f"Context limited: {original_count} -> {len(ollama_messages)} messages")
        
        return ollama_messages
    
    def _limit_context_length(self, messages: List[Dict[str, str]], max_chars: int = 16000) -> List[Dict[str, str]]:
        """Limit total context length to prevent server errors"""
        # Calculate total length
        total_length = sum(len(msg['content']) for msg in messages)
        
        if total_length <= max_chars:
            return messages
        
        self.logger.warning(f"Context too long ({total_length} chars), truncating to {max_chars}")
        
        # Keep system message and recent messages
        if messages and messages[0]['role'] == 'system':
            system_msg = messages[0]
            other_messages = messages[1:]
        else:
            system_msg = None
            other_messages = messages
        
        # Keep messages from the end until we exceed limit
        kept_messages = []
        current_length = len(system_msg['content']) if system_msg else 0
        
        for msg in reversed(other_messages):
            msg_length = len(msg['content'])
            if current_length + msg_length > max_chars:
                break
            kept_messages.insert(0, msg)
            current_length += msg_length
        
        # Reconstruct messages
        result = []
        if system_msg:
            result.append(system_msg)
        result.extend(kept_messages)
        
        self.logger.info(f"Kept {len(result)} messages out of {len(messages)}")
        return result

    def _add_message_to_ollama(self, msg: Any, ollama_messages: List[Dict[str, str]]):
        """Add a single message to the Ollama messages list"""
        self.logger.debug(f"Processing message type: {type(msg).__name__}")
        
        if isinstance(msg, TextPrompt):
            content = msg.text
            role = "user"
            self.logger.debug(f"TextPrompt -> user: {content[:100]}...")
            ollama_messages.append({"role": role, "content": content})
            
        elif isinstance(msg, TextResult):
            content = msg.text
            role = "assistant"
            self.logger.debug(f"TextResult -> assistant: {content[:100]}...")
            ollama_messages.append({"role": role, "content": content})
            
        elif isinstance(msg, AssistantContentBlock):
            content = getattr(msg, 'text', str(msg))
            role = "assistant"
            self.logger.debug(f"AssistantContentBlock -> assistant: {content[:100]}...")
            ollama_messages.append({"role": role, "content": content})
            
        elif isinstance(msg, ToolCall):
            # For native tool support, we might need to handle this differently
            # For now, format as content
            role = "assistant"
            tool_json = {
                "tool_name": msg.tool_name,
                "tool_input": msg.tool_input
            }
            content = f"I need to use the {msg.tool_name} tool:\n```json\n{json.dumps(tool_json, indent=2)}\n```"
            
            self.logger.debug(f"ToolCall -> assistant: {msg.tool_name}")
            ollama_messages.append({"role": role, "content": content})
            
        elif isinstance(msg, ToolFormattedResult):
            # Tool results should be user messages
            tool_name = getattr(msg, 'tool_name', 'unknown_tool')
            
            # Extract result content
            result_text = None
            for attr in ['output', 'result', 'content', 'tool_output', 'message', 'data']:
                if hasattr(msg, attr):
                    result_text = str(getattr(msg, attr))
                    break
            
            if result_text is None:
                result_text = str(msg)
            
            content = f"Tool '{tool_name}' returned:\n{result_text}"
            role = "user"
            
            self.logger.debug(f"ToolFormattedResult -> user: {tool_name} result")
            ollama_messages.append({"role": role, "content": content})
            
        elif hasattr(msg, 'text'):
            # Generic text message
            if not ollama_messages or ollama_messages[-1]["role"] == "assistant":
                role = "user"
            else:
                role = "assistant"
            content = str(getattr(msg, 'text', msg))
            self.logger.debug(f"Generic text message -> {role}: {content[:100]}...")
            ollama_messages.append({"role": role, "content": content})
            
        else:
            self.logger.warning(f"Unknown message type: {type(msg)}")
            self.logger.warning(f"Message attributes: {dir(msg)}")

    def _validate_message_structure(self, messages: List[Dict[str, str]]):
        """Ensure messages have valid structure for Ollama"""
        if not messages:
            return
        
        prev_role = None
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                raise ValueError(f"Invalid message format at index {i}: {msg}")
            
            # Ensure content is a string
            if not isinstance(msg['content'], str):
                msg['content'] = str(msg['content'])
            
            # Check for empty content
            if not msg['content'].strip():
                msg['content'] = "[empty message]"
            
            # Check for consecutive same roles (skip system messages)
            if prev_role == msg['role'] and msg['role'] != 'system':
                self.logger.warning(f"Consecutive {msg['role']} messages at index {i}")
            
            prev_role = msg['role']

    def generate(
        self,
        messages: LLMMessages,
        max_tokens: int,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[ToolParam] = [],
        tool_choice: dict[str, str] | None = None,
        thinking_tokens: int | None = None,
    ) -> Tuple[list[AssistantContentBlock], dict[str, Any]]:
        """Generate response with native tool calling support"""
        try:
            # Prepare messages
            ollama_messages = self._prepare_messages(messages, system_prompt)
            
            # Prepare request payload
            request_payload = {
                "model": self.model_name,
                "messages": ollama_messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }
            
            # Add tools if provided and supported
            if tools and self.supports_native_tools:
                ollama_tools = self._convert_tools_to_ollama_format(tools)
                request_payload["tools"] = ollama_tools
                
                # Add tool choice if specified
                if tool_choice:
                    request_payload["tool_choice"] = tool_choice
                    
                self.logger.debug(f"Added {len(ollama_tools)} tools to request")
            elif tools:
                # Fallback: Add tool descriptions to system prompt for older versions
                tool_prompt = self._create_tool_prompt(tools)
                if ollama_messages and ollama_messages[0]["role"] == "system":
                    ollama_messages[0]["content"] += "\n\n" + tool_prompt
                else:
                    ollama_messages.insert(0, {"role": "system", "content": tool_prompt})
            
            # Log request for debugging
            self.logger.debug(f"Ollama request payload: {json.dumps(request_payload, indent=2)[:1000]}...")
            
            # Execute request
            response_data = self._execute_with_retry(request_payload)
            
            # Log raw response for debugging
            self.logger.debug(f"Raw Ollama response: {json.dumps(response_data, indent=2)[:1000]}...")
            
            # Process response
            response_blocks = self._process_response(response_data, tools)
            
            # Extract token usage
            usage_metadata = self._extract_token_usage(response_data, ollama_messages)
            
            return response_blocks, usage_metadata
        
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise

    def _convert_tools_to_ollama_format(self, tools: List[ToolParam]) -> List[Dict[str, Any]]:
        """Convert tools to Ollama's native format (latest spec)"""
        ollama_tools = []
        
        for tool in tools:
            # Ensure schema is properly formatted
            parameters = tool.input_schema.copy()
            
            # Ensure it has the required structure
            if "type" not in parameters:
                parameters["type"] = "object"
            if "properties" not in parameters:
                parameters["properties"] = {}
            
            # Ollama's format as per latest documentation
            ollama_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters
                }
            }
            
            ollama_tools.append(ollama_tool)
            self.logger.debug(f"Added tool: {tool.name}")
        
        return ollama_tools

    def _create_tool_prompt(self, tools: List[ToolParam]) -> str:
        """Create tool prompt for models that don't support native tool calling"""
        if not tools:
            return ""
        
        # Create a detailed prompt for tool usage
        tool_descriptions = []
        for tool in tools:
            tool_descriptions.append(f"- {tool.name}: {tool.description}")
        
        return (
            "You have access to the following tools:\n\n" +
            "\n".join(tool_descriptions) +
            "\n\nWhen you need to use a tool, respond with a JSON object in this format:\n" +
            "```json\n{\n  \"tool_name\": \"<tool_name>\",\n  \"tool_input\": {<parameters>}\n}\n```\n" +
            "After using a tool, wait for the result before proceeding."
        )

    def _execute_with_retry(self, payload: Dict[str, Any], max_retries: int = 1) -> Dict[str, Any]:
        """Execute request with detailed error reporting"""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = self.client.post(
                    f"{self.base_url}/api/chat", 
                    json=payload
                )
                self.logger.debug(f"Ollama response status: {response.status_code}")
                
                # Handle different error codes
                if response.status_code == 500:
                    error_detail = response.text
                    self.logger.error(f"Ollama server error (500): {error_detail}")
                    
                    try:
                        error_json = response.json()
                        error_msg = error_json.get('error', 'Unknown server error')
                        
                        if 'out of memory' in error_msg.lower():
                            raise RuntimeError("Ollama ran out of memory. Try a smaller model or reduce context length.")
                        elif 'context length' in error_msg.lower():
                            raise RuntimeError("Context length exceeded. Try reducing the conversation history.")
                        else:
                            raise RuntimeError(f"Ollama server error: {error_msg}")
                    except json.JSONDecodeError:
                        raise RuntimeError(f"Ollama server error (500): {error_detail[:200]}")
                
                elif response.status_code == 400:
                    error_detail = response.text
                    self.logger.error(f"Bad request to Ollama: {error_detail}")
                    
                    try:
                        error_json = response.json()
                        error_msg = error_json.get('error', 'Unknown error')
                        raise ValueError(f"Ollama API error: {error_msg}")
                    except json.JSONDecodeError:
                        raise ValueError(f"Ollama API error (400): {error_detail}")
                
                response.raise_for_status()
                return response.json()
            
            except httpx.RequestError as e:
                last_error = e
                if attempt == max_retries - 1:
                    self.logger.error(f"Request failed after {max_retries} attempts: {e}")
                    raise
                
                wait_time = (2 ** attempt) + random.random()
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            
            except RuntimeError:
                raise
            
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                raise
        
        if last_error:
            raise last_error

    def _process_response(self, response_data: Dict[str, Any], tools: List[ToolParam]) -> List[AssistantContentBlock]:
        """Process Ollama response with native tool call support"""
        response_blocks = []
        
        # Extract message from response
        message = response_data.get('message', {})
        content = message.get('content', '')
        
        # Check for native tool calls (Ollama's latest format)
        tool_calls = message.get('tool_calls', [])
        
        if tool_calls:
            # Native tool calling response
            self.logger.info(f"Found {len(tool_calls)} native tool calls")
            
            # Add any content before tool calls
            if content and content.strip():
                response_blocks.append(TextResult(text=content))
            
            # Process each tool call
            for tool_call in tool_calls:
                # Ollama format: {function: {name, arguments}}
                function = tool_call.get('function', {})
                
                # Parse arguments - might be string or dict
                arguments = function.get('arguments', {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Failed to parse arguments: {arguments}")
                        arguments = {}
                
                response_blocks.append(
                    ToolCall(
                        tool_name=function.get('name', ''),
                        tool_input=arguments,
                        tool_call_id=tool_call.get('id', str(uuid.uuid4()))
                    )
                )
                
                self.logger.debug(f"Processed native tool call: {function.get('name')} with args: {arguments}")
        
        elif content:
            # No native tool calls, check for embedded format (backward compatibility)
            embedded_tool_calls = self._parse_embedded_tool_calls(content)
            
            if embedded_tool_calls:
                self.logger.info(f"Found {len(embedded_tool_calls)} embedded tool calls")
                self._process_mixed_content(content, embedded_tool_calls, response_blocks)
            else:
                # Pure text response
                response_blocks.append(TextResult(text=content))
        
        else:
            self.logger.warning("Empty response from Ollama")
            response_blocks.append(TextResult(text="[No response generated]"))
        
        return response_blocks

    def _parse_embedded_tool_calls(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse tool calls from various formats Ollama might use"""
        tool_calls = []
        
        # First, check if the entire response is a JSON tool call
        response_stripped = response_text.strip()
        if response_stripped.startswith('{') and response_stripped.endswith('}'):
            try:
                parsed = json.loads(response_stripped)
                if 'tool_name' in parsed and 'tool_input' in parsed:
                    self.logger.debug(f"Entire response is a tool call: {parsed}")
                    return [parsed]
            except json.JSONDecodeError:
                pass
        # Try to find JSON objects with various tool call patterns
        json_patterns = [
            # Standard code blocks
            r'```(?:json)?\s*\n?(\{[^`]+\})\s*\n?```',
            # Inline JSON objects
            r'(\{[^{}]*(?:"tool_name"|"name"|"function")[^{}]*\})',
        ]
        
        matches = []
        for pattern in json_patterns:
            found = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if found:
                matches.extend(found)
                break
        
        for match in matches:
            try:
                json_str = match.strip()
                # Clean up common JSON issues
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                
                parsed = json.loads(json_str)
                
                # Normalize to standard format
                tool_call = self._normalize_tool_call(parsed)
                
                if tool_call:
                    tool_calls.append(tool_call)
                    self.logger.debug(f"Parsed embedded tool call: {tool_call}")
                    
            except json.JSONDecodeError as e:
                self.logger.debug(f"Failed to parse JSON: {match[:100]}... Error: {e}")
        
        return tool_calls

    def _normalize_tool_call(self, tool_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize different tool call formats to standard format"""
        # Format 1: {tool_name, tool_input} - Our expected format
        if 'tool_name' in tool_data:
            return {
                'tool_name': tool_data['tool_name'],
                'tool_input': tool_data.get('tool_input', {})
            }
        
        # Format 2: {name, parameters} or {name, arguments} - Ollama's various formats
        elif 'name' in tool_data:
            return {
                'tool_name': tool_data['name'],
                'tool_input': tool_data.get('parameters', tool_data.get('arguments', {}))
            }
        
        # Format 3: {function: {name, arguments}} - OpenAI-style format
        elif 'function' in tool_data:
            func = tool_data['function']
            return {
                'tool_name': func.get('name'),
                'tool_input': func.get('arguments', {})
            }
        
        # Unknown format
        self.logger.warning(f"Unknown tool call format: {tool_data}")
        return None

    def _process_mixed_content(self, content: str, tool_calls: List[Dict[str, Any]], response_blocks: List[AssistantContentBlock]):
        """Process content that contains both text and tool calls"""
        # For simplicity, add text first, then tool calls
        # More sophisticated parsing could split text around tool calls
        
        # Remove JSON blocks from content
        cleaned_content = content
        for tool_call in tool_calls:
            # Try to remove the JSON representation
            json_str = json.dumps(tool_call)
            patterns = [
                f'```json\\s*\\n?{re.escape(json_str)}\\s*\\n?```',
                f'```\\s*\\n?{re.escape(json_str)}\\s*\\n?```',
                re.escape(json_str)
            ]
            
            for pattern in patterns:
                cleaned_content = re.sub(pattern, '', cleaned_content, flags=re.DOTALL)
        
        # Add any remaining text
        cleaned_content = cleaned_content.strip()
        if cleaned_content and not self._is_just_tool_announcement(cleaned_content):
            response_blocks.append(TextResult(text=cleaned_content))
        
        # Add tool calls
        for tool_call in tool_calls:
            response_blocks.append(
                ToolCall(
                    tool_name=tool_call['tool_name'],
                    tool_input=tool_call['tool_input'],
                    tool_call_id=str(uuid.uuid4())
                )
            )

    def _is_just_tool_announcement(self, text: str) -> bool:
        """Check if text is just announcing tool usage"""
        # Common patterns that indicate tool announcements
        tool_phrases = [
            r"i(?:'ll| will) (?:use|call|invoke) (?:the )?.*tool",
            r"let me (?:use|call|invoke)",
            r"using (?:the )?.*tool",
            r"calling (?:the )?.*tool",
            r"here'?s? (?:the )?(?:tool call|response)",
            r"i need to (?:use|call)",
        ]
        
        text_lower = text.lower().strip()
        
        # Check if it's very short
        if len(text_lower) < 100:
            # Check if it matches tool announcement patterns
            for pattern in tool_phrases:
                if re.search(pattern, text_lower):
                    return True
        
        return False

    def _extract_token_usage(self, response_data: Dict[str, Any], input_messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Extract token usage information"""
        # Ollama provides these in the response
        usage = {
            "input_tokens": response_data.get('prompt_eval_count', 0),
            "output_tokens": response_data.get('eval_count', 0),
            "total_tokens": 0
        }
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        
        # If no token counts provided, estimate
        if usage["input_tokens"] == 0:
            input_text = " ".join(msg['content'] for msg in input_messages)
            usage["input_tokens"] = len(input_text) // 4
        
        if usage["output_tokens"] == 0:
            output_text = response_data.get('message', {}).get('content', '')
            usage["output_tokens"] = len(output_text) // 4
        
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        
        return usage

    async def generate_stream(
        self,
        messages: LLMMessages,
        max_tokens: int,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        tools: list[ToolParam] = [],
        tool_choice: dict[str, str] | None = None,
        on_chunk: Optional[Callable] = None,
    ) -> Tuple[list[AssistantContentBlock], dict[str, Any]]:
        """Generate response with streaming support for WebSocket delivery"""
        try:
            # Prepare messages
            ollama_messages = self._prepare_messages(messages, system_prompt)
            
            # Prepare request payload with streaming enabled
            request_payload = {
                "model": self.model_name,
                "messages": ollama_messages,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }
            
            # Add tools if provided and supported
            if tools and self.supports_native_tools:
                ollama_tools = self._convert_tools_to_ollama_format(tools)
                request_payload["tools"] = ollama_tools
                if tool_choice:
                    request_payload["tool_choice"] = tool_choice
            elif tools:
                # Add tool prompt for older versions
                tool_prompt = self._create_tool_prompt(tools)
                if ollama_messages and ollama_messages[0]["role"] == "system":
                    ollama_messages[0]["content"] += "\n\n" + tool_prompt
                else:
                    ollama_messages.insert(0, {"role": "system", "content": tool_prompt})
            
            # Stream the response
            response_blocks = []
            accumulated_content = ""
            tool_calls = []
            usage_metadata = {}
            
            async for chunk in self._stream_request(request_payload):
                if chunk.get('done', False):
                    # Final chunk with usage metadata
                    usage_metadata = self._extract_token_usage_from_stream(chunk, ollama_messages)
                    break
                
                # Process streaming chunk
                message = chunk.get('message', {})
                
                # Handle content chunks
                if 'content' in message:
                    content_chunk = message['content']
                    accumulated_content += content_chunk
                    
                    # Send to WebSocket if callback provided
                    if on_chunk and content_chunk:
                        await on_chunk({
                            'type': 'content',
                            'data': content_chunk
                        })
                
                # Handle tool calls (may come in chunks or complete)
                if 'tool_calls' in message:
                    for tool_call in message['tool_calls']:
                        tool_calls.append(tool_call)
                        
                        # Send tool call to WebSocket
                        if on_chunk:
                            await on_chunk({
                                'type': 'tool_call',
                                'data': tool_call
                            })
            
            # Process final response
            if tool_calls:
                # Add any accumulated content
                if accumulated_content.strip():
                    response_blocks.append(TextResult(text=accumulated_content))
                
                # Add tool calls
                for tool_call in tool_calls:
                    function = tool_call.get('function', {})
                    response_blocks.append(
                        ToolCall(
                            tool_name=function.get('name', ''),
                            tool_input=function.get('arguments', {}),
                            tool_call_id=tool_call.get('id', str(uuid.uuid4()))
                        )
                    )
            else:
                # Check for embedded tool calls in accumulated content
                embedded_tool_calls = self._parse_embedded_tool_calls(accumulated_content)
                
                if embedded_tool_calls:
                    self._process_mixed_content(accumulated_content, embedded_tool_calls, response_blocks)
                else:
                    response_blocks.append(TextResult(text=accumulated_content))
            
            return response_blocks, usage_metadata
            
        except Exception as e:
            self.logger.error(f"Streaming generation failed: {e}")
            raise

    async def _stream_request(self, payload: Dict[str, Any]) -> AsyncGenerator:
        """Execute streaming request with chunk handling"""
        if not self.async_client:
            self.async_client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
        
        try:
            async with self.async_client.stream(
                'POST',
                f"{self.base_url}/api/chat",
                json=payload
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            yield chunk
                        except json.JSONDecodeError:
                            self.logger.warning(f"Failed to parse streaming chunk: {line}")
        except Exception as e:
            self.logger.error(f"Streaming request failed: {e}")
            raise

    def _extract_token_usage_from_stream(self, final_chunk: Dict[str, Any], input_messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Extract token usage from the final streaming chunk"""
        # Similar to non-streaming version but uses final chunk data
        usage = {
            "input_tokens": final_chunk.get('prompt_eval_count', 0),
            "output_tokens": final_chunk.get('eval_count', 0),
            "total_tokens": 0
        }
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        
        # Estimate if not provided
        if usage["input_tokens"] == 0:
            input_text = " ".join(msg['content'] for msg in input_messages)
            usage["input_tokens"] = len(input_text) // 4
        
        return usage

    async def __aenter__(self):
        """Async context manager entry"""
        if not self.async_client:
            self.async_client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.async_client:
            await self.async_client.aclose()
            self.async_client = None

# Context manager classes remain the same
from src.ii_agent.llm.context_manager.simple import SimpleContextManager, OllamaOptimizedContextManager
from src.ii_agent.llm.context_manager.llm_summarizing import LLMSummarizingContextManager
from src.ii_agent.llm.token_counter import TokenCounter

class OllamaContextManager:
    """
    Factory class that returns the appropriate context manager for Ollama
    """
    
    @staticmethod
    def create(
        token_counter: TokenCounter,
        logger: logging.Logger,
        token_budget: int = 4096,
        strategy: str = "simple",  # "simple", "optimized", or "summarizing"
        client=None  # Required only for summarizing strategy
    ):
        """
        Create the appropriate context manager for Ollama
        
        Args:
            token_counter: Token counter instance
            logger: Logger instance  
            token_budget: Maximum token budget
            strategy: Which truncation strategy to use
            client: LLM client (required for summarizing strategy)
            
        Returns:
            Appropriate context manager instance
        """
        if strategy == "summarizing":
            if not client:
                raise ValueError("LLM client required for summarizing strategy")
            
            # Use LLMSummarizingContextManager with Ollama-friendly settings
            return LLMSummarizingContextManager(
                client=client,
                token_counter=token_counter,
                logger=logger,
                token_budget=token_budget,
                max_size=20,  # Smaller for Ollama
                max_event_length=500  # Shorter events for Ollama
            )
            
        elif strategy == "optimized":
            # Use OllamaOptimizedContextManager
            return OllamaOptimizedContextManager(
                token_counter=token_counter,
                logger=logger,
                token_budget=token_budget,
                max_message_length=1000
            )
            
        else:  # "simple" or default
            # Use SimpleContextManager
            return SimpleContextManager(
                token_counter=token_counter,
                logger=logger,
                token_budget=token_budget,
                keep_recent=10
            )


class OllamaAdaptiveContextManager:
    """
    Adaptive context manager that chooses strategy based on model capabilities
    """
    
    def __init__(
        self, 
        token_counter=None, 
        logger=None, 
        token_budget: int = 4096,
        model_name: str = None,
        client=None
    ):
        if not token_counter:
            token_counter = TokenCounter()
        if not logger:
            logger = logging.getLogger(__name__)
            
        self.model_name = model_name or "unknown"
        self.logger = logger
        
        # Choose strategy based on model
        strategy = self._choose_strategy(model_name)
        
        # Create the actual context manager
        self._context_manager = OllamaContextManager.create(
            token_counter=token_counter,
            logger=logger,
            token_budget=token_budget,
            strategy=strategy,
            client=client
        )
        
        self.logger.info(f"Using {strategy} context strategy for model {model_name}")
    
    def _choose_strategy(self, model_name: str) -> str:
        """Choose strategy based on model capabilities"""
        if not model_name:
            return "simple"
            
        model_lower = model_name.lower()
        
        # Models that can handle summarization
        if any(name in model_lower for name in ["llama3.1:70b", "mixtral", "qwen:72b"]):
            return "summarizing"
        
        # Models that benefit from optimization
        elif any(name in model_lower for name in ["llama3.1:8b", "mistral:latest", "phi3:latest"]):
            return "optimized"
            
        # Default to simple for smaller models
        else:
            return "simple"
    
    # Delegate all methods to the actual context manager
    def __getattr__(self, name):
        return getattr(self._context_manager, name)