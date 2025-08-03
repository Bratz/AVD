"""
TCS BaNCS Specialist Agent using II-Agent Framework
"""
import asyncio
from datetime import datetime
import json
import sys
import os
import time
from tkinter import EventType
from typing import Dict, Any, List, Optional

import jsonschema

# Existing imports remain the same
from src.ii_agent.core.agent import IIAgent, AgentContext, AgentState
from src.ii_agent.utils.logging_config import get_logger
from wrappers.ollama_wrapper import OllamaWrapper

from src.ii_agent.core.event import EventType,RealtimeEvent
from src.ii_agent.tools.message_tool import MessageTool
from src.ii_agent.tools.complete_tool import ReturnControlToUserTool
from src.ii_agent.tools.sequential_thinking_tool import SequentialThinkingTool
from src.ii_agent.llm.base import TextPrompt, TextResult  # Fix TextPrompt import
from src.ii_agent.core.parameter_extraction import DomainAwareExtractor
from src.ii_agent.core.planning import Plan  # Add this import

# Dynamic Tool Registration
class TCSBancsToolRegistry:
    def __init__(self, mcp_wrapper):
        self.mcp_wrapper = mcp_wrapper
        self.available_tools = {}
        self.logger = get_logger("BaNCS.ToolRegistry")

    async def discover_tools(self):
        """
        Dynamically discover tools from MCP server
        """
        try:
            tools = await self.mcp_wrapper.list_available_tools()
            for tool in tools:
                self.available_tools[tool['name']] = {
                    'description': tool.get('description', ''),
                    'category': tool.get('category', 'default'),
                    'parameters_schema': tool.get('input_schema', {})
                }
            
            self.logger.info(f"Discovered {len(self.available_tools)} tools")
        except Exception as e:
            self.logger.error(f"Tool discovery failed: {e}")

    def validate_tool_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> bool:
        """
        Validate tool parameters against its schema
        """
        tool_info = self.available_tools.get(tool_name)
        if not tool_info:
            raise ValueError(f"Tool {tool_name} not found")
        
        schema = tool_info.get('parameters_schema', {})
        try:
            jsonschema.validate(instance=parameters, schema=schema)
            return True
        except jsonschema.ValidationError as e:
            self.logger.warning(f"Parameter validation failed for {tool_name}: {e}")
            return False

class TCSBancsSpecialistAgent(IIAgent):
    """Specialized II-Agent for TCS BaNCS banking operations with full event support."""
    
    def __init__(self, ollama_wrapper=None, event_stream=None):
        super().__init__(
            agent_id="tcs_bancs_specialist",
            name="TCS BaNCS Banking Specialist",
            description="Expert agent for comprehensive banking operations using TCS BaNCS platform"
        )
        self.ollama_wrapper = ollama_wrapper
        self.logger = get_logger("tcs_bancs_agent")
        self.tool_registry = None
        
        if ollama_wrapper:
            adapted_client = LLMClientAdapter(ollama_wrapper)
            self.parameter_extractor = DomainAwareExtractor(
                llm_client=adapted_client,
                domain="banking"
            )
        else:
            self.parameter_extractor = DomainAwareExtractor(
                llm_client=None,
                domain="banking"
            )

        # Use provided event stream or create new one
        if event_stream:
            self.event_stream = event_stream
        # else: parent class already creates one
        
        # Core II-Agent tools
        self.message_tool = MessageTool()
        self.return_control_tool = ReturnControlToUserTool()
        self.sequential_thinking_tool = SequentialThinkingTool()
        
    async def initialize(self, context: AgentContext, mcp_wrapper, ollama_wrapper=None):
        """Initialize the agent with dynamic tool discovery"""
        await super().initialize(context, mcp_wrapper, ollama_wrapper)
        
        # Initialize Ollama wrapper if provided
        if ollama_wrapper:
            self.ollama_wrapper = ollama_wrapper
        
        # Setup Tool Registry
        self.tool_registry = TCSBancsToolRegistry(mcp_wrapper)
        await self.tool_registry.discover_tools()
        
        if mcp_wrapper and hasattr(mcp_wrapper, 'tool_registry'):
            list_tool = mcp_wrapper.tool_registry.get_tool('list_api_endpoints')
            if list_tool:
                self.logger.info(f"list_api_endpoints schema: {json.dumps(list_tool.input_schema, indent=2)}")

        # Update available operations dynamically
        self.available_operations = {
            tool_name: {
                "description": tool_info['description'],
                "category": tool_info['category'],
                "required_params": list(tool_info['parameters_schema'].get('required', []))
            }
            for tool_name, tool_info in self.tool_registry.available_tools.items()
        }
        
        # Emit specialized initialization event
        await self.event_stream.emit_event(
            EventType.AGENT_INITIALIZED,
            {
                "agent_type": "TCS_BaNCS_Specialist",
                "tools_discovered": len(self.available_operations),
                "capabilities": ["banking_operations", "mcp_integration", "sequential_planning"],
                "session_id": context.session_id
            }
        )
    
    async def execute(self, goal: str) -> Dict[str, Any]:
        """Execute without pre-planning, like original ii-agent"""
        
        # Just start the conversation
        messages = [{"role": "user", "content": goal}]
        
        while not self.done:
            # Let LLM choose next tool dynamically
            response = await self.llm.generate(
                messages=messages,
                tools=self.available_tools
            )
            
            # Execute chosen tools
            for tool_call in response.tool_calls:
                result = await self.execute_tool(tool_call)
                messages.append({"role": "tool", "content": result})

    # async def execute(self, goal: str) -> Dict[str, Any]:
    #     """Execute with enhanced sequential thinking"""
    #     try:
    #         # Step 1: Emit user message event
    #         await self.event_stream.emit_event(
    #             EventType.USER_MESSAGE,
    #             {"message": goal, "timestamp": datetime.now().isoformat()}
    #         )
            
    #         # Step 2: Parse goal with LLM
    #         self.goal_parameters = await self.parse_goal_with_llm(goal)
    #         await self.add_thought("planning", f"Parsed parameters: {self.goal_parameters}")
            
    #         # Store extraction for visualization
    #         if 'raw_extraction' in self.goal_parameters:
    #             self._last_extraction = self.goal_parameters['raw_extraction']
                
    #         # Log extraction visualization
    #         self.logger.info(self.get_extraction_visualization())

    #         # Step 3: Acknowledge
    #         await self._acknowledge_user_message(goal)
            
    #         # Step 4: Enhanced sequential thinking
    #         thinking_results = await self._sequential_thinking_planning(goal)
            
    #         # Step 5: Create plan using thinking insights
    #         plan = await self.create_plan(goal, planning_context={
    #             "thinking_results": thinking_results,
    #             "goal_parameters": self.goal_parameters,
    #             "available_tools": list(self.available_operations.keys()) if hasattr(self, 'available_operations') else []
    #         })
    #         self.logger.info("GENERATED PLAN:")
    #         self.logger.info(f"Plan ID: {plan.id}")
    #         self.logger.info(f"Goal: {plan.goal}")
    #         self.logger.info(f"Total Steps: {len(plan.steps)}")
    #         for i, step in enumerate(plan.steps):
    #             self.logger.info(f"  Step {i+1}:")
    #             self.logger.info(f"    Description: {step.description}")
    #             self.logger.info(f"    Action: {step.action}")
    #             self.logger.info(f"    Parameters: {step.parameters}")
    #             self.logger.info(f"    Status: {step.status.value}")
    #         self.logger.info("="*60)

    #         # NEW: Step 5.5: Explore alternatives for complex scenarios
    #         alternative_plan = await self._explore_alternatives_if_needed(goal, plan)
    #         if alternative_plan:
    #             await self.add_thought("planning", "Using alternative plan based on exploration")
    #             plan = alternative_plan
            


    #         await self.add_thought("planning", f"Generated plan with {len(plan.steps)} steps")        
    #         # Step 6: Execute plan with dynamic parameter resolution
    #         results = []
    #         execution_context = {}
            
    #         for i, step in enumerate(plan.steps):
    #             # Resolve parameters dynamically based on previous results
    #             if i > 0 and step.action == "get_api_structure":
    #                 # Use API name from previous step
    #                 if 'api_list' in execution_context:
    #                     best_api = await self._select_best_api_with_llm(
    #                         execution_context['api_list'], 
    #                         goal
    #                     )
    #                     step.parameters['endpoint_name'] = best_api
                        
    #             elif step.action == "invoke_banking_api":
    #                 # Use schema from previous step
    #                 if 'selected_api' in execution_context:
    #                     step.parameters['endpoint_name'] = execution_context['selected_api']
    #                     step.parameters['params'] = self._map_params_to_schema(
    #                         self.goal_parameters,
    #                         execution_context.get('api_schema', {})
    #                     )
                
    #             # Execute step
    #             await self.add_thought("action", f"Executing step {i+1}: {step.description}")
                
    #             result = await self._execute_plan_step(step)
    #             results.append(result)
                
    #             # Store context for next steps
    #             if step.action == "list_banking_apis" and 'output' in result:
    #                 execution_context['api_list'] = result['output'].get('apis', [])
    #             elif step.action == "get_api_structure" and 'output' in result:
    #                 execution_context['api_schema'] = result['output']
    #                 execution_context['selected_api'] = step.parameters.get('endpoint_name')
            
    #         # Step 7: Reflect on results
    #         final_result = await self.reflect(results, plan)
            
    #         # Step 8: Send final response
    #         await self._send_final_response(final_result, goal)
            
    #         # Step 9: Return control
    #         await self._return_control_to_user()
            
    #         return {
    #             "status": "completed",
    #             "result": final_result,
    #             "plan_summary": {
    #                 "total_steps": len(plan.steps),
    #                 "successful_steps": len([r for r in results if not isinstance(r, dict) or "error" not in r])
    #             }
    #         }
            
    #     except Exception as e:
    #         self.logger.error(f"Execution failed: {e}", exc_info=True)
    #         raise

    async def _select_best_api_with_llm(self, api_list: List[Dict], goal: str) -> str:
        """Use LLM to select the best API"""
        if not api_list:
            return ""
            
        if not self.ollama_wrapper:
            # Simple matching
            goal_lower = goal.lower()
            for api in api_list:
                if 'balance' in goal_lower and 'balance' in api.get('name', '').lower():
                    return api['name']
            return api_list[0].get('name', '')
        
        # Use LLM for intelligent selection
        selection_prompt = f"""Select the best API for: "{goal}"

    Available APIs:
    {json.dumps([{'name': api['name'], 'description': api.get('description', '')} for api in api_list[:10]], indent=2)}

    Return only the exact API name:"""

        try:
            messages = [[TextPrompt(text=selection_prompt)]]
            response_blocks, _ = await self.ollama_wrapper.generate(
                messages=messages,
                max_tokens=100,
                temperature=0.1
            )
            
            api_name = ""
            for block in response_blocks:
                if isinstance(block, TextResult):
                    api_name += block.text
                    
            return api_name.strip().strip('"').strip("'")
            
        except Exception as e:
            self.logger.error(f"API selection failed: {e}")
            return api_list[0].get('name', '') if api_list else ""

    def _map_params_to_schema(self, extracted_params: Dict, schema: Dict) -> Dict:
        """Map extracted parameters to API schema"""
        mapped = {}
        
        # Simple mapping rules
        param_mapping = {
            'account_id': ['accountId', 'account_id', 'accountNumber'],
            'amount': ['amount', 'transferAmount', 'value'],
            'recipient_account': ['toAccount', 'recipientAccount', 'destinationAccount']
        }
        
        schema_props = schema.get('properties', {})
        
        for our_param, possible_names in param_mapping.items():
            if our_param in extracted_params and extracted_params[our_param]:
                for schema_name in possible_names:
                    if schema_name in schema_props:
                        mapped[schema_name] = extracted_params[our_param]
                        break
                        
        return mapped
            
    async def _acknowledge_user_message(self, goal: str):
        """Use message_user tool to acknowledge"""
        await self.add_thought("observation", f"Received banking request: {goal}")
        
        # Execute message tool
        result = self.message_tool.run(
            {"text": "I understand your banking request. Let me analyze what needs to be done..."},
            None
        )
        
        await self.event_stream.emit_event(
            EventType.AGENT_RESPONSE,
            {
                "message": result.tool_output if hasattr(result, 'tool_output') else str(result),
                "type": "acknowledgment"
            }
        )
        
    # async def _sequential_thinking_planning(self, goal: str) -> List[Dict[str, Any]]:
    #     """Use sequential thinking tool for planning"""
    #     await self.add_thought("planning", "Starting sequential thinking for planning")
        
    #     thinking_results = []
    #     next_needed = True
    #     thought_number = 1
    #     total_thoughts = 5
        
    #     while next_needed and thought_number <= total_thoughts:
    #         # Emit thinking start
    #         await self.event_stream.emit_event(
    #             EventType.AGENT_THINKING,
    #             {
    #                 "thought_number": thought_number,
    #                 "total_thoughts": total_thoughts,
    #                 "status": "thinking"
    #             }
    #         )
            
    #         result = self.sequential_thinking_tool.run({
    #             "thought": f"Planning step {thought_number} for banking request: {goal}",
    #             "thoughtNumber": thought_number,
    #             "totalThoughts": total_thoughts,
    #             "nextThoughtNeeded": thought_number < total_thoughts
    #         })
            
    #         thinking_results.append({
    #             "step": thought_number,
    #             "result": result
    #         })
            
    #         # Extract next_needed from result
    #         if hasattr(result, 'auxiliary_data'):
    #             thought_data = result.auxiliary_data.get('thought_data', {})
    #             next_needed = thought_data.get('nextThoughtNeeded', False)
    #         else:
    #             next_needed = thought_number < total_thoughts
            
    #         thought_number += 1
                
    #     await self.add_thought("planning", f"Completed sequential thinking with {len(thinking_results)} steps")
    #     return thinking_results

    async def _sequential_thinking_planning(self, goal: str) -> List[Dict[str, Any]]:
        """Enhanced usage of sequential thinking tool with full CoT capabilities"""
        await self.add_thought("planning", "Starting enhanced sequential thinking for planning")
        
        thinking_results = []
        thought_number = 1
        total_thoughts = 6  # Initial estimate, can be adjusted dynamically
        
        # Step 1: Initial Analysis with parameter extraction
        analysis_result = self.sequential_thinking_tool.run({
            "thought": f"""Analyzing banking request: {goal}
            
    Let me extract key information:
    - Account numbers: {self._extract_account_ids(goal)}
    - Amounts: {self._extract_amounts(goal)}
    - Operation type: {self._identify_operation_type(goal)}

    Understanding the user's intent...""",
            "thoughtNumber": thought_number,
            "totalThoughts": total_thoughts,
            "nextThoughtNeeded": True
        })
        thinking_results.append(self._extract_tool_result(analysis_result))
        thought_number += 1
        
        # Step 2: API Discovery Strategy
        api_strategy_result = self.sequential_thinking_tool.run({
            "thought": f"""Based on the request, I need to find the right banking API.

    Strategy:
    1. Use list_banking_apis with:
    - tag: {self._determine_api_tag(goal)}
    - search_query: {self._determine_search_query(goal)}
    2. Then get_api_structure to understand parameters
    3. Finally invoke_banking_api with the extracted data

    This approach should find the most relevant API...""",
            "thoughtNumber": thought_number,
            "totalThoughts": total_thoughts,
            "nextThoughtNeeded": True
        })
        thinking_results.append(self._extract_tool_result(api_strategy_result))
        thought_number += 1
        
        # Step 3: Parameter Mapping Plan
        param_result = self.sequential_thinking_tool.run({
            "thought": f"""Planning parameter mapping:

    From the goal, I've identified:
    {json.dumps(self._extract_all_parameters(goal), indent=2)}

    I'll need to:
    1. Map these to the API's expected parameter names
    2. Handle any missing required parameters
    3. Validate data formats

    The key is matching user's natural language to API specifications...""",
            "thoughtNumber": thought_number,
            "totalThoughts": total_thoughts,
            "nextThoughtNeeded": True
        })
        thinking_results.append(self._extract_tool_result(param_result))
        thought_number += 1
        
        # Step 4: Error Handling Strategy
        error_handling_result = self.sequential_thinking_tool.run({
            "thought": """Considering potential issues:

    What could go wrong:
    - No matching API found → broaden search terms
    - Missing parameters → ask user or use defaults
    - API errors → retry with modified parameters

    I should build resilience into each step...""",
            "thoughtNumber": thought_number,
            "totalThoughts": total_thoughts,
            "nextThoughtNeeded": True
        })
        thinking_results.append(self._extract_tool_result(error_handling_result))
        thought_number += 1
        
        # Step 5: Reflection and Optimization
        reflection_result = self.sequential_thinking_tool.run({
            "thought": """Let me review my approach:

    Actually, I should also consider:
    - Checking if this is a multi-step operation
    - Whether I need to chain multiple API calls
    - How to format the response naturally

    Let me revise my strategy to be more comprehensive...""",
            "thoughtNumber": thought_number,
            "totalThoughts": total_thoughts + 1,  # Dynamically extend
            "nextThoughtNeeded": True,
            "isRevision": True,
            "revisesThought": 2
        })
        thinking_results.append(self._extract_tool_result(reflection_result))
        thought_number += 1
        
        # Step 6: Final Execution Plan
        hypothesis_result = self.sequential_thinking_tool.run({
            "thought": f"""Final execution plan for '{goal}':

    1. list_banking_apis with tag='{self._determine_api_tag(goal)}', search='{self._determine_search_query(goal)}'
    2. Select the most relevant API based on description
    3. get_api_structure for the selected API
    4. Map parameters: {json.dumps(self._extract_all_parameters(goal), indent=2)}
    5. invoke_banking_api with mapped parameters
    6. Format response in user-friendly way

    This comprehensive approach should successfully complete the request.""",
            "thoughtNumber": thought_number,
            "totalThoughts": thought_number,
            "nextThoughtNeeded": False
        })
        thinking_results.append(self._extract_tool_result(hypothesis_result))
        
        await self.add_thought("planning", f"Completed sequential thinking with {len(thinking_results)} thoughts")
        return thinking_results
    async def _send_final_response(self, final_result: Dict[str, Any], original_goal: str):
        """Send final response using message_user tool"""
        # Format the final message
        message = f"I've completed your banking request: '{original_goal}'\n\n"
        
        if "summary" in final_result:
            message += f"Summary: {final_result['summary']}\n"
        
        if "recommendations" in final_result:
            message += f"\nRecommendations:\n"
            for rec in final_result['recommendations']:
                message += f"- {rec}\n"
        
        # Use message tool
        result = self.message_tool.run({"text": message}, None)
        
        await self.event_stream.emit_event(
            EventType.AGENT_RESPONSE,
            {
                "message": message,
                "type": "final_response",
                "result_summary": final_result.get("summary", "Task completed")
            }
        )
        
    async def _return_control_to_user(self):
        """Use return_control_to_user tool"""
        result = self.return_control_tool.run({})
        
        await self.event_stream.emit_event(
            EventType.AGENT_RESPONSE,
            {
                "message": "Task completed. Control returned to user.",
                "type": "completion",
                "timestamp": datetime.now().isoformat()
            }
        )
        
    async def _execute_plan_step(self, step):
        """Execute a single plan step - override parent to use banking tools"""
        # Map action to actual tool execution
        if step.action == "list_banking_apis":
            return await self._execute_list_apis(step.parameters)
        elif step.action == "get_api_structure":
            return await self._execute_get_structure(step.parameters)
        elif step.action == "invoke_banking_api":
            return await self._execute_invoke_api(step.parameters)
        else:
            # Fallback to dynamic tool execution
            return await self.execute_dynamic_tool(step.action, step.parameters)
    
    async def _execute_list_apis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute list_banking_apis through tool manager"""
        if hasattr(self, 'tool_manager') and self.tool_manager:
            try:
                tool = self.tool_manager.get_tool("list_banking_apis")
                result = tool.run(parameters, None)
                return {"output": result.tool_output if hasattr(result, 'tool_output') else str(result)}
            except:
                pass
        
        # Fallback to MCP
        return await self.mcp_wrapper.execute_tool("list_api_endpoints", parameters)
    
    async def _execute_get_structure(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute get_api_structure through tool manager"""
        if hasattr(self, 'tool_manager') and self.tool_manager:
            try:
                tool = self.tool_manager.get_tool("get_api_structure")
                result = tool.run(parameters, None)
                return {"output": result.tool_output if hasattr(result, 'tool_output') else str(result)}
            except:
                pass
        
        # Fallback to MCP
        return await self.mcp_wrapper.execute_tool("get_api_endpoint_schema", parameters)
    
    async def _execute_invoke_api(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute invoke_banking_api through tool manager"""
        if hasattr(self, 'tool_manager') and self.tool_manager:
            try:
                tool = self.tool_manager.get_tool("invoke_banking_api")
                result = tool.run(parameters, None)
                return {"output": result.tool_output if hasattr(result, 'tool_output') else str(result)}
            except:
                pass
        
        # Fallback to MCP
        return await self.mcp_wrapper.execute_tool("invoke_api_endpoint", parameters)
    
    async def _execute_step(self, action: str, parameters: Dict[str, Any] = None) -> Any:
        """
        Execute a single step - implementation of abstract method from IIAgent
        """
        parameters = parameters or {}
        
        self.logger.info(f"Executing step: {action} with parameters: {parameters}")
        
        # First, check if it's a system tool (from tool_manager)
        if hasattr(self, 'tool_manager') and self.tool_manager:
            try:
                tool = self.tool_manager.get_tool(action)
                if tool:
                    self.logger.info(f"Executing system tool: {action}")
                    result = tool.run(parameters, None)
                    if hasattr(result, 'tool_output'):
                        return {"output": result.tool_output, "tool_type": "system"}
                    else:
                        return {"output": str(result), "tool_type": "system"}
            except ValueError:
                # Tool not found in tool_manager, continue
                pass
        
        # Map banking tool names to MCP tool names
        tool_mapping = {
            "list_banking_apis": "list_api_endpoints",
            "get_api_structure": "get_api_endpoint_schema", 
            "invoke_banking_api": "invoke_api_endpoint"
        }
        
        mcp_tool_name = tool_mapping.get(action, action)
        
        # Route generic actions based on context
        if action in ["extract_data", "execute_step", "fetch_data", "analyze_data"]:
            # For generic actions in banking context, use list_api_endpoints
            if self._is_banking_context():
                mcp_tool_name = "list_api_endpoints"
                parameters = parameters
                # parameters 
                # {
                #     "compact": True,
                #     "max_results": 50,
                #     **parameters
                # }
            else:
                return {"status": "handled", "action": action, "result": "generic action"}
        
        # Try to execute as MCP tool
        if self.mcp_wrapper:
            try:
                # Check if it's a valid MCP tool
                if hasattr(self.mcp_wrapper, 'tool_registry'):
                    tool_metadata = self.mcp_wrapper.tool_registry.get_tool(mcp_tool_name)
                    if tool_metadata:
                        self.logger.info(f"Executing MCP tool: {mcp_tool_name}")
                        result = await self.mcp_wrapper.execute_tool(mcp_tool_name, parameters)
                        return {"output": result, "tool": mcp_tool_name, "tool_type": "mcp"}
            except Exception as e:
                self.logger.error(f"MCP tool execution failed for {mcp_tool_name}: {e}")
        
        # Handle other standard actions
        if action == "analyze_goal":
            return {"analysis": f"Analyzed: {parameters.get('goal', 'unknown')}", "status": "completed"}
        elif action == "execute_task":
            return {"status": "Task executed", "parameters": parameters}
        elif action == "format_results":
            return {"formatted": True, "data": parameters.get('results', [])}
        else:
            self.logger.warning(f"Unknown action: {action}")
            return {"status": "completed", "action": action, "parameters": parameters}

    def _is_banking_context(self) -> bool:
        """Determine if current context is banking-related"""
        if self.context and self.context.current_goal:
            goal_lower = self.context.current_goal.lower()
            banking_keywords = ["api", "account", "balance", "transfer", "banking", "payment"]
            return any(keyword in goal_lower for keyword in banking_keywords)
        return False
    
    # The existing _execute_plan_step method should now call _execute_step
    async def _execute_plan_step(self, step):
        """Execute a single plan step using the _execute_step method"""
        return await self._execute_step(step.action, step.parameters)

    async def _generate_natural_language_response(self, data: Any, context: str, goal: str) -> str:
        """Use Ollama to generate natural language response from banking data"""
    
        # Check if we have Ollama available
        if not (self.ollama_wrapper or (hasattr(self, 'ollama_client') and self.ollama_client)):
            # Fallback to structured formatting
            return self._format_results_for_display([{'output': data}], goal)
        
        # Prepare the prompt for Ollama
        prompt = f"""You are a friendly banking assistant. Convert this banking data into a natural, conversational response.

    User's Request: {goal}
    Data Type: {context}
    Data: {json.dumps(data, indent=2) if isinstance(data, (dict, list)) else str(data)}

    Instructions:
    1. Provide a warm, conversational response
    2. Highlight the most important information first
    3. Use natural language, not technical jargon
    4. Format numbers as currency where appropriate (e.g., $1,234.56)
    5. If it's a list, summarize key points rather than listing everything
    6. If there's an error, explain it helpfully
    7. End with a relevant follow-up question or offer to help further

    Generate a natural response:"""

        try:
            # Use whichever LLM is available
            llm = self.ollama_wrapper or getattr(self, 'ollama_client', None)
            
            messages = [[TextPrompt(text=prompt)]]
            
            response_blocks, _ = llm.generate(
                messages=messages,
                max_tokens=500,
                temperature=0.7  # Slightly creative for natural language
            )
            
            # Extract response
            natural_response = ""
            for block in response_blocks:
                if isinstance(block, TextResult):
                    natural_response += block.text
                elif hasattr(block, 'text'):
                    natural_response += block.text
            
            return natural_response.strip()
        
        except Exception as e:
            self.logger.error(f"Ollama natural language generation failed: {e}")
            # Fallback to structured formatting
            return self._format_results_for_display([{'output': data}], goal)

    async def _send_final_response(self, final_result: Dict[str, Any], original_goal: str):
        """Send final response using natural language generation"""
        
        # Get execution results
        execution_results = getattr(self, '_execution_results', [])
        
        if execution_results:
            # Find the primary result
            primary_result = None
            for result in execution_results:
                if isinstance(result, dict) and 'output' in result:
                    primary_result = result
                    break
            
            if primary_result:
                # Extract data from MCP format
                data = self._extract_mcp_data(primary_result['output'])
                context = primary_result.get('tool', 'banking operation')
                
                # Generate natural language response
                natural_message = await self._generate_natural_language_response(
                    data, 
                    context, 
                    original_goal
                )
                
                # Add a footer with reflection if available
                if "summary" in final_result:
                    natural_message += f"\n\n📝 _Summary: {final_result['summary']}_"
            else:
                natural_message = f"I've completed your request: {original_goal}"
        else:
            natural_message = f"✅ Task completed: {original_goal}\n\n{final_result.get('summary', '')}"
        
        # Send via MessageTool
        self.message_tool.run({"text": natural_message}, None)
        
        # Emit structured event for web clients
        await self.event_stream.emit_event(
            EventType.AGENT_RESPONSE,
            {
                "message": natural_message,
                "type": "final_response",
                "rich_data": {
                    "goal": original_goal,
                    "execution_results": execution_results,
                    "natural_language": True,
                    "structured_data": self._extract_mcp_data(execution_results[0]['output']) if execution_results else None
                }
            }
        )

    async def _stream_natural_response(self, data: Any, context: str, goal: str):
        """Stream natural language response for real-time display"""
        
        if not (self.ollama_wrapper or getattr(self, 'ollama_client', None)):
            # Can't stream without Ollama
            response = self._format_results_for_display([{'output': data}], goal)
            await self.event_stream.emit_event(
                EventType.AGENT_RESPONSE,
                {"message": response, "type": "streaming_complete"}
            )
            return
        
        prompt = f"""Convert this banking data into a friendly conversation:
    User asked: {goal}
    Data: {json.dumps(data, indent=2)[:1000]}  # Limit size

    Respond naturally and conversationally:"""

        try:
            llm = self.ollama_wrapper or getattr(self, 'ollama_client', None)
            
            # Use streaming if available
            if hasattr(llm, 'generate_stream'):
                accumulated = ""
                
                async def on_chunk(chunk):
                    if chunk['type'] == 'content':
                        accumulated += chunk['data']
                        # Emit streaming event
                        await self.event_stream.emit_event(
                            EventType.AGENT_RESPONSE,
                            {
                                "type": "streaming",
                                "content": chunk['data'],
                                "accumulated": accumulated
                            }
                        )
                
                messages = [[TextPrompt(text=prompt)]]
                
                response_blocks, _ = await llm.generate_stream(
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7,
                    on_chunk=on_chunk
                )
                
                # Final event
                await self.event_stream.emit_event(
                    EventType.AGENT_RESPONSE,
                    {"type": "streaming_complete", "message": accumulated}
                )
                
            else:
                # Non-streaming fallback
                await self._generate_natural_language_response(data, context, goal)
                
        except Exception as e:
            self.logger.error(f"Streaming failed: {e}")

    async def _generate_intelligent_response(self, data: Any, goal: str, tool_used: str) -> str:
        """Use Ollama to intelligently interpret and present data"""
        
        if not self.ollama_wrapper:
            return self._format_results_for_display([{'output': data}], goal)
        
        # Create a smart prompt based on the tool and goal
        prompt = f"""
    You are a helpful banking assistant. The user asked: "{goal}"

    I used the "{tool_used}" tool and received this data:
    {json.dumps(data, indent=2) if isinstance(data, (dict, list)) else str(data)}

    Please provide a natural, helpful response that:
    1. Directly answers what the user asked for
    2. Highlights the most important information
    3. Formats it clearly with bullet points or sections
    4. Adds helpful context or next steps if appropriate
    5. Is friendly and professional

    Important: Focus on presenting the actual data, not describing what you did.
    """
        
        try:
            # Use Ollama's generate method directly for more control
            messages = [[TextPrompt(text=prompt)]]
            response_blocks, _ = await self.ollama_wrapper.generate(
                messages=messages,
                max_tokens=500,
                temperature=0.3,
                system_prompt="You are a banking assistant presenting data to customers."
            )
            
            # Extract response
            response = ""
            for block in response_blocks:
                if hasattr(block, 'text'):
                    response += block.text
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to generate intelligent response: {e}")
            return self._format_results_for_display([{'output': data}], goal)

    async def parse_goal_with_llm(self, goal: str) -> Dict[str, Any]:
        """Use universal parameter extraction"""
        
        # Define expected parameters for banking
        expected_params = [
            {"name": "account_id", "type": "identifier", "description": "Account number or ID"},
            {"name": "amount", "type": "numeric", "description": "Monetary amount"},
            {"name": "recipient", "type": "identifier", "description": "Recipient account or name"},
            {"name": "date_range", "type": "temporal", "description": "Date or time period"},
            {"name": "transaction_type", "type": "selection", "description": "Type of transaction"}
        ]
        
        # Extract parameters
        extraction_results = await self.parameter_extractor.extract_parameters(
            text=goal,
            expected_params=expected_params,
            domain_context={
                "customer_id": self.context.working_memory.get('customer_id'),
                "session_id": self.context.session_id
            }
        )
        
        # Convert to agent format
        params = {
            "raw_extraction": extraction_results,
            "operation_type": self._determine_operation_from_extraction(extraction_results),
            "confidence": extraction_results['summary'].get('total_parameters', 0) > 0
        }
        
        # Add specific parameters
        for param_type, values in extraction_results['parameters'].items():
            if param_type == 'identifier' and values:
                # First identifier is likely account
                params['account_id'] = values[0]['value']
                # Second might be recipient
                if len(values) > 1:
                    params['recipient_account'] = values[1]['value']
            
            elif param_type == 'numeric' and values:
                params['amount'] = values[0]['value']
            
            elif param_type == 'temporal' and values:
                params['date'] = values[0]['value']
        
        # Determine API search parameters
        params['api_category'] = self._determine_api_tag_from_extraction(extraction_results)
        params['search_terms'] = self._extract_search_terms_from_extraction(extraction_results)
        
        self.logger.info(f"Universal extraction found {extraction_results['summary']['total_parameters']} parameters")
        
        return params

    async def _explore_alternatives_if_needed(self, goal: str, initial_plan: Plan) -> Optional[Plan]:
        """Use branching thoughts to explore alternatives for complex scenarios"""
        
        # Only explore alternatives for complex goals
        if self._assess_complexity(goal) != "complex":
            return None
        
        await self.add_thought("planning", "Exploring alternative approaches using branching thoughts")
        
        # Get the current thought count for branching
        thought_history_length = len(self.sequential_thinking_tool.thought_history) if hasattr(self.sequential_thinking_tool, 'thought_history') else 0
        
        # Create branch for alternative approach
        alt_result = self.sequential_thinking_tool.run({
            "thought": f"""Let me think of an alternative approach for '{goal}'.
            
    Instead of the standard discovery flow, what if:
    1. I use more specific search terms?
    2. I combine multiple API calls?
    3. I use a different category filter?

    Alternative approach:""",
            "thoughtNumber": 1,
            "totalThoughts": 3,
            "nextThoughtNeeded": True,
            "branchFromThought": thought_history_length,
            "branchId": "alternative_approach"
        })
        
        # Store alternative result
        alternative_steps = []
        
        # Continue branch thinking
        detailed_alt = self.sequential_thinking_tool.run({
            "thought": """Specifically, I could:
    1. Search with tag='accounts' AND search_query='balance statement'
    2. Or try tag='customers' for customer-centric APIs
    3. Use regex matching on API descriptions

    This might find more specific APIs...""",
            "thoughtNumber": 2,
            "totalThoughts": 3,
            "nextThoughtNeeded": True,
            "branchId": "alternative_approach"
        })
        
        # Evaluate alternatives
        evaluation = self.sequential_thinking_tool.run({
            "thought": """Comparing approaches:
    - Original: Standard, reliable, might be generic
    - Alternative: More specific, might miss APIs, could be faster

    Decision: I'll stick with original but incorporate specific search terms from alternative.""",
            "thoughtNumber": 3,
            "totalThoughts": 3,
            "nextThoughtNeeded": False,
            "branchId": "alternative_approach"
        })
        
        # Extract insights from alternative thinking
        if hasattr(evaluation, 'auxiliary_data'):
            insights = evaluation.auxiliary_data.get('thought_data', {})
            
            # If alternative is better, create new plan
            if "use alternative" in str(insights).lower():
                # Create alternative plan steps
                alternative_steps = [
                    {
                        "description": "Search APIs with refined parameters",
                        "action": "list_banking_apis",
                        "parameters": {
                            "tag": "accounts",
                            "search_query": "balance statement account details"
                        }
                    },
                    {
                        "description": "Get structure of most specific API",
                        "action": "get_api_structure",
                        "parameters": {"endpoint_name": ""}
                    },
                    {
                        "description": "Execute with enhanced parameters",
                        "action": "invoke_banking_api",
                        "parameters": {"endpoint_name": "", "params": {}}
                    }
                ]
                
                # Create new plan using planning engine
                if hasattr(self, 'planning_engine'):
                    alternative_plan = self.planning_engine.create_plan(
                        f"Alternative approach for: {goal}",
                        alternative_steps
                    )
                    
                    await self.add_thought("planning", "Created alternative plan based on branching thoughts")
                    return alternative_plan
        
        # Log the exploration even if we stick with original
        await self.add_thought("planning", "Explored alternatives, refined original approach with insights")
        
        # Could modify the original plan here based on insights
        if initial_plan and hasattr(initial_plan, 'steps'):
            # Example: Update search parameters in first step
            for step in initial_plan.steps:
                if step.action == "list_banking_apis" and "search_query" in step.parameters:
                    # Enhance search query based on alternative thinking
                    step.parameters["search_query"] += " detailed specific"
        
        return None  # Return None to use original plan with modifications

    def _determine_operation_from_extraction(self, extraction: Dict[str, Any]) -> str:
        """Determine operation type from extraction results"""
        # Check metadata for intent
        for param in extraction.get('all_extractions', []):
            intent = param.get('metadata', {}).get('intent', '').lower()
            if 'balance' in intent:
                return 'balance_inquiry'
            elif 'transfer' in intent or 'send' in intent:
                return 'fund_transfer'
            elif 'history' in intent or 'transaction' in intent:
                return 'transaction_history'
        
        # Fallback to checking extracted values
        params = extraction.get('parameters', {})
        if 'numeric' in params and len(params.get('identifier', [])) > 1:
            return 'fund_transfer'  # Multiple IDs + amount suggests transfer
        
        return 'general_inquiry'

    def _determine_api_tag_from_extraction(self, extraction: Dict[str, Any]) -> str:
        """Determine API category from extraction"""
        operation = self._determine_operation_from_extraction(extraction)
        return {
            'balance_inquiry': 'accounts',
            'fund_transfer': 'transfers',
            'transaction_history': 'accounts',
            'general_inquiry': ''
        }.get(operation, '')

    def _extract_search_terms_from_extraction(self, extraction: Dict[str, Any]) -> List[str]:
        """Extract search terms from extraction results"""
        terms = []
        
        # Add high-confidence entity names
        for param in extraction.get('all_extractions', []):
            if param['type'] == 'entity' and param['confidence'] > 0.7:
                terms.append(param['value'].lower())
        
        # Add operation keywords
        operation = self._determine_operation_from_extraction(extraction)
        if operation == 'balance_inquiry':
            terms.extend(['balance', 'account'])
        elif operation == 'fund_transfer':
            terms.extend(['transfer', 'payment'])
        
        return list(set(terms))[:5]  # Unique, limited to 5

    # Add visualization method:
    def get_extraction_visualization(self) -> str:
        """Visualize parameter extraction results"""
        if not hasattr(self, '_last_extraction'):
            return "No extraction results available"
        
        extraction = self._last_extraction
        viz = "🔍 Parameter Extraction Visualization:\n" + "="*50 + "\n\n"
        
        # Summary
        summary = extraction.get('summary', {})
        viz += f"📊 Summary:\n"
        viz += f"  Total parameters: {summary.get('total_parameters', 0)}\n"
        viz += f"  High confidence: {len(summary.get('high_confidence', []))}\n"
        viz += f"  By type:\n"
        for ptype, count in summary.get('by_type', {}).items():
            viz += f"    {ptype}: {count}\n"
        
        # Parameters
        viz += f"\n📌 Extracted Parameters:\n"
        for param_type, values in extraction.get('parameters', {}).items():
            viz += f"  {param_type.upper()}:\n"
            for val in values:
                viz += f"    - {val['name']}: {val['value']} (conf: {val['confidence']:.2f})\n"
        
        # Detailed extractions
        viz += f"\n🔎 All Extractions:\n"
        for ext in extraction.get('all_extractions', [])[:10]:  # First 10
            viz += f"  - {ext['name']}: '{ext['value']}' "
            viz += f"[{ext['type']}] (conf: {ext['confidence']:.2f})\n"
            if ext['metadata'].get('reasoning'):
                viz += f"    Reasoning: {ext['metadata']['reasoning']}\n"
        
        return viz
        # Add these methods to the TCSBancsSpecialistAgent class:

    def _extract_tool_result(self, tool_result) -> Dict[str, Any]:
        """Extract result from tool execution"""
        if hasattr(tool_result, 'tool_output'):
            try:
                return json.loads(tool_result.tool_output)
            except:
                return {"output": tool_result.tool_output}
        elif hasattr(tool_result, 'auxiliary_data'):
            return tool_result.auxiliary_data
        else:
            return {"result": str(tool_result)}

    def _extract_all_parameters(self, goal: str) -> Dict[str, Any]:
        """Extract all parameters using combined approach"""
        return {
            "operation_type": self._identify_operation_type(goal),
            "account_ids": self._extract_account_ids(goal),
            "amounts": self._extract_amounts(goal),
            "dates": self._extract_dates(goal),
            "api_category": self._determine_api_tag(goal),
            "search_terms": self._determine_search_query(goal).split(),
            **self._extract_operation_params(goal)
        }

    def _identify_operation_type(self, goal: str) -> str:
        """Identify the type of banking operation"""
        goal_lower = goal.lower()
        
        operations = {
            "balance_inquiry": ["balance", "how much", "amount in"],
            "fund_transfer": ["transfer", "send", "pay", "payment to"],
            "transaction_history": ["transaction", "history", "statement", "recent"],
            "account_details": ["account detail", "account info", "my account"]
        }
        
        for op_type, keywords in operations.items():
            if any(kw in goal_lower for kw in keywords):
                return op_type
                
        return "general_inquiry"

    def _extract_account_ids(self, goal: str) -> List[str]:
        """Extract account IDs using pattern matching"""
        import re
        patterns = [
            r'\b\d{10,12}\b',
            r'account\s*(?:number\s*)?(\d+)',
            r'acc\s*(?:no\s*)?(\d+)'
        ]
        
        account_ids = []
        for pattern in patterns:
            matches = re.findall(pattern, goal, re.IGNORECASE)
            account_ids.extend(matches)
            
        return list(set(filter(None, account_ids)))

    def _extract_amounts(self, goal: str) -> List[str]:
        """Extract monetary amounts"""
        import re
        patterns = [
            r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:dollars?|usd)',
        ]
        
        amounts = []
        for pattern in patterns:
            matches = re.findall(pattern, goal, re.IGNORECASE)
            amounts.extend(matches)
            
        return [amt.replace(',', '') for amt in amounts]

    def _extract_dates(self, goal: str) -> List[str]:
        """Extract date references"""
        date_keywords = ['today', 'yesterday', 'last week', 'last month', 'this month']
        found_dates = [kw for kw in date_keywords if kw in goal.lower()]
        return found_dates

    def _extract_operation_params(self, goal: str) -> Dict[str, Any]:
        """Extract operation-specific parameters"""
        import re
        params = {}
        
        if 'transfer' in goal.lower():
            to_match = re.search(r'to\s+(?:account\s*)?(\d+)', goal, re.IGNORECASE)
            if to_match:
                params['recipient_account'] = to_match.group(1)
                
        return params

    def _determine_api_tag(self, goal: str) -> str:
        """Determine API tag for searching"""
        op_type = self._identify_operation_type(goal)
        return {
            "balance_inquiry": "accounts",
            "fund_transfer": "transfers",
            "transaction_history": "accounts",
            "account_details": "accounts",
        }.get(op_type, "")

    def _determine_search_query(self, goal: str) -> str:
        """Determine search query for API discovery"""
        op_type = self._identify_operation_type(goal)
        queries = {
            "balance_inquiry": "balance account",
            "fund_transfer": "transfer payment send",
            "transaction_history": "transaction history statement",
            "account_details": "account details information"
        }
        return queries.get(op_type, "account")

    def _assess_complexity(self, goal: str) -> str:
        """Assess complexity of the request"""
        # Simple heuristic based on length and keywords
        if len(goal.split()) < 10:
            return "simple"
        elif any(word in goal.lower() for word in ['and', 'then', 'also', 'multiple']):
            return "complex"
        else:
            return "medium"

class ModularReasoner:
    """Implements modular reasoning for complex banking queries"""
    
    def __init__(self, ollama_wrapper):
        self.ollama_wrapper = ollama_wrapper
        self.modules = {
            "intent_analysis": self._analyze_intent,
            "entity_extraction": self._extract_entities,
            "api_mapping": self._map_to_apis,
            "parameter_planning": self._plan_parameters,
            "execution_strategy": self._plan_execution
        }
    
    async def reason(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply modular reasoning to understand and plan for a goal"""
        
        reasoning_results = {}
        
        # Execute each reasoning module
        for module_name, module_func in self.modules.items():
            try:
                result = await module_func(goal, context, reasoning_results)
                reasoning_results[module_name] = result
            except Exception as e:
                reasoning_results[module_name] = {"error": str(e)}
        
        return reasoning_results
    
    async def _analyze_intent(self, goal: str, context: Dict, previous: Dict) -> Dict:
        """Deep intent analysis module"""
        
        prompt = f"""Analyze the deep intent behind this banking request:

Request: "{goal}"

Consider:
1. What is the primary intent?
2. Are there any secondary intents?
3. What's the urgency level?
4. What's the complexity level?
5. What domain knowledge is required?

Provide a detailed intent analysis:"""

        # Use Ollama with thinking model
        messages = [[TextPrompt(text=prompt)]]
        response_blocks, _ = await self.ollama_wrapper.generate(
            messages=messages,
            max_tokens=300,
            temperature=0.3
        )
        
        # Parse response
        analysis = self._extract_text_from_blocks(response_blocks)
        
        return {
            "analysis": analysis,
            "primary_intent": self._classify_intent(analysis),
            "complexity": self._assess_complexity(analysis)
        }

class LLMClientAdapter:
    """Adapter to provide consistent interface for parameter extraction"""
    
    def __init__(self, wrapped_client):
        self.wrapped_client = wrapped_client
    
    async def generate(self, messages, max_tokens, temperature, system_prompt=None):
        # Try different methods based on what's available
        if hasattr(self.wrapped_client, 'generate'):
            return await self.wrapped_client.generate(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt
            )
        elif hasattr(self.wrapped_client, 'client') and hasattr(self.wrapped_client.client, 'generate'):
            return await self.wrapped_client.client.generate(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt
            )
        else:
            raise AttributeError(f"No generate method found on {type(self.wrapped_client)}")