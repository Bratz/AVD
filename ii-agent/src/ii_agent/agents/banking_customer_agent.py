from src.ii_agent.core.agent import IIAgent, AgentContext
from wrappers.ollama_wrapper import OllamaWrapper, OllamaConfig
from typing import Any, Dict, List
import json

class BankingCustomerAgent(IIAgent):
    """II-Agent for customer-related banking operations with LLM capabilities."""
    
    def __init__(self):
        super().__init__(
            agent_id="banking_customer_agent",
            name="Banking Customer Agent",
            description="Specialized agent for customer profile and account management"
        )
        self.ollama_wrapper = None
    async def initialize(self, context: AgentContext, mcp_wrapper, ollama_wrapper: OllamaWrapper = None):
        """Initialize agent with context, MCP wrapper, and Ollama wrapper."""
        await super().initialize(context, mcp_wrapper)
        
        # Initialize Ollama wrapper
        if ollama_wrapper:
            self.ollama_wrapper = ollama_wrapper
        else:
            self.ollama_wrapper = OllamaWrapper()
            await self.ollama_wrapper.initialize()
            
        await self.add_thought("observation", f"Agent {self.name} initialized with LLM capabilities")
        
    async def _create_plan(self, goal: str) -> List[str]:
        """Create plan for customer-related tasks using LLM reasoning."""
        await self.add_thought("thought", f"Using LLM to analyze goal: {goal}")
        
        # Use Ollama to generate intelligent plan
        context = {
            "agent_type": "banking_customer_agent",
            "available_tools": ["customer_profile", "account_balance", "transaction"],
            "working_memory": self.context.working_memory
        }
        
        try:
            plan = await self.ollama_wrapper.generate_plan(goal, context)
            await self.add_thought("thought", f"LLM generated plan: {plan}")
            return plan
        except Exception as e:
            await self.add_thought("thought", f"LLM planning failed, using fallback: {e}")
            # Fallback to rule-based planning
            return await self._fallback_create_plan(goal)
            
    async def _fallback_create_plan(self, goal: str) -> List[str]:
        """Fallback rule-based planning if LLM fails."""
        goal_lower = goal.lower()
        
        if "profile" in goal_lower or "customer" in goal_lower:
            return [
                "extract_customer_id",
                "fetch_customer_profile", 
                "analyze_customer_data",
                "format_response"
            ]
        elif "balance" in goal_lower or "account" in goal_lower:
            return [
                "extract_customer_id",
                "fetch_account_balance",
                "analyze_balance_data", 
                "format_response"
            ]
        elif "transaction" in goal_lower:
            return [
                "extract_customer_id", 
                "fetch_transaction_history",
                "analyze_transactions",
                "format_response"
            ]
        else:
            return [
                "analyze_request",
                "determine_required_tools",
                "execute_tools",
                "synthesize_results"
            ]
            
    async def _execute_step(self, step: str) -> Any:
        """Execute individual step of the plan."""
        await self.add_thought("action", f"Executing step: {step}")
        
        if step == "extract_customer_id":
            return await self._extract_customer_id()
        elif step == "fetch_customer_profile":
            return await self._fetch_customer_profile()
        elif step == "fetch_account_balance":
            return await self._fetch_account_balance()
        elif step == "fetch_transaction_history":
            return await self._fetch_transaction_history()
        elif step == "analyze_customer_data":
            return await self._analyze_customer_data()
        elif step == "analyze_balance_data":
            return await self._analyze_balance_data()
        elif step == "analyze_transactions":
            return await self._analyze_transactions()
        elif step == "format_response":
            return await self._format_response()
        else:
            await self.add_thought("thought", f"Unknown step: {step}, skipping")
            return None
            
    async def _extract_customer_id(self) -> str:
        """Extract customer ID from context or goal."""
        goal = self.context.current_goal or ""
        # Simple extraction - in real implementation, use NLP
        if "customer_id" in self.context.working_memory:
            customer_id = self.context.working_memory["customer_id"]
        else:
            # Default customer ID for demo
            customer_id = "12345"
            
        self.context.working_memory["customer_id"] = customer_id
        await self.add_thought("observation", f"Extracted customer ID: {customer_id}")
        return customer_id
        
    async def _fetch_customer_profile(self) -> Dict[str, Any]:
        """Fetch customer profile using MCP tools."""
        customer_id = self.context.working_memory["customer_id"]
        
        result = await self.mcp_wrapper.execute_tool(
            "customer_profile", 
            {"customer_id": customer_id}
        )
        
        self.context.working_memory["customer_profile"] = result
        await self.add_thought("observation", f"Fetched customer profile: {result}")
        return result
        
    async def _fetch_account_balance(self) -> Dict[str, Any]:
        """Fetch account balance using MCP tools."""
        customer_id = self.context.working_memory["customer_id"]
        
        result = await self.mcp_wrapper.execute_tool(
            "account_balance",
            {"customer_id": customer_id}
        )
        
        self.context.working_memory["account_balance"] = result
        await self.add_thought("observation", f"Fetched account balance: {result}")
        return result
        
    async def _fetch_transaction_history(self) -> Dict[str, Any]:
        """Fetch transaction history using MCP tools."""
        customer_id = self.context.working_memory["customer_id"]
        
        result = await self.mcp_wrapper.execute_tool(
            "transaction",
            {"customer_id": customer_id}
        )
        
        self.context.working_memory["transactions"] = result
        await self.add_thought("observation", f"Fetched transactions: {result}")
        return result
        
    async def _analyze_customer_data(self) -> Dict[str, Any]:
        """Analyze customer profile data using LLM reasoning."""
        profile = self.context.working_memory.get("customer_profile", {})
        
        if self.ollama_wrapper:
            try:
                # Use LLM for intelligent analysis
                reasoning = await self.ollama_wrapper.generate_reasoning(
                    f"Analyzing customer profile data: {profile}",
                    {"context": "customer_data_analysis", "data": profile}
                )
                
                await self.add_thought("thought", f"LLM customer analysis: {reasoning}")
                
                analysis = {
                    "customer_status": "active" if profile else "not_found",
                    "data_completeness": len(profile.keys()) if profile else 0,
                    "risk_assessment": "low",  # Simplified
                    "llm_insights": reasoning
                }
            except Exception as e:
                await self.add_thought("thought", f"LLM analysis failed, using fallback: {e}")
                analysis = self._fallback_analyze_customer_data(profile)
        else:
            analysis = self._fallback_analyze_customer_data(profile)
        
        self.context.working_memory["customer_analysis"] = analysis
        await self.add_thought("thought", f"Customer analysis: {analysis}")
        return analysis
        
    def _fallback_analyze_customer_data(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis if LLM is unavailable."""
        return {
            "customer_status": "active" if profile else "not_found",
            "data_completeness": len(profile.keys()) if profile else 0,
            "risk_assessment": "low"
        }
        
    async def _analyze_balance_data(self) -> Dict[str, Any]:
        """Analyze account balance data using LLM reasoning."""
        balance = self.context.working_memory.get("account_balance", {})
        
        if self.ollama_wrapper:
            try:
                # Use LLM for intelligent analysis
                reasoning = await self.ollama_wrapper.generate_reasoning(
                    f"Analyzing account balance data: {balance}",
                    {"context": "balance_analysis", "data": balance}
                )
                
                await self.add_thought("thought", f"LLM balance analysis: {reasoning}")
                
                analysis = {
                    "balance_status": "positive" if balance.get("amount", 0) > 0 else "negative",
                    "balance_level": "high" if balance.get("amount", 0) > 1000 else "low",
                    "currency": balance.get("currency", "USD"),
                    "llm_insights": reasoning
                }
            except Exception as e:
                await self.add_thought("thought", f"LLM analysis failed, using fallback: {e}")
                analysis = self._fallback_analyze_balance_data(balance)
        else:
            analysis = self._fallback_analyze_balance_data(balance)
        
        self.context.working_memory["balance_analysis"] = analysis
        await self.add_thought("thought", f"Balance analysis: {analysis}")
        return analysis
        
    def _fallback_analyze_balance_data(self, balance: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback balance analysis if LLM is unavailable."""
        return {
            "balance_status": "positive" if balance.get("amount", 0) > 0 else "negative",
            "balance_level": "high" if balance.get("amount", 0) > 1000 else "low",
            "currency": balance.get("currency", "USD")
        }
        
    async def _analyze_transactions(self) -> Dict[str, Any]:
        """Analyze transaction data using LLM reasoning."""
        transactions = self.context.working_memory.get("transactions", [])
        
        if self.ollama_wrapper:
            try:
                # Use LLM for intelligent analysis
                reasoning = await self.ollama_wrapper.generate_reasoning(
                    f"Analyzing transaction data: {transactions}",
                    {"context": "transaction_analysis", "data": transactions}
                )
                
                await self.add_thought("thought", f"LLM transaction analysis: {reasoning}")
                
                analysis = {
                    "transaction_count": len(transactions) if isinstance(transactions, list) else 0,
                    "spending_pattern": "regular",  # Simplified
                    "recent_activity": "active" if transactions else "inactive",
                    "llm_insights": reasoning
                }
            except Exception as e:
                await self.add_thought("thought", f"LLM analysis failed, using fallback: {e}")
                analysis = self._fallback_analyze_transactions(transactions)
        else:
            analysis = self._fallback_analyze_transactions(transactions)
        
        self.context.working_memory["transaction_analysis"] = analysis
        await self.add_thought("thought", f"Transaction analysis: {analysis}")
        return analysis
        
    def _fallback_analyze_transactions(self, transactions: Any) -> Dict[str, Any]:
        """Fallback transaction analysis if LLM is unavailable."""
        return {
            "transaction_count": len(transactions) if isinstance(transactions, list) else 0,
            "spending_pattern": "regular",
            "recent_activity": "active" if transactions else "inactive"
        }
        
    async def _format_response(self) -> Dict[str, Any]:
        """Format final response using LLM for natural language generation."""
        base_response = {
            "agent": self.name,
            "session_id": self.context.session_id,
            "goal_achieved": self.context.current_goal,
            "data": self.context.working_memory,
            "thought_trail_summary": len(self.context.thought_trail)
        }
        
        if self.ollama_wrapper:
            try:
                # Generate natural language summary
                natural_response = await self.ollama_wrapper.generate_response(
                    self.context.working_memory,
                    "Provide a clear, professional summary of the banking information retrieved."
                )
                
                base_response["natural_language_summary"] = natural_response
                await self.add_thought("thought", "Generated natural language summary using LLM")
            except Exception as e:
                await self.add_thought("thought", f"LLM response generation failed: {e}")
        
        await self.add_thought("thought", "Response formatted successfully")
        return base_response
        
    async def _perform_reflection(self, results: List[Any]) -> Dict[str, Any]:
        """Perform reflection on execution results using LLM."""
        await self.add_thought("reflection", "Starting reflection on banking customer task")
        
        if self.ollama_wrapper:
            try:
                # Use LLM for intelligent reflection
                reflection = await self.ollama_wrapper.generate_reflection(
                    results, 
                    self.context.current_goal or "banking customer task"
                )
                
                await self.add_thought("reflection", f"LLM reflection: {reflection}")
                
                # Store reflection in context
                self.context.working_memory["reflection"] = reflection
                
                return {
                    "status": "completed",
                    "reflection": reflection,
                    "final_data": self.context.working_memory
                }
            except Exception as e:
                await self.add_thought("reflection", f"LLM reflection failed: {e}")
        
        # Fallback reflection
        successful_steps = [r for r in results if r is not None]
        failed_steps = len(results) - len(successful_steps)
        
        reflection = {
            "goal_achievement": "complete" if failed_steps == 0 else "partial",
            "result_quality": "high" if self.context.working_memory else "low",
            "execution_effectiveness": "high" if failed_steps == 0 else "medium",
            "success_rating": max(0, 10 - failed_steps * 2),
            "lessons_learned": [
                "Customer data retrieved successfully" if successful_steps else "Retry with different parameters"
            ],
            "recommendations": [
                "Continue with current approach" if successful_steps else "Review input parameters"
            ],
            "summary": f"Completed {len(successful_steps)}/{len(results)} steps successfully"
        }
        
        self.context.working_memory["reflection"] = reflection
        
        return {
            "status": "completed",
            "reflection": reflection,
            "final_data": self.context.working_memory
        }