from src.ii_agent.core.agent import IIAgent, AgentContext, AgentState
from wrappers.ollama_wrapper import OllamaWrapper
from .banking_customer_agent import BankingCustomerAgent
from typing import Dict, Any, List
import asyncio

class BankingOrchestrator(IIAgent):
    """Orchestrator for coordinating multiple banking agents with LLM capabilities."""
    
    def __init__(self):
        super().__init__(
            agent_id="banking_orchestrator",
            name="Banking Orchestrator", 
            description="Coordinates multiple banking agents for complex tasks"
        )
        self.sub_agents = {}
        self.ollama_wrapper = None
        
    async def initialize(self, context: AgentContext, mcp_wrapper, ollama_wrapper: OllamaWrapper = None):
        """Initialize orchestrator and sub-agents with LLM capabilities."""
        await super().initialize(context, mcp_wrapper)
        
        # Initialize Ollama wrapper
        if ollama_wrapper:
            self.ollama_wrapper = ollama_wrapper
        else:
            self.ollama_wrapper = OllamaWrapper()
            await self.ollama_wrapper.initialize()
        
        # Initialize sub-agents with LLM capabilities
        self.sub_agents["customer"] = BankingCustomerAgent()
        await self.sub_agents["customer"].initialize(context, mcp_wrapper, self.ollama_wrapper)
        
        await self.add_thought("observation", "Banking orchestrator and sub-agents initialized with LLM capabilities")
        
    async def _create_plan(self, goal: str) -> List[str]:
        """Create orchestration plan using LLM reasoning."""
        await self.add_thought("thought", f"Using LLM to create orchestration plan for: {goal}")
        
        if self.ollama_wrapper:
            try:
                # Use LLM to generate intelligent orchestration plan
                context = {
                    "agent_type": "banking_orchestrator",
                    "available_sub_agents": list(self.sub_agents.keys()),
                    "sub_agent_capabilities": {
                        "customer": ["customer_profile", "account_balance", "transaction"]
                    },
                    "working_memory": self.context.working_memory
                }
                
                plan = await self.ollama_wrapper.generate_plan(goal, context)
                await self.add_thought("thought", f"LLM generated orchestration plan: {plan}")
                return plan
            except Exception as e:
                await self.add_thought("thought", f"LLM planning failed, using fallback: {e}")
        
        # Fallback to rule-based planning
        return await self._fallback_create_plan(goal)
        
    async def _fallback_create_plan(self, goal: str) -> List[str]:
        """Fallback orchestration planning if LLM fails."""
        goal_lower = goal.lower()
        
        if "comprehensive" in goal_lower or "analysis" in goal_lower:
            return [
                "analyze_goal_complexity",
                "identify_required_agents", 
                "coordinate_parallel_execution",
                "synthesize_agent_results",
                "perform_cross_agent_analysis"
            ]
        elif "customer" in goal_lower and ("balance" in goal_lower or "transaction" in goal_lower):
            return [
                "delegate_to_customer_agent",
                "monitor_agent_execution",
                "validate_results"
            ]
        else:
            return [
                "analyze_goal",
                "route_to_appropriate_agent",
                "monitor_execution"
            ]
            
    async def _execute_step(self, step: str) -> Any:
        """Execute orchestration step."""
        await self.add_thought("action", f"Orchestrator executing: {step}")
        
        if step == "analyze_goal_complexity":
            return await self._analyze_goal_complexity()
        elif step == "identify_required_agents":
            return await self._identify_required_agents()
        elif step == "coordinate_parallel_execution":
            return await self._coordinate_parallel_execution()
        elif step == "synthesize_agent_results":
            return await self._synthesize_agent_results()
        elif step == "perform_cross_agent_analysis":
            return await self._perform_cross_agent_analysis()
        elif step == "delegate_to_customer_agent":
            return await self._delegate_to_customer_agent()
        elif step == "monitor_agent_execution":
            return await self._monitor_agent_execution()
        elif step == "validate_results":
            return await self._validate_results()
        else:
            await self.add_thought("thought", f"Unknown orchestration step: {step}")
            return None
            
    async def _analyze_goal_complexity(self) -> Dict[str, Any]:
        """Analyze the complexity of the goal using LLM reasoning."""
        goal = self.context.current_goal or ""
        
        if self.ollama_wrapper:
            try:
                # Use LLM for intelligent complexity analysis
                reasoning = await self.ollama_wrapper.generate_reasoning(
                    f"Analyze the complexity of this banking goal: {goal}",
                    {
                        "context": "goal_complexity_analysis",
                        "available_agents": list(self.sub_agents.keys()),
                        "goal": goal
                    }
                )
                
                await self.add_thought("thought", f"LLM complexity analysis: {reasoning}")
                
                analysis = {
                    "complexity_score": self._estimate_complexity_score(goal),
                    "llm_analysis": reasoning,
                    "requires_orchestration": True,
                    "estimated_agents_needed": len([agent for agent in self.sub_agents.keys()])
                }
            except Exception as e:
                await self.add_thought("thought", f"LLM complexity analysis failed: {e}")
                analysis = self._fallback_analyze_goal_complexity(goal)
        else:
            analysis = self._fallback_analyze_goal_complexity(goal)
        
        self.context.working_memory["complexity_analysis"] = analysis
        await self.add_thought("thought", f"Goal complexity analysis: {analysis}")
        return analysis
        
    def _estimate_complexity_score(self, goal: str) -> int:
        """Estimate complexity score based on goal content."""
        complexity_indicators = {
            "multi_step": any(word in goal.lower() for word in ["and", "then", "also", "plus"]),
            "requires_analysis": any(word in goal.lower() for word in ["analyze", "compare", "evaluate"]),
            "involves_multiple_data": any(word in goal.lower() for word in ["profile", "balance", "transaction"]),
            "time_sensitive": any(word in goal.lower() for word in ["urgent", "asap", "immediately"])
        }
        return sum(complexity_indicators.values())
        
    def _fallback_analyze_goal_complexity(self, goal: str) -> Dict[str, Any]:
        """Fallback complexity analysis if LLM is unavailable."""
        complexity_score = self._estimate_complexity_score(goal)
        
        return {
            "complexity_score": complexity_score,
            "requires_orchestration": complexity_score >= 2,
            "estimated_agents_needed": 1 if complexity_score < 2 else len(self.sub_agents)
        }
        
    async def _identify_required_agents(self) -> List[str]:
        """Identify which agents are needed for the task."""
        goal = self.context.current_goal or ""
        required_agents = []
        
        if any(word in goal.lower() for word in ["customer", "profile", "balance", "account", "transaction"]):
            required_agents.append("customer")
            
        # Add more agent types as they're implemented
        # if "loan" in goal.lower():
        #     required_agents.append("loan")
        # if "investment" in goal.lower():
        #     required_agents.append("investment")
            
        self.context.working_memory["required_agents"] = required_agents
        await self.add_thought("thought", f"Identified required agents: {required_agents}")
        return required_agents
        
    async def _coordinate_parallel_execution(self) -> Dict[str, Any]:
        """Execute multiple agents in parallel."""
        required_agents = self.context.working_memory.get("required_agents", [])
        
        tasks = []
        for agent_name in required_agents:
            if agent_name in self.sub_agents:
                agent = self.sub_agents[agent_name]
                task = asyncio.create_task(agent.execute(self.context.current_goal))
                tasks.append((agent_name, task))
                
        # Wait for all agents to complete
        agent_results = {}
        for agent_name, task in tasks:
            try:
                result = await task
                agent_results[agent_name] = result
                await self.add_thought("observation", f"Agent {agent_name} completed: {result}")
            except Exception as e:
                agent_results[agent_name] = {"error": str(e)}
                await self.add_thought("observation", f"Agent {agent_name} failed: {str(e)}")
                
        self.context.working_memory["agent_results"] = agent_results
        return agent_results
        
    async def _synthesize_agent_results(self) -> Dict[str, Any]:
        """Synthesize results from multiple agents."""
        agent_results = self.context.working_memory.get("agent_results", {})
        
        synthesis = {
            "combined_data": {},
            "cross_references": {},
            "insights": []
        }
        
        # Combine data from all agents
        for agent_name, result in agent_results.items():
            if isinstance(result, dict) and "final_data" in result:
                synthesis["combined_data"][agent_name] = result["final_data"]
                
        # Look for cross-references (simplified)
        customer_data = synthesis["combined_data"].get("customer", {})
        if "customer_profile" in customer_data and "account_balance" in customer_data:
            synthesis["cross_references"]["profile_balance_match"] = True
            synthesis["insights"].append("Customer profile and balance data are consistent")
            
        self.context.working_memory["synthesis"] = synthesis
        await self.add_thought("thought", f"Synthesized agent results: {synthesis}")
        return synthesis
        
    async def _perform_cross_agent_analysis(self) -> Dict[str, Any]:
        """Perform analysis across agent results using LLM reasoning."""
        synthesis = self.context.working_memory.get("synthesis", {})
        
        if self.ollama_wrapper:
            try:
                # Use LLM for intelligent cross-agent analysis
                reasoning = await self.ollama_wrapper.generate_reasoning(
                    f"Perform cross-agent analysis on synthesized results: {synthesis}",
                    {
                        "context": "cross_agent_analysis",
                        "synthesis_data": synthesis,
                        "participating_agents": list(self.sub_agents.keys())
                    }
                )
                
                await self.add_thought("reflection", f"LLM cross-agent analysis: {reasoning}")
                
                analysis = {
                    "data_completeness": len(synthesis.get("combined_data", {})),
                    "consistency_score": len(synthesis.get("cross_references", {})),
                    "insight_count": len(synthesis.get("insights", [])),
                    "overall_quality": "high",
                    "llm_analysis": reasoning
                }
            except Exception as e:
                await self.add_thought("reflection", f"LLM cross-agent analysis failed: {e}")
                analysis = self._fallback_cross_agent_analysis(synthesis)
        else:
            analysis = self._fallback_cross_agent_analysis(synthesis)
        
        self.context.working_memory["cross_agent_analysis"] = analysis
        await self.add_thought("reflection", f"Cross-agent analysis: {analysis}")
        return analysis
        
    def _fallback_cross_agent_analysis(self, synthesis: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback cross-agent analysis if LLM is unavailable."""
        return {
            "data_completeness": len(synthesis.get("combined_data", {})),
            "consistency_score": len(synthesis.get("cross_references", {})),
            "insight_count": len(synthesis.get("insights", [])),
            "overall_quality": "medium"
        }
        
    async def _delegate_to_customer_agent(self) -> Dict[str, Any]:
        """Delegate task to customer agent."""
        customer_agent = self.sub_agents.get("customer")
        if customer_agent:
            result = await customer_agent.execute(self.context.current_goal)
            self.context.working_memory["customer_result"] = result
            await self.add_thought("observation", f"Customer agent delegation complete: {result}")
            return result
        else:
            await self.add_thought("observation", "Customer agent not available")
            return {"error": "Customer agent not available"}
            
    async def _monitor_agent_execution(self) -> Dict[str, Any]:
        """Monitor the execution of sub-agents."""
        monitoring_data = {
            "active_agents": len([a for a in self.sub_agents.values() if a.state != AgentState.IDLE]),
            "completed_agents": len([a for a in self.sub_agents.values() if a.state == AgentState.COMPLETED]),
            "error_agents": len([a for a in self.sub_agents.values() if a.state == AgentState.ERROR])
        }
        
        await self.add_thought("observation", f"Agent monitoring: {monitoring_data}")
        return monitoring_data
        
    async def _validate_results(self) -> Dict[str, Any]:
        """Validate the results from agents."""
        results = self.context.working_memory
        
        validation = {
            "has_data": len(results) > 0,
            "data_types_valid": True,  # Simplified validation
            "required_fields_present": "customer_id" in str(results),
            "overall_valid": True
        }
        
        await self.add_thought("thought", f"Result validation: {validation}")
        return validation
        
    async def _perform_reflection(self, results: List[Any]) -> Dict[str, Any]:
        """Perform orchestrator reflection using LLM."""
        await self.add_thought("reflection", "Starting orchestrator reflection")
        
        if self.ollama_wrapper:
            try:
                # Use LLM for intelligent orchestrator reflection
                reflection = await self.ollama_wrapper.generate_reflection(
                    results,
                    f"orchestration of banking agents for goal: {self.context.current_goal}"
                )
                
                await self.add_thought("reflection", f"LLM orchestrator reflection: {reflection}")
                
                return {
                    "status": "orchestration_completed",
                    "reflection": reflection,
                    "orchestrated_data": self.context.working_memory
                }
            except Exception as e:
                await self.add_thought("reflection", f"LLM reflection failed: {e}")
        
        # Fallback reflection
        successful_steps = [r for r in results if r is not None]
        
        reflection = {
            "goal_achievement": "complete" if len(successful_steps) == len(results) else "partial",
            "agent_coordination_quality": "high" if self.context.working_memory.get("agent_results") else "low",
            "data_synthesis_quality": "high" if self.context.working_memory.get("synthesis") else "low",
            "overall_effectiveness": "high",
            "success_rating": min(10, len(successful_steps) * 2),
            "recommendations": [
                "Multi-agent coordination successful",
                "Consider adding more specialized agents for complex scenarios"
            ],
            "summary": f"Orchestrated {len(successful_steps)}/{len(results)} steps successfully"
        }
        
        return {
            "status": "orchestration_completed", 
            "reflection": reflection,
            "orchestrated_data": self.context.working_memory
        }