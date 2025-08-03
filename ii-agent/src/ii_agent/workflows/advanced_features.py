# src/ii_agent/workflows/advanced_features.py
"""
Advanced workflow features for II-Agent
Including subworkflows, concurrent execution, and dynamic routing
"""

from typing import Dict, List, Any, Optional, Union, Callable
import asyncio
import uuid
from datetime import datetime
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver

from src.ii_agent.workflows.langgraph_integration import BaNCSLangGraphBridge, WorkflowState
from src.ii_agent.workflows.definitions import WorkflowDefinition, WorkflowEdge, EdgeConditionType


class DynamicRouter:
    """Dynamic routing based on runtime conditions"""
    
    def __init__(self):
        self.routing_rules: Dict[str, Callable] = {}
        self.fallback_route: Optional[str] = None
    
    def add_rule(self, condition: str, route_fn: Callable[[WorkflowState], str]):
        """Add a routing rule"""
        self.routing_rules[condition] = route_fn
    
    def set_fallback(self, route: str):
        """Set fallback route when no rules match"""
        self.fallback_route = route
    
    def route(self, state: WorkflowState) -> str:
        """Determine route based on state"""
        for condition, route_fn in self.routing_rules.items():
            try:
                if self._evaluate_condition(condition, state):
                    return route_fn(state)
            except Exception:
                continue
        
        return self.fallback_route or END
    
    def _evaluate_condition(self, condition: str, state: WorkflowState) -> bool:
        """Safely evaluate a condition"""
        # Create safe evaluation context
        safe_context = {
            "state": state,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "any": any,
            "all": all,
        }
        
        try:
            return eval(condition, {"__builtins__": {}}, safe_context)
        except Exception:
            return False


class SubWorkflowManager:
    """Manage subworkflows within parent workflows"""
    
    def __init__(self, parent_workflow: StateGraph):
        self.parent_workflow = parent_workflow
        self.subworkflows: Dict[str, StateGraph] = {}
        self.bridge = BaNCSLangGraphBridge()
    
    def add_subworkflow(
        self,
        name: str,
        subworkflow_def: Union[WorkflowDefinition, Dict[str, Any]],
        parent_node: str
    ) -> StateGraph:
        """Add a subworkflow to parent"""
        
        if isinstance(subworkflow_def, WorkflowDefinition):
            subworkflow_def = subworkflow_def.dict()
        
        # Create subworkflow graph
        subgraph = self._create_subgraph(name, subworkflow_def)
        
        # Register in parent
        self.subworkflows[name] = subgraph
        
        # Add subworkflow node to parent
        self.parent_workflow.add_node(
            f"subworkflow_{name}",
            lambda state: self._execute_subworkflow(name, state)
        )
        
        # Connect parent node to subworkflow
        self.parent_workflow.add_edge(parent_node, f"subworkflow_{name}")
        
        return subgraph
    
    def _create_subgraph(self, name: str, subworkflow_def: Dict[str, Any]) -> StateGraph:
        """Create a subworkflow graph"""
        subgraph = StateGraph(WorkflowState)
        
        # Add nodes for each agent
        for agent_config in subworkflow_def["agents"]:
            node_name = f"{name}.{agent_config['name']}"
            subgraph.add_node(
                node_name,
                self.bridge._create_agent_node(agent_config)
            )
        
        # Add edges
        for edge in subworkflow_def["edges"]:
            from_node = f"{name}.{edge['from_agent']}"
            
            if edge.get("condition"):
                targets = {}
                for result, target in edge.get("to_agents", {}).items():
                    if target == "__end__":
                        targets[result] = END
                    else:
                        targets[result] = f"{name}.{target}"
                
                subgraph.add_conditional_edges(
                    from_node,
                    self.bridge._create_condition_function(edge["condition"]),
                    targets
                )
            else:
                to_node = f"{name}.{edge['to_agent']}"
                subgraph.add_edge(from_node, to_node)
        
        # Set entry point
        entry_point = f"{name}.{subworkflow_def['entry_point']}"
        subgraph.set_entry_point(entry_point)
        
        return subgraph
    
    async def _execute_subworkflow(self, name: str, state: WorkflowState) -> WorkflowState:
        """Execute a subworkflow"""
        subgraph = self.subworkflows[name]
        
        # Create subworkflow state
        sub_state = WorkflowState(
            messages=state["messages"],
            current_agent=f"{name}.start",
            global_state=state["global_state"].copy(),
            metadata={
                **state["metadata"],
                "parent_workflow": state["metadata"].get("workflow_id"),
                "subworkflow": name
            }
        )
        
        # Execute subworkflow
        checkpointer = MemorySaver()
        result = await subgraph.ainvoke(sub_state, {"checkpointer": checkpointer})
        
        # Merge results back to parent state
        state["messages"].extend(result["messages"])
        state["global_state"].update(result["global_state"])
        
        return state


class ConcurrentWorkflowManager:
    """Manage multiple concurrent workflow executions"""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.active_workflows: Dict[str, asyncio.Task] = {}
        self.workflow_queue = asyncio.Queue()
        self.results: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
    
    async def submit_workflow(
        self,
        workflow_id: str,
        workflow: StateGraph,
        input_data: Dict[str, Any]
    ):
        """Submit a workflow for execution"""
        await self.workflow_queue.put((workflow_id, workflow, input_data))
    
    async def start(self):
        """Start the workflow manager"""
        # Create worker tasks
        workers = []
        for i in range(self.max_concurrent):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            workers.append(worker)
        
        # Wait for all workers
        await asyncio.gather(*workers)
    
    async def _worker(self, worker_id: str):
        """Worker to process workflows from queue"""
        while True:
            try:
                workflow_id, workflow, input_data = await self.workflow_queue.get()
                
                async with self._lock:
                    # Check if we can start new workflow
                    if len(self.active_workflows) >= self.max_concurrent:
                        # Put back in queue
                        await self.workflow_queue.put((workflow_id, workflow, input_data))
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Start workflow execution
                    task = asyncio.create_task(
                        self._execute_workflow(workflow_id, workflow, input_data)
                    )
                    self.active_workflows[workflow_id] = task
                
                # Mark task as done
                self.workflow_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
    
    async def _execute_workflow(
        self,
        workflow_id: str,
        workflow: StateGraph,
        input_data: Dict[str, Any]
    ):
        """Execute a single workflow"""
        try:
            checkpointer = MemorySaver()
            result = await workflow.ainvoke(input_data, {"checkpointer": checkpointer})
            
            self.results[workflow_id] = {
                "status": "completed",
                "result": result,
                "completed_at": datetime.utcnow()
            }
        except Exception as e:
            self.results[workflow_id] = {
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.utcnow()
            }
        finally:
            async with self._lock:
                self.active_workflows.pop(workflow_id, None)
    
    def get_result(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow result if available"""
        return self.results.get(workflow_id)
    
    def get_status(self, workflow_id: str) -> Optional[str]:
        """Get workflow status"""
        if workflow_id in self.active_workflows:
            return "running"
        elif workflow_id in self.results:
            return self.results[workflow_id]["status"]
        else:
            return None


class WorkflowOptimizer:
    """Optimize workflow execution"""
    
    def __init__(self):
        self.execution_stats: Dict[str, Dict[str, Any]] = {}
    
    def analyze_workflow(self, workflow_def: WorkflowDefinition) -> Dict[str, Any]:
        """Analyze workflow for optimization opportunities"""
        
        analysis = {
            "parallelizable_branches": [],
            "potential_bottlenecks": [],
            "redundant_paths": [],
            "optimization_suggestions": []
        }
        
        # Find parallelizable branches
        agent_dependencies = self._build_dependency_graph(workflow_def)
        parallel_groups = self._find_parallel_groups(agent_dependencies)
        
        if parallel_groups:
            analysis["parallelizable_branches"] = parallel_groups
            analysis["optimization_suggestions"].append(
                "Consider using parallel execution for independent agent groups"
            )
        
        # Identify potential bottlenecks
        bottlenecks = self._find_bottlenecks(workflow_def, agent_dependencies)
        if bottlenecks:
            analysis["potential_bottlenecks"] = bottlenecks
            analysis["optimization_suggestions"].append(
                "Consider adding caching or alternative paths for bottleneck agents"
            )
        
        return analysis
    
    def _build_dependency_graph(
        self,
        workflow_def: WorkflowDefinition
    ) -> Dict[str, List[str]]:
        """Build agent dependency graph"""
        dependencies = {agent.name: [] for agent in workflow_def.agents}
        
        for edge in workflow_def.edges:
            if edge.to_agent:
                dependencies[edge.to_agent].append(edge.from_agent)
            elif edge.to_agents:
                for target in edge.to_agents.values():
                    if target != "__end__" and target in dependencies:
                        dependencies[target].append(edge.from_agent)
        
        return dependencies
    
    def _find_parallel_groups(
        self,
        dependencies: Dict[str, List[str]]
    ) -> List[List[str]]:
        """Find groups of agents that can run in parallel"""
        parallel_groups = []
        
        # Find agents with same dependencies
        dep_groups = {}
        for agent, deps in dependencies.items():
            dep_key = tuple(sorted(deps))
            if dep_key not in dep_groups:
                dep_groups[dep_key] = []
            dep_groups[dep_key].append(agent)
        
        # Groups with more than one agent can run in parallel
        for group in dep_groups.values():
            if len(group) > 1:
                parallel_groups.append(group)
        
        return parallel_groups
    
    def _find_bottlenecks(
        self,
        workflow_def: WorkflowDefinition,
        dependencies: Dict[str, List[str]]
    ) -> List[str]:
        """Find potential bottleneck agents"""
        bottlenecks = []
        
        # Count incoming edges for each agent
        incoming_counts = {agent: 0 for agent in dependencies}
        
        for edge in workflow_def.edges:
            if edge.to_agent:
                incoming_counts[edge.to_agent] += 1
            elif edge.to_agents:
                for target in edge.to_agents.values():
                    if target != "__end__" and target in incoming_counts:
                        incoming_counts[target] += 1
        
        # Agents with many incoming edges are potential bottlenecks
        avg_incoming = sum(incoming_counts.values()) / len(incoming_counts)
        for agent, count in incoming_counts.items():
            if count > avg_incoming * 2:  # 2x average
                bottlenecks.append(agent)
        
        return bottlenecks