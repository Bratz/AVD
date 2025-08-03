# src/ii_agent/testing/rowboat_ai_simulator.py
"""
ROWBOAT AI Testing System
Natural language testing and simulation for multi-agent workflows
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import asyncio
import json
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import logging
import random

from src.ii_agent.llm.chutes_openai import ChutesOpenAIClient
from src.ii_agent.llm.model_registry import ChutesModelRegistry
from src.ii_agent.sdk.client import ROWBOATClient

logger = logging.getLogger(__name__)

class TestComplexity(Enum):
    """Test scenario complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EDGE_CASE = "edge_case"
    STRESS = "stress"

class PersonaType(Enum):
    """User persona types for testing"""
    HAPPY_PATH = "happy_path"
    CONFUSED = "confused"
    TECHNICAL = "technical"
    NON_TECHNICAL = "non_technical"
    ADVERSARIAL = "adversarial"
    IMPATIENT = "impatient"
    DETAILED = "detailed"

@dataclass
class TestScenario:
    """Structured test scenario"""
    id: str
    name: str
    description: str
    complexity: TestComplexity
    conversation_plan: List[Dict[str, Any]]
    expected_behaviors: List[str]
    edge_cases: List[str]
    success_criteria: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class TestResult:
    """Test execution result"""
    scenario_id: str
    success: bool
    conversation: List[Dict[str, Any]]
    evaluation: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    issues_found: List[str]
    suggestions: List[str]
    timestamp: datetime

class ROWBOATTestSimulator:
    """ROWBOAT AI-powered testing environment"""
    
    def __init__(self, llm_client: Optional[ChutesOpenAIClient] = None):
        self.llm = llm_client or ChutesModelRegistry.create_llm_client(
            model_key="deepseek-v3",
            use_native_tools=False
        )
        self.scenario_cache = {}
        self.test_history = []
        
        logger.info("ROWBOAT Test Simulator initialized")
    
    async def generate_test_suite(
        self,
        workflow_description: str,
        focus_areas: Optional[List[str]] = None,
        num_scenarios: int = 5
    ) -> List[TestScenario]:
        """Generate comprehensive test suite from workflow description
        
        Args:
            workflow_description: Natural language workflow description
            focus_areas: Areas to focus testing on (e.g., "error handling", "edge cases")
            num_scenarios: Number of scenarios to generate
            
        Example:
            scenarios = await simulator.generate_test_suite(
                "Customer support workflow for refunds",
                focus_areas=["high value refunds", "angry customers"],
                num_scenarios=10
            )
        """
        
        # Generate diverse scenarios
        scenarios = []
        complexities = list(TestComplexity)
        
        for i in range(num_scenarios):
            # Vary complexity
            complexity = complexities[i % len(complexities)]
            
            # Generate scenario
            scenario = await self._generate_scenario(
                workflow_description,
                complexity,
                focus_areas,
                scenario_index=i
            )
            scenarios.append(scenario)
        
        # Cache scenarios
        suite_id = f"suite_{datetime.now().timestamp()}"
        self.scenario_cache[suite_id] = scenarios
        
        return scenarios
    
    async def _generate_scenario(
        self,
        workflow_description: str,
        complexity: TestComplexity,
        focus_areas: Optional[List[str]],
        scenario_index: int
    ) -> TestScenario:
        """Generate individual test scenario"""
        
        prompt = f"""Generate a test scenario for this workflow:
        {workflow_description}
        
        Complexity: {complexity.value}
        Focus areas: {', '.join(focus_areas or ['general functionality'])}
        
        Create a realistic scenario that tests the workflow thoroughly.
        Include:
        1. Scenario name and description
        2. Step-by-step conversation plan
        3. Expected agent behaviors
        4. Edge cases to test
        5. Success criteria
        
        Return as JSON with structure:
        {{
            "name": "scenario name",
            "description": "what this tests",
            "conversation_plan": [
                {{"step": 1, "user_intent": "...", "expected_response_type": "..."}},
                ...
            ],
            "expected_behaviors": ["list of expected behaviors"],
            "edge_cases": ["list of edge cases"],
            "success_criteria": {{
                "must_mention": ["key points"],
                "must_not_do": ["things to avoid"],
                "performance": {{"max_response_time_ms": 5000}}
            }}
        }}"""
        
        try:
            response = await self.llm.agenerate([prompt])
            scenario_data = json.loads(response.generations[0][0].text)
            
            return TestScenario(
                id=f"scenario_{scenario_index}_{datetime.now().timestamp()}",
                name=scenario_data["name"],
                description=scenario_data["description"],
                complexity=complexity,
                conversation_plan=scenario_data["conversation_plan"],
                expected_behaviors=scenario_data["expected_behaviors"],
                edge_cases=scenario_data["edge_cases"],
                success_criteria=scenario_data["success_criteria"],
                metadata={
                    "focus_areas": focus_areas,
                    "generated_at": datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Failed to generate scenario: {e}")
            # Fallback scenario
            return self._create_fallback_scenario(complexity, scenario_index)
    
    def _create_fallback_scenario(
        self,
        complexity: TestComplexity,
        index: int
    ) -> TestScenario:
        """Create fallback scenario if generation fails"""
        
        return TestScenario(
            id=f"fallback_scenario_{index}",
            name=f"{complexity.value.title()} Test Scenario {index}",
            description=f"Testing {complexity.value} workflow interactions",
            complexity=complexity,
            conversation_plan=[
                {
                    "step": 1,
                    "user_intent": "Initial request",
                    "expected_response_type": "acknowledgment"
                },
                {
                    "step": 2,
                    "user_intent": "Provide details",
                    "expected_response_type": "processing"
                },
                {
                    "step": 3,
                    "user_intent": "Confirm action",
                    "expected_response_type": "completion"
                }
            ],
            expected_behaviors=[
                "Respond appropriately",
                "Handle requests correctly",
                "Complete workflow successfully"
            ],
            edge_cases=[],
            success_criteria={
                "must_mention": [],
                "must_not_do": ["error", "fail"],
                "performance": {"max_response_time_ms": 5000}
            },
            metadata={}
        )
    
    async def simulate_scenario(
        self,
        workflow_id: str,
        scenario: TestScenario,
        persona_type: PersonaType = PersonaType.HAPPY_PATH,
        client: Optional[ROWBOATClient] = None
    ) -> TestResult:
        """Simulate a test scenario with AI-generated user interactions
        
        Args:
            workflow_id: Workflow to test
            scenario: Test scenario to execute
            persona_type: Type of user to simulate
            client: ROWBOAT client (creates one if not provided)
        """
        
        logger.info(f"Simulating scenario: {scenario.name} with {persona_type.value} persona")
        
        # Initialize client
        if not client:
            client = ROWBOATClient()
        
        # Generate persona
        persona = await self._generate_persona(persona_type, scenario)
        
        # Initialize conversation
        conv = client.conversation(workflow_id)
        conversation_log = []
        issues_found = []
        
        # Performance tracking
        start_time = datetime.now()
        response_times = []
        
        try:
            # Execute conversation plan
            for step in scenario.conversation_plan:
                # Generate user message based on persona
                user_message = await self._generate_user_message(
                    step,
                    persona,
                    conversation_log
                )
                
                # Send message and track timing
                message_start = datetime.now()
                response = await conv.say(user_message)
                response_time = (datetime.now() - message_start).total_seconds() * 1000
                response_times.append(response_time)
                
                # Log conversation
                conversation_log.append({
                    "step": step["step"],
                    "user": user_message,
                    "assistant": response,
                    "response_time_ms": response_time,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Evaluate response
                step_evaluation = await self._evaluate_step(
                    step,
                    response,
                    scenario.expected_behaviors,
                    response_time
                )
                
                if not step_evaluation["success"]:
                    issues_found.extend(step_evaluation["issues"])
                
                # Check if should continue
                if step_evaluation.get("should_stop", False):
                    break
            
            # Final evaluation
            total_time = (datetime.now() - start_time).total_seconds()
            evaluation = await self._final_evaluation(
                scenario,
                conversation_log,
                response_times,
                total_time
            )
            
            # Generate suggestions
            suggestions = await self._generate_improvement_suggestions(
                scenario,
                conversation_log,
                evaluation
            )
            
            # Create result
            result = TestResult(
                scenario_id=scenario.id,
                success=len(issues_found) == 0 and evaluation["success"],
                conversation=conversation_log,
                evaluation=evaluation,
                performance_metrics={
                    "total_time_seconds": total_time,
                    "average_response_time_ms": sum(response_times) / len(response_times) if response_times else 0,
                    "max_response_time_ms": max(response_times) if response_times else 0,
                    "steps_completed": len(conversation_log),
                    "total_steps": len(scenario.conversation_plan)
                },
                issues_found=issues_found,
                suggestions=suggestions,
                timestamp=datetime.now()
            )
            
            # Store in history
            self.test_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return TestResult(
                scenario_id=scenario.id,
                success=False,
                conversation=conversation_log,
                evaluation={"error": str(e)},
                performance_metrics={},
                issues_found=[f"Simulation error: {str(e)}"],
                suggestions=["Fix the error and rerun the test"],
                timestamp=datetime.now()
            )
    
    async def _generate_persona(
        self,
        persona_type: PersonaType,
        scenario: TestScenario
    ) -> Dict[str, Any]:
        """Generate user persona for testing"""
        
        persona_traits = {
            PersonaType.HAPPY_PATH: {
                "communication_style": "clear and direct",
                "patience_level": "high",
                "technical_knowledge": "average",
                "cooperation": "very cooperative"
            },
            PersonaType.CONFUSED: {
                "communication_style": "unclear and rambling",
                "patience_level": "medium",
                "technical_knowledge": "low",
                "cooperation": "needs guidance"
            },
            PersonaType.TECHNICAL: {
                "communication_style": "precise and technical",
                "patience_level": "low",
                "technical_knowledge": "expert",
                "cooperation": "expects efficiency"
            },
            PersonaType.ADVERSARIAL: {
                "communication_style": "challenging and skeptical",
                "patience_level": "very low",
                "technical_knowledge": "varies",
                "cooperation": "tests boundaries"
            },
            PersonaType.IMPATIENT: {
                "communication_style": "brief and urgent",
                "patience_level": "very low",
                "technical_knowledge": "average",
                "cooperation": "wants quick results"
            }
        }
        
        base_traits = persona_traits.get(
            persona_type,
            persona_traits[PersonaType.HAPPY_PATH]
        )
        
        return {
            "type": persona_type.value,
            "traits": base_traits,
            "background": f"User testing {scenario.description}",
            "goals": scenario.success_criteria.get("user_goals", ["Complete the task"]),
            "frustrations": self._generate_frustrations(persona_type)
        }
    
    def _generate_frustrations(self, persona_type: PersonaType) -> List[str]:
        """Generate persona-specific frustrations"""
        
        frustration_map = {
            PersonaType.CONFUSED: ["Complex instructions", "Technical jargon"],
            PersonaType.IMPATIENT: ["Slow responses", "Multiple steps"],
            PersonaType.ADVERSARIAL: ["Being told what to do", "Limited options"],
            PersonaType.TECHNICAL: ["Simplified explanations", "Lack of details"]
        }
        
        return frustration_map.get(persona_type, [])
    
    async def _generate_user_message(
        self,
        step: Dict[str, Any],
        persona: Dict[str, Any],
        conversation_history: List[Dict[str, Any]]
    ) -> str:
        """Generate user message based on persona and step"""
        
        prompt = f"""Generate a user message for this test step:
        
        Step: {json.dumps(step)}
        Persona: {json.dumps(persona)}
        Conversation so far: {len(conversation_history)} messages
        
        The message should:
        - Match the persona's communication style: {persona['traits']['communication_style']}
        - Express the intent: {step['user_intent']}
        - Be realistic and natural
        
        Message:"""
        
        try:
            response = await self.llm.agenerate([prompt])
            return response.generations[0][0].text.strip()
        except Exception as e:
            logger.error(f"Failed to generate user message: {e}")
            # Fallback message
            return self._get_fallback_message(step, persona)
    
    def _get_fallback_message(self, step: Dict[str, Any], persona: Dict[str, Any]) -> str:
        """Get fallback message if generation fails"""
        
        intent = step.get("user_intent", "continue")
        persona_type = persona["type"]
        
        fallback_messages = {
            PersonaType.HAPPY_PATH.value: {
                "Initial request": "Hi, I need help with something",
                "Provide details": "Here are the details you asked for",
                "Confirm action": "Yes, please go ahead"
            },
            PersonaType.CONFUSED.value: {
                "Initial request": "Um, I'm not sure how to explain this but I need help",
                "Provide details": "I think this is what you need? Not sure",
                "Confirm action": "Wait, what exactly will happen?"
            },
            PersonaType.IMPATIENT.value: {
                "Initial request": "Need this done ASAP",
                "Provide details": "Here. Hurry up.",
                "Confirm action": "Yes yes, just do it"
            }
        }
        
        persona_messages = fallback_messages.get(
            persona_type,
            fallback_messages[PersonaType.HAPPY_PATH.value]
        )
        
        return persona_messages.get(intent, "Continue please")
    
    async def _evaluate_step(
        self,
        step: Dict[str, Any],
        response: str,
        expected_behaviors: List[str],
        response_time: float
    ) -> Dict[str, Any]:
        """Evaluate a single conversation step"""
        
        issues = []
        
        # Check response time
        max_time = step.get("max_response_time_ms", 5000)
        if response_time > max_time:
            issues.append(f"Response time ({response_time}ms) exceeded limit ({max_time}ms)")
        
        # Check expected response type
        expected_type = step.get("expected_response_type", "")
        if expected_type and not self._matches_response_type(response, expected_type):
            issues.append(f"Response type mismatch. Expected: {expected_type}")
        
        # Check for errors
        error_indicators = ["error", "exception", "failed", "unable to"]
        for indicator in error_indicators:
            if indicator.lower() in response.lower():
                issues.append(f"Potential error detected: '{indicator}' in response")
        
        return {
            "success": len(issues) == 0,
            "issues": issues,
            "response_time_ok": response_time <= max_time,
            "should_stop": "critical" in " ".join(issues).lower()
        }
    
    def _matches_response_type(self, response: str, expected_type: str) -> bool:
        """Check if response matches expected type"""
        
        type_indicators = {
            "acknowledgment": ["understand", "got it", "sure", "okay", "will help"],
            "processing": ["working on", "processing", "analyzing", "checking"],
            "completion": ["done", "completed", "finished", "here is", "successfully"],
            "question": ["?", "what", "which", "how", "please provide"],
            "confirmation": ["confirm", "is this correct", "please verify"]
        }
        
        indicators = type_indicators.get(expected_type.lower(), [])
        response_lower = response.lower()
        
        return any(indicator in response_lower for indicator in indicators)
    
    async def _final_evaluation(
        self,
        scenario: TestScenario,
        conversation_log: List[Dict[str, Any]],
        response_times: List[float],
        total_time: float
    ) -> Dict[str, Any]:
        """Perform final evaluation of test execution"""
        
        # Check success criteria
        criteria = scenario.success_criteria
        criteria_met = {}
        
        # Check must_mention items
        full_conversation = " ".join([
            log["assistant"] for log in conversation_log
        ])
        
        for item in criteria.get("must_mention", []):
            criteria_met[f"mentions_{item}"] = item.lower() in full_conversation.lower()
        
        # Check must_not_do items
        for item in criteria.get("must_not_do", []):
            criteria_met[f"avoids_{item}"] = item.lower() not in full_conversation.lower()
        
        # Performance criteria
        perf_criteria = criteria.get("performance", {})
        if "max_response_time_ms" in perf_criteria:
            criteria_met["performance_ok"] = all(
                t <= perf_criteria["max_response_time_ms"] for t in response_times
            )
        
        # Overall success
        success = all(criteria_met.values()) if criteria_met else True
        
        return {
            "success": success,
            "criteria_results": criteria_met,
            "completion_rate": len(conversation_log) / len(scenario.conversation_plan),
            "average_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "total_time": total_time,
            "quality_score": self._calculate_quality_score(conversation_log, criteria_met)
        }
    
    def _calculate_quality_score(
        self,
        conversation_log: List[Dict[str, Any]],
        criteria_met: Dict[str, bool]
    ) -> float:
        """Calculate overall quality score (0-100)"""
        
        # Base score from criteria
        criteria_score = (sum(criteria_met.values()) / len(criteria_met) * 50) if criteria_met else 50
        
        # Response quality (simplified - would use LLM in production)
        response_lengths = [len(log["assistant"]) for log in conversation_log]
        avg_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0
        
        # Penalize very short or very long responses
        length_score = 50
        if avg_length < 20:
            length_score = 30
        elif avg_length > 500:
            length_score = 40
        
        return criteria_score + (length_score * 0.5)
    
    async def _generate_improvement_suggestions(
        self,
        scenario: TestScenario,
        conversation_log: List[Dict[str, Any]],
        evaluation: Dict[str, Any]
    ) -> List[str]:
        """Generate suggestions for workflow improvement"""
        
        suggestions = []
        
        # Performance suggestions
        if evaluation.get("average_response_time", 0) > 3000:
            suggestions.append("Consider optimizing agent response times - average exceeds 3 seconds")
        
        # Completion suggestions
        if evaluation.get("completion_rate", 1) < 1:
            suggestions.append("Workflow did not complete all expected steps - investigate blocking issues")
        
        # Quality suggestions
        if evaluation.get("quality_score", 100) < 70:
            suggestions.append("Response quality could be improved - consider refining agent instructions")
        
        # Criteria-based suggestions
        criteria_results = evaluation.get("criteria_results", {})
        for criterion, met in criteria_results.items():
            if not met:
                if criterion.startswith("mentions_"):
                    item = criterion.replace("mentions_", "")
                    suggestions.append(f"Ensure agents mention '{item}' when relevant")
                elif criterion.startswith("avoids_"):
                    item = criterion.replace("avoids_", "")
                    suggestions.append(f"Agents should avoid '{item}' in responses")
        
        return suggestions
    
    # Batch testing methods
    
    async def run_test_suite(
        self,
        workflow_id: str,
        scenarios: List[TestScenario],
        parallel: bool = True,
        max_concurrent: int = 3
    ) -> List[TestResult]:
        """Run multiple test scenarios
        
        Args:
            workflow_id: Workflow to test
            scenarios: List of test scenarios
            parallel: Run tests in parallel
            max_concurrent: Max concurrent tests
        """
        
        if parallel:
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def run_with_limit(scenario):
                async with semaphore:
                    return await self.simulate_scenario(workflow_id, scenario)
            
            tasks = [run_with_limit(scenario) for scenario in scenarios]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert exceptions to failed results
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    final_results.append(TestResult(
                        scenario_id=scenarios[i].id,
                        success=False,
                        conversation=[],
                        evaluation={"error": str(result)},
                        performance_metrics={},
                        issues_found=[f"Test execution error: {str(result)}"],
                        suggestions=[],
                        timestamp=datetime.now()
                    ))
                else:
                    final_results.append(result)
            
            return final_results
        else:
            # Sequential execution
            results = []
            for scenario in scenarios:
                result = await self.simulate_scenario(workflow_id, scenario)
                results.append(result)
            return results
    
    def generate_test_report(self, results: List[TestResult]) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r.success)
        
        # Aggregate metrics
        all_response_times = []
        all_issues = []
        all_suggestions = set()
        
        for result in results:
            if result.performance_metrics.get("average_response_time_ms"):
                all_response_times.append(result.performance_metrics["average_response_time_ms"])
            all_issues.extend(result.issues_found)
            all_suggestions.update(result.suggestions)
        
        # Issue categories
        issue_categories = {}
        for issue in all_issues:
            category = self._categorize_issue(issue)
            issue_categories[category] = issue_categories.get(category, 0) + 1
        
        return {
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "test_date": datetime.now().isoformat()
            },
            "performance": {
                "average_response_time_ms": sum(all_response_times) / len(all_response_times) if all_response_times else 0,
                "max_response_time_ms": max(all_response_times) if all_response_times else 0,
                "min_response_time_ms": min(all_response_times) if all_response_times else 0
            },
            "issues": {
                "total_issues": len(all_issues),
                "by_category": issue_categories,
                "critical_issues": [i for i in all_issues if "critical" in i.lower()]
            },
            "suggestions": list(all_suggestions),
            "test_details": [
                {
                    "scenario": r.scenario_id,
                    "success": r.success,
                    "issues": r.issues_found,
                    "quality_score": r.evaluation.get("quality_score", 0)
                }
                for r in results
            ]
        }
    
    def _categorize_issue(self, issue: str) -> str:
        """Categorize issue for reporting"""
        
        issue_lower = issue.lower()
        
        if "response time" in issue_lower:
            return "performance"
        elif "error" in issue_lower or "exception" in issue_lower:
            return "errors"
        elif "mismatch" in issue_lower:
            return "behavior"
        else:
            return "other"


# Testing Playground UI Component
class ROWBOATTestingPlayground:
    """Interactive testing UI with natural language controls"""
    
    template = """
    import React, { useState, useEffect } from 'react';
    import { Card, Button, Select, Progress, Alert, Tabs } from '@/components/ui';
    
    const ROWBOATTestingPlayground = ({ workflowId }) => {
        const [testMode, setTestMode] = useState('natural');
        const [scenarios, setScenarios] = useState([]);
        const [activeTests, setActiveTests] = useState([]);
        const [results, setResults] = useState([]);
        const [report, setReport] = useState(null);
        
        // Natural language test generation
        const generateTestsFromDescription = async (description) => {
            const response = await api.testing.generateTestSuite({
                workflow_id: workflowId,
                description: description,
                auto_focus: true  // AI determines focus areas
            });
            setScenarios(response.scenarios);
        };
        
        // Quick test presets
        const quickTests = [
            {
                name: "Happy Path",
                description: "Test normal user interactions",
                persona: "happy_path",
                count: 3
            },
            {
                name: "Edge Cases", 
                description: "Test boundary conditions and errors",
                persona: "adversarial",
                count: 5
            },
            {
                name: "Stress Test",
                description: "Test with impatient and confused users",
                persona: "mixed",
                count: 10
            }
        ];
        
        const runQuickTest = async (preset) => {
            const scenarios = await api.testing.generateTestSuite({
                workflow_id: workflowId,
                description: preset.description,
                num_scenarios: preset.count
            });
            
            const results = await api.testing.runTestSuite({
                workflow_id: workflowId,
                scenarios: scenarios,
                persona_type: preset.persona,
                parallel: true
            });
            
            setResults(prev => [...prev, ...results]);
            updateReport(results);
        };
        
        return (
            <div className="testing-playground">
                <div className="header">
                    <h2>ROWBOAT Testing Playground</h2>
                    <Select 
                        value={testMode}
                        onChange={setTestMode}
                        options={[
                            { value: 'natural', label: 'Natural Language' },
                            { value: 'quick', label: 'Quick Tests' },
                            { value: 'manual', label: 'Manual Testing' }
                        ]}
                    />
                </div>
                
                {testMode === 'natural' && (
                    <NaturalLanguageTestGenerator
                        onGenerate={generateTestsFromDescription}
                        scenarios={scenarios}
                        onRunTests={runGeneratedTests}
                    />
                )}
                
                {testMode === 'quick' && (
                    <div className="quick-tests">
                        {quickTests.map((preset, idx) => (
                            <Card key={idx} className="test-preset">
                                <h3>{preset.name}</h3>
                                <p>{preset.description}</p>
                                <Button 
                                    onClick={() => runQuickTest(preset)}
                                    variant="primary"
                                >
                                    Run {preset.count} Tests
                                </Button>
                            </Card>
                        ))}
                    </div>
                )}
                
                {testMode === 'manual' && (
                    <ManualTestInterface 
                        workflowId={workflowId}
                        showDebugInfo={true}
                    />
                )}
                
                {activeTests.length > 0 && (
                    <div className="active-tests">
                        <h3>Running Tests</h3>
                        {activeTests.map((test, idx) => (
                            <TestProgress key={idx} test={test} />
                        ))}
                    </div>
                )}
                
                {results.length > 0 && (
                    <TestResults 
                        results={results}
                        onViewDetails={viewTestDetails}
                        onRerun={rerunTest}
                    />
                )}
                
                {report && (
                    <TestReport 
                        report={report}
                        onExport={exportReport}
                    />
                )}
            </div>
        );
    };
    
    const NaturalLanguageTestGenerator = ({ onGenerate, scenarios, onRunTests }) => {
        const [description, setDescription] = useState('');
        const [generating, setGenerating] = useState(false);
        
        const handleGenerate = async () => {
            setGenerating(true);
            await onGenerate(description);
            setGenerating(false);
        };
        
        return (
            <div className="natural-test-generator">
                <textarea
                    placeholder="Describe what you want to test... e.g., 'Test the refund process with angry customers who have high-value orders'"
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    rows={4}
                />
                <Button 
                    onClick={handleGenerate}
                    disabled={!description || generating}
                    variant="primary"
                >
                    {generating ? 'Generating Tests...' : 'Generate Test Scenarios'}
                </Button>
                
                {scenarios.length > 0 && (
                    <div className="generated-scenarios">
                        <h3>Generated Scenarios ({scenarios.length})</h3>
                        {scenarios.map((scenario, idx) => (
                            <ScenarioCard 
                                key={idx}
                                scenario={scenario}
                                onRun={() => onRunTests([scenario])}
                            />
                        ))}
                        <Button 
                            onClick={() => onRunTests(scenarios)}
                            variant="primary"
                            size="large"
                        >
                            Run All Tests
                        </Button>
                    </div>
                )}
            </div>
        );
    };
    """


# Quick usage functions

async def quick_test(workflow_id: str, description: str = "Test basic functionality") -> Dict[str, Any]:
    """Quick test with natural language description
    
    Example:
        result = await quick_test("wf_123", "Test refund process with edge cases")
    """
    simulator = ROWBOATTestSimulator()
    scenarios = await simulator.generate_test_suite(description, num_scenarios=3)
    results = await simulator.run_test_suite(workflow_id, scenarios)
    return simulator.generate_test_report(results)


async def stress_test(workflow_id: str, num_users: int = 10) -> Dict[str, Any]:
    """Run stress test with multiple personas
    
    Example:
        report = await stress_test("wf_123", num_users=50)
    """
    simulator = ROWBOATTestSimulator()
    
    # Generate diverse scenarios
    scenarios = await simulator.generate_test_suite(
        "Comprehensive stress test with various user types",
        focus_areas=["performance", "error handling", "concurrent users"],
        num_scenarios=num_users
    )
    
    # Run with different personas
    persona_types = list(PersonaType)
    results = []
    
    for i, scenario in enumerate(scenarios):
        persona = persona_types[i % len(persona_types)]
        result = await simulator.simulate_scenario(workflow_id, scenario, persona)
        results.append(result)
    
    return simulator.generate_test_report(results)