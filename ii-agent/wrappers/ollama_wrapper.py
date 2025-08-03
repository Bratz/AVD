import httpx
import json
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

class ModelType(Enum):
    INTENT = "intent"
    RESPONSE = "response" 
    EMBEDDING = "embedding"
    REASONING = "reasoning"

@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    intent_model: str = "llama3.1:8b"
    response_model: str = "mistral:latest"
    embedding_model: str = "nomic-embed-text:latest"
    reasoning_model: str = "llama3.1:8b"  # For II-Agent reasoning
    timeout: int = 300

class OllamaWrapper:
    """Wrapper for Ollama API integration with II-Agent framework."""
    
    def __init__(self, config: OllamaConfig = None):
        self.config = config or OllamaConfig()
        self.client = httpx.AsyncClient(timeout=self.config.timeout)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"OllamaWrapper initialized with models:")
        self.logger.info(f"  intent_model: {self.config.intent_model}")
        self.logger.info(f"  response_model: {self.config.response_model}")
        self.logger.info(f"  reasoning_model: {self.config.reasoning_model}")
    async def initialize(self):
        """Initialize and verify Ollama connection."""
        try:
            # Test connection
            response = await self.client.get(f"{self.config.base_url}/api/tags")
            response.raise_for_status()
            
            # Verify required models are available
            models_data = response.json()
            available_models = [model["name"] for model in models_data.get("models", [])]
            
            required_models = [
                self.config.intent_model,
                self.config.response_model, 
                self.config.embedding_model,
                self.config.reasoning_model
            ]
            
            missing_models = [model for model in required_models if model not in available_models]
            if missing_models:
                self.logger.warning(f"Missing models: {missing_models}")
                # Auto-pull missing models
                for model in missing_models:
                    await self._pull_model(model)
            
            self.logger.info("Ollama wrapper initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama wrapper: {e}")
            raise
            
    async def _pull_model(self, model_name: str):
        """Pull a model if not available."""
        try:
            self.logger.info(f"Pulling model: {model_name}")
            response = await self.client.post(
                f"{self.config.base_url}/api/pull",
                json={"name": model_name}
            )
            response.raise_for_status()
            self.logger.info(f"Successfully pulled model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to pull model {model_name}: {e}")
            
    async def generate_plan(self, goal: str, context: Dict[str, Any] = None) -> List[str]:
        """Generate a step-by-step plan for achieving a goal."""
        prompt = f"""
You are an AI planning agent for a banking system. Generate a detailed step-by-step plan to achieve the following goal:

Goal: {goal}

Context: {json.dumps(context or {}, indent=2)}

Instructions:
1. Break down the goal into specific, actionable steps
2. Consider the banking domain (customers, accounts, transactions, compliance)
3. Each step should be clear and executable
4. Return ONLY a JSON array of step strings, no other text

Example format: ["step1", "step2", "step3"]
"""
        
        try:
            response = await self._generate_completion(
                prompt, 
                model=self.config.reasoning_model,
                system="You are a banking domain expert and planning specialist."
            )
            
            # Parse JSON response
            plan_text = response.strip()
            if plan_text.startswith('```json'):
                plan_text = plan_text.split('```json')[1].split('```')[0]
            elif plan_text.startswith('```'):
                plan_text = plan_text.split('```')[1].split('```')[0]
                
            plan = json.loads(plan_text)
            
            if isinstance(plan, list):
                return plan
            else:
                self.logger.warning(f"Invalid plan format: {plan}")
                return ["analyze_goal", "execute_task", "validate_result"]
                
        except Exception as e:
            self.logger.error(f"Plan generation failed: {e}")
            return ["analyze_goal", "execute_task", "validate_result"]
            
    async def generate_reasoning(self, situation: str, context: Dict[str, Any] = None) -> str:
        """Generate reasoning for a given situation."""
        prompt = f"""
You are an AI reasoning agent in a banking system. Analyze the following situation and provide your reasoning:

Situation: {situation}

Context: {json.dumps(context or {}, indent=2)}

Provide clear, logical reasoning about:
1. What you observe in this situation
2. What this means in the banking context
3. What actions or conclusions follow from this analysis
4. Any risks or considerations to be aware of

Keep your response concise but thorough.
"""
        
        return await self._generate_completion(
            prompt,
            model=self.config.reasoning_model,
            system="You are a banking domain expert with strong analytical skills."
        )
        
    async def generate_reflection(self, execution_results: List[Any], goal: str) -> Dict[str, Any]:
        """Generate reflection on execution results."""
        prompt = f"""
You are an AI reflection agent for a banking system. Analyze the execution results and provide reflection:

Original Goal: {goal}

Execution Results: {json.dumps(execution_results, indent=2, default=str)}

Provide a reflection analysis covering:
1. Goal achievement assessment (complete/partial/failed)
2. Quality of results obtained
3. Effectiveness of the execution approach
4. Lessons learned or improvements for next time
5. Overall success rating (0-10)

Return your analysis as a JSON object with these keys:
{
  "goal_achievement": "complete|partial|failed",
  "result_quality": "high|medium|low", 
  "execution_effectiveness": "high|medium|low",
  "success_rating": 8,
  "lessons_learned": ["lesson1", "lesson2"],
  "recommendations": ["rec1", "rec2"],
  "summary": "Overall reflection summary"
}
"""
        
        try:
            response = await self._generate_completion(
                prompt,
                model=self.config.reasoning_model,
                system="You are an expert at analyzing and reflecting on task execution."
            )
            
            # Parse JSON response
            reflection_text = response.strip()
            if reflection_text.startswith('```json'):
                reflection_text = reflection_text.split('```json')[1].split('```')[0]
            elif reflection_text.startswith('```'):
                reflection_text = reflection_text.split('```')[1].split('```')[0]
                
            reflection = json.loads(reflection_text)
            return reflection
            
        except Exception as e:
            self.logger.error(f"Reflection generation failed: {e}")
            return {
                "goal_achievement": "partial",
                "result_quality": "medium",
                "execution_effectiveness": "medium", 
                "success_rating": 5,
                "lessons_learned": ["Reflection generation encountered errors"],
                "recommendations": ["Retry with simpler approach"],
                "summary": f"Reflection failed due to: {str(e)}"
            }
            
    async def detect_intent(self, user_input: str) -> Dict[str, Any]:
        """Detect user intent using your existing LLM service pattern."""
        prompt = f"""
Analyze this banking query and classify the intent:

Query: "{user_input}"

Classify into one of these categories:
- customer_profile: Customer information queries
- account_balance: Balance and account inquiries  
- transaction_history: Transaction-related queries
- financial_advice: Requests for financial guidance
- market_analysis: Market trends and analysis
- comprehensive_analysis: Complex multi-step analysis
- general_chat: General conversation

Return JSON format:
{
  "intent": "category_name",
  "confidence": 0.95,
  "entities": {
    "customer_id": "extracted_id_if_any",
    "account_type": "extracted_type_if_any"
  },
  "complexity": "simple|medium|complex"
}
"""
        
        try:
            response = await self._generate_completion(
                prompt,
                model=self.config.intent_model,
                system="You are an intent classification specialist for banking systems."
            )
            
            # Parse JSON response
            intent_text = response.strip()
            if intent_text.startswith('```json'):
                intent_text = intent_text.split('```json')[1].split('```')[0]
            elif intent_text.startswith('```'):
                intent_text = intent_text.split('```')[1].split('```')[0]
                
            intent_data = json.loads(intent_text)
            return intent_data
            
        except Exception as e:
            self.logger.error(f"Intent detection failed: {e}")
            return {
                "intent": "general_chat",
                "confidence": 0.5,
                "entities": {},
                "complexity": "simple"
            }
            
    async def generate_response(self, context: Dict[str, Any], format_instructions: str = None) -> str:
        """Generate natural language response from context."""
        format_part = f"\n\nFormat Instructions: {format_instructions}" if format_instructions else ""
        
        prompt = f"""
You are a helpful banking assistant. Generate a natural, professional response based on this context:

Context: {json.dumps(context, indent=2, default=str)}

Instructions:
1. Provide a clear, helpful response based on the context data
2. Use professional but friendly tone appropriate for banking
3. If there are errors in the context, acknowledge them appropriately
4. Include relevant details but keep the response concise
5. If data seems incomplete, mention what additional information might be helpful{format_part}
"""
        
        return await self._generate_completion(
            prompt,
            model=self.config.response_model,
            system="You are a professional banking customer service representative."
        )
        
    async def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for text using Ollama."""
        try:
            response = await self.client.post(
                f"{self.config.base_url}/api/embeddings",
                json={
                    "model": self.config.embedding_model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("embedding", [])
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return []
            
    async def _generate_completion(self, prompt: str, model: str, system: str = None) -> str:
        """Generate completion using Ollama chat API."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Log the request details
            self.logger.debug(f"Ollama request - Model: {model}, Base URL: {self.config.base_url}")
            self.logger.debug(f"Prompt preview: {prompt[:200]}...")
            
            # Make the request
            response = await self.client.post(
                f"{self.config.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False
                }
            )
            
            # Check response status
            if response.status_code != 200:
                self.logger.error(f"Ollama returned status {response.status_code}: {response.text}")
                raise Exception(f"Ollama API error: {response.status_code}")
            
            # Parse response
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse Ollama response: {response.text[:500]}")
                raise Exception(f"Invalid JSON from Ollama: {e}")
            
            # Extract content
            if "error" in data:
                self.logger.error(f"Ollama error: {data['error']}")
                raise Exception(f"Ollama error: {data['error']}")
                
            content = data.get("message", {}).get("content", "")
            
            if not content:
                self.logger.warning(f"Empty response from Ollama. Full response: {data}")
                # Return a default response for parameter extraction
                return "{}"
            
            return content
            
        except httpx.ConnectError:
            self.logger.error(f"Cannot connect to Ollama at {self.config.base_url}. Is Ollama running?")
            raise Exception("Cannot connect to Ollama. Ensure Ollama is running with 'ollama serve'")
        except httpx.TimeoutException:
            self.logger.error(f"Ollama request timed out after {self.config.timeout}s")
            raise Exception("Ollama request timed out")
        except Exception as e:
            self.logger.error(f"Completion generation failed: {type(e).__name__}: {e}")
            raise
            
    async def generate(self, messages, max_tokens=500, temperature=0.1, system_prompt=None, **kwargs):
        """Generate method compatible with LLMClient interface - ASYNC"""
        # Extract text from messages
        prompt = ""
        if isinstance(messages, list) and messages:
            for message_group in messages:
                if isinstance(message_group, list):
                    for msg in message_group:
                        if hasattr(msg, 'text'):
                            prompt += msg.text + "\n"
                        elif isinstance(msg, str):
                            prompt += msg + "\n"
                elif hasattr(message_group, 'text'):
                    prompt += message_group.text + "\n"
                elif isinstance(message_group, str):
                    prompt += message_group + "\n"
        
        try:
            # Since we're already async, just await directly
            response = await self._generate_completion(
                prompt=prompt.strip(),
                model=self.config.reasoning_model,
                system=system_prompt
            )
            
            # Return in expected format
            from src.ii_agent.llm.base import TextResult
            return [TextResult(text=response)], {}
            
        except Exception as e:
            self.logger.error(f"Generate method failed: {e}")
            from src.ii_agent.llm.base import TextResult
            return [TextResult(text='{"parameters": []}')], {"error": str(e)}
                
    async def _check_model_availability(self, model: str) -> bool:
        """Check if a model is available in Ollama"""
        try:
            response = await self.client.get(f"{self.config.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                available_models = [m["name"] for m in data.get("models", [])]
                return model in available_models
        except Exception as e:
            self.logger.error(f"Failed to check model availability: {e}")
        return False

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()