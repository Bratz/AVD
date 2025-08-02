import logging
from typing import List, Optional
from langchain_ollama import OllamaLLM
from cachetools import TTLCache
from schemas.models import DecisionRequest, DecisionResponse
from services.audit.file_logger import FileLogger
from utils.exceptions import JudgeLLMError

class JudgeLLM:
    """LLM for evaluating banking queries with guardrails."""

    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434"):
        self.logger = logging.getLogger(__name__)
        self.audit_logger = FileLogger()
        self.llm = OllamaLLM(model=model_name, base_url=base_url)
        self.cache = TTLCache(maxsize=1000, ttl=3600)
        self.logger.info(f"Initialized JudgeLLM with model {model_name}")

    async def evaluate_async(self, request: DecisionRequest) -> DecisionResponse:
        """Evaluate a banking query with guardrails and caching."""
        from services.guardrails.unified import UnifiedGuardrail  # Moved import here

        try:
            cache_key = f"{request.customer_id}:{request.query}"
            if cache_key in self.cache:
                self.logger.info(f"Cache hit for {cache_key}")
                return self.cache[cache_key]

            # Initialize guardrail
            guardrail = UnifiedGuardrail()

            # Process input through guardrails
            guardrail_result = await guardrail.process(request.query, decisions=[])
            if not guardrail_result.input_compliant:
                self.logger.warning(f"Guardrail violation for input: {request.query}")
                self.audit_logger.log_event(
                    event_type="guardrail_violation",
                    details={"customer_id": request.customer_id, "query": request.query, "guardrail_result": guardrail_result.dict()}
                )
                raise JudgeLLMError("Input failed guardrail checks")

            # Generate prompt
            prompt = self._generate_prompt(request)
            self.logger.debug(f"Generated prompt: {prompt}")

            # Call LLM
            response = await self.llm.agenerate([prompt])
            verdict, text, confidence = self._parse_response(response)

            # Create response
            decision_response = DecisionResponse(
                customer_id=request.customer_id,
                verdict=verdict,
                text=text,
                confidence=confidence
            )

            # Cache result
            self.cache[cache_key] = decision_response
            self.logger.info(f"Cached result for {cache_key}")

            # Log decision
            self.audit_logger.log_event(
                event_type="judge_decision",
                details={
                    "customer_id": request.customer_id,
                    "query": request.query,
                    "verdict": verdict,
                    "confidence": confidence
                }
            )

            return decision_response

        except Exception as e:
            self.logger.error(f"Error evaluating query: {str(e)}")
            self.audit_logger.log_event(
                event_type="judge_error",
                details={"customer_id": request.customer_id, "query": request.query, "error": str(e)}
            )
            raise JudgeLLMError(f"Failed to evaluate query: {str(e)}")

    def _generate_prompt(self, request: DecisionRequest) -> str:
        """Generate a prompt for the LLM based on the request."""
        context_str = "\n".join([f"{k}: {v}" for k, v in request.context.items()])
        return f"""
        You are a banking compliance assistant. Evaluate the following query for compliance with RBI regulations.
        Customer ID: {request.customer_id}
        Query: {request.query}
        Context:
        {context_str}
        Provide a verdict (1 for approved, 0 for denied), an explanation, and a confidence score (0.0 to 1.0).
        Format: verdict: <int>, text: <str>, confidence: <float>
        """

    def _parse_response(self, response: str) -> tuple[int, str, float]:
        """Parse the LLM response into verdict, text, and confidence."""
        try:
            lines = response.strip().split("\n")
            verdict_line = next((line for line in lines if line.startswith("verdict:")), "verdict: 0")
            text_line = next((line for line in lines if line.startswith("text:")), "text: No response")
            confidence_line = next((line for line in lines if line.startswith("confidence:")), "confidence: 0.5")

            verdict = int(verdict_line.split(":")[1].strip())
            text = text_line.split(":", 1)[1].strip()
            confidence = float(confidence_line.split(":")[1].strip())

            if verdict not in [0, 1] or not 0.0 <= confidence <= 1.0:
                raise ValueError("Invalid verdict or confidence")

            return verdict, text, confidence
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {str(e)}")
            raise JudgeLLMError(f"Invalid LLM response format: {str(e)}")