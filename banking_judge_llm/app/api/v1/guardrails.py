from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer
import pandas as pd
from schemas.models import GuardrailRequest, GuardrailResult, DecisionRequest, DecisionResponse
from services.guardrails.unified import UnifiedGuardrail
from core.llm.judge import JudgeLLM
from app.middleware.auth import verify_jwt
from app.middleware.rate_limiter import rate_limit
from utils.websocket import WebSocketManager

router = APIRouter(prefix="/v1/guardrails", tags=["guardrails"])
security = HTTPBearer()

@router.post("/process", response_model=GuardrailResult)
@rate_limit(limit=100, window=60)  # 100 requests per minute
async def process_guardrails(request: GuardrailRequest, token: str = Depends(security)):
    """
    Process input and output through guardrails, returning sanitized results and metrics.
    """
    try:
        verify_jwt(token)  # JWT validation
        guardrail = UnifiedGuardrail()
        decisions_df = pd.DataFrame(request.decisions)
        result = guardrail.process(request.input_text, decisions_df)
        
        await WebSocketManager().send_progress(
            session_id="guardrail_session",
            message={
                "stage": "guardrail_processed",
                "input_compliant": result.input_compliant,
                "output_compliant": result.output_compliant
            }
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Guardrail processing failed: {str(e)}")

@router.post("/evaluate", response_model=DecisionResponse)
@rate_limit(limit=50, window=60)  # 50 requests per minute
async def evaluate_with_guardrails(request: DecisionRequest, token: str = Depends(security)):
    """
    Evaluate a banking decision with guardrail checks.
    """
    try:
        verify_jwt(token)
        judge = JudgeLLM()
        result = await judge.evaluate_async(request)
        
        await WebSocketManager().send_progress(
            session_id=f"decision_{request.customer_id}",
            message={
                "stage": "decision_processed",
                "verdict": result.verdict,
                "confidence": result.confidence
            }
        )
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")