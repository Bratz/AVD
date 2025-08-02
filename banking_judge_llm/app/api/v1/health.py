from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer
from app.middleware.auth import verify_jwt
from core.llm.judge import JudgeLLM
from core.llm.retriever import Retriever
from utils.websocket import WebSocketManager
from services.audit.file_logger import FileLogger
from config.settings import Settings
from utils.exceptions import BankingJudgeError
import logging
import httpx

router = APIRouter(prefix="/v1/health", tags=["health"])
security = HTTPBearer()
settings = Settings()
logger = logging.getLogger(__name__)
audit_logger = FileLogger()

@router.get("/status")
async def health_check(token: dict = Depends(security)):
    """Check the health of the API and its components."""
    try:
        # Verify JWT
        payload = verify_jwt(token)
        user_id = payload.get("sub", "unknown")

        # Initialize components
        judge_llm = JudgeLLM()
        retriever = Retriever()
        websocket_manager = WebSocketManager()

        # Check Ollama connectivity
        ollama_status = await check_ollama()
        # Check FAISS index
        faiss_status = check_faiss(retriever)
        # Check WebSocket
        websocket_status = await check_websocket(websocket_manager)

        # Compile health status
        status = {
            "status": "healthy" if all([ollama_status["healthy"], faiss_status["healthy"], websocket_status["healthy"]]) else "unhealthy",
            "components": {
                "api": {"healthy": True, "message": "API is running"},
                "ollama": ollama_status,
                "faiss": faiss_status,
                "websocket": websocket_status
            },
            "message": "Banking Judge LLM API health check completed"
        }

        # Log health check
        audit_logger.log_event(
            event_type="health_check",
            details={"user_id": user_id, "status": status["status"], "components": status["components"]}
        )

        return status

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        audit_logger.log_event(
            event_type="health_check_error",
            details={"user_id": user_id, "error": str(e)}
        )
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

async def check_ollama() -> dict:
    """Check connectivity to Ollama server."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            response.raise_for_status()
            return {"healthy": True, "message": "Ollama server is reachable"}
    except Exception as e:
        logger.warning(f"Ollama health check failed: {str(e)}")
        return {"healthy": False, "message": f"Ollama server error: {str(e)}"}

def check_faiss(retriever: Retriever) -> dict:
    """Check FAISS index availability."""
    try:
        # Perform a simple query to test FAISS
        retriever.vectorstore.similarity_search("test", k=1)
        return {"healthy": True, "message": "FAISS index is accessible"}
    except Exception as e:
        logger.warning(f"FAISS health check failed: {str(e)}")
        return {"healthy": False, "message": f"FAISS index error: {str(e)}"}

async def check_websocket(websocket_manager: WebSocketManager) -> dict:
    """Check WebSocket manager initialization."""
    try:
        await websocket_manager.initialize()
        return {"healthy": True, "message": "WebSocket manager is operational"}
    except Exception as e:
        logger.warning(f"WebSocket health check failed: {str(e)}")
        return {"healthy": False, "message": f"WebSocket manager error: {str(e)}"}