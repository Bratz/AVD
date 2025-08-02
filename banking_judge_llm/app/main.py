from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from app.middleware.auth import verify_jwt
from app.middleware.rate_limiter import rate_limit
from app.api.v1.guardrails import router as guardrails_router
from app.api.v1.health import router as health_router
from utils.websocket import WebSocketManager
from config.settings import Settings
import logging

def create_app() -> FastAPI:
    app = FastAPI(
        title="Banking Judge LLM API",
        description="API for banking decision-making with guardrails",
        version="1.0.0"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    rate_limit(app,60)  # Add rate limiter middleware

    app.include_router(guardrails_router)
    app.include_router(health_router)

    websocket_manager = WebSocketManager()

    @app.websocket("/ws/progress/{session_id}")
    async def websocket_endpoint(websocket, session_id: str):
        await websocket_manager.connect(websocket, session_id)
        try:
            while True:
                await websocket.receive_text()
        except Exception as e:
            await websocket_manager.disconnect(websocket, session_id)

    @app.on_event("startup")
    async def startup_event():
        settings = Settings()
        logging.info("Starting Banking Judge LLM API")
        await websocket_manager.initialize()

    @app.on_event("shutdown")
    async def shutdown_event():
        logging.info("Shutting down Banking Judge LLM API")
        await websocket_manager.cleanup()

    return app

app = create_app()