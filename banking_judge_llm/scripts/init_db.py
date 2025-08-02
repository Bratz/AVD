import asyncio
import logging
from typing import List, Dict
from core.llm.retriever import Retriever
from config.settings import Settings
from services.audit.file_logger import FileLogger
from utils.websocket import WebSocketManager

async def initialize_rbi_regulations():
    logger = logging.getLogger(__name__)
    settings = Settings()
    audit_logger = FileLogger()
    websocket_manager = WebSocketManager()
    retriever = Retriever()

    regulations = [
        {
            "text": (
                "All Payment System Providers must store data relating to payment systems only in India, "
                "as per RBI circular DPSS.CO.OD.No 2785/06.08.005/2017-18 dated April 06, 2018. "
                "This includes customer data (Name, Mobile Number, Email, Aadhaar Number, PAN number), "
                "payment credentials (OTP, PIN, Passwords), and transaction data."
            ),
            "metadata": {
                "id": "RBI-2018-001",
                "source": "https://www.rbi.org.in/scripts/NotificationUser.aspx?Id=11244",
                "date": "2018-04-06",
                "category": "Payment Systems"
            }
        },
        {
            "text": (
                "The Payment and Settlement Systems Act, 2007 regulates payment systems in India, "
                "ensuring secure and efficient operations of banks, payment gateways, and intermediaries."
            ),
            "metadata": {
                "id": "RBI-2007-001",
                "source": "https://www.india.gov.in",
                "date": "2007-12-20",
                "category": "Payment Systems"
            }
        },
        {
            "text": (
                "RBI Payments Regulatory Board Regulations, 2025, establish guidelines for regulatory "
                "approvals, licenses, and filings across India’s financial sector."
            ),
            "metadata": {
                "id": "RBI-2025-001",
                "source": "https://t.co/FOiUbNKuby",
                "date": "2025-05-22",
                "category": "Regulatory Framework"
            }
        },
        {
            "text": (
                "Revised Directions on Investment by Regulated Entities in Alternative Investment Funds, "
                "draft released on May 19, 2025, for public comments."
            ),
            "metadata": {
                "id": "RBI-2025-002",
                "source": "https://t.co/8uWyphXitL",
                "date": "2025-05-19",
                "category": "Investment Regulations"
            }
        }
    ]

    try:
        texts = [reg["text"] for reg in regulations]
        metadatas = [reg["metadata"] for reg in regulations]

        await retriever.add_regulations(texts, metadatas)
        logger.info(f"Initialized FAISS index with {len(texts)} RBI regulations")
        audit_logger.log_event(
            event_type="init_rbi_regulations",
            details={"count": len(texts), "ids": [meta["id"] for meta in metadatas]}
        )

        await websocket_manager.send_progress(
            session_id="init_db_session",
            message={
                "stage": "rbi_regulations_initialized",
                "count": len(texts)
            }
        )

    except Exception as e:
        logger.error(f"Failed to initialize RBI regulations: {str(e)}")
        audit_logger.log_event(
            event_type="init_rbi_regulations_error",
            details={"error": str(e)}
        )
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(initialize_rbi_regulations())