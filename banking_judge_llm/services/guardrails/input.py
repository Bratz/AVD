from typing import List, Dict
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import logging
import json
from config.settings import Settings
from services.audit.file_logger import FileLogger
from utils.exceptions import GuardrailViolationError
from utils.websocket import WebSocketManager

class InputGuardrail:
    """Guardrail for input validation and PII detection."""

    def __init__(self):
        self.settings = Settings()
        self.logger = logging.getLogger(__name__)
        self.audit_logger = FileLogger()
        self.websocket_manager = WebSocketManager()
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.blocked_phrases = self._load_blocked_phrases()

    def _load_blocked_phrases(self) -> List[str]:
        """Load blocked phrases from file."""
        try:
            with open(self.settings.blocked_phrases_path, "r") as f:
                phrases = [line.strip().lower() for line in f if line.strip()]
            self.logger.info(f"Loaded {len(phrases)} blocked phrases")
            return phrases
        except Exception as e:
            self.logger.error(f"Failed to load blocked phrases: {str(e)}")
            self.audit_logger.log_event(
                event_type="blocked_phrases_load_error",
                details={"error": str(e)}
            )
            raise GuardrailViolationError(f"Failed to load blocked phrases: {str(e)}")

    def process(self, input_text: str) -> Dict:
        """Process input for PII and blocked phrases, return compliance result."""
        try:
            # Detect PII
            pii_entities = self.analyzer.analyze(
                text=input_text,
                entities=["PHONE_NUMBER", "IN_AADHAAR", "IN_PAN", "EMAIL_ADDRESS", "CREDIT_CARD"],
                language="en"
            )
            pii_detected = [
                {
                    "entity_type": entity.entity_type,
                    "start": entity.start,
                    "end": entity.end,
                    "score": entity.score
                }
                for entity in pii_entities
            ]

            # Anonymize PII
            sanitized_text = self.anonymizer.anonymize(
                text=input_text,
                entities=[entity.entity_type for entity in pii_entities],
                language="en"
            ).text

            # Check for blocked phrases
            input_lower = input_text.lower()
            blocked_phrases_found = [
                phrase for phrase in self.blocked_phrases if phrase in input_lower
            ]

            # Determine compliance
            is_compliant = not (pii_detected or blocked_phrases_found)

            # Log results
            details = {
                "input_text": input_text,
                "sanitized_text": sanitized_text,
                "pii_detected": pii_detected,
                "blocked_phrases_found": blocked_phrases_found,
                "compliant": is_compliant
            }
            self.audit_logger.log_event(
                event_type="input_guardrail_check",
                details=details
            )

            # Notify via WebSocket if non-compliant
            if not is_compliant:
                asyncio.create_task(
                    self.websocket_manager.send_progress(
                        session_id="guardrail_session",
                        message={
                            "stage": "input_guardrail_violation",
                            "details": details,
                            "timestamp": str(logging.time())
                        }
                    )
                )

            return {
                "sanitized_input": sanitized_text,
                "pii_detected": pii_detected,
                "blocked_phrases_found": blocked_phrases_found,
                "input_compliant": is_compliant
            }

        except Exception as e:
            self.logger.error(f"Input guardrail processing failed: {str(e)}")
            self.audit_logger.log_event(
                event_type="input_guardrail_error",
                details={"input_text": input_text, "error": str(e)}
            )
            raise GuardrailViolationError(f"Input processing failed: {str(e)}")