import asyncio
import json
import logging
from typing import Dict, List
from datetime import datetime
from pathlib import Path
from config.settings import Settings
from utils.websocket import WebSocketManager
from utils.exceptions import BankingJudgeError
from services.audit.file_logger import FileLogger

class LogProcessor:
    """Process and summarize audit logs for compliance reporting."""

    def __init__(self):
        self.settings = Settings()
        self.logger = logging.getLogger(__name__)
        self.audit_logger = FileLogger()
        self.websocket_manager = WebSocketManager()
        self.log_path = Path(self.settings.audit_log_path)
        self.summary_path = Path(self.settings.audit_summary_path)

    async def process_logs(self, session_id: str = "log_processor_session") -> Dict:
        """Parse audit logs, generate summary, and send WebSocket updates."""
        try:
            self.logger.info("Starting log processing")
            await self.websocket_manager.send_progress(
                session_id=session_id,
                message={"stage": "log_processing_started", "timestamp": str(datetime.now())}
            )

            # Initialize summary
            summary = {
                "timestamp": str(datetime.now()),
                "total_events": 0,
                "event_types": {},
                "errors": [],
                "guardrail_violations": 0,
                "retrievals": 0,
                "rbi_compliance_events": 0
            }

            # Parse logs
            events = await self._parse_logs()
            summary["total_events"] = len(events)

            # Analyze events
            for event in events:
                event_type = event.get("event_type")
                details = event.get("details", {})

                # Update event type counts
                summary["event_types"][event_type] = summary["event_types"].get(event_type, 0) + 1

                # Track errors
                if "error" in event_type.lower():
                    summary["errors"].append({"type": event_type, "details": details})

                # Track guardrail violations
                if event_type == "regulatory_check" and not details.get("compliant", True):
                    summary["guardrail_violations"] += 1

                # Track retrievals
                if event_type == "retriever_query":
                    summary["retrievals"] += 1

                # Track RBI compliance events
                if "rbi" in event_type.lower() or "regulatory" in event_type.lower():
                    summary["rbi_compliance_events"] += 1

            # Save summary
            await self._save_summary(summary)

            # Log and notify completion
            self.logger.info(f"Processed {summary['total_events']} events")
            self.audit_logger.log_event(
                event_type="log_processing_completed",
                details={"summary": summary}
            )
            await self.websocket_manager.send_progress(
                session_id=session_id,
                message={
                    "stage": "log_processing_completed",
                    "total_events": summary["total_events"],
                    "guardrail_violations": summary["guardrail_violations"]
                }
            )

            return summary

        except Exception as e:
            self.logger.error(f"Log processing failed: {str(e)}")
            self.audit_logger.log_event(
                event_type="log_processing_error",
                details={"error": str(e)}
            )
            await self.websocket_manager.send_progress(
                session_id=session_id,
                message={"stage": "log_processing_failed", "error": str(e)}
            )
            raise BankingJudgeError(f"Log processing failed: {str(e)}")

    async def _parse_logs(self) -> List[Dict]:
        """Parse audit log file into structured events."""
        events = []
        if not self.log_path.exists():
            self.logger.warning(f"Log file not found: {self.log_path}")
            return events

        with open(self.log_path, "r") as f:
            for line in f:
                try:
                    # Extract event from log line
                    if "Event:" not in line:
                        continue
                    parts = line.split(" - INFO - Event: ")
                    if len(parts) < 2:
                        continue
                    event_data = parts[1].split(", Details: ")
                    event_type = event_data[0].strip()
                    details = json.loads(event_data[1].strip()) if len(event_data) > 1 else {}
                    events.append({"event_type": event_type, "details": details})
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse log line: {line.strip()} - {str(e)}")
                    continue

        return events

    async def _save_summary(self, summary: Dict):
        """Save summary to file."""
        try:
            self.summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            self.logger.info(f"Saved summary to {self.summary_path}")
        except Exception as e:
            self.logger.error(f"Failed to save summary: {str(e)}")
            raise BankingJudgeError(f"Failed to save summary: {str(e)}")

async def main():
    """Run the log processor."""
    logging.basicConfig(level=logging.INFO)
    processor = LogProcessor()
    summary = await processor.process_logs()
    print(f"Log Processing Summary:\n{json.dumps(summary, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())