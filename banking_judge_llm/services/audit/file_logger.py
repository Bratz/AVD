import logging
import json
from typing import Dict
from config.settings import Settings
from logging.handlers import RotatingFileHandler

class FileLogger:
    """Audit logger with local file rotation."""
    
    def __init__(self):
        self.settings = Settings()
        self.logger = logging.getLogger(__name__)
        handler = RotatingFileHandler(
            self.settings.audit_log_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_event(self, event_type: str, details: Dict):
        """Log an audit event."""
        self.logger.info(
            f"Event: {event_type}, Details: {json.dumps(details, default=str)}"
        )