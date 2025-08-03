"""
II-Agent Logging Configuration
Adapted from II-Agent repository patterns
"""
import logging
import sys
from typing import Optional

def setup_logging(level: str = "DEBUG", log_file: Optional[str] = None) -> None:
    """Setup logging configuration for II-Agent."""
    
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(console_handler)
    
    # Setup file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Setup II-Agent specific logger
    ii_agent_logger = logging.getLogger('ii_agent')
    ii_agent_logger.setLevel(numeric_level)
    
    logging.info("II-Agent logging configured")

def get_logger(name: str) -> logging.Logger:
    """Get a logger for II-Agent components."""
    return logging.getLogger(f"ii_agent.{name}")