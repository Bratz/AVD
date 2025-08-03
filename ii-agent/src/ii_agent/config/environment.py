import os
from typing import Optional, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class EnvironmentConfig:
    """Centralized environment configuration"""
    
    _instance = None
    _env_vars: Dict[str, Any] = {}
    _loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._loaded:
            self._load_environment()
    
    def _load_environment(self):
        """Load environment variables from multiple sources"""
        
        # 1. Load from .env file if exists
        self._load_env_file()
        
        # 2. Load from system environment
        self._load_system_env()
        
        # 3. Load from parent process environment (for subprocess scenarios)
        self._load_parent_env()
        
        self._loaded = True
        logger.info(f"Loaded {len(self._env_vars)} environment variables")
    
    def _load_env_file(self):
        """Load from .env file"""
        env_files = [
            Path(".env"),
            Path(".env.local"),
            Path("../.env"),  # Parent directory
            Path(os.path.expanduser("~/.ii_agent/.env"))  # User directory
        ]
        
        for env_file in env_files:
            if env_file.exists():
                logger.info(f"Loading environment from {env_file}")
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, value = line.split("=", 1)
                            self._env_vars[key.strip()] = value.strip().strip('"\'')
    
    def _load_system_env(self):
        """Load from system environment"""
        for key, value in os.environ.items():
            self._env_vars[key] = value
    
    def _load_parent_env(self):
        """Load from parent process environment file if exists"""
        parent_env_file = Path("/tmp/ii_agent_env.json")
        if parent_env_file.exists():
            import json
            try:
                with open(parent_env_file) as f:
                    parent_env = json.load(f)
                    self._env_vars.update(parent_env)
                    logger.info(f"Loaded {len(parent_env)} variables from parent process")
            except Exception as e:
                logger.warning(f"Failed to load parent environment: {e}")
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable"""
        # First check our loaded variables
        if key in self._env_vars:
            return self._env_vars[key]
        
        # Fallback to os.getenv
        value = os.getenv(key, default)
        if value is not None:
            self._env_vars[key] = value
        
        return value
    
    def set(self, key: str, value: str):
        """Set environment variable"""
        self._env_vars[key] = value
        os.environ[key] = value
    
    def get_all(self) -> Dict[str, Any]:
        """Get all environment variables"""
        return self._env_vars.copy()
    
    def ensure_required(self, required_vars: list) -> bool:
        """Ensure required variables are present"""
        missing = []
        for var in required_vars:
            if not self.get(var):
                missing.append(var)
        
        if missing:
            logger.error(f"Missing required environment variables: {missing}")
            return False
        
        return True

# Global instance
env_config = EnvironmentConfig()
