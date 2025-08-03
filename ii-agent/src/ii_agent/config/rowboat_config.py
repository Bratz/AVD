# src/ii_agent/config/rowboat_config.py
"""ROWBOAT-specific configuration using the environment system"""

from src.ii_agent.config.environment import env_config
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class ROWBOATConfig:
    """ROWBOAT model and feature configuration"""
    
    def __init__(self):
        # Ensure ROWBOAT environment variables are loaded
        self._validate_config()
    
    def _validate_config(self):
        """Validate ROWBOAT configuration"""
        # Optional: Add required ROWBOAT vars
        # env_config.ensure_required(["CHUTES_API_KEY"])
        pass
    
    @property
    def copilot_model(self) -> str:
        """Model for workflow generation (copilot) - should output clean JSON"""
        # DeepSeek-R1 includes thinking tags, so use V3 for cleaner output
        default = "deepseek-ai/DeepSeek-V3"  # Better for JSON output
        
        # Allow override via environment
        model = env_config.get("ROWBOAT_COPILOT_MODEL", default)
        
        # Validate it exists in registry
        from src.ii_agent.llm.model_registry import ChutesModelRegistry
        model_key = ChutesModelRegistry.get_model_key_by_id(model)
        
        if not model_key:
            logger.warning(f"Copilot model {model} not found in registry, using default {default}")
            return default
        
        return model

    @property
    def default_agent_model(self) -> str:
        """Default model for agents"""
        default = "deepseek-ai/DeepSeek-V3"
        model = env_config.get("ROWBOAT_DEFAULT_AGENT_MODEL", default)
        
        # Validate
        from src.ii_agent.llm.model_registry import ChutesModelRegistry
        model_key = ChutesModelRegistry.get_model_key_by_id(model)
        
        if not model_key:
            logger.warning(f"Default agent model {model} not found in registry, using {default}")
            return default
        
        return model
    
    @property
    def hub_model(self) -> str:
        """Model for hub/orchestrator agents"""
        return env_config.get("ROWBOAT_HUB_MODEL", "deepseek-ai/DeepSeek-V3-0324")
    
    @property
    def info_model(self) -> str:
        """Model for info/Q&A agents"""
        return env_config.get("ROWBOAT_INFO_MODEL", "moonshotai/Kimi-K2-Instruct")
    
    @property
    def procedural_model(self) -> str:
        """Model for procedural/workflow agents"""
        return env_config.get("ROWBOAT_PROCEDURAL_MODEL", "Qwen/Qwen3")
    
    @property
    def internal_model(self) -> str:
        """Model for internal (non-user-facing) agents"""
        return env_config.get("ROWBOAT_INTERNAL_MODEL", "NousResearch/DeepHermes-3-Llama-3-Preview")
    
    @property
    def code_model(self) -> str:
        """Model for code-related agents"""
        return env_config.get("ROWBOAT_CODE_MODEL", "Qwen/Qwen2.5-Coder-Instruct")
    
    @property
    def fallback_models(self) -> List[str]:
        """Get fallback models from registry instead of hardcoding"""
        from src.ii_agent.llm.model_registry import ChutesModelRegistry
        
        # Check if user provided custom fallback models
        custom_fallbacks = env_config.get("ROWBOAT_FALLBACK_MODELS", "")
        if custom_fallbacks:
            return [m.strip() for m in custom_fallbacks.split(",") if m.strip()]
        
        # Otherwise, get intelligent fallbacks from registry
        model_key = ChutesModelRegistry.get_model_key_by_id(self.default_agent_model)
        if model_key:
            return ChutesModelRegistry.get_fallback_models(model_key)
        else:
            return ChutesModelRegistry.get_default_fallback_chain()
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration with intelligent fallbacks"""
        from src.ii_agent.llm.model_registry import ChutesModelRegistry, ModelCapability
        
        return {
            "copilot_model": self.copilot_model,
            "default_agent_model": self.default_agent_model,
            "hub_model": self.hub_model,
            "info_model": self.info_model,
            "procedural_model": self.procedural_model,
            "internal_model": self.internal_model,
            "code_model": self.code_model,
            "fallback_models": self.fallback_models,
            # Add capability-specific fallbacks
            "reasoning_fallbacks": ChutesModelRegistry.get_reasoning_fallback_chain(),
            "vision_fallbacks": ChutesModelRegistry.get_vision_fallback_chain(),
            "tool_calling_fallbacks": ChutesModelRegistry.get_tool_calling_fallback_chain(),
        }
    
    def get_model_for_role(self, role: str) -> str:
        """Get appropriate model for a specific role"""
        from src.ii_agent.llm.model_registry import ChutesModelRegistry
        
        # Check if we have a specific model configured for this role
        role_model_map = {
            "copilot": self.copilot_model,
            "hub": self.hub_model,
            "info": self.info_model,
            "procedural": self.procedural_model,
            "internal": self.internal_model,
            "code": self.code_model,
        }
        
        if role in role_model_map:
            return role_model_map[role]
        
        # Otherwise use registry recommendations
        return ChutesModelRegistry.get_model_for_role(role)
    
    def get_fallbacks_for_model(self, model_id: str, max_fallbacks: int = 5) -> List[str]:
        """Get appropriate fallback models for a specific model"""
        from src.ii_agent.llm.model_registry import ChutesModelRegistry
        
        # Try to find the model key
        model_key = ChutesModelRegistry.get_model_key_by_id(model_id)
        
        if model_key:
            return ChutesModelRegistry.get_fallback_models(model_key, max_fallbacks)
        else:
            # If model not in registry, use capability-based fallbacks
            logger.warning(f"Model {model_id} not found in registry, using default fallbacks")
            return ChutesModelRegistry.get_default_fallback_chain()[:max_fallbacks]
    
    def get_fallbacks_for_capability(self, capabilities: List[str], max_fallbacks: int = 5) -> List[str]:
        """Get fallback models based on required capabilities"""
        from src.ii_agent.llm.model_registry import ChutesModelRegistry, ModelCapability
        
        # Convert string capabilities to enums
        capability_enums = []
        for cap_str in capabilities:
            for cap_enum in ModelCapability:
                if cap_enum.value == cap_str:
                    capability_enums.append(cap_enum)
                    break
        
        if not capability_enums:
            logger.warning(f"No valid capabilities found in {capabilities}, using default fallbacks")
            return ChutesModelRegistry.get_default_fallback_chain()[:max_fallbacks]
        
        return ChutesModelRegistry.get_fallback_models_by_capability(
            capability_enums,
            max_fallbacks=max_fallbacks
        )
    
    @property
    def max_parallel_agents(self) -> int:
        """Maximum number of agents running in parallel"""
        return int(env_config.get("ROWBOAT_MAX_PARALLEL_AGENTS", "5"))
    
    @property
    def enable_hot_models(self) -> bool:
        """Whether to prioritize hot (high-usage) models"""
        return env_config.get("ROWBOAT_ENABLE_HOT_MODELS", "true").lower() == "true"
    
    @property
    def enable_cost_optimization(self) -> bool:
        """Whether to optimize for cost in model selection"""
        return env_config.get("ROWBOAT_ENABLE_COST_OPTIMIZATION", "true").lower() == "true"
    
    def log_config(self):
        """Log current ROWBOAT configuration"""
        logger.info("ROWBOAT Configuration:")
        logger.info(f"  Copilot Model: {self.copilot_model}")
        logger.info(f"  Default Agent Model: {self.default_agent_model}")
        logger.info(f"  Hub Model: {self.hub_model}")
        logger.info(f"  Info Model: {self.info_model}")
        logger.info(f"  Procedural Model: {self.procedural_model}")
        logger.info(f"  Internal Model: {self.internal_model}")
        logger.info(f"  Code Model: {self.code_model}")
        logger.info(f"  Fallback Models: {len(self.fallback_models)} configured")
        logger.info(f"  Max Parallel Agents: {self.max_parallel_agents}")
        logger.info(f"  Hot Models Enabled: {self.enable_hot_models}")
        logger.info(f"  Cost Optimization: {self.enable_cost_optimization}")

# Create singleton instance
rowboat_config = ROWBOATConfig()