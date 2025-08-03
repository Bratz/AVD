# src/ii_agent/llm/model_registry.py
"""
Chutes Model Registry for BaNCS
Manages available Chutes AI models and their capabilities
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging

from src.ii_agent.llm.chutes_openai import ChutesOpenAIClient
from src.ii_agent.config.environment import env_config

logger = logging.getLogger(__name__)

class ModelCapability(Enum):
    """Model capabilities"""
    TEXT = "text"
    VISION = "vision"
    TOOL_CALLING = "tool_calling"
    LONG_CONTEXT = "long_context"
    FAST_INFERENCE = "fast_inference"
    QA = "qa"
    REASONING = "reasoning"
    CODE = "code"
    INSTRUCTIONS = "instructions"

@dataclass
class ModelInfo:
    """Information about a Chutes model"""
    model_id: str
    name: str
    capabilities: List[ModelCapability]
    context_window: int
    recommended_for: List[str]
    cost_tier: str  # "economy", "standard", "premium"
    # Optional fields for hot models
    pricing_per_m: Optional[float] = None
    usage_7d: Optional[str] = None
    default_temperature: Optional[float] = None  # Add default temperature
    supports_json_mode: Optional[bool] = None    # Add JSON mode support
    
class ChutesModelRegistry:
    """Registry of available Chutes AI models with their capabilities"""
    
    AVAILABLE_MODELS = {
        # Ultra Large Models
        "deepseek-v3": ModelInfo(
            model_id="deepseek-ai/DeepSeek-V3-0324",
            name="DeepSeek V3",
            capabilities=[ModelCapability.TEXT, ModelCapability.VISION, ModelCapability.TOOL_CALLING],
            context_window=128000,
            recommended_for=["complex_reasoning", "research", "analysis", "code_generation"],
            cost_tier="premium",
            pricing_per_m=0.27,
            usage_7d="28.95M"
        ),
        "nemotron-340b": ModelInfo(
            model_id="NVIDIA/Nemotron-4-340B-Chat",
            name="Nemotron 340B",
            capabilities=[ModelCapability.TEXT, ModelCapability.TOOL_CALLING],
            context_window=4096,
            recommended_for=["research", "technical_writing", "complex_analysis", "fallback", "general_agents"],
            cost_tier="premium",
            pricing_per_m=0.10,
            usage_7d="Unknown"
        ),
        
        # Large Models
        "deepseek-chat": ModelInfo(
            model_id="deepseek-ai/DeepSeek-V3-Chat",
            name="DeepSeek Chat",
            capabilities=[ModelCapability.TEXT, ModelCapability.TOOL_CALLING],
            context_window=32000,
            recommended_for=["conversation", "general_tasks", "customer_support"],
            cost_tier="standard"
        ),
        "qwen-72b": ModelInfo(
            model_id="Qwen/Qwen3-72B-Instruct",
            name="Qwen 72B",
            capabilities=[ModelCapability.TEXT],
            context_window=32000,
            recommended_for=["writing", "translation", "content_generation"],
            cost_tier="standard"
        ),
        "qwen-235b": ModelInfo(
            model_id="Qwen/Qwen3-235B-A22B",
            name="Qwen 235B",
            capabilities=[ModelCapability.TEXT],
            context_window=8192,
            recommended_for=["complex_writing", "analysis"],
            cost_tier="premium"
        ),
        
        # Vision Models
        "llama-maverick": ModelInfo(
            model_id="chutesai/Llama-4-Maverick-17B-128E-Instruct-FP8",
            name="Llama Maverick Vision",
            capabilities=[ModelCapability.TEXT, ModelCapability.VISION, ModelCapability.TOOL_CALLING],
            context_window=128000,
            recommended_for=["vision_tasks", "multimodal_analysis", "image_understanding"],
            cost_tier="standard"
        ),
        "qwen-vl": ModelInfo(
            model_id="Qwen/Qwen2.5-VL-32B-Instruct",
            name="Qwen Vision Language",
            capabilities=[ModelCapability.TEXT, ModelCapability.VISION],
            context_window=32000,
            recommended_for=["image_understanding", "visual_qa", "ocr_tasks"],
            cost_tier="standard"
        ),
        
        # Specialized Models
        "deepseek-r1": ModelInfo(
            model_id="deepseek-ai/DeepSeek-R1",
            name="DeepSeek R1",
            capabilities=[ModelCapability.TEXT, ModelCapability.REASONING, ModelCapability.TOOL_CALLING],
            context_window=128000,
            recommended_for=["complex_reasoning", "workflow_design", "copilot", "problem_solving", "math"],
            cost_tier="economy",
            pricing_per_m=0.27,
            usage_7d="44.26M"
        ),
        "deepseek-r1-distill": ModelInfo(
            model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            name="DeepSeek R1 Distill",
            capabilities=[ModelCapability.TEXT, ModelCapability.FAST_INFERENCE],
            context_window=8000,
            recommended_for=["fast_reasoning", "quick_responses"],
            cost_tier="economy"
        ),
        
        # Premium External Models
        "claude-opus-4": ModelInfo(
            model_id="claude-opus-4-0",
            name="Claude Opus 4",
            capabilities=[ModelCapability.TEXT, ModelCapability.TOOL_CALLING, ModelCapability.LONG_CONTEXT],
            context_window=200000,
            recommended_for=["creative_writing", "complex_analysis", "long_documents"],
            cost_tier="premium"
        ),
        "gemini-2.5-pro": ModelInfo(
            model_id="google/gemini-2.5-pro-preview",
            name="Gemini 2.5 Pro",
            capabilities=[ModelCapability.TEXT, ModelCapability.TOOL_CALLING, ModelCapability.LONG_CONTEXT],
            context_window=1000000,
            recommended_for=["long_context", "document_analysis", "research"],
            cost_tier="premium"
        ),
        "gpt-4.1": ModelInfo(
            model_id="openai/gpt-4.1",
            name="GPT-4.1",
            capabilities=[ModelCapability.TEXT, ModelCapability.TOOL_CALLING],
            context_window=128000,
            recommended_for=["general_tasks", "code_generation", "analysis"],
            cost_tier="premium"
        ),
        "gemini-flash-thinking": ModelInfo(
            model_id="google/gemini-2.5-flash-preview-05-20:thinking",
            name="Gemini Flash Thinking",
            capabilities=[ModelCapability.TEXT, ModelCapability.FAST_INFERENCE],
            context_window=32000,
            recommended_for=["quick_analysis", "fast_responses"],
            cost_tier="economy"
        ),

        # Additional DeepSeek Models (Hot - High Usage)
        "deepseek-r1-0528": ModelInfo(
            model_id="deepseek-ai/DeepSeek-R1-0528",
            name="DeepSeek R1 0528",
            capabilities=[ModelCapability.TEXT, ModelCapability.REASONING, ModelCapability.TOOL_CALLING],
            context_window=128000,
            recommended_for=["reasoning_agents"],
            cost_tier="economy",
            pricing_per_m=0.27,
            usage_7d="4.31M"
        ),
        "deepseek-v3-base": ModelInfo(
            model_id="deepseek-ai/DeepSeek-V3",
            name="DeepSeek V3 Base",
            capabilities=[ModelCapability.TEXT, ModelCapability.TOOL_CALLING],
            context_window=128000,
            recommended_for=["general_agents"],
            cost_tier="standard",
            pricing_per_m=0.27,
            usage_7d="1.30M"
        ),
        
        # Moonshot AI Models (Hot)
        "kimi-k2": ModelInfo(
            model_id="moonshotai/Kimi-K2-Instruct",
            name="Kimi K2",
            capabilities=[ModelCapability.TEXT, ModelCapability.QA],
            context_window=80000,
            recommended_for=["question_answering", "info_agents"],
            cost_tier="standard",
            pricing_per_m=0.45,
            usage_7d="764.73K"
        ),
        
        # NousResearch Models (Hot - Cost Effective)
        "deephermes-3-llama": ModelInfo(
            model_id="NousResearch/DeepHermes-3-Llama-3-Preview",
            name="DeepHermes 3 Llama",
            capabilities=[ModelCapability.TEXT],
            context_window=8000,
            recommended_for=["internal_agents", "cost_effective"],
            cost_tier="economy",
            pricing_per_m=0.06,
            usage_7d="1.86M"
        ),
        "deephermes-3-mistral": ModelInfo(
            model_id="NousResearch/DeepHermes-3-Mistral-Preview",
            name="DeepHermes 3 Mistral",
            capabilities=[ModelCapability.TEXT],
            context_window=24000,
            recommended_for=["internal_agents"],
            cost_tier="economy",
            pricing_per_m=0.14,
            usage_7d="2.57M"
        ),
        
        # Qwen Models (Hot)
        "qwen3-32b": ModelInfo(
            model_id="Qwen/Qwen3",
            name="Qwen 3 32B",
            capabilities=[ModelCapability.TEXT, ModelCapability.INSTRUCTIONS],
            context_window=32000,
            recommended_for=["task_execution", "procedural_agents"],
            cost_tier="economy",
            pricing_per_m=0.03,
            usage_7d="895.96K"
        ),
        "qwq": ModelInfo(
            model_id="Qwen/QwQ",
            name="QwQ",
            capabilities=[ModelCapability.TEXT, ModelCapability.REASONING],
            context_window=32000,
            recommended_for=["reasoning", "analysis"],
            cost_tier="economy",
            pricing_per_m=0.02,
            usage_7d="479.99K"
        ),
        "qwen-coder": ModelInfo(
            model_id="Qwen/Qwen2.5-Coder-Instruct",
            name="Qwen Coder",
            capabilities=[ModelCapability.TEXT, ModelCapability.CODE],
            context_window=32000,
            recommended_for=["code_generation", "code_agents"],
            cost_tier="economy",
            pricing_per_m=0.03,
            usage_7d="285.06K"
        ),
        "qwen2.5-vl": ModelInfo(
            model_id="Qwen/Qwen2.5-VL-Instruct",
            name="Qwen 2.5 VL",
            capabilities=[ModelCapability.TEXT, ModelCapability.VISION],
            context_window=32000,
            recommended_for=["multimodal_agents"],
            cost_tier="economy",
            pricing_per_m=0.03,
            usage_7d="276.02K"
        ),
        
        # THUDM Models (Hot)
        "glm-z1": ModelInfo(
            model_id="THUDM/GLM-Z1-0414",
            name="GLM Z1",
            capabilities=[ModelCapability.TEXT],
            context_window=32000,
            recommended_for=["general_agents"],
            cost_tier="economy",
            pricing_per_m=0.03,
            usage_7d="582.09K"
        ),
        
        # Mistral Models (Hot)
        "mistral-nemo": ModelInfo(
            model_id="Unsloth/Mistral-Nemo-Instruct-2407",
            name="Mistral Nemo",
            capabilities=[ModelCapability.TEXT],
            context_window=12000,
            recommended_for=["general_agents"],
            cost_tier="economy",
            pricing_per_m=0.02,
            usage_7d="469.75K"
        ),
        
        # Chutes AI Models
        "mistral-small-3.1": ModelInfo(
            model_id="Chutesai/Mistral-Small-3.1-Instruct-2503",
            name="Mistral Small 3.1",
            capabilities=[ModelCapability.TEXT],
            context_window=24000,
            recommended_for=["general_agents"],
            cost_tier="economy",
            pricing_per_m=0.03,
            usage_7d="609.97K"
        ),
    }
    
    # Role-based model recommendations
    ROLE_MODEL_MAPPING = {
        "researcher": "nemotron-340b",
        "analyzer": "deepseek-chat",
        "writer": "qwen-72b",
        "coder": "deepseek-v3",
        "vision_analyst": "llama-maverick",
        "conversationalist": "deepseek-chat",
        "reviewer": "claude-opus-4",
        "fast_responder": "deepseek-r1-distill",
        "document_processor": "gemini-2.5-pro"
    }
    
    # Workflow-specific configurations
    WORKFLOW_CONFIGS = {
        "research_workflow": {
            "researcher": "nemotron-340b",
            "analyzer": "deepseek-chat",
            "writer": "qwen-72b",
            "reviewer": "claude-opus-4"
        },
        "code_generation": {
            "architect": "deepseek-v3",
            "coder": "deepseek-v3",
            "tester": "deepseek-chat",
            "documenter": "qwen-72b"
        },
        "vision_analysis": {
            "image_analyst": "llama-maverick",
            "caption_writer": "qwen-vl",
            "report_generator": "deepseek-chat"
        },
        "customer_support": {
            "intake": "deepseek-chat",
            "specialist": "deepseek-v3",
            "resolution": "qwen-72b"
        },
        "document_processing": {
            "reader": "gemini-2.5-pro",
            "summarizer": "deepseek-chat",
            "analyzer": "deepseek-v3"
        }
    }
    
    @classmethod
    def get_fallback_models(cls, primary_model_key: str, max_fallbacks: int = 5) -> List[str]:
        """
        Get intelligent fallback models based on the primary model's characteristics
        
        Args:
            primary_model_key: The key of the primary model
            max_fallbacks: Maximum number of fallback models to return
            
        Returns:
            List of model IDs to use as fallbacks
        """
        primary_info = cls.AVAILABLE_MODELS.get(primary_model_key)
        if not primary_info:
            # If primary model not found, return default fallback chain
            return cls.get_default_fallback_chain()[:max_fallbacks]
        
        fallback_candidates = []
        
        # Score each potential fallback model
        for key, info in cls.AVAILABLE_MODELS.items():
            if key == primary_model_key:
                continue
            
            score = 0
            
            # Capability overlap (higher score for more overlap)
            capability_overlap = len(set(primary_info.capabilities) & set(info.capabilities))
            score += capability_overlap * 10
            
            # Prefer different cost tier for diversity
            if info.cost_tier != primary_info.cost_tier:
                score += 5
            
            # Prefer models with good usage stats
            if info.usage_7d and info.usage_7d != "Unknown":
                score += 3
            
            # Prefer models with similar context window
            context_diff = abs(info.context_window - primary_info.context_window)
            if context_diff < 50000:
                score += 2
            
            # Add cost tier priority (economy gets bonus for fallbacks)
            cost_priority = {"economy": 3, "standard": 1, "premium": 0}
            score += cost_priority.get(info.cost_tier, 0)
            
            fallback_candidates.append({
                "key": key,
                "model_id": info.model_id,
                "score": score,
                "cost_tier": info.cost_tier,
                "capabilities": info.capabilities
            })
        
        # Sort by score (descending)
        fallback_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Ensure diversity in fallback chain
        selected_fallbacks = []
        selected_tiers = set()
        
        for candidate in fallback_candidates:
            # Try to get at least one model from each cost tier
            if len(selected_fallbacks) < max_fallbacks:
                if candidate["cost_tier"] not in selected_tiers or len(selected_fallbacks) < 3:
                    selected_fallbacks.append(candidate["model_id"])
                    selected_tiers.add(candidate["cost_tier"])
        
        # Fill remaining slots with highest scoring models
        for candidate in fallback_candidates:
            if candidate["model_id"] not in selected_fallbacks and len(selected_fallbacks) < max_fallbacks:
                selected_fallbacks.append(candidate["model_id"])
        
        return selected_fallbacks

    @classmethod
    def get_fallback_models_by_capability(
        cls, 
        required_capabilities: List[ModelCapability], 
        exclude_model: Optional[str] = None,
        max_fallbacks: int = 5
    ) -> List[str]:
        """
        Get fallback models that have specific capabilities
        
        Args:
            required_capabilities: List of required capabilities
            exclude_model: Model key to exclude from fallbacks
            max_fallbacks: Maximum number of fallback models to return
            
        Returns:
            List of model IDs sorted by cost tier (economy first)
        """
        suitable_models = []
        
        for key, info in cls.AVAILABLE_MODELS.items():
            if key == exclude_model:
                continue
            
            # Check if model has all required capabilities
            if all(cap in info.capabilities for cap in required_capabilities):
                # Calculate a score based on cost and usage
                score = 0
                
                # Cost tier priority (economy is preferred for fallbacks)
                cost_priority = {"economy": 10, "standard": 5, "premium": 0}
                score += cost_priority.get(info.cost_tier, 0)
                
                # Usage popularity bonus
                if info.usage_7d and info.usage_7d != "Unknown":
                    if info.usage_7d.endswith("M"):
                        score += 5
                    elif info.usage_7d.endswith("K"):
                        score += 2
                
                suitable_models.append({
                    "model_id": info.model_id,
                    "cost_tier": info.cost_tier,
                    "score": score
                })
        
        # Sort by score (higher is better)
        suitable_models.sort(key=lambda x: x["score"], reverse=True)
        
        return [m["model_id"] for m in suitable_models[:max_fallbacks]]

    @classmethod
    def get_default_fallback_chain(cls) -> List[str]:
        """Get a default fallback chain for general use"""
        # Return a curated list of reliable models across different tiers
        return [
            cls.AVAILABLE_MODELS["deepseek-v3"].model_id,         # Premium, high capability
            cls.AVAILABLE_MODELS["deepseek-chat"].model_id,       # Standard, good general model
            cls.AVAILABLE_MODELS["qwen-72b"].model_id,           # Standard, text specialist
            cls.AVAILABLE_MODELS["deepseek-r1-distill"].model_id, # Economy, fast
            cls.AVAILABLE_MODELS["glm-z1"].model_id               # Economy, general purpose
        ]

    @classmethod
    def get_reasoning_fallback_chain(cls) -> List[str]:
        """Get fallback chain optimized for reasoning tasks"""
        return cls.get_fallback_models_by_capability(
            [ModelCapability.TEXT, ModelCapability.REASONING],
            max_fallbacks=5
        )

    @classmethod
    def get_vision_fallback_chain(cls) -> List[str]:
        """Get fallback chain for vision tasks"""
        return cls.get_fallback_models_by_capability(
            [ModelCapability.VISION],
            max_fallbacks=3
        )

    @classmethod
    def get_tool_calling_fallback_chain(cls) -> List[str]:
        """Get fallback chain for tool calling tasks"""
        return cls.get_fallback_models_by_capability(
            [ModelCapability.TOOL_CALLING],
            max_fallbacks=5
        )
    
    @classmethod
    def create_llm_client_by_model_id(
        cls,
        model_id: str,
        use_native_tools: bool = True,
        fallback_models: Optional[List[str]] = None,
        auto_fallbacks: bool = True,
        temperature: Optional[float] = None,  # Add temperature parameter
        **kwargs
    ) -> ChutesOpenAIClient:
        """
        Create LLM client by model ID directly
        """
        logger.info(f"Creating Chutes client for model ID: {model_id}")
        
        # If temperature is provided, add it to kwargs
        if temperature is not None:
            kwargs['temperature'] = temperature
        
        # Check if we have metadata for this model
        model_info = None
        model_key = None
        
        for key, info in cls.AVAILABLE_MODELS.items():
            if info.model_id == model_id:
                model_info = info
                model_key = key
                logger.info(f"Found model info - Capabilities: {[cap.value for cap in info.capabilities]}, Context: {info.context_window}")
                break
        
        # Generate fallback models if not provided
        if fallback_models is None and auto_fallbacks:
            if model_key:
                fallback_models = cls.get_fallback_models(model_key)
                logger.info(f"Generated {len(fallback_models)} fallback models based on {model_key}")
            else:
                fallback_models = cls.get_default_fallback_chain()
                logger.info(f"Using default fallback chain with {len(fallback_models)} models")
        
        if not model_info:
            logger.warning(f"Model {model_id} not in registry, creating with defaults")
        
        return ChutesOpenAIClient(
            model_name=model_id,
            use_native_tool_calling=use_native_tools,
            fallback_models=fallback_models,
            **kwargs  # Temperature will be passed through kwargs
        )

    @classmethod
    def get_model_key_by_id(cls, model_id: str) -> Optional[str]:
        """
        Get the registry key for a given model ID
        
        Args:
            model_id: The model ID to look up
            
        Returns:
            The registry key if found, None otherwise
        """
        for key, info in cls.AVAILABLE_MODELS.items():
            if info.model_id == model_id:
                return key
        return None

    @classmethod
    def get_hot_models(cls, min_usage: str = "100K") -> List[Dict[str, Any]]:
        """
        Get models with high usage (hot models)
        
        Args:
            min_usage: Minimum usage threshold (e.g., "100K", "1M")
            
        Returns:
            List of hot models
        """
        hot_models = []
        
        # Convert min_usage to number
        min_val = 0
        if min_usage.endswith("K"):
            min_val = float(min_usage[:-1]) * 1000
        elif min_usage.endswith("M"):
            min_val = float(min_usage[:-1]) * 1000000
        
        for key, info in cls.AVAILABLE_MODELS.items():
            # Only include models with usage data
            if info.usage_7d and isinstance(info.usage_7d, str):
                # Parse usage string
                usage_val = 0
                if info.usage_7d.endswith("K"):
                    usage_val = float(info.usage_7d[:-1]) * 1000
                elif info.usage_7d.endswith("M"):
                    usage_val = float(info.usage_7d[:-1]) * 1000000
                
                if usage_val >= min_val:
                    hot_models.append({
                        "key": key,
                        "model_id": info.model_id,
                        "usage": info.usage_7d,
                        "pricing": info.pricing_per_m if info.pricing_per_m else "N/A",
                        "capabilities": [cap.value for cap in info.capabilities]
                    })
        
        return sorted(hot_models, key=lambda x: x["usage"], reverse=True)

    @classmethod
    def get_models_by_capability(cls, capability: str) -> List[Dict[str, Any]]:
        """
        Get models that have a specific capability
        
        Args:
            capability: The capability to filter by (e.g., "tool_calling", "vision", "code")
            
        Returns:
            List of models with the capability
        """
        models = []
        
        # Try to convert string to ModelCapability enum
        capability_enum = None
        for cap in ModelCapability:
            if cap.value == capability:
                capability_enum = cap
                break
        
        if not capability_enum:
            logger.warning(f"Unknown capability: {capability}")
            return models
        
        for key, info in cls.AVAILABLE_MODELS.items():
            if capability_enum in info.capabilities:
                models.append({
                    "key": key,
                    "model_id": info.model_id,
                    "pricing": info.pricing_per_m if info.pricing_per_m else "N/A",
                    "context_window": info.context_window
                })
        return models

    @classmethod
    def get_model_info(cls, model_key: str) -> Optional[ModelInfo]:
        """Get information about a specific model"""
        return cls.AVAILABLE_MODELS.get(model_key)
    
    @classmethod
    def get_model_for_role(cls, role: str) -> str:
        """Get recommended model ID for a specific agent role"""
        model_key = cls.ROLE_MODEL_MAPPING.get(role, "deepseek-v3")
        model_info = cls.AVAILABLE_MODELS.get(model_key)
        return model_info.model_id if model_info else "deepseek-ai/DeepSeek-V3-0324"
    
    @classmethod
    def get_models_by_capability_enum(cls, capability: ModelCapability) -> List[str]:
        """Get all models with a specific capability (using enum)"""
        models = []
        for key, info in cls.AVAILABLE_MODELS.items():
            if capability in info.capabilities:
                models.append(key)
        return models
    
    @classmethod
    def get_models_by_cost_tier(cls, tier: str) -> List[str]:
        """Get all models in a specific cost tier"""
        models = []
        for key, info in cls.AVAILABLE_MODELS.items():
            if info.cost_tier == tier:
                models.append(key)
        return models
    
    @classmethod
    def create_llm_client(
        cls,
        model_key: Optional[str] = "deepseek-v3",
        model_id: Optional[str] = None,
        use_native_tools: bool = True,
        auto_fallbacks: bool = True,
        temperature: Optional[float] = None,  # Add temperature parameter
        **kwargs
    ) -> ChutesOpenAIClient:
        """Create a Chutes LLM client with specified model"""
        
        # If temperature is provided, add it to kwargs
        if temperature is not None:
            kwargs['temperature'] = temperature
        
        # Use model_id if provided, otherwise look up by key
        if model_id:
            return cls.create_llm_client_by_model_id(
                model_id=model_id,
                use_native_tools=use_native_tools,
                auto_fallbacks=auto_fallbacks,
                temperature=temperature,  # Pass temperature to the other method
                **kwargs
            )
        
        model_info = cls.AVAILABLE_MODELS.get(model_key)
        if not model_info:
            raise ValueError(f"Unknown model key: {model_key}")
        
        # Check if model supports native tool calling
        supports_tools = ModelCapability.TOOL_CALLING in model_info.capabilities
        
        # Generate appropriate fallback models if not provided
        if auto_fallbacks and "fallback_models" not in kwargs:
            kwargs["fallback_models"] = cls.get_fallback_models(model_key)
        
        logger.info(f"Creating Chutes client for {model_info.name} (ID: {model_info.model_id})")
        logger.info(f"Capabilities: {[c.value for c in model_info.capabilities]}")
        logger.info(f"Context window: {model_info.context_window}")
            # Remove temperature from kwargs - ChutesOpenAIClient doesn't accept it
        if temperature is not None:
            logger.info(f"Temperature {temperature} requested (not supported by ChutesOpenAIClient)")
            kwargs.pop('temperature', None)  # Remove if it exists

        return ChutesOpenAIClient(
            model_name=model_info.model_id,
            use_native_tool_calling=use_native_tools and supports_tools,
            **kwargs  # Temperature will be passed through kwargs
        )
    
    @classmethod
    def get_workflow_config(cls, workflow_type: str) -> Dict[str, str]:
        """Get model configuration for a specific workflow type"""
        return cls.WORKFLOW_CONFIGS.get(workflow_type, cls.WORKFLOW_CONFIGS["research_workflow"])
    
    @classmethod
    def estimate_cost_tier(cls, workflow_config: Dict[str, str]) -> str:
        """Estimate the cost tier for a workflow configuration"""
        tiers = []
        for role, model_key in workflow_config.items():
            model_info = cls.AVAILABLE_MODELS.get(model_key)
            if model_info:
                tiers.append(model_info.cost_tier)
        
        # Return highest cost tier
        if "premium" in tiers:
            return "premium"
        elif "standard" in tiers:
            return "standard"
        else:
            return "economy"
    
    @classmethod
    def list_models(cls, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List available models with optional filtering"""
        models = []
        
        for key, info in cls.AVAILABLE_MODELS.items():
            # Apply filters if provided
            if filters:
                if "capability" in filters:
                    cap = ModelCapability(filters["capability"])
                    if cap not in info.capabilities:
                        continue
                
                if "cost_tier" in filters:
                    if info.cost_tier != filters["cost_tier"]:
                        continue
                
                if "min_context" in filters:
                    if info.context_window < filters["min_context"]:
                        continue
            
            models.append({
                "key": key,
                "model_id": info.model_id,
                "name": info.name,
                "capabilities": [c.value for c in info.capabilities],
                "context_window": info.context_window,
                "recommended_for": info.recommended_for,
                "cost_tier": info.cost_tier,
                "pricing_per_m": info.pricing_per_m if info.pricing_per_m else "N/A",
                "usage_7d": info.usage_7d if info.usage_7d else "N/A"
            })
        
        return models