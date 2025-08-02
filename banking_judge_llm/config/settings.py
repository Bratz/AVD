from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ollama_model: str = "llama3"
    ollama_embeddings_model: str = "llama3"
    ollama_base_url: str = "http://localhost:11434"
    bias_threshold: float = 0.1
    faiss_index_path: str = "./data/faiss_index"
    retriever_max_retries: int = 3
    retriever_cache_ttl: int = 3600
    cache_ttl: int = 3600
    guardrail_rules_path: str = "./config/guardrail_rules.json"
    blocked_phrases_path: str = "./config/blocked_phrases.txt"
    audit_log_path: str = "./logs/audit.log"
    audit_summary_path: str = "./logs/audit_summary.json"
    jwt_secret_key: str = "your_jwt_secret_key"  # Set in .env
    rate_limit_requests: int = 100  # Requests per window
    rate_limit_window: int = 60  # Window in seconds