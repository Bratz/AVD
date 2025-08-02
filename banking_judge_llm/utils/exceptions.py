class BankingJudgeError(Exception):
    """Base exception for Banking Judge LLM errors."""
    pass

class GuardrailViolationError(BankingJudgeError):
    """Raised when guardrail checks fail."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}

class RetrieverError(BankingJudgeError):
    """Raised when FAISS retriever fails."""
    pass

class LLMError(BankingJudgeError):
    """Raised when JudgeLLM fails."""
    pass

class JudgeLLMError(Exception):
    """Exception raised when JudgeLLM evaluation fails."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)