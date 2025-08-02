from pydantic import BaseModel, Field
from typing import Dict, Optional, Any, List
from datetime import datetime
import pandas as pd

class DecisionRequest(BaseModel):
    """Request model for JudgeLLM evaluation."""
    customer_id: str = Field(..., min_length=1, description="Unique customer identifier")
    query: str = Field(..., min_length=1, description="Banking request query")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context for evaluation")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
        extra = "forbid"

class DecisionResponse(BaseModel):
    """Response model for JudgeLLM evaluation."""
    customer_id: str = Field(..., min_length=1, description="Unique customer identifier")
    verdict: int = Field(..., ge=0, le=1, description="Decision verdict: 1 for approved, 0 for denied")
    text: str = Field(..., min_length=1, description="Explanation of the decision")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the decision")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
        extra = "forbid"

class Regulation(BaseModel):
    """Model for RBI regulation data."""
    text: str = Field(..., min_length=1, description="Regulation text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata such as ID, source, date")
    score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Similarity score from FAISS")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
        extra = "forbid"

class GuardrailRequest(BaseModel):
    """Request model for guardrail processing."""
    input_text: str = Field(..., min_length=1, description="Input text to validate")
    decisions: Optional[List[Dict[str, Any]]] = Field(None, description="List of decisions for output validation")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
        extra = "forbid"

class GuardrailResult(BaseModel):
    """Result model for guardrail processing."""
    sanitized_input: str = Field(..., min_length=1, description="Sanitized input text after PII removal")
    pii_detected: List[Dict[str, Any]] = Field(default_factory=list, description="Detected PII entities")
    blocked_phrases_found: List[str] = Field(default_factory=list, description="Blocked phrases found in input")
    input_compliant: bool = Field(..., description="Whether input is compliant")
    output_compliant: Optional[bool] = Field(None, description="Whether output is compliant")
    bias_metrics: List[Dict[str, Any]] = Field(default_factory=list, description="Bias metrics for output")
    compliance_issues: List[Dict[str, Any]] = Field(default_factory=list, description="Regulatory compliance issues")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
        extra = "forbid"