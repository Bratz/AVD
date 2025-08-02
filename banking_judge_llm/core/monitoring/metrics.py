from prometheus_client import Counter, Histogram

# Metrics for LLM requests
llm_requests = Counter(
    "llm_requests_total",
    "Total number of LLM requests processed"
)

# Metrics for guardrail outcomes
guardrail_metrics = Counter(
    "guardrail_checks_total",
    "Total number of guardrail checks",
    ["input_compliant", "output_compliant"]
)

# Metrics for retriever
retriever_requests = Counter(
    "retriever_requests_total",
    "Total number of retriever queries"
)

retriever_latency = Histogram(
    "retriever_latency_seconds",
    "Latency of retriever queries"
)