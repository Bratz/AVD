from typing import List, Dict
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset
import pandas as pd
import json
import logging
from config.settings import Settings
from services.audit.file_logger import FileLogger
from utils.exceptions import GuardrailViolationError
from utils.websocket import WebSocketManager

class OutputGuardrail:
    """Guardrail for output bias and compliance checking."""

    def __init__(self):
        self.settings = Settings()
        self.logger = logging.getLogger(__name__)
        self.audit_logger = FileLogger()
        self.websocket_manager = WebSocketManager()
        self.bias_threshold = self.settings.bias_threshold
        self.regulatory_rules = self._load_regulatory_rules()

    def _load_regulatory_rules(self) -> Dict:
        """Load regulatory rules from file."""
        try:
            with open(self.settings.guardrail_rules_path, "r") as f:
                rules = json.load(f)
            self.logger.info(f"Loaded {len(rules.get('output_rules', []))} output regulatory rules")
            return rules
        except Exception as e:
            self.logger.error(f"Failed to load regulatory rules: {str(e)}")
            self.audit_logger.log_event(
                event_type="regulatory_rules_load_error",
                details={"error": str(e)}
            )
            raise GuardrailViolationError(f"Failed to load regulatory rules: {str(e)}")

    def process(self, decisions: pd.DataFrame) -> Dict:
        """Process LLM output for bias and regulatory compliance."""
        try:
            # Initialize result
            result = {
                "output_compliant": True,
                "bias_metrics": [],
                "compliance_issues": []
            }

            # Skip if no decisions
            if decisions.empty:
                self.logger.debug("No decisions provided for output guardrail")
                return result

            # Prepare dataset for bias analysis
            dataset = BinaryLabelDataset(
                df=decisions[["verdict"]].assign(protected_attribute=0),  # Simplified: assume single group
                label_names=["verdict"],
                protected_attribute_names=["protected_attribute"]
            )

            # Calculate bias metrics
            metric = BinaryLabelDatasetMetric(
                dataset,
                privileged_groups=[{"protected_attribute": 0}],
                unprivileged_groups=[{"protected_attribute": 0}]
            )
            bias_score = metric.disparate_impact()
            bias_metrics = [{"name": "disparate_impact", "value": bias_score}]

            # Check bias threshold
            if bias_score < 1 - self.bias_threshold or bias_score > 1 + self.bias_threshold:
                result["output_compliant"] = False
                result["bias_metrics"].append({"name": "disparate_impact", "value": bias_score})

            # Check regulatory compliance
            for decision in decisions.itertuples():
                decision_text = getattr(decision, "text", "")
                for rule in self.regulatory_rules.get("output_rules", []):
                    if rule["pattern"].lower() in decision_text.lower():
                        result["output_compliant"] = False
                        result["compliance_issues"].append({
                            "rule_id": rule["id"],
                            "description": rule["description"],
                            "decision_text": decision_text
                        })

            # Log results
            details = {
                "decisions": decisions.to_dict(orient="records"),
                "bias_metrics": result["bias_metrics"],
                "compliance_issues": result["compliance_issues"],
                "output_compliant": result["output_compliant"]
            }
            self.audit_logger.log_event(
                event_type="output_guardrail_check",
                details=details
            )

            # Notify via WebSocket if non-compliant
            if not result["output_compliant"]:
                asyncio.create_task(
                    self.websocket_manager.send_progress(
                        session_id="guardrail_session",
                        message={
                            "stage": "output_guardrail_violation",
                            "details": details,
                            "timestamp": str(logging.time())
                        }
                    )
                )

            return result

        except Exception as e:
            self.logger.error(f"Output guardrail processing failed: {str(e)}")
            self.audit_logger.log_event(
                event_type="output_guardrail_error",
                details={"decisions": decisions.to_dict(orient="records"), "error": str(e)}
            )
            raise GuardrailViolationError(f"Output processing failed: {str(e)}")