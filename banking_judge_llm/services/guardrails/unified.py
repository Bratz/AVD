import logging
import json
import pandas as pd
from typing import List, Dict, Tuple
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.inprocessing import AdversarialDebiasing
from schemas.models import GuardrailResult, Regulation
from core.llm.retriever import Retriever
from services.audit.file_logger import FileLogger
from config.settings import Settings

class UnifiedGuardrail:
    """Unified guardrail with differentiated input/output rules."""
    
    def __init__(self):
        self.settings = Settings()
        self.input_guardrail = InputGuardrail()
        self.output_guardrail = OutputGuardrail()
        self.retriever = Retriever()
        self.logger = logging.getLogger(__name__)
        self.audit_logger = FileLogger()
        self.input_rules, self.output_rules = self._load_rules()
        self.blocked_phrases = self._load_blocked_phrases()

    def _load_rules(self) -> Tuple[Dict, Dict]:
        """Load input and output rules from guardrail_rules.json."""
        try:
            with open(self.settings.guardrail_rules_path, "r") as f:
                rules = json.load(f)
                return rules.get("input", {}), rules.get("output", {})
        except Exception as e:
            self.logger.error(f"Failed to load guardrail rules: {str(e)}")
            self.audit_logger.log_event(
                event_type="guardrail_rules_load_error",
                details={"error": str(e)}
            )
            return {}, {}

    def _load_blocked_phrases(self) -> List[str]:
        """Load blocked phrases for input filtering."""
        try:
            with open(self.settings.blocked_phrases_path, "r") as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            self.logger.error(f"Failed to load blocked phrases: {str(e)}")
            self.audit_logger.log_event(
                event_type="blocked_phrases_load_error",
                details={"error": str(e)}
            )
            return []

    def check_regulatory_compliance(self, text: str, is_input: bool = True) -> bool:
        """Check compliance using FAISS retriever and specific rules."""
        try:
            rules = self.input_rules if is_input else self.output_rules
            relevant_rules = self.retriever.get_relevant_rules(text)
            rule_violations = []

            # Check FAISS-retrieved regulations
            for rule in relevant_rules:
                if rule.score > (rules.get("score_threshold", 0.8)):
                    rule_violations.append(rule.text)

            # Check blocked phrases for input
            if is_input:
                for phrase in self.blocked_phrases:
                    if phrase.lower() in text.lower():
                        rule_violations.append(f"Blocked phrase: {phrase}")

            if rule_violations:
                self.logger.warning(f"Compliance violations: {rule_violations}")
                self.audit_logger.log_event(
                    event_type="regulatory_check",
                    details={"text": text, "is_input": is_input, "violations": rule_violations}
                )
                return False
            return True
        except Exception as e:
            self.logger.error(f"Regulatory check failed: {str(e)}")
            self.audit_logger.log_event(
                event_type="regulatory_check_error",
                details={"text": text, "is_input": is_input, "error": str(e)}
            )
            return False

    def process(self, input_text: str, decisions: pd.DataFrame) -> GuardrailResult:
        """Process input and output through guardrails with differentiated rules."""
        sanitized_text, pii_entities = self.input_guardrail.sanitize(input_text)
        self.audit_logger.log_event(
            event_type="input_sanitization",
            details={"original_text": input_text, "sanitized_text": sanitized_text, "pii": pii_entities}
        )

        input_compliant = self.check_regulatory_compliance(sanitized_text, is_input=True)
        bias_metrics = self.output_guardrail.check_bias(decisions)
        output_compliant = all(metric["value"] < self.settings.bias_threshold for metric in bias_metrics)
        output_text = decisions.get("text", "").iloc[0] if not decisions.empty else ""
        output_compliant = output_compliant and self.check_regulatory_compliance(output_text, is_input=False)

        self.audit_logger.log_event(
            event_type="guardrail_process",
            details={
                "input_compliant": input_compliant,
                "output_compliant": output_compliant,
                "bias_metrics": bias_metrics
            }
        )

        return GuardrailResult(
            sanitized_input=sanitized_text,
            pii_detected=pii_entities,
            input_compliant=input_compliant,
            output_compliant=output_compliant,
            bias_metrics=bias_metrics
        )

class InputGuardrail:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.logger = logging.getLogger(__name__)

    def sanitize(self, text: str) -> Tuple[str, List[Dict]]:
        analysis = self.analyzer.analyze(text=text, language="en")
        if analysis:
            self.logger.warning(f"PII detected: {analysis}")
            anonymized = self.anonymizer.anonymize(text=text, analyzer_results=analysis)
            return anonymized.text, [{"entity": a.entity_type, "start": a.start, "end": a.end} for a in analysis]
        return text, []

class OutputGuardrail:
    def check_bias(self, decisions: pd.DataFrame) -> List[Dict]:
        dataset = BinaryLabelDataset(df=decisions, label_names=["verdict"])
        return AdversarialDebiasing().fit(dataset).metrics()