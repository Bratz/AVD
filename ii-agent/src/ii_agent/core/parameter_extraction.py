"""
Universal parameter extraction module for II-Agent
Domain-agnostic, LLM-powered parameter extraction
"""
import json
import re
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.ii_agent.llm.base import TextPrompt, TextResult, LLMClient
from src.ii_agent.utils.logging_config import get_logger


class ParameterType(Enum):
    """Common parameter types across domains"""
    IDENTIFIER = "identifier"  # IDs, account numbers, codes
    NUMERIC = "numeric"  # Amounts, quantities, measurements
    TEMPORAL = "temporal"  # Dates, times, durations
    ENTITY = "entity"  # Names, organizations, places
    BOOLEAN = "boolean"  # Yes/no, true/false
    SELECTION = "selection"  # From a list of options
    TEXT = "text"  # Free text, descriptions
    STRUCTURED = "structured"  # Complex objects, JSON


@dataclass
class ExtractedParameter:
    """Represents an extracted parameter"""
    name: str
    value: Any
    type: ParameterType
    confidence: float
    source_span: Tuple[int, int]  # Start and end position in text
    context: str = ""  # Surrounding context
    metadata: Dict[str, Any] = field(default_factory=dict)


class UniversalParameterExtractor:
    """
    LLM-powered parameter extraction that works across domains
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client
        self.logger = get_logger("ParameterExtractor")
        
        # Domain-agnostic patterns
        self.patterns = {
            ParameterType.IDENTIFIER: [
                (r'\b[A-Z0-9]{6,20}\b', 'alphanumeric_id'),  # Generic IDs
                (r'\b\d{8,16}\b', 'numeric_id'),  # Numeric IDs
                (r'#\s*(\w+)', 'hash_reference'),  # References like #ORDER123
                (r'\b[A-Z]{2,5}-\d{4,10}\b', 'prefixed_id'),  # Like ABC-12345
            ],
            ParameterType.NUMERIC: [
                (r'(\d+(?:\.\d+)?)\s*(%|percent)', 'percentage'),
                (r'(\d+(?:,\d{3})*(?:\.\d+)?)', 'number'),
                (r'(\d+(?:\.\d+)?)\s*(kg|g|mg|lb|oz|km|m|cm|mm|mi|ft|in)', 'measurement'),
                (r'([+-]?\d+(?:\.\d+)?)', 'signed_number'),
            ],
            ParameterType.TEMPORAL: [
                (r'\d{4}-\d{2}-\d{2}', 'iso_date'),
                (r'\d{1,2}/\d{1,2}/\d{2,4}', 'us_date'),
                (r'\d{1,2}-\d{1,2}-\d{2,4}', 'eu_date'),
                (r'\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?', 'time'),
                (r'(?:today|tomorrow|yesterday)', 'relative_date'),
                (r'(?:next|last)\s+(?:week|month|year)', 'relative_period'),
            ],
            ParameterType.BOOLEAN: [
                (r'\b(?:yes|no|true|false|enable|disable|on|off)\b', 'boolean_value'),
            ]
        }
    
    async def extract_parameters(
        self, 
        text: str, 
        domain_context: Optional[Dict[str, Any]] = None,
        expected_params: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Extract parameters using LLM with optional domain context
        
        Args:
            text: Input text to extract from
            domain_context: Optional domain-specific context
            expected_params: Optional list of expected parameters with types
        """
        
        # Step 1: Basic pattern extraction
        pattern_results = self._extract_with_patterns(text)
        
        # Step 2: LLM-powered extraction if available
        if self.llm_client:
            llm_results = await self._extract_with_llm(
                text, 
                domain_context, 
                expected_params,
                pattern_results
            )
            
            # Step 3: Merge and validate
            final_results = self._merge_results(pattern_results, llm_results)
        else:
            final_results = pattern_results
        
        # Step 4: Structure the output
        return self._structure_output(final_results, text)
    
    def _extract_with_patterns(self, text: str) -> List[ExtractedParameter]:
        """Extract using regex patterns"""
        results = []
        
        for param_type, patterns in self.patterns.items():
            for pattern, label in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    # Get the matched value
                    value = match.group(1) if match.groups() else match.group(0)
                    
                    # Get context (20 chars before and after)
                    start = max(0, match.start() - 20)
                    end = min(len(text), match.end() + 20)
                    context = text[start:end]
                    
                    results.append(ExtractedParameter(
                        name=f"{param_type.value}_{label}",
                        value=self._parse_value(value, param_type),
                        type=param_type,
                        confidence=0.7,  # Pattern matching confidence
                        source_span=(match.start(), match.end()),
                        context=context,
                        metadata={"extraction_method": "pattern", "label": label}
                    ))
        
        return results
    
    async def _extract_with_llm(
        self, 
        text: str,
        domain_context: Optional[Dict[str, Any]],
        expected_params: Optional[List[Dict[str, str]]],
        pattern_results: List[ExtractedParameter]
    ) -> List[ExtractedParameter]:
        """Extract parameters using LLM"""
        
        # Build context-aware prompt
        prompt = self._build_extraction_prompt(
            text, 
            domain_context, 
            expected_params,
            pattern_results
        )
        
        try:
            messages = [[TextPrompt(text=prompt)]]
            
            # Check which method is available
            if hasattr(self.llm_client, 'generate'):
                response_blocks, _ = await self.llm_client.generate(
                    messages=messages,
                    max_tokens=500,
                    temperature=0.1,
                    system_prompt="You are an expert at extracting structured information from text. Be precise and comprehensive."
                )
            elif hasattr(self.llm_client, 'generate_completion'):
                # Use OllamaWrapper's method
                response = await self.llm_client.generate_completion(
                    prompt=prompt,  # OllamaWrapper might expect a string prompt
                    max_tokens=500,
                    temperature=0.1
                )
                # Convert response to expected format
                response_blocks = [TextResult(text=response)]
                _ = {}  # Empty metadata
            else:
                self.logger.warning(f"No compatible generation method found on {type(self.llm_client)}")
                return []
                        
            # Parse LLM response
            response_text = self._extract_text_from_blocks(response_blocks)
            
            # Parse JSON response
            json_str = response_text.strip()
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0]
            elif '```' in json_str:
                json_str = json_str.split('```')[1].split('```')[0]
            
            extracted_data = json.loads(json_str)
            
            # Convert to ExtractedParameter objects
            return self._parse_llm_extraction(extracted_data, text)
            
        except Exception as e:
            self.logger.error(f"LLM extraction failed: {e}")
            return []
    
    def _build_extraction_prompt(
        self,
        text: str,
        domain_context: Optional[Dict[str, Any]],
        expected_params: Optional[List[Dict[str, str]]],
        pattern_results: List[ExtractedParameter]
    ) -> str:
        """Build comprehensive extraction prompt"""
        
        prompt_parts = [
            f'Extract all parameters from this text: "{text}"'
        ]
        
        # Add domain context if provided
        if domain_context:
            prompt_parts.append(f"\nDomain context: {json.dumps(domain_context, indent=2)}")
        
        # Add expected parameters if provided
        if expected_params:
            prompt_parts.append("\nExpected parameters:")
            for param in expected_params:
                prompt_parts.append(f"- {param.get('name', 'unknown')}: {param.get('type', 'any')} - {param.get('description', '')}")
        
        # Add pattern results for validation
        if pattern_results:
            found_values = [f"{p.type.value}: {p.value}" for p in pattern_results[:5]]
            prompt_parts.append(f"\nAlready found: {', '.join(found_values)}")
        
        # Add extraction instructions
        prompt_parts.extend([
            "\nExtract ALL parameters including:",
            "1. Identifiers (IDs, codes, references)",
            "2. Numeric values (amounts, quantities, measurements)",
            "3. Temporal information (dates, times, durations)",
            "4. Entities (names, organizations, locations)",
            "5. Boolean values (yes/no, true/false)",
            "6. Selections from implied options",
            "7. Any domain-specific parameters",
            "",
            "Return a JSON object with this structure:",
            "{",
            '  "parameters": [',
            "    {",
            '      "name": "parameter_name",',
            '      "value": "extracted_value",',
            '      "type": "identifier|numeric|temporal|entity|boolean|selection|text|structured",',
            '      "confidence": 0.95,',
            '      "reasoning": "why this was extracted"',
            "    }",
            "  ],",
            '  "intent": "overall intent of the text",',
            '  "domain": "detected domain (e.g., banking, healthcare, retail)"',
            "}"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_llm_extraction(self, extracted_data: Dict, original_text: str) -> List[ExtractedParameter]:
        """Parse LLM extraction results"""
        results = []
        
        for param in extracted_data.get('parameters', []):
            # Find the parameter in original text
            value_str = str(param['value'])
            source_span = self._find_span(value_str, original_text)
            
            # Map type string to enum
            type_mapping = {
                'identifier': ParameterType.IDENTIFIER,
                'numeric': ParameterType.NUMERIC,
                'temporal': ParameterType.TEMPORAL,
                'entity': ParameterType.ENTITY,
                'boolean': ParameterType.BOOLEAN,
                'selection': ParameterType.SELECTION,
                'text': ParameterType.TEXT,
                'structured': ParameterType.STRUCTURED
            }
            
            param_type = type_mapping.get(param.get('type', 'text'), ParameterType.TEXT)
            
            results.append(ExtractedParameter(
                name=param['name'],
                value=self._parse_value(param['value'], param_type),
                type=param_type,
                confidence=param.get('confidence', 0.8),
                source_span=source_span,
                context=self._get_context(original_text, source_span),
                metadata={
                    "extraction_method": "llm",
                    "reasoning": param.get('reasoning', ''),
                    "domain": extracted_data.get('domain', 'general'),
                    "intent": extracted_data.get('intent', '')
                }
            ))
        
        return results
    
    def _merge_results(
        self, 
        pattern_results: List[ExtractedParameter],
        llm_results: List[ExtractedParameter]
    ) -> List[ExtractedParameter]:
        """Merge pattern and LLM results, removing duplicates"""
        merged = {}
        
        # Add pattern results
        for param in pattern_results:
            key = (param.type, str(param.value).lower())
            if key not in merged or param.confidence > merged[key].confidence:
                merged[key] = param
        
        # Add/update with LLM results (usually higher confidence)
        for param in llm_results:
            key = (param.type, str(param.value).lower())
            if key not in merged or param.confidence > merged[key].confidence:
                merged[key] = param
        
        return list(merged.values())
    
    def _structure_output(self, parameters: List[ExtractedParameter], original_text: str) -> Dict[str, Any]:
        """Structure the final output"""
        output = {
            "text": original_text,
            "parameters": {},
            "all_extractions": [],
            "summary": {
                "total_parameters": len(parameters),
                "by_type": {},
                "high_confidence": []
            }
        }
        
        # Group by type
        for param in parameters:
            # Add to parameters dict
            if param.confidence > 0.7:
                if param.type.value not in output["parameters"]:
                    output["parameters"][param.type.value] = []
                output["parameters"][param.type.value].append({
                    "name": param.name,
                    "value": param.value,
                    "confidence": param.confidence
                })
            
            # Add to all extractions
            output["all_extractions"].append({
                "name": param.name,
                "value": param.value,
                "type": param.type.value,
                "confidence": param.confidence,
                "span": param.source_span,
                "context": param.context,
                "metadata": param.metadata
            })
            
            # Update summary
            type_name = param.type.value
            if type_name not in output["summary"]["by_type"]:
                output["summary"]["by_type"][type_name] = 0
            output["summary"]["by_type"][type_name] += 1
            
            if param.confidence >= 0.8:
                output["summary"]["high_confidence"].append(param.name)
        
        return output
    
    def _parse_value(self, value: str, param_type: ParameterType) -> Any:
        """Parse value based on type"""
        if param_type == ParameterType.NUMERIC:
            # Remove commas and try to parse
            clean_value = value.replace(',', '')
            try:
                if '.' in clean_value:
                    return float(clean_value)
                else:
                    return int(clean_value)
            except:
                return value
        
        elif param_type == ParameterType.BOOLEAN:
            lower_value = value.lower()
            if lower_value in ['yes', 'true', 'enable', 'on']:
                return True
            elif lower_value in ['no', 'false', 'disable', 'off']:
                return False
            return value
        
        elif param_type == ParameterType.TEMPORAL:
            # Keep as string but could parse to datetime
            return value
        
        else:
            return value
    
    def _find_span(self, value: str, text: str) -> Tuple[int, int]:
        """Find the span of a value in text"""
        try:
            start = text.lower().index(str(value).lower())
            return (start, start + len(str(value)))
        except ValueError:
            return (0, 0)
    
    def _get_context(self, text: str, span: Tuple[int, int], context_size: int = 30) -> str:
        """Get context around a span"""
        start = max(0, span[0] - context_size)
        end = min(len(text), span[1] + context_size)
        return text[start:end]
    
    def _extract_text_from_blocks(self, blocks) -> str:
        """Extract text from LLM response blocks"""
        text = ""
        for block in blocks:
            if isinstance(block, TextResult):
                text += block.text
            elif hasattr(block, 'text'):
                text += block.text
        return text


class DomainAwareExtractor(UniversalParameterExtractor):
    """
    Domain-aware parameter extractor that can be configured for specific domains
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None, domain: str = "general"):
        super().__init__(llm_client)
        self.domain = domain
        self.domain_patterns = self._load_domain_patterns()
    
    def _load_domain_patterns(self) -> Dict[str, List[Tuple[str, str]]]:
        """Load domain-specific patterns"""
        domain_patterns = {
            "banking": [
                (r'\b(?:account|acc)\s*(?:no|number|#)?\s*[:=-]?\s*(\d{8,16})\b', 'account_number'),
                (r'(?:balance|amount|sum)\s*[:=-]?\s*\$?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', 'monetary_amount'),
                (r'(?:transfer|send|pay)\s+(?:to\s+)?(\w+)', 'recipient'),
            ],
            "healthcare": [
                (r'\b(?:patient|pt)\s*(?:id|#)?\s*[:=-]?\s*(\w+)\b', 'patient_id'),
                (r'(\d{1,3}/\d{1,3})\s*(?:mmHg|mm Hg)', 'blood_pressure'),
                (r'(\d{1,3})\s*(?:bpm|beats)', 'heart_rate'),
            ],
            "retail": [
                (r'(?:order|invoice)\s*(?:#|number|no)?\s*[:=-]?\s*(\w+)', 'order_id'),
                (r'(?:sku|product)\s*[:=-]?\s*([A-Z0-9-]+)', 'product_sku'),
                (r'quantity\s*[:=-]?\s*(\d+)', 'quantity'),
            ]
        }
        
        return domain_patterns.get(self.domain, {})
    
    async def extract_parameters(
        self, 
        text: str, 
        domain_context: Optional[Dict[str, Any]] = None,
        expected_params: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Extract with domain awareness"""
        
        # Add domain patterns to base patterns
        if self.domain_patterns:
            for pattern, label in self.domain_patterns:
                # Determine type based on label
                if any(kw in label for kw in ['id', 'number', 'sku']):
                    param_type = ParameterType.IDENTIFIER
                elif any(kw in label for kw in ['amount', 'quantity', 'rate']):
                    param_type = ParameterType.NUMERIC
                else:
                    param_type = ParameterType.TEXT
                
                if param_type not in self.patterns:
                    self.patterns[param_type] = []
                self.patterns[param_type].append((pattern, f"{self.domain}_{label}"))
        
        # Set domain context
        if domain_context is None:
            domain_context = {}
        domain_context['domain'] = self.domain
        
        # Use parent extraction
        return await super().extract_parameters(text, domain_context, expected_params)