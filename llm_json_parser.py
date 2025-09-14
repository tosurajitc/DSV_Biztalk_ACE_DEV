"""
LLM JSON Parser Module
======================

A robust JSON parser specifically designed to handle unpredictable LLM responses.
Handles various formats: markdown-wrapped JSON, mixed text+JSON, partial responses, etc.

"""

import json
import re
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Result of JSON parsing attempt"""
    success: bool
    data: Optional[Dict[Any, Any]]
    error_message: str
    raw_content: str
    method_used: str
    cleaned_content: Optional[str] = None


class LLMJSONParser:
    """
    Robust JSON parser for LLM responses that handles various edge cases:
    - Markdown code blocks (```json ... ```)
    - Mixed text and JSON content
    - Partial/truncated JSON
    - Empty responses
    - Invalid JSON with common LLM mistakes
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.parsing_stats = {
            'total_attempts': 0,
            'direct_json_success': 0,
            'markdown_extraction_success': 0,
            'regex_extraction_success': 0,
            'fuzzy_extraction_success': 0,
            'failures': 0
        }
    
    def parse(self, llm_response: str, expected_schema: Optional[Dict] = None) -> ParseResult:
        """
        Main parsing method that tries multiple strategies to extract JSON from LLM response.
        
        Args:
            llm_response: Raw response from LLM
            expected_schema: Optional schema to validate against (dict with expected keys)
            
        Returns:
            ParseResult object with success status and extracted data
        """
        self.parsing_stats['total_attempts'] += 1
        
        if self.debug:
            logger.info(f"Parsing LLM response (length: {len(llm_response)})")
            logger.debug(f"Raw content preview: {llm_response[:200]}...")
        
        # Strategy 1: Direct JSON parsing
        result = self._try_direct_json(llm_response)
        if result.success:
            self.parsing_stats['direct_json_success'] += 1
            return self._validate_schema(result, expected_schema)
        
        # Strategy 2: Extract from markdown code blocks
        result = self._try_markdown_extraction(llm_response)
        if result.success:
            self.parsing_stats['markdown_extraction_success'] += 1
            return self._validate_schema(result, expected_schema)
        
        # Strategy 3: Regex-based JSON extraction
        result = self._try_regex_extraction(llm_response)
        if result.success:
            self.parsing_stats['regex_extraction_success'] += 1
            return self._validate_schema(result, expected_schema)
        
        # Strategy 4: Fuzzy JSON extraction (handle common LLM mistakes)
        result = self._try_fuzzy_extraction(llm_response)
        if result.success:
            self.parsing_stats['fuzzy_extraction_success'] += 1
            return self._validate_schema(result, expected_schema)
        
        # All strategies failed
        self.parsing_stats['failures'] += 1
        return ParseResult(
            success=False,
            data=None,
            error_message=f"All parsing strategies failed. Content preview: {llm_response[:100]}...",
            raw_content=llm_response,
            method_used="none"
        )
    
    def _try_direct_json(self, content: str) -> ParseResult:
        """Try parsing content directly as JSON"""
        try:
            cleaned = content.strip()
            if not cleaned:
                return ParseResult(False, None, "Empty content", content, "direct_json")
            
            data = json.loads(cleaned)
            return ParseResult(True, data, "", content, "direct_json", cleaned)
        except json.JSONDecodeError as e:
            return ParseResult(False, None, f"Direct JSON parse failed: {e}", content, "direct_json")
    
    def _try_markdown_extraction(self, content: str) -> ParseResult:
        """Extract JSON from markdown code blocks"""
        # Look for ```json ... ``` or ``` ... ``` blocks
        patterns = [
            r'```json\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```',
            r'`json\s*\n(.*?)\n`',
            r'`(.*?)`'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    cleaned = match.strip()
                    data = json.loads(cleaned)
                    return ParseResult(True, data, "", content, "markdown_extraction", cleaned)
                except json.JSONDecodeError:
                    continue
        
        return ParseResult(False, None, "No valid JSON found in markdown blocks", content, "markdown_extraction")
    
    def _try_regex_extraction(self, content: str) -> ParseResult:
        """Use regex to find JSON-like structures"""
        # Look for content between curly braces
        patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested braces
            r'\{.*?\}',  # Simple braces
            r'\[.*?\]'   # Arrays
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    cleaned = match.strip()
                    data = json.loads(cleaned)
                    return ParseResult(True, data, "", content, "regex_extraction", cleaned)
                except json.JSONDecodeError:
                    continue
        
        return ParseResult(False, None, "No valid JSON structures found via regex", content, "regex_extraction")
    
    def _try_fuzzy_extraction(self, content: str) -> ParseResult:
        """Handle common LLM JSON mistakes and try to fix them"""
        try:
            # Common fixes for LLM JSON responses
            fixed_content = content.strip()
            
            # Remove common prefixes/suffixes
            prefixes_to_remove = [
                "Here's the JSON:",
                "Here is the JSON:",
                "JSON Response:",
                "```json",
                "```",
                "The JSON is:",
                "Response:"
            ]
            
            for prefix in prefixes_to_remove:
                if fixed_content.lower().startswith(prefix.lower()):
                    fixed_content = fixed_content[len(prefix):].strip()
            
            # Remove common suffixes
            suffixes_to_remove = ["```", "That's the JSON response.", "End of JSON."]
            for suffix in suffixes_to_remove:
                if fixed_content.lower().endswith(suffix.lower()):
                    fixed_content = fixed_content[:-len(suffix)].strip()
            
            # Fix common JSON issues
            fixes = [
                # Single quotes to double quotes
                (r"'([^']*)':", r'"\1":'),
                # Trailing commas
                (r',\s*}', '}'),
                (r',\s*]', ']'),
                # Missing quotes on keys
                (r'(\w+):', r'"\1":'),
                # Python True/False/None to JSON equivalents
                (r'\bTrue\b', 'true'),
                (r'\bFalse\b', 'false'),
                (r'\bNone\b', 'null'),
            ]
            
            for pattern, replacement in fixes:
                fixed_content = re.sub(pattern, replacement, fixed_content)
            
            # Try to parse the fixed content
            data = json.loads(fixed_content)
            return ParseResult(True, data, "", content, "fuzzy_extraction", fixed_content)
            
        except json.JSONDecodeError as e:
            # If still fails, try to extract just the first complete JSON object
            try:
                # Find first { and matching }
                start = fixed_content.find('{')
                if start != -1:
                    brace_count = 0
                    for i, char in enumerate(fixed_content[start:], start):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_candidate = fixed_content[start:i+1]
                                data = json.loads(json_candidate)
                                return ParseResult(True, data, "", content, "fuzzy_extraction_partial", json_candidate)
            except json.JSONDecodeError:
                pass
        
        return ParseResult(False, None, f"Fuzzy extraction failed. Content: {fixed_content[:100]}...", content, "fuzzy_extraction")
    
    def _validate_schema(self, result: ParseResult, expected_schema: Optional[Dict]) -> ParseResult:
        """Validate parsed JSON against expected schema"""
        if not expected_schema or not result.success:
            return result
        
        try:
            # Basic schema validation - check if expected keys exist
            if isinstance(result.data, dict):
                missing_keys = []
                for key in expected_schema.keys():
                    if key not in result.data:
                        missing_keys.append(key)
                
                if missing_keys:
                    logger.warning(f"Schema validation warning: Missing keys {missing_keys}")
                    # Don't fail, just warn
            
            return result
        except Exception as e:
            logger.error(f"Schema validation error: {e}")
            return result
    
    def parse_business_requirements(self, llm_response: str) -> Dict[str, List[str]]:
        """
        Specialized parser for business requirements with fallback structure
        """
        expected_schema = {
            "message_flows": [],
            "transformation_requirements": [],
            "integration_endpoints": [],
            "database_lookups": [],
            "business_entities": [],
            "ace_library_indicators": [],
            "processing_patterns": [],
            "technical_specifications": [],
            "data_enrichment_rules": [],
            "routing_logic": []
        }
        
        result = self.parse(llm_response, expected_schema)
        
        if result.success and isinstance(result.data, dict):
            # Ensure all expected keys exist with empty lists as defaults
            for key in expected_schema.keys():
                if key not in result.data:
                    result.data[key] = []
                elif not isinstance(result.data[key], list):
                    # Convert single items to lists
                    result.data[key] = [result.data[key]] if result.data[key] else []
            
            return result.data
        else:
            logger.error(f"Failed to parse business requirements: {result.error_message}")
            # Return empty structure instead of failing completely
            return expected_schema
    
    def parse_component_mappings(self, llm_response: str) -> List[Dict[str, Any]]:
        """
        Specialized parser for component mappings
        """
        result = self.parse(llm_response)
        
        if result.success:
            if isinstance(result.data, list):
                return result.data
            elif isinstance(result.data, dict) and 'mappings' in result.data:
                return result.data['mappings']
        
        logger.error(f"Failed to parse component mappings: {result.error_message}")
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get parsing statistics"""
        total = self.parsing_stats['total_attempts']
        if total == 0:
            return self.parsing_stats
        
        stats = self.parsing_stats.copy()
        stats['success_rate'] = (total - stats['failures']) / total * 100
        return stats
    
    def reset_stats(self):
        """Reset parsing statistics"""
        for key in self.parsing_stats:
            self.parsing_stats[key] = 0


# Convenience functions for easy import and use
def parse_llm_json(llm_response: str, expected_schema: Optional[Dict] = None, debug: bool = False) -> ParseResult:
    """
    Quick function to parse LLM JSON response
    
    Usage:
        result = parse_llm_json(response.choices[0].message.content)
        if result.success:
            data = result.data
        else:
            print(f"Parsing failed: {result.error_message}")
    """
    parser = LLMJSONParser(debug=debug)
    return parser.parse(llm_response, expected_schema)


def safe_json_loads(llm_response: str, fallback: Any = None) -> Any:
    """
    Safe JSON loading with fallback value
    
    Usage:
        data = safe_json_loads(llm_response, fallback={})
    """
    result = parse_llm_json(llm_response)
    return result.data if result.success else fallback


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_cases = [
        # Valid JSON
        '{"key": "value", "list": [1, 2, 3]}',
        
        # Markdown wrapped
        '```json\n{"key": "value"}\n```',
        
        # Mixed content
        'Here is the JSON response:\n{"result": "success"}\nThat concludes the analysis.',
        
        # Python-style (need fixing)
        "{'key': True, 'value': None}",
        
        # Trailing comma
        '{"key": "value",}',
        
        # Empty response
        '',
        
        # Invalid JSON
        'This is not JSON at all.',
    ]
    
    parser = LLMJSONParser(debug=True)
    
    for i, test in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Input: {test[:50]}...")
        result = parser.parse(test)
        print(f"Success: {result.success}")
        print(f"Method: {result.method_used}")
        if result.success:
            print(f"Data: {result.data}")
        else:
            print(f"Error: {result.error_message}")
    
    print(f"\n--- Parsing Statistics ---")
    print(parser.get_stats())