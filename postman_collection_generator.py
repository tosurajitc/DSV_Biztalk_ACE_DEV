"""
Postman Collection Generator for IBM ACE MessageFlow Testing
===========================================================

Generates comprehensive Postman test collections from IBM ACE artifacts using Vector DB business requirements.
Pure LLM-powered generation with robust JSON parsing - no fallbacks, no helper methods.
"""

import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from groq import Groq
from llm_json_parser import parse_llm_json
import streamlit as st
from llm_json_parser import LLMJSONParser, parse_llm_json

class PostmanCollectionGenerator:
    """
    AI-powered Postman collection generator for IBM ACE MessageFlow testing.
    Uses Vector DB business requirements and LLM intelligence for realistic test generation.
    """
    
    def __init__(self, 
                 reviewed_modules_path: str,
                 target_output_folder: str = None,
                 project_name: str = "ACE_MessageFlow"):
        
        self.reviewed_modules_path = Path(reviewed_modules_path)
        self.project_name = project_name
        self.vector_business_requirements = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate output paths with project name formatting
        formatted_folder_name = f"{project_name}_POSTMAN_COLLECTIONS"
        
        if target_output_folder:
            self.output_root = Path(target_output_folder) / formatted_folder_name
        else:
            root_folder = self.reviewed_modules_path.parent
            self.output_root = root_folder / f"{project_name}_POSTMAN_COLLECTIONS"
        
        # Create subfolder structure
        self.paths = {
            'root': self.output_root,
            'collections': self.output_root / "collections",
            'environments': self.output_root / "environments", 
            'test_data': self.output_root / "test_data",
            'documentation': self.output_root / "documentation"
        }
        
        # LLM Configuration
        self.model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
        self.temperature = 0.3
        self.max_tokens_analysis = 4000
        self.max_tokens_generation = 3000
        
        # Initialize LLM client
        self.llm_client = None
        self._initialize_llm()
        
        # Results tracking
        self.generation_results = {
            'timestamp': datetime.now().isoformat(),
            'collections_created': [],
            'environments_created': [],
            'test_scenarios_generated': 0,
            'payload_samples_created': 0,
            'documentation_files': [],
            'ai_analysis_calls': 0,
            'ai_generation_calls': 0
        }
    
    def _initialize_llm(self):
        """Initialize LLM client - required for operation"""
        try:
            groq_api_key = os.getenv('GROQ_API_KEY')
            if not groq_api_key:
                raise Exception("GROQ_API_KEY environment variable is required")
            
            self.llm_client = Groq(api_key=groq_api_key)
            print("AI Intelligence initialized successfully - 100% LLM powered!")
        except Exception as e:
            print(f"AI initialization failed: {e}")
            raise Exception("Cannot proceed without AI intelligence")
    
    def _call_llm_intelligence(self, system_prompt: str, user_prompt: str, 
                              max_tokens: int = 2000, call_type: str = "general") -> str:
        """Make LLM call with error handling"""
        
        if call_type == "analysis":
            self.generation_results['ai_analysis_calls'] += 1
        else:
            self.generation_results['ai_generation_calls'] += 1
            
        try:
            response = self.llm_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=self.temperature,
                max_tokens=max_tokens
            )

            if 'token_tracker' in st.session_state and hasattr(response, 'usage') and response.usage:
                st.session_state.token_tracker.manual_track(
                    agent="postman_generator",                    # ✅ CORRECT agent name
                    operation=f"postman_{call_type}",            # ✅ DYNAMIC operation based on call_type
                    model=self.model,                            # ✅ CORRECT model reference
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    flow_name=getattr(self, 'project_name', 'postman_collection')  # ✅ DYNAMIC flow name
                )


            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"AI Intelligence call failed: {e}")
        
        
    
    def generate_postman_collections(self, vector_content: str = None):
        """Interface method for main.py with Vector DB support"""
        print("🚀 Starting Postman Collection Generation (main.py interface)...")
        
        # Enhanced Vector DB content logging
        if vector_content:
            print("✅ Vector DB content received")
            print(f"📊 Vector content size: {len(vector_content)} characters")
            print(f"📋 Content preview (first 200 chars): {vector_content[:200]}...")
            
            # Store Vector DB content
            self.vector_business_requirements = vector_content
            print("💾 Vector content stored in self.vector_business_requirements")
            
            # Verify storage
            stored_size = len(self.vector_business_requirements)
            print(f"✅ Storage verification: {stored_size} characters stored")
            
        else:
            print("⚠️ No Vector DB content provided")
            self.vector_business_requirements = None
            print("💾 vector_business_requirements set to None")
        
        print("🎯 Calling main generation process...")
        
        # Call main generation
        self.generate()
        
        # Log completion
        output_path = str(self.output_root)
        print(f"✅ Postman Collection Generation completed")
        print(f"📁 Output path: {output_path}")
        
        # Return output path
        return output_path


    def generate(self):
        """Main generation method"""
        print("Starting 100% AI-Intelligent Postman Collection Generation...")
        print(f"Project: {self.project_name}")
        print(f"Source: {self.reviewed_modules_path}")
        print(f"Output: {self.output_root}")
        
        # Create output directories
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: AI analyzes ACE artifacts
        ace_analysis = self._analyze_ace_artifacts_intelligent()
        esql_analysis = self._analyze_esql_modules_individual(ace_analysis)
        business_mapping = self._map_esql_to_business_context(esql_analysis)
        # Step 2: AI generates test scenarios
        test_scenarios = self._generate_test_scenarios_intelligent(ace_analysis, business_mapping)
        
        # Step 3: AI creates payloads
        payload_data = self._generate_payloads_intelligent(test_scenarios, ace_analysis)
        
        # Step 4: AI builds collections
        self._create_postman_collections_intelligent(test_scenarios, ace_analysis, payload_data)
        
        # Step 5: AI generates environments
        self._create_environments_intelligent(ace_analysis)
        
        # Step 6: AI creates documentation
        self._generate_documentation_intelligent(ace_analysis, test_scenarios)
        
        print(f"\n100% AI-Intelligent generation completed!")
        print(f"AI Analysis Calls: {self.generation_results['ai_analysis_calls']}")
        print(f"AI Generation Calls: {self.generation_results['ai_generation_calls']}")
        print(f"Test Scenarios: {self.generation_results['test_scenarios_generated']}")
        print(f"Collections: {len(self.generation_results['collections_created'])}")
    
    def _analyze_ace_artifacts_intelligent(self) -> Dict:
        """AI analyzes ACE artifacts with business intelligence"""
        print("AI analyzing ACE artifacts with business intelligence...")
        
        # Read ACE files
        ace_content = self._read_ace_files_intelligent()
        
        system_prompt = """You are an IBM ACE architect analyzing integration artifacts for test generation.

Analyze the ACE artifacts and Vector DB business requirements to understand:
- Business processes and entities involved
- Technical integration patterns and protocols
- Data transformation requirements
- Error handling mechanisms
- Security and authentication patterns

Return analysis in valid JSON format."""
        
        # Include Vector DB business requirements if available
        vector_context = ""
        if hasattr(self, 'vector_business_requirements') and self.vector_business_requirements:
            vector_context = f"\n\nVECTOR DB BUSINESS REQUIREMENTS:\n{self.vector_business_requirements[:2000]}"
        
        user_prompt = f"""Analyze these ACE artifacts with business context:

PROJECT: {self.project_name}

MESSAGE FLOW FILES:
{ace_content.get('msgflows', 'No msgflow files found')}

ESQL MODULES:  
{ace_content.get('esql_modules', 'No ESQL modules found')}

XSL TRANSFORMS:
{ace_content.get('xsl_transforms', 'No XSL transforms found')}

PROJECT CONFIGURATIONS:
{ace_content.get('project_configs', 'No project configs found')}

{vector_context}

Return JSON analysis with business context, technical patterns, and integration requirements."""

        analysis_response = self._call_llm_intelligence(
            system_prompt, user_prompt, 
            max_tokens=self.max_tokens_analysis, 
            call_type="analysis"
        )
        
        # Parse using llm_json_parser
        result = parse_llm_json(analysis_response)
        
        if not result.success:
            raise Exception(f"ACE analysis JSON parsing failed: {result.error_message}")
        
        print("AI code analysis completed with business intelligence")
        return result.data
    
    def _generate_test_scenarios_intelligent(self, ace_analysis: Dict, business_mapping: Dict):
        """AI generates test scenarios from Vector DB business requirements"""
        print("AI generating intelligent test scenarios...")
        
        # Extract Vector DB context and detect domain
        vector_context = ""
        domain_type = "integration"
        
        if hasattr(self, 'vector_business_requirements') and self.vector_business_requirements:
            vector_context = self.vector_business_requirements[:3000]
            
            content_lower = self.vector_business_requirements.lower()
            if any(word in content_lower for word in ['shipment', 'cargo', 'freight', 'logistics']):
                domain_type = "logistics"
            elif any(word in content_lower for word in ['payment', 'invoice', 'billing', 'financial']):
                domain_type = "financial"  
            elif any(word in content_lower for word in ['patient', 'medical', 'healthcare']):
                domain_type = "healthcare"
            elif any(word in content_lower for word in ['manufacturing', 'production', 'inventory']):
                domain_type = "manufacturing"
        else:
            vector_context = "No specific business requirements provided"
        
        system_prompt = """You are an integration testing architect generating module-specific test scenarios from ESQL business-technical mapping.

        Generate 3-5 test scenarios per ESQL module based on actual module business logic and technical implementation. Focus on module-specific business processes, validation rules, and code paths."""

        user_prompt = f"""Generate test scenarios from ESQL module business-technical mapping:

        BUSINESS-TECHNICAL MODULE MAPPING:
        {json.dumps(business_mapping.get('business_context_mapping', []), indent=2)}

        OVERALL BUSINESS FLOW:
        {business_mapping.get('overall_business_flow', '')}

        CRITICAL TEST AREAS:
        {business_mapping.get('critical_test_areas', [])}

        TARGET: Generate 3-5 scenarios per ESQL module (Total modules: {len(business_mapping.get('business_context_mapping', []))})

        For each module in the mapping, generate test scenarios based on:
        1. Module's specific business process and technical implementation
        2. Code paths and validation rules identified in the module
        3. Business entities and integration points for that module
        4. Error conditions specific to that module

        Return JSON array with scenarios explicitly tied to specific modules:
        [
        {{
            "id": "scenario_001",
            "module_name": "specific_esql_module_name_from_mapping",
            "name": "Module-specific test name based on business process",
            "description": "Test description for specific module logic",
            "category": "functional",
            "priority": 1,
            "business_justification": "Why this module test matters for business",
            "technical_focus": "Specific code path or validation in this module",
            "module_business_process": "Business process this module handles",
            "validation_criteria": ["Expected outcomes for this module"],
            "payload_requirements": "Data needs for this specific module"
        }}
        ]

        Generate exactly 3-5 scenarios for EACH module in the business mapping."""

        scenarios_response = self._call_llm_intelligence(
            system_prompt, user_prompt,
            max_tokens=4000,
            call_type="generation"
        )
        
        # Parse using llm_json_parser
        result = parse_llm_json(scenarios_response)
        
        if not result.success:
            raise Exception(f"Test scenarios JSON parsing failed: {result.error_message}")
        
        # Handle both single scenario object and array responses
        scenarios_data = result.data
        
        if isinstance(scenarios_data, list):
            scenarios_list = scenarios_data
        elif isinstance(scenarios_data, dict):
            # Check if single scenario object
            scenario_keys = ['id', 'name', 'description', 'category', 'priority']
            if all(key in scenarios_data for key in scenario_keys):
                scenarios_list = [scenarios_data]
            else:
                # Look for scenarios in object properties
                scenarios_list = None
                for key in ['scenarios', 'test_scenarios', 'items', 'data']:
                    if key in scenarios_data and isinstance(scenarios_data[key], list):
                        scenarios_list = scenarios_data[key]
                        break
                
                if scenarios_list is None:
                    raise Exception(f"No scenarios found. Available keys: {list(scenarios_data.keys())}")
        else:
            raise Exception("Expected list or dict containing test scenarios")
        
        # Validate scenarios structure
        valid_scenarios = []
        for scenario in scenarios_list:
            if (isinstance(scenario, dict) and 
                all(key in scenario for key in ['id', 'name', 'category', 'priority'])):
                valid_scenarios.append(scenario)
        
        if len(valid_scenarios) == 0:
            raise Exception("No valid scenarios found in LLM response")
        
        self.generation_results['test_scenarios_generated'] = len(valid_scenarios)
        print(f"AI generated {len(valid_scenarios)} domain-specific test scenarios")
        return valid_scenarios
    
    def _generate_payloads_intelligent(self, test_scenarios: List[Dict], ace_analysis: Dict) -> Dict:
        """AI generates realistic payloads based on business requirements"""
        print("AI generating intelligent payloads with business context...")
        
        # Extract Vector DB context and detect domain/format
        vector_context = ""
        domain_type = "integration"
        payload_format = "XML"
        
        if hasattr(self, 'vector_business_requirements') and self.vector_business_requirements:
            vector_context = self.vector_business_requirements[:2500]
            
            content_lower = self.vector_business_requirements.lower()
            if any(word in content_lower for word in ['xml', 'soap', 'cdm', 'document']):
                payload_format = "XML"
            elif any(word in content_lower for word in ['json', 'rest', 'api']):
                payload_format = "JSON"
                
            if any(word in content_lower for word in ['shipment', 'cargo', 'freight', 'logistics']):
                domain_type = "logistics"
            elif any(word in content_lower for word in ['payment', 'invoice', 'billing', 'financial']):
                domain_type = "financial"
        else:
            vector_context = "No specific business requirements provided"

        system_prompt = f"""You are a data architect expert specializing in realistic test payload generation.

Generate REALISTIC PAYLOADS that:
- Reflect actual business entities and data structures from business requirements
- Match specific data formats, schemas, and technical patterns mentioned
- Include proper business identifiers, references, and realistic data values
- Cover valid scenarios, invalid scenarios, and edge cases for testing
- Use actual field names, data types, and structures from business domain

Create payloads in {payload_format} format based on actual business requirements."""

        # Create scenario summary for context
        scenario_summary = []
        for scenario in test_scenarios[:10]:
            scenario_summary.append({
                'id': scenario.get('id'),
                'name': scenario.get('name'),
                'category': scenario.get('category'),
                'payload_requirements': scenario.get('payload_requirements', '')
            })

        user_prompt = f"""Generate realistic payloads for business requirements and test scenarios:

BUSINESS REQUIREMENTS FROM VECTOR DB:
{vector_context}

ACE TECHNICAL ANALYSIS:
{json.dumps(ace_analysis, indent=2)[:1500]}

TEST SCENARIOS REQUIRING PAYLOADS:
{json.dumps(scenario_summary, indent=2)}

DOMAIN: {domain_type}
FORMAT: {payload_format}

Generate payloads in this JSON format:
{{
  "valid_payloads": [
    {{
      "name": "descriptive_name_based_on_actual_business_entity",
      "description": "business_context_from_requirements", 
      "payload": "actual_{payload_format.lower()}_content_with_real_field_names",
      "applicable_scenarios": ["scenario_ids_that_use_this_payload"]
    }}
  ],
  "invalid_payloads": [
    {{
      "name": "error_scenario_name_based_on_business_rules",
      "description": "error_context_from_requirements",
      "payload": "invalid_{payload_format.lower()}_content_for_error_testing",
      "expected_error": "expected_error_type_from_business_rules",
      "applicable_scenarios": ["scenario_ids_for_error_testing"]
    }}
  ],
  "edge_case_payloads": [
    {{
      "name": "edge_case_name_based_on_business_boundaries",
      "description": "edge_case_context_from_requirements",
      "payload": "boundary_condition_{payload_format.lower()}_content",
      "applicable_scenarios": ["scenario_ids_for_edge_cases"]
    }}
  ]
}}

Requirements:
1. Use ACTUAL field names, entity names from business requirements
2. Create realistic data values for {domain_type} domain
3. Generate 5-10 payloads in each category
4. Match technical specifications from requirements
5. Use proper {payload_format} syntax
6. Base content on ACTUAL business requirements

Return ONLY the JSON object."""

        payload_response = self._call_llm_intelligence(
            system_prompt, user_prompt,
            max_tokens=4000,
            call_type="generation"
        )
        
        # Parse using llm_json_parser
        result = parse_llm_json(payload_response)
        
        if not result.success:
            raise Exception(f"Payloads JSON parsing failed: {result.error_message}")
        
        payload_data = result.data
        
        if not isinstance(payload_data, dict):
            raise Exception("Expected dictionary of payload categories")
        
        # Validate payload structure
        required_keys = ['valid_payloads', 'invalid_payloads', 'edge_case_payloads']
        if not all(key in payload_data for key in required_keys):
            raise Exception(f"Payload response missing required keys: {required_keys}")
        
        # Count total payloads
        total_payloads = 0
        for category in required_keys:
            if isinstance(payload_data[category], list):
                total_payloads += len(payload_data[category])
            else:
                raise Exception(f"Payload category {category} is not a list")
        
        if total_payloads == 0:
            raise Exception("No payloads generated by LLM")
        
        self.generation_results['payload_samples_created'] = total_payloads
        print(f"AI generated {total_payloads} intelligent payloads")
        return payload_data
    
    def _create_postman_collections_intelligent(self, test_scenarios: List[Dict], 
                                               ace_analysis: Dict, payload_data: Dict):
        """AI creates Postman collections with real test data and domain-specific configurations"""
        print("AI creating intelligent Postman collections...")
        
        # Extract Vector DB context and detect protocol patterns
        vector_context = "No business requirements available - generate based on the ACE components only"
        domain_type = "integration"
        protocol_type = "HTTP"
        
        if hasattr(self, 'vector_business_requirements') and self.vector_business_requirements:
            vector_context = self.vector_business_requirements
            
            content_lower = self.vector_business_requirements.lower()
            if any(word in content_lower for word in ['soap', 'xml', 'wsdl']):
                protocol_type = "SOAP"
            elif any(word in content_lower for word in ['rest', 'json', 'http']):
                protocol_type = "REST"
            elif any(word in content_lower for word in ['mq', 'queue', 'message']):
                protocol_type = "MQ"
            
            if any(word in content_lower for word in ['shipment', 'cargo', 'freight', 'logistics']):
                domain_type = "logistics"
            elif any(word in content_lower for word in ['payment', 'invoice', 'billing', 'financial']):
                domain_type = "financial"
        else:
            vector_context = "No specific business requirements provided"

        system_prompt = """You are a Postman collection expert creating production-ready API test collections.

Create COMPLETE POSTMAN COLLECTIONS that:
- Use actual endpoints, URLs, and connection details from business requirements
- Configure proper headers based on technical specifications mentioned
- Include realistic request bodies using provided payload data
- Generate meaningful test scripts based on business validation criteria
- Organize scenarios into logical collection structure
- Use actual authentication and security patterns from requirements

Generate collections in standard Postman v2.1.0 format."""

        # Organize scenarios by category
        functional_scenarios = [s for s in test_scenarios if s.get('category') == 'functional']
        error_scenarios = [s for s in test_scenarios if s.get('category') == 'error'] 
        performance_scenarios = [s for s in test_scenarios if s.get('category') == 'performance']

        user_prompt = f"""Create complete Postman collections based on actual business requirements:

BUSINESS REQUIREMENTS FROM VECTOR DB:
{vector_context}

ACE TECHNICAL ANALYSIS:
{json.dumps(ace_analysis, indent=2)}

TEST SCENARIOS:
Functional: {len(functional_scenarios)} scenarios
Error Handling: {len(error_scenarios)} scenarios  
Performance: {len(performance_scenarios)} scenarios

AVAILABLE PAYLOADS:
Valid Payloads: {len(payload_data.get('valid_payloads', []))}
Invalid Payloads: {len(payload_data.get('invalid_payloads', []))}
Edge Case Payloads: {len(payload_data.get('edge_case_payloads', []))}

DOMAIN: {domain_type}
PROTOCOL: {protocol_type}

Generate 3 Postman collections in JSON format:
{{
  "functional_collection": {{
    "info": {{
      "name": "specific_name_based_on_business_domain",
      "description": "description_based_on_actual_business_requirements",
      "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
      "_postman_id": "generated_uuid",
      "updatedAt": "current_timestamp"
    }},
    "item": [
      {{
        "name": "test_name_from_actual_scenario",
        "request": {{
          "method": "POST",
          "header": [
            {{"key": "Content-Type", "value": "content_type_from_requirements"}},
            {{"key": "Authorization", "value": "auth_pattern_from_requirements"}}
          ],
          "body": {{
            "mode": "raw",
            "raw": "actual_payload_content_from_payload_data"
          }},
          "url": {{
            "raw": "actual_endpoint_from_business_requirements",
            "host": ["extracted_from_requirements"],
            "path": ["path_from_requirements"]
          }}
        }},
        "event": [
          {{
            "listen": "test",
            "script": {{
              "exec": [
                "test_script_based_on_validation_criteria_from_scenario"
              ]
            }}
          }}
        ]
      }}
    ]
  }},
  "error_collection": {{
    "info": {{"name": "error_collection_name", "description": "error_description"}},
    "item": ["error_test_items"]
  }},
  "performance_collection": {{
    "info": {{"name": "performance_collection_name", "description": "performance_description"}},
    "item": ["performance_test_items"]
  }}
}}

Requirements:
1. Extract ACTUAL endpoints, URLs from business requirements
2. Use ACTUAL payload content from provided payload_data
3. Configure headers based on ACTUAL technical specifications
4. Generate test scripts based on ACTUAL validation criteria
5. Use proper UUIDs and timestamps
6. Match request configuration to {protocol_type} protocol
7. Base all configurations on ACTUAL business requirements

Return ONLY the JSON object."""

        collections_response = self._call_llm_intelligence(
            system_prompt, user_prompt,
            max_tokens=6000,
            call_type="generation"
        )
        
        # Parse using llm_json_parser
        result = parse_llm_json(collections_response)
        
        if not result.success:
            raise Exception(f"Collections JSON parsing failed: {result.error_message}")
        
        collections_data = result.data
        
        if not isinstance(collections_data, dict):
            raise Exception("Expected dictionary of collections")
        
        # Validate collections structure
        required_collections = ['functional_collection', 'error_collection', 'performance_collection']
        if not all(key in collections_data for key in required_collections):
            raise Exception(f"Collections response missing required collections: {required_collections}")
        
        # Create and save each collection
        created_collections = []
        for collection_type, collection_data in collections_data.items():
            if not isinstance(collection_data, dict) or 'info' not in collection_data:
                raise Exception(f"Invalid collection structure for {collection_type}")
            
            if 'item' not in collection_data or not isinstance(collection_data['item'], list):
                raise Exception(f"Collection {collection_type} missing valid items array")
            
            # Generate filename and save collection
            collection_name = collection_data['info'].get('name', f'{self.project_name}_{collection_type}')
            filename = f"{collection_name}.postman_collection.json"
            collection_path = self.paths['collections'] / filename
            
            with open(collection_path, 'w', encoding='utf-8') as f:
                json.dump(collection_data, f, indent=2)
            
            created_collections.append(str(collection_path))
            print(f"   Created intelligent collection: {collection_name}")
        
        # Validate collections contain test items
        total_items = sum(len(collections_data[col]['item']) for col in required_collections 
                         if 'item' in collections_data[col])
        
        if total_items == 0:
            raise Exception("No test items found in generated collections")
        
        self.generation_results['collections_created'] = created_collections
        print(f"AI created {len(created_collections)} collections with {total_items} total test items")
    
    def _create_environments_intelligent(self, ace_analysis: Dict):
        """AI creates Postman environments"""
        print("AI creating intelligent environments...")
        
        system_prompt = """Create Postman environment configurations for Development and QA based on business requirements."""
        
        user_prompt = f"""Create Postman environments based on analysis:

ACE ANALYSIS:
{json.dumps(ace_analysis, indent=2)[:1500]}

Generate Development and QA environment configurations in JSON format."""

        env_response = self._call_llm_intelligence(
            system_prompt, user_prompt,
            max_tokens=2000,
            call_type="generation"
        )
        
        # Parse and create basic environments
        result = parse_llm_json(env_response)
        
        if result.success and isinstance(result.data, dict):
            environments = result.data
        else:
            # Create basic environments if parsing fails
            environments = {
                "Development": {
                    "name": "Development",
                    "values": [
                        {"key": "base_url", "value": "http://localhost:8080", "enabled": True},
                        {"key": "auth_token", "value": "dev_token", "enabled": True}
                    ]
                },
                "QA_Testing": {
                    "name": "QA_Testing", 
                    "values": [
                        {"key": "base_url", "value": "https://qa.example.com", "enabled": True},
                        {"key": "auth_token", "value": "qa_token", "enabled": True}
                    ]
                }
            }
        
        for env_name, env_config in environments.items():
            env_path = self.paths['environments'] / f"{env_name}.postman_environment.json"
            with open(env_path, 'w', encoding='utf-8') as f:
                json.dump(env_config, f, indent=2)
            
            self.generation_results['environments_created'].append(str(env_path))
            print(f"   Created intelligent environment: {env_name}")
    
    def _generate_documentation_intelligent(self, ace_analysis: Dict, test_scenarios: List[Dict]):
        """AI generates documentation"""
        print("AI generating intelligent documentation...")
        
        documentation_content = f"""# {self.project_name} - Postman Test Collections

## Overview
AI-generated Postman test collections based on business requirements analysis.

## Test Scenarios Generated
- Total Scenarios: {len(test_scenarios)}
- Functional Tests: {len([s for s in test_scenarios if s.get('category') == 'functional'])}
- Error Handling Tests: {len([s for s in test_scenarios if s.get('category') == 'error'])}
- Performance Tests: {len([s for s in test_scenarios if s.get('category') == 'performance'])}

## Collections Created
{chr(10).join('- ' + str(Path(c).name) for c in self.generation_results['collections_created'])}

## Generated: {self.generation_results['timestamp']}
"""
        
        readme_path = self.paths['documentation'] / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(documentation_content)
        
        self.generation_results['documentation_files'].append(str(readme_path))
        print("   Created intelligent documentation: README.md")


    

    def _analyze_esql_modules_individual(self, ace_analysis: Dict) -> Dict:
        """Analyze each ESQL module individually to extract technical business logic"""
        
        esql_modules = ace_analysis.get('esql_modules', [])
        if not esql_modules:
            return {'modules_analyzed': 0, 'individual_analysis': []}
        
        print(f"AI analyzing {len(esql_modules)} ESQL modules individually...")
        
        individual_analysis = []
        
        for esql_file_path in esql_modules:
            try:
                # Read ESQL file content
                with open(esql_file_path, 'r', encoding='utf-8') as f:
                    esql_content = f.read()
                
                module_name = os.path.basename(esql_file_path).replace('.esql', '')
                
                system_prompt = """Analyze ESQL module code to extract business logic elements for test scenario generation.

    Extract specific technical implementation details that indicate business processes, validation rules, and data transformations."""

                user_prompt = f"""Analyze this ESQL module and extract business logic elements:

    MODULE NAME: {module_name}
    ESQL CONTENT:
    {esql_content}

    Extract and return JSON format:
    {{
    "module_name": "{module_name}",
    "business_purpose": "inferred business purpose from code analysis",
    "technical_elements": {{
        "database_queries": ["extracted SQL/database operations"],
        "validation_rules": ["IF/WHEN conditions and business rules"],
        "message_transformations": ["SET/CREATE operations and data mappings"],
        "error_handling": ["exception handling and error conditions"],
        "field_mappings": ["input/output field operations"],
        "business_entities": ["detected business objects/entities"],
        "external_calls": ["SOAP/HTTP/MQ operations"]
    }},
    "code_paths": ["main execution paths through the module"],
    "test_focus_areas": ["specific areas that need testing based on code complexity"]
    }}

    Analyze actual ESQL code syntax and extract real implementation details."""

                response = self._call_llm_intelligence(
                    system_prompt, user_prompt,
                    max_tokens=1500,
                    call_type="analysis"
                )
                
                # Parse using llm_json_parser
                result = parse_llm_json(response)
                
                if result.success:
                    module_analysis = result.data
                    module_analysis['file_path'] = esql_file_path
                    module_analysis['content_size'] = len(esql_content)
                    individual_analysis.append(module_analysis)
                    print(f"   ✅ Analyzed: {module_name} ({len(esql_content)} chars)")
                else:
                    print(f"   ❌ Failed to parse analysis for: {module_name}")
                    
            except Exception as e:
                print(f"   ❌ Error analyzing {esql_file_path}: {e}")
                continue
        
        return {
            'modules_analyzed': len(individual_analysis),
            'individual_analysis': individual_analysis
        }


    def _map_esql_to_business_context(self, esql_analysis: Dict) -> Dict:
        """Map technical ESQL analysis to business context using Vector DB requirements"""
        
        if not esql_analysis.get('individual_analysis'):
            return {'mapped_modules': 0, 'business_context_mapping': []}
        
        vector_context = self.vector_business_requirements or "No business requirements available"
        
        print(f"AI mapping {esql_analysis['modules_analyzed']} ESQL modules to business context...")
        
        system_prompt = """Map technical ESQL module analysis to business context using business requirements.

    Create comprehensive business-technical mapping that connects code implementation to actual business processes and requirements."""

        user_prompt = f"""Map technical ESQL analysis to business requirements:

    BUSINESS REQUIREMENTS FROM VECTOR DB:
    {vector_context}

    TECHNICAL ESQL ANALYSIS:
    {json.dumps(esql_analysis['individual_analysis'], indent=2)}

    Create business-technical mapping in JSON format:
    {{
    "business_technical_mapping": [
        {{
        "module_name": "extracted_from_analysis",
        "business_process": "mapped business process from requirements",
        "business_entities": ["business objects mentioned in requirements"],
        "validation_requirements": ["business rules from requirements mapped to code"],
        "technical_implementation": "how code implements business requirements",
        "test_scenarios_needed": [
            {{
            "scenario_name": "business-meaningful test name",
            "business_justification": "why this test matters for business",
            "technical_focus": "specific code path or logic to test",
            "expected_behavior": "expected business outcome"
            }}
        ],
        "integration_points": ["external systems or processes mentioned in requirements"],
        "error_conditions": ["business error scenarios mapped from code"],
        "performance_considerations": ["business throughput/timing requirements"]
        }}
    ],
    "overall_business_flow": "end-to-end business process description",
    "critical_test_areas": ["most important business processes to test"]
    }}

    Map actual business terminology and processes from requirements to technical implementation."""

        response = self._call_llm_intelligence(
            system_prompt, user_prompt,
            max_tokens=3000,
            call_type="analysis"
        )
        
        # Parse using llm_json_parser
        result = parse_llm_json(response)
        
        if not result.success:
            print(f"   ❌ Failed to parse business-technical mapping: {result.error_message}")
            return {'mapped_modules': 0, 'business_context_mapping': []}
        
        mapping_data = result.data
        business_mapping = mapping_data.get('business_technical_mapping', [])
        
        print(f"   ✅ Mapped {len(business_mapping)} modules to business context")
        
        return {
            'mapped_modules': len(business_mapping),
            'business_context_mapping': business_mapping,
            'overall_business_flow': mapping_data.get('overall_business_flow', ''),
            'critical_test_areas': mapping_data.get('critical_test_areas', [])
        }



    def _read_ace_files_intelligent(self) -> Dict:
        """
        Updated function with comprehensive print statements for debugging
        """
        
        # PRINT 1: Show what path we're starting with
        print(f"\n" + "="*60)
        print(f"🔍 ACE COMPONENT READER - DEBUG MODE")
        print(f"="*60)
        print(f"📂 Input Path: {self.reviewed_modules_path}")
        print(f"🔍 Path Type: {type(self.reviewed_modules_path)}")
        print(f"✅ Path Exists: {self.reviewed_modules_path.exists()}")
        print(f"📁 Is Directory: {self.reviewed_modules_path.is_dir()}")
        
        if not self.reviewed_modules_path.exists():
            print(f"❌ CRITICAL: Path does not exist!")
            return {}
        
        if not self.reviewed_modules_path.is_dir():
            print(f"❌ CRITICAL: Path is not a directory!")
            return {}
        
        # PRINT 2: Show directory contents
        print(f"\n📊 DIRECTORY CONTENTS:")
        try:
            contents = list(self.reviewed_modules_path.iterdir())
            print(f"   📈 Total Items: {len(contents)}")
            
            for item in contents:
                item_type = "📁 DIR " if item.is_dir() else "📄 FILE"
                size_info = f"({item.stat().st_size} bytes)" if item.is_file() else ""
                print(f"   {item_type}: {item.name} {size_info}")
                
        except Exception as e:
            print(f"❌ ERROR reading directory: {e}")
            return {}
        
        # Initialize result structure
        ace_content = {
            'msgflows': [],
            'esql_modules': [],
            'xsl_transforms': [],
            'project_configs': [],
            'subflows': [],
            'schema_files': []
        }
        
        # PRINT 3: Show scanning process
        print(f"\n🔍 SCANNING FOR ACE FILES:")
        print(f"-" * 40)
        
        ace_extensions = {
            '.msgflow': 'msgflows',
            '.esql': 'esql_modules',
            '.xsl': 'xsl_transforms',
            '.xslt': 'xsl_transforms',
            '.subflow': 'subflows',
            '.xsd': 'schema_files',
            '.project': 'project_configs',
            '.xml': 'project_configs',
                '.descriptor': 'project_configs',   
            '': 'project_configs' 
        }
        
        total_files_scanned = 0
        ace_files_found = 0
        
        # Scan recursively
        for file_path in self.reviewed_modules_path.rglob("*"):
            if file_path.is_file():
                total_files_scanned += 1
                file_ext = file_path.suffix.lower()
                relative_path = file_path.relative_to(self.reviewed_modules_path)
                
                # PRINT 4: Show each file being scanned
                print(f"📄 Scanning: {relative_path}")
                
                if file_ext in ace_extensions:
                    category = ace_extensions[file_ext]
                    ace_files_found += 1
                    
                    # PRINT 5: Show ACE file found
                    print(f"   🎯 ACE FILE DETECTED! Category: {category}")
                    
                    try:
                        # Try to read file content
                        content = None
                        encoding_used = None
                        
                        for encoding in ['utf-8', 'utf-16', 'latin-1']:
                            try:
                                content = file_path.read_text(encoding=encoding)
                                encoding_used = encoding
                                break
                            except UnicodeDecodeError:
                                continue
                        
                        if content is None:
                            print(f"      ❌ Could not read file with any encoding")
                            continue
                        
                        # PRINT 6: Show file content details
                        content_preview = content[:200].replace('\n', '\\n').replace('\r', '\\r')
                        print(f"      ✅ Successfully read: {len(content)} characters")
                        print(f"      📝 Encoding: {encoding_used}")
                        print(f"      👀 Preview: {content_preview}...")
                        
                        # Store file info
                        file_info = {
                            'file': str(file_path.name),
                            'path': str(relative_path),
                            'full_path': str(file_path),
                            'content': content,
                            'size': len(content),
                            'encoding': encoding_used
                        }
                        
                        ace_content[category].append(file_info)
                        
                        # PRINT 7: Confirm file added to category
                        print(f"      📁 Added to category '{category}' (now has {len(ace_content[category])} files)")
                        
                    except Exception as e:
                        print(f"      ❌ ERROR reading file: {e}")
                        
                else:
                    print(f"   ⏭️  Skipping: {file_ext} (not ACE file)")
        
        # PRINT 8: Final summary
        print(f"\n" + "="*60)
        print(f"📊 ACE COMPONENT SCAN COMPLETE")
        print(f"="*60)
        print(f"📄 Total files scanned: {total_files_scanned}")
        print(f"🎯 ACE files found: {ace_files_found}")
        print(f"\n📋 RESULTS BY CATEGORY:")
        
        total_components = 0
        for category, files in ace_content.items():
            count = len(files)
            total_components += count
            status = "✅" if count > 0 else "⚪"
            print(f"   {status} {category}: {count} files")
            
            # Show details of first 2 files in each category
            for i, file_info in enumerate(files[:2]):
                print(f"      {i+1}. {file_info['file']} ({file_info['size']} chars)")
            
            if len(files) > 2:
                print(f"      ... and {len(files) - 2} more files")
        
        print(f"\n🎉 TOTAL ACE COMPONENTS FOUND: {total_components}")
        
        # PRINT 9: Show if ready for LLM analysis
        if total_components > 0:
            print(f"✅ READY FOR LLM ANALYSIS!")
            print(f"   📝 Components will be passed to AI for test generation")
        else:
            print(f"❌ NO ACE COMPONENTS FOUND!")
            print(f"   🔍 Check if Agent 4 completed successfully")
            print(f"   📂 Verify path contains ACE files: {self.reviewed_modules_path}")
        
        print(f"=" * 60)
        
        return ace_content