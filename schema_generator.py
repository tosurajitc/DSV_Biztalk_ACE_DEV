#!/usr/bin/env python3
"""
Schema Generator Module v1.0 - ACE Module Creator
Purpose: Extract schema requirements from Confluence PDF + component mapping JSON + LLM 
         Generate ACE-compatible XSD files with ZERO hardcoded fallbacks
Input: Confluence PDF + component mapping JSON + LLM analysis
Output: Creates schemas/ folder with all schema files needed for the project
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv
import streamlit as st
import time
import logging


load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SchemaGenerator:
    """
    ACE Schema Generator with complete LLM integration
    NO HARDCODED FALLBACKS - All schemas generated via LLM analysis
    """
    
    def __init__(self, groq_api_key: str = None):
        """Initialize with Groq LLM client"""
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY must be provided or set in environment")
        
        self.llm = Groq(api_key=self.groq_api_key)
        self.groq_model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')


        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY must be provided or set in environment")
        
        self.llm = Groq(api_key=self.groq_api_key)
        self.groq_model = os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')
        
        # Processing results tracking
        self.processed_pdf_content = None
        self.processed_json_mappings = None
        self.generated_schemas = []
        self.llm_analysis_calls = 0
        self.llm_generation_calls = 0
    

    def _call_llm_with_retry(self, prompt: str, max_retries: int = 5, timeout: int = 180) -> Optional[str]:
        """
        Call LLM with retry logic for timeouts.
        Private method - for internal use only.
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"🤖 LLM attempt {attempt + 1}/{max_retries}...")
                
                response = self.llm.chat.completions.create(
                    model=self.groq_model,
                    messages=[
                        {"role": "system", "content": "You are an expert IBM ACE developer."},
                        {"role": "user", "content": prompt}
                    ],
                    timeout=timeout,
                    temperature=0.3
                )
                
                result = response.choices[0].message.content
                logger.info(f"✅ LLM call succeeded")
                return result
                
            except TimeoutError as e:
                if attempt == max_retries - 1:
                    logger.error(f"❌ Failed after {max_retries} attempts")
                    return None
                
                delay = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                logger.warning(f"⏱️ Timeout. Retrying in {delay}s...")
                time.sleep(delay)
                
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    time.sleep(delay)
                else:
                    return None
        
        return None



    def generate_schemas(self, 
                    vector_content: str,  # New: Vector DB content instead of PDF path
                    component_mapping_json_path: str,
                    output_dir: str) -> Dict[str, Any]:
        """
        Main entry point - Generate all XSD schemas using LLM analysis
        
        Args:
            confluence_pdf_path: Path to Confluence PDF with business requirements
            component_mapping_json_path: Path to component mapping JSON
            output_dir: Output directory for generated schemas
            
        Returns:
            Dict with generation results and metadata
        """
        print("🎯 Starting Schema Generation - Full LLM Processing Mode")
        print("📋 NO HARDCODED FALLBACKS - Pure AI-driven schema extraction")
        
        # Step 1: Process ALL inputs completely
        print("\n📄 Step 1: Processing ALL inputs for LLM analysis...")
        vector_processed_content = self._process_vector_content(vector_content)
        json_mappings = self._extract_complete_json_mappings(component_mapping_json_path)
        
        # Step 2: LLM Analysis of requirements
        print("\n🧠 Step 2: LLM analyzing schema requirements...")
        schema_requirements = self._llm_analyze_schema_requirements(vector_processed_content, json_mappings)
        
        # Step 3: LLM Generation of XSD schemas
        print("\n⚡ Step 3: LLM generating ACE-compatible XSD schemas...")
        generated_schemas = self._llm_generate_xsd_schemas(schema_requirements, output_dir)
        
        # Step 4: Create output structure
        schemas_dir = os.path.join(output_dir, 'schemas')
        os.makedirs(schemas_dir, exist_ok=True)
        
        # Step 5: Write generated schemas to files
        print("\n💾 Step 5: Writing generated schemas to files...")
        schema_files = self._write_schema_files(generated_schemas, schemas_dir)
        
        return {
            'status': 'success',
            'schemas_generated': len(schema_files),
            'schema_files': schema_files,
            'output_directory': schemas_dir,
            'llm_analysis_calls': self.llm_analysis_calls,
            'llm_generation_calls': self.llm_generation_calls,
            'processing_metadata': {
            'vector_content_length': len(vector_content),
            'schema_focus_keywords': vector_processed_content.get('focus_keywords', []),
            'json_components_processed': len(json_mappings.get('components', [])),
            'timestamp': datetime.now().isoformat()
        }
        }
    
    def _process_vector_content(self, vector_content: str) -> Dict[str, Any]:
        """
        Process Vector DB content for schema analysis
        Simple wrapper around Vector DB focused content
        """
        return {
            'schema_content': vector_content,
            'content_length': len(vector_content),
            'focus_keywords': ['schema', 'xsd', 'data structure', 'validation'],
            'data_structures': self._extract_data_structure_hints(vector_content),
            'source': 'vector_db'
        }

    def _extract_data_structure_hints(self, content: str) -> List[str]:
        """Extract data structure patterns from Vector content"""
        patterns = ['CDM', 'Document', 'Message', 'Header', 'Target', 'Source']
        found = [pattern for pattern in patterns if pattern.lower() in content.lower()]
        return found
    
    def _get_vector_content_for_schema_generator(self):
        """Get Vector DB content focused on schema requirements"""
        if not hasattr(self, 'vector_pipeline') or not self.vector_pipeline:
            # Get from session state (already established)
            import streamlit as st
            if not st.session_state.get('vector_pipeline'):
                raise Exception("Vector DB pipeline not available")
            return st.session_state.vector_pipeline.search_engine.get_agent_content("schema_generator")
        return self.vector_pipeline.search_engine.get_agent_content("schema_generator")
    
    def _extract_complete_json_mappings(self, json_path: str) -> Dict[str, Any]:
        """
        Extract COMPLETE JSON mapping content for LLM processing
        NO filtering - all mapping data goes to LLM
        """
        print(f"  🗂️ Extracting complete JSON mappings from: {json_path}")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Component mapping JSON not found: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Validate JSON structure
            if not isinstance(json_data, dict):
                raise ValueError("JSON must be a dictionary object")
            
            print(f"  ✅ JSON processed: {len(str(json_data))} characters")
            self.processed_json_mappings = json_data
            return json_data
            
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to process JSON: {str(e)}")
        
        
    
    def _llm_analyze_schema_requirements(self, vector_processed_content: Dict, json_mappings: Dict) -> Dict[str, Any]:
        """
        LLM analyzes ALL content to extract schema requirements
        NO hardcoded analysis - pure LLM intelligence
        """
        print("  🧠 LLM analyzing schema requirements from ALL inputs...")
        
        system_prompt = """You are an expert IBM ACE (App Connect Enterprise) schema architect with deep expertise in XSD schema design and BizTalk-to-ACE migrations.

        Your task is to analyze the complete business requirements and component mappings to extract ALL schema requirements needed for an ACE project.

        ANALYSIS FOCUS:
        - Identify all data structures mentioned in the business requirements
        - Extract entity relationships and data flows  
        - Determine message formats and transformation requirements
        - Identify database schemas and external system interfaces
        - Map BizTalk components to required ACE schemas
        - Consider validation rules and constraints
        - Analyze error handling and exception schemas

        CRITICAL: You MUST return ONLY valid JSON in the EXACT format specified below. No explanations, no markdown, no additional text.

        REQUIRED JSON STRUCTURE:
        {
        "schema_requirements": [
            {
            "name": "SchemaName",
            "description": "Purpose and usage description",
            "category": "message|database|transformation|error|interface",
            "priority": "high|medium|low",
            "fields": [
                {
                "name": "fieldName",
                "type": "string|integer|decimal|boolean|date|complex",
                "required": true/false,
                "description": "Field purpose"
                }
            ],
            "constraints": ["validation rule 1", "validation rule 2"],
            "ace_considerations": "ACE-specific implementation notes"
            }
        ],
        "data_structures": [
            {
            "entity": "EntityName",
            "relationships": ["related entity 1", "related entity 2"],
            "source": "biztalk|database|external_system",
            "target_schema": "corresponding schema name"
            }
        ],
        "validation_rules": [
            {
            "schema": "schema name",
            "rule": "validation rule description",
            "type": "business|technical|format"
            }
        ],
        "relationships": [
            {
            "source_schema": "Schema1",
            "target_schema": "Schema2", 
            "relationship_type": "transformation|lookup|reference|aggregation",
            "description": "How schemas relate"
            }
        ],
        "ace_considerations": [
            {
            "schema": "schema name",
            "consideration": "ACE-specific implementation note",
            "impact": "performance|security|integration|maintenance"
            }
        ]
        }

        EXAMPLE OUTPUT:
        {
        "schema_requirements": [
            {
            "name": "CDMDocument",
            "description": "Common Data Model for customer messages",
            "category": "message",
            "priority": "high",
            "fields": [
                {
                "name": "customerId",
                "type": "string",
                "required": true,
                "description": "Unique customer identifier"
                },
                {
                "name": "documentType",
                "type": "string", 
                "required": true,
                "description": "Type of document being processed"
                }
            ],
            "constraints": ["customerId must be alphanumeric", "documentType from predefined list"],
            "ace_considerations": "Use ACE message sets for validation"
            }
        ],
        "data_structures": [
            {
            "entity": "Customer",
            "relationships": ["Order", "Address"],
            "source": "biztalk",
            "target_schema": "CDMDocument"
            }
        ],
        "validation_rules": [
            {
            "schema": "CDMDocument",
            "rule": "Customer ID must be 8-12 characters",
            "type": "business"
            }
        ],
        "relationships": [
            {
            "source_schema": "CDMDocument",
            "target_schema": "CW1EventFormat",
            "relationship_type": "transformation",
            "description": "CDM transforms to CW1 universal event format"
            }
        ],
        "ace_considerations": [
            {
            "schema": "CDMDocument",
            "consideration": "Implement as shared library for reusability",
            "impact": "maintenance"
            }
        ]
        }

        Be thorough and capture EVERY schema requirement from the provided content. Return ONLY the JSON structure above."""

        user_prompt = f"""Analyze the complete business requirements and component mappings to extract ALL schema requirements:

    ## VECTOR DB BUSINESS REQUIREMENTS:
    **Schema-Focused Content ({len(vector_processed_content.get('schema_content', ''))} characters):**
    {vector_processed_content.get('schema_content', '')}
    **Data Structure Requirements:**
    {json.dumps(vector_processed_content.get('data_structures', []), indent=2)}

    **Focus Keywords:**
    {json.dumps(vector_processed_content.get('focus_keywords', []), indent=2)}

    ## COMPONENT MAPPING JSON:
    {json.dumps(json_mappings, indent=2)}

    ## ANALYSIS REQUIREMENTS:
    Extract ALL schema requirements and return comprehensive JSON with:
    1. schema_requirements: List of all required schemas
    2. data_structures: Detailed structure for each schema
    3. validation_rules: Business rules and constraints
    4. relationships: How schemas relate to each other
    5. ace_considerations: ACE-specific implementation notes

    Analyze every piece of information provided and ensure no schema requirement is missed."""

        try:
            response = self.llm.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=3000
            )
            
            # Token tracking (if available)
            if 'token_tracker' in st.session_state and hasattr(response, 'usage') and response.usage:
                st.session_state.token_tracker.manual_track(
                    agent="schema_generator",
                    operation="schema_analysis", 
                    model=self.groq_model,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    flow_name=getattr(self, 'current_schema_name', 'schema_analysis')
                )
            
            raw_analysis = response.choices[0].message.content.strip()
            self.llm_analysis_calls += 1
            
            # Parse LLM response as JSON
            try:
                schema_requirements = json.loads(raw_analysis)
                print(f"  ✅ LLM analysis complete: {len(schema_requirements.get('schema_requirements', []))} schemas identified")
                return schema_requirements
            except json.JSONDecodeError:
                # If not valid JSON, try to extract JSON from response
                json_match = re.search(r'\{.*\}', raw_analysis, re.DOTALL)
                if json_match:
                    schema_requirements = json.loads(json_match.group())
                    print(f"  ✅ LLM analysis complete: {len(schema_requirements.get('schema_requirements', []))} schemas identified")
                    return schema_requirements
                else:
                    raise Exception("LLM did not return valid JSON analysis")
                    
        except Exception as e:
            raise Exception(f"LLM schema analysis failed: {str(e)}")
        


        
    
    def _llm_generate_xsd_schemas(self, schema_requirements: Dict, output_dir: str) -> List[Dict]:
        """
        LLM generates actual XSD schemas based on requirements analysis
        ✅ NOW WITH: Retry logic, rate limiting, and partial success handling
        """
        print("  ⚡ LLM generating ACE-compatible XSD schemas...")
        
        schemas = schema_requirements.get('schema_requirements', [])
        if not schemas:
            raise Exception("No schema requirements found from LLM analysis")
        
        generated_schemas = []
        failed_schemas = []
        total = len(schemas)
        
        for idx, schema_req in enumerate(schemas, 1):
            # Type validation
            if not isinstance(schema_req, dict):
                print(f"    ⚠️ Warning: Schema requirement is not a dict (type: {type(schema_req)}), skipping")
                failed_schemas.append({'name': 'Unknown', 'reason': 'Invalid type'})
                continue
            
            schema_name = schema_req.get('name', 'UnknownSchema')
            print(f"\n    {'='*50}")
            print(f"    🔨 [{idx}/{total}] Generating {schema_name}.xsd...")
            print(f"    {'='*50}")
            
            # Create prompt
            prompt = f"""Generate a complete XSD schema based on these requirements:

    ## SCHEMA SPECIFICATION:
    **Name:** {schema_name}
    **Purpose:** {schema_req.get('purpose', 'Data structure definition')}
    **Target Namespace:** urn:schemas:ace:{schema_name.lower()}

    ## DATA STRUCTURE REQUIREMENTS:
    {json.dumps(schema_req.get('data_structure', {}), indent=2)}

    ## VALIDATION RULES:
    {json.dumps(schema_req.get('validation_rules', []), indent=2)}

    ## BUSINESS CONTEXT:
    {schema_req.get('business_context', 'Enterprise data exchange')}

    ## ACE CONSIDERATIONS:
    {json.dumps(schema_req.get('ace_considerations', {}), indent=2)}

    Generate ONLY the complete XSD schema content. No explanations or markdown.

    Example structure:
    <?xml version="1.0" encoding="UTF-8"?>
    <xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
            targetNamespace="urn:schemas:ace:{schema_name.lower()}"
            elementFormDefault="qualified">
    <!-- Schema elements here -->
    </xs:schema>"""

            # ✅ USE RETRY LOGIC
            xsd_content = self._call_llm_with_retry(
                prompt=prompt,
                max_retries=5,
                timeout=180
            )
            
            if xsd_content:
                # Clean up markdown formatting
                xsd_content = re.sub(r'^```xml\s*\n?', '', xsd_content)
                xsd_content = re.sub(r'\n?```\s*$', '', xsd_content)
                xsd_content = xsd_content.strip()
                
                # Track token usage if available
                if 'token_tracker' in st.session_state:
                    try:
                        # Note: We can't track tokens from retry method, but that's ok
                        pass
                    except:
                        pass
                
                generated_schemas.append({
                    'name': schema_name,
                    'filename': f"{schema_name}.xsd",
                    'content': xsd_content,
                    'requirements': schema_req
                })
                
                self.llm_generation_calls += 1
                print(f"    ✅ [{idx}/{total}] {schema_name}.xsd generated ({len(xsd_content)} chars)")
            else:
                # ✅ DON'T RAISE EXCEPTION - CONTINUE
                failed_schemas.append({
                    'name': schema_name,
                    'reason': 'LLM call failed after retries'
                })
                print(f"    ❌ [{idx}/{total}] {schema_name}.xsd generation failed")
            
            # ✅ RATE LIMITING - 3 second delay between schemas
            if idx < total:
                print(f"    ⏳ Cooling down 3s before next schema...")
                time.sleep(3)
        
        # Final summary
        print(f"\n  {'='*50}")
        print(f"  📊 XSD Generation Summary:")
        print(f"  ✅ Successful: {len(generated_schemas)}/{total}")
        print(f"  ❌ Failed: {len(failed_schemas)}/{total}")
        print(f"  {'='*50}")
        
        if failed_schemas:
            print(f"\n  ⚠️ Failed schemas:")
            for failed in failed_schemas:
                print(f"    - {failed['name']}: {failed['reason']}")
        
        if not generated_schemas:
            raise Exception("All schema generation attempts failed")
        
        print(f"\n  🎉 Successfully generated {len(generated_schemas)} schemas")
        return generated_schemas
    


    
    def _write_schema_files(self, generated_schemas: List[Dict], schemas_dir: str) -> List[str]:
        """Write generated XSD schemas to files"""
        print("  💾 Writing XSD schemas to files...")
        
        schema_files = []
        
        for schema in generated_schemas:
            filename = schema['filename']
            content = schema['content']
            
            file_path = os.path.join(schemas_dir, filename)
            
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                schema_files.append(file_path)
                print(f"    📄 Created: {filename}")
                
            except Exception as e:
                print(f"    ❌ Failed to write {filename}: {str(e)}")
                raise
        
        print(f"  ✅ All schema files written to: {schemas_dir}")
        return schema_files

def main():
    """Test harness for schema generator - Vector DB Mode"""
    generator = SchemaGenerator()
    
    # Get Vector DB content using existing pipeline (NO PDF processing)
    import streamlit as st
    vector_content = st.session_state.vector_pipeline.search_engine.get_agent_content("schema_generator")
    
    # Generate schemas using Vector DB content
    result = generator.generate_schemas(
        vector_content=vector_content,                      # ✅ Vector DB content
        component_mapping_json_path="component_mapping.json",  # ✅ Direct path for testing
        output_dir="test_output"                           # ✅ Direct path for testing
    )
    
    print(f"\n🎯 Schema Generation Results:")
    print(f"✅ Status: {result['status']}")
    print(f"📊 Schemas Generated: {result['schemas_generated']}")
    print(f"🧠 LLM Analysis Calls: {result['llm_analysis_calls']}")
    print(f"⚡ LLM Generation Calls: {result['llm_generation_calls']}")


if __name__ == "__main__":
    main()