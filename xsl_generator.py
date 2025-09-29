#!/usr/bin/env python3
"""
XSL Generator Module v1.0 - ACE Module Creator
Purpose: Generate XSL transformation files from BizTalk mapping requirements
Input: Confluence PDF + component mapping JSON + LLM → Generate XSL transformation files from BizTalk mapping requirements
Output: Creates transforms/ folder with all .xsl transformation files
Prompt Focus: "Analyze transformation requirements from Confluence PDF and component mappings, generate XSL transformation files"
LLM Task: XSL/XSLT transformation logic generation
"""

import os
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

class XSLGenerator:
    """
    ACE XSL Transformation Generator with complete Vector DB + LLM integration
    NO HARDCODED FALLBACKS - All XSL transformations generated via Vector DB + LLM analysis
    NO PDF PROCESSING - Uses Vector DB focused content only
    """
    
    def __init__(self, groq_api_key: str = None):
        """Initialize with Groq LLM client"""
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY must be provided or set in environment")
        
        self.llm = Groq(api_key=self.groq_api_key)
        self.groq_model = os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')
        
        # Processing results tracking
        self.processed_vector_content = None
        self.processed_json_mappings = None
        self.generated_xsl_transforms = []
        self.llm_analysis_calls = 0
        self.llm_generation_calls = 0
    
    def generate_xsl_transformations(self, 
                                vector_content: str,
                                component_mapping_json_path: str,
                                output_dir: str) -> Dict[str, Any]:
        """
        Main entry point - Generate XSL transformation files from Vector DB business requirements
        
        Args:
            vector_content: Vector DB focused content with transformation requirements
            component_mapping_json_path: Path to component mapping JSON with transformation definitions
            output_dir: Output directory for generated XSL transformation files
            
        Returns:
            Dict with generation results and metadata
        """
        print("🎯 Starting XSL Transformation Generation - Full Vector DB + LLM Processing Mode")
        print("📋 NO HARDCODED FALLBACKS - Pure AI-driven XSL/XSLT transformation generation")
        
        # Step 1: Process Vector DB content for transformation analysis
        print("\n📄 Step 1: Processing Vector DB content for transformation requirements analysis...")
        vector_processed_content = self._process_vector_content_for_xsl(vector_content)
        json_mappings = self._extract_complete_json_mappings(component_mapping_json_path)
        
        # Step 2: LLM Analysis of transformation requirements from Vector DB content
        print("\n🧠 Step 2: LLM analyzing transformation requirements from Vector DB content...")
        transformation_requirements = self._llm_analyze_transformation_requirements(vector_processed_content, json_mappings)
        
        # Step 3: LLM Generation of XSL transformation files
        print("\n⚡ Step 3: LLM generating XSL transformation files...")
        generated_xsl_files = self._llm_generate_xsl_transformation_files(transformation_requirements)
        
        # Step 4: Create output structure
        transforms_dir = os.path.join(output_dir, 'transforms')
        os.makedirs(transforms_dir, exist_ok=True)
        
        # Step 5: Write generated XSL transformation files
        print("\n💾 Step 5: Writing XSL transformation files to transforms/ folder...")
        xsl_files = self._write_xsl_transformation_files(generated_xsl_files, transforms_dir)
        
        return {
            'status': 'success',
            'xsl_transformations_generated': len(xsl_files),
            'xsl_files': xsl_files,
            'output_directory': transforms_dir,
            'llm_analysis_calls': self.llm_analysis_calls,
            'llm_generation_calls': self.llm_generation_calls,
            'processing_metadata': {
                'vector_content_length': len(vector_content),
                'transformation_requirements_identified': len(transformation_requirements.get('transformations', [])),
                'json_mappings_processed': len(json_mappings.get('transformation_mappings', [])),
                'timestamp': datetime.now().isoformat()
            }
        }
        
    def _process_vector_content_for_xsl(self, vector_content: str) -> Dict[str, Any]:
        """
        Process Vector DB content for XSL transformation analysis
        Simple wrapper focused on XSL transformation requirements
        """
        print(f"  🔍 Processing Vector DB content for XSL transformation focus: {len(vector_content)} characters")
        
        return {
            'transformation_content': vector_content,
            'content_length': len(vector_content),
            'focus_keywords': ['transformation', 'mapping', 'xsl', 'xslt', 'field mapping', 'data conversion'],
            'source': 'vector_db',
            'transformation_indicators': self._extract_transformation_indicators(vector_content)
        }

    def _extract_transformation_indicators(self, content: str) -> List[str]:
        """Extract XSL transformation patterns from Vector content"""
        indicators = ['biztalk map', 'source schema', 'target schema', 'field mapping', 'data transformation', 'xpath', 'xslt']
        content_lower = content.lower()
        found = [indicator for indicator in indicators if indicator in content_lower]
        return found
    
    def _extract_complete_json_mappings(self, json_path: str) -> Dict[str, Any]:
        """
        Extract COMPLETE JSON mapping content for transformation analysis
        NO filtering - all component mapping data goes to LLM for transformation analysis
        """
        print(f"  🗂️ Extracting complete JSON component mappings for transformation analysis from: {json_path}")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Component mapping JSON not found: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            if not isinstance(json_data, dict):
                raise ValueError("JSON must be a dictionary object")
            
            # Enhance JSON data for transformation analysis
            enhanced_json = {
                'original_data': json_data,
                'transformation_mappings': self._extract_transformation_mappings(json_data),
                'component_relationships': self._extract_component_relationships(json_data),
                'data_flow_patterns': self._extract_data_flow_patterns(json_data)
            }
            
            print(f"  ✅ JSON processed: {len(str(json_data))} characters, {len(enhanced_json['transformation_mappings'])} transformation mappings identified")
            self.processed_json_mappings = enhanced_json
            return enhanced_json
            
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to process JSON for transformation analysis: {str(e)}")
        

    
    def _llm_analyze_transformation_requirements(self, vector_processed_content: Dict, json_mappings: Dict) -> Dict[str, Any]:
        """
        LLM analyzes transformation requirements from Vector DB content and component mappings
        NO hardcoded analysis - pure LLM intelligence for XSL transformation analysis
        """
        print("  🧠 LLM analyzing XSL transformation requirements from Vector DB content...")
        
        system_prompt = """You are an expert XSL/XSLT transformation architect with deep expertise in:
    - BizTalk mapping and transformation patterns
    - XSL/XSLT development and best practices
    - Data transformation and field mapping
    - XML schema transformation logic
    - Enterprise integration transformation requirements

    Your task is to analyze Vector DB transformation requirements to generate XSL transformation files.

    TRANSFORMATION ANALYSIS FOCUS:
    - Identify all data transformation requirements from Vector DB content
    - Extract field mapping specifications and business rules
    - Analyze component mapping relationships for transformation logic
    - Determine source-to-target schema transformations
    - Identify data conversion and validation requirements
    - Extract transformation patterns that need XSL implementation
    - Analyze complex transformation logic and conditional mappings

    OUTPUT REQUIREMENTS:
    Return comprehensive JSON analysis with:
    1. All required XSL transformations with their purposes
    2. Source and target schema specifications
    3. Field mapping requirements and transformation rules
    4. Business logic for data conversion and validation
    5. Conditional transformation logic and error handling
    6. XSL template patterns and transformation methods

    Be thorough and capture EVERY transformation requirement from the Vector DB content."""

        user_prompt = f"""Analyze Vector DB transformation requirements to generate XSL transformation files:

    ## VECTOR DB TRANSFORMATION REQUIREMENTS:
    **Transformation-Focused Content ({vector_processed_content.get('content_length', 0)} characters):**
    {vector_processed_content.get('transformation_content', '')}

    **Transformation Indicators Found:**
    {json.dumps(vector_processed_content.get('transformation_indicators', []), indent=2)}

    ## COMPONENT MAPPING JSON:
    **Original Component Data:**
    {json.dumps(json_mappings.get('original_data', {}), indent=2)}

    **Transformation Mappings:**
    {json.dumps(json_mappings.get('transformation_mappings', []), indent=2)}

    **Component Relationships:**
    {json.dumps(json_mappings.get('component_relationships', []), indent=2)}

    **Data Flow Patterns:**
    {json.dumps(json_mappings.get('data_flow_patterns', []), indent=2)}

    ## TRANSFORMATION ANALYSIS REQUIREMENTS:
    Extract ALL transformation requirements and return comprehensive JSON with:
    1. transformations: List of all required XSL transformations
    2. field_mappings: Detailed field-to-field mapping specifications
    3. business_rules: Transformation business logic and validation rules
    4. source_schemas: Source data structure specifications
    5. target_schemas: Target data structure specifications
    6. transformation_patterns: XSL/XSLT implementation patterns

    Focus on analyzing transformation requirements from Vector DB content to generate XSL transformation files. Ensure no transformation requirement is missed."""

        try:
            response = self.llm.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            if 'token_tracker' in st.session_state and hasattr(response, 'usage') and response.usage:
                st.session_state.token_tracker.manual_track(
                    agent="xsl_generator",
                    operation="xsl_transformation_generation",
                    model=self.groq_model,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    flow_name="xsl_generation"
                )
            
            raw_analysis = response.choices[0].message.content.strip()
            self.llm_analysis_calls += 1
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', raw_analysis, re.DOTALL)
            if json_match:
                transformation_requirements = json.loads(json_match.group())
                print(f"  ✅ LLM transformation analysis complete: {len(transformation_requirements.get('transformations', []))} XSL transformations identified")
                return transformation_requirements
            else:
                raise Exception("LLM did not return valid JSON transformation analysis")
                
        except Exception as e:
            raise Exception(f"LLM transformation analysis failed: {str(e)}")
        

    
    def _llm_generate_xsl_transformation_files(self, transformation_requirements: Dict) -> List[Dict]:
        """
        LLM generates XSL transformation files based on Vector DB requirements analysis
        NO hardcoded XSL templates - pure LLM XSL/XSLT transformation logic generation
        """
        print("  ⚡ LLM generating XSL transformation files from Vector DB requirements...")
        
        transformations = transformation_requirements.get('transformations', [])
        if not transformations:
            raise Exception("No transformation requirements found from Vector DB LLM analysis")
        
        generated_xsl_files = []
        
        for transform_req in transformations:
            transform_name = transform_req.get('name', 'UnknownTransform')
            print(f"    🔨 Generating {transform_name}.xsl from Vector DB requirements...")
            
            system_prompt = """You are an expert XSL/XSLT developer specializing in enterprise data transformations.

    Generate production-ready XSL transformation files that are:
    - Fully compliant with XSLT 1.0/2.0 standards
    - Optimized for performance and maintainability
    - Include comprehensive field mapping and data conversion logic
    - Follow enterprise transformation patterns and best practices
    - Include proper error handling and validation
    - Include detailed comments and documentation
    - Support complex conditional transformation logic

    Return ONLY the complete XSL transformation content, no explanations or markdown."""

            user_prompt = f"""Generate a complete XSL transformation file based on Vector DB transformation requirements:

    ## TRANSFORMATION SPECIFICATION:
    **Name:** {transform_name}
    **Purpose:** {transform_req.get('purpose', 'Data transformation')}
    **Type:** {transform_req.get('type', 'XSL Transform')}

    ## FIELD MAPPING REQUIREMENTS:
    {json.dumps(transform_req.get('field_mappings', []), indent=2)}

    ## BUSINESS RULES:
    {json.dumps(transform_req.get('business_rules', []), indent=2)}

    ## SOURCE SCHEMA:
    {json.dumps(transform_req.get('source_schema', {}), indent=2)}

    ## TARGET SCHEMA:
    {json.dumps(transform_req.get('target_schema', {}), indent=2)}

    ## TRANSFORMATION PATTERNS:
    {json.dumps(transform_req.get('transformation_patterns', []), indent=2)}

    ## XSL REQUIREMENTS:
    - Include proper XSL namespace declarations
    - Implement all field mapping requirements with proper XPath expressions
    - Include business rule validation and conditional logic
    - Add error handling for missing or invalid data
    - Follow enterprise XSL transformation patterns
    - Include comprehensive comments explaining transformation logic
    - Support both simple field mappings and complex data conversions

    Generate the complete XSL transformation file that satisfies all Vector DB transformation requirements."""

            try:
                response = self.llm.chat.completions.create(
                    model=self.groq_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=4000
                )
                if 'token_tracker' in st.session_state and hasattr(response, 'usage') and response.usage:
                    st.session_state.token_tracker.manual_track(
                        agent="xsl_generator",
                        operation="xsl_transformation_generation",
                        model=self.groq_model,
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                        flow_name="xsl_generation"
                    )
                
                xsl_content = response.choices[0].message.content.strip()
                xsl_content = self._clean_markdown_blocks(xsl_content)  # ← CLEANUP: Remove ```xsl and ``` markers
                self.llm_generation_calls += 1

                generated_xsl_files.append({
                    'filename': f"{transform_name}.xsl",
                    'content': xsl_content,
                    'purpose': transform_req.get('purpose', 'Data transformation'),
                    'content_length': len(xsl_content)
                })
                
                print(f"    ✅ {transform_name}.xsl generated: {len(xsl_content)} characters")
                
            except Exception as e:
                raise Exception(f"Failed to generate XSL for {transform_name}: {str(e)}")
        
        print(f"  ✅ All XSL transformations generated: {len(generated_xsl_files)} files")
        return generated_xsl_files
    
    

    def _clean_markdown_blocks(self, content: str) -> str:
        """
        Remove markdown code block markers from LLM-generated content
        Handles: ```xsl, ```xml, or plain ``` markers
        """
        # Remove opening markers: ```xsl, ```xml, or ```
        content = re.sub(r'^```(?:xsl|xml)?\s*\n?', '', content, flags=re.MULTILINE)
        
        # Remove closing markers: ```
        content = re.sub(r'\n?```\s*$', '', content, flags=re.MULTILINE)
        
        # Remove any lingering ``` in the middle (just in case)
        content = re.sub(r'\n```\n', '\n', content)
        
        return content.strip()


    
    def _write_xsl_transformation_files(self, generated_xsl_files: List[Dict], transforms_dir: str) -> List[str]:
        """Write generated XSL transformation files to transforms/ folder"""
        print("  💾 Writing XSL transformation files to transforms/ folder...")
        
        xsl_files = []
        
        for xsl_file in generated_xsl_files:
            filename = xsl_file['filename']
            content = xsl_file['content']
            
            file_path = os.path.join(transforms_dir, filename)
            
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                xsl_files.append(file_path)
                print(f"    📄 Created: {filename}")
                
            except Exception as e:
                print(f"    ❌ Failed to write {filename}: {str(e)}")
                raise
        
        print(f"  ✅ All XSL transformation files written to: {transforms_dir}")
        return xsl_files
    
    # Helper methods for transformation analysis
    def _is_transformation_table(self, table_data: List[List]) -> bool:
        """Determine if a table contains transformation mapping information"""
        if not table_data or len(table_data) < 2:
            return False
        
        # Check for transformation-related headers
        header_row = [str(cell).lower() if cell else '' for cell in table_data[0]]
        transformation_indicators = [
            'source', 'target', 'mapping', 'transform', 'field', 'xpath', 'value'
        ]
        
        return any(indicator in ' '.join(header_row) for indicator in transformation_indicators)
    
    def _extract_transformation_mappings(self, json_data: Dict) -> List[Dict]:
        """Extract transformation mapping information from JSON"""
        mappings = []
        
        # Look for transformation-related keys
        if 'transformations' in json_data:
            mappings.extend(json_data['transformations'])
        if 'mappings' in json_data:
            mappings.extend(json_data['mappings'])
        if 'components' in json_data:
            for component in json_data['components']:
                if 'transformations' in component:
                    mappings.extend(component['transformations'])
        
        return mappings
    
    def _extract_component_relationships(self, json_data: Dict) -> List[Dict]:
        """Extract component relationship information for transformation logic"""
        relationships = []
        
        if 'components' in json_data:
            for component in json_data['components']:
                if 'relationships' in component:
                    relationships.extend(component['relationships'])
                if 'dependencies' in component:
                    relationships.extend(component['dependencies'])
        
        return relationships
    
    def _extract_data_flow_patterns(self, json_data: Dict) -> List[Dict]:
        """Extract data flow patterns for transformation analysis"""
        patterns = []
        
        if 'data_flows' in json_data:
            patterns.extend(json_data['data_flows'])
        if 'workflows' in json_data:
            patterns.extend(json_data['workflows'])
        
        return patterns


def main():
    """
    Test harness for XSL Generator with Vector DB integration
    Run this through main.py with Vector Knowledge Base setup
    """
    import streamlit as st
    
    # Check if Vector DB pipeline is available
    if not hasattr(st, 'session_state') or not st.session_state.get('vector_pipeline'):
        print("❌ Vector DB pipeline not available")
        print("💡 Run this through main.py with Vector DB setup")
        print("📝 Steps: 1) Upload PDF in Agent 1, 2) Setup Vector Knowledge Base, 3) Run XSL generation")
        return
    
    print("🚀 Starting Vector DB XSL generation test...")
    
    # Get Vector DB content for XSL generation
    vector_content = st.session_state.vector_pipeline.search_engine.get_agent_content("xsl_generator")
    
    if not vector_content:
        print("❌ No Vector DB content found for 'xsl_generator'")
        print("💡 Ensure Vector Knowledge Base contains XSL/transformation-related content")
        return
    
    print(f"📊 Vector DB content retrieved: {len(vector_content)} characters")
    
    # Initialize XSL Generator
    generator = XSLGenerator()
    
    # Generate XSL transformations using Vector DB content
    result = generator.generate_xsl_transformations(
        vector_content=vector_content,                           # ✅ Vector DB content
        component_mapping_json_path='component_mapping.json',    # ✅ Direct path for testing
        output_dir='output'                                      # ✅ Output directory
    )
    
    print(f"\n🎯 XSL Transformation Generation Results:")
    print(f"✅ Status: {result['status']}")
    print(f"📊 Transformations Generated: {result['xsl_transformations_generated']}")
    print(f"🧠 LLM Analysis Calls: {result['llm_analysis_calls']}")
    print(f"⚡ LLM Generation Calls: {result['llm_generation_calls']}")
    
    # Display individual XSL files
    if result.get('xsl_files'):
        print(f"\n📁 Generated XSL Files:")
        for i, xsl_file in enumerate(result['xsl_files'], 1):
            filename = os.path.basename(xsl_file)
            print(f"  {i}. {filename}")
    
    # Display processing metadata
    if result.get('processing_metadata'):
        metadata = result['processing_metadata']
        print(f"\n📈 Processing Summary:")
        print(f"  • Vector content: {metadata.get('vector_content_length', 0)} characters")
        print(f"  • Transformation requirements: {metadata.get('transformation_requirements_identified', 0)}")
        print(f"  • JSON mappings processed: {metadata.get('json_mappings_processed', 0)}")
        print(f"  • Output directory: {result.get('output_directory', 'N/A')}")


if __name__ == "__main__":
    main()