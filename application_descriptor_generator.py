#!/usr/bin/env python3
"""
Application Descriptor Generator Module v1.0 - ACE Module Creator
Purpose: Generate final application descriptor with library references
Input: application_descriptor.xml template + component mapping JSON + LLM → Generate final application descriptor with library references
Output: Creates application.descriptor file in project root with proper shared library dependencies
Prompt Focus: "Use application descriptor template, extract shared library dependencies from component mappings, generate final descriptor"
LLM Task: Library dependency analysis and descriptor generation
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

class ApplicationDescriptorGenerator:
    """
    ACE Application Descriptor Generator with complete LLM integration
    NO HARDCODED FALLBACKS - All application descriptor generation via LLM analysis
    """
    
    def __init__(self, groq_api_key: str = None):
        """Initialize with Groq LLM client"""
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY must be provided or set in environment")
        
        self.llm = Groq(api_key=self.groq_api_key)
        self.groq_model = os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')
        
        # Processing results tracking
        self.processed_template_content = None
        self.processed_json_mappings = None
        self.generated_descriptor = None
        self.llm_analysis_calls = 0
        self.llm_generation_calls = 0


    
    def generate_application_descriptor(self, vector_content, template_path, component_mapping_json_path, output_dir):
        
        # ADD: Convert string to expected dict format
        if isinstance(vector_content, str):
            vector_content = {
                'library_content': vector_content,
                'content_length': len(vector_content),
                'configuration_indicators': [],
                'shared_library_references': [],
                'runtime_requirements': [],
                'application_context': []
            }
        """
        Main entry point - Generate final application descriptor using Vector DB + DSV Template
        
        Args:
            vector_content: Vector DB focused content for library dependencies and configuration
            template_path: Path to application_descriptor.xml DSV standard template
            component_mapping_json_path: Path to component mapping JSON with library dependencies
            output_dir: Output directory (project root) for application.descriptor file
            
        Returns:
            Dict with generation results and metadata
        """
        print("🎯 Starting Application Descriptor Generation - Vector DB + DSV Template Processing")
        print("📋 NO HARDCODED FALLBACKS - Pure AI-driven analysis using Vector DB business context + DSV standard template")
        
        from datetime import datetime
        
        # Step 1: Process ALL inputs completely
        print("\n📄 Step 1: Processing ALL inputs for library dependency analysis...")
        template_content = self._extract_complete_template_content(template_path)
        json_mappings = self._extract_complete_json_mappings(component_mapping_json_path)
        
        # Step 2: LLM Analysis of library dependencies using Vector DB + template + mappings
        print("\n🧠 Step 2: LLM analyzing shared library dependencies from Vector DB + DSV template...")
        dependency_analysis = self._llm_analyze_library_dependencies(
            vector_content=vector_content,  # ← Vector DB business requirements
            json_mappings=json_mappings,
            template_content=template_content  # ← DSV standard template
        )
        
        # Step 3: LLM Generation of final application descriptor
        print("\n⚡ Step 3: LLM generating final application descriptor...")
        generated_descriptor = self._llm_generate_application_descriptor(
            dependency_analysis=dependency_analysis, 
            vector_content=vector_content,  # ← Vector DB business context
            template_content=template_content  # ← DSV standard structure
        )
        
        # Step 4: Write application descriptor to project root
        print("\n💾 Step 4: Writing application.descriptor file to project root...")
        descriptor_file = self._write_application_descriptor_file(generated_descriptor, output_dir)
        
        return {
            'status': 'success',
            'application_descriptor_generated': True,
            'descriptor_file': descriptor_file,
            'output_directory': output_dir,
            'llm_analysis_calls': self.llm_analysis_calls,
            'llm_generation_calls': self.llm_generation_calls,
            'processing_metadata': {
                'vector_content_processed': bool(vector_content),
                'template_content_processed': bool(template_content),
                'json_components_processed': len(json_mappings.get('components', [])),
                'dependencies_identified': len(dependency_analysis.get('shared_libraries', [])),
                'dsv_template_applied': bool(template_content.get('raw_content')),
                'vector_business_context_applied': bool(vector_content.get('library_content')),
                'timestamp': datetime.now().isoformat()
            }
        }
    


    def _llm_analyze_library_dependencies(self, vector_content: Dict, json_mappings: Dict, template_content: Dict) -> Dict[str, Any]:
        """
        LLM analyzes shared library dependencies using Vector DB business requirements AND DSV standard template
        NO hardcoded analysis - pure LLM intelligence combining Vector business context with template compliance
        """
        print("  🧠 LLM analyzing shared library dependencies from Vector DB business requirements + DSV template...")
        
        system_prompt = """You are an expert IBM ACE (App Connect Enterprise) application architect with deep expertise in:
    - Application descriptor configuration and structure
    - Shared library dependency management
    - Component integration and library references
    - ACE runtime dependency resolution
    - Enterprise application deployment patterns

    Your task is to analyze Vector DB business requirements AND DSV standard template to extract shared library dependencies from component mappings.

    DEPENDENCY ANALYSIS FOCUS:
    - Analyze Vector DB business requirements for library and configuration patterns
    - Use DSV standard template structure for compliance and format standards
    - Extract all shared library dependencies from component mappings
    - Identify required ACE runtime libraries and external dependencies
    - Determine proper library reference configurations from business context
    - Analyze component relationships for dependency requirements
    - Apply template standards while incorporating business requirements context
    - Identify version constraints and compatibility requirements

    OUTPUT REQUIREMENTS:
    Return comprehensive JSON analysis with:
    1. shared_libraries: List of all required shared library dependencies with versions
    2. library_configurations: Configuration settings for each library from business requirements
    3. dependency_relationships: Component-to-library dependency mappings
    4. runtime_requirements: ACE runtime library requirements
    5. external_dependencies: External system and service dependencies
    6. application_settings: Application configuration parameters

    Focus on combining Vector DB business requirements with DSV template standards to extract library dependencies."""

        user_prompt = f"""Analyze Vector DB business requirements, DSV template structure, and component mappings to extract shared library dependencies:

    ## VECTOR DB BUSINESS REQUIREMENTS:
    **Library Dependencies Focused Content ({vector_content.get('content_length', 0)} characters):**
    {vector_content.get('library_content', '')}

    **Configuration Indicators Found:**
    {json.dumps(vector_content.get('configuration_indicators', []), indent=2)}

    **Shared Library References:**
    {json.dumps(vector_content.get('shared_library_references', []), indent=2)}

    **Runtime Requirements:**
    {json.dumps(vector_content.get('runtime_requirements', []), indent=2)}

    ## DSV STANDARD TEMPLATE:
    **Template Structure ({template_content.get('content_length', 0)} characters):**
    {template_content.get('raw_content', '')}

    **XML Structure Analysis:**
    {json.dumps(template_content.get('xml_structure', {}), indent=2)}

    **Library References in Template:**
    {json.dumps(template_content.get('library_references', []), indent=2)}

    **Configuration Sections:**
    {json.dumps(template_content.get('configuration_sections', []), indent=2)}

    ## COMPONENT MAPPING JSON:
    **Original Component Data:**
    {json.dumps(json_mappings.get('original_data', {}), indent=2)}

    **Library Dependencies:**
    {json.dumps(json_mappings.get('library_dependencies', []), indent=2)}

    **Shared Library References:**
    {json.dumps(json_mappings.get('shared_library_references', []), indent=2)}

    **Component Dependencies:**
    {json.dumps(json_mappings.get('component_dependencies', []), indent=2)}

    **External References:**
    {json.dumps(json_mappings.get('external_references', []), indent=2)}

    ## DEPENDENCY ANALYSIS REQUIREMENTS:
    Extract ALL library dependencies and return comprehensive JSON with:
    1. shared_libraries: List of all required shared library dependencies
    2. library_configurations: Configuration settings for each library
    3. dependency_relationships: Component-to-library dependency mappings
    4. runtime_requirements: ACE runtime library requirements
    5. external_dependencies: External system and service dependencies
    6. application_settings: Application configuration parameters

    Focus on combining Vector DB business requirements with DSV template standards to extract shared library dependencies from component mappings. Ensure no library dependency is missed."""

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
            
            # Add token tracking if available
            import streamlit as st
            if 'token_tracker' in st.session_state and hasattr(response, 'usage') and response.usage:
                st.session_state.token_tracker.manual_track(
                    agent="application_descriptor_generator",
                    operation="library_dependency_analysis",
                    model=os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant'),
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    flow_name="application_descriptor_generation"
                )
            
            raw_analysis = response.choices[0].message.content.strip()
            self.llm_analysis_calls += 1
            
            # Parse LLM response as JSON
            try:
                dependency_analysis = json.loads(raw_analysis)
                print(f"  ✅ LLM dependency analysis complete: {len(dependency_analysis.get('shared_libraries', []))} shared libraries identified")
                return dependency_analysis
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', raw_analysis, re.DOTALL)
                if json_match:
                    dependency_analysis = json.loads(json_match.group())
                    print(f"  ✅ LLM dependency analysis complete: {len(dependency_analysis.get('shared_libraries', []))} shared libraries identified")
                    return dependency_analysis
                else:
                    raise Exception("LLM did not return valid JSON dependency analysis")
                    
        except Exception as e:
            raise Exception(f"LLM library dependency analysis failed: {str(e)}")
    

        
    def _extract_complete_template_content(self, template_path: str) -> Dict[str, Any]:
        """
        Extract COMPLETE application descriptor template content for LLM processing
        NO filtering - entire template structure goes to LLM for analysis
        """
        print(f"  📝 Extracting complete application descriptor template from: {template_path}")
        
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Application descriptor template not found: {template_path}")
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_xml = f.read()
            
            # Parse XML structure for comprehensive analysis
            try:
                root = ET.fromstring(template_xml)
                xml_structure = self._analyze_xml_structure(root)
            except ET.ParseError:
                # If not valid XML, treat as text template
                xml_structure = {'type': 'text_template', 'valid_xml': False}
            
            template_data = {
                'raw_content': template_xml,
                'content_length': len(template_xml),
                'xml_structure': xml_structure,
                'library_references': self._extract_library_references(template_xml),
                'dependency_patterns': self._extract_dependency_patterns(template_xml),
                'configuration_sections': self._extract_configuration_sections(template_xml),
                'template_variables': self._extract_template_variables(template_xml)
            }
            
            print(f"  ✅ Template processed: {len(template_xml)} characters, {len(template_data['library_references'])} library references detected")
            self.processed_template_content = template_data
            return template_data
            
        except Exception as e:
            raise Exception(f"Failed to process application descriptor template: {str(e)}")
    
    def _extract_complete_json_mappings(self, json_path: str) -> Dict[str, Any]:
        """
        Extract COMPLETE JSON mapping content for library dependency analysis
        NO filtering - all component mapping data goes to LLM for dependency analysis
        """
        print(f"  🗂️ Extracting complete JSON component mappings for library dependency analysis from: {json_path}")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Component mapping JSON not found: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            if not isinstance(json_data, dict):
                raise ValueError("JSON must be a dictionary object")
            
            # Enhance JSON data for library dependency analysis
            enhanced_json = {
                'original_data': json_data,
                'library_dependencies': self._extract_library_dependencies(json_data),
                'shared_library_references': self._extract_shared_library_references(json_data),
                'component_dependencies': self._extract_component_dependencies(json_data),
                'external_references': self._extract_external_references(json_data)
            }
            
            print(f"  ✅ JSON processed: {len(str(json_data))} characters, {len(enhanced_json['library_dependencies'])} library dependencies identified")
            self.processed_json_mappings = enhanced_json
            return enhanced_json
            
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to process JSON for library dependency analysis: {str(e)}")
        

    

    
    def _llm_generate_application_descriptor(self, dependency_analysis: Dict, vector_content: Dict, template_content: Dict) -> Dict[str, Any]:
        """
        LLM generates final application descriptor based on dependency analysis, Vector DB business context, and DSV template
        NO hardcoded descriptor generation - pure LLM application descriptor generation using Vector insights + template standards
        """
        print("  ⚡ LLM generating final application descriptor from Vector DB business context + DSV template...")
        
        system_prompt = """You are an expert IBM ACE application descriptor developer specializing in enterprise application configuration.

    Your task is to generate a complete, production-ready application.descriptor file based on dependency analysis, Vector DB business requirements context, and DSV standard template.

    GENERATION FOCUS:
    - Create proper IBM ACE application descriptor XML structure using DSV template as base
    - Include all shared library dependencies with correct references
    - Apply business requirements context from Vector DB for configuration
    - Follow DSV standard template structure and format compliance
    - Generate runtime library configurations based on business needs
    - Include proper version constraints and compatibility settings
    - Follow enterprise deployment patterns and ACE best practices
    - Apply Vector DB business context for application-specific settings

    DESCRIPTOR REQUIREMENTS:
    - Use DSV standard template XML format as structural foundation
    - Include proper namespace declarations and XML structure from template
    - Add all identified shared library dependencies with proper references
    - Include runtime configurations based on business requirements
    - Apply proper version management and compatibility constraints
    - Include comprehensive configuration parameters from business context
    - Follow ACE deployment standards and enterprise patterns
    - Maintain DSV template compliance while incorporating business requirements

    Generate production-ready application.descriptor content that is:
    - Fully compliant with IBM ACE application descriptor standards
    - Based on DSV template structure for organizational compliance
    - Include all required shared library dependencies and references
    - Follow enterprise application deployment patterns from business context
    - Include proper version management and compatibility settings
    - Include comprehensive configuration parameters from Vector DB insights
    - Follow ACE runtime requirements and best practices

    Return ONLY the complete application.descriptor XML content, no explanations or markdown."""

        user_prompt = f"""Generate a complete application.descriptor file based on dependency analysis, Vector DB business context, and DSV template:

    ## DEPENDENCY ANALYSIS:
    **Shared Libraries:**
    {json.dumps(dependency_analysis.get('shared_libraries', []), indent=2)}

    **Library Configurations:**
    {json.dumps(dependency_analysis.get('library_configurations', {}), indent=2)}

    **Dependency Relationships:**
    {json.dumps(dependency_analysis.get('dependency_relationships', []), indent=2)}

    **Runtime Requirements:**
    {json.dumps(dependency_analysis.get('runtime_requirements', {}), indent=2)}

    **External Dependencies:**
    {json.dumps(dependency_analysis.get('external_dependencies', []), indent=2)}

    **Application Settings:**
    {json.dumps(dependency_analysis.get('application_settings', {}), indent=2)}

    ## VECTOR DB BUSINESS CONTEXT:
    **Application Configuration Context ({vector_content.get('content_length', 0)} characters):**
    {vector_content.get('library_content', '')}

    **Configuration Patterns from Business Requirements:**
    {json.dumps(vector_content.get('configuration_indicators', []), indent=2)}

    **Business-Driven Library References:**
    {json.dumps(vector_content.get('shared_library_references', []), indent=2)}

    **Runtime Context from Business Requirements:**
    {json.dumps(vector_content.get('runtime_requirements', []), indent=2)}

    **Application Context Indicators:**
    {json.dumps(vector_content.get('application_context', []), indent=2)}

    ## DSV STANDARD TEMPLATE:
    **Template Structure ({template_content.get('content_length', 0)} characters):**
    {template_content.get('raw_content', '')}

    **XML Structure Analysis:**
    {json.dumps(template_content.get('xml_structure', {}), indent=2)}

    **Template Library References:**
    {json.dumps(template_content.get('library_references', []), indent=2)}

    **Template Configuration Sections:**
    {json.dumps(template_content.get('configuration_sections', []), indent=2)}

    **Template Variables:**
    {json.dumps(template_content.get('template_variables', []), indent=2)}

    ## APPLICATION DESCRIPTOR REQUIREMENTS:
    - Use DSV standard template XML structure as the foundation with proper namespaces
    - Include all identified shared library dependencies with proper references and versions
    - Apply business context from Vector DB for application-specific configurations
    - Follow DSV template format and organizational compliance standards
    - Include required runtime library configurations based on business requirements
    - Add proper version constraints and compatibility settings from dependency analysis
    - Include application configuration parameters derived from business context
    - Follow enterprise deployment patterns and ACE runtime best practices
    - Maintain template structure while incorporating all business-driven requirements

    Generate the complete application.descriptor file content that combines DSV template compliance with Vector DB business requirements and all identified library dependencies."""

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
            
            # Add token tracking if available
            import streamlit as st
            if 'token_tracker' in st.session_state and hasattr(response, 'usage') and response.usage:
                st.session_state.token_tracker.manual_track(
                    agent="application_descriptor_generator",
                    operation="descriptor_generation",
                    model=os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant'),
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    flow_name="application_descriptor_generation"
                )
            
            raw_descriptor = response.choices[0].message.content.strip()
            self.llm_generation_calls += 1
            
            # Clean up the generated content (remove markdown if present)
            if raw_descriptor.startswith('```xml'):
                raw_descriptor = raw_descriptor.replace('```xml', '').replace('```', '').strip()
            elif raw_descriptor.startswith('```'):
                raw_descriptor = raw_descriptor.replace('```', '').strip()
            
            print(f"  ✅ LLM application descriptor generation complete: {len(raw_descriptor)} characters generated")
            
            return {
                'content': raw_descriptor,
                'length': len(raw_descriptor),
                'generation_method': 'llm_vector_db_template',
                'metadata': {
                    'dependencies_included': len(dependency_analysis.get('shared_libraries', [])),
                    'vector_context_applied': bool(vector_content.get('library_content')),
                    'template_structure_applied': bool(template_content.get('raw_content')),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            raise Exception(f"LLM application descriptor generation failed: {str(e)}")
    


    
    def _write_application_descriptor_file(self, generated_descriptor: Dict, output_dir: str) -> str:
        """Write generated application descriptor to application.descriptor file in project root"""
        print("  💾 Writing application.descriptor file to project root...")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Write to application.descriptor (not .xml extension)
        descriptor_file_path = os.path.join(output_dir, 'application.descriptor')
        
        try:
            with open(descriptor_file_path, 'w', encoding='utf-8') as f:
                f.write(generated_descriptor['content'])
            
            print(f"  ✅ application.descriptor file written to: {descriptor_file_path}")
            return descriptor_file_path
            
        except Exception as e:
            print(f"  ❌ Failed to write application.descriptor: {str(e)}")
            raise
    
    # Helper methods for content analysis
    def _analyze_xml_structure(self, root: ET.Element) -> Dict[str, Any]:
        """Analyze XML structure of application descriptor template"""
        return {
            'root_tag': root.tag,
            'root_attributes': dict(root.attrib),
            'child_elements': [child.tag for child in root],
            'total_elements': len(list(root.iter())),
            'valid_xml': True
        }
    
    def _extract_library_references(self, content: str) -> List[str]:
        """Extract library reference patterns from template"""
        patterns = [
            r'<library[^>]*name\s*=\s*["\']([^"\']+)["\']',
            r'<libraryReference[^>]*>([^<]+)</libraryReference>',
            r'library["\']?\s*:\s*["\']([^"\']+)["\']'
        ]
        references = []
        for pattern in patterns:
            references.extend(re.findall(pattern, content, re.IGNORECASE))
        return list(set(references))
    
    def _extract_dependency_patterns(self, content: str) -> List[str]:
        """Extract dependency patterns from template"""
        patterns = [
            r'<dependency[^>]*>([^<]+)</dependency>',
            r'depends\s*=\s*["\']([^"\']+)["\']',
            r'require[sd]?\s*["\']([^"\']+)["\']'
        ]
        dependencies = []
        for pattern in patterns:
            dependencies.extend(re.findall(pattern, content, re.IGNORECASE))
        return list(set(dependencies))
    
    def _extract_configuration_sections(self, content: str) -> List[str]:
        """Extract configuration section patterns from template"""
        patterns = [
            r'<configuration[^>]*>.*?</configuration>',
            r'<properties[^>]*>.*?</properties>',
            r'<settings[^>]*>.*?</settings>'
        ]
        sections = []
        for pattern in patterns:
            sections.extend(re.findall(pattern, content, re.IGNORECASE | re.DOTALL))
        return sections
    
    def _extract_template_variables(self, content: str) -> List[str]:
        """Extract template variable patterns"""
        patterns = [
            r'\$\{([^}]+)\}',
            r'%([^%]+)%',
            r'\{\{([^}]+)\}\}'
        ]
        variables = []
        for pattern in patterns:
            variables.extend(re.findall(pattern, content))
        return list(set(variables))
    
    def _extract_library_dependencies(self, json_data: Dict) -> List[Dict]:
        """Extract library dependency information from JSON"""
        dependencies = []
        
        # Look for library-related keys
        if 'libraries' in json_data:
            dependencies.extend(json_data['libraries'])
        if 'dependencies' in json_data:
            dependencies.extend(json_data['dependencies'])
        if 'components' in json_data:
            for component in json_data['components']:
                if 'libraries' in component:
                    dependencies.extend(component['libraries'])
                if 'dependencies' in component:
                    dependencies.extend(component['dependencies'])
        
        return dependencies
    
    def _extract_shared_library_references(self, json_data: Dict) -> List[Dict]:
        """Extract shared library reference information from JSON"""
        references = []
        
        if 'shared_libraries' in json_data:
            references.extend(json_data['shared_libraries'])
        if 'components' in json_data:
            for component in json_data['components']:
                if 'shared_libraries' in component:
                    references.extend(component['shared_libraries'])
        
        return references
    
    def _extract_component_dependencies(self, json_data: Dict) -> List[Dict]:
        """Extract component dependency information from JSON"""
        dependencies = []
        
        if 'components' in json_data:
            for component in json_data['components']:
                if 'component_dependencies' in component:
                    dependencies.extend(component['component_dependencies'])
                if 'requires' in component:
                    dependencies.extend(component['requires'])
        
        return dependencies
    
    def _extract_external_references(self, json_data: Dict) -> List[Dict]:
        """Extract external reference information from JSON"""
        references = []
        
        if 'external_systems' in json_data:
            references.extend(json_data['external_systems'])
        if 'external_libraries' in json_data:
            references.extend(json_data['external_libraries'])
        if 'components' in json_data:
            for component in json_data['components']:
                if 'external_references' in component:
                    references.extend(component['external_references'])
        
        return references


def main():
    """Test harness for application descriptor generator"""
    generator = ApplicationDescriptorGenerator()
    
    # Test with sample inputs
    result = generator.generate_application_descriptor(
        template_path="application_descriptor.xml",
        component_mapping_json_path="component_mapping.json",
        output_dir="test_output"
    )
    
    print(f"\n🎯 Application Descriptor Generation Results:")
    print(f"✅ Status: {result['status']}")
    print(f"📊 Application Descriptor Generated: {result['application_descriptor_generated']}")
    print(f"🧠 LLM Analysis Calls: {result['llm_analysis_calls']}")
    print(f"⚡ LLM Generation Calls: {result['llm_generation_calls']}")


if __name__ == "__main__":
    main()