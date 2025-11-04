#!/usr/bin/env python3
"""
Simple Project Generator v2.0 - Template Based Only
Purpose: Generate .project file using template with simple placeholder replacement
Input: project.xml template → Replace {project_name} with folder name → Write .project file
Output: Creates .project file for IBM ACE Toolkit compatibility
No LLM Required: Pure template-based generation
"""

import os
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import json as json_module

class ProjectGenerator:
    """
    Simple Project Generator - Template-based only
    NO LLM CALLS - Direct template processing with placeholder replacement
    """
    
    def __init__(self, groq_api_key: str = None, **kwargs):
        """
        Initialize simple project generator
        
        Args:
            groq_api_key (str): Not used - kept for compatibility
            **kwargs: Other parameters - ignored for compatibility
        """
        # Ignore all LLM-related parameters - we don't need them!
        print("🎯 Simple Project Generator initialized - No LLM required")
        print("   ✅ Parameters ignored (LLM not needed): groq_api_key, etc.")


        
    def generate_project_file(self, 
                            vector_content: str,
                            template_path: str,
                            business_requirements_json_path: str,
                            output_dir: str,
                            biztalk_folder: str = None,  # âœ… FIXED: Added parameter
                            generated_components_dir: str = None) -> Dict[str, Any]:
        """
        Generate final .project file using MINIMAL functionality - NO undefined method calls
        âœ… USES ONLY: os, json, basic file operations - NO extraction methods
        """
        print("Starting Simple Project File Generation")
        
        # Extract project name from output directory
        project_name = os.path.basename(os.path.abspath(output_dir))
        if not project_name or project_name == '.':
            project_name = "Enhanced_ACE_Project"
        
        print(f"Project name: {project_name}")
        
        try:
            # ✅ STEP 1: Read template file (template-only, no fallback)
            print("📄 Step 1: Reading template...")
            
            # Try multiple possible template locations
            template_locations = [
                template_path,
                os.path.join(os.path.dirname(__file__), 'templates', 'project_template.xml'),
                os.path.join(os.getcwd(), 'templates', 'project_template.xml'),
            ]
            
            template_content = None
            for template_location in template_locations:
                if os.path.exists(template_location):
                    with open(template_location, 'r', encoding='utf-8') as f:
                        template_content = f.read()
                    print(f"  ✅ Template loaded from: {template_location}")
                    break
            
            if template_content is None:
                error_msg = "❌ project_template.xml not found. Searched:\n" + "\n".join(f"  - {loc}" for loc in template_locations)
                print(error_msg)
                raise FileNotFoundError(error_msg)

            
            # STEP 2: Read JSON file (basic JSON reading only)
            print("Step 2: Reading JSON mappings...")
            if os.path.exists(business_requirements_json_path):
                with open(business_requirements_json_path, 'r', encoding='utf-8') as f:
                    json_data = json_module.load(f)
                print(f" JSON loaded: {len(str(json_data))} characters")
            else:
                json_data = {}
                print("  JSON not found - using empty data")
            
            # STEP 3: Count component files (basic file counting only)
            print("Step 3: Counting component files...")
            component_count = 0
            if generated_components_dir and os.path.exists(generated_components_dir):
                for root, dirs, files in os.walk(generated_components_dir):
                    for file in files:
                        if file.endswith(('.esql', '.xsd', '.xsl', '.msgflow', '.xml')):
                            component_count += 1
            print(f" Found {component_count} component files")
            
            # STEP 4: Generate project content (simple string replacement only)
            print("Step 4: Generating project content...")
            
            # Simple template replacement - This handles {project_name} placeholder correctly âœ…
            project_content = template_content.replace('{project_name}', project_name)
            project_content = project_content.replace('${project_name}', project_name)
            project_content = project_content.replace('PROJECT_NAME', project_name)
            
            # DELETED PROBLEMATIC REGEX BLOCK - No longer needed!
            # The above replacements already handle the <name>{project_name}</name> correctly
            # Removed lines that were:
            # if '<name>' in project_content and '</name>' in project_content:
            #     import re
            #     project_content = re.sub(r'<name>.*?</name>', f'<name>{project_name}</name>', project_content)
            
            print(f"  Project content generated: {len(project_content)} characters")
            
            # STEP 5: Write project file (basic file writing only)
            print("Step 5: Writing project file...")
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Write .project file
            project_file_path = os.path.join(output_dir, '.project')
            with open(project_file_path, 'w', encoding='utf-8') as f:
                f.write(project_content)
            
            print(f" Project file written: {project_file_path}")
            print("Project generation completed successfully!")
            
            # Return success with minimal data
            return {
                'status': 'success',
                'project_file_path': project_file_path,
                'project_name': project_name,
                'project_file_generated': True,
                'content_source': 'simple_template',
                'content_length': len(project_content),
                'llm_analysis_calls': 0,  # No LLM used
                'llm_generation_calls': 0,  # No LLM used
                'processing_metadata': {
                    'template_found': os.path.exists(template_path),
                    'json_found': os.path.exists(business_requirements_json_path),
                    'components_found': component_count,
                    'vector_content_length': len(vector_content) if vector_content else 0
                }
            }
            
        except Exception as e:
            print(f"Project generation failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'project_file_generated': False,
                'llm_analysis_calls': 0,
                'llm_generation_calls': 0
            }
        

    
    def _extract_project_name(self, output_dir: str) -> str:
        """
        Extract project name from output directory path
        
        Args:
            output_dir (str): Output directory path
            
        Returns:
            str: Project name extracted from folder name
        """
        project_name = os.path.basename(os.path.abspath(output_dir))
        
        # Validate project name
        if not project_name or project_name == '.' or project_name == '..':
            raise ValueError(f"Invalid output directory - cannot extract project name: {output_dir}")
        
        # Clean project name (remove invalid characters for XML)
        project_name = project_name.replace(' ', '_').replace('-', '_')
        
        return project_name
    
    def _load_template(self, template_path: str) -> str:
        """
        Load template content from file
        
        Args:
            template_path (str): Path to template file
            
        Returns:
            str: Template content
        """
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found: {template_path}")
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            
            if not template_content.strip():
                raise ValueError(f"Template file is empty: {template_path}")
            
            return template_content
            
        except UnicodeDecodeError:
            # Try with different encodings
            try:
                with open(template_path, 'r', encoding='latin-1') as f:
                    template_content = f.read()
                return template_content
            except Exception as e:
                raise Exception(f"Failed to read template file with multiple encodings: {e}")
        
        except Exception as e:
            raise Exception(f"Failed to load template: {e}")
    
    def _replace_placeholder(self, template_content: str, project_name: str) -> str:
        """
        Replace {project_name} placeholder in template with actual project name
        
        Args:
            template_content (str): Original template content
            project_name (str): Project name to replace placeholder with
            
        Returns:
            str: Template content with placeholder replaced
        """
        # Primary placeholder replacement
        project_content = template_content.replace('{project_name}', project_name)
        
        # Also handle variations (for robustness)
        variations = [
            '{{project_name}}',
            '{PROJECT_NAME}',
            '{{PROJECT_NAME}}',
            '<name>{project_name}</name>',
            '<name>{{project_name}}</name>'
        ]
        
        for variation in variations:
            if variation in template_content:
                if '<name>' in variation:
                    # For XML name tags, replace with proper XML format
                    project_content = project_content.replace(variation, f'<name>{project_name}</name>')
                else:
                    project_content = project_content.replace(variation, project_name)
        
        # Verify replacement occurred
        if project_content == template_content:
            print("⚠️ Warning: No placeholder found in template. Template used as-is.")
        else:
            print(f"✅ Placeholder replacement successful")
        
        return project_content
    
    def _write_project_file(self, project_content: str, output_dir: str) -> str:
        """
        Write project content to .project file in output directory
        
        Args:
            project_content (str): Generated project file content
            output_dir (str): Output directory
            
        Returns:
            str: Path to written .project file
        """
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Write to .project file (note the dot prefix)
        project_file_path = os.path.join(output_dir, '.project')
        
        try:
            with open(project_file_path, 'w', encoding='utf-8') as f:
                f.write(project_content)
            
            # Verify file was written
            if not os.path.exists(project_file_path):
                raise Exception("File write verification failed")
            
            return project_file_path
            
        except Exception as e:
            raise Exception(f"Failed to write .project file: {e}")
    
    def validate_template(self, template_path: str) -> Dict[str, Any]:
        """
        Validate template file and check for placeholders
        
        Args:
            template_path (str): Path to template file
            
        Returns:
            Dict[str, Any]: Validation results
        """
        try:
            template_content = self._load_template(template_path)
            
            # Check for placeholders
            placeholders_found = []
            placeholder_patterns = [
                '{project_name}', '{{project_name}}',
                '{PROJECT_NAME}', '{{PROJECT_NAME}}',
                '<name>{project_name}</name>',
                '<name>{{project_name}}</name>'
            ]
            
            for pattern in placeholder_patterns:
                if pattern in template_content:
                    placeholders_found.append(pattern)
            
            # Basic XML structure check
            has_xml_declaration = template_content.strip().startswith('<?xml')
            has_project_description = '<projectDescription>' in template_content
            has_name_tag = '<name>' in template_content
            has_natures = '<natures>' in template_content
            has_build_spec = '<buildSpec>' in template_content
            
            validation_result = {
                'status': 'valid',
                'template_path': template_path,
                'content_length': len(template_content),
                'placeholders_found': placeholders_found,
                'has_placeholders': len(placeholders_found) > 0,
                'xml_structure': {
                    'has_xml_declaration': has_xml_declaration,
                    'has_project_description': has_project_description,
                    'has_name_tag': has_name_tag,
                    'has_natures': has_natures,
                    'has_build_spec': has_build_spec
                },
                'recommendations': []
            }
            
            # Add recommendations
            if not validation_result['has_placeholders']:
                validation_result['recommendations'].append(
                    "No {project_name} placeholder found. Template will be used as-is."
                )
            
            if not has_name_tag:
                validation_result['recommendations'].append(
                    "Template missing <name> tag. Consider adding <name>{project_name}</name>"
                )
            
            return validation_result
            
        except Exception as e:
            return {
                'status': 'error',
                'error_message': str(e),
                'template_path': template_path
            }


def main():
    """
    Test harness and example usage for Simple Project Generator
    """
    print("🧪 Simple Project Generator - Test Mode")
    
    generator = ProjectGenerator()
    
    # Example usage
    template_path = "project.xml"  # Update with actual template path
    output_dir = "test_project_output"  # This folder name becomes project name
    
    print(f"\n📋 Test Configuration:")
    print(f"   Template: {template_path}")
    print(f"   Output Dir: {output_dir}")
    print(f"   Expected Project Name: {os.path.basename(os.path.abspath(output_dir))}")
    
    # Validate template first (if exists)
    if os.path.exists(template_path):
        print(f"\n🔍 Validating template...")
        validation = generator.validate_template(template_path)
        print(f"   Status: {validation['status']}")
        if validation.get('has_placeholders'):
            print(f"   Placeholders: {validation['placeholders_found']}")
        if validation.get('recommendations'):
            for rec in validation['recommendations']:
                print(f"   💡 {rec}")
    
        # Generate project file
        print(f"\n🚀 Generating project file...")
        result = generator.generate_project_file(template_path, output_dir)
        
        print(f"\n📊 Generation Results:")
        print(f"   Status: {result['status']}")
        if result['status'] == 'success':
            print(f"   Project Name: {result['project_name']}")
            print(f"   File Path: {result['project_file_path']}")
            print(f"   Content Length: {result['content_length']} characters")
            print(f"   LLM Calls: {result['llm_calls']}")
            print(f"   Method: {result['method']}")
        else:
            print(f"   Error: {result.get('error_message')}")
    else:
        print(f"❌ Template file not found: {template_path}")
        print("   Create a project.xml template file to test the generator")
        
        # Show example template
        print(f"\n📝 Example Template Content (save as {template_path}):")
        example_template = '''<?xml version="1.0" encoding="UTF-8"?>
<projectDescription>
	<name>{project_name}</name>
	<comment></comment>
	<projects>
		<project>EPIS_CommonUtils_Lib</project>
		<project>EPIS_Consumer_Lib_v2</project>
		<project>EPIS_BlobStorage_Lib</project>
		<project>EPIS_MessageEnrichment_StaticLib</project>
		<project>EPIS_CommonFlows_Lib</project>
        <project>EPIS_CargoWiseOne_eAdapter_Lib</project>
        <project>EPIS_CargoWiseOne_Schemas_Lib</project>
	</projects>
	<buildSpec>
		<buildCommand>
			<name>com.ibm.etools.mft.applib.applibbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.applib.applibresourcevalidator</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.connector.policy.ui.PolicyBuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.applib.mbprojectbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.msg.validation.dfdl.mlibdfdlbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.flow.adapters.adapterbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.flow.sca.scabuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.msg.validation.dfdl.mbprojectresourcesbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.esql.lang.esqllangbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.map.builder.mslmappingbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.flow.msgflowxsltbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.flow.msgflowbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.decision.service.ui.decisionservicerulebuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.pattern.capture.PatternBuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.json.builder.JSONBuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.restapi.ui.restApiDefinitionsBuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.policy.ui.policybuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.msg.assembly.messageAssemblyBuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.msg.validation.dfdl.dfdlqnamevalidator</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.bar.ext.barbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.unittest.ui.TestCaseBuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
	</buildSpec>
	<natures>
		<nature>com.ibm.etools.msgbroker.tooling.applicationNature</nature>
		<nature>com.ibm.etools.msgbroker.tooling.messageBrokerProjectNature</nature>
	</natures>
</projectDescription>'''
        print(example_template)


if __name__ == "__main__":
    main()