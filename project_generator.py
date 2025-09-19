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
                            template_path: str,
                            component_mapping_json_path: str = None,
                            output_dir: str = None,
                            generated_components_dir: str = None,
                            vector_content: str = None) -> Dict[str, Any]:
        """
        Generate .project file using template with simple placeholder replacement
        
        Args:
            template_path (str): Path to project.xml template file
            component_mapping_json_path (str): Not used - kept for compatibility
            output_dir (str): Output directory (project name extracted from this path)
            generated_components_dir (str): Not used - kept for compatibility  
            vector_content (str): Not used - kept for compatibility
        
        Returns:
            Dict[str, Any]: Generation results
        """
        print("🚀 Starting Simple Project File Generation")
        
        # Handle parameter compatibility - output_dir might be in different positions
        if output_dir is None:
            if component_mapping_json_path and os.path.isdir(component_mapping_json_path):
                # Likely output_dir was passed as second parameter
                output_dir = component_mapping_json_path
                component_mapping_json_path = None
        
        if not output_dir:
            raise ValueError("output_dir parameter is required")
        
        try:
            # Step 1: Extract project name from output directory
            project_name = self._extract_project_name(output_dir)
            print(f"📁 Project name extracted: {project_name}")
            
            # Step 2: Load template content
            template_content = self._load_template(template_path)
            print(f"📄 Template loaded: {len(template_content)} characters")
            
            # Step 3: Replace placeholder with project name
            project_content = self._replace_placeholder(template_content, project_name)
            print(f"🔄 Placeholder replaced: {project_name}")
            
            # Step 4: Write .project file
            project_file_path = self._write_project_file(project_content, output_dir)
            print(f"💾 .project file written: {project_file_path}")
            
            # Return results
            result = {
                'status': 'success',
                'project_name': project_name,
                'project_file_path': project_file_path,
                'template_path': template_path,
                'output_dir': output_dir,
                'content_length': len(project_content),
                'generation_time': datetime.now().isoformat(),
                'llm_calls': 0,  # Zero LLM calls!
                'method': 'template_replacement'
            }
            
            print("✅ Simple Project Generation Complete!")
            return result
            
        except Exception as e:
            error_result = {
                'status': 'error',
                'error_message': str(e),
                'generation_time': datetime.now().isoformat(),
                'llm_calls': 0
            }
            print(f"❌ Generation failed: {str(e)}")
            return error_result
    
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