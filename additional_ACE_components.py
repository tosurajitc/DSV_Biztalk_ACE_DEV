#!/usr/bin/env python3
"""
additional_ACE_components.py - Subflow Generator for ACE Projects

Analyzes messageflow XML and generates required subflow components 
based on references in the messageflow file. Uses external template files only.
"""

import os
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional

class AdditionalComponentsGenerator:
    """
    Analyzes messageflow XML and generates additional components 
    like subflows (RECSATInput, RECSATOutput) using external templates.
    """
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize the component generator
        
        Args:
            template_dir: Directory containing template files (defaults to project root)
        """
        # Set template directory
        self.template_dir = template_dir or str(Path.cwd())
        print(f"  📁 Template directory: {self.template_dir}")
        
        # Template mapping
        self.template_mapping = {
            "RECSATInput": "templates\RECSAT_template.xml",
            "RECSATOutput": "templates\SNDSAT_template.xml"
        }
    
    def analyze_and_generate(self, msgflow_path: str, output_dir: str, business_req_path: str = None) -> Dict[str, List[str]]:
        """
        Analyze messageflow and generate additional components
        
        Args:
            msgflow_path: Path to the messageflow XML file
            output_dir: Directory to output generated files
            business_req_path: Path to business requirements JSON (optional)
            
        Returns:
            Dict with keys: 'subflows', 'esql', etc. and values as lists of generated component names
        """
        print("🔍 Analyzing messageflow for additional components...")
        
        # Load business requirements if available (for future use)
        business_requirements = {}
        if business_req_path and os.path.exists(business_req_path):
            try:
                with open(business_req_path, 'r') as f:
                    business_requirements = json.load(f)
                print(f"  ✓ Loaded business requirements from {business_req_path}")
            except Exception as e:
                print(f"  ⚠️ Could not load business requirements: {e}")
        
        # Initialize result dictionary
        additional_components = {
            'subflows': [],
            'esql': [],
            'wsdl': [],
            'additional_xsl': []
        }
        
        try:
            # Open messageflow file and read content
            with open(msgflow_path, 'r') as f:
                content = f.read()
                
            # First try parsing as XML
            try:
                tree = ET.parse(msgflow_path)
                root = tree.getroot()
                
                # Find subflow components by looking for componentName attributes
                for elem in root.findall(".//*[@componentName]"):
                    subflow_name = elem.get('componentName')
                    if subflow_name and subflow_name not in additional_components['subflows']:
                        print(f"  → Found subflow reference: {subflow_name}")
                        additional_components['subflows'].append(subflow_name)
            except ET.ParseError:
                print("  ⚠️ XML parsing failed, using regex fallback")
            
            # If XML parsing didn't find subflows, or as a backup, use regex
            if not additional_components['subflows']:
                # Check for RECSATInput reference
                if 'RECSATInput' in content and 'RECSATInput' not in additional_components['subflows']:
                    print(f"  → Found RECSATInput reference")
                    additional_components['subflows'].append('RECSATInput')
                    
                # Check for RECSATOutput reference  
                if 'RECSATOutput' in content and 'RECSATOutput' not in additional_components['subflows']:
                    print(f"  → Found RECSATOutput reference")
                    additional_components['subflows'].append('RECSATOutput')
            
            # Generate the subflows
            if additional_components['subflows']:
                self._generate_subflows(
                    additional_components['subflows'], 
                    msgflow_path, 
                    output_dir
                )
            
            return additional_components
            
        except Exception as e:
            print(f"  ❌ Error analyzing messageflow: {e}")
            return additional_components
    
    def _generate_subflows(self, subflow_names: List[str], msgflow_path: str, output_dir: str):
        """
        Generate required subflows based on messageflow analysis using template files
        
        Args:
            subflow_names: List of subflow names to generate
            msgflow_path: Path to the messageflow file
            output_dir: Directory to output generated files
        """
        print(f"  📦 Generating {len(subflow_names)} subflow(s)...")
        
        # Create subflows directory
        subflows_dir = os.path.join(output_dir, 'subflows')
        os.makedirs(subflows_dir, exist_ok=True)
        
        # Extract flow name from messageflow path for template references
        flow_name = os.path.basename(msgflow_path).replace('.msgflow', '')
        
        # Generate each required subflow
        for subflow_name in subflow_names:
            # Find template file
            template_file = self._find_template_file(subflow_name)
            
            if template_file and os.path.exists(template_file):
                try:
                    # Copy and process the template
                    self._process_template_file(
                        template_file, 
                        subflow_name,
                        flow_name,
                        subflows_dir
                    )
                    print(f"    ✅ Generated subflow: {subflow_name}.subflow from template")
                except Exception as e:
                    print(f"    ❌ Error processing template for {subflow_name}: {e}")
            else:
                print(f"    ⚠️ No template found for {subflow_name}. Subflow will not be generated.")
    
    def _find_template_file(self, subflow_name: str) -> Optional[str]:
        """
        Find the template file for a given subflow
        
        Args:
            subflow_name: Name of the subflow
            
        Returns:
            Path to template file or None if not found
        """
        # Check if we have a mapping for this subflow
        if subflow_name in self.template_mapping:
            template_filename = self.template_mapping[subflow_name]
            template_path = os.path.join(self.template_dir, template_filename)
            
            # Check if file exists
            if os.path.exists(template_path):
                return template_path
            
            # Check in project directory (if template_dir is not project dir)
            project_path = os.path.join(str(Path.cwd()), template_filename)
            if os.path.exists(project_path):
                return project_path
                
            # Not found in expected locations
            print(f"    ⚠️ Template file not found: {template_filename}")
            print(f"    ⚠️ Looked in: {template_path} and {project_path}")
            return None
        else:
            print(f"    ⚠️ No template mapping for subflow: {subflow_name}")
            return None
        
        
    def _process_template_file(self, template_path: str, subflow_name: str, flow_name: str, output_dir: str):
        """
        Process template file using an external template with dynamic node configuration
        
        Args:
            template_path: Path to the template file
            subflow_name: Name of the subflow or component to create
            flow_name: Name of the parent flow
            output_dir: Directory to save processed file
        """
        try:
            # First look for the standard template in the templates folder
            subflow_template_path = os.path.join(os.path.dirname(template_path), "subflow_template.xml")
            
            if os.path.exists(subflow_template_path):
                with open(subflow_template_path, 'r') as f:
                    template_content = f.read()
                print(f"  ✓ Using standard subflow template: {subflow_template_path}")
            elif os.path.exists(template_path):
                with open(template_path, 'r') as f:
                    template_content = f.read()
                print(f"  ✓ Using specific template file: {template_path}")
            else:
                print(f"  ⚠️ Template file not found, component will not be generated")
                return False
                
            # Load business requirements
            business_req = self._get_business_requirements()
            
            # Configure node-specific properties based on component type
            extension = ".subflow"  # Default extension
            
            # Define node configuration based on component type
            if subflow_name == "RECSATInput":
                # Input subflow with FileRead node
                node_namespace = "ComIbmFileRead.msgnode"
                node_label = "File Read"
                
                # Get properties from business requirements or use defaults
                input_directory = business_req.get('input_directory', '/var/mqsi/input')
                file_pattern = business_req.get('file_pattern', '*.xml')
                message_domain = business_req.get('message_domain', 'XMLNSC')
                
                node_properties = f'''inputDirectory="{input_directory}" 
                filenamePattern="{file_pattern}" 
                messageDomainProperty="{message_domain}"'''
                
            elif subflow_name == "RECSATOutput":
                # Output subflow with FileOutput node
                node_namespace = "ComIbmFileOutput.msgnode"
                node_label = "File Output"
                
                # Get properties from business requirements or use defaults
                output_directory = business_req.get('output_directory', '/var/mqsi/output')
                
                node_properties = f'''outputDirectory="{output_directory}" 
                outputFilename="${{Root.MQMD.MsgId}}.xml"
                outputMode="archiveAndDeleteInput"'''
                
            elif "FileInput" in subflow_name:
                # FileInput node (not a subflow)
                node_namespace = "ComIbmFileInput.msgnode"
                node_label = "File Input"
                extension = ".msgnode"
                
                input_directory = business_req.get('input_directory', '/var/mqsi/input')
                file_pattern = business_req.get('file_pattern', '*.xml')
                message_domain = business_req.get('message_domain', 'XMLNSC')
                
                node_properties = f'''inputDirectory="{input_directory}" 
                filenamePattern="{file_pattern}" 
                messageDomainProperty="{message_domain}"'''
                
            elif "Compute" in subflow_name:
                # Compute node (not a subflow)
                node_namespace = "ComIbmCompute.msgnode"
                node_label = "Compute"
                extension = ".msgnode"
                
                # ESQL expression reference
                esql_module = business_req.get('esql_module', flow_name)
                esql_method = business_req.get('esql_method', 'Main')
                
                node_properties = f'''computeExpression="esql://routine/{flow_name}#{esql_module}.{esql_method}"'''
                
            else:
                # Generic passthrough for unknown types
                node_namespace = "ComIbmPassthru.msgnode"
                node_label = "Pass through"
                node_properties = ""
            
            # Replace all placeholders in the template
            replacements = {
                '{SUBFLOW_NAME}': subflow_name,
                '{FLOW_NAME}': flow_name,
                '{NODE_NAMESPACE}': node_namespace,
                '{NODE_PROPERTIES}': node_properties,
                '{NODE_LABEL}': node_label
            }
            
            processed_content = template_content
            for placeholder, value in replacements.items():
                processed_content = processed_content.replace(placeholder, value)
            
            # Ensure correct URI and namespace based on component type
            nsURI_value = f"{subflow_name}{extension}"
            processed_content = processed_content.replace('{COMPONENT_URI}', nsURI_value)
            
            # Create output file with appropriate extension
            output_file = os.path.join(output_dir, f"{subflow_name}{extension}")
            with open(output_file, 'w') as f:
                f.write(processed_content)
                
            print(f"  ✅ Generated {subflow_name}{extension} with dynamic configuration")
            return True
            
        except Exception as e:
            print(f"  ❌ Error processing component template: {e}")
            return False
    
            
    def _get_business_requirements(self):
        """Get business requirements from available sources"""
        business_req = {}
        
        # Try to find business_requirements.json in known locations
        potential_paths = [
            os.path.join(str(Path.cwd()), 'business_requirements.json'),
            os.path.join(str(Path.cwd()), 'output', 'business_requirements.json')
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        business_req = json.load(f)
                    print(f"  ✓ Loaded business requirements from {path}")
                    break
                except Exception as e:
                    print(f"  ⚠️ Failed to load business requirements from {path}: {e}")
        
        return business_req
    
    def _update_namespace_uris(self, content: str, subflow_name: str) -> str:
        """
        Ensure namespace URIs and prefixes are correct in the template
        
        Args:
            content: XML content
            subflow_name: Name of the subflow
            
        Returns:
            Updated XML content
        """
        # Simple string replacement to update namespaces if needed
        # More sophisticated parsing could be done with XML libraries if needed
        if 'nsURI="' in content:
            content = content.replace('nsURI="RECSAT_template.subflow"', f'nsURI="{subflow_name}.subflow"')
            content = content.replace('nsURI="SNDSAT_template.subflow"', f'nsURI="{subflow_name}.subflow"')
            content = content.replace('nsPrefix="RECSAT_template.subflow"', f'nsPrefix="{subflow_name}.subflow"')
            content = content.replace('nsPrefix="SNDSAT_template.subflow"', f'nsPrefix="{subflow_name}.subflow"')
        
        return content


def main():
    """Test function for direct execution"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python additional_ACE_components.py <msgflow_path> <output_dir> [business_req_path] [template_dir]")
        sys.exit(1)
    
    msgflow_path = sys.argv[1]
    output_dir = sys.argv[2]
    business_req_path = sys.argv[3] if len(sys.argv) > 3 else None
    template_dir = sys.argv[4] if len(sys.argv) > 4 else None
    
    generator = AdditionalComponentsGenerator(template_dir)
    result = generator.analyze_and_generate(msgflow_path, output_dir, business_req_path)
    
    print("\nGeneration Results:")
    print(f"Subflows: {len(result['subflows'])}")
    print(f"Additional ESQL: {len(result['esql'])}")
    print(f"WSDL: {len(result['wsdl'])}")
    print(f"XSL: {len(result['additional_xsl'])}")


if __name__ == "__main__":
    main()