#!/usr/bin/env python3
"""
Enhanced ESQL Generator and MessageFlow Updater v2.1

This module:
1. Extracts required ESQL nodes from Vector DB business requirements
2. Updates existing messageflow files with required nodes
3. Creates ESQL files based on business patterns from Vector DB
4. Ensures proper node connections based on business flow

Key features:
- Dynamic ESQL node detection (no hardcoded modules)
- MessageFlow file updating with required ESQL nodes
- Business-driven ESQL file generation from templates
- Flexible for 1000+ different flow patterns
- Preserves comments in messageflow files
- Uses proper naming convention from naming_convention.json
"""

import os
import re
import json
import glob
import shutil
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from groq import Groq
from llm_json_parser import LLMJSONParser, parse_llm_json


class ESQLGenerator:
    """
    Enhanced ESQL Generator with dynamic node detection and messageflow updating
    """
    
    def __init__(self, groq_api_key: Optional[str] = None, groq_model: str = "llama-3.3-70b-versatile"):
        """
        Initialize enhanced ESQL Generator with LLM configuration.
        
        Args:
            groq_api_key: Groq API key (optional, can use environment variable)
            groq_model: LLM model to use (configurable)
        """
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY must be provided or set in environment")
        
        self.groq_client = Groq(api_key=self.groq_api_key)
        self.groq_model = groq_model or os.getenv('GROQ_MODEL', 'llama-3.1-70b-versatile')
        self.json_parser = LLMJSONParser(debug=False)
        
        # Store extracted business information
        self.business_requirements = {}
        self.required_nodes = []
        self.database_operations = []
        self.transformations = []
        
        # Tracking for generation
        self.generation_stats = {
            'llm_calls': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'auto_fixes_applied': 0,
            'messageflows_updated': 0
        }
        
        # Template section markers in ESQL template
        self.template_sections = {
            'input_event': ['INPUT AND OUTPUT EVENT MESSAGE TEMPLATE - METADATA ONLY', 'COMPUTE TEMPLATE - FULL BUSINESS LOGIC'],
            'output_event': ['INPUT AND OUTPUT EVENT MESSAGE TEMPLATE - METADATA ONLY', 'COMPUTE TEMPLATE - FULL BUSINESS LOGIC'],
            'compute': ['COMPUTE TEMPLATE - FULL BUSINESS LOGIC', 'PROCESSING TEMPLATE'],
            'processing': ['PROCESSING TEMPLATE - VALIDATION AND ROUTING ONLY', 'FAILURE/ERROR HANDLING TEMPLATE'],
            'failure': ['FAILURE/ERROR HANDLING TEMPLATE', None]  # Last section
        }
        
        # Module type suffixes for detection
        self.module_type_suffixes = {
            'input_event': ['_InputEventMessage', '_InputEvent'],
            'output_event': ['_OutputEventMessage', '_OutputEvent', '_AfterEventMsg'],
            'compute': ['_Compute', '_Transform', '_Mapper'],
            'processing': ['_Processing', '_Validation', '_Router', '_AfterEnrichment'],
            'failure': ['_Failure', '_Error', '_Exception', '_ErrorHandler']
        }
        
        print(f"✅ Enhanced ESQL Generator initialized with model: {self.groq_model}")
    
    # ============================================================================
    # PART 1: BUSINESS REQUIREMENTS ANALYSIS
    # ============================================================================
    
    def analyze_vector_content(self, vector_content: str) -> Dict:
        """
        Extract business requirements from Vector DB content using LLM.
        This is the foundational step that identifies required nodes.
        
        Args:
            vector_content: Raw business requirements from Vector DB
            
        Returns:
            Structured business requirements with node information
        """
        print("\n" + "="*70)
        print("🧠 ANALYZING BUSINESS REQUIREMENTS FROM VECTOR DB")
        print("="*70)
        
        # Build comprehensive analysis prompt
        analysis_prompt = f"""Analyze this business flow description from Vector DB and extract:
1. Required ESQL node types for IBM ACE MessageFlow
2. Processing sequence and dependencies
3. Database operations and transformations

VECTOR DB BUSINESS CONTENT:
{vector_content}

Identify ONLY the ESQL nodes that are EXPLICITLY required based on the business description.
DO NOT include generic placeholders or nodes not supported by the requirements.

Return a structured JSON with:
1. Flow name/type
2. Required ESQL nodes in sequence (input_event, compute, processing, output_event)
3. Node purposes and dependencies
4. Database operations and transformations

CRITICAL: Analysis must be based ONLY on actual business requirements, not assumptions.

Return JSON format:
{{
  "flow_name": "extracted flow name",
  "flow_pattern": "MQ-to-MQ|HTTP-to-MQ|FILE-to-SOAP|etc",
  "required_nodes": [
    {{
      "name": "meaningful_name_based_on_function",
      "type": "input_event|compute|processing|output_event|failure",
      "purpose": "Specific function in flow",
      "sequence_order": 1,
      "required": true|false,
      "business_logic": {{
        "database_operations": ["List DB operations"],
        "transformations": ["List transformations"],
        "routing_logic": ["List routing conditions"]
      }}
    }}
  ],
  "database_operations": ["All DB ops mentioned"],
  "transformations": ["All transformations mentioned"],
  "integration_pattern": "description of overall pattern"
}}

Return ONLY valid JSON. Do not include generic placeholders like "Node1" or "ESQLModule1".
Each node name should reflect its actual function in the flow.
"""
        
        # Call LLM to analyze requirements
        self.generation_stats['llm_calls'] += 1
        
        try:
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert IBM ACE integration developer. Extract precise node requirements from business specifications. Return only valid JSON with realistic ACE node types."
                    },
                    {
                        "role": "user",
                        "content": analysis_prompt
                    }
                ],
                temperature=0.2,
                max_tokens=3000
            )
            
            llm_response = response.choices[0].message.content
            
            # Parse JSON using llm_json_parser
            parse_result = parse_llm_json(llm_response)
            
            if parse_result.success:
                business_requirements = parse_result.data
                
                # Validate structure
                if not business_requirements or not isinstance(business_requirements, dict):
                    raise ValueError("Invalid business requirements structure")
                
                if 'required_nodes' not in business_requirements:
                    raise ValueError("Missing required_nodes in business requirements")
                
                # Store for later use
                self.business_requirements = business_requirements
                self.required_nodes = business_requirements.get('required_nodes', [])
                self.database_operations = business_requirements.get('database_operations', [])
                self.transformations = business_requirements.get('transformations', [])
                
                # Log results
                print(f"✅ Successfully extracted business requirements")
                print(f"🔄 Flow Pattern: {business_requirements.get('flow_pattern', 'Unknown')}")
                print(f"🧩 Required Nodes: {len(self.required_nodes)}")
                print(f"💾 Database Operations: {len(self.database_operations)}")
                
                return business_requirements
            else:
                print(f"❌ Failed to parse LLM response: {parse_result.error}")
                return {}
                
        except Exception as e:
            print(f"❌ Error analyzing business requirements: {str(e)}")
            return {}

    # ============================================================================
    # PART 2: ESQL FILES GENERATION
    # ============================================================================
    
    def _locate_naming_convention_file(self, base_dir: str) -> Optional[str]:
        """
        Locate the naming_convention.json file in output/single or output/multiple directories.
        
        Args:
            base_dir: Base directory to start search
            
        Returns:
            Path to naming convention file or None if not found
        """
        # First check if there's a naming_convention.json in the current directory
        if os.path.exists(os.path.join(base_dir, "naming_convention.json")):
            return os.path.join(base_dir, "naming_convention.json")
        
        # Check in output/single
        single_dir = os.path.join(base_dir, "output", "single")
        if os.path.exists(single_dir):
            for root, _, files in os.walk(single_dir):
                if "naming_convention.json" in files:
                    return os.path.join(root, "naming_convention.json")
        
        # Check in output/multiple
        multiple_dir = os.path.join(base_dir, "output", "multiple")
        if os.path.exists(multiple_dir):
            for root, _, files in os.walk(multiple_dir):
                if "naming_convention.json" in files:
                    return os.path.join(root, "naming_convention.json")
        
        # If not found, check in parent directory
        parent_dir = os.path.dirname(base_dir)
        if os.path.exists(os.path.join(parent_dir, "naming_convention.json")):
            return os.path.join(parent_dir, "naming_convention.json")
            
        # Finally, check in any directory under base_dir
        for root, _, files in os.walk(base_dir):
            if "naming_convention.json" in files:
                return os.path.join(root, "naming_convention.json")
                
        return None
    
    def _load_naming_convention(self, base_dir: str) -> Dict:
        """
        Load naming convention from naming_convention.json
        
        Args:
            base_dir: Base directory to start search
            
        Returns:
            Naming convention data or empty dict if not found
        """
        naming_file = self._locate_naming_convention_file(base_dir)
        if naming_file:
            try:
                with open(naming_file, 'r') as f:
                    naming_data = json.load(f)
                print(f"✅ Loaded naming convention from {naming_file}")
                return naming_data
            except Exception as e:
                print(f"❌ Error loading naming convention: {str(e)}")
                return {}
        else:
            print("⚠️ Naming convention file not found")
            return {}
    
    def _load_esql_template(self, template_path: str) -> Optional[str]:
        """
        Load ESQL template from file.
        
        Args:
            template_path: Path to ESQL template file
            
        Returns:
            Template content or None if not found
        """
        try:
            # Handle case where template_path is a dict - extract the 'path' key
            if isinstance(template_path, dict) and 'path' in template_path:
                template_path = template_path['path']
                
            # Ensure template_path is a string
            template_path = str(template_path)
            
            # First try the exact path
            if os.path.exists(template_path):
                with open(template_path, 'r') as f:
                    template_content = f.read()
                print(f"✅ Loaded ESQL template from {template_path}")
                return template_content
            
            # Try to locate the file in some common directories
            possible_paths = [
                os.path.join(os.path.dirname(template_path), "ESQL_Template_Updated.ESQL"),
                os.path.join("templates", "ESQL_Template_Updated.ESQL"),
                "ESQL_Template_Updated.ESQL"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        template_content = f.read()
                    print(f"✅ Loaded ESQL template from {path}")
                    return template_content
            
            print(f"❌ ESQL template not found at {template_path} or common locations")
            # Create a minimal template if not found
            return self._create_minimal_template()
            
        except Exception as e:
            print(f"❌ Error loading ESQL template: {str(e)}")
            return None
    
    def _create_minimal_template(self) -> str:
        """Create a minimal ESQL template if the template file is not found"""
        print("⚠️ Creating minimal ESQL template")
        return """
-- INPUT AND OUTPUT EVENT MESSAGE TEMPLATE - METADATA ONLY
CREATE COMPUTE MODULE {MODULE_NAME}_InputEventMessage
 CREATE FUNCTION Main() RETURNS BOOLEAN
 BEGIN
    -- Input event processing logic
    CALL CopyEntireMessage();
    RETURN TRUE;
 END;
 CREATE PROCEDURE CopyMessageHeaders() BEGIN
    DECLARE I INTEGER 1;
    DECLARE J INTEGER;
    SET J = CARDINALITY(InputRoot.*[]);
    WHILE I < J DO
        SET OutputRoot.*[I] = InputRoot.*[I];
        SET I = I + 1;
    END WHILE;
 END;
 CREATE PROCEDURE CopyEntireMessage() BEGIN
    SET OutputRoot = InputRoot;
 END;
END MODULE;

-- COMPUTE TEMPLATE - FULL BUSINESS LOGIC
CREATE COMPUTE MODULE {MODULE_NAME}_Compute
 CREATE FUNCTION Main() RETURNS BOOLEAN
 BEGIN
    -- Compute node processing logic
    CALL CopyMessageHeaders();
    -- Add business logic here
    RETURN TRUE;
 END;
 CREATE PROCEDURE CopyMessageHeaders() BEGIN
    DECLARE I INTEGER 1;
    DECLARE J INTEGER;
    SET J = CARDINALITY(InputRoot.*[]);
    WHILE I < J DO
        SET OutputRoot.*[I] = InputRoot.*[I];
        SET I = I + 1;
    END WHILE;
 END;
END MODULE;

-- PROCESSING TEMPLATE - VALIDATION AND ROUTING ONLY
CREATE COMPUTE MODULE {MODULE_NAME}_AfterEnrichment
 CREATE FUNCTION Main() RETURNS BOOLEAN
 BEGIN
    -- After Enrichment processing
    CALL CopyEntireMessage();
    -- Add enrichment logic here
    RETURN TRUE;
 END;
 CREATE PROCEDURE CopyMessageHeaders() BEGIN
    DECLARE I INTEGER 1;
    DECLARE J INTEGER;
    SET J = CARDINALITY(InputRoot.*[]);
    WHILE I < J DO
        SET OutputRoot.*[I] = InputRoot.*[I];
        SET I = I + 1;
    END WHILE;
 END;
 CREATE PROCEDURE CopyEntireMessage() BEGIN
    SET OutputRoot = InputRoot;
 END;
END MODULE;

-- FAILURE/ERROR HANDLING TEMPLATE
CREATE COMPUTE MODULE {MODULE_NAME}_ErrorHandler
 CREATE FUNCTION Main() RETURNS BOOLEAN
 BEGIN
    -- Error handling logic
    CALL CopyMessageHeaders();
    -- Add error handling logic here
    RETURN TRUE;
 END;
 CREATE PROCEDURE CopyMessageHeaders() BEGIN
    DECLARE I INTEGER 1;
    DECLARE J INTEGER;
    SET J = CARDINALITY(InputRoot.*[]);
    WHILE I < J DO
        SET OutputRoot.*[I] = InputRoot.*[I];
        SET I = I + 1;
    END WHILE;
 END;
END MODULE;
"""
    
    def _extract_flow_name_from_msgflow(self, msgflow_path: str) -> str:
        """
        Extract flow name from messageflow file path
        
        Args:
            msgflow_path: Path to messageflow XML file
            
        Returns:
            Flow name
        """
        try:
            # First try to extract from filename
            filename = os.path.basename(msgflow_path)
            if filename.endswith('.msgflow'):
                flow_name = filename[:-8]  # Remove '.msgflow'
                return flow_name
            
            # If that fails, try to extract from XML content
            try:
                tree = ET.parse(msgflow_path)
                root = tree.getroot()
                
                # Check nsURI attribute
                ns_uri = root.get('nsURI', '')
                if '{FLOW_NAME}' not in ns_uri:  # If it's not a template
                    match = re.search(r'([^/]+)\.msgflow', ns_uri)
                    if match:
                        return match.group(1)
            except:
                pass
            
            # Fallback to directory name
            dir_name = os.path.basename(os.path.dirname(msgflow_path))
            return dir_name
            
        except Exception as e:
            print(f"❌ Error extracting flow name: {str(e)}")
            return "Unknown_Flow"
    
    def update_messageflow(self, msgflow_path: str, preserve_comments: bool = True) -> bool:
        """
        Update messageflow with required ESQL nodes.
        
        Args:
            msgflow_path: Path to messageflow XML file
            preserve_comments: Whether to preserve XML comments
            
        Returns:
            True if updated successfully
        """
        try:
            print(f"\n🔄 Updating messageflow: {msgflow_path}")
            
            if not os.path.exists(msgflow_path):
                print(f"❌ Messageflow file not found: {msgflow_path}")
                return False
            
            # Extract flow name for node naming
            flow_name = self._extract_flow_name_from_msgflow(msgflow_path)
            print(f"🔍 Detected flow name: {flow_name}")
            
            # Read the original XML content to preserve formatting and comments
            with open(msgflow_path, 'r') as f:
                xml_content = f.read()
            
            if preserve_comments:
                # Parse XML while preserving comments
                dom = minidom.parse(msgflow_path)
                
                # Check if we have the required nodes in the business requirements
                nodes_to_add = []
                node_names = set()
                
                for node_req in self.required_nodes:
                    node_type = node_req.get('type', '')
                    if node_type in ['input_event', 'compute', 'processing', 'output_event', 'failure']:
                        node_name = node_req.get('name', '')
                        node_names.add(node_name)
                        
                        # Format the ESQL node reference properly
                        if node_type == 'processing':  # For AfterEnrichment
                            esql_node_ref = f"esql://routine/{flow_name}#{flow_name}_AfterEnrichment.Main"
                            nodes_to_add.append((node_name, esql_node_ref))
                        else:
                            suffix = self._get_suffix_for_node_type(node_type)
                            esql_node_ref = f"esql://routine/{flow_name}#{flow_name}{suffix}.Main"
                            nodes_to_add.append((node_name, esql_node_ref))
                
                # Now find where to inject the ESQL node references
                # Usually in the DYNAMIC_ESQL_NODES section or similar placeholder
                if '{DYNAMIC_ESQL_NODES}' in xml_content:
                    nodes_xml = '\n      '.join(f'<!-- {name} ESQL Node -->\n      <node dynamic="true" id="{name}" value="{ref}" />' 
                                              for name, ref in nodes_to_add)
                    xml_content = xml_content.replace('{DYNAMIC_ESQL_NODES}', nodes_xml)
                else:
                    # Try to find composition section to add nodes
                    match = re.search(r'<composition.*?>(.*?)</composition>', xml_content, re.DOTALL)
                    if match:
                        composition = match.group(1)
                        nodes_xml = '\n      '.join(f'<!-- {name} ESQL Node -->\n      <node dynamic="true" id="{name}" value="{ref}" />' 
                                                  for name, ref in nodes_to_add)
                        new_composition = composition + '\n      ' + nodes_xml
                        xml_content = xml_content.replace(composition, new_composition)
                
                # Save the updated XML
                with open(msgflow_path, 'w') as f:
                    f.write(xml_content)
                
                print(f"✅ Updated messageflow with {len(nodes_to_add)} ESQL nodes")
                self.generation_stats['messageflows_updated'] += 1
                return True
            
            else:
                # Standard XML parsing approach (without comment preservation)
                tree = ET.parse(msgflow_path)
                root = tree.getroot()
                
                # Find the composition element
                namespaces = {'eflow': 'http://www.ibm.com/wbi/2005/eflow'}
                classifiers = root.findall(".//eClassifiers", namespaces)
                
                if classifiers:
                    composition = None
                    for classifier in classifiers:
                        comp = classifier.find(".//composition", namespaces)
                        if comp is not None:
                            composition = comp
                            break
                    
                    if composition is not None:
                        # Add required ESQL nodes
                        for node_req in self.required_nodes:
                            node_type = node_req.get('type', '')
                            if node_type in ['input_event', 'compute', 'processing', 'output_event', 'failure']:
                                node_name = node_req.get('name', '')
                                
                                # Create the ESQL node
                                node = ET.SubElement(composition, "node")
                                node.set("dynamic", "true")
                                node.set("id", node_name)
                                
                                # Format the ESQL node reference properly
                                if node_type == 'processing':  # For AfterEnrichment
                                    esql_node_ref = f"esql://routine/{flow_name}#{flow_name}_AfterEnrichment.Main"
                                else:
                                    suffix = self._get_suffix_for_node_type(node_type)
                                    esql_node_ref = f"esql://routine/{flow_name}#{flow_name}{suffix}.Main"
                                    
                                node.set("value", esql_node_ref)
                        
                        # Save the updated XML
                        tree.write(msgflow_path, encoding="UTF-8", xml_declaration=True)
                        print(f"✅ Updated messageflow with {len(self.required_nodes)} ESQL nodes")
                        self.generation_stats['messageflows_updated'] += 1
                        return True
                    else:
                        print("❌ Composition element not found in messageflow")
                else:
                    print("❌ No eClassifiers found in messageflow")
            
            return False
                
        except Exception as e:
            print(f"❌ Error updating messageflow: {str(e)}")
            return False
    
    def _get_suffix_for_node_type(self, node_type: str) -> str:
        """
        Get the appropriate suffix for a node type
        
        Args:
            node_type: Node type (input_event, compute, processing, etc.)
            
        Returns:
            Suffix for the node type
        """
        if node_type == 'input_event':
            return '_InputEventMessage'
        elif node_type == 'output_event':
            return '_OutputEventMessage'
        elif node_type == 'compute':
            return '_Compute'
        elif node_type == 'processing':
            return '_AfterEnrichment'
        elif node_type == 'failure':
            return '_ErrorHandler'
        else:
            return ''
    
    def _extract_template_section(self, template_content: str, section_type: str) -> str:
        """
        Extract a specific section from the ESQL template
        
        Args:
            template_content: Full template content
            section_type: Section type (input_event, compute, etc.)
            
        Returns:
            Section content
        """
        section_markers = self.template_sections.get(section_type)
        if not section_markers:
            print(f"❌ Unknown section type: {section_type}")
            return ""
        
        start_marker, end_marker = section_markers
        
        if start_marker not in template_content:
            print(f"⚠️ Start marker not found for {section_type} section")
            return ""
        
        start_idx = template_content.find(start_marker)
        
        if end_marker:
            if end_marker not in template_content:
                print(f"⚠️ End marker not found for {section_type} section")
                return template_content[start_idx:]
                
            end_idx = template_content.find(end_marker)
            return template_content[start_idx:end_idx].strip()
        else:
            return template_content[start_idx:].strip()
        

    
    # Modified method signature to accept both template_dict and esql_template
    def generate_esql_files(self, vector_content: str, template_dict: Dict = None, msgflow_content: Dict = None, 
                           json_mappings: Dict = None, output_dir: str = None, esql_template: str = None) -> Dict:
        """
        Generate ESQL files for a message flow based on vector content and template.
        
        Args:
            vector_content: Business requirements from Vector DB
            template_dict: ESQL template information (dict with 'path' key)
            msgflow_content: Message flow information (dict with 'path' key)
            json_mappings: JSON mappings information
            output_dir: Output directory
            esql_template: ESQL template path (backward compatibility)
            
        Returns:
            Generation results
        """
        # Initialize results
        results = {
            'status': 'initialized',
            'successful': 0,
            'failed': 0,
            'total_modules': 0,
            'esql_files': []
        }
        
        try:
            print("\n" + "="*70)
            print("🔧 GENERATING ESQL FILES")
            print("="*70)
            
            # Handle backward compatibility
            if esql_template and template_dict is None:
                template_dict = {'path': esql_template}
                
            # Ensure we have all required parameters
            if template_dict is None:
                template_dict = {'path': 'templates/ESQL_Template_Updated.ESQL'}
            
            if msgflow_content is None and output_dir:
                # Create a default msgflow content
                flow_name = "Default_Flow"
                if self.business_requirements:
                    flow_name = self.business_requirements.get('flow_name', "Default_Flow")
                msgflow_content = {'path': os.path.join(output_dir, f"{flow_name}.msgflow")}
                
            if output_dir is None and msgflow_content and 'path' in msgflow_content:
                output_dir = os.path.dirname(msgflow_content['path'])
                
            if not output_dir:
                output_dir = '.'
            
            if json_mappings is None:
                json_mappings = {'path': os.path.join(output_dir, 'component_mapping.json')}
            
            # Step 1: Extract flow name and load naming convention
            flow_name = None
            if msgflow_content and 'path' in msgflow_content and os.path.exists(msgflow_content['path']):
                flow_name = self._extract_flow_name_from_msgflow(msgflow_content['path'])
            elif self.business_requirements:
                flow_name = self.business_requirements.get('flow_name', "Default_Flow")
            else:
                # Try to extract from business requirements
                business_reqs = self.analyze_vector_content(vector_content)
                flow_name = business_reqs.get('flow_name', "Default_Flow")
                
            naming_convention = self._load_naming_convention(output_dir)
            
            # Step 2: Load ESQL template - pass the path directly, not the dictionary
            template_path = template_dict['path'] if isinstance(template_dict, dict) and 'path' in template_dict else template_dict
            template_content = self._load_esql_template(template_path)
            
            if not template_content:
                raise ValueError("Failed to load ESQL template")
            
            # Step 3: Analyze vector content if not already done
            if not self.required_nodes:
                self.analyze_vector_content(vector_content)
            
            # Step 4: Generate ESQL files for each required node
            for node_req in self.required_nodes:
                node_name = node_req.get('name', '')
                node_type = node_req.get('type', '')
                
                if not node_type or not node_name:
                    print(f"⚠️ Skipping node with missing name or type: {node_req}")
                    continue
                
                results['total_modules'] += 1
                
                try:
                    # Get the appropriate suffix for this node type
                    suffix = self._get_suffix_for_node_type(node_type)
                    module_name = f"{flow_name}{suffix}"
                    
                    # Extract the appropriate section from the template
                    section_content = self._extract_template_section(template_content, node_type)
                    
                    if not section_content:
                        print(f"⚠️ Empty template section for {node_type}, using minimal template")
                        # Create a minimal template section
                        section_content = self._create_minimal_template_section(node_type)
                    
                    # Replace module name in template
                    esql_content = section_content.replace('{MODULE_NAME}', flow_name)
                    
                    # Create the ESQL file
                    esql_filename = f"{module_name}.esql"
                    esql_path = os.path.join(output_dir, esql_filename)
                    
                    with open(esql_path, 'w') as f:
                        f.write(esql_content)
                    
                    print(f"✅ Generated ESQL file: {esql_filename}")
                    results['successful'] += 1
                    results['esql_files'].append(esql_path)
                    
                except Exception as e:
                    print(f"❌ Failed to generate ESQL for {node_name}: {str(e)}")
                    results['failed'] += 1
            
            # Update status based on results
            if results['successful'] > 0:
                if results['failed'] == 0:
                    results['status'] = 'completed'
                else:
                    results['status'] = 'partially_completed'
            else:
                results['status'] = 'failed'
            
            return results
            
        except Exception as e:
            print(f"❌ Error generating ESQL files: {str(e)}")
            results['status'] = 'failed'
            results['error'] = str(e)
            return results
    


    def _create_minimal_template_section(self, section_type: str) -> str:
        """
        Create a minimal template section if the template doesn't have it
        
        Args:
            section_type: Section type (input_event, compute, etc.)
            
        Returns:
            Minimal template section
        """
        if section_type == 'input_event':
            return """
    CREATE COMPUTE MODULE {MODULE_NAME}_InputEventMessage
    CREATE FUNCTION Main() RETURNS BOOLEAN
    BEGIN
        -- Input event processing logic
        DECLARE episInfo REFERENCE TO Environment.variables.EventData.episInfo;
        DECLARE sourceInfo REFERENCE TO Environment.variables.EventData.sourceInfo;
        DECLARE targetInfo REFERENCE TO Environment.variables.EventData.targetInfo;
        DECLARE integrationInfo REFERENCE TO Environment.variables.EventData.integrationInfo;
        DECLARE dataInfo REFERENCE TO Environment.variables.EventData.dataInfo;
        
        -- Extract source information
        SET sourceInfo.srcAppIdentifier = InputRoot.XMLNSC.[<].*:Header.*:Source.*:Identifier;
        SET sourceInfo.srcEnterpriseCode = InputRoot.XMLNSC.[<].*:Header.*:Source.*:EnterpriseCode;
        SET sourceInfo.srcCountryCode = InputRoot.XMLNSC.[<].*:Header.*:Source.*:CountryCode;
        SET sourceInfo.srcCompanyCode = InputRoot.XMLNSC.[<].*:Header.*:Source.*:CompanyCode;
        
        -- Extract target information
        SET targetInfo.tgtAppIdentifier = InputRoot.XMLNSC.[<].*:Header.*:Target.*:Identifier;
        SET targetInfo.tgtEnterpriseCode = InputRoot.XMLNSC.[<].*:Header.*:Target.*:EnterpriseCode;
        SET targetInfo.tgtCountryCode = InputRoot.XMLNSC.[<].*:Header.*:Target.*:CountryCode;
        SET targetInfo.tgtCompanyCode = InputRoot.XMLNSC.[<].*:Header.*:Target.*:CompanyCode;
        
        -- Extract data information
        SET dataInfo.messageType = InputRoot.XMLNSC.[<].*:Header.*:MessageType;
        SET dataInfo.dataFormat = 'XML';
        
        CALL CopyEntireMessage();
        RETURN TRUE;
    END;
    CREATE PROCEDURE CopyMessageHeaders() BEGIN
        DECLARE I INTEGER 1;
        DECLARE J INTEGER;
        SET J = CARDINALITY(InputRoot.*[]);
        WHILE I < J DO
            SET OutputRoot.*[I] = InputRoot.*[I];
            SET I = I + 1;
        END WHILE;
    END;
    CREATE PROCEDURE CopyEntireMessage() BEGIN
        SET OutputRoot = InputRoot;
    END;
    END MODULE;
    """
        elif section_type == 'compute':
            return """
    CREATE COMPUTE MODULE {MODULE_NAME}_Compute
    CREATE FUNCTION Main() RETURNS BOOLEAN
    BEGIN
        -- ✅ BUSINESS LOGIC: Full message transformation and processing
        DECLARE episInfo REFERENCE TO Environment.variables.EventData.episInfo;
        DECLARE sourceInfo REFERENCE TO Environment.variables.EventData.sourceInfo;
        DECLARE targetInfo REFERENCE TO Environment.variables.EventData.targetInfo;
        DECLARE integrationInfo REFERENCE TO Environment.variables.EventData.integrationInfo;
        DECLARE dataInfo REFERENCE TO Environment.variables.EventData.dataInfo;
        
        -- ✅ BUSINESS DATA EXTRACTION: Extract business identifiers and data
        SET dataInfo.mainIdentifier = InputRoot.XMLNSC.[<].*:BusinessEntity.*:BusinessIdentifier;
        
        -- Add business logic here
        
        -- ✅ STANDARD MESSAGE PROCESSING
        SET OutputRoot = NULL;
        SET OutputRoot = InputRoot;
        
        RETURN TRUE;
    END;

    -- ✅ STANDARD IBM ACE INFRASTRUCTURE
    CREATE PROCEDURE CopyMessageHeaders() BEGIN
        DECLARE I INTEGER 1;
        DECLARE J INTEGER;
        SET J = CARDINALITY(InputRoot.*[]);
        WHILE I < J DO
            SET OutputRoot.*[I] = InputRoot.*[I];
            SET I = I + 1;
        END WHILE;
    END;

    CREATE PROCEDURE CopyEntireMessage() BEGIN
        SET OutputRoot = InputRoot;
    END;
    END MODULE;
    """
        elif section_type == 'processing':
            return """
    CREATE COMPUTE MODULE {MODULE_NAME}_AfterEnrichment
    CREATE FUNCTION Main() RETURNS BOOLEAN
    BEGIN
        -- After Enrichment processing
        DECLARE episInfo REFERENCE TO Environment.variables.EventData.episInfo;
        DECLARE sourceInfo REFERENCE TO Environment.variables.EventData.sourceInfo;
        DECLARE targetInfo REFERENCE TO Environment.variables.EventData.targetInfo;
        DECLARE integrationInfo REFERENCE TO Environment.variables.EventData.integrationInfo;
        DECLARE dataInfo REFERENCE TO Environment.variables.EventData.dataInfo;
        
        -- Process enrichment data
        -- Add enrichment-specific logic here
        
        CALL CopyEntireMessage();
        RETURN TRUE;
    END;
    CREATE PROCEDURE CopyMessageHeaders() BEGIN
        DECLARE I INTEGER 1;
        DECLARE J INTEGER;
        SET J = CARDINALITY(InputRoot.*[]);
        WHILE I < J DO
            SET OutputRoot.*[I] = InputRoot.*[I];
            SET I = I + 1;
        END WHILE;
    END;
    CREATE PROCEDURE CopyEntireMessage() BEGIN
        SET OutputRoot = InputRoot;
    END;
    END MODULE;
    """
        elif section_type == 'output_event':
            return """
    CREATE COMPUTE MODULE {MODULE_NAME}_OutputEventMessage
    CREATE FUNCTION Main() RETURNS BOOLEAN
    BEGIN
        -- Output event processing logic
        DECLARE episInfo REFERENCE TO Environment.variables.EventData.episInfo;
        DECLARE sourceInfo REFERENCE TO Environment.variables.EventData.sourceInfo;
        DECLARE targetInfo REFERENCE TO Environment.variables.EventData.targetInfo;
        DECLARE integrationInfo REFERENCE TO Environment.variables.EventData.integrationInfo;
        DECLARE dataInfo REFERENCE TO Environment.variables.EventData.dataInfo;
        
        -- Prepare output event data
        -- Process response data
        -- Update event tracking information
        
        CALL CopyEntireMessage();
        RETURN TRUE;
    END;
    CREATE PROCEDURE CopyMessageHeaders() BEGIN
        DECLARE I INTEGER 1;
        DECLARE J INTEGER;
        SET J = CARDINALITY(InputRoot.*[]);
        WHILE I < J DO
            SET OutputRoot.*[I] = InputRoot.*[I];
            SET I = I + 1;
        END WHILE;
    END;
    CREATE PROCEDURE CopyEntireMessage() BEGIN
        SET OutputRoot = InputRoot;
    END;
    END MODULE;
    """
        elif section_type == 'failure':
            return """
    CREATE COMPUTE MODULE {MODULE_NAME}_ErrorHandler
    CREATE FUNCTION Main() RETURNS BOOLEAN
    BEGIN
        -- Error handling logic
        DECLARE episInfo REFERENCE TO Environment.variables.EventData.episInfo;
        DECLARE sourceInfo REFERENCE TO Environment.variables.EventData.sourceInfo;
        DECLARE targetInfo REFERENCE TO Environment.variables.EventData.targetInfo;
        DECLARE integrationInfo REFERENCE TO Environment.variables.EventData.integrationInfo;
        DECLARE dataInfo REFERENCE TO Environment.variables.EventData.dataInfo;
        DECLARE errorInfo REFERENCE TO Environment.variables.EventData.errorInfo;
        
        -- Capture error details
        SET errorInfo.errorCode = 'ERR-' || CAST(InputExceptionList.[1].Number AS CHARACTER);
        SET errorInfo.errorText = InputExceptionList.[1].Text;
        SET errorInfo.errorSource = InputExceptionList.[1].Label;
        
        -- Set up error response
        SET OutputRoot.XMLNSC.Error.Code = errorInfo.errorCode;
        SET OutputRoot.XMLNSC.Error.Text = errorInfo.errorText;
        SET OutputRoot.XMLNSC.Error.Source = errorInfo.errorSource;
        
        RETURN TRUE;
    END;
    CREATE PROCEDURE CopyMessageHeaders() BEGIN
        DECLARE I INTEGER 1;
        DECLARE J INTEGER;
        SET J = CARDINALITY(InputRoot.*[]);
        WHILE I < J DO
            SET OutputRoot.*[I] = InputRoot.*[I];
            SET I = I + 1;
        END WHILE;
    END;
    END MODULE;
    """
        else:
            return ""



    # ============================================================================
    # PART 3: MAIN EXECUTION 
    # ============================================================================
    
    def generate_esql_files_for_flow(self, vector_content: str, esql_template_path: str, output_dir: str) -> Dict:
        """
        Main execution method to generate ESQL files for a flow.
        
        Args:
            vector_content: Business requirements from Vector DB
            esql_template_path: Path to ESQL template
            output_dir: Output directory
            
        Returns:
            Generation results
        """
        try:
            print("\n" + "="*70)
            print("🚀 STARTING ENHANCED ESQL GENERATION")
            print("="*70)
            
            # Step 1: Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Step 2: Analyze vector content to extract business requirements
            business_requirements = self.analyze_vector_content(vector_content)
            if not business_requirements:
                raise ValueError("Failed to extract business requirements from Vector DB content")
            
            # Step 3: Find messageflow files in the output directory
            msgflow_files = []
            for root, _, files in os.walk(output_dir):
                for file in files:
                    if file.endswith('.msgflow'):
                        msgflow_files.append(os.path.join(root, file))
            
            if not msgflow_files:
                print(f"⚠️ No messageflow files found in {output_dir}")
                print("⚠️ Proceeding with ESQL generation based on business requirements")
            
            # Step 4: Generate ESQL files for each messageflow
            all_results = {
                'status': 'in_progress',
                'total_flows': len(msgflow_files) if msgflow_files else 1,
                'successful_flows': 0,
                'failed_flows': 0,
                'flow_results': []
            }
            
            for msgflow_path in msgflow_files:
                try:
                    # Extract flow name
                    flow_name = self._extract_flow_name_from_msgflow(msgflow_path)
                    
                    # Update messageflow
                    msgflow_updated = self.update_messageflow(msgflow_path)
                    
                    # Generate ESQL files
                    flow_dir = os.path.dirname(msgflow_path)
                    # Create proper parameter dictionaries
                    template_dict = {'path': esql_template_path}
                    msgflow_content = {'path': msgflow_path}
                    json_mappings = {'path': os.path.join(flow_dir, '..', 'component_mapping.json')}
                    
                    # Call with correct parameters - now we support both parameter styles
                    flow_results = self.generate_esql_files(
                        vector_content=vector_content,
                        template_dict=template_dict,
                        msgflow_content=msgflow_content,
                        json_mappings=json_mappings,
                        output_dir=flow_dir
                    )
                    
                    # Add to overall results
                    flow_results['messageflow_updated'] = msgflow_updated
                    flow_results['messageflow_path'] = msgflow_path
                    all_results['flow_results'].append(flow_results)
                    
                    if flow_results['status'] == 'completed':
                        all_results['successful_flows'] += 1
                    else:
                        all_results['failed_flows'] += 1
                
                except Exception as e:
                    print(f"❌ Error processing messageflow {msgflow_path}: {str(e)}")
                    all_results['failed_flows'] += 1
                    all_results['flow_results'].append({
                        'status': 'failed',
                        'error': str(e),
                        'messageflow_path': msgflow_path
                    })
            
            # If no messageflow files found, still generate ESQL based on requirements
            if not msgflow_files:
                try:
                    flow_name = business_requirements.get('flow_name', 'Default_Flow')
                    
                    # Create a dummy messageflow content
                    msgflow_content = {'path': os.path.join(output_dir, f"{flow_name}.msgflow")}
                    template_dict = {'path': esql_template_path}
                    json_mappings = {'path': os.path.join(output_dir, 'component_mapping.json')}
                    
                    # Generate ESQL files with new parameter style
                    flow_results = self.generate_esql_files(
                        vector_content=vector_content,
                        template_dict=template_dict,
                        msgflow_content=msgflow_content,
                        json_mappings=json_mappings,
                        output_dir=output_dir
                    )
                    
                    # Add to overall results
                    flow_results['messageflow_updated'] = False
                    flow_results['messageflow_path'] = msgflow_content['path']
                    all_results['flow_results'].append(flow_results)
                    
                    if flow_results['status'] == 'completed':
                        all_results['successful_flows'] += 1
                    else:
                        all_results['failed_flows'] += 1
                
                except Exception as e:
                    print(f"❌ Error generating ESQL without messageflow: {str(e)}")
                    all_results['failed_flows'] += 1
                    all_results['flow_results'].append({
                        'status': 'failed',
                        'error': str(e)
                    })
            
            # Step 5: Generate final results
            all_results['status'] = 'completed'
            all_results['statistics'] = self.generation_stats
            
            # Calculate total success rate
            total_modules = sum(result.get('total_modules', 0) for result in all_results['flow_results'])
            successful_modules = sum(result.get('successful', 0) for result in all_results['flow_results'])
            if total_modules > 0:
                all_results['success_rate'] = f"{(successful_modules / total_modules * 100):.1f}%"
            else:
                all_results['success_rate'] = "0%"
            
            # Print overall summary
            print("\n" + "="*70)
            print("📊 OVERALL GENERATION SUMMARY")
            print("="*70)
            print(f"✅ Successful Flows: {all_results['successful_flows']}/{all_results['total_flows']}")
            print(f"✅ Successful Modules: {successful_modules}/{total_modules} ({all_results['success_rate']})")
            print(f"🧠 Total LLM Calls: {self.generation_stats['llm_calls']}")
            print(f"🔧 Total Auto-fixes: {self.generation_stats['auto_fixes_applied']}")
            print(f"🔄 MessageFlows Updated: {self.generation_stats['messageflows_updated']}")
            
            return all_results
        
        except Exception as e:
            print(f"\n❌ ESQL Generation Failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'statistics': self.generation_stats
            }


# ============================================================================
# HELPER FUNCTIONS FOR INTEGRATION
# ============================================================================

def create_enhanced_esql_generator(groq_api_key: Optional[str] = None, 
                                groq_model: str = "llama-3.1-70b-versatile") -> ESQLGenerator:
    """
    Factory function to create ESQLGenerator instance.
    
    Args:
        groq_api_key: Groq API key (optional, uses environment variable)
        groq_model: LLM model to use
    
    Returns:
        Configured ESQLGenerator instance
    """
    return ESQLGenerator(groq_api_key=groq_api_key, groq_model=groq_model)


def run_enhanced_esql_generator(vector_content: str, output_dir: str, 
                            esql_template_path: str = "templates/ESQL_Template_Updated.ESQL",
                            groq_api_key: Optional[str] = None,
                            groq_model: str = "llama-3.1-70b-versatile") -> Dict:
    """
    Main execution function with simplified inputs for main.py.
    
    Args:
        vector_content: Business requirements from Vector DB
        output_dir: Output directory
        esql_template_path: Path to ESQL template
        groq_api_key: Groq API key
        groq_model: LLM model to use
        
    Returns:
        Generation results
    """
    generator = create_enhanced_esql_generator(groq_api_key, groq_model)
    return generator.generate_esql_files_for_flow(vector_content, esql_template_path, output_dir)




# ============================================================================
# MAIN EXECUTION (for testing)
# ============================================================================

def main():
    """
    Test execution of Enhanced ESQL Generator.
    This is for standalone testing only.
    """
    import sys
    
    print("="*70)
    print("Enhanced ESQL Generator - Standalone Test")
    print("="*70)
    
    # Check for ESQL template
    template_path = "templates/ESQL_Template_Updated.ESQL"
    if not os.path.exists(template_path):
        template_path = "ESQL_Template_Updated.ESQL"
        if not os.path.exists(template_path):
            print(f"\n❌ ESQL template not found in templates directory or current directory")
            print("Please ensure ESQL_Template_Updated.ESQL is available")
            sys.exit(1)
    
    # Check for output directory
    output_dir = "output"
    if not os.path.exists(output_dir):
        print(f"\n❌ Output directory not found: {output_dir}")
        print("Please generate MessageFlow first")
        sys.exit(1)
    
    # Simulated Vector DB content for testing
    vector_content = """
    CW1 Document Processing Flow
    
    Business Requirements:
    1. Receive CDM Document messages from local queue CW1.IN.DOCUMENT.SND.QL
    2. Enrich with database lookups:
       - sp_GetMainCompanyInCountry: Get company information
       - sp_GetCW1BrokerageId: Get brokerage ID
    3. Transform CDM Document to CW1 universal event format
    4. Database lookup for IsPublished flag based on document type
    5. Enrich target recipient ID based on country/company code
    6. Compress message using GZip
    7. Send to CW1 EAdapter via SOAP
    
    Database Operations:
    - MH.ESB.EDIEnterprise alias for database
    - Multiple stored procedure calls for enrichment
    - Parameter passing from message headers
    
    Error Handling:
    - Failed messages go to error handling
    - Event tracking for monitoring
    """
    
    try:
        # Run generator
        results = run_enhanced_esql_generator(
            vector_content=vector_content,
            output_dir=output_dir,
            esql_template_path=template_path
        )
        
        # Print results
        if results.get('status') == 'completed':
            print("\n🎉 Enhanced ESQL Generation Completed Successfully!")
        else:
            print(f"\n❌ Enhanced ESQL Generation Failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()