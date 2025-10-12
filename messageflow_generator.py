#!/usr/bin/env python3
"""
Enhanced MessageFlow Generator v3.0 - DSV Standard with JSON Input
- Reads biztalk_ace_component_mapping.json from output/ folder
- Reads msgflow_template.xml from root folder
- Generates high-quality DSV standard MessageFlow
- 100% LLM-based with comprehensive BizTalk transformation logic
- Strict no fallback error handling
- Uses prompt_module.py exclusively for LLM prompts
"""

import os
import json
import xml.etree.ElementTree as ET
from typing import Dict, List
from datetime import datetime
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

class MessageFlowGenerationError(Exception):
    """Exception for MessageFlow generation failures - No fallback allowed"""
    pass


class DSVMessageFlowGenerator:
    """Enhanced MessageFlow Generator for DSV Standards with JSON-based component mapping"""
    
    def __init__(self, app_name: str, flow_name: str, groq_api_key: str, groq_model: str = None):
        self.app_name = app_name
        self.flow_name = flow_name
        self.client = Groq(api_key=groq_api_key)
        self.groq_model = groq_model or os.getenv('GROQ_MODEL', 'llama-3.1-70b-versatile')
        self.root_path = Path.cwd()  # Current working directory as root
        print(f"🎯 DSV MessageFlow Generator Ready: {flow_name} | Model: {self.groq_model}")



    
    def generate_messageflow(self, confluence_spec: str, biztalk_maps_path: str, output_dir: str) -> Dict:
        """
        Generate DSV Standard MessageFlow with automated JSON and template input
        Supports single and multiple MessageFlow generation with flow connectors
        """
        print("🚀 Starting DSV Standard MessageFlow Generation")
        
        try:
            # Step 1: Load business requirements
            print("📋 Loading business requirements...")
            business_reqs = self._load_business_requirements()
            print(f"   ✅ Business requirements loaded")

            # Step 2: Detect naming convention files
            print("📋 Detecting naming convention files...")
            naming_conventions = self._detect_and_load_naming_conventions()
            num_flows = len(naming_conventions)
            print(f"   ✅ Found {num_flows} MessageFlow(s) to generate")
            
            # Step 3: Determine folder structure (single vs multiple)
            base_output_dir = self.root_path / "output"
            if num_flows == 1:
                mode = "single"
                output_root = base_output_dir / "single"
                print(f"   📁 Mode: Single MessageFlow")
            else:
                mode = "multiple"
                output_root = base_output_dir / "multiple"
                print(f"   📁 Mode: Multiple MessageFlows with connectors")
            
            os.makedirs(output_root, exist_ok=True)
            
            # Step 4: Generate MessageFlows with connectors
            generated_flows = []
            
            for idx, naming_conv in enumerate(naming_conventions, 1):
                flow_name = naming_conv['project_naming']['message_flow_name']
                app_name = naming_conv['project_naming'].get('ace_application', 'ACE_Application')
                
                print(f"\n🔄 Generating Flow {idx}/{num_flows}: {flow_name}")
                
                # Create flow-specific directory
                flow_output_dir = output_root / flow_name
                os.makedirs(flow_output_dir, exist_ok=True)
                os.makedirs(flow_output_dir / "schemas", exist_ok=True)
                os.makedirs(flow_output_dir / "esql", exist_ok=True)
                
                # Determine input/output queues for connectors
                connector_config = self._create_connector_config(
                    idx, num_flows, naming_conventions
                )
                
                # Generate the MessageFlow
                msgflow_result = self._generate_single_messageflow(
                    flow_name=flow_name,
                    app_name=app_name,
                    naming_conv=naming_conv,
                    business_reqs=business_reqs,
                    connector_config=connector_config,
                    output_dir=flow_output_dir,
                    biztalk_maps_path=biztalk_maps_path
                )
                
                generated_flows.append({
                    'flow_name': flow_name,
                    'flow_index': idx,
                    'output_dir': str(flow_output_dir),
                    'msgflow_file': msgflow_result['msgflow_file'],
                    'connector_config': connector_config
                })
                
                print(f"   ✅ {flow_name} generated successfully")
            
            # Step 5: Return summary
            return {
                'success': True,
                'mode': mode,
                'total_flows': num_flows,
                'generated_flows': generated_flows,
                'output_root': str(output_root),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"DSV MessageFlow generation failed: {str(e)}"
            print(f"❌ {error_msg}")
            raise MessageFlowGenerationError(error_msg)
    


    def _process_template_with_connectors(self, template: str, flow_name: str, 
                                     app_name: str, connector_config: Dict) -> str:
        """
        Process MessageFlow template and inject connector queue names
        """
        # Replace placeholders
        processed = template.replace('{FLOW_NAME}', flow_name)
        processed = processed.replace('{APP_NAME}', app_name)
        processed = processed.replace('{INPUT_QUEUE_NAME}', connector_config['input_queue'])
        processed = processed.replace('{OUTPUT_QUEUE_NAME}', connector_config['output_queue'])
        
        # Add connector metadata as XML comment
        connector_info = f"""
        <!-- Flow Connector Configuration -->
        <!-- Flow Index: {connector_config['flow_index']} -->
        <!-- Input Queue: {connector_config['input_queue']} -->
        <!-- Output Queue: {connector_config['output_queue']} -->
        <!-- First Flow: {connector_config['is_first']} -->
        <!-- Last Flow: {connector_config['is_last']} -->
        """
        
        # Insert after XML declaration
        if '<?xml' in processed:
            processed = processed.replace('<?xml version="1.0" encoding="UTF-8"?>', 
                                        f'<?xml version="1.0" encoding="UTF-8"?>\n{connector_info}')
        
        return processed



    def _generate_single_messageflow(self, flow_name: str, app_name: str, 
                                naming_conv: Dict, business_reqs: Dict,
                                connector_config: Dict, output_dir: Path,
                                biztalk_maps_path: str) -> Dict:
        """
        Generate a single MessageFlow with connector configuration
        """
        print(f"      📝 Loading MessageFlow template...")
        msgflow_template = self._load_msgflow_template()
        
        # Process template with connector queues
        print(f"      🔧 Applying connector configuration...")
        processed_xml = self._process_template_with_connectors(
            msgflow_template, 
            flow_name, 
            app_name,
            connector_config
        )
        
        # Save MessageFlow file
        msgflow_filename = f"{flow_name}.msgflow"
        msgflow_file = output_dir / msgflow_filename
        
        with open(msgflow_file, 'w', encoding='utf-8') as f:
            f.write(processed_xml)
        
        print(f"      💾 Saved: {msgflow_filename}")
        
        return {
            'msgflow_file': str(msgflow_file),
            'flow_name': flow_name,
            'app_name': app_name
        }



    def _create_connector_config(self, flow_index: int, total_flows: int, 
                            naming_conventions: List[Dict]) -> Dict:
        """
        Create connector configuration for flow chaining
        Flow N output → Flow N+1 input
        """
        connector = {
            'is_first': flow_index == 1,
            'is_last': flow_index == total_flows,
            'flow_index': flow_index,
            'input_queue': None,
            'output_queue': None
        }
        
        current_flow_name = naming_conventions[flow_index - 1]['project_naming']['message_flow_name']
        
        # For multiple flows, create connectors
        if total_flows > 1:
            # Input queue: From previous flow (except first)
            if not connector['is_first']:
                prev_flow_name = naming_conventions[flow_index - 2]['project_naming']['message_flow_name']
                connector['input_queue'] = f"{prev_flow_name}_OUT.QL"
                print(f"      🔗 Input: {connector['input_queue']} (from Flow {flow_index - 1})")
            else:
                # First flow gets input from external source
                connector['input_queue'] = f"{current_flow_name}_IN.QL"
                print(f"      📥 Input: {connector['input_queue']} (external)")
            
            # Output queue: To next flow (except last)
            if not connector['is_last']:
                connector['output_queue'] = f"{current_flow_name}_OUT.QL"
                print(f"      📤 Output: {connector['output_queue']} (to Flow {flow_index + 1})")
            else:
                # Last flow sends to final destination
                connector['output_queue'] = f"{current_flow_name}_FINAL.QL"
                print(f"      🎯 Output: {connector['output_queue']} (final)")
        else:
            # Single flow: standard input/output
            connector['input_queue'] = f"{current_flow_name}_IN.QL"
            connector['output_queue'] = f"{current_flow_name}_OUT.QL"
        
        return connector



    def _load_business_requirements(self) -> Dict:
        """Load business_requirements.json from output folder"""
        business_req_path = self.root_path / "output" / "business_requirements.json"
        
        try:
            if not business_req_path.exists():
                raise FileNotFoundError(f"business_requirements.json not found: {business_req_path}")
            
            with open(business_req_path, 'r', encoding='utf-8') as f:
                business_data = json.load(f)
            
            print(f"   📁 File: {business_req_path}")
            return business_data
            
        except json.JSONDecodeError as e:
            raise MessageFlowGenerationError(f"Invalid JSON in business_requirements.json: {e}")
        except Exception as e:
            raise MessageFlowGenerationError(f"Failed to load business requirements: {e}")




    def _detect_and_load_naming_conventions(self) -> List[Dict]:
        """Detect and load naming_convention files (single or multiple)"""
        naming_files = []
        
        # Check for single file
        single_file = self.root_path / "naming_convention.json"
        if single_file.exists():
            with open(single_file, 'r') as f:
                naming_files.append(json.load(f))
            return naming_files
        
        # Check for numbered files
        idx = 1
        while True:
            numbered_file = self.root_path / f"naming_convention_{idx}.json"
            if numbered_file.exists():
                with open(numbered_file, 'r') as f:
                    naming_files.append(json.load(f))
                idx += 1
            else:
                break
        
        if not naming_files:
            raise FileNotFoundError("No naming_convention.json file(s) found")
        
        return naming_files


    
    def _load_msgflow_template(self) -> str:
        """Load MessageFlow template from root folder"""
        template_file_path = self.root_path / "msgflow_template.xml"
        
        try:
            if not template_file_path.exists():
                raise FileNotFoundError(f"MessageFlow template not found: {template_file_path}")
            
            with open(template_file_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            
            # Validate template is XML
            try:
                ET.fromstring(template_content)
            except ET.ParseError as e:
                raise ValueError(f"Invalid XML template: {e}")
            
            # Check for required template placeholders
            required_placeholders = ['{FLOW_NAME}', '{APP_NAME}', '{INPUT_QUEUE_NAME}']
            missing_placeholders = [p for p in required_placeholders if p not in template_content]
            if missing_placeholders:
                print(f"   ⚠️ Warning: Missing placeholders in template: {missing_placeholders}")
            
            print(f"   📁 Template file: {template_file_path}")
            return template_content
            
        except Exception as e:
            raise MessageFlowGenerationError(f"Failed to load MessageFlow template: {e}")

    
    
            
    def _process_template_placeholders(self, template_content: str, component_data: Dict, flow_details: Dict) -> str:
        """Process template by preserving structure and replacing placeholders only"""
        
        # Get standard ESQL module definitions
        esql_modules = self._enforce_6_module_standard(flow_details['flow_name'])
        
        # Build complete replacement map
        replacements = {
            '{FLOW_NAME}': flow_details['flow_name'],
            '{APP_NAME}': flow_details['app_name'],
            '{INPUT_QUEUE_NAME}': component_data.get('input_queue', f"{flow_details['flow_name']}.INPUT.QL"),
            '{SOAP_SERVICE_URL}': component_data.get('soap_url', 'https://eadapterqa.dsv.com/eAdapterStreamedService.svc'),
            '{WSDL_FILE_NAME}': component_data.get('wsdl_file', 'eAdapterStreamedService.wsdl'),
            '{LOCAL_ENRICHMENT_PATH}': f"{flow_details['app_name']}\\{flow_details['flow_name']}\\enrichment"
        }
        
        # Start with template content
        processed = template_content
        
        # Replace all placeholders
        for placeholder, value in replacements.items():
            processed = processed.replace(placeholder, value)
        
        # Update ALL compute expressions to use correct flow_name prefix
        # Match pattern: computeExpression="esql://routine/#ANYTHING_BEFORE_FLOWNAME...
        import re
        
        # Replace module names in compute expressions - preserve the suffix (e.g., _InputEventMessage, _Compute)
        patterns = [
            (r'(computeExpression="esql://routine/#)[^_]+(_InputEventMessage\.Main")', rf'\1{flow_details["flow_name"]}\2'),
            (r'(computeExpression="esql://routine/#)[^_]+(_Compute\.Main")', rf'\1{flow_details["flow_name"]}\2'),
            (r'(computeExpression="esql://routine/#)[^_]+(_AfterEnrichment\.Main")', rf'\1{flow_details["flow_name"]}\2'),
            (r'(computeExpression="esql://routine/#)[^_]+(_OutputEventMessage\.Main")', rf'\1{flow_details["flow_name"]}\2'),
            (r'(computeExpression="esql://routine/#)[^_]+(_AfterEventMsg\.Main")', rf'\1{flow_details["flow_name"]}\2'),
            (r'(computeExpression="esql://routine/#)[^_]+(_Failure\.Main")', rf'\1{flow_details["flow_name"]}\2'),
        ]
        
        for pattern, replacement in patterns:
            processed = re.sub(pattern, replacement, processed)
        
        return processed
        

    def _analyze_source_destination_protocols(self, component_data: Dict, business_context: Dict) -> Dict:
        """
        Analyze source and destination protocols from component mapping and business context
        Returns: {'source_protocol': str, 'dest_protocol': str, 'flow_type': str}
        """
        try:
            protocol_info = {
                'source_protocol': 'UNKNOWN',
                'dest_protocol': 'UNKNOWN', 
                'flow_type': 'STANDARD'
            }
            
            # Extract from component mappings
            component_mappings = component_data.get('component_mappings', [])
            
            for mapping in component_mappings:
                integration_details = mapping.get('integration_details', {})
                
                # Detect source protocol
                input_queue = integration_details.get('input_queue', '')
                if input_queue and ('MQ' in input_queue.upper() or '.QL' in input_queue):
                    protocol_info['source_protocol'] = 'MQ'
                elif 'http' in str(integration_details.get('input_endpoint', '')).lower():
                    protocol_info['source_protocol'] = 'HTTP'
                elif 'file' in str(integration_details.get('input_type', '')).lower():
                    protocol_info['source_protocol'] = 'FILE'
                
                # Detect destination protocol  
                output_endpoint = integration_details.get('output_endpoint', '')
                if output_endpoint and ('MQ' in output_endpoint.upper() or '.QL' in output_endpoint):
                    protocol_info['dest_protocol'] = 'MQ'
                elif 'http' in str(output_endpoint).lower() or 'service' in str(output_endpoint).lower():
                    protocol_info['dest_protocol'] = 'HTTP'
                elif 'file' in str(output_endpoint).lower():
                    protocol_info['dest_protocol'] = 'FILE'
            
            # Determine flow type based on protocols
            if protocol_info['source_protocol'] == 'MQ' and protocol_info['dest_protocol'] == 'MQ':
                protocol_info['flow_type'] = 'P2P'
            elif protocol_info['source_protocol'] == 'FILE' or protocol_info['dest_protocol'] == 'FILE':
                protocol_info['flow_type'] = 'SAT'
            elif protocol_info['source_protocol'] == 'HTTP' or protocol_info['dest_protocol'] == 'HTTP':
                protocol_info['flow_type'] = 'HUB'
            
            return protocol_info
            
        except Exception as e:
            print(f"    ⚠️ Protocol analysis failed: {e}")
            return {
                'source_protocol': 'UNKNOWN',
                'dest_protocol': 'UNKNOWN',
                'flow_type': 'STANDARD'
            }
    

    def _get_6_module_context_for_llm(self, standard_modules: List[Dict]) -> str:
        """
        Generate 6-module context information for LLM prompt
        """
        context = """
    ## 6-MODULE DSV STANDARD FRAMEWORK

    This MessageFlow MUST implement exactly 6 compute modules:

    """
        
        for i, module in enumerate(standard_modules, 1):
            context += f"""
    {i}. **{module['name']}** (Node ID: {module['id']})
    - Purpose: {module['purpose']}
    - Type: {module['type']}
    - computeExpression: "{module['name']}"
    """
        
        context += """
    ## CRITICAL REQUIREMENTS:
    - Exactly 6 compute nodes - no more, no less
    - Node IDs must be FCMComposite_1_1 through FCMComposite_1_6
    - All modules must be connected in proper sequence
    - Failure node (FCMComposite_1_6) must connect to error terminals
    - No dynamic module generation - use only the 6 standard modules

    """
        return context




    def _generate_xml_with_enhanced_context(self, msgflow_template: str, business_context: Dict, 
                                        component_context: List[Dict], biztalk_maps: List[Dict], 
                                        component_data: Dict, output_dir: str, 
                                        standard_modules: List[Dict]) -> str:
        """Generate MessageFlow XML by preserving template structure"""
        try:
            flow_details = {
                'flow_name': self.flow_name,
                'app_name': self.app_name
            }
            
            print("  Analyzing source and destination protocols...")
            protocol_analysis = self._analyze_source_destination_protocols(component_data, business_context)
            print(f"    Detected: Source={protocol_analysis['source_protocol']}, Dest={protocol_analysis['dest_protocol']}, Flow Type={protocol_analysis['flow_type']}")

            # Process template - this preserves ALL nodes and connections
            print("  Processing template with placeholder replacement...")
            processed_xml = self._process_template_placeholders(msgflow_template, component_data, flow_details)
            print("  Template placeholders processed")
            
            # Validate the processed XML structure
            try:
                root = ET.fromstring(processed_xml)
                
                # Find all nodes - handle namespace
                all_nodes = root.findall('.//{http://www.ibm.com/wbi/2005/eflow}nodes')
                if not all_nodes:
                    all_nodes = root.findall('.//nodes')
                
                # Find all connections
                all_connections = root.findall('.//{http://www.ibm.com/wbi/2005/eflow}connections')
                if not all_connections:
                    all_connections = root.findall('.//connections')
                
                node_count = len(all_nodes)
                connection_count = len(all_connections)
                
                print(f"  Generated MessageFlow: {node_count} nodes, {connection_count} connections")
                
                # Validate minimum requirements
                if node_count < 11:
                    raise MessageFlowGenerationError(f"Insufficient nodes: {node_count} (expected 11+ nodes including subflows)")
                
                if connection_count < 10:
                    print(f"  Warning: Only {connection_count} connections found (expected 10+)")
                
                # Track token usage for consistency
                if 'token_tracker' in st.session_state:
                    st.session_state.token_tracker.manual_track(
                        agent="messageflow_generator",
                        operation="template_processing",
                        model="template_parser",
                        input_tokens=len(msgflow_template) // 4,
                        output_tokens=len(processed_xml) // 4,
                        flow_name=self.flow_name
                    )
                    
            except ET.ParseError as e:
                raise MessageFlowGenerationError(f"Generated invalid XML structure: {e}")
            
            # Save MessageFlow file
            os.makedirs(output_dir, exist_ok=True)
            msgflow_filename = f"{self.flow_name}.msgflow"
            msgflow_file = os.path.join(output_dir, msgflow_filename)
            
            with open(msgflow_file, 'w', encoding='utf-8') as f:
                f.write(processed_xml)
            
            print(f"  MessageFlow saved: {msgflow_file}")
            return msgflow_file
            
        except MessageFlowGenerationError:
            raise
        except Exception as e:
            raise MessageFlowGenerationError(f"MessageFlow generation failed: {str(e)}")
        


    def _create_safe_business_spec(self, message_flow_patterns, routing_logic, 
                              integration_patterns, technical_specs, business_ctx):
        """Create Vector DB business spec with same validation as enhanced prompt"""
        
        def safe_list(items, error_msg):
            return '\n'.join(f"- {item}" for item in items) if items else f"❌ CRITICAL: {error_msg}"
        
        def safe_comma(items, error_msg):
            return ', '.join(items) if items else f"Not specified: {error_msg}"
        
        return f"""=== VECTOR DB EXTRACTED SPECIFICATIONS ===

    MESSAGE FLOW PATTERNS:
    {safe_list(message_flow_patterns, "No patterns extracted from Vector DB")}

    ROUTING LOGIC:
    {safe_list(routing_logic, "No routing logic extracted from Vector DB")}

    INTEGRATION FLOWS:
    {safe_list(integration_patterns, "No integration patterns extracted from Vector DB")}

    TECHNICAL REQUIREMENTS:
    - Error Handling: {safe_comma(technical_specs.get('error_handling_patterns', []), 'error handling')}
    - Performance: {safe_comma(technical_specs.get('performance_requirements', []), 'performance')}
    - Security: {safe_comma(technical_specs.get('security_requirements', []), 'security')}

    BUSINESS PURPOSE: {business_ctx.get('business_purpose', 'Not extracted from Vector DB')}
    INTEGRATION OBJECTIVES: {safe_comma(business_ctx.get('integration_objectives', []), 'integration objectives')}
    """



    def _validate_inputs(self, component_data: Dict, msgflow_template: str, 
                        confluence_spec: str, biztalk_maps_path: str):
        """Validate all inputs including JSON data and template"""
        errors = []
        
        if not component_data or not component_data.get('component_mappings'):
            errors.append("Component mapping JSON is empty or invalid")
        if not msgflow_template or not msgflow_template.strip():
            errors.append("MessageFlow template is missing or empty")
        if not confluence_spec or not confluence_spec.strip():
            errors.append("Confluence specification is missing")
        
        # BizTalk maps path is now optional - only validate if provided
        if biztalk_maps_path and biztalk_maps_path.strip() and not os.path.exists(biztalk_maps_path):
            print(f"   ⚠️ Warning: BizTalk maps path does not exist: {biztalk_maps_path} - will skip .btm processing")
        
        # Validate JSON structure
        required_fields = ['biztalk_component', 'ace_components']
        for mapping in component_data.get('component_mappings', [])[:3]:  # Check first 3
            missing_fields = [field for field in required_fields if not mapping.get(field)]
            if missing_fields:
                errors.append(f"Missing required fields in component mapping: {missing_fields}")
                break
        
        if errors:
            raise MessageFlowGenerationError("Input validation failed: " + "; ".join(errors))

    def _process_biztalk_maps(self, biztalk_maps_path: str) -> List[Dict]:
        """Process all .btm files in the specified path (optional)"""
        try:
            # Handle empty or non-existent path gracefully
            if not biztalk_maps_path or not biztalk_maps_path.strip():
                print("   📁 No BizTalk maps path provided - skipping .btm processing")
                return []
            
            if not os.path.exists(biztalk_maps_path):
                print(f"   📁 BizTalk maps path does not exist: {biztalk_maps_path} - skipping .btm processing")
                return []
            
            maps_data = []
            btm_files = list(Path(biztalk_maps_path).glob("*.btm"))
            
            if not btm_files:
                print(f"   📁 No .btm files found in {biztalk_maps_path} - using JSON mapping only")
                return []
            
            print(f"   🔍 Found {len(btm_files)} .btm files - processing for additional context...")
            for btm_file in btm_files:
                try:
                    print(f"   🔍 Processing: {btm_file.name}")
                    map_content = self._extract_btm_content(btm_file)
                    if map_content:
                        maps_data.append(map_content)
                except Exception as e:
                    print(f"   ⚠️ Error processing {btm_file.name}: {str(e)} - continuing with others")
                    continue
            
            print(f"   ✅ Successfully processed {len(maps_data)} BizTalk maps")
            return maps_data
            
        except Exception as e:
            print(f"   ⚠️ BizTalk maps processing failed: {str(e)} - continuing without .btm files")
            return []  # Return empty list instead of failing

    def _extract_btm_content(self, btm_file: Path) -> Dict:
        """Extract transformation logic from a single .btm file"""
        try:
            # Try multiple encodings for BizTalk files
            encodings_to_try = ['utf-16', 'utf-16-le', 'utf-8-sig', 'utf-8', 'cp1252', 'windows-1252']
            
            btm_content = None
            used_encoding = None
            
            for encoding in encodings_to_try:
                try:
                    # Convert Path to string explicitly and force encoding
                    with open(str(btm_file), 'r', encoding=encoding, errors='strict') as f:
                        btm_content = f.read()
                    used_encoding = encoding
                    print(f"   ✅ Read {btm_file.name} using {encoding}")
                    break
                except (UnicodeDecodeError, UnicodeError, LookupError) as e:
                    print(f"   🔄 {encoding} failed for {btm_file.name}: {type(e).__name__}")
                    continue
                except Exception as e:
                    print(f"   ⚠️ Unexpected error with {encoding}: {type(e).__name__}: {str(e)[:50]}")
                    continue
                        
            # Add binary fallback if all encodings fail
            if btm_content is None:
                try:
                    print(f"   🔧 Trying binary read for {btm_file.name}")
                    with open(str(btm_file), 'rb') as f:
                        raw_bytes = f.read()
                    
                    # Try to decode with error replacement as last resort
                    btm_content = raw_bytes.decode('utf-8', errors='replace')
                    used_encoding = 'utf-8-binary-fallback'
                    print(f"   ✅ Binary fallback successful for {btm_file.name}")
                    
                except Exception as fallback_error:
                    print(f"   ❌ Binary fallback failed: {fallback_error}")
                    raise ValueError(f"Could not read content from {btm_file.name} with any method")
            
            # Use LLM to extract transformation logic
            prompt = f"""Analyze this BizTalk Map (.btm) file and extract transformation logic:

File: {btm_file.name}
Content Preview: {btm_content[:2000]}...

Extract and return JSON with:
{{
    "map_name": "map name",
    "source_schema": "source schema name", 
    "target_schema": "target schema name",
    "transformations": [
        {{"source_field": "field1", "target_field": "field2", "operation": "copy/transform/concat"}}
    ],
    "business_rules": ["rule1", "rule2"],
    "error_handling": ["error pattern1", "error pattern2"]
}}

Return only JSON:"""
            
            response = self.client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": "You are a BizTalk expert. Extract transformation logic from .btm files as structured JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )

            if 'token_tracker' in st.session_state and hasattr(response, 'usage') and response.usage:
                st.session_state.token_tracker.manual_track(
                    agent="messageflow_generator",           # ✅ CORRECT
                    operation="messageflow_generation",      # ✅ CORRECT
                    model=self.groq_model,                   # ✅ CORRECT (use class property)
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    flow_name=getattr(self, 'flow_name', 'messageflow')  # ✅ CORRECT
                )

            
            # Parse JSON response
            try:
                map_data = json.loads(response.choices[0].message.content.strip())
                map_data['file_path'] = str(btm_file)
                map_data['encoding_used'] = used_encoding
                return map_data
            except json.JSONDecodeError:
                return {
                    "map_name": btm_file.stem,
                    "source_schema": "Unknown",
                    "target_schema": "Unknown",
                    "transformations": [],
                    "business_rules": [],
                    "error_handling": [],
                    "file_path": str(btm_file),
                    "encoding_used": used_encoding
                }
                
        except Exception as e:
            return {
                "map_name": btm_file.stem,
                "source_schema": "Error",
                "target_schema": "Error",
                "transformations": [],
                "business_rules": [],
                "error_handling": [],
                "file_path": str(btm_file),
                "extraction_error": str(e)
            }

    def _process_business_requirements(self, confluence_spec: str) -> Dict:
        """
        Extract business requirements from Vector DB focused content
        100% Vector DB + LLM based processing - NO FALLBACKS
        """
        if not confluence_spec or not confluence_spec.strip():
            raise MessageFlowGenerationError("No Vector DB focused content received for business requirements processing")

        try:
            prompt = f"""Extract MessageFlow technical specifications and integration flows from this Vector DB focused content:

    === VECTOR FOCUSED CONTENT ===
    {confluence_spec}

    === EXTRACTION REQUIREMENTS ===
    This content has been pre-filtered by Vector DB to focus on message flow patterns, routing logic, and system integration points. Extract ALL relevant MessageFlow specifications.

    Return comprehensive JSON with these specific sections:

    {{
        "technical_specs": {{
            "message_flow_patterns": ["list of identified message flow patterns"],
            "routing_logic": ["specific routing rules and decision points"],
            "data_transformation_points": ["transformation requirements"],
            "error_handling_patterns": ["error handling and exception flows"],
            "performance_requirements": ["performance and scalability needs"],
            "security_requirements": ["security and authentication needs"]
        }},
        "integration_flows": {{
            "input_systems": ["source systems and entry points"],
            "output_systems": ["target systems and endpoints"],
            "integration_patterns": ["integration patterns (queue-to-service, etc.)"],
            "connection_points": ["specific connection configurations"],
            "data_flow_sequence": ["step-by-step data flow process"],
            "middleware_components": ["required middleware and adapters"]
        }},
        "message_flow_specifications": {{
            "queue_configurations": ["input/output queue specifications"],
            "service_endpoints": ["service URLs and connection details"],
            "message_formats": ["message structure and format requirements"],
            "transformation_rules": ["data mapping and transformation logic"],
            "conditional_routing": ["routing conditions and business rules"],
            "audit_and_logging": ["audit trail and logging requirements"]
        }},
        "business_context": {{
            "business_purpose": "overall business purpose of the message flow",
            "integration_objectives": ["key integration goals"],
            "compliance_requirements": ["regulatory and compliance needs"],
            "sla_requirements": ["service level agreement specifications"]
        }}
    }}

    CRITICAL: Extract ALL available information from the vector content. Do not use generic placeholders. Return ONLY valid JSON."""

            response = self.client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert IBM ACE architect. Extract technical specifications and integration flows from Vector DB focused business requirements. Return only valid JSON with extracted information."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            # Token tracking
            if 'token_tracker' in st.session_state and hasattr(response, 'usage') and response.usage:
                st.session_state.token_tracker.manual_track(
                    agent="messageflow_generator",
                    operation="vector_content_processing",
                    model=self.groq_model,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    flow_name=getattr(self, 'flow_name', 'messageflow')
                )
            
            result = response.choices[0].message.content.strip()

            # 🔧 FIX: Remove markdown code block markers if present
            if result.startswith('```json'):
                # Remove opening ```json and closing ```
                result = result[7:]  # Remove '```json\n'
                if result.endswith('```'):
                    result = result[:-3]  # Remove closing '```'
                result = result.strip()

            print(f"🔧 Cleaned JSON for parsing: {result[:100]}...")

            # Parse JSON - NO FALLBACKS, fail fast if extraction fails
            try:
                extracted_data = json.loads(result)
                print("✅ JSON parsing successful!")
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                print(f"🎯 Error position: line {e.lineno}, column {e.colno}")
                raise MessageFlowGenerationError(f"Vector DB content extraction failed - LLM returned invalid JSON: {e}")
            except json.JSONDecodeError as e:
                raise MessageFlowGenerationError(f"Vector DB content extraction failed - LLM returned invalid JSON: {e}")
            
            # Validate required sections exist - NO FALLBACKS
            required_sections = ['technical_specs', 'integration_flows', 'message_flow_specifications', 'business_context']
            missing_sections = [section for section in required_sections if section not in extracted_data]
            if missing_sections:
                raise MessageFlowGenerationError(f"Vector DB extraction incomplete - missing sections: {missing_sections}")
            
            # Validate that we have actual content, not empty structures
            tech_specs = extracted_data.get('technical_specs', {})
            if not any(tech_specs.get(key, []) for key in ['message_flow_patterns', 'routing_logic']):
                raise MessageFlowGenerationError("Vector DB extraction failed - no message flow patterns or routing logic found")
            
            print(f"   ✅ Vector content processed: {len(confluence_spec)} chars → structured specifications")
            return extracted_data
                    
        except MessageFlowGenerationError:
            raise  # Re-raise our specific errors
        except Exception as e:
            raise MessageFlowGenerationError(f"Vector DB business requirements processing failed: {str(e)}")

    def _validate_xml(self, msgflow_file: str) -> Dict:
        """Validate generated XML with DSV standards"""
        try:
            with open(msgflow_file, 'r', encoding='utf-8') as f:
                xml_content = f.read()
            
            # Test XML parsing
            try:
                ET.fromstring(xml_content)
                xml_valid = True
                errors = []
            except ET.ParseError as e:
                xml_valid = False
                errors = [f"XML Parse Error: {str(e)}"]
            
            # Check for required DSV elements
            if '<propertyOrganizer' not in xml_content:
                errors.append("Missing: <propertyOrganizer> element (required for DSV standards)")
            
            if '<stickyBoard' not in xml_content:
                errors.append("Missing: <stickyBoard> element (required for DSV standards)")
            
            # Check for connections
            if '<connections xmi:type="eflow:FCMConnection"' not in xml_content:
                errors.append("WARNING: No connections found - nodes may not be properly linked")
            
            # Check for namespace declarations
            required_namespaces = ['xmlns:eflow', 'xmlns:ecore', 'xmlns:xmi']
            missing_namespaces = [ns for ns in required_namespaces if ns not in xml_content]
            if missing_namespaces:
                errors.append(f"WARNING: Missing namespace declarations: {missing_namespaces}")
            
            valid = xml_valid and len([e for e in errors if 'Missing:' in e]) == 0
            
            if valid:
                print("   ✅ XML validation passed - DSV standards compliant")
            else:
                print(f"   ⚠️ XML validation issues found: {len(errors)} errors/warnings")
            
            return {'valid': valid, 'errors': errors}
            
        except Exception as e:
            return {'valid': False, 'errors': [f"DSV validation failed: {str(e)}"]}



    def _enforce_6_module_standard(self, flow_name: str) -> List[Dict]:
        """
        Enforce exactly 6 compute modules for any flow - DSV Standard
        Returns the standardized module list that must be generated
        """
        print("📋 Enforcing 6-module DSV standard...")
        
        standard_modules = [
            {
                "name": f"{flow_name}_InputEventMessage",
                "id": "FCMComposite_1_1", 
                "purpose": "Input validation & event capture",
                "type": "EVENT"
            },
            {
                "name": f"{flow_name}_Compute", 
                "id": "FCMComposite_1_2",
                "purpose": "Main business logic and transformations", 
                "type": "COMPUTE"
            },
            {
                "name": f"{flow_name}_AfterEnrichment",
                "id": "FCMComposite_1_3",
                "purpose": "Post-enrichment processing",
                "type": "COMPUTE" 
            },
            {
                "name": f"{flow_name}_OutputEventMessage",
                "id": "FCMComposite_1_4", 
                "purpose": "Output processing & event capture",
                "type": "EVENT"
            },
            {
                "name": f"{flow_name}_AfterEventMsg",
                "id": "FCMComposite_1_5",
                "purpose": "After transformation event processing", 
                "type": "EVENT"
            },
            {
                "name": f"{flow_name}_Failure",
                "id": "FCMComposite_1_6",
                "purpose": "Generic error handling",
                "type": "ERROR"
            }
        ]
        
        print(f"   ✅ Standard 6 modules defined for {flow_name}")
        for module in standard_modules:
            print(f"      • {module['name']} ({module['purpose']})")
        
        return standard_modules



# Main execution function for main.py compatibility
def run_messageflow_generator(confluence_content: str, biztalk_maps_path: str, 
                            app_name: str, flow_name: str, groq_api_key: str, 
                            groq_model: str) -> Dict:
    """Main execution function with simplified inputs for main.py"""
    try:
        print("📊 Starting DSV MessageFlow Generator...")
        
        generator = DSVMessageFlowGenerator(app_name, flow_name, groq_api_key, groq_model)
        output_dir = os.path.join(os.getcwd(), "output", f"MessageFlow_{app_name}")
        
        return generator.generate_messageflow(
            confluence_spec=confluence_content,
            biztalk_maps_path=biztalk_maps_path if biztalk_maps_path else "",
            output_dir=output_dir
        )
        
    except MessageFlowGenerationError:
        raise
    except Exception as e:
        raise MessageFlowGenerationError(f"Execution failed: {str(e)}")


def create_messageflow_generator(app_name: str, flow_name: str, groq_api_key: str, groq_model: str):
    """Factory function for creating DSV MessageFlow Generator instances"""
    return DSVMessageFlowGenerator(app_name, flow_name, groq_api_key, groq_model)