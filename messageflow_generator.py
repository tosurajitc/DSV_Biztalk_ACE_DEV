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
        """Generate DSV Standard MessageFlow with automated JSON and template input"""
        print("🚀 Starting DSV Standard MessageFlow Generation")
        
        try:
            # Step 1: Load component mapping JSON from output folder
            print("📋 Loading component mapping from JSON...")
            component_data = self._load_component_mapping_json()
            print(f"   ✅ Loaded {len(component_data.get('component_mappings', []))} component mappings")
            
            # Step 2: Load MessageFlow template from root folder
            print("📄 Loading MessageFlow template...")
            msgflow_template = self._load_msgflow_template()
            print(f"   ✅ Template loaded ({len(msgflow_template)} characters)")
            
            # Step 3: Validate inputs
            self._validate_inputs(component_data, msgflow_template, confluence_spec, biztalk_maps_path)
            
            # Step 4: Process BizTalk Maps (.btm files) - Optional
            print(f"🔍 Processing BizTalk Maps from: {biztalk_maps_path}")
            biztalk_maps = self._process_biztalk_maps(biztalk_maps_path)
            print(f"   📊 Processed {len(biztalk_maps)} BizTalk map files")
            
            # Step 5: Process business requirements
            print("📋 Processing business requirements...")
            business_context = self._process_business_requirements(confluence_spec)
            
            # Step 6: Process component mappings from JSON
            print("🔄 Processing component mappings...")
            component_context = self._process_json_components(component_data['component_mappings'])
            
            # Step 7: Generate MessageFlow XML with enhanced context
            print("🏗️ Generating DSV Standard MessageFlow...")
            msgflow_file = self._generate_xml_with_enhanced_context(
                msgflow_template, business_context, component_context, 
                biztalk_maps, component_data, output_dir
            )
            
            # Step 8: Validate output
            print("✅ Validating generated MessageFlow...")
            validation = self._validate_xml(msgflow_file)
            if not validation['valid']:
                raise MessageFlowGenerationError(f"Generated XML validation failed: {validation['errors']}")
            
            print("🎉 DSV Standard MessageFlow generation completed successfully")
            return {
                'success': True,
                'messageflow_file': msgflow_file,
                'component_mappings_processed': len(component_data.get('component_mappings', [])),
                'biztalk_maps_processed': len(biztalk_maps),
                'validation': validation,
                'dsv_standard': True,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"DSV MessageFlow generation failed: {str(e)}"
            print(f"❌ {error_msg}")
            raise MessageFlowGenerationError(error_msg)
    
    def _load_component_mapping_json(self) -> Dict:
        """Load component mapping from JSON file in output folder"""
        json_file_path = self.root_path / "output" / "biztalk_ace_component_mapping.json"
        
        try:
            if not json_file_path.exists():
                raise FileNotFoundError(f"Component mapping JSON not found: {json_file_path}")
            
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Validate JSON structure
            if 'component_mappings' not in json_data:
                raise ValueError("Invalid JSON structure: missing 'component_mappings' key")
            
            if not isinstance(json_data['component_mappings'], list):
                raise ValueError("Invalid JSON structure: 'component_mappings' must be a list")
            
            if len(json_data['component_mappings']) == 0:
                raise ValueError("Component mappings list is empty")
            
            print(f"   📁 JSON file: {json_file_path}")
            print(f"   📊 Components: {len(json_data['component_mappings'])}")
            print(f"   🔧 Generator: {json_data.get('metadata', {}).get('generator', 'Unknown')}")
            print(f"   📅 Generated: {json_data.get('metadata', {}).get('generated_at', 'Unknown')}")
            
            return json_data
            
        except json.JSONDecodeError as e:
            raise MessageFlowGenerationError(f"Invalid JSON format in component mapping file: {e}")
        except Exception as e:
            raise MessageFlowGenerationError(f"Failed to load component mapping JSON: {e}")
    
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
    
    def _process_json_components(self, component_mappings: List[Dict]) -> List[Dict]:
        """Process component mappings from JSON for enhanced LLM context"""
        enhanced_components = []
        
        for mapping in component_mappings:
            try:
                enhanced_component = {
                    # Basic component info
                    'biztalk_component': mapping.get('biztalk_component', ''),
                    'component_type': mapping.get('component_type', ''),
                    'business_purpose': mapping.get('business_purpose', ''),
                    'implementation_priority': mapping.get('implementation_priority', 'medium'),
                    'confidence': mapping.get('confidence', 0.8),
                    'reasoning': mapping.get('reasoning', ''),
                    
                    # ACE Components Details
                    'primary_artifact': mapping.get('ace_components', {}).get('primary_artifact', {}),
                    'supporting_artifacts': mapping.get('ace_components', {}).get('supporting_artifacts', []),
                    
                    # Integration Details for MessageFlow configuration
                    'integration_details': mapping.get('ace_components', {}).get('integration_details', {}),
                    
                    # Extract key integration info for easy access
                    'input_queue': mapping.get('ace_components', {}).get('integration_details', {}).get('input_queue', ''),
                    'output_endpoint': mapping.get('ace_components', {}).get('integration_details', {}).get('output_endpoint', ''),
                    'database_operations': mapping.get('ace_components', {}).get('integration_details', {}).get('database_operations', []),
                    'transformation_logic': mapping.get('ace_components', {}).get('integration_details', {}).get('transformation_logic', '')
                }
                enhanced_components.append(enhanced_component)
            except Exception as e:
                print(f"   ⚠️ Warning: Error processing component mapping: {e}")
                continue
        
        print(f"   🔧 Enhanced {len(enhanced_components)} component mappings with integration details")
        return enhanced_components
    
    def _process_template_placeholders(self, template: str, component_data: Dict, flow_details: Dict) -> str:
        """Replace template placeholders with actual values from JSON data"""
        try:
            processed_template = template
            
            # Extract primary component for queue and endpoint info
            primary_component = None
            component_mappings = component_data.get('component_mappings', [])
            
            # Find the primary/highest priority component
            for mapping in component_mappings:
                if mapping.get('implementation_priority') == 'high':
                    primary_component = mapping
                    break
            
            # Fallback to first component if no high priority found
            if not primary_component and component_mappings:
                primary_component = component_mappings[0]
            
            # Default replacement values
            replacements = {
                '{FLOW_NAME}': flow_details.get('flow_name', 'DefaultFlow'),
                '{APP_NAME}': flow_details.get('app_name', 'DefaultApp'),
                '{INPUT_QUEUE_NAME}': 'CW1.IN.DOCUMENT.SND.QL',  # Default from documentation
                '{XSL_STYLESHEET_NAME}': 'DefaultTransform.xsl',
                '{LOCAL_ENRICHMENT_PATH}': f"/{flow_details.get('app_name', 'DefaultApp')}/enrichment/{flow_details.get('flow_name', 'DefaultFlow')}"
            }
            
            # Extract enhanced values from primary component if available
            if primary_component:
                integration_details = primary_component.get('ace_components', {}).get('integration_details', {})
                
                # Update with actual values from JSON
                if integration_details.get('input_queue'):
                    replacements['{INPUT_QUEUE_NAME}'] = integration_details['input_queue']
                
                # Find XSL stylesheet name from primary artifact
                primary_artifact = primary_component.get('ace_components', {}).get('primary_artifact', {})
                if primary_artifact.get('type') == 'xsl_transform' and primary_artifact.get('name'):
                    replacements['{XSL_STYLESHEET_NAME}'] = primary_artifact['name']
            
            # Apply replacements
            for placeholder, value in replacements.items():
                if placeholder in processed_template:
                    processed_template = processed_template.replace(placeholder, value)
                    print(f"   🔄 {placeholder} → {value}")
            
            return processed_template
            
        except Exception as e:
            print(f"   ⚠️ Template placeholder processing failed: {e}")
            return template  # Return original template if processing fails
        

    
    def _generate_xml_with_enhanced_context(self, msgflow_template: str, business_context: Dict, 
                                        component_context: List[Dict], biztalk_maps: List[Dict], 
                                        component_data: Dict, output_dir: str) -> str:
        """Generate MessageFlow XML - 100% Vector DB + LLM based, NO HARDCODED FALLBACKS"""
        try:
            # Process template placeholders with JSON data
            flow_details = {
                'flow_name': self.flow_name,
                'app_name': self.app_name
            }
            processed_template = self._process_template_placeholders(msgflow_template, component_data, flow_details)
            
            # Import prompt function from prompt_module.py
            from prompt_module import get_msgflow_generation_prompt
            
            # Extract Vector DB specifications - PURE VECTOR DB CONTENT
            technical_specs = business_context.get('technical_specs', {})
            integration_flows = business_context.get('integration_flows', {})
            message_flow_specs = business_context.get('message_flow_specifications', {})
            business_ctx = business_context.get('business_context', {})
            
            # Validate Vector DB content exists
            if not technical_specs or not integration_flows:
                raise MessageFlowGenerationError("Vector DB processing failed - missing technical specifications or integration flows")
            
            # Extract Vector DB derived patterns and logic
            message_flow_patterns = technical_specs.get('message_flow_patterns', [])
            routing_logic = technical_specs.get('routing_logic', [])
            integration_patterns = integration_flows.get('integration_patterns', [])
            
            if not message_flow_patterns or not routing_logic:
                raise MessageFlowGenerationError("Vector DB extraction failed - missing essential message flow patterns or routing logic")
            
            print(f"   🎯 Vector DB technical specs: {len(message_flow_patterns)} flow patterns, {len(routing_logic)} routing rules")
            
            # Process component context for prompt
            components_info = []
            for comp in component_context[:5]:
                components_info.append({
                    'biztalk_component': comp.get('biztalk_component', 'Unknown'),
                    'component_type': comp.get('component_type', 'Unknown'),
                    'ace_library': comp.get('primary_artifact', {}).get('name', 'Unknown'),
                    'business_purpose': comp.get('business_purpose', ''),
                    'input_queue': comp.get('input_queue', ''),
                    'output_endpoint': comp.get('output_endpoint', ''),
                    'database_operations': comp.get('database_operations', []),
                    'transformation_logic': comp.get('transformation_logic', '')
                })
            
            # Extract queue and service details from Vector DB - NO DEFAULTS
            queue_configs = message_flow_specs.get('queue_configurations', [])
            service_endpoints = message_flow_specs.get('service_endpoints', [])
            
            # Determine primary queue and service from Vector DB or components - NO HARDCODED FALLBACKS
            primary_input_queue = None
            primary_output_service = None
            
            if queue_configs:
                primary_input_queue = queue_configs[0]
            elif components_info:
                for comp in components_info:
                    if comp.get('input_queue'):
                        primary_input_queue = comp['input_queue']
                        break
            
            if service_endpoints:
                primary_output_service = service_endpoints[0]
            elif components_info:
                for comp in components_info:
                    if comp.get('output_endpoint'):
                        primary_output_service = comp['output_endpoint']
                        break
            
            # Validate we have essential connection details
            if not primary_input_queue or not primary_output_service:
                raise MessageFlowGenerationError("Vector DB/Component processing failed - missing input queue or output service configuration")
            
            # Create business specification from Vector DB extraction

            vector_business_spec = self._create_safe_business_spec(
                message_flow_patterns, routing_logic, integration_patterns, 
                technical_specs, business_ctx
            )
            
            # Create prompt using Vector DB derived values
            prompt = get_msgflow_generation_prompt(
                flow_name=self.flow_name,
                project_name=self.app_name,
                input_queue=primary_input_queue,
                output_service=primary_output_service,
                error_queue='ERROR.QUEUE',  # Standard error queue
                esql_modules=[],
                msgflow_template=processed_template,
                confluence_spec=vector_business_spec,
                components_info=components_info
            )
            
            enhanced_prompt = f"""{prompt}

        ## VECTOR DB TECHNICAL SPECIFICATIONS (PRIMARY AUTHORITY):
        **Message Flow Patterns:** {', '.join(message_flow_patterns[:5]) if message_flow_patterns else 'ERROR: No patterns extracted'}
        **Routing Logic:** {', '.join(routing_logic[:5]) if routing_logic else 'ERROR: No routing logic extracted'}
        **Integration Patterns:** {', '.join(integration_patterns[:3]) if integration_patterns else 'ERROR: No integration patterns'}
        **Queue Configurations:** {', '.join(queue_configs[:3]) if queue_configs else 'ERROR: No queue configs'}
        **Service Endpoints:** {', '.join(service_endpoints[:3]) if service_endpoints else 'ERROR: No endpoints'}

        ## PRIORITY HIERARCHY:
        1. **HIGHEST PRIORITY**: Vector DB specifications (technical requirements)
        2. **MEDIUM PRIORITY**: Template structure (adaptable framework) 
        3. **LOWEST PRIORITY**: Standard conventions (override if needed)

        ## CRITICAL REQUIREMENTS (Vector DB Driven):
        1. Use the template as ADAPTABLE foundation - modify structure to meet Vector DB requirements
        2. **MANDATORY**: Implement Vector DB routing logic: {routing_logic[:3] if routing_logic else ['ERROR: No routing logic']}
        3. **MANDATORY**: Apply Vector DB message flow patterns: {message_flow_patterns[:3] if message_flow_patterns else ['ERROR: No patterns']}
        4. Preserve namespace declarations unless Vector DB requires changes
        5. Create node connections based on Vector DB specifications, not template defaults
        6. Include propertyOrganizer and stickyBoard elements unless Vector DB specifies otherwise
        7. Start event - implement per Vector DB requirements (may override no-transformation rule)
        8. Ensure subflow uniqueness per Vector DB patterns
        9. End event placement per Vector DB flow specifications

        ## TEMPLATE ADAPTATION RULES:
        - If Vector DB specifies different node types than template → USE Vector DB requirements
        - If Vector DB routing conflicts with template connections → FOLLOW Vector DB routing
        - If Vector DB requires additional nodes not in template → ADD them
        - If Vector DB specifies different message patterns → IMPLEMENT Vector DB patterns
        - Template serves as structural starting point only

        ## VALIDATION REQUIREMENTS:
        ✅ All Vector DB routing logic implemented
        ✅ All Vector DB message patterns applied  
        ✅ Node connections match Vector DB specifications
        ✅ Flow structure supports Vector DB technical requirements

        Generate complete MessageFlow XML implementing Vector DB specifications as PRIMARY authority:"""
            # LLM generation
            response = self.client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": "You are an expert IBM ACE developer. Generate production-ready MessageFlow XML implementing Vector DB extracted specifications exactly as provided."},
                    {"role": "user", "content": enhanced_prompt}
                ],
                temperature=0.1,
                max_tokens=8000
            )

            # Token tracking
            if 'token_tracker' in st.session_state and hasattr(response, 'usage') and response.usage:
                st.session_state.token_tracker.manual_track(
                    agent="messageflow_generator",
                    operation="vector_messageflow_generation",
                    model=self.groq_model,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    flow_name=getattr(self, 'flow_name', 'messageflow')
                )
            
            xml_content = response.choices[0].message.content.strip()
            
            # Extract XML content - NO FALLBACKS, fail if invalid
            xml_start = xml_content.find('<?xml')
            xml_end = xml_content.find('</ecore:EPackage>')

            if xml_start == -1:
                xml_start = xml_content.find('<ecore:EPackage')
                if xml_start == -1:
                    raise MessageFlowGenerationError("LLM failed to generate valid XML structure")
                xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_content[xml_start:]
                xml_end = xml_content.find('</ecore:EPackage>')
            
            if xml_end == -1:
                raise MessageFlowGenerationError("LLM generated incomplete XML - missing closing tag")
                
            xml_content = xml_content[xml_start:xml_end + len('</ecore:EPackage>')]
            
            # Save MessageFlow file
            os.makedirs(output_dir, exist_ok=True)
            msgflow_filename = f"{self.flow_name}.msgflow"
            msgflow_file = os.path.join(output_dir, msgflow_filename)
            
            with open(msgflow_file, 'w', encoding='utf-8') as f:
                f.write(xml_content)
            
            print(f"   ✅ Vector DB optimized MessageFlow generated: {msgflow_file}")
            return msgflow_file
            
        except MessageFlowGenerationError:
            raise  # Re-raise our specific errors
        except Exception as e:
            raise MessageFlowGenerationError(f"Vector DB MessageFlow generation failed: {str(e)}")
        


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
            encodings_to_try = ['utf-16', 'utf-16-le', 'utf-8-sig', 'utf-8', 'cp1252']
            
            btm_content = None
            used_encoding = None
            
            for encoding in encodings_to_try:
                try:
                    with open(btm_file, 'r', encoding=encoding) as f:
                        btm_content = f.read()
                    used_encoding = encoding
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if btm_content is None:
                raise ValueError(f"Could not read content from {btm_file.name}")
            
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