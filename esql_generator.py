"""
Generates IBM ACE ESQL modules from:
1. Vector DB business requirements (business context)
2. MessageFlow XML (source of truth for module names)
3. ESQL Template (structural foundation)
4. Component mappings (transformation patterns)

Key Optimizations:
- MessageFlow-driven requirements (eliminates generic placeholders)
- Template-first generation (LLM fills business logic only)
- Integrated validation pipeline (auto-fix before write)
- Unified LLM integration with llm_json_parser

"""

import os
import json
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from groq import Groq
from llm_json_parser import LLMJSONParser, parse_llm_json


class ESQLGenerator:
    """
    Optimized ESQL Generator with MessageFlow-driven requirements extraction
    and template-first generation approach.
    """
    
    def __init__(self, groq_api_key: Optional[str] = None, groq_model: str = "llama-3.3-70b-versatile"):
        """
        Initialize ESQL Generator with LLM configuration.
        
        Args:
            groq_api_key: Groq API key (optional, can use environment variable)
            groq_model: LLM model to use (no hardcoding, configurable)
        """
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY must be provided or set in environment")
        
        self.groq_client = Groq(api_key=self.groq_api_key)
        self.groq_model = groq_model
        self.json_parser = LLMJSONParser(debug=False)
        
        # Generation tracking
        self.generation_stats = {
            'llm_calls': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'auto_fixes_applied': 0
        }
        
        print(f"✅ ESQLGenerator initialized with model: {self.groq_model}")
    
    
    def _load_inputs(self, esql_template: Dict, msgflow_content: Dict, 
                     json_mappings: Dict) -> Tuple[str, str, Dict, Dict]:
        """
        Load all input files needed for ESQL generation.
        
        Args:
            esql_template: Dict with 'path' key pointing to ESQL_Template_Updated.ESQL
            msgflow_content: Dict with 'path' key pointing to .msgflow file
            json_mappings: Dict with 'path' key pointing to component_mapping.json
        
        Returns:
            Tuple of (template_content, msgflow_xml, mappings_data, naming_data)
        """
        print("📂 Loading input files...")
        
        # Load ESQL Template
        template_path = esql_template.get('path')
        if not template_path or not os.path.exists(template_path):
            raise FileNotFoundError(f"ESQL template not found: {template_path}")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        print(f"  ✅ Template loaded: {len(template_content)} chars")
        
        # Load MessageFlow XML
        msgflow_path = msgflow_content.get('path')
        if not msgflow_path:
            # Auto-discover from output directory
            import glob
            msgflow_files = glob.glob("output/**/*.msgflow", recursive=True)
            if msgflow_files:
                msgflow_path = msgflow_files[0]
                print(f"  🔍 Auto-discovered: {msgflow_path}")
            else:
                raise FileNotFoundError("No .msgflow file found")
        
        if not os.path.exists(msgflow_path):
            raise FileNotFoundError(f"MessageFlow not found: {msgflow_path}")
        
        with open(msgflow_path, 'r', encoding='utf-8') as f:
            msgflow_xml = f.read()
        print(f"  ✅ MessageFlow loaded: {len(msgflow_xml)} chars")
        
        # Load Component Mappings
        mappings_path = json_mappings.get('path')
        if not mappings_path or not os.path.exists(mappings_path):
            raise FileNotFoundError(f"Component mappings not found: {mappings_path}")
        
        with open(mappings_path, 'r', encoding='utf-8') as f:
            mappings_data = json.load(f)
        print(f"  ✅ Mappings loaded: {len(mappings_data)} items")
        
        # Load Naming Convention
        naming_data = self._load_naming_convention()
        
        return template_content, msgflow_xml, mappings_data, naming_data
    
    
    def _load_naming_convention(self) -> Dict:
        """
        Load naming convention from naming_convention.json.
        Supports both old and new formats.
        
        Returns:
            Dict with flow naming information
        """
        naming_path = "naming_convention.json"
        if not os.path.exists(naming_path):
            raise FileNotFoundError(f"Naming convention not found: {naming_path}")
        
        with open(naming_path, 'r', encoding='utf-8') as f:
            naming_data = json.load(f)
        
        print(f"  ✅ Naming convention loaded")
        return naming_data
    
    
    def _extract_requirements_from_sources(self, vector_content: str, msgflow_xml: str, 
                                          mappings_data: Dict, naming_data: Dict) -> List[Dict]:
        """
        Extract ESQL module requirements using MessageFlow-first approach.
        
        This is the KEY optimization: MessageFlow compute nodes are the source of truth.
        Vector DB and mappings provide business context only.
        
        Args:
            vector_content: Business requirements from Vector DB
            msgflow_xml: MessageFlow XML content
            mappings_data: Component mappings
            naming_data: Flow naming convention
        
        Returns:
            List of module requirements with structure:
            [
                {
                    'name': 'CW1_IN_Document_SND_Compute',
                    'type': 'compute',
                    'purpose': 'Business logic processing',
                    'business_logic': {...},
                    'source': 'messageflow'
                },
                ...
            ]
        """
        print("🔍 Extracting ESQL requirements (MessageFlow-first approach)...")
        
        # STEP 1: Extract compute nodes from MessageFlow (SOURCE OF TRUTH)
        msgflow_modules = self._parse_msgflow_compute_nodes(msgflow_xml, naming_data)
        print(f"  📋 Found {len(msgflow_modules)} compute nodes in MessageFlow")
        
        if not msgflow_modules:
            raise Exception("No compute nodes found in MessageFlow XML")
        
        # STEP 2: Enrich with business context from Vector DB via LLM
        print("  🧠 Enriching modules with Vector DB business context...")
        
        enrichment_prompt = f"""Analyze business requirements and enrich ESQL module specifications.

MESSAGEFLOW MODULES (SOURCE OF TRUTH - DO NOT ADD OR REMOVE):
{json.dumps(msgflow_modules, indent=2)}

VECTOR DB BUSINESS REQUIREMENTS:
{vector_content[:4000]}

COMPONENT MAPPINGS CONTEXT:
{json.dumps(mappings_data.get('component_mappings', [])[:5], indent=2)}

TASK:
For EACH module in the MessageFlow list, extract relevant business logic from Vector DB.
Return the SAME modules with enriched business_logic.

CRITICAL RULES:
1. Return EXACTLY {len(msgflow_modules)} modules - no more, no less
2. Keep ALL module names exactly as provided from MessageFlow
3. DO NOT create generic names like "ESQLModule_1" or "Module_1"
4. Only ADD business context to existing modules
5. Match business requirements to module purposes

Return JSON format:
{{
    "esql_modules": [
        {{
            "name": "exact_name_from_messageflow",
            "type": "compute|input_event|output_event|failure",
            "purpose": "description from business requirements",
            "business_logic": {{
                "database_operations": ["list of DB operations from Vector DB"],
                "transformations": ["list of transformations needed"],
                "validation_rules": ["business validation rules"],
                "error_handling": ["error scenarios to handle"]
            }},
            "source": "messageflow"
        }}
    ],
    "database_operations": ["global list of all stored procedures mentioned"],
    "transformations": ["global transformation patterns"]
}}

Return ONLY valid JSON."""

        # Call LLM with integrated JSON parsing
        enriched_data = self._call_llm_with_parsing(enrichment_prompt, "requirements_enrichment")
        
        # Validate structure
        if not enriched_data or 'esql_modules' not in enriched_data:
            print("  ⚠️ LLM enrichment returned invalid structure, using MessageFlow modules as-is")
            return msgflow_modules
        
        enriched_modules = enriched_data['esql_modules']
        
        # STEP 3: Validate and clean enriched modules
        validated_modules = []
        for module in enriched_modules:
            # Ensure it's a dict
            if isinstance(module, str):
                # Skip if it's a generic placeholder
                if module.startswith('ESQLModule_') or module.startswith('Module_'):
                    print(f"  ⚠️ Skipping generic placeholder: {module}")
                    continue
                # Convert to dict
                module = {'name': module, 'type': 'compute', 'purpose': 'Processing', 'business_logic': {}}
            
            # Skip generic placeholders
            if module.get('name', '').startswith('ESQLModule_') or module.get('name', '').startswith('Module_'):
                print(f"  ⚠️ Skipping generic placeholder: {module.get('name')}")
                continue
            
            # Ensure required keys
            module.setdefault('name', 'UnknownModule')
            module.setdefault('type', 'compute')
            module.setdefault('purpose', 'Processing')
            module.setdefault('business_logic', {})
            module.setdefault('source', 'messageflow')
            
            validated_modules.append(module)
        
        print(f"  ✅ Requirements extracted: {len(validated_modules)} modules validated")
        
        # Store global context
        self.database_operations = enriched_data.get('database_operations', [])
        self.transformations = enriched_data.get('transformations', [])
        
        return validated_modules
    
    
    def _parse_msgflow_compute_nodes(self, msgflow_xml: str, naming_data: Dict) -> List[Dict]:
        """
        Parse MessageFlow XML to extract compute node expressions.
        These are the ONLY modules we should generate.
        
        Args:
            msgflow_xml: MessageFlow XML content
            naming_data: Naming convention for flow name
        
        Returns:
            List of module dictionaries with names from MessageFlow
        """
        print("  🔍 Parsing MessageFlow compute nodes...")
        
        try:
            root = ET.fromstring(msgflow_xml)
        except ET.ParseError as e:
            raise Exception(f"Failed to parse MessageFlow XML: {e}")
        
        # Extract flow name
        flow_name = self._extract_flow_name(naming_data)
        
        # Find all compute nodes with namespaces
        namespaces = {
            'eflow': 'http://www.ibm.com/wbi/2005/eflow',
            'xmi': 'http://www.omg.org/XMI'
        }
        
        compute_nodes = []
        
        # Find compute nodes (handle both with and without namespace)
        for node in root.findall(".//{http://www.ibm.com/wbi/2005/eflow}nodes", namespaces):
            compute_expr = node.get('computeExpression', '')
            if compute_expr and 'esql://routine/' in compute_expr:
                # Extract module name from expression
                # Format: esql://routine/{flow_name}#{module_name}.Main
                match = re.search(r'esql://routine/[^#]+#([^.]+)\.Main', compute_expr)
                if match:
                    module_name = match.group(1)
                    
                    # Determine module type from name suffix
                    module_type = self._determine_module_type(module_name)
                    
                    # Get node label for purpose
                    translation = node.find('.//{http://www.ibm.com/wbi/2005/eflow_utility}ConstantString')
                    purpose = translation.get('string', 'Processing') if translation is not None else 'Processing'
                    
                    compute_nodes.append({
                        'name': module_name,
                        'type': module_type,
                        'purpose': purpose,
                        'business_logic': {},
                        'source': 'messageflow',
                        'compute_expression': compute_expr
                    })
        
        # Fallback: try without namespace
        if not compute_nodes:
            for node in root.findall('.//nodes'):
                compute_expr = node.get('computeExpression', '')
                if compute_expr and 'esql://routine/' in compute_expr:
                    match = re.search(r'esql://routine/[^#]+#([^.]+)\.Main', compute_expr)
                    if match:
                        module_name = match.group(1)
                        module_type = self._determine_module_type(module_name)
                        
                        compute_nodes.append({
                            'name': module_name,
                            'type': module_type,
                            'purpose': 'Processing',
                            'business_logic': {},
                            'source': 'messageflow',
                            'compute_expression': compute_expr
                        })
        
        print(f"    Found {len(compute_nodes)} compute expressions")
        for node in compute_nodes:
            print(f"      • {node['name']} ({node['type']})")
        
        return compute_nodes
    
    
    def _extract_flow_name(self, naming_data: Dict) -> str:
        """
        Extract flow name from naming convention.
        Supports both old and new formats.
        
        Args:
            naming_data: Naming convention dictionary
        
        Returns:
            Flow name string
        """
        # Try new format first
        project_naming = naming_data.get('project_naming', {})
        if project_naming:
            flow_name = project_naming.get('message_flow_name') or project_naming.get('flow_name')
            if flow_name:
                return flow_name
        
        # Try old format
        flow_name = naming_data.get('message_flow_name') or naming_data.get('flow_name')
        if flow_name:
            return flow_name
        
        raise Exception("Could not extract flow_name from naming_convention.json")
    
    
    def _determine_module_type(self, module_name: str) -> str:
        """
        Determine module type from module name suffix.
        
        Args:
            module_name: Full module name (e.g., "CW1_IN_Document_SND_Compute")
        
        Returns:
            Module type: 'input_event', 'output_event', 'compute', 'failure', 'processing'
        """
        if module_name.endswith('_InputEventMessage'):
            return 'input_event'
        elif module_name.endswith('_OutputEventMessage'):
            return 'output_event'
        elif module_name.endswith('_AfterEventMsg'):
            return 'output_event'
        elif module_name.endswith('_Failure'):
            return 'failure'
        elif module_name.endswith('_Compute'):
            return 'compute'
        elif module_name.endswith('_AfterEnrichment') or module_name.endswith('_Processing'):
            return 'processing'
        else:
            return 'compute'  # Default
    
    
    def _call_llm_with_parsing(self, prompt: str, call_type: str = "generation") -> Dict:
        """
        Unified LLM call with integrated JSON parsing using llm_json_parser.
        
        Args:
            prompt: Full prompt to send to LLM
            call_type: Type of call for tracking (e.g., "requirements", "generation")
        
        Returns:
            Parsed JSON dictionary
        """
        self.generation_stats['llm_calls'] += 1
        
        try:
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert IBM ACE ESQL developer. Return ONLY valid JSON, no markdown, no explanations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=4000
            )
            
            llm_response = response.choices[0].message.content
            
            # Parse using llm_json_parser
            parse_result = parse_llm_json(llm_response)
            
            if parse_result.success:
                return parse_result.data
            else:
                print(f"  ⚠️ JSON parsing failed ({call_type}): {parse_result.error_message}")
                print(f"  📄 Raw response preview: {llm_response[:200]}...")
                raise Exception(f"Failed to parse LLM JSON response: {parse_result.error_message}")
        
        except Exception as e:
            print(f"  ❌ LLM call failed ({call_type}): {str(e)}")
            raise




    # ============================================================================
    # PART 2: GENERATION METHODS
    # ============================================================================
    
    def _is_event_message_module(self, module_name: str) -> bool:
        """
        Check if module is an event message type (metadata capture only).
        Event messages use template copying, no LLM generation needed.
        
        Args:
            module_name: Full module name
        
        Returns:
            True if InputEventMessage or OutputEventMessage type
        """
        return (module_name.endswith('_InputEventMessage') or 
                module_name.endswith('_OutputEventMessage') or
                module_name.endswith('_AfterEventMsg'))
    
    
    def _generate_event_message_esql(self, module_name: str, naming_data: Dict, 
                                    template_content: str) -> str:
        """
        Generate event message ESQL using template copying (Tier 1 - No LLM).
        Event messages contain only metadata capture logic from template.
        
        Args:
            module_name: Full module name (e.g., "CW1_IN_Document_SND_InputEventMessage")
            naming_data: Naming convention dictionary
            template_content: Full ESQL template content
        
        Returns:
            Complete ESQL code ready to write
        """
        print(f"    📋 Template copy for event message: {module_name}")
        
        # Extract flow name
        flow_name = self._extract_flow_name(naming_data)
        
        # Extract base flow name (without suffix)
        base_flow_name = self._extract_base_flow_name(module_name)
        
        # Load INPUT EVENT MESSAGE template section
        template_section = self._load_template_section(template_content, 'input_event')
        
        # Apply naming to template
        esql_content = self._apply_event_naming(template_section, base_flow_name, module_name)
        
        # Validate and auto-fix
        esql_content, fixes = self._validate_and_fix_esql_structure(
            esql_content, module_name, base_flow_name
        )
        
        if fixes:
            print(f"      🔧 Applied {len(fixes)} auto-fixes")
            self.generation_stats['auto_fixes_applied'] += len(fixes)
        
        return esql_content
    
    
    def _apply_event_naming(self, template_content: str, flow_name: str, 
                           full_module_name: str) -> str:
        """
        Replace placeholders in event message template with actual names.
        
        Args:
            template_content: Raw template section from ESQL_Template_Updated.ESQL
            flow_name: Base flow name (e.g., "CW1_IN_Document_SND")
            full_module_name: Complete module name with suffix
        
        Returns:
            Complete ESQL code with names applied
        """
        lines = template_content.split('\n')
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Replace BROKER SCHEMA line (uses base flow_name without suffix)
            if stripped.startswith('BROKER SCHEMA'):
                lines[i] = f'BROKER SCHEMA {flow_name}'
            
            # Replace CREATE COMPUTE MODULE line (uses full_module_name with suffix)
            elif stripped.startswith('CREATE COMPUTE MODULE'):
                lines[i] = f'CREATE COMPUTE MODULE {full_module_name}'
        
        return '\n'.join(lines)
    
    
    def _generate_business_logic_esql(self, module_req: Dict, naming_data: Dict, 
                                     template_content: str, vector_context: str = "") -> str:
        """
        Generate business logic ESQL using template-first approach with LLM injection.
        
        This is the KEY generation method:
        1. Load template section as foundation
        2. Prepare structure (set BROKER SCHEMA and MODULE names upfront)
        3. LLM injects business logic into marker
        4. Validate and auto-fix structure
        
        Args:
            module_req: Module requirements with business_logic
            naming_data: Naming convention
            template_content: Full ESQL template
            vector_context: Vector DB context for business logic
        
        Returns:
            Complete ESQL code ready to write
        """
        module_name = module_req['name']
        module_type = module_req.get('type', 'compute')
        
        print(f"    🧠 LLM generation for: {module_name} ({module_type})")
        
        # STEP 1: Load appropriate template section
        template_section = self._load_template_section(template_content, module_type)
        
        # STEP 2: Prepare template structure (set names BEFORE LLM)
        flow_name = self._extract_flow_name(naming_data)
        base_flow_name = self._extract_base_flow_name(module_name)
        
        prepared_template = self._prepare_template_structure(
            template_section, base_flow_name, module_name
        )
        
        # STEP 3: LLM injects business logic into prepared template
        esql_content = self._llm_inject_business_logic(
            prepared_template, module_req, flow_name, vector_context
        )
        
        # STEP 4: Validate and auto-fix structure
        esql_content, fixes = self._validate_and_fix_esql_structure(
            esql_content, module_name, base_flow_name
        )
        
        if fixes:
            print(f"      🔧 Applied {len(fixes)} auto-fixes")
            self.generation_stats['auto_fixes_applied'] += len(fixes)
        
        return esql_content
    
    
    def _load_template_section(self, template_content: str, template_type: str) -> str:
        """
        Extract specific template section from ESQL_Template_Updated.ESQL.
        
        Template sections:
        - 'input_event': INPUT AND OUTPUT EVENT MESSAGE TEMPLATE
        - 'compute': COMPUTE TEMPLATE - FULL BUSINESS LOGIC
        - 'processing': PROCESSING TEMPLATE - VALIDATION AND ROUTING
        - 'failure': FAILURE/ERROR HANDLING TEMPLATE
        
        Args:
            template_content: Full ESQL template file content
            template_type: Type of template section to extract
        
        Returns:
            Extracted template section from BROKER SCHEMA to END MODULE;
        """
        # Section markers in template
        section_markers = {
            'input_event': ['INPUT AND OUTPUT EVENT MESSAGE TEMPLATE - METADATA ONLY', 'COMPUTE TEMPLATE - FULL BUSINESS LOGIC'],
            'compute': ['COMPUTE TEMPLATE - FULL BUSINESS LOGIC', 'PROCESSING TEMPLATE'],
            'processing': ['PROCESSING TEMPLATE - VALIDATION AND ROUTING ONLY', 'FAILURE/ERROR HANDLING TEMPLATE'],
            'failure': ['FAILURE/ERROR HANDLING TEMPLATE', None]  # Last section
        }
        
        # Map generic to compute
        if template_type == 'generic':
            template_type = 'compute'
        
        if template_type not in section_markers:
            raise Exception(f"Invalid template_type: {template_type}. Valid: input_event, compute, processing, failure")
        
        start_marker, end_marker = section_markers[template_type]
        lines = template_content.split('\n')
        
        # Find section boundaries
        section_start = None
        section_end = len(lines)
        
        for i, line in enumerate(lines):
            if start_marker in line:
                section_start = i
            if end_marker and end_marker in line and section_start is not None:
                section_end = i
                break
        
        if section_start is None:
            raise Exception(f"Template section marker not found: '{start_marker}'")
        
        # Find first BROKER SCHEMA or CREATE COMPUTE MODULE line (actual ESQL start)
        esql_start_line = None
        for i in range(section_start, section_end):
            line = lines[i].strip()
            if line.startswith('BROKER SCHEMA') or line.startswith('CREATE COMPUTE MODULE'):
                esql_start_line = i
                break
        
        if esql_start_line is None:
            raise Exception(f"No ESQL structure found in template section: {template_type}")
        
        # Extract section
        selected_template = '\n'.join(lines[esql_start_line:section_end]).strip()
        
        return selected_template
    
    
    def _prepare_template_structure(self, template_section: str, base_flow_name: str, 
                                    module_name: str) -> str:
        """
        Prepare template structure by setting BROKER SCHEMA and CREATE COMPUTE MODULE BEFORE LLM.
        This ensures correct naming regardless of LLM output.
        
        Args:
            template_section: Template section from _load_template_section
            base_flow_name: Base flow name for BROKER SCHEMA (e.g., "CW1_IN_Document_SND")
            module_name: Full module name for CREATE COMPUTE MODULE
        
        Returns:
            Template with correct BROKER SCHEMA and MODULE names set
        """
        lines = template_section.split('\n')
        
        # Update BROKER SCHEMA (line 1)
        for i, line in enumerate(lines):
            if line.strip().startswith('BROKER SCHEMA'):
                lines[i] = f'BROKER SCHEMA {base_flow_name}'
                break
        
        # Update CREATE COMPUTE MODULE (line 2)
        for i, line in enumerate(lines):
            if line.strip().startswith('CREATE COMPUTE MODULE'):
                lines[i] = f'CREATE COMPUTE MODULE {module_name}'
                break
        
        # Replace placeholder in template
        placeholder = "_SYSTEM___MSG_TYPE___FLOW_PROCESS___SYSTEM2___FLOW_TYPE"
        prepared = '\n'.join(lines)
        prepared = prepared.replace(placeholder, module_name)
        
        return prepared
    
    
    def _llm_inject_business_logic(self, prepared_template: str, module_req: Dict, 
                                   flow_name: str, vector_context: str) -> str:
        """
        LLM injects business logic into the prepared template at marker location.
        LLM does NOT generate full module - only fills business logic section.
        
        Args:
            prepared_template: Template with BROKER SCHEMA and MODULE names already set
            module_req: Module requirements with business_logic
            flow_name: Flow name for context
            vector_context: Vector DB business requirements
        
        Returns:
            Complete ESQL with business logic injected
        """
        module_name = module_req['name']
        business_logic = module_req.get('business_logic', {})
        purpose = module_req.get('purpose', 'Processing')
        
        # Build business logic context
        db_operations = business_logic.get('database_operations', [])
        transformations = business_logic.get('transformations', [])
        validation_rules = business_logic.get('validation_rules', [])
        
        # Add global context
        db_operations.extend(self.database_operations[:5])
        transformations.extend(self.transformations[:5])
        
        # Build focused business logic prompt
        business_logic_prompt = f"""You are injecting business logic into an ESQL template.

PREPARED TEMPLATE (DO NOT MODIFY STRUCTURE):
{prepared_template}

MODULE SPECIFICATION:
- Name: {module_name}
- Purpose: {purpose}
- Flow: {flow_name}

BUSINESS LOGIC TO INJECT:
Database Operations:
{json.dumps(db_operations[:6], indent=2)}

Transformations:
{transformations[:5]}

Validation Rules:
{validation_rules[:5]}

VECTOR DB CONTEXT:
{vector_context[:2000]}

CRITICAL INSTRUCTIONS:
1. Find the marker: -- [[[INSERT_BUSINESS_LOGIC_HERE]]]
2. Replace ONLY that marker with business logic code
3. PRESERVE all template structure exactly:
   - Keep BROKER SCHEMA line (line 1)
   - Keep CREATE COMPUTE MODULE line (line 2)
   - Keep CREATE FUNCTION Main() RETURNS BOOLEAN
   - Keep all DECLARE statements
   - Keep RETURN TRUE; at end of Main()
   - Keep CopyMessageHeaders() procedure
   - Keep CopyEntireMessage() procedure
   - Keep END MODULE; as LAST LINE

4. Business logic must use:
   - SET statements for data manipulation
   - OutputRoot for output (InputRoot is READ-ONLY)
   - Environment.variables for shared data
   - PASSTHRU for database calls
   - Proper ESQL syntax (no @ symbols, no "esql" prefix)

5. FORBIDDEN DATA TYPES:
   - NEVER use: VARCHAR, XML, RECORD, STRING, JSON, Database
   - ONLY use: BOOLEAN, INTEGER, DECIMAL, FLOAT, CHARACTER, BIT, BLOB, DATE, TIME, TIMESTAMP, REFERENCE, ROW

6. Database operation pattern:
DECLARE result CHARACTER;
SET result = PASSTHRU('CALL sp_GetCompany(?, ?)',
InputRoot.XMLNSC.:Header.:CompanyCode,
InputRoot.XMLNSC.:Header.:CountryCode
);
SET OutputRoot.XMLNSC.*:CompanyData = result;

7. The LAST line MUST be: END MODULE;

Return the COMPLETE template with business logic injected.
NO markdown, NO code blocks, just pure ESQL."""

        # Call LLM
        self.generation_stats['llm_calls'] += 1
        
        try:
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an ESQL code injector. Replace ONLY the business logic marker in the template. Preserve ALL template structure. Return pure ESQL code only."
                    },
                    {
                        "role": "user",
                        "content": business_logic_prompt
                    }
                ],
                temperature=0.3,
                max_tokens=3500
            )
            
            esql_content = response.choices[0].message.content.strip()
            
            # Clean common LLM artifacts
            if esql_content.startswith('```esql'):
                esql_content = esql_content[7:]
            if esql_content.startswith('```'):
                esql_content = esql_content[3:]
            if esql_content.endswith('```'):
                esql_content = esql_content[:-3]
            
            esql_content = esql_content.strip()
            
            return esql_content
        
        except Exception as e:
            print(f"      ❌ LLM injection failed: {str(e)}")
            # Fallback: return prepared template as-is
            print(f"      📋 Using template without business logic injection")
            return prepared_template
    
    
    def _extract_base_flow_name(self, module_name: str) -> str:
        """
        Extract base flow name from module name by removing standard suffixes.
        
        Args:
            module_name: Full module name (e.g., "CW1_IN_Document_SND_Compute")
        
        Returns:
            Base flow name (e.g., "CW1_IN_Document_SND")
        """
        suffixes = [
            '_InputEventMessage',
            '_OutputEventMessage',
            '_AfterEventMsg',
            '_Compute',
            '_AfterEnrichment',
            '_Processing',
            '_Failure'
        ]
        
        base_name = module_name
        for suffix in suffixes:
            if module_name.endswith(suffix):
                base_name = module_name[:-len(suffix)]
                break
        
        return base_name
    

# ============================================================================
    # PART 3: VALIDATION & AUTO-FIX
    # ============================================================================
    
    def _validate_and_fix_esql_structure(self, esql_content: str, module_name: str, 
                                        flow_name: str) -> Tuple[str, List[str]]:
        """
        Validate ESQL structure and automatically fix common issues.
        This is the CRITICAL method that ensures all ESQL files are production-ready.
        
        Validates and fixes:
        1. BROKER SCHEMA line (must be line 1)
        2. CREATE COMPUTE MODULE line (must match module_name)
        3. CopyMessageHeaders procedure (required)
        4. CopyEntireMessage procedure (required)
        5. END MODULE; statement (must be last line)
        
        Args:
            esql_content: Generated ESQL content
            module_name: Expected module name
            flow_name: Base flow name for BROKER SCHEMA
        
        Returns:
            Tuple of (fixed_content, list_of_fixes_applied)
        """
        fixes_applied = []
        fixed_content = esql_content.strip()
        
        # ============================================================
        # PART 1: Fix Header Structure
        # ============================================================
        
        lines = fixed_content.split('\n')
        
        # Check/Fix BROKER SCHEMA (must be line 1)
        if not lines[0].strip().startswith('BROKER SCHEMA'):
            # Missing BROKER SCHEMA - prepend it
            lines.insert(0, f'BROKER SCHEMA {flow_name}')
            fixed_content = '\n'.join(lines)
            fixes_applied.append(f"Added BROKER SCHEMA {flow_name}")
        else:
            # Verify BROKER SCHEMA name
            current_schema = lines[0].strip().replace('BROKER SCHEMA ', '')
            if current_schema != flow_name:
                lines[0] = f'BROKER SCHEMA {flow_name}'
                fixed_content = '\n'.join(lines)
                fixes_applied.append(f"Fixed BROKER SCHEMA name to {flow_name}")
        
        # Check/Fix CREATE COMPUTE MODULE
        if 'CREATE COMPUTE MODULE' not in fixed_content:
            # Missing CREATE COMPUTE MODULE - insert after BROKER SCHEMA
            lines = fixed_content.split('\n')
            if lines[0].startswith('BROKER SCHEMA'):
                lines.insert(1, f'CREATE COMPUTE MODULE {module_name}')
                fixed_content = '\n'.join(lines)
                fixes_applied.append("Added CREATE COMPUTE MODULE")
        else:
            # Verify module name is correct
            module_pattern = r'CREATE COMPUTE MODULE\s+(\S+)'
            match = re.search(module_pattern, fixed_content)
            if match:
                current_module_name = match.group(1)
                # Check if it's a placeholder or incorrect
                if '_SYSTEM___MSG_TYPE___FLOW_PROCESS___SYSTEM2___FLOW_TYPE' in current_module_name or current_module_name != module_name:
                    fixed_content = re.sub(
                        r'CREATE COMPUTE MODULE\s+\S+',
                        f'CREATE COMPUTE MODULE {module_name}',
                        fixed_content
                    )
                    fixes_applied.append(f"Fixed module name to {module_name}")
        
        # ============================================================
        # PART 2: Fix Required Procedures
        # ============================================================
        
        # Required procedure templates
        required_copy_headers = """CREATE PROCEDURE CopyMessageHeaders() BEGIN
\t\tDECLARE I INTEGER 1;
\t\tDECLARE J INTEGER;
\t\tSET J = CARDINALITY(InputRoot.*[]);
\t\tWHILE I < J DO
\t\t\tSET OutputRoot.*[I] = InputRoot.*[I];
\t\t\tSET I = I + 1;
\t\tEND WHILE;
\tEND;"""
        
        required_copy_entire = """CREATE PROCEDURE CopyEntireMessage() BEGIN
\t\tSET OutputRoot = InputRoot;
\tEND;"""
        
        # Check for CopyMessageHeaders procedure
        if 'CREATE PROCEDURE CopyMessageHeaders()' not in fixed_content:
            # Add before END MODULE; if it exists, or at the end
            if 'END MODULE;' in fixed_content:
                fixed_content = fixed_content.replace('END MODULE;', f'\n\n\t{required_copy_headers}\n\nEND MODULE;')
            else:
                fixed_content = fixed_content + f'\n\n\t{required_copy_headers}\n'
            fixes_applied.append("Added CopyMessageHeaders procedure")
        
        # Check for CopyEntireMessage procedure
        if 'CREATE PROCEDURE CopyEntireMessage()' not in fixed_content:
            # Add before END MODULE; if it exists, or at the end
            if 'END MODULE;' in fixed_content:
                fixed_content = fixed_content.replace('END MODULE;', f'\n\t{required_copy_entire}\n\nEND MODULE;')
            else:
                fixed_content = fixed_content + f'\n\n\t{required_copy_entire}\n'
            fixes_applied.append("Added CopyEntireMessage procedure")
        
        # ============================================================
        # PART 3: Fix END MODULE
        # ============================================================
        
        # Check for END MODULE; at the end
        if not fixed_content.strip().endswith('END MODULE;'):
            fixed_content = fixed_content.strip() + '\n\nEND MODULE;'
            fixes_applied.append("Added END MODULE;")
        
        # ============================================================
        # PART 4: Final Validation
        # ============================================================
        
        validation_errors = []
        
        # Check 1: Has BROKER SCHEMA
        if 'BROKER SCHEMA' not in fixed_content:
            validation_errors.append("Missing BROKER SCHEMA (auto-fix failed)")
        
        # Check 2: Has CREATE COMPUTE MODULE
        if 'CREATE COMPUTE MODULE' not in fixed_content:
            validation_errors.append("Missing CREATE COMPUTE MODULE (auto-fix failed)")
        
        # Check 3: Has CopyMessageHeaders
        if 'CREATE PROCEDURE CopyMessageHeaders()' not in fixed_content:
            validation_errors.append("Missing CopyMessageHeaders procedure (auto-fix failed)")
        
        # Check 4: Has CopyEntireMessage
        if 'CREATE PROCEDURE CopyEntireMessage()' not in fixed_content:
            validation_errors.append("Missing CopyEntireMessage procedure (auto-fix failed)")
        
        # Check 5: Ends with END MODULE;
        if not fixed_content.strip().endswith('END MODULE;'):
            validation_errors.append("Missing END MODULE; (auto-fix failed)")
        
        if validation_errors:
            print(f"      ⚠️ Validation errors after auto-fix: {validation_errors}")
        
        return fixed_content, fixes_applied
    
    
    # ============================================================================
    # PART 4: MAIN GENERATION LOOP
    # ============================================================================
    
    def generate_esql_files(self, vector_content: str, esql_template: Dict, 
                           msgflow_content: Dict, json_mappings: Dict,
                           output_dir: str = "output") -> Dict:
        """
        Main method to generate all ESQL files for a MessageFlow.
        
        Workflow:
        1. Load all inputs (template, msgflow, mappings, naming)
        2. Extract requirements (MessageFlow-first approach)
        3. For each module:
           - If event message: template copy
           - If business logic: template + LLM injection
           - Validate and auto-fix structure
           - Write to file
        4. Return generation results
        
        Args:
            vector_content: Business requirements from Vector DB
            esql_template: Dict with 'path' to ESQL_Template_Updated.ESQL
            msgflow_content: Dict with 'path' to .msgflow file
            json_mappings: Dict with 'path' to component_mapping.json
            output_dir: Output directory for ESQL files
        
        Returns:
            Dict with generation results and statistics
        """
        print("\n" + "="*70)
        print("🚀 ESQL GENERATION STARTED")
        print("="*70)
        
        # Reset statistics
        self.generation_stats = {
            'llm_calls': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'auto_fixes_applied': 0
        }
        
        try:
            # STEP 1: Load all inputs
            template_content, msgflow_xml, mappings_data, naming_data = self._load_inputs(
                esql_template, msgflow_content, json_mappings
            )
            
            # STEP 2: Extract requirements (MessageFlow-first)
            module_requirements = self._extract_requirements_from_sources(
                vector_content, msgflow_xml, mappings_data, naming_data
            )
            
            if not module_requirements:
                raise Exception("No ESQL modules found in requirements extraction")
            
            print(f"\n📋 Generating {len(module_requirements)} ESQL modules...")
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Track results
            successful_modules = []
            failed_modules = []
            
            # STEP 3: Generate each module
            for idx, module_req in enumerate(module_requirements, 1):
                module_name = module_req['name']
                module_type = module_req.get('type', 'compute')
                
                print(f"\n  [{idx}/{len(module_requirements)}] Processing: {module_name}")
                print(f"      Type: {module_type}")
                
                try:
                    # Determine generation method
                    if self._is_event_message_module(module_name):
                        # Template copy for event messages
                        esql_content = self._generate_event_message_esql(
                            module_name, naming_data, template_content
                        )
                        generation_method = "TEMPLATE_COPY"
                    else:
                        # Template + LLM for business logic
                        esql_content = self._generate_business_logic_esql(
                            module_req, naming_data, template_content, vector_content
                        )
                        generation_method = "LLM_GENERATION"
                    
                    # Final structure check
                    if not esql_content.strip().endswith('END MODULE;'):
                        print(f"      ❌ FAILED: Structure incomplete after generation")
                        failed_modules.append({
                            'name': module_name,
                            'error': 'Incomplete structure (missing END MODULE;)',
                            'type': module_type
                        })
                        self.generation_stats['failed_generations'] += 1
                        continue
                    
                    # Write to file
                    file_path = self._write_esql_file(module_name, esql_content, output_dir)
                    
                    # Record success
                    successful_modules.append({
                        'name': module_name,
                        'type': module_type,
                        'file_path': file_path,
                        'generation_method': generation_method,
                        'content_length': len(esql_content)
                    })
                    
                    self.generation_stats['successful_generations'] += 1
                    print(f"      ✅ SUCCESS: Written to {file_path}")
                
                except Exception as e:
                    print(f"      ❌ FAILED: {str(e)}")
                    failed_modules.append({
                        'name': module_name,
                        'error': str(e),
                        'type': module_type
                    })
                    self.generation_stats['failed_generations'] += 1
            
            # STEP 4: Generate results summary
            results = {
                'status': 'completed',
                'total_modules': len(module_requirements),
                'successful': len(successful_modules),
                'failed': len(failed_modules),
                'success_rate': f"{(len(successful_modules) / len(module_requirements) * 100):.1f}%",
                'successful_modules': successful_modules,
                'failed_modules': failed_modules,
                'output_directory': output_dir,
                'statistics': self.generation_stats,
                    # UI-compatible keys (add these)
                'llm_calls': self.generation_stats['llm_calls'],
                'llm_calls_made': self.generation_stats['llm_calls'],
                'files_generated': len(successful_modules),
                'total_files': len(successful_modules),
                'esql_files_generated': len(successful_modules),
                'generation_method': 'Hybrid: Template Copy + LLM Generation'

            }
            
            # Print summary
            print("\n" + "="*70)
            print("📊 GENERATION SUMMARY")
            print("="*70)
            print(f"✅ Successful: {results['successful']}/{results['total_modules']} ({results['success_rate']})")
            print(f"❌ Failed: {results['failed']}/{results['total_modules']}")
            print(f"🧠 LLM Calls: {self.generation_stats['llm_calls']}")
            print(f"🔧 Auto-fixes Applied: {self.generation_stats['auto_fixes_applied']}")
            print(f"📁 Output Directory: {output_dir}")
            
            if failed_modules:
                print("\n⚠️  Failed Modules:")
                for module in failed_modules:
                    print(f"   • {module['name']}: {module['error']}")
            
            print("="*70 + "\n")
            
            return results
        
        except Exception as e:
            print(f"\n❌ ESQL Generation Failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'total_modules': 0,
                'successful': 0,
                'failed': 0,
                'statistics': self.generation_stats,

                        # UI-compatible keys for error case
                'llm_calls': self.generation_stats.get('llm_calls', 0),
                'llm_calls_made': self.generation_stats.get('llm_calls', 0),
                'files_generated': 0,
                'total_files': 0,
                'esql_files_generated': 0,
                'generation_method': 'Failed'
            }
    
    
    # ============================================================================
    # PART 5: FILE OPERATIONS
    # ============================================================================
    
    def _write_esql_file(self, module_name: str, esql_content: str, output_dir: str) -> str:
        """
        Write ESQL content to file.
        
        Args:
            module_name: Module name (becomes filename)
            esql_content: Complete ESQL code
            output_dir: Output directory
        
        Returns:
            Full path to written file
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename
        filename = f"{module_name}.esql"
        file_path = os.path.join(output_dir, filename)
        
        # Write file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(esql_content)
        
        return file_path


# ============================================================================
# HELPER FUNCTIONS FOR INTEGRATION
# ============================================================================

def create_esql_generator(groq_api_key: Optional[str] = None, 
                         groq_model: str = "llama-3.1-70b-versatile") -> ESQLGenerator:
    """
    Factory function to create ESQLGenerator instance.
    
    Args:
        groq_api_key: Groq API key (optional, uses environment variable if not provided)
        groq_model: LLM model to use
    
    Returns:
        Configured ESQLGenerator instance
    """
    return ESQLGenerator(groq_api_key=groq_api_key, groq_model=groq_model)


# ============================================================================
# MAIN EXECUTION (for testing)
# ============================================================================

def main():
    """
    Test execution of ESQL Generator.
    This is for standalone testing only.
    """
    import sys
    
    print("="*70)
    print("ESQL Generator - Standalone Test")
    print("="*70)
    
    # Check for required files
    required_files = [
        'ESQL_Template_Updated.ESQL',
        'naming_convention.json',
        'component_mapping.json'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"\n❌ Missing required files: {missing_files}")
        print("Please ensure all required files are in the current directory.")
        sys.exit(1)
    
    # Check for MessageFlow
    import glob
    msgflow_files = glob.glob("output/**/*.msgflow", recursive=True)
    if not msgflow_files:
        print("\n❌ No .msgflow file found in output directory")
        print("Please generate MessageFlow first.")
        sys.exit(1)
    
    print(f"\n✅ Found MessageFlow: {msgflow_files[0]}")
    
    # Check for Vector DB content (simulated)
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
        # Create generator
        generator = create_esql_generator()
        
        # Generate ESQL files
        results = generator.generate_esql_files(
            vector_content=vector_content,
            esql_template={'path': 'ESQL_Template_Updated.ESQL'},
            msgflow_content={'path': msgflow_files[0]},
            json_mappings={'path': 'component_mapping.json'},
            output_dir='output/esql'
        )
        
        # Print results
        if results['status'] == 'completed':
            print("\n🎉 ESQL Generation Completed Successfully!")
            print(f"📁 Check output in: {results['output_directory']}")
        else:
            print(f"\n❌ ESQL Generation Failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

