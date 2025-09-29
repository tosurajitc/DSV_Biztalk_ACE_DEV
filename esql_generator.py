"""
Enhanced ESQL Generator - 100% LLM Based with Token Management
NO HARDCODED FALLBACKS - Pure LLM Generation Only
"""

import os
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import streamlit as st
from groq import Groq
class ESQLGenerator:
    """
    100% LLM-based ESQL generator with intelligent token management
    CRITICAL: NO hardcoded ESQL generation - pure LLM only
    """
    
    def __init__(self, groq_api_key: str = None):
        """Initialize with Groq LLM client and required attributes"""
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY must be provided or set in environment")
        
        self.llm = Groq(api_key=self.groq_api_key)
        self.groq_client = Groq(api_key=self.groq_api_key)
        self.groq_model = os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')

        self.max_tokens_per_request = 8192
        self.estimated_chars_per_token = 3.2  
        self.max_total_tokens = 32768
        
        # ✅ ADD: Initialize missing attributes
        self.output_dir = None  # Will be set by generate_esql_files method
        self.llm_calls_count = 0
        self.esql_modules = []
        self.processing_results = {}
        # ENHANCED: Valid data types for constraint validation
        self.valid_data_types = {
            'BOOLEAN', 'INTEGER', 'DECIMAL', 'FLOAT', 'CHARACTER',
            'BIT', 'BLOB', 'DATE', 'TIME', 'TIMESTAMP', 'REFERENCE', 'ROW'
        }
        
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        return len(text) // self.avg_chars_per_token
    
    def chunk_input_data(self, input_data: Dict) -> List[Dict]:
        """
        Intelligently chunk input data to stay under token limits
        """
        chunks = []
        
        # Calculate base system prompt tokens
        system_prompt = self._get_system_prompt()
        base_tokens = self.estimate_tokens(system_prompt)
        
        # Reserve tokens for output and safety margin
        available_input_tokens = self.max_total_tokens - base_tokens - self.max_tokens_per_request - 1000
        
        print(f"🧮 Token Budget: Total={self.max_total_tokens}, Base={base_tokens}, Available={available_input_tokens}")
        
        # Priority order for data inclusion
        priority_data = [
            ('esql_template', input_data.get('esql_template', {})),
            ('msgflow_structure', input_data.get('msgflow_content', {})),
            ('business_requirements', input_data.get('pdf_content', {})),
            ('component_mappings', input_data.get('json_mappings', {}))
        ]
        
        current_chunk = {}
        current_tokens = 0
        
        for data_type, data_content in priority_data:
            data_str = json.dumps(data_content, indent=2) if isinstance(data_content, dict) else str(data_content)
            data_tokens = self.estimate_tokens(data_str)
            
            if current_tokens + data_tokens <= available_input_tokens:
                current_chunk[data_type] = data_content
                current_tokens += data_tokens
                print(f"✅ Added {data_type}: {data_tokens} tokens (total: {current_tokens})")
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    print(f"📦 Created chunk with {current_tokens} tokens")
                
                # Start new chunk
                current_chunk = {data_type: data_content}
                current_tokens = data_tokens
                print(f"🆕 Started new chunk with {data_type}: {data_tokens} tokens")
        
        if current_chunk:
            chunks.append(current_chunk)
            print(f"📦 Final chunk with {current_tokens} tokens")
        
        print(f"🔄 Created {len(chunks)} chunks for processing")
        return chunks
    
    def _get_system_prompt(self) -> str:
        """
        Enhanced system prompt with all requirements
        """
        return """You are an expert IBM ACE ESQL developer specializing in generating production-ready ESQL modules. You MUST follow these CRITICAL requirements exactly:
## ⚠️ CRITICAL DATA TYPE RESTRICTIONS (READ FIRST) ⚠️

### APPROVED DATA TYPES ONLY - NO EXCEPTIONS:
You MUST ONLY use these data types in ALL DECLARE statements:
✅ BOOLEAN, INTEGER, DECIMAL, FLOAT, CHARACTER
✅ BIT, BLOB, DATE, TIME, TIMESTAMP, REFERENCE, ROW

### ABSOLUTELY FORBIDDEN DATA TYPES - NEVER USE:
❌ XML - Use REFERENCE TO InputRoot.XMLNSC instead
❌ RECORD - Use REFERENCE TO instead  
❌ STRING - Use CHARACTER instead
❌ VARCHAR - Use CHARACTER instead
❌ JSON - Use REFERENCE TO InputRoot.JSON instead
❌ Database - Use REFERENCE TO instead

### XML PROCESSING RULE:
Instead of: DECLARE xmlData XML;
Use this: DECLARE xmlRef REFERENCE TO InputRoot.XMLNSC;

## MANDATORY STRUCTURAL REQUIREMENTS

### 0. MODULE NAME RULES (CRITICAL):
- CREATE COMPUTE MODULE statement must use module name WITHOUT .esql extension
- Example: CREATE COMPUTE MODULE AzureBlob_To_CDM_Document  (CORRECT)
- Example: CREATE COMPUTE MODULE AzureBlob_To_CDM_Document.esql  (WRONG)
- File extension (.esql) is handled separately, NOT in the module declaration
- Do not call any procedures directly in the ESQL file

### 1. InputRoot/OutputRoot Rules (CRITICAL):
- **InputRoot**: READ-ONLY - NEVER modify InputRoot directly
- **OutputRoot**: WRITABLE - Always use OutputRoot for modifications
- Always start message processing with: SET OutputRoot = InputRoot; 

### 2. MANDATORY PROCEDURES (MUST BE INCLUDED EXACTLY):
Every ESQL file MUST include these two procedures at the bottom

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


### FORBIDDEN ELEMENTS:
- NO comments starting with "--" (e.g., "-- Declare variables")
- NO lines starting with "esql"
- NO "@" symbols anywhere
- NO code block markers
- NO custom SQLEXCEPTION handlers in procedures
- NO direct procedure calls using CALL statements
- NO CALL functionName() patterns - use SET statements instead

## DATA TYPE RESTRICTIONS (CRITICAL):
ONLY use these approved data types in DECLARE statements:
- BOOLEAN, INTEGER, DECIMAL, FLOAT, CHARACTER, BIT, BLOB, DATE, TIME, TIMESTAMP, REFERENCE, ROW

NEVER use these forbidden data types:
- XML, RECORD, STRING, VARCHAR, Database (use REFERENCE TO InputRoot.XMLNSC for XML processing)
- For text data: use CHARACTER instead of STRING or VARCHAR
- For XML processing: use REFERENCE TO InputRoot.XMLNSC instead of XML type

### 4. TEMPLATE STRUCTURE (MUST BE CUSTOMIZED BASED ON BUSINESS FLOW):
- Update dataInfo.mainIdentifier XPath based on business entities
- Populate customReference1-4 based on database requirements
- Populate customProperty1-4 based on transformation requirements
- Remove unused sourceInfo/targetInfo fields if not required

## OUTPUT REQUIREMENTS:
- Generate COMPLETE ESQL modules with NO truncation
- Include ALL mandatory procedures exactly as specified
- NO explanatory text outside the ESQL code
- Follow exact syntax and indentation patterns
- Module names MUST match MessageFlow node names exactly

## CRITICAL: 100% LLM GENERATION
- NO hardcoded ESQL patterns
- NO template substitution fallbacks
- Generate everything through LLM reasoning
- Adapt to specific business requirements dynamically"""




    def _get_chunk_analysis_prompt(self, chunk_data: Dict, chunk_index: int) -> str:
        return f"""Analyze this data chunk ({chunk_index + 1}) to extract ESQL requirements:

    ## DATA CHUNK:
    {json.dumps(chunk_data, indent=2)}

    ## ANALYSIS REQUIREMENTS:
    Extract and return JSON with:
    1. "esql_modules": List of required ESQL modules with names, purposes, and types
    2. "business_logic": Business logic requirements mapped to specific modules
    3. "database_operations": List of database procedures and operations mentioned
    4. "transformations": List of data transformation requirements  
    5. "message_structure": Input/output message structure and XPath details
    6. "customizations": Specific customizations needed for template

    ## SPECIFIC EXTRACTION FOCUS:
    - **Database Operations**: Extract ALL stored procedures, database lookups, and conditional logic patterns
    - **XPath Expressions**: Extract message field paths and data extraction patterns
    - **Business Rules**: Extract conditional logic, validation rules, and processing requirements
    - **Module Distribution**: Map comprehensive business logic to _Compute modules only
    - **Event Capture**: Map metadata extraction to _InputEventMessage/_OutputEventMessage modules

    ## REQUIRED BUSINESS LOGIC DETAILS:
    For each database operation found, include:
    - Procedure name (e.g., sp_Shipment_GetIdBySSN)
    - Input parameters and XPath sources
    - Conditional logic (IF/THEN patterns)
    - Result handling

    For message processing, include:
    - Source message format and structure
    - Target message format requirements  
    - Field mapping and transformation rules
    - Error handling requirements

    Return valid JSON only with ALL required fields:"""




    def _determine_module_type(self, module_name: str) -> str:
        """Determine module type based on naming pattern - returns template-compatible values"""
        if module_name.endswith('_InputEventMessage'):
            return 'input_event'
        elif module_name.endswith('_Compute'):
            return 'compute'  # THE business logic module
        elif module_name.endswith('_AfterEnrichment'):
            return 'processing'
        elif module_name.endswith('_OutputEventMessage'):
            return 'input_event'  # Same template as input_event
        elif module_name.endswith('_AfterEventMsg'):
            return 'processing'  # Same template as processing
        elif module_name.endswith('_Failure'):
            return 'failure'
        else:
            return 'compute'  # Default to compute template
    


    def _get_esql_generation_prompt(self, module_requirements: Dict, template_info: Dict) -> str:
        module_name = module_requirements.get('name', 'UnknownModule')
        if module_name.lower().endswith('.esql'):
            module_name = module_name[:-5]

        # Determine module type using our new method
        module_type = self._determine_module_type(module_name)
        purpose = module_requirements.get('purpose', 'Message processing')
        
        # Base prompt structure
        prompt = f"""Generate a complete ESQL module for IBM ACE:

    ## MODULE SPECIFICATION:
    - **Name**: {module_name}
    - **Purpose**: {purpose}
    - **Type**: {module_type}

    """
        
        # Add module-type specific content
        if module_type == 'COMPUTE':
            prompt += self._get_compute_module_prompt(module_requirements, module_name)
        elif module_type in ['INPUT_EVENT', 'OUTPUT_EVENT']:
            prompt += self._get_event_module_prompt(module_requirements, module_type)
        elif module_type in ['POST_ENRICHMENT', 'POST_EVENT']:
            prompt += self._get_processing_module_prompt(module_requirements, module_type)
        elif module_type == 'FAILURE':
            prompt += self._get_failure_module_prompt(module_requirements)
        else:
            prompt += self._get_default_module_prompt(module_requirements)
        
        return prompt


    def _generate_esql_from_template(self, template_type: str, module_name: str, business_data: Dict) -> str:
        """
        Generate complete ESQL code from template with business data injection
        Single self-sufficient method for 1000+ different flows - NO hardcoded values
        
        Args:
            template_type: 'input_event', 'compute', 'processing', or 'failure'  
            module_name: Name of the ESQL module
            business_data: Extracted business data for placeholder replacement
        
        Returns:
            str: Complete ESQL code ready for use
        """
        
        # Step 1: Load naming convention from JSON - REQUIRED for dynamic naming
        try:
            with open("naming_convention.json", 'r', encoding='utf-8') as f:
                naming_data = json.load(f)
        except FileNotFoundError:
            raise Exception("naming_convention.json not found - required for dynamic naming across 1000+ flows")
        
        # Step 2: Read ESQL template file - REQUIRED for template sections
        try:
            with open("ESQL_Template_Updated.ESQL", 'r', encoding='utf-8') as f:
                template_content = f.read()
        except FileNotFoundError:
            raise Exception("ESQL_Template_Updated.ESQL not found - required for template generation")

        # Step 3: Extract correct template section based on module type - NO hardcoded sections
        section_markers = {
            'input_event': ['INPUT EVENT MESSAGE TEMPLATE - METADATA ONLY', 'COMPUTE TEMPLATE - FULL BUSINESS LOGIC'],
            'compute': ['COMPUTE TEMPLATE - FULL BUSINESS LOGIC', 'PROCESSING TEMPLATE - VALIDATION AND ROUTING ONLY'],
            'processing': ['PROCESSING TEMPLATE - VALIDATION AND ROUTING ONLY', 'FAILURE/ERROR HANDLING TEMPLATE'],
            'failure': ['FAILURE/ERROR HANDLING TEMPLATE', None]  # Last section
        }
        
        # Map legacy types to new structure
        if template_type == 'generic':
            template_type = 'compute'
        
        if template_type not in section_markers:
            raise Exception(f"Unknown template_type: {template_type}. Valid types: {list(section_markers.keys())}")
        
        start_marker, end_marker = section_markers[template_type]
        
        # Find template section boundaries
        start_pos = template_content.find(start_marker)
        if start_pos == -1:
            raise Exception(f"Template section '{start_marker}' not found in ESQL_Template_Updated.ESQL")
        
        if end_marker:
            end_pos = template_content.find(end_marker, start_pos + len(start_marker))
            if end_pos == -1:
                end_pos = len(template_content)
            else:
                # CRITICAL FIX: Include END MODULE; from current section
                current_section = template_content[start_pos:end_pos]
                end_module_pos = current_section.rfind('END MODULE;')
                if end_module_pos != -1:
                    # Adjust end_pos to include the END MODULE; from current section
                    end_pos = start_pos + end_module_pos + len('END MODULE;')
        else:
            end_pos = len(template_content)
        
        # Extract section and find ESQL code start
        section_content = template_content[start_pos:end_pos]
        lines = section_content.split('\n')
        
        # Find actual ESQL code (starts with BROKER SCHEMA MODULE)
        esql_start_line = None
        for i, line in enumerate(lines):
            if line.strip().startswith('BROKER SCHEMA MODULE'):
                esql_start_line = i
                break

        if esql_start_line is None:
            # Fallback to CREATE COMPUTE MODULE if BROKER SCHEMA not found
            for i, line in enumerate(lines):
                if line.strip().startswith('CREATE COMPUTE MODULE'):
                    esql_start_line = i
                    break

        if esql_start_line is None:
            raise Exception(f"No 'BROKER SCHEMA MODULE' or 'CREATE COMPUTE MODULE' found in {template_type} template section")

        selected_template = '\n'.join(lines[esql_start_line:]).strip()
        
        # Step 4: Extract naming components from naming_convention.json - FULLY DYNAMIC
        project_naming = naming_data.get('project_naming', {})
        if not project_naming:
            raise Exception("project_naming section missing from naming_convention.json")
        
        ace_app_name = project_naming.get('ace_application_name', '')
        message_flow_name = project_naming.get('message_flow_name', '')
        
        if not ace_app_name or not message_flow_name:
            raise Exception("ace_application_name and message_flow_name required in naming_convention.json")
        
        # Parse naming components dynamically from naming convention
        app_parts = ace_app_name.split('_')
        flow_parts = message_flow_name.split('_')
        
        if len(app_parts) < 2 or len(flow_parts) < 3:
            raise Exception(f"Invalid naming format: ace_application_name='{ace_app_name}', message_flow_name='{message_flow_name}'")
        
        system1 = app_parts[0]          # Extract from ace_application_name
        system2 = app_parts[1]          # Extract from ace_application_name
        flow_process = flow_parts[1]    # Extract from message_flow_name
        msg_type = flow_parts[2]        # Extract from message_flow_name
        flow_type = flow_parts[-1]      # Extract from message_flow_name
        
        # Step 5: Replace template placeholders with dynamic values
        placeholder = "_SYSTEM___MSG_TYPE___FLOW_PROCESS___SYSTEM2___FLOW_TYPE"
        dynamic_name = f"{system1}_{msg_type}_{flow_process}_{system2}_{flow_type}"
        
        # Replace main placeholder with dynamic naming
        esql_code = selected_template.replace(placeholder, dynamic_name)
        
        # Replace with specific module name if provided and different
        if module_name and module_name != dynamic_name:
            # Update module name in CREATE COMPUTE MODULE statement
            pattern = r'(CREATE COMPUTE MODULE\s+)[\w_]+(\s*\n)'
            esql_code = re.sub(pattern, rf'\1{module_name}\2', esql_code)
        
        # Step 6: Inject business logic based on template type and Vector DB data
        if template_type == 'compute':
            # Inject comprehensive business logic for compute modules
            database_operations = business_data.get('database_operations', [])
            transformations = business_data.get('transformations', [])
            xpath_mappings = business_data.get('xpath_mappings', [])
            
            # Replace business data placeholders with Vector DB extracted paths
            if xpath_mappings:
                # Replace business entity and identifier placeholders
                for xpath in xpath_mappings[:5]:  # Limit for code size
                    if isinstance(xpath, str) and 'mainIdentifier' not in esql_code:
                        esql_code = esql_code.replace('_BUSINESS_ENTITY_.*:_BUSINESS_IDENTIFIER_', xpath)
                        break
            
            # Inject database operations as comments (no actual stored procedure calls)
            if database_operations:
                db_comment_block = "\n\t\t-- Database Operations from Vector DB:\n"
                for db_op in database_operations[:3]:
                    if isinstance(db_op, dict):
                        proc_name = db_op.get('procedure', 'unknown')
                        params = db_op.get('parameters', [])
                        db_comment_block += f"\t\t-- {proc_name}({', '.join(params)})\n"
                
                # Insert database operations comments
                esql_code = esql_code.replace(
                    "-- ✅ DATABASE OPERATIONS: Business enrichment and lookups",
                    f"-- ✅ DATABASE OPERATIONS: Business enrichment and lookups{db_comment_block}"
                )
            
            # Inject transformation logic as comments
            if transformations:
                transform_comment_block = "\n\t\t-- Transformations from Vector DB:\n"
                for transform in transformations[:3]:
                    if isinstance(transform, dict):
                        transform_type = transform.get('type', 'unknown')
                        source = transform.get('source', 'unknown')
                        target = transform.get('target', 'unknown')
                        transform_comment_block += f"\t\t-- {transform_type}: {source} -> {target}\n"
                
                # Insert transformation comments
                esql_code = esql_code.replace(
                    "-- ✅ BUSINESS TRANSFORMATIONS: Message format conversions",
                    f"-- ✅ BUSINESS TRANSFORMATIONS: Message format conversions{transform_comment_block}"
                )
        
        elif template_type == 'processing':
            # Inject validation and routing logic for processing modules
            validation_rules = business_data.get('validation_rules', [])
            routing_decisions = business_data.get('routing_decisions', [])
            
            if validation_rules:
                validation_block = "\n\t\t-- Validation Rules from Vector DB:\n"
                for rule in validation_rules[:3]:
                    validation_block += f"\t\t-- Validate: {rule}\n"
                
                esql_code = esql_code.replace(
                    "-- ✅ VALIDATION RULES: Message validation and routing decisions",
                    f"-- ✅ VALIDATION RULES: Message validation and routing decisions{validation_block}"
                )
            
            if routing_decisions:
                routing_block = "\n\t\t-- Routing Logic from Vector DB:\n"
                for decision in routing_decisions[:3]:
                    routing_block += f"\t\t-- Route: {decision}\n"
                
                esql_code = esql_code.replace(
                    "-- ✅ ROUTING LOGIC: Conditional routing based on message content",
                    f"-- ✅ ROUTING LOGIC: Conditional routing based on message content{routing_block}"
                )
        
        # Step 7: Auto-fix BROKER SCHEMA if missing (APPLIES TO ALL TYPES)
        esql_code = self._ensure_broker_schema(esql_code, module_name)

        # Step 8: Validate final structure with BROKER SCHEMA requirement (APPLIES TO ALL TYPES)
        lines = esql_code.strip().split('\n')

        # Check Line 1: Must start with BROKER SCHEMA
        if not lines[0].strip().startswith('BROKER SCHEMA'):
            raise Exception(f"Generated ESQL must start with 'BROKER SCHEMA' on line 1")

        # Check Line 2: Must have CREATE COMPUTE MODULE
        if len(lines) < 2 or not lines[1].strip().startswith('CREATE COMPUTE MODULE'):
            raise Exception(f"Generated ESQL must have 'CREATE COMPUTE MODULE' on line 2")

        # Check ending
        if not esql_code.strip().endswith('END MODULE;'):
            raise Exception(f"Generated ESQL must end with 'END MODULE;'")

        return esql_code
    


    def _get_compute_module_prompt(self, module_req: Dict, module_name: str) -> str:
        """
        Generate business logic code that will be DIRECTLY inserted into template
        NEW STRATEGY: Generate the business code separately, clearly marked for insertion
        """
        
        business_logic = module_req.get('business_logic', {})
        database_operations = module_req.get('database_operations', [])
        transformations = module_req.get('transformations', [])
        message_structure = module_req.get('message_structure', {})
        
        # Build complete business logic as a single code block
        business_code_block = ""
        
        # Add database operations
        if database_operations:
            business_code_block += "\n\t\t-- DATABASE ENRICHMENT OPERATIONS\n"
            for idx, db_op in enumerate(database_operations[:6], 1):
                procedure = db_op.get('procedure', f'sp_Operation{idx}')
                description = db_op.get('description', 'Database lookup')
                business_code_block += f"\t\t-- {description}\n"
                business_code_block += f"\t\tDECLARE result{idx} CHARACTER;\n"
                business_code_block += f"\t\tSET result{idx} = PASSTHRU(\n"
                business_code_block += f"\t\t\t'CALL {procedure}(?, ?)',\n"
                business_code_block += f"\t\t\tInputRoot.XMLNSC.*:Header.*:CompanyCode,\n"
                business_code_block += f"\t\t\tInputRoot.XMLNSC.*:Header.*:CountryCode\n"
                business_code_block += f"\t\t);\n"
                business_code_block += f"\t\tSET OutputRoot.XMLNSC.*:Result{idx} = result{idx};\n\n"
        
        # Add transformations
        if transformations:
            business_code_block += "\n\t\t-- MESSAGE TRANSFORMATIONS\n"
            for idx, transform in enumerate(transformations[:5], 1):
                source = transform.get('source_field', f'SourceField{idx}')
                target = transform.get('target_field', f'TargetField{idx}')
                business_code_block += f"\t\t-- Transform: {source} -> {target}\n"
                business_code_block += f"\t\tSET OutputRoot.XMLNSC.*:{target} = InputRoot.XMLNSC.*:{source};\n\n"
        
        # If no business logic, add passthrough
        if not database_operations and not transformations:
            business_code_block = "\n\t\t-- Direct passthrough - no business logic required\n\t\tSET OutputRoot = InputRoot;\n"


        print(f"\n🔍 DEBUG: _get_compute_module_prompt for {module_name}")
        print(f"   📊 database_operations length: {len(database_operations)}")
        print(f"   📊 transformations length: {len(transformations)}")
        print(f"   📊 business_code_block length: {len(business_code_block)}")
        print(f"   📝 business_code_block preview (first 200 chars):")
        print(f"      {business_code_block[:200]}")    
        
        prompt = f"""
            Find the marker line in the template:
            -- [[[INSERT_BUSINESS_LOGIC_HERE]]]

            Replace that ONE line with this exact code:
            {business_code_block}

            RULES:
            1. Only replace the marker line
            2. Do not modify anything else
            3. Return the complete template with marker replaced
            """
        
        return prompt


    def _get_event_module_prompt(self, module_req: Dict, module_type: str) -> str:
        """Generate prompt instructing LLM to use template for Event module with metadata only"""
        
        # Extract EVENT-SPECIFIC business data (metadata and correlation only)
        business_logic = module_req.get('business_logic', {})
        message_structure = module_req.get('message_structure', {})
        
        # Event-specific business data
        event_business_logic = {
            **business_logic,
            'module_type': module_type,
            'purpose': 'event_capture_only',
            'metadata_extraction': True,
            'business_processing': False,
            'interface_name': message_structure.get('interface_name', 'DYNAMIC_INTERFACE'),
            'correlation_fields': message_structure.get('correlation_fields', ['correlationId']),
            'metadata_fields': message_structure.get('metadata_fields', ['messageId', 'timestamp'])
        }
        
        # Generate dynamic module name
        interface_name = message_structure.get('interface_name', 'DYNAMIC_INTERFACE')
        event_module_name = f"{interface_name}InputEventMessage"
        
        return f"""
        ### EVENT MODULE - METADATA CAPTURE ONLY:
        - **Template Type**: 'input_event'
        - **Module Name**: {event_module_name} (NO .esql extension)
        - **Purpose**: Event capture with metadata only - NO business logic

        ### EVENT-SPECIFIC Business Data Extraction (Metadata only):
        {json.dumps(event_business_logic, indent=2)}

        ### Message Structure for Event Capture (Dynamic for 1000+ flows):
        {json.dumps(message_structure, indent=2)}


    ### EVENT CAPTURE RESPONSIBILITIES:
    - Interface: {event_business_logic.get('interface_name', 'DYNAMIC_INTERFACE')}
    - Event type: {module_type}
    - Correlation fields: {event_business_logic.get('correlation_fields', [])}
    - Metadata fields: {event_business_logic.get('metadata_fields', [])}
    - NO database operations: {event_business_logic.get('business_processing', False)}

    ### STRICT EVENT COMPLIANCE:
    - ONLY metadata extraction from headers
    - NO business data from message payload
    - NO database operations or transformations
    - Dynamic for 1000+ different flows  
    - Preserve template structure while enhancing
    """


    def _get_processing_module_prompt(self, module_req: Dict, module_type: str) -> str:
        """Generate prompt instructing LLM to use template for Processing module with validation logic"""
        
        # Extract VALIDATION-SPECIFIC business data (routing and validation only)
        business_logic = module_req.get('business_logic', {})
        message_structure = module_req.get('message_structure', {})

        module_name = module_req.get('name', f"{message_structure.get('interface_name', 'DYNAMIC_INTERFACE')}{module_type}Module")
        # Processing-specific business data
        processing_business_logic = {
            **business_logic,
            'module_type': module_type,
            'purpose': 'light_processing',
            'validation_only': True,
            'routing_logic': True,
            'interface_name': message_structure.get('interface_name', 'DYNAMIC_INTERFACE'),
            'validation_rules': business_logic.get('validation_rules', []),
            'routing_decisions': business_logic.get('routing_logic', [])
        }
        
        return f"""

        ### PROCESSING MODULE - VALIDATION AND ROUTING ONLY:
        - **Template Type**: 'generic'
        - **Module Name**: {module_name} (NO .esql extension)
        - **Purpose**: Light processing with validation and routing logic

        ### VALIDATION-SPECIFIC Business Data Extraction (Routing/validation only):
        {json.dumps(processing_business_logic, indent=2)}

        ### Message Structure for Processing (Dynamic for 1000+ flows):
        {json.dumps(message_structure, indent=2)}

    ### PROCESSING MODULE RESPONSIBILITIES:
    - Interface: {processing_business_logic.get('interface_name', 'DYNAMIC_INTERFACE')}
    - Processing type: {module_type}
    - Validation rules: {len(processing_business_logic.get('validation_rules', []))}
    - Routing decisions: {len(processing_business_logic.get('routing_decisions', []))}
    - Lightweight operations only

    ### PROCESSING CONSTRAINTS:
    - NO database operations for business enrichment
    - NO business transformations
    - VALIDATION and ROUTING focus only
    - Dynamic for 1000+ different flows  
    - Preserve template structure while enhancing
    """




    def _get_failure_module_prompt(self, module_req: Dict) -> str:
        """Generate prompt instructing LLM to use template for Failure module with error handling"""
        
        # Extract ERROR-SPECIFIC business data (error handling and debugging only)
        business_logic = module_req.get('business_logic', {})
        message_structure = module_req.get('message_structure', {})
        
        # Error-specific business data
        error_business_logic = {
            **business_logic,
            'module_type': 'FAILURE',
            'purpose': 'error_handling',
            'error_capture': True,
            'exception_processing': True,
            'interface_name': message_structure.get('interface_name', 'DYNAMIC_INTERFACE'),
            'error_queue': message_structure.get('error_queue', f"{message_structure.get('interface_name', 'DYNAMIC')}.ERROR.QUEUE"),
            'fault_tolerance_level': business_logic.get('fault_tolerance', 'STANDARD')
        }
        
        # Generate dynamic module name
        interface_name = message_structure.get('interface_name', 'DYNAMIC_INTERFACE')
        failure_module_name = f"{interface_name}FailureEventMessage"
        
        return f"""

        ### FAILURE MODULE - ERROR HANDLING ONLY:
        - **Template Type**: 'failure'
        - **Module Name**: {failure_module_name} (NO .esql extension)
        - **Purpose**: Error handling and exception processing

        ### ERROR-SPECIFIC Business Data Extraction (Error handling only):
        {json.dumps(error_business_logic, indent=2)}

        ### Message Structure for Error Handling (Dynamic for 1000+ flows):
        {json.dumps(message_structure, indent=2)}


    ### ERROR HANDLING RESPONSIBILITIES:
    - Interface: {error_business_logic.get('interface_name', 'DYNAMIC_INTERFACE')}
    - Error queue: {error_business_logic.get('error_queue', 'DYNAMIC.ERROR.QUEUE')}
    - Fault tolerance: {error_business_logic.get('fault_tolerance_level', 'STANDARD')}
    - Exception processing with GetFaultDetailAsString function
    - Production support debugging assistance

    ### ERROR HANDLING CONSTRAINTS:
    - NO database operations for business enrichment
    - NO business transformations   
    - ERROR capture and logging focus only
    - Dynamic for 1000+ different flows  
    - Preserve template structure while enhancing
    
    """



    def _get_default_module_prompt(self, module_req: Dict) -> str:
        """Generate prompt instructing LLM to use template for Default module with minimal processing"""
        
        # Extract MINIMAL business data (basic processing only)
        business_logic = module_req.get('business_logic', {})
        message_structure = module_req.get('message_structure', {})
        
        # Default-specific business data
        default_business_logic = {
            **business_logic,
            'module_type': 'DEFAULT',
            'purpose': 'basic_processing',
            'basic_processing_type': True,
            'interface_name': message_structure.get('interface_name', 'DYNAMIC_INTERFACE'),
            'lightweight_operations': True,
            'standard_message_processing': True
        }
        
        # Extract module name
        module_name = module_req.get('name', f"{message_structure.get('interface_name', 'DYNAMIC_INTERFACE')}DefaultModule")
        
        return f"""

        ### DEFAULT MODULE - MINIMAL PROCESSING:
        - **Template Type**: 'generic'
        - **Module Name**: {module_name} (NO .esql extension)
        - **Purpose**: Basic message processing and forwarding

        ### MINIMAL Business Data Extraction (Basic processing only):
        {json.dumps(default_business_logic, indent=2)}

        ### Message Structure for Basic Processing (Dynamic for 1000+ flows):
        {json.dumps(message_structure, indent=2)}


    ### DEFAULT MODULE RESPONSIBILITIES:
    - Interface: {default_business_logic.get('interface_name', 'DYNAMIC_INTERFACE')}
    - Processing: Basic message handling and forwarding
    - Operations: Lightweight standard processing
    - Fallback for unknown/basic module types
    - Template-based generation with minimal complexity

    ### DEFAULT CONSTRAINTS:
    - NO database operations
    - NO business transformations
    - BASIC processing only
    - Dynamic for 1000+ different flows  
    - Preserve template structure while enhancing
    """




    def analyze_requirements_with_chunking(self, input_data: Dict) -> Dict:
        """
        Analyze requirements using chunked approach - 100% LLM
        """
        print("🔍 Starting LLM-based requirements analysis with chunking...")
        
        # Chunk the input data
        data_chunks = self.chunk_input_data(input_data)
        
        combined_requirements = {
            'esql_modules': [],
            'business_logic': {},
            'database_operations': [],
            'transformations': [],
            'message_structure': {},
            'customizations': {}
        }
        

        print("🔍 DEBUG: Global business logic extracted:")
        print(json.dumps(combined_requirements.get('business_logic', {}), indent=2))
        print("🔍 DEBUG: Database operations extracted:")  
        print(json.dumps(combined_requirements.get('database_operations', []), indent=2))


        # Analyze each chunk with LLM
        for i, chunk in enumerate(data_chunks):
            print(f"📊 Analyzing chunk {i + 1}/{len(data_chunks)} with LLM...")
            
            try:
                chunk_analysis = self._llm_analyze_chunk(chunk, i)
                
                # Merge results
                if 'esql_modules' in chunk_analysis:
                    combined_requirements['esql_modules'].extend(chunk_analysis['esql_modules'])
                
                if 'business_logic' in chunk_analysis:
                    combined_requirements['business_logic'].update(chunk_analysis['business_logic'])
                
                if 'database_operations' in chunk_analysis:
                    combined_requirements['database_operations'].extend(chunk_analysis['database_operations'])
                
                if 'transformations' in chunk_analysis:
                    combined_requirements['transformations'].extend(chunk_analysis['transformations'])
                
                if 'message_structure' in chunk_analysis:
                    combined_requirements['message_structure'].update(chunk_analysis['message_structure'])
                
                if 'customizations' in chunk_analysis:
                    combined_requirements['customizations'].update(chunk_analysis['customizations'])
                
                print(f"✅ Chunk {i + 1} analysis complete")
                
            except Exception as e:
                print(f"⚠️ Chunk {i + 1} analysis failed: {e}")
                continue
        
        # Deduplicate modules
        unique_modules = []
        seen_names = set()
        for module in combined_requirements['esql_modules']:
            name = module.get('name', '')
            if name and name not in seen_names:
                unique_modules.append(module)
                seen_names.add(name)
        
        combined_requirements['esql_modules'] = unique_modules
        
        print(f"🎯 Analysis complete: {len(unique_modules)} unique ESQL modules identified")
        return combined_requirements
    



    def _llm_analyze_chunk(self, chunk_data: Dict, chunk_index: int) -> Dict:
        """LLM analysis of a single chunk"""
        if not self.groq_client:
            raise Exception("LLM client not available")
        
        try:
            prompt = self._get_chunk_analysis_prompt(chunk_data, chunk_index)
            
            # DEBUG: Print what we're sending to LLM
            print(f"🔍 DEBUG: Chunk {chunk_index} data keys: {list(chunk_data.keys())}")
            print(f"🔍 DEBUG: Chunk {chunk_index} prompt length: {len(prompt)} chars")
            
            response = self.groq_client.chat.completions.create(...)
            
            raw_response = response.choices[0].message.content.strip()
            
            # DEBUG: Print what LLM returned
            print(f"🔍 DEBUG: LLM raw response for chunk {chunk_index}:")
            print(f"   Response length: {len(raw_response)} chars")
            print(f"   First 200 chars: {raw_response[:200]}...")
            
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                parsed_json = json.loads(json_match.group())
                if isinstance(parsed_json, dict):
                    print(f"🔍 DEBUG: Parsed JSON keys: {list(parsed_json.keys())}")
                    print(f"🔍 DEBUG: Business logic found: {len(parsed_json.get('business_logic', {}))}")
                    print(f"🔍 DEBUG: Database operations found: {len(parsed_json.get('database_operations', []))}")
                    return parsed_json
                elif isinstance(parsed_json, list):
                    print(f"🔍 DEBUG: Parsed JSON is list with {len(parsed_json)} items")
                    # Convert list to expected dict format
                    return {"business_logic": {}, "database_operations": [], "raw_list": parsed_json}
                else:
                    print(f"🔍 DEBUG: Unexpected JSON type: {type(parsed_json)}")
                    return {}
            else:
                print(f"❌ DEBUG: No JSON found in LLM response for chunk {chunk_index}")
                return {}
                
        except Exception as e:
            print(f"❌ DEBUG: LLM chunk analysis failed for chunk {chunk_index}: {e}")
            return {}
        




    def generate_esql_modules(self, requirements: Dict, template_info: Dict) -> List[Dict]:
        """
        Generate ESQL modules - 100% LLM based
        """
        print("🏭 Starting LLM-based ESQL module generation...")
        
        modules = requirements.get('esql_modules', [])
        if not modules:
            print("⚠️ No ESQL modules identified in requirements")
            return []
        
        generated_modules = []
        
        for module_req in modules:
            module_name = module_req.get('name', 'UnknownModule')
            if module_name.lower().endswith('.esql'):
                module_name = module_name[:-5]

            print(f"⚡ Generating {module_name}.esql with LLM...")
            
            try:
                # Enrich module requirements with global context
                enriched_module = self._enrich_module_requirements(module_req, requirements)
                
                # Generate ESQL using LLM
                esql_content = self._llm_generate_esql_module(enriched_module, template_info)
                
                # Validate and enhance if needed
                validated_esql = self._llm_validate_and_enhance(esql_content, module_name)
                
                generated_modules.append({
                    'name': module_name,
                    'content': validated_esql,
                    'purpose': module_req.get('purpose', 'Processing'),
                    'type': module_req.get('type', 'COMPUTE'),
                    'validation_status': 'LLM_VALIDATED'
                })
                
                print(f"✅ {module_name}.esql generated successfully")
                
            except Exception as e:
                print(f"❌ Failed to generate {module_name}.esql: {e}")
                continue
        
        print(f"🎉 Generated {len(generated_modules)} ESQL modules via LLM")
        return generated_modules
    


    def _enrich_module_requirements(self, module_req: Dict, global_requirements: Dict) -> Dict:
        enriched = module_req.copy()

        # Determine module type using our new method
        module_name = module_req.get('name', '')
        module_type = self._determine_module_type(module_name)
        
        print(f"🔍 DEBUG: Enriching module {module_name} (type: {module_type})")
        print(f"    Global database_operations available: {len(global_requirements.get('database_operations', []))}")
        print(f"    Global transformations available: {len(global_requirements.get('transformations', []))}")
        
        # Distribute business logic based on module type
        if module_type == 'COMPUTE':
            # FULL business logic for Compute module only
            enriched['business_logic'] = {
                **global_requirements.get('business_logic', {}),
                **module_req.get('business_logic', {}),
                'module_type': 'COMPUTE',
                'comprehensive_logic': True,
                'has_database_operations': len(global_requirements.get('database_operations', [])) > 0,
                'has_transformations': len(global_requirements.get('transformations', [])) > 0
            }
            # Include ALL database operations for Compute module
            enriched['database_operations'] = global_requirements.get('database_operations', [])
            enriched['transformations'] = global_requirements.get('transformations', [])
            
            print(f"    ✅ COMPUTE module enriched with {len(enriched['database_operations'])} database ops, {len(enriched['transformations'])} transformations")
            
        elif module_type in ['INPUT_EVENT', 'OUTPUT_EVENT']:
            # Event capture modules - metadata only
            enriched['business_logic'] = {
                'module_type': module_type,
                'purpose': 'event_capture_only',
                'metadata_extraction': True,
                'business_processing': False,
                'has_database_operations': False,
                'has_transformations': False
            }
            enriched['database_operations'] = []  # No database operations
            enriched['transformations'] = []     # No transformations
            
            print(f"    ✅ EVENT module configured for metadata capture only")
            
        elif module_type in ['POST_ENRICHMENT', 'POST_EVENT']:
            # Light processing modules - some business logic but no database operations
            enriched['business_logic'] = {
                'module_type': module_type,
                'purpose': 'light_processing',
                'validation_only': True,
                'routing_logic': True,
                'has_database_operations': False,
                'has_transformations': False
            }
            enriched['database_operations'] = []  # No database operations
            enriched['transformations'] = []     # No transformations
            
            print(f"    ✅ POST_PROCESSING module configured for light processing")
            
        elif module_type == 'FAILURE':
            # Error handling module
            enriched['business_logic'] = {
                'module_type': 'FAILURE',
                'purpose': 'error_handling',
                'error_capture': True,
                'exception_processing': True,
                'has_database_operations': False,
                'has_transformations': False
            }
            enriched['database_operations'] = []  # No database operations
            enriched['transformations'] = []     # No transformations
            
            print(f"    ✅ FAILURE module configured for error handling")
            
        else:
            # Default lightweight logic
            enriched['business_logic'] = {
                **module_req.get('business_logic', {}),
                'module_type': 'UNKNOWN',
                'purpose': 'default_processing',
                'has_database_operations': False,
                'has_transformations': False
            }
            enriched['database_operations'] = []
            enriched['transformations'] = []
            
            print(f"    ⚠️ UNKNOWN module type - using default configuration")
        
        # Common attributes for all modules
        enriched['message_structure'] = global_requirements.get('message_structure', {})
        enriched['customizations'] = global_requirements.get('customizations', {})
        
        # Debug final enrichment result
        has_business_logic = bool(enriched.get('business_logic'))
        print(f"    🎯 Final: {module_name} business logic = {has_business_logic}")
        print(f"    📊 Module database_operations: {len(enriched.get('database_operations', []))}")
        print(f"    📊 Module transformations: {len(enriched.get('transformations', []))}")
        
        return enriched





    def _llm_generate_esql_module(self, module_requirements: Dict, template_info: Dict) -> str:
        """
        Generate ESQL module using LLM - NO hardcoded fallbacks
        """
        if not self.groq_client:
            raise Exception("LLM client not available")
        
        try:
            prompt = self._get_esql_generation_prompt(module_requirements, template_info)
            
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.05,  # Very low for consistent structure
                max_tokens=self.max_tokens_per_request
            )
            print(f"🔍 DEBUG: LLM call completed in method: {__name__}")
            print(f"🔍 DEBUG: token_tracker in session_state: {'token_tracker' in st.session_state}")
            try:
                if 'token_tracker' in st.session_state and hasattr(response, 'usage') and response.usage:
                    st.session_state.token_tracker.manual_track(
                        agent="esql_generator",
                        operation="llm_call_detected",  # Generic operation name
                        model=self.groq_model,
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                        flow_name="esql_generation"
                    )
                    print(f"📊 esql_generator/llm_call_detected: {response.usage.total_tokens} tokens")
                else:
                    print("🔍 DEBUG: Token tracking skipped - conditions not met")
            except Exception as e:
                print(f"🔍 DEBUG: Token tracking error: {e}")


            self.llm_calls_count += 1
            esql_content = response.choices[0].message.content.strip()
            
            # Remove any markdown code blocks if present
            esql_content = re.sub(r'```esql\n?', '', esql_content)
            esql_content = re.sub(r'```\n?', '', esql_content)
            esql_content = self._clean_esql_content(esql_content)

            # Extract module name from requirements
            module_name = module_requirements.get('name', '')

            # Auto-fix: Add BROKER SCHEMA if missing
            esql_content = self._ensure_broker_schema(esql_content, module_name)


            return esql_content
            
        except Exception as e:
            raise Exception(f"LLM ESQL generation failed: {str(e)}")

    def _llm_validate_and_enhance(self, esql_content: str, module_name: str) -> str:
        """
        LLM-based validation and enhancement - NO hardcoded fixes
        """
        if not self.groq_client:
            return esql_content
        
        validation_prompt = f"""Validate and enhance this ESQL module to ensure it meets all requirements:

## MODULE NAME: {module_name}

## ESQL CODE TO VALIDATE:
{esql_content}

## VALIDATION REQUIREMENTS:
1. ✅ Contains CREATE COMPUTE MODULE {module_name}
2. ✅ Contains CREATE FUNCTION Main() RETURNS BOOLEAN
3. ✅ Contains CREATE PROCEDURE CopyMessageHeaders() exactly as required
4. ✅ Contains CREATE PROCEDURE CopyEntireMessage() exactly as required
5. ✅ Contains END MODULE;
6. ✅ No comments starting with "--"
7. ✅ InputRoot treated as READ-ONLY
8. ✅ OutputRoot used for modifications
9. ✅ All required DECLARE statements present
10. ✅ RETURN TRUE; statement present

## ENHANCEMENT REQUIREMENTS:
- Fix any missing procedures
- Ensure proper structure
- Maintain all business logic
- Keep all customizations intact

If the code is valid, return it unchanged.
If fixes are needed, return the corrected complete ESQL module.

Return ONLY the ESQL code:"""

        try:
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": "You are an ESQL validation expert. Fix issues while preserving all business logic."},
                    {"role": "user", "content": validation_prompt}
                ],
                temperature=0.1,
                max_tokens=self.max_tokens_per_request
            )
            print(f"🔍 DEBUG: LLM call completed in method: {__name__}")
            print(f"🔍 DEBUG: token_tracker in session_state: {'token_tracker' in st.session_state}")
            try:
                if 'token_tracker' in st.session_state and hasattr(response, 'usage') and response.usage:
                    st.session_state.token_tracker.manual_track(
                        agent="esql_generator",
                        operation="llm_call_detected",  # Generic operation name
                        model=self.groq_model,
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                        flow_name="esql_generation"
                    )
                    print(f"📊 esql_generator/llm_call_detected: {response.usage.total_tokens} tokens")
                else:
                    print("🔍 DEBUG: Token tracking skipped - conditions not met")
            except Exception as e:
                print(f"🔍 DEBUG: Token tracking error: {e}")
            
            self.llm_calls_count += 1
            enhanced_content = response.choices[0].message.content.strip()
            
            # Remove any markdown if present
            enhanced_content = re.sub(r'```esql\n?', '', enhanced_content)
            enhanced_content = re.sub(r'```\n?', '', enhanced_content)
            
            return enhanced_content
            
        except Exception as e:
            print(f"⚠️ LLM validation failed for {module_name}: {e}")
            return esql_content  # Return original if validation fails



    def generate_esql_files(self, vector_content: str, esql_template: Dict, 
                        msgflow_content: Dict, json_mappings: Dict, output_dir: str = None) -> Dict:
        """
        Generate ESQL files using Vector DB content and 100% LLM processing
        STANDALONE METHOD: Individual error isolation, no helper methods, no fallbacks
        Saves successful modules immediately, continues processing if some modules fail
        """
        print("🚀 Starting Vector DB + LLM ESQL generation...")
        
        # Set output_dir at the beginning
        self.output_dir = output_dir or 'output'
        
        if not self.groq_client:
            raise Exception("LLM client not available - Vector DB processing requires LLM")
        
        # Initialize tracking arrays
        successful_modules = []
        failed_modules = []
        
        try:
            # Step 1: Load supporting files for LLM analysis
            print("📁 Loading supporting files for LLM analysis...")
            
            # Load ESQL template
            esql_template_content = ""
            if esql_template.get('path') and os.path.exists(esql_template['path']):
                with open(esql_template['path'], 'r', encoding='utf-8') as f:
                    esql_template_content = f.read()
                print(f"  ✅ ESQL Template loaded: {len(esql_template_content)} characters")
            else:
                raise Exception(f"ESQL template not found: {esql_template.get('path', 'No path provided')}")
            
            # Load MessageFlow content
            msgflow_content_text = ""
            msgflow_file_found = None

            # Search for any .msgflow file in output directory
            import glob
            msgflow_pattern = os.path.join("output", "**", "*.msgflow")
            msgflow_files = glob.glob(msgflow_pattern, recursive=True)

            if msgflow_files:
                msgflow_file_found = msgflow_files[0]  # Use first .msgflow file found
                with open(msgflow_file_found, 'r', encoding='utf-8') as f:
                    msgflow_content_text = f.read()
                print(f"  ✅ MessageFlow auto-discovered: {msgflow_file_found} ({len(msgflow_content_text)} characters)")
            else:
                print(f"  ⚠️ No .msgflow files found in output directory")
                print(f"  🔄 Continuing with Vector DB content only...")
                msgflow_content_text = "" 
            
            # Load JSON mappings
            json_mappings_data = {}
            if json_mappings.get('path') and os.path.exists(json_mappings['path']):
                with open(json_mappings['path'], 'r', encoding='utf-8') as f:
                    json_mappings_data = json.load(f)
                print(f"  ✅ JSON mappings loaded: {len(json_mappings_data)} components")
            else:
                raise Exception(f"JSON mappings not found: {json_mappings.get('path', 'No path provided')}")
            
            # Extract ESQL requirements from MessageFlow compute expressions via LLM
            print("🔍 LLM extracting ESQL requirements from MessageFlow compute expressions...")
            
            msgflow_analysis_prompt = f"""Analyze MessageFlow XML to extract ALL ESQL module requirements from compute expressions:

    {msgflow_content_text}

    Extract and return ONLY valid JSON with:
    1. "esql_modules": List of modules from computeExpression="esql://routine/#ModuleName.Main" patterns
    2. "compute_nodes": Details of each compute node found
    3. "flow_structure": Processing flow pattern

    Return only JSON:"""

            msgflow_response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": "Extract ESQL requirements from MessageFlow XML. Return valid JSON only."},
                    {"role": "user", "content": msgflow_analysis_prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )

            try:
                if 'token_tracker' in st.session_state and hasattr(generation_response, 'usage') and generation_response.usage:
                    st.session_state.token_tracker.manual_track(
                        agent="esql_generator",
                        operation="esql_module_generation",
                        model=self.groq_model,
                        input_tokens=generation_response.usage.prompt_tokens,
                        output_tokens=generation_response.usage.completion_tokens,
                        flow_name="esql_generation"
                    )
                    print(f"📊 esql_generator/esql_module_generation: {generation_response.usage.total_tokens} tokens | ${generation_response.usage.total_tokens * 0.0008:.4f} | {self.groq_model}")
            except Exception as e:
                print(f"⚠️ Token tracking failed: {e}")
            
            self.llm_calls_count += 1
            
            # Parse MessageFlow analysis
            msgflow_analysis_content = msgflow_response.choices[0].message.content.strip()
            json_start = msgflow_analysis_content.find('{')
            json_end = msgflow_analysis_content.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise Exception("LLM failed to extract ESQL requirements from MessageFlow")
            
            try:
                msgflow_analysis = json.loads(msgflow_analysis_content[json_start:json_end])
                msgflow_esql_modules = msgflow_analysis.get('esql_modules', [])
            except json.JSONDecodeError as e:
                raise Exception(f"LLM generated invalid JSON for MessageFlow analysis: {str(e)}")
            
            print(f"  ✅ LLM extracted {len(msgflow_esql_modules)} ESQL modules from MessageFlow")
            
            # Step 2: Comprehensive LLM Analysis combining ALL sources
            print("🧠 LLM Analysis: Extracting ESQL requirements from ALL sources...")
            
            analysis_prompt = f"""Extract ESQL requirements with SPECIFIC FOCUS on database operations and stored procedures:

            ## VECTOR DB BUSINESS REQUIREMENTS (PRIMARY SOURCE - SCAN FOR DATABASE OPERATIONS):
            {vector_content}

            ## COMPONENT MAPPINGS (BUSINESS PATTERNS AND LOGIC):
            {json.dumps(json_mappings_data.get('component_mappings', []), indent=2)}

            ## MESSAGEFLOW COMPUTE EXPRESSIONS (MODULE STRUCTURE):
            {json.dumps(msgflow_esql_modules, indent=2)}

            ## CRITICAL DATABASE OPERATION EXTRACTION - FIND THESE SPECIFIC PATTERNS:
            **HIGH PRIORITY - Search for these EXACT terms in Vector DB content:**
            - "sp_GetMainCompanyInCountry" 
            - "sp_Shipment_GetIdBySSN"
            - "sp_" (any stored procedure starting with sp_)
            - "BEGIN SELECT"
            - "database lookup"
            - "CompanyCode"
            - "CountryCode" 
            - "enrichment"
            - "CW1.IN.DOCUMENT.SND.QL"
            - "DocumentMessage"

            ## INTEGRATION RULES:
            - Create exactly {len(msgflow_esql_modules)} ESQL modules as defined by MessageFlow
            - **CRITICAL**: Extract ALL database operations from Vector DB content and map to _Compute module
            - **CRITICAL**: Extract ALL stored procedures and their parameters from Vector DB content  
            - Ensure _Compute module contains comprehensive business logic INCLUDING database operations
            - Ensure event modules contain only metadata capture logic

            ## MESSAGEFLOW STRUCTURE:
            {msgflow_content_text[:5000]}

            ## BUSINESS LOGIC EXTRACTION REQUIREMENTS:
            **For _Compute module specifically, extract:**
            - All stored procedure calls found in Vector DB content
            - Database lookup operations and conditional logic
            - Parameter mappings (CompanyCode, CountryCode, ShipmentReference, etc.)
            - XPath expressions for data extraction
            - Error handling and validation patterns

            Extract and return JSON with:
            {{
                "esql_modules": [
                    {{"name": "CW1_IN_Document_SND_Compute", "type": "COMPUTE", "purpose": "Main business logic with database operations"}},
                    {{"name": "CW1_IN_Document_SND_InputEventMessage", "type": "INPUT_EVENT", "purpose": "Event capture only"}},
                    {{"name": "CW1_IN_Document_SND_OutputEventMessage", "type": "OUTPUT_EVENT", "purpose": "Event capture only"}},
                    {{"name": "CW1_IN_Document_SND_AfterEnrichment", "type": "POST_ENRICHMENT", "purpose": "Post processing"}},
                    {{"name": "CW1_IN_Document_SND_AfterEventMsg", "type": "POST_EVENT", "purpose": "Post processing"}},
                    {{"name": "CW1_IN_Document_SND_Failure", "type": "FAILURE", "purpose": "Error handling"}}
                ],
                "business_logic": {{
                    "comprehensive_processing": true,
                    "database_enrichment": true,
                    "stored_procedures_required": true,
                    "message_transformation": true
                }},
                "database_operations": [
                    {{"procedure": "sp_GetMainCompanyInCountry", "parameters": ["CompanyCode", "CountryCode"], "purpose": "Company lookup", "xpath_source": "//Header//CompanyCode"}},
                    {{"procedure": "sp_Shipment_GetIdBySSN", "parameters": ["ShipmentReference"], "purpose": "Shipment validation", "xpath_source": "//ShipmentDetails//ShipmentId"}}
                ],
                "transformations": [
                    {{"type": "message_format", "source": "CDM Document", "target": "CW1 format", "operation": "database_enrichment"}},
                    {{"type": "data_lookup", "source": "CompanyCode", "target": "enriched_company_data"}}
                ],
                "message_structure": {{
                    "input_format": "XML",
                    "queue": "CW1.IN.DOCUMENT.SND.QL",
                    "xpath_mappings": ["//Header//CompanyCode", "//Header//CountryCode"]
                }},
                "customizations": {{
                    "event_capture": true,
                    "database_enrichment": true,
                    "error_handling": true
                }}
            }}

            ## EXTRACTION PRIORITY:
            1. **HIGHEST**: Scan Vector DB content for stored procedures (sp_) and database operations  
            2. **HIGH**: Extract XPath expressions and parameter mappings for database calls
            3. **MEDIUM**: Message transformation requirements
            4. **LOW**: General processing patterns

            Return ONLY valid JSON with ALL database operations and stored procedures found in Vector DB content."""

            # LLM call for comprehensive requirements analysis
            analysis_response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": "You are an ESQL template enhancer. Your task is to ADD business logic to existing template while PRESERVING the exact template structure. CRITICAL RULES: 1) The output MUST end with exactly 'END MODULE;' as provided in the template. 2) NEVER use CALL statements - they are forbidden. 3) Use SET statements instead of CALL statements. 4) Do NOT remove or modify the template infrastructure - only enhance business logic sections marked with comments."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )

            try:
                if 'token_tracker' in st.session_state and hasattr(generation_response, 'usage') and generation_response.usage:
                    st.session_state.token_tracker.manual_track(
                        agent="esql_generator",
                        operation="esql_module_generation",
                        model=self.groq_model,
                        input_tokens=generation_response.usage.prompt_tokens,
                        output_tokens=generation_response.usage.completion_tokens,
                        flow_name="esql_generation"
                    )
                    print(f"📊 esql_generator/esql_module_generation: {generation_response.usage.total_tokens} tokens | ${generation_response.usage.total_tokens * 0.0008:.4f} | {self.groq_model}")
            except Exception as e:
                print(f"⚠️ Token tracking failed: {e}")
            
            self.llm_calls_count += 1
            
            # Parse comprehensive analysis
            analysis_content = analysis_response.choices[0].message.content.strip()
            json_start = analysis_content.find('{')
            json_end = analysis_content.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise Exception("LLM failed to generate valid JSON requirements analysis")
            
            try:
                requirements = json.loads(analysis_content[json_start:json_end])

                print("🔍 DEBUG: Business logic extraction verification:")
                print(f"   📊 Total modules identified: {len(requirements.get('esql_modules', []))}")
                print(f"   📊 Business logic keys: {list(requirements.get('business_logic', {}).keys())}")
                print(f"   📊 Database operations: {len(requirements.get('database_operations', []))}")
                print(f"   📊 Transformations: {len(requirements.get('transformations', []))}")

                for module in requirements.get('esql_modules', []):
                    module_name = module.get('name', 'Unknown')
                    has_business_logic = bool(module.get('business_logic'))
                    print(f"   🎯 {module_name}: Business logic = {has_business_logic}")

                # 🎯 FORCE BUSINESS LOGIC ASSIGNMENT - Fix #4 Implementation
                print("🔧 FORCING proper business logic distribution to individual modules...")

                # Get global requirements for distribution
                global_database_ops = requirements.get('database_operations', [])
                global_transformations = requirements.get('transformations', [])
                global_business_logic = requirements.get('business_logic', {})

                print(f"📦 Available for distribution: {len(global_database_ops)} DB ops, {len(global_transformations)} transforms")

                # Force assign business logic to each module based on module type
                for module in requirements.get('esql_modules', []):
                    module_name = module.get('name', 'Unknown')
                    
                    # Determine module type and assign appropriate logic
                    if module_name.endswith('_Compute'):
                        # COMPUTE module gets ALL business logic and database operations
                        module['business_logic'] = {
                            **global_business_logic,
                            'module_type': 'COMPUTE',
                            'comprehensive_processing': True,
                            'database_enrichment': True,
                            'stored_procedures_required': True,
                            'has_database_operations': len(global_database_ops) > 0,
                            'has_transformations': len(global_transformations) > 0,
                            'primary_business_module': True
                        }
                        module['database_operations'] = global_database_ops.copy()
                        module['transformations'] = global_transformations.copy()
                        print(f"   ✅ COMPUTE: {module_name} → FULL business logic + {len(global_database_ops)} DB ops + {len(global_transformations)} transforms")
                        
                    elif module_name.endswith('_InputEventMessage') or module_name.endswith('_OutputEventMessage'):
                        # EVENT modules get metadata extraction only
                        module['business_logic'] = {
                            'module_type': 'EVENT_CAPTURE',
                            'metadata_extraction': True,
                            'event_logging': True,
                            'business_processing': False,
                            'has_database_operations': False,
                            'has_transformations': False,
                            'purpose': 'metadata_capture_only'
                        }
                        module['database_operations'] = []
                        module['transformations'] = []
                        print(f"   ✅ EVENT: {module_name} → Metadata capture only (no DB ops, no transforms)")
                        
                    elif module_name.endswith('_AfterEnrichment'):
                        # POST-ENRICHMENT gets validation and routing only
                        module['business_logic'] = {
                            'module_type': 'POST_ENRICHMENT',
                            'light_processing': True,
                            'validation_only': True,
                            'routing_logic': True,
                            'enrichment_validation': True,
                            'has_database_operations': False,
                            'has_transformations': False,
                            'purpose': 'validate_enriched_data_and_route'
                        }
                        module['database_operations'] = []
                        module['transformations'] = []
                        print(f"   ✅ POST-ENRICHMENT: {module_name} → Validation & routing only (no DB ops, no transforms)")
                        
                    elif module_name.endswith('_AfterEventMsg'):
                        # POST-EVENT gets message handling only
                        module['business_logic'] = {
                            'module_type': 'POST_EVENT',
                            'message_handling': True,
                            'cleanup_processing': True,
                            'final_routing': True,
                            'has_database_operations': False,
                            'has_transformations': False,
                            'purpose': 'post_event_cleanup_and_routing'
                        }
                        module['database_operations'] = []
                        module['transformations'] = []
                        print(f"   ✅ POST-EVENT: {module_name} → Message handling only (no DB ops, no transforms)")
                        
                    elif module_name.endswith('_Failure'):
                        # FAILURE module gets error handling only
                        module['business_logic'] = {
                            'module_type': 'FAILURE',
                            'error_handling': True,
                            'exception_processing': True,
                            'error_logging': True,
                            'fault_tolerance': True,
                            'has_database_operations': False,
                            'has_transformations': False,
                            'purpose': 'error_capture_and_logging'
                        }
                        module['database_operations'] = []
                        module['transformations'] = []
                        print(f"   ✅ FAILURE: {module_name} → Error handling only (no DB ops, no transforms)")
                        
                    else:
                        # DEFAULT fallback for unknown module types
                        module['business_logic'] = {
                            'module_type': 'DEFAULT',
                            'basic_processing': True,
                            'lightweight_logic': True,
                            'has_database_operations': False,
                            'has_transformations': False,
                            'purpose': 'basic_message_processing'
                        }
                        module['database_operations'] = []
                        module['transformations'] = []
                        print(f"   ⚠️ DEFAULT: {module_name} → Basic processing (no DB ops, no transforms)")

                # Verify the fix worked by re-running debug verification
                print("🔍 DEBUG: AFTER FORCING business logic distribution:")
                print("=" * 60)
                for module in requirements.get('esql_modules', []):
                    module_name = module.get('name', 'Unknown')
                    has_business_logic = bool(module.get('business_logic'))
                    db_ops_count = len(module.get('database_operations', []))
                    transforms_count = len(module.get('transformations', []))
                    module_type = module.get('business_logic', {}).get('module_type', 'UNKNOWN')

                    purpose = module.get('business_logic', {}).get('purpose', 'No purpose defined')
                    
                    print(f"   🎯 {module_name}:")
                    print(f"      Type: {module_type}")
                    print(f"      Business logic: {has_business_logic}")
                    print(f"      DB operations: {db_ops_count}")
                    print(f"      Transformations: {transforms_count}")
                    print(f"      Purpose: {purpose}")
                    print()

                print("🎉 Business logic distribution completed!")
                print("=" * 60)

            except json.JSONDecodeError as e:
                raise Exception(f"LLM generated invalid JSON for requirements: {str(e)}")
            
            total_modules = len(requirements.get('esql_modules', []))
            component_modules = len([m for m in requirements.get('esql_modules', []) if m.get('source') == 'component_mapping'])
            msgflow_modules = len([m for m in requirements.get('esql_modules', []) if m.get('source') == 'messageflow_compute'])
            
            print(f"  ✅ Requirements analysis completed: {total_modules} modules identified")
            print(f"    📊 Component mappings: {component_modules} modules")
            print(f"    📊 MessageFlow compute: {msgflow_modules} modules")
            
            if total_modules == 0:
                raise Exception("No ESQL modules identified from Vector DB + MessageFlow analysis")
            
            # Create template info for helper methods
            template_info = {
                'content': esql_template_content,
                'length': len(esql_template_content)
            }
            
            # Step 3: Generate ESQL modules using node-specific helper methods
            print("🔧 LLM Generation: Creating ESQL modules using node-specific prompts...")
            
            esql_modules = requirements.get('esql_modules', [])
            
            # Process each module individually with error isolation
            for i, module_req in enumerate(esql_modules):
                module_name = module_req.get('name', f'ESQLModule_{i+1}')
                
                try:
                    print(f"  🎯 Generating {module_name}.esql with node-specific prompt...")
                    
                    # Determine module type and get appropriate prompt using helper methods
                    module_type = self._determine_module_type(module_name)
                    
                    if module_type == 'COMPUTE':
                        generation_prompt = self._get_compute_module_prompt(module_req, module_name)
                    elif module_type in ['INPUT_EVENT', 'OUTPUT_EVENT']:
                        generation_prompt = self._get_event_module_prompt(module_req, module_type)
                    elif module_type in ['POST_ENRICHMENT', 'POST_EVENT']:
                        generation_prompt = self._get_processing_module_prompt(module_req, module_type)
                    elif module_type == 'FAILURE':
                        generation_prompt = self._get_failure_module_prompt(module_req)
                    else:
                        generation_prompt = self._get_default_module_prompt(module_req)
                    
                    print(f"      📝 Using {module_type} specific prompt template")


                    basic_data = {
                        'module_name': module_name,
                        'module_type': module_type,
                        'purpose': module_req.get('purpose', 'Processing'),
                        'source': module_req.get('source', 'component_mapping')
                    }
                    try:
                        print(f"🔍 ABOUT TO GENERATE TEMPLATE for {module_name} with type '{module_type}'")
                        template_foundation = self._generate_esql_from_template(module_type, module_name, basic_data)
                        print(f"   ✅ Template generation succeeded for {module_name}")
                    except Exception as e:
                        print(f"   ❌ Template generation failed for {module_name}: {str(e)}")
                        raise


                    print(f"\n🚨 TEMPLATE FOUNDATION DEBUG for {module_name}:")
                    print(f"   📊 Length: {len(template_foundation)} chars")
                    print(f"   🔍 Last 100 chars: ...{repr(template_foundation[-100:])}")
                    print(f"   ✅ Ends with 'END MODULE;': {template_foundation.rstrip().endswith('END MODULE;')}")
                    print(f"   📝 Template content preview:")
                    print("   " + "\n   ".join(template_foundation.split('\n')[-5:]))  # Last 5 lines

                    print(f"🔍 ABOUT TO MAKE LLM CALL for {module_name}")
                    print(f"   📊 Template foundation length: {len(template_foundation)}")
                    print(f"   📊 Generation prompt length: {len(generation_prompt)}")
                    print(f"   📊 Total prompt size: {len(template_foundation) + len(generation_prompt)}")


                    try:
                        # LLM call for ESQL generation with node-specific prompt
                        generation_response = self.groq_client.chat.completions.create(
                            model=self.groq_model,
                            messages=[
                                {"role": "system", "content": "You are an ESQL code editor. Your only task is to find the marker '-- [[[INSERT_BUSINESS_LOGIC_HERE]]]' in the template and replace that single line with the code provided. Do not interpret, modify, or enhance the code. Just perform the replacement and return the result."},
                                {"role": "user", "content": f"""Here is the complete ESQL template:

                                    {template_foundation}

                                    Task: {generation_prompt}

                                    CRITICAL: Return the ENTIRE template from CREATE COMPUTE MODULE to END MODULE; 
                                    with the modifications applied. Do not omit any lines from the original template."""}
                            ],
                            temperature=0.1,
                            max_tokens=4000
                        )
                        print(f"   ✅ LLM call succeeded for {module_name}")
                    except Exception as e:
                        print(f"   ❌ LLM call failed for {module_name}: {str(e)}")
                        raise    

                    try:
                        if 'token_tracker' in st.session_state and hasattr(generation_response, 'usage') and generation_response.usage:
                            st.session_state.token_tracker.manual_track(
                                agent="esql_generator",
                                operation="esql_module_generation",
                                model=self.groq_model,
                                input_tokens=generation_response.usage.prompt_tokens,
                                output_tokens=generation_response.usage.completion_tokens,
                                flow_name="esql_generation"
                            )
                            print(f"📊 esql_generator/esql_module_generation: {generation_response.usage.total_tokens} tokens | ${generation_response.usage.total_tokens * 0.0008:.4f} | {self.groq_model}")
                    except Exception as e:
                        print(f"⚠️ Token tracking failed: {e}")
                    
                    self.llm_calls_count += 1
                    
                    esql_content = generation_response.choices[0].message.content.strip()

                    print(f"🔍 DEBUG LLM OUTPUT for {module_name}:")
                    print(f"   📊 Length: {len(esql_content)} chars")
                    print(f"   🔍 Last 200 chars: ...{repr(esql_content[-200:])}")
                    print(f"   ✅ Ends with 'END MODULE;': {esql_content.rstrip().endswith('END MODULE;')}")
                    if not esql_content.rstrip().endswith('END MODULE;'):
                        print(f"   ❌ Actually ends with: {repr(esql_content[-50:])}")

                    
                    # Clean LLM output - remove code markers if present
                    esql_content = re.sub(r'```[\w]*\n?', '', esql_content)
                    esql_content = re.sub(r'\n?```\s*$', '', esql_content)
                    esql_content = self._clean_esql_content(esql_content)

                    # Auto-fix: Add BROKER SCHEMA MODULE if missing
                    esql_content = self._ensure_broker_schema(esql_content, module_name)

                    # Validate ESQL format compliance with BROKER SCHEMA requirement
                    lines = esql_content.strip().split('\n')
                    
                    # Check Line 1: Must start with BROKER SCHEMA (with or without MODULE keyword)
                    if not lines[0].strip().startswith('BROKER SCHEMA'):
                        raise Exception(f"ESQL format violation: {module_name} must start with 'BROKER SCHEMA' on line 1")
                    
                    # Check Line 2: Must have CREATE COMPUTE MODULE with module name
                    if len(lines) < 2 or not lines[1].strip().startswith('CREATE COMPUTE MODULE'):
                        raise Exception(f"ESQL format violation: {module_name} must have 'CREATE COMPUTE MODULE {module_name}' on line 2")
                    
                    # Verify module name appears in line 2
                    if module_name not in lines[1]:
                        raise Exception(f"ESQL format violation: Module name '{module_name}' not found in CREATE COMPUTE MODULE declaration on line 2")
                    
                    # Check ending
                    if not esql_content.rstrip().endswith('END MODULE;'):
                        raise Exception(f"ESQL format violation: {module_name} must end with 'END MODULE;'")
                    
                    # Check for forbidden symbols
                    if '@' in esql_content:
                        raise Exception(f"ESQL format violation: {module_name} contains forbidden '@' symbols")

                    # ENHANCED: Apply constraint validation
                    constraint_result = self.validate_esql_constraints(esql_content)

                    # Use the modified content (which may have commented-out CALL statements)
                    if 'modified_content' in constraint_result:
                        esql_content = constraint_result['modified_content']

                    # Handle warnings (auto-fixes like commented CALL statements)
                    if constraint_result.get('warnings'):
                        print(f"    ⚠️  Auto-fixes applied for {module_name}: {constraint_result['warnings']}")

                    # Only fail if there are actual errors (not just warnings)
                    if not constraint_result['valid']:
                        print(f"    ❌ Constraint violations for {module_name}: {constraint_result['errors']}")
                        failed_modules.append({
                            'name': module_name,
                            'error': f"Constraint violations: {', '.join(constraint_result['errors'])}"
                        })
                        continue
                    
                    # Validate required elements
                    required_elements = [
                        f'CREATE COMPUTE MODULE {module_name}',
                        'CREATE FUNCTION Main() RETURNS BOOLEAN',
                        'RETURN TRUE;',
                        'CREATE PROCEDURE CopyMessageHeaders()',
                        'CREATE PROCEDURE CopyEntireMessage()',
                        'END MODULE;'
                    ]
                    
                    missing_elements = [elem for elem in required_elements if elem not in esql_content]
                    if missing_elements:
                        raise Exception(f"LLM generated incomplete ESQL for {module_name}. Missing: {missing_elements}")
                    
                    # Save ESQL file immediately upon successful generation
                    esql_filename = f"{module_name}.esql"
                    esql_file_path = os.path.join(self.output_dir, esql_filename)
                    
                    # Ensure output directory exists
                    os.makedirs(os.path.dirname(esql_file_path), exist_ok=True)
                    
                    with open(esql_file_path, 'w', encoding='utf-8') as f:
                        f.write(esql_content)
                    
                    # Track successful module
                    successful_modules.append({
                        'name': module_name,
                        'filename': esql_filename,
                        'file_path': esql_file_path,
                        'content_length': len(esql_content),
                        'purpose': module_req.get('purpose', 'Processing'),
                        'type': module_req.get('type', 'COMPUTE'),
                        'source': module_req.get('source', 'component_mapping'),
                        'validation_status': 'FORMAT_VALIDATED',
                        'module_type': module_type,
                        'prompt_method': f"_{module_type.lower()}_module_prompt"
                    })
                    
                    print(f"    ✅ {esql_filename} generated successfully ({len(esql_content)} characters) using {module_type} prompt")
                    
                except Exception as module_error:
                    # Individual module failed - log and continue with next module
                    error_message = str(module_error)
                    print(f"💥 ESQL generation failed: {error_message}")
                    
                    failed_modules.append({
                        'name': module_name,
                        'error_message': error_message,
                        'purpose': module_req.get('purpose', 'Processing'),
                        'source': module_req.get('source', 'component_mapping')
                    })
                    
                    # Continue processing remaining modules
                    continue
            
            # Step 4: Determine final status and return results
            total_attempted = len(esql_modules)
            successful_count = len(successful_modules)
            failed_count = len(failed_modules)
            
            if successful_count == 0:
                # Complete failure - no modules generated
                raise Exception(f"All {total_attempted} ESQL modules failed to generate")
            elif failed_count > 0:
                # Partial success
                print(f"✅ Successful modules: {successful_count}/{total_attempted}")
                print(f"📊 Total files generated: {successful_count}")
                print(f"❌ Failed modules: {failed_count}")
                for failed in failed_modules:
                    print(f"  - {failed['name']}: {failed['error_message']}")
            else:
                # Complete success
                print(f"🎉 ESQL generation complete: {successful_count} modules generated")
            
            return {
                'status': 'success' if failed_count == 0 else 'partial_success',
                'generated_modules': successful_modules,
                'failed_modules': failed_modules,
                'requirements_analysis': requirements,
                'msgflow_analysis': msgflow_analysis,
                'llm_calls_made': self.llm_calls_count,
                'total_modules': successful_count,
                'total_attempted': total_attempted,
                'successful_count': successful_count,
                'failed_count': failed_count,
                'generation_method': '100% LLM Based with Node-Specific Prompts',
                'token_management': 'Vector DB + MessageFlow Optimized',
                'processing_summary': {
                    'vector_content_length': len(vector_content),
                    'component_mappings_processed': len(json_mappings_data.get('component_mappings', [])),
                    'msgflow_esql_modules_extracted': len(msgflow_esql_modules),
                    'total_esql_modules_identified': len(esql_modules),
                    'component_mapping_modules': component_modules,
                    'messageflow_compute_modules': msgflow_modules,
                    'business_logic_rules': len(requirements.get('business_logic', {})),
                    'database_operations': len(requirements.get('database_operations', [])),
                    'transformations': len(requirements.get('transformations', []))
                }
            }
            
        except Exception as e:
            # Only complete failure if we couldn't do initial analysis or have no successful modules
            if len(successful_modules) > 0:
                # We have some successful modules - return partial success instead of complete failure
                return {
                    'status': 'partial_success',
                    'generated_modules': successful_modules,
                    'failed_modules': failed_modules,
                    'llm_calls_made': self.llm_calls_count,
                    'total_modules': len(successful_modules),
                    'successful_count': len(successful_modules),
                    'failed_count': len(failed_modules),
                    'generation_method': '100% LLM Based with Error Recovery',
                    'error_message': f"Partial failure during generation: {str(e)}"
                }
            else:
                print(f"💥 ESQL generation failed: {str(e)}")
                raise Exception(f"Vector DB + MessageFlow ESQL Generation Failed: {str(e)}")



    def validate_esql_constraints(self, esql_content: str) -> Dict[str, Any]:
        """Validate ESQL constraints - NEW METHOD"""
        validation = {'valid': True, 'errors': [], 'warnings': []}
        
        # 1. Check for forbidden VARCHAR
        if re.search(r'\bVARCHAR\b', esql_content, re.IGNORECASE):
            validation['errors'].append("VARCHAR is forbidden - use CHARACTER instead")
            validation['valid'] = False
        
        # 2. Check for Database DECLARE statements  
        if re.search(r'DECLARE\s+\w+\s+Database\b', esql_content, re.IGNORECASE):
            validation['errors'].append("DECLARE with Database is forbidden")
            validation['valid'] = False
        
        # 3. Check for direct procedure calls
        call_pattern = r'(\s*)(CALL\s+\w+\s*\([^)]*\)\s*;?)'
        call_matches = list(re.finditer(call_pattern, esql_content, re.IGNORECASE))

        if call_matches:
            # Start with the original content
            modified_content = esql_content
            
            # Process matches in reverse order to maintain string positions
            for match in reversed(call_matches):
                indentation = match.group(1)  # Preserve indentation
                call_statement = match.group(2)  # The CALL statement
                
                # Create commented version
                commented_version = f"{indentation}-- AUTO-COMMENTED: {call_statement.strip()}"
                
                # Replace in content (update modified_content for next iteration)
                modified_content = modified_content[:match.start()] + commented_version + modified_content[match.end():]
            
            validation['warnings'].append(f"Auto-commented {len(call_matches)} CALL statement(s)")
            validation['modified_content'] = modified_content
        else:
            # No CALL matches found, return original content
            validation['modified_content'] = esql_content
        
        # 4. Check for invalid data types
        valid_types = {'BOOLEAN', 'INTEGER', 'DECIMAL', 'FLOAT', 'CHARACTER', 
                    'BIT', 'BLOB', 'DATE', 'TIME', 'TIMESTAMP', 'REFERENCE', 'ROW'}
        
        declare_pattern = r'DECLARE\s+\w+\s+(\w+)'
        for match in re.finditer(declare_pattern, esql_content, re.IGNORECASE):
            data_type = match.group(1).upper()
            if data_type not in valid_types:
                validation['errors'].append(f"Invalid data type '{data_type}' - only approved types allowed")
                validation['valid'] = False
        
        return validation



    def _clean_esql_content(self, esql_content: str) -> str:
        """
        Clean ESQL content to ensure compliance:
        - Remove all '@' symbols
        - Remove lines starting with 'esql'
        - Ensure proper CREATE COMPUTE start
        - Ensure proper END MODULE; ending
        """
        
        # Step 1: Remove all '@' symbols completely
        cleaned_content = esql_content.replace('@', '')
        
        # Step 2: Process line by line to remove 'esql' prefixes and clean structure
        lines = cleaned_content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped_line = line.strip()
            
            # Skip lines that start with 'esql'
            if stripped_line.lower().startswith('esql'):
                continue
                
            # Keep all other lines
            cleaned_lines.append(line)
        
        # Step 3: Rejoin lines
        cleaned_content = '\n'.join(cleaned_lines)
        
        # Step 4: Ensure proper structure (CREATE COMPUTE start)
        stripped_content = cleaned_content.strip()
        
        # If doesn't start with CREATE COMPUTE, try to fix
        if not stripped_content.startswith('CREATE COMPUTE'):
            # Look for CREATE COMPUTE MODULE pattern and ensure it's at the start
            lines = stripped_content.split('\n')
            create_line_index = -1
            
            for i, line in enumerate(lines):
                if line.strip().startswith('CREATE COMPUTE MODULE'):
                    create_line_index = i
                    break
            
            if create_line_index > 0:
                # Move CREATE COMPUTE MODULE to the beginning
                lines = lines[create_line_index:]
                stripped_content = '\n'.join(lines)
            elif create_line_index == -1:
                # If no CREATE COMPUTE MODULE found, we'll let validation catch this
                pass
        
        # Step 5: Ensure proper ending (END MODULE;)
        if not stripped_content.rstrip().endswith('END MODULE;'):
            # If it ends with just END MODULE (without semicolon), add semicolon
            if stripped_content.rstrip().endswith('END MODULE'):
                stripped_content = stripped_content.rstrip() + ';'
            # If it doesn't end with END MODULE at all, we'll let validation catch this
        
        # Step 6: Clean up extra whitespace
        cleaned_content = stripped_content.strip()
        
        return cleaned_content



    def _ensure_broker_schema(self, esql_content: str, module_name: str) -> str:
        """
        Automatically add BROKER SCHEMA line if missing.
        Extracts base flow name from module_name (e.g., CW1_IN_Document_SND_Compute -> CW1_IN_Document_SND)
        """
        lines = esql_content.strip().split('\n')
        
        # Extract base flow name from module_name
        suffixes = ['_Compute', '_InputEventMessage', '_OutputEventMessage', 
                    '_AfterEnrichment', '_AfterEventMsg', '_Failure']
        
        base_flow_name = module_name
        for suffix in suffixes:
            if module_name.endswith(suffix):
                base_flow_name = module_name[:-len(suffix)]
                break
        
        # Construct correct BROKER SCHEMA line
        broker_line = f'BROKER SCHEMA {base_flow_name}'
        
        # Check if BROKER SCHEMA is present on line 1
        if not lines[0].strip().startswith('BROKER SCHEMA'):
            # BROKER SCHEMA is missing - prepend it
            esql_content = broker_line + '\n' + esql_content
        
        return esql_content



# Helper function for integration
def create_enhanced_esql_generator(groq_api_key=None) -> ESQLGenerator:
    return ESQLGenerator(groq_api_key=groq_api_key)


def validate_data_type_constraints(self, esql_content: str) -> Dict:
    """Validate ESQL data type constraints"""
    validation = {'valid': True, 'errors': [], 'warnings': []}
    
    # Check for forbidden VARCHAR
    if 'VARCHAR' in esql_content:
        validation['errors'].append("VARCHAR is forbidden - use CHARACTER instead")
        validation['valid'] = False
    
    # Check for invalid data types
    valid_types = ['BOOLEAN', 'INTEGER', 'DECIMAL', 'FLOAT', 'CHARACTER', 
                   'BIT', 'BLOB', 'DATE', 'TIME', 'TIMESTAMP', 'REFERENCE', 'ROW']
    
    # Extract DECLARE statements and validate types
    declare_pattern = r'DECLARE\s+\w+\s+(\w+)'
    for match in re.finditer(declare_pattern, esql_content):
        data_type = match.group(1)
        if data_type not in valid_types:
            validation['errors'].append(f"Invalid data type: {data_type}")
            validation['valid'] = False
    
    return validation








# Example usage function
def example_usage():
    """
    Example of how to use the enhanced ESQL generator
    """
    from groq import Groq
    
    # Initialize
    groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    generator = create_enhanced_esql_generator(os.getenv('GROQ_API_KEY'))
    
    # Example input data
    input_data = {
        'esql_template': {'content': 'template content'},
        'msgflow_content': {'nodes': [], 'connections': []},
        'vector_content': 'business requirements',
        'json_mappings': {'mappings': []}
    }
    
    # Generate ESQL files
    results = generator.generate_esql_files(**input_data)
    
    return results


def main():
    """Test harness for ESQL generator - Vector DB Mode"""
    generator = ESQLGenerator()
    
    
    if not st.session_state.get('vector_pipeline'):
        print("❌ Vector DB pipeline not available")
        print("💡 Run this through main.py with Vector DB setup")
        print("📝 Steps: 1) Upload PDF in Agent 1, 2) Setup Vector Knowledge Base, 3) Run ESQL generation")
        return
    
    print("🚀 Starting Vector DB ESQL generation test...")
    
    # Get Vector DB content for ESQL generation
    vector_content = st.session_state.vector_pipeline.search_engine.get_agent_content("esql_generator")
    
    if not vector_content:
        print("❌ No Vector DB content found for 'esql_generator'")
        print("💡 Ensure Vector Knowledge Base contains ESQL-related content")
        return
    
    print(f"📊 Vector DB content retrieved: {len(vector_content)} characters")
    
    # Generate ESQL using Vector DB content
    result = generator.generate_esql_files(
        vector_content=vector_content,                      # ✅ Vector DB content
        esql_template={'path': 'ESQL_Template_Updated.ESQL'},  # ✅ Direct path for testing
        msgflow_content={'path': 'output/generated_messageflow.msgflow'},  # ✅ Direct path
        json_mappings={'path': 'component_mapping.json'}    # ✅ Direct path for testing
    )
    
    print(f"\n🎯 ESQL Generation Results:")
    print(f"✅ Status: Success")
    print(f"📊 Modules Generated: {result['total_modules']}")
    print(f"🧠 LLM Calls: {result['llm_calls_made']}")
    print(f"🎯 Generation Method: {result['generation_method']}")
    print(f"⚡ Token Management: {result['token_management']}")
    
    # Display individual modules
    if result.get('generated_modules'):
        print(f"\n📝 Generated ESQL Modules:")
        for i, module in enumerate(result['generated_modules'], 1):
            print(f"  {i}. {module['filename']} - {module['purpose']} ({module['content_length']} chars)")
    
    # Display processing summary
    if result.get('processing_summary'):
        summary = result['processing_summary']
        print(f"\n📈 Processing Summary:")
        print(f"  • Vector content: {summary.get('vector_content_length', 0)} characters")
        print(f"  • ESQL modules identified: {summary.get('esql_modules_identified', 0)}")
        print(f"  • Business logic rules: {summary.get('business_logic_rules', 0)}")
        print(f"  • Database operations: {summary.get('database_operations', 0)}")
        print(f"  • Transformations: {summary.get('transformations', 0)}")


if __name__ == "__main__":
    main()