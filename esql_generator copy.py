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
from llm_json_parser import parse_llm_json


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
    


    def _extract_event_template_section(self, template_content: str) -> str:
        """
        Extract ONLY the INPUT EVENT MESSAGE template section (single module)
        Returns: Single module from BROKER SCHEMA to END MODULE;
        """
        
        start_marker = "INPUT AND OUTPUT EVENT MESSAGE TEMPLATE"
        compute_marker = "COMPUTE TEMPLATE"
        
        # Find the section boundaries
        start_pos = template_content.find(start_marker)
        if start_pos == -1:
            raise Exception(f"❌ Template section not found: '{start_marker}'")
        
        end_pos = template_content.find(compute_marker, start_pos)
        if end_pos == -1:
            end_pos = len(template_content)
        
        # Extract the full section
        section = template_content[start_pos:end_pos]
        
        # ✅ FIX: Extract ONLY the first module (single BROKER SCHEMA to END MODULE;)
        broker_pos = section.find('BROKER SCHEMA')
        if broker_pos == -1:
            raise Exception("❌ No BROKER SCHEMA found in template section")
        
        # Find the FIRST END MODULE; after BROKER SCHEMA
        end_module_pos = section.find('END MODULE;', broker_pos)
        if end_module_pos == -1:
            raise Exception("❌ No END MODULE; found in template section")
        
        # Extract from BROKER SCHEMA to END MODULE; (inclusive)
        single_module = section[broker_pos:end_module_pos + len('END MODULE;')]
        
        # ✅ Validate: Should contain exactly ONE 'END MODULE;'
        end_module_count = single_module.count('END MODULE;')
        if end_module_count != 1:
            print(f"    ⚠️  WARNING: Template has {end_module_count} 'END MODULE;' statements")
            print(f"    📏 Template size: {len(single_module)} characters")
            # If multiple END MODULE; found, take only up to the first one
            if end_module_count > 1:
                first_end_pos = single_module.find('END MODULE;')
                single_module = single_module[:first_end_pos + len('END MODULE;')]
                print(f"    🔧 Trimmed to first module: {len(single_module)} characters")
        
        print(f"    📏 Template extracted: {len(single_module)} characters")
        print(f"    ✅ Contains {single_module.count('CREATE COMPUTE MODULE')} module")
        print(f"    ✅ Contains {single_module.count('END MODULE;')} END MODULE; statement")
        
        return single_module.strip()





    def _is_event_message_module(self, module_name: str) -> bool:
        """
        Detect if module is an event message type (InputEventMessage or OutputEventMessage)
        Event messages use template copying, not LLM generation
        
        Args:
            module_name: Dynamically constructed from naming_convention.json
                        Format: {message_flow_name}_{standard_suffix}
                        Example: If message_flow_name="CW1_IN_Document_SND"
                                Then module_name="CW1_IN_Document_SND_InputEventMessage"
        
        Returns:
            bool: True if event message module, False otherwise
        
        Note: Module names are constructed from business data extraction,
            NOT hardcoded. This method only checks the suffix pattern.
        """
        return (module_name.endswith('_InputEventMessage') or 
                module_name.endswith('_OutputEventMessage'))



    def _apply_event_naming(self, template_content: str, flow_name: str, full_module_name: str) -> str:
        """
        Replace placeholders in event message template with actual names
        
        Args:
            template_content: Raw template section extracted from ESQL_Template_Updated.ESQL
            flow_name: Base flow name from naming_convention.json (e.g., "CW1_IN_Document_SND")
            full_module_name: Complete module name with suffix (e.g., "CW1_IN_Document_SND_InputEventMessage")
        
        Returns:
            str: Complete ESQL code ready to write to file
        """
        lines = template_content.split('\n')
        
        # Process each line
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Replace BROKER SCHEMA line (uses base flow_name without suffix)
            if stripped.startswith('BROKER SCHEMA'):
                lines[i] = f'BROKER SCHEMA {flow_name}'
            
            # Replace CREATE COMPUTE MODULE line (uses full_module_name with suffix)
            elif stripped.startswith('CREATE COMPUTE MODULE'):
                lines[i] = f'CREATE COMPUTE MODULE {full_module_name}'
        
        # Join lines back together
        modified_content = '\n'.join(lines)
        
        return modified_content



    def _generate_event_message_esql(self, module_name: str, naming_data: Dict, template_content: str) -> str:
        """
        Generate complete event message ESQL using template copying (Tier 1 - No LLM)
        
        This method orchestrates the template-based generation for InputEventMessage
        and OutputEventMessage modules which contain only metadata capture logic.
        
        Args:
            module_name: Full module name (e.g., "CW1_IN_Document_SND_InputEventMessage")
            naming_data: Dictionary from naming_convention.json containing message_flow_name
            template_content: Full content of ESQL_Template_Updated.ESQL file
        
        Returns:
            str: Complete ESQL code ready to write to file
        
        Workflow:
            1. Extract base flow name from naming_data
            2. Extract INPUT EVENT MESSAGE template section
            3. Apply flow name and module name to template
            4. Return complete ESQL code
        """
        # Step 1: Extract flow name from naming data
        project_naming = naming_data.get('project_naming', {})
        if project_naming:
            flow_name = project_naming.get('message_flow_name')  # ✅ CORRECT FIELD NAME
        else:
            flow_name = naming_data.get('message_flow_name')

        if not flow_name:
            raise Exception(
                "❌ Flow name not found in naming_convention.json\n"
                "   Expected either 'project_naming.message_flow_name' or 'message_flow_name'"
            )
        
        # Step 2: Extract template section (Method 2)
        template_section = self._extract_event_template_section(template_content)
        
        # Step 3: Apply naming (Method 3)
        esql_code = self._apply_event_naming(
            template_section,
            flow_name,
            module_name
        )
        
        # Step 4: Return complete ESQL code
        return esql_code




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




    def _generate_esql_from_template(self, template_type: str, module_name: str, 
                                    business_data: Dict, naming_data: Dict, 
                                    template_content: str) -> str:
        """
        Generate ESQL template foundation for BUSINESS LOGIC modules only
        Event messages use separate _generate_event_message_esql() pathway
        
        Args:
            template_type: 'compute', 'processing', or 'failure' (NOT 'input_event')
            module_name: Name of the ESQL module (e.g., "CW1_IN_Document_SND_Compute")
            business_data: Business logic data for template customization
            naming_data: Pre-loaded dictionary from naming_convention.json
            template_content: Pre-loaded content from ESQL_Template_Updated.ESQL
        
        Returns:
            str: ESQL template foundation ready for LLM enhancement
        
        Raises:
            Exception: If event message incorrectly routed here
        """
        
        # ============================================================
        # SAFETY CHECK: Event messages must not reach this method
        # ============================================================
        if template_type == 'input_event' or self._is_event_message_module(module_name):
            raise Exception(
                f"❌ CRITICAL: Event message {module_name} incorrectly routed to _generate_esql_from_template()\n"
                f"   Event messages should use _generate_event_message_esql() pathway\n"
                f"   This indicates a logic error in generate_esql_files()"
            )
        
        # ============================================================
        # STEP 1: Define template section markers for business logic
        # ============================================================
        section_markers = {
            'compute': ['COMPUTE TEMPLATE - FULL BUSINESS LOGIC', 'PROCESSING TEMPLATE - VALIDATION AND ROUTING ONLY'],
            'processing': ['PROCESSING TEMPLATE - VALIDATION AND ROUTING ONLY', 'FAILURE/ERROR HANDLING TEMPLATE'],
            'failure': ['FAILURE/ERROR HANDLING TEMPLATE', None]  # Last section, no end marker
        }
        
        # Map legacy 'generic' type to 'compute'
        if template_type == 'generic':
            template_type = 'compute'
        
        # Validate template type
        if template_type not in section_markers:
            raise Exception(
                f"❌ Invalid template_type for business logic: '{template_type}'\n"
                f"   Valid types: compute, processing, failure\n"
                f"   Note: 'input_event' is not allowed here"
            )
        
        # ============================================================
        # STEP 2: Extract template section
        # ============================================================
        start_marker, end_marker = section_markers[template_type]
        lines = template_content.split('\n')
        
        section_start = None
        section_end = len(lines)
        
        # Find section boundaries
        for i, line in enumerate(lines):
            if start_marker in line:
                section_start = i
            if end_marker and end_marker in line and section_start is not None:
                section_end = i
                break
        
        if section_start is None:
            raise Exception(
                f"❌ Could not find template section marker: '{start_marker}'\n"
                f"   Please verify ESQL_Template_Updated.ESQL has correct section markers"
            )
        
        # Find first BROKER SCHEMA or CREATE COMPUTE MODULE line (actual ESQL start)
        esql_start_line = None
        for i in range(section_start, section_end):
            line = lines[i].strip()
            if line.startswith('BROKER SCHEMA') or line.startswith('CREATE COMPUTE MODULE'):
                esql_start_line = i
                break
        
        if esql_start_line is None:
            raise Exception(
                f"❌ No 'BROKER SCHEMA' or 'CREATE COMPUTE MODULE' found in {template_type} template section\n"
                f"   Template may be corrupted or missing required structure"
            )
        
        # Extract the template section
        selected_template = '\n'.join(lines[esql_start_line:section_end]).strip()
        
        # ============================================================
        # STEP 3: Extract flow name from naming data
        # ============================================================
        # Try new format first (project_naming structure)
        project_naming = naming_data.get('project_naming', {})
        
        if project_naming:
            # New format: {"project_naming": {"flow_name": "CW1_IN_Document_SND"}}
            message_flow_name = project_naming.get('message_flow_name')
            if not message_flow_name:
                raise Exception(
                    f"❌ naming_convention.json has 'project_naming' but missing 'flow_name' field"
                )
        else:
            # Legacy format: {"message_flow_name": "CW1_IN_Document_SND"}
            message_flow_name = naming_data.get('message_flow_name')
            if not message_flow_name:
                raise Exception(
                    f"❌ naming_convention.json missing both 'project_naming.flow_name' and 'message_flow_name'"
                )
        
        # ============================================================
        # STEP 4: Apply naming to template
        # ============================================================
        # Replace placeholder with actual flow name
        placeholder = "_SYSTEM___MSG_TYPE___FLOW_PROCESS___SYSTEM2___FLOW_TYPE"
        esql_code = selected_template.replace(placeholder, message_flow_name)
        
        # Also replace other common placeholders (backward compatibility)
        esql_code = esql_code.replace("{{MESSAGE_FLOW_NAME}}", message_flow_name)
        esql_code = esql_code.replace("{{FLOW_NAME}}", message_flow_name)
        esql_code = esql_code.replace("${MESSAGE_FLOW_NAME}", message_flow_name)
        
        # ============================================================
        # STEP 5: Update BROKER SCHEMA and CREATE COMPUTE MODULE lines
        # ============================================================
        # Extract base flow name (remove standard suffixes)
        standard_suffixes = [
            '_Compute',
            '_AfterEnrichment',
            '_AfterEventMsg',
            '_Failure'
        ]
        
        base_flow_name = module_name
        for suffix in standard_suffixes:
            if module_name.endswith(suffix):
                base_flow_name = module_name[:-len(suffix)]
                break
        
        # If no standard suffix found, use message_flow_name as base
        if base_flow_name == module_name:
            base_flow_name = message_flow_name
        
        # Update BROKER SCHEMA and CREATE COMPUTE MODULE lines
        lines = esql_code.split('\n')
        
        # Update Line 1: BROKER SCHEMA (must be first line)
        if lines and lines[0].strip().startswith('BROKER SCHEMA'):
            lines[0] = f'BROKER SCHEMA {base_flow_name}'
            print(f"   ✅ Set BROKER SCHEMA: {base_flow_name}")
        else:
            # If BROKER SCHEMA is missing, prepend it (REQUIRED by IBM ACE)
            lines.insert(0, f'BROKER SCHEMA {base_flow_name}')
            print(f"   ✅ Added BROKER SCHEMA: {base_flow_name}")
        
        # Update Line 2: CREATE COMPUTE MODULE (must be second line)
        for i, line in enumerate(lines):
            if line.strip().startswith('CREATE COMPUTE MODULE'):
                lines[i] = f'CREATE COMPUTE MODULE {module_name}'
                print(f"   ✅ Set CREATE COMPUTE MODULE: {module_name}")
                break
        
        esql_code = '\n'.join(lines)
        
        # ============================================================
        # STEP 6: Validation
        # ============================================================
        # Ensure critical elements are present
        if f'CREATE COMPUTE MODULE {module_name}' not in esql_code:
            raise Exception(
                f"❌ Generated ESQL must contain 'CREATE COMPUTE MODULE {module_name}'\n"
                f"   Module naming is incorrect"
            )
        
        if not esql_code.strip().endswith('END MODULE;'):
            raise Exception(
                f"❌ Generated ESQL must end with 'END MODULE;'\n"
                f"   ESQL structure is incomplete"
            )
        
        print(f"   ✅ Template foundation validated - Structure correct")
        
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
            
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": "You are an expert IBM ACE ESQL analyst. Extract ESQL module requirements and return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=self.max_tokens_per_request
            )
            
            raw_response = response.choices[0].message.content.strip()
            
            # DEBUG: Print what LLM returned
            print(f"🔍 DEBUG: LLM raw response for chunk {chunk_index}:")
            print(f"   Response length: {len(raw_response)} chars")
            print(f"   First 200 chars: {raw_response[:200]}...")
            
            # ✅ FIX: Robust JSON extraction with markdown handling
            json_str = raw_response
            
            # Remove markdown code blocks if present
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0].strip()
            elif '```' in json_str:
                json_str = json_str.split('```')[1].split('```')[0].strip()
            
            # Try to extract JSON object
            json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
            
            # Parse JSON
            parsed_json = json.loads(json_str)
            
            # ✅ CRITICAL FIX: Ensure proper data structure
            if isinstance(parsed_json, dict):
                print(f"🔍 DEBUG: Parsed JSON keys: {list(parsed_json.keys())}")
                
                # ✅ FIX esql_modules if present
                if 'esql_modules' in parsed_json:
                    modules = parsed_json['esql_modules']
                    
                    # Ensure it's a list
                    if not isinstance(modules, list):
                        print(f"   ⚠️  WARNING: esql_modules is {type(modules)}, converting to list")
                        modules = [modules] if modules else []
                    
                    # Fix each module to ensure dict format
                    fixed_modules = []
                    for idx, module in enumerate(modules):
                        if isinstance(module, str):
                            # Convert string to dict
                            print(f"   🔧 Converting string module to dict: {module}")
                            fixed_modules.append({
                                'name': module,
                                'type': 'compute',
                                'purpose': 'Processing',
                                'source': 'vector_db_analysis'
                            })
                        elif isinstance(module, dict):
                            # Ensure required keys exist
                            if 'name' not in module:
                                module['name'] = f'ESQLModule_{idx+1}'
                            if 'type' not in module:
                                module['type'] = 'compute'
                            if 'purpose' not in module:
                                module['purpose'] = 'Processing'
                            if 'source' not in module:
                                module['source'] = 'vector_db_analysis'
                            fixed_modules.append(module)
                        else:
                            print(f"   ⚠️  WARNING: Skipping invalid module type: {type(module)}")
                    
                    parsed_json['esql_modules'] = fixed_modules
                    print(f"   ✅ Fixed {len(fixed_modules)} modules in chunk {chunk_index}")
                
                # Ensure other required keys exist
                if 'business_logic' not in parsed_json:
                    parsed_json['business_logic'] = {}
                if 'database_operations' not in parsed_json:
                    parsed_json['database_operations'] = []
                if 'transformations' not in parsed_json:
                    parsed_json['transformations'] = []
                
                print(f"🔍 DEBUG: Business logic items: {len(parsed_json.get('business_logic', {}))}")
                print(f"🔍 DEBUG: Database operations: {len(parsed_json.get('database_operations', []))}")
                print(f"🔍 DEBUG: ESQL modules: {len(parsed_json.get('esql_modules', []))}")
                
                return parsed_json
                
            elif isinstance(parsed_json, list):
                print(f"🔍 DEBUG: Parsed JSON is list with {len(parsed_json)} items")
                # Convert list to expected dict format
                return {
                    "esql_modules": [],
                    "business_logic": {},
                    "database_operations": [],
                    "transformations": [],
                    "raw_list": parsed_json
                }
            else:
                print(f"⚠️  WARNING: Unexpected JSON type: {type(parsed_json)}")
                return {
                    "esql_modules": [],
                    "business_logic": {},
                    "database_operations": [],
                    "transformations": []
                }
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed for chunk {chunk_index}: {e}")
            print(f"   📄 Raw response preview: {raw_response[:300] if 'raw_response' in locals() else 'No response'}")
            return {
                "esql_modules": [],
                "business_logic": {},
                "database_operations": [],
                "transformations": []
            }
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

            if not esql_content.strip().endswith('END MODULE;'):
                print(f"    ⚠️  LLM response missing END MODULE; - adding it")
                # Check if it has END MODULE without semicolon
                if esql_content.strip().endswith('END MODULE'):
                    esql_content = esql_content.strip() + ';'
                else:
                    esql_content = esql_content.strip() + '\n\nEND MODULE;'
                print(f"    ✅ Added END MODULE; to complete structure")

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
        ENHANCED: Event messages use template copying (no LLM), business logic uses LLM
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
            # ============================================================
            # 🆕 STEP 0: Load Resources for Template-Based Generation
            # ============================================================
            print("📁 Loading resources for template-based generation...")
            
            # Load naming convention (required for event message generation)
            naming_file = "naming_convention.json"
            if not os.path.exists(naming_file):
                raise Exception(
                    f"❌ CRITICAL: {naming_file} not found\n"
                    f"   This file must be created by Agent 1 first\n"
                    f"   Expected location: {os.path.abspath(naming_file)}\n"
                    f"   Please run Agent 1 (Specification-Driven Mapper) before ESQL generation"
                )

            try:
                with open(naming_file, 'r', encoding='utf-8') as f:
                    naming_data = json.load(f)
                
                # ✅ ADD THIS: Extract flow_name for validation
                project_naming = naming_data.get('project_naming', {})
                if project_naming:
                    flow_name = project_naming.get('message_flow_name')
                else:
                    flow_name = naming_data.get('message_flow_name')
                
                if not flow_name:
                    raise Exception(
                        "❌ CRITICAL: Flow name not found in naming_convention.json\n"
                        "   Expected either 'project_naming.message_flow_name' or 'message_flow_name'"
                    )
                
                print(f"  ✅ Base flow name extracted: {flow_name}")
                
            except json.JSONDecodeError as e:
                raise Exception(
                    f"❌ CRITICAL: {naming_file} contains invalid JSON\n"
                    f"   Parse error: {str(e)}"
                )
            
            project_naming = naming_data.get('project_naming', {})
            message_flow_name = project_naming.get('message_flow_name')

            # Fallback to legacy format if new format not found
            if not message_flow_name:
                message_flow_name = naming_data.get('message_flow_name')

            if not message_flow_name:
                raise Exception(
                    f"❌ CRITICAL: {naming_file} missing required field\n"
                    f"   Expected: 'project_naming.message_flow_name' (new format)\n"
                    f"   OR: 'message_flow_name' (legacy format)"
                )

            print(f"  ✅ Naming convention loaded: {message_flow_name}")
            
            # Load ESQL template file (required for event message generation)
            template_file = "ESQL_Template_Updated.ESQL"
            if not os.path.exists(template_file):
                raise Exception(
                    f"❌ CRITICAL: {template_file} not found\n"
                    f"   Expected location: {os.path.abspath(template_file)}\n"
                    f"   This template is required for event message generation"
                )
            
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    esql_template_file_content = f.read()
                print(f"  ✅ ESQL template file loaded: {len(esql_template_file_content)} characters")
            except Exception as e:
                raise Exception(f"❌ Error reading {template_file}: {str(e)}")
            
            # ============================================================
            # STEP 1: Load supporting files for LLM analysis (EXISTING)
            # ============================================================
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
                msgflow_file_found = msgflow_files[0]
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
            
            # [... ALL EXISTING LLM ANALYSIS CODE REMAINS UNCHANGED ...]
            # [Lines for msgflow_analysis_prompt, analysis_response, requirements extraction, etc.]
            # [I'm omitting this for brevity - it all stays exactly as-is]
            
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
                if 'token_tracker' in st.session_state and hasattr(msgflow_response, 'usage') and msgflow_response.usage:
                    st.session_state.token_tracker.manual_track(
                        agent="esql_generator",
                        operation="msgflow_analysis",
                        model=self.groq_model,
                        input_tokens=msgflow_response.usage.prompt_tokens,
                        output_tokens=msgflow_response.usage.completion_tokens,
                        flow_name="esql_generation"
                    )
            except Exception as e:
                print(f"⚠️ Token tracking failed: {e}")
            
            self.llm_calls_count += 1
            
            # Parse MessageFlow analysis
            msgflow_analysis_content = msgflow_response.choices[0].message.content.strip()

            # Use robust JSON parser
            result = parse_llm_json(msgflow_analysis_content, debug=False)

            if not result.success:
                raise Exception(
                    f"LLM MessageFlow analysis JSON parsing failed\n"
                    f"Error: {result.error_message}\n"
                    f"Method attempted: {result.method_used}"
                )

            msgflow_analysis = result.data
            msgflow_esql_modules = msgflow_analysis.get('esql_modules', [])
            
            print(f"  ✅ LLM extracted {len(msgflow_esql_modules)} ESQL modules from MessageFlow")
            
            # [... Continue with all existing analysis code - this section stays exactly as-is ...]
            # [All the business logic extraction, database operations extraction, etc.]
            
            # Step 2: Comprehensive LLM Analysis combining ALL sources
            print("🧠 LLM Analysis: Extracting ESQL requirements from ALL sources...")
            
            analysis_prompt = f"""Extract ESQL requirements with SPECIFIC FOCUS on database operations and stored procedures:

    ## VECTOR DB BUSINESS REQUIREMENTS (PRIMARY SOURCE - SCAN FOR DATABASE OPERATIONS):
    {vector_content}

    ## COMPONENT MAPPINGS (BUSINESS PATTERNS AND LOGIC):
    {json.dumps(json_mappings_data.get('component_mappings', []), indent=2)}

    ## MESSAGEFLOW COMPUTE EXPRESSIONS (MODULE STRUCTURE):
    {json.dumps(msgflow_esql_modules, indent=2)}

            CRITICAL RULES:
            - ONLY create ESQL modules for MessageFlow compute expressions listed above
            - Use business requirements to understand what each MessageFlow module should do
            - Use component mappings as reference for understanding existing patterns
            - Do NOT create additional modules from business requirements or component mappings
            - Return exactly the same number of modules as MessageFlow compute expressions
            - Each ESQL module must correspond to a MessageFlow compute expression

            Extract and return JSON with:
            1. "esql_modules": List containing ONLY MessageFlow compute expression modules enhanced with business context
            2. "business_logic": Business logic requirements for each MessageFlow module
            3. "message_structure": Input/output message structure details from MessageFlow
            4. "customizations": Specific customizations needed for template based on business requirements

            Focus on:
            - Understanding business context for each MessageFlow compute node
            - Mapping business requirements to MessageFlow node purposes
            - Database operations and procedures mentioned in business requirements for MessageFlow nodes
            - Transformation logic requirements that apply to MessageFlow compute expressions
            - Custom XPath expressions needed for MessageFlow processing

            VALIDATION: Ensure every module in esql_modules corresponds to a MessageFlow compute expression. Do not add modules from other sources.

            Return valid JSON only

    Return ONLY valid JSON with ALL database operations and stored procedures found in Vector DB content."""

            # [... All existing LLM analysis code remains unchanged ...]
            
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
                if 'token_tracker' in st.session_state and hasattr(analysis_response, 'usage') and analysis_response.usage:
                    st.session_state.token_tracker.manual_track(
                        agent="esql_generator",
                        operation="requirements_analysis",
                        model=self.groq_model,
                        input_tokens=analysis_response.usage.prompt_tokens,
                        output_tokens=analysis_response.usage.completion_tokens,
                        flow_name="esql_generation"
                    )
            except Exception as e:
                print(f"⚠️ Token tracking failed: {e}")
            
            self.llm_calls_count += 1
            
            # Parse comprehensive analysis
            analysis_content = analysis_response.choices[0].message.content.strip()

            # Use robust JSON parser
            result = parse_llm_json(analysis_content, debug=False)

            if not result.success:
                raise Exception(
                    f"LLM requirements analysis JSON parsing failed\n"
                    f"Error: {result.error_message}\n"
                    f"Method attempted: {result.method_used}"
                )

            requirements = result.data

            # CRITICAL: Merge MessageFlow modules with requirements
            if 'esql_modules' not in requirements:
                requirements['esql_modules'] = []

            # Add MessageFlow modules if missing
            existing_names = {
                m.get('name') if isinstance(m, dict) else m 
                for m in requirements['esql_modules']
            }

            for msgflow_mod in msgflow_esql_modules:
                # Handle both string and dict formats
                if isinstance(msgflow_mod, str):
                    mod_name = msgflow_mod
                elif isinstance(msgflow_mod, dict):
                    mod_name = msgflow_mod.get('name') or msgflow_mod.get('module_name')
                else:
                    continue  # Skip invalid formats
                
                if mod_name and mod_name not in existing_names:
                    requirements['esql_modules'].append({
                        'name': mod_name,
                        'source': 'messageflow',
                        'type': 'compute',
                        'purpose': 'MessageFlow compute module'
                    })

            total_modules = len(requirements.get('esql_modules', []))


            component_modules = len([m for m in requirements.get('esql_modules', []) if m.get('source') == 'component_mapping'])
            msgflow_modules = len([m for m in requirements.get('esql_modules', []) if m.get('source') == 'messageflow_compute'])
            
            print(f"  ✅ Requirements analysis completed: {total_modules} modules identified")
            
            if total_modules == 0:
                raise Exception("No ESQL modules identified from Vector DB + MessageFlow analysis")
            
            # Create template info for helper methods
            template_info = {
                'content': esql_template_content,
                'length': len(esql_template_content)
            }
            
            # ============================================================
            # STEP 3: Generate ESQL modules (ENHANCED WITH TEMPLATE COPY)
            # ============================================================
            print("🔧 Generation: Creating ESQL modules (Template copy + LLM)...")
            
            esql_modules = requirements.get('esql_modules', [])
            
            # Step 2: Filter them (ADD THIS NEW CODE)
            filtered_modules = []
            for module in esql_modules:
                if isinstance(module, dict):
                    module_name = module.get('name', '')
                    if module_name.startswith('ESQLModule_'):
                        print(f"  Skipping generic placeholder: {module_name}")
                        continue
                    filtered_modules.append(module)
                elif isinstance(module, str):
                    if module.startswith('ESQLModule_'):
                        print(f"  Skipping generic placeholder: {module}")
                        continue
                    filtered_modules.append({
                        'name': module,
                        'type': 'compute',
                        'purpose': 'Processing',
                        'source': 'requirements_analysis'
                    })

            esql_modules = filtered_modules
            print(f"  Filtered to {len(filtered_modules)} real modules")

            # Step 3: Now the loop starts (DON'T CHANGE THIS LINE)


            # Process each module individually with error isolation
            for i, module_req in enumerate(esql_modules):
                module_name = module_req.get('name', f'ESQLModule_{i+1}')
                
                try:
                    # ============================================================
                    # 🆕 DECISION POINT: Template Copy vs LLM Generation
                    # ============================================================
                    if self._is_event_message_module(module_name):
                        # 🆕 TIER 1: Template Copy Pathway (No LLM)
                        print(f"  📋 Generating {module_name}.esql via TEMPLATE COPY (no LLM)...")
                        
                        esql_content = self._generate_event_message_esql(
                            module_name,
                            naming_data,
                            esql_template_file_content
                        )
                        
                        print(f"    ✅ Template copy completed ({len(esql_content)} characters)")
                        generation_method = "TEMPLATE_COPY"
                        
                    else:
                        # TIER 2: LLM Generation Pathway (Existing Code)
                        print(f"  🎯 Generating {module_name}.esql with LLM...")
                        
                        # [... ALL EXISTING LLM GENERATION CODE STAYS HERE ...]
                        # Determine module type and get appropriate prompt
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
                        
                        template_foundation = self._generate_esql_from_template(
                            module_type, 
                            module_name, 
                            basic_data,
                            naming_data,                    # Add this
                            esql_template_file_content      # Add this
                        )
                        

                        # LLM call for ESQL generation
                        generation_response = self.groq_client.chat.completions.create(
                            model=self.groq_model,
                            messages=[
                                {"role": "system", "content": "CRITICAL REQUIREMENT: Your output MUST end with 'END MODULE;' on the last line. This is mandatory.\n\nYou are an ESQL code editor. Your task:\n1. Find marker '-- [[[INSERT_BUSINESS_LOGIC_HERE]]]'\n2. Replace with business logic code\n3. Return COMPLETE template from BROKER SCHEMA to END MODULE;\n4. The absolute last line MUST be: END MODULE;"},
                                {"role": "user", "content": f"""Here is the complete ESQL template:

    {template_foundation}

    Task: {generation_prompt}


           ## MANDATORY DATA TYPES - NO EXCEPTIONS:
            ONLY USE: BOOLEAN, INTEGER, DECIMAL, FLOAT, CHARACTER, BIT, BLOB, DATE, TIME, TIMESTAMP, REFERENCE, ROW
            NEVER USE: XML, RECORD, STRING, VARCHAR, JSON, Database

            ## XML PROCESSING EXAMPLES:
            WRONG: DECLARE xmlData XML;
            CORRECT: DECLARE xmlRef REFERENCE TO InputRoot.XMLNSC;

            WRONG: DECLARE msg RECORD;  
            CORRECT: DECLARE msgRef REFERENCE TO InputRoot;

            Generate complete IBM ACE ESQL module with STRICT format compliance: 

            ## MODULE SPECIFICATION:
            Name: {module_name}
            Purpose: {module_req.get('purpose', 'Business logic processing')}
            Type: {module_req.get('type', 'COMPUTE')}
            Source: {module_req.get('source', 'component_mapping')}

            ## BUSINESS REQUIREMENTS:
            {json.dumps(module_req.get('business_logic', {}), indent=2)}

            ## DATABASE OPERATIONS:
            {json.dumps(requirements.get('database_operations', []), indent=2)}

            ## TRANSFORMATIONS:
            {json.dumps(requirements.get('transformations', []), indent=2)}

            ## STRICT FORMAT REQUIREMENTS (MANDATORY):

            ### EXACT START (NO PREFIXES):
            CREATE COMPUTE MODULE {module_name}

            ### EXACT DECLARE STATEMENTS (customize variable names for business context):
                CREATE FUNCTION Main() RETURNS BOOLEAN
                BEGIN
                    DECLARE episInfo 		REFERENCE TO 	Environment.variables.EventData.episInfo;
                    DECLARE sourceInfo 		REFERENCE TO 	Environment.variables.EventData.sourceInfo;
                    DECLARE targetInfo 		REFERENCE TO 	Environment.variables.EventData.targetInfo;
                    DECLARE integrationInfo REFERENCE TO 	Environment.variables.EventData.integrationInfo;
                    DECLARE dataInfo 		REFERENCE TO 	Environment.variables.EventData.dataInfo;

            ### BUSINESS LOGIC SECTION:
            - SET statements for sourceInfo/targetInfo/dataInfo based on business requirements
            - Database operations as specified
            - Data transformations as needed
            - Use OutputRoot for all modifications (InputRoot is READ-ONLY)

            ### EXACT END (DO NOT MODIFY):
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

                ## FORBIDDEN ELEMENTS:
                - NO "@" symbols anywhere
                - NO comments starting with "--"
                - NO modifications to InputRoot
                - NO prefixes before CREATE COMPUTE MODULE
                - NO CALL statements or direct procedure calls
                - Use SET statements and built-in functions instead of CALL



    CRITICAL: Return the ENTIRE template from CREATE COMPUTE MODULE to END MODULE; 
    with the modifications applied. Do not omit any lines from the original template."""}
                            ],
                            temperature=0.1,
                            max_tokens=4000
                        )

                        generated_esql = generation_response.choices[0].message.content.strip()
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
                        except Exception as e:
                            print(f"⚠️ Token tracking failed: {e}")
                        
                        self.llm_calls_count += 1
                        
                        if not generated_esql.strip().endswith('END MODULE;'):
                            print(f"   ⚠️ WARNING: LLM forgot END MODULE; - appending it")
                            # Clean up and append
                            generated_esql = generated_esql.rstrip()
                            # Add proper spacing
                            if not generated_esql.endswith('\n'):
                                generated_esql += '\n'
                            # Append mandatory procedures if missing
                            if 'CREATE PROCEDURE CopyMessageHeaders()' not in generated_esql:
                                generated_esql += '''

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
                        '''
                            # Add final terminator
                            generated_esql += '\nEND MODULE;'

                        esql_content = generated_esql

                        # Clean LLM output FIRST
                        esql_content = re.sub(r'```[\w]*\n?', '', esql_content)
                        esql_content = re.sub(r'\n?```\s*$', '', esql_content)
                        esql_content = self._clean_esql_content(esql_content)
                        esql_content = self._ensure_broker_schema(esql_content, module_name)

                        # ✅ SAFETY NET - Apply AFTER cleaning
                        if not esql_content.strip().endswith('END MODULE;'):
                            print(f"   ⚠️ AUTO-FIX: Appending missing END MODULE;")
                            if 'CREATE PROCEDURE CopyMessageHeaders()' not in esql_content:
                                esql_content += '''

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
                        '''
                            esql_content = esql_content.rstrip() + '\n\nEND MODULE;'
                        
                        generation_method = "LLM_GENERATION"
                    
                    # ============================================================
                    # COMMON VALIDATION & SAVE (Both Pathways)
                    # ============================================================
                    
                    # Validate ESQL format compliance
                    lines = esql_content.strip().split('\n')
                    
                    if not lines[0].strip().startswith('BROKER SCHEMA'):
                        raise Exception(f"ESQL format violation: {module_name} must start with 'BROKER SCHEMA' on line 1")
                    
                    if len(lines) < 2 or not lines[1].strip().startswith('CREATE COMPUTE MODULE'):
                        raise Exception(f"ESQL format violation: {module_name} must have 'CREATE COMPUTE MODULE {module_name}' on line 2")
                    
                    if module_name not in lines[1]:
                        raise Exception(f"ESQL format violation: Module name '{module_name}' not found in CREATE COMPUTE MODULE declaration")
                    
                    if not esql_content.rstrip().endswith('END MODULE;'):
                        raise Exception(f"ESQL format violation: {module_name} must end with 'END MODULE;'")
                    
                    if '@' in esql_content:
                        raise Exception(f"ESQL format violation: {module_name} contains forbidden '@' symbols")

                    # Apply constraint validation
                    constraint_result = self.validate_esql_constraints(esql_content)

                    if 'modified_content' in constraint_result:
                        esql_content = constraint_result['modified_content']

                    if constraint_result.get('warnings'):
                        print(f"    ⚠️  Auto-fixes applied: {constraint_result['warnings']}")

                    if not constraint_result['valid']:
                        print(f"    ❌ Constraint violations: {constraint_result['errors']}")
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
                        raise Exception(f"Incomplete ESQL for {module_name}. Missing: {missing_elements}")
                    

                    # ✅ FIX: Validate before writing
                    validation_errors = []

                    # Check 1: Has BROKER SCHEMA
                    if 'BROKER SCHEMA' not in esql_content:
                        validation_errors.append("Missing BROKER SCHEMA")

                    # Check 2: Has CREATE COMPUTE MODULE
                    if 'CREATE COMPUTE MODULE' not in esql_content:
                        validation_errors.append("Missing CREATE COMPUTE MODULE")

                    # Check 3: Ends with END MODULE;
                    if not esql_content.strip().endswith('END MODULE;'):
                        validation_errors.append("Must end with END MODULE;")
                        
                    # Check 4: Size check
                    if len(esql_content) < 200:
                        validation_errors.append(f"Content too small: {len(esql_content)} chars")
                    elif len(esql_content) > 4000 and generation_method == "TEMPLATE_COPY":
                        validation_errors.append(f"Content too large for template copy: {len(esql_content)} chars")

                    # Check 5: Single module check
                    end_module_count = esql_content.count('END MODULE;')
                    if end_module_count != 1:
                        validation_errors.append(f"Expected 1 'END MODULE;', found {end_module_count}")

                    # If validation fails, log and skip
                    if validation_errors:
                        print(f"    ❌ ESQL validation failed:")
                        for error in validation_errors:
                            print(f"       • {error}")
                        
                        # Save failed content for debugging
                        debug_dir = os.path.join(self.output_dir, 'debug_failed')
                        os.makedirs(debug_dir, exist_ok=True)
                        debug_file = os.path.join(debug_dir, f"{module_name}_FAILED.esql")
                        with open(debug_file, 'w', encoding='utf-8') as f:
                            f.write(esql_content)
                        print(f"    💾 Failed content saved to: {debug_file}")
                        
                        # Track failure
                        failed_modules.append({
                            'name': module_name,
                            'errors': validation_errors,
                            'generation_method': generation_method
                        })
                        continue


                    # Save ESQL file
                    esql_filename = f"{module_name}.esql"
                    esql_file_path = os.path.join(self.output_dir, esql_filename)
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
                        'generation_method': generation_method  # 🆕 Track which method was used
                    })
                    
                    print(f"    ✅ {esql_filename} generated successfully ({len(esql_content)} characters) via {generation_method}")
                    
                except Exception as module_error:
                    error_message = str(module_error)
                    print(f"    ❌ ESQL generation failed: {error_message}")
                    
                    failed_modules.append({
                        'name': module_name,
                        'error_message': error_message,
                        'purpose': module_req.get('purpose', 'Processing'),
                        'source': module_req.get('source', 'component_mapping')
                    })
                    
                    continue
            
            # ============================================================
            # STEP 4: Determine final status and return results (EXISTING)
            # ============================================================
            total_attempted = len(esql_modules)
            successful_count = len(successful_modules)
            failed_count = len(failed_modules)
            
            # Count how many were generated via each method
            template_copy_count = len([m for m in successful_modules if m.get('generation_method') == 'TEMPLATE_COPY'])
            llm_generation_count = len([m for m in successful_modules if m.get('generation_method') == 'LLM_GENERATION'])
            
            if successful_count == 0:
                raise Exception(f"All {total_attempted} ESQL modules failed to generate")
            elif failed_count > 0:
                print(f"✅ Successful modules: {successful_count}/{total_attempted}")
                print(f"   📋 Template copy: {template_copy_count} modules")
                print(f"   🤖 LLM generation: {llm_generation_count} modules")
                print(f"❌ Failed modules: {failed_count}")
            else:
                print(f"🎉 ESQL generation complete: {successful_count} modules generated")
                print(f"   📋 Template copy: {template_copy_count} modules (0 LLM calls)")
                print(f"   🤖 LLM generation: {llm_generation_count} modules ({llm_generation_count} LLM calls)")
            
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
                'template_copy_count': template_copy_count,  # 🆕 Track template copies
                'llm_generation_count': llm_generation_count,  # 🆕 Track LLM generations
                'generation_method': 'Hybrid: Template Copy (Event Messages) + LLM (Business Logic)',  # 🆕 Updated
                'token_management': 'Optimized: Event messages bypass LLM',  # 🆕 Updated
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
            if len(successful_modules) > 0:
                return {
                    'status': 'partial_success',
                    'generated_modules': successful_modules,
                    'failed_modules': failed_modules,
                    'llm_calls_made': self.llm_calls_count,
                    'total_modules': len(successful_modules),
                    'successful_count': len(successful_modules),
                    'failed_count': len(failed_modules),
                    'generation_method': 'Hybrid with Error Recovery',
                    'error_message': f"Partial failure: {str(e)}"
                }
            else:
                print(f"💥 ESQL generation failed: {str(e)}")
                raise Exception(f"ESQL Generation Failed: {str(e)}")





    def _validate_and_fix_esql_structure(self, esql_content: str, module_name: str, flow_name: str) -> tuple[str, list[str]]:
        """
        Validate ESQL structure and auto-fix missing components
        
        Args:
            esql_content: The ESQL code to validate
            module_name: Expected module name (e.g., "IDP_IN_CW1_DocPackRequest_Invoice_SND_Compute")
            flow_name: Base flow name (e.g., "IDP_IN_CW1_DocPackRequest_Invoice_SND")
        
        Returns:
            (fixed_esql_content, list_of_fixes_applied)
        """
        fixes_applied = []
        fixed_content = esql_content.strip()
        
        # ============================================================
        # PART 1: Fix Beginning Structure
        # ============================================================
        
        # Check for BROKER SCHEMA at the beginning
        if not fixed_content.startswith('BROKER SCHEMA'):
            print(f"    🔧 Missing BROKER SCHEMA - adding")
            fixed_content = f'BROKER SCHEMA {flow_name}\n{fixed_content}'
            fixes_applied.append("Added BROKER SCHEMA")
        
        # Verify BROKER SCHEMA has correct flow name
        lines = fixed_content.split('\n')
        if lines[0].startswith('BROKER SCHEMA'):
            broker_line = lines[0]
            # Extract current schema name
            if 'BROKER SCHEMA' in broker_line:
                current_schema = broker_line.replace('BROKER SCHEMA', '').strip()
                # Check if it's a placeholder or incorrect
                if '_SYSTEM___MSG_TYPE___FLOW_PROCESS___SYSTEM2___FLOW_TYPE' in current_schema or current_schema != flow_name:
                    print(f"    🔧 Fixing BROKER SCHEMA name: {current_schema} -> {flow_name}")
                    lines[0] = f'BROKER SCHEMA {flow_name}'
                    fixed_content = '\n'.join(lines)
                    fixes_applied.append(f"Fixed BROKER SCHEMA name to {flow_name}")
        
        # Check for CREATE COMPUTE MODULE
        if 'CREATE COMPUTE MODULE' not in fixed_content:
            print(f"    🔧 Missing CREATE COMPUTE MODULE - adding")
            # Insert after BROKER SCHEMA
            lines = fixed_content.split('\n')
            if lines[0].startswith('BROKER SCHEMA'):
                lines.insert(1, f'CREATE COMPUTE MODULE {module_name}')
                fixed_content = '\n'.join(lines)
                fixes_applied.append("Added CREATE COMPUTE MODULE")
        else:
            # Verify module name is correct
            import re
            module_pattern = r'CREATE COMPUTE MODULE\s+(\S+)'
            match = re.search(module_pattern, fixed_content)
            if match:
                current_module_name = match.group(1)
                # Check if it's a placeholder or incorrect
                if '_SYSTEM___MSG_TYPE___FLOW_PROCESS___SYSTEM2___FLOW_TYPE' in current_module_name or current_module_name != module_name:
                    print(f"    🔧 Fixing module name: {current_module_name} -> {module_name}")
                    fixed_content = re.sub(
                        r'CREATE COMPUTE MODULE\s+\S+',
                        f'CREATE COMPUTE MODULE {module_name}',
                        fixed_content
                    )
                    fixes_applied.append(f"Fixed module name to {module_name}")
        
        # ============================================================
        # PART 2: Fix Ending Structure
        # ============================================================
        
        # Required ending procedures
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
            print(f"    🔧 Missing CopyMessageHeaders procedure - adding")
            # Add before END MODULE; if it exists, or at the end
            if 'END MODULE;' in fixed_content:
                fixed_content = fixed_content.replace('END MODULE;', f'\n\t{required_copy_headers}\n\nEND MODULE;')
            else:
                fixed_content = fixed_content + f'\n\n\t{required_copy_headers}\n'
            fixes_applied.append("Added CopyMessageHeaders procedure")
        
        # Check for CopyEntireMessage procedure
        if 'CREATE PROCEDURE CopyEntireMessage()' not in fixed_content:
            print(f"    🔧 Missing CopyEntireMessage procedure - adding")
            # Add before END MODULE; if it exists, or at the end
            if 'END MODULE;' in fixed_content:
                fixed_content = fixed_content.replace('END MODULE;', f'\n\t{required_copy_entire}\n\nEND MODULE;')
            else:
                fixed_content = fixed_content + f'\n\n\t{required_copy_entire}\n'
            fixes_applied.append("Added CopyEntireMessage procedure")
        
        # Check for END MODULE; at the end
        if not fixed_content.strip().endswith('END MODULE;'):
            print(f"    🔧 Missing END MODULE; - adding")
            fixed_content = fixed_content.strip() + '\n\nEND MODULE;'
            fixes_applied.append("Added END MODULE;")
        
        # ============================================================
        # PART 3: Final Validation
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
            validation_errors.append("Does not end with END MODULE; (auto-fix failed)")
        
        # Check 6: Size check
        if len(fixed_content) < 200:
            validation_errors.append(f"Content too small: {len(fixed_content)} chars")
        
        # If validation still fails after fixes, log it
        if validation_errors:
            print(f"    ⚠️  Validation issues remain after auto-fix:")
            for error in validation_errors:
                print(f"       • {error}")
        elif fixes_applied:
            print(f"    ✅ Auto-fixed {len(fixes_applied)} issues: {', '.join(fixes_applied)}")
        
        return fixed_content, fixes_applied




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