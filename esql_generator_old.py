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
        
        # âœ… ADD: Initialize missing attributes
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
        
        print(f"ðŸ§® Token Budget: Total={self.max_total_tokens}, Base={base_tokens}, Available={available_input_tokens}")
        
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
                print(f"âœ… Added {data_type}: {data_tokens} tokens (total: {current_tokens})")
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    print(f"ðŸ“¦ Created chunk with {current_tokens} tokens")
                
                # Start new chunk
                current_chunk = {data_type: data_content}
                current_tokens = data_tokens
                print(f"ðŸ†• Started new chunk with {data_type}: {data_tokens} tokens")
        
        if current_chunk:
            chunks.append(current_chunk)
            print(f"ðŸ“¦ Final chunk with {current_tokens} tokens")
        
        print(f"ðŸ”„ Created {len(chunks)} chunks for processing")
        return chunks
    
    def _get_system_prompt(self) -> str:
        """
        Enhanced system prompt with all requirements
        """
        return """You are an expert IBM ACE ESQL developer specializing in generating production-ready ESQL modules. You MUST follow these CRITICAL requirements exactly:
## âš ï¸ CRITICAL DATA TYPE RESTRICTIONS (READ FIRST) âš ï¸

### APPROVED DATA TYPES ONLY - NO EXCEPTIONS:
You MUST ONLY use these data types in ALL DECLARE statements:
âœ… BOOLEAN, INTEGER, DECIMAL, FLOAT, CHARACTER
âœ… BIT, BLOB, DATE, TIME, TIMESTAMP, REFERENCE, ROW

### ABSOLUTELY FORBIDDEN DATA TYPES - NEVER USE:
âŒ XML - Use REFERENCE TO InputRoot.XMLNSC instead
âŒ RECORD - Use REFERENCE TO instead  
âŒ STRING - Use CHARACTER instead
âŒ VARCHAR - Use CHARACTER instead
âŒ JSON - Use REFERENCE TO InputRoot.JSON instead
âŒ Database - Use REFERENCE TO instead

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
- Always start message processing with: SET OutputRoot = InputRoot; OR CALL CopyEntireMessage();

### 2. MANDATORY PROCEDURES (MUST BE INCLUDED EXACTLY):
Every ESQL file MUST include these two procedures at the bottom, just before END MODULE;

```
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
```

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
        """
        Create analysis prompt for a data chunk
        """
        return f"""Analyze this data chunk ({chunk_index + 1}) to extract ESQL requirements:

## DATA CHUNK:
{json.dumps(chunk_data, indent=2)}

## ANALYSIS REQUIREMENTS:
Extract and return JSON with:
1. "esql_modules": List of required ESQL modules with names and purposes
2. "business_logic": Business logic requirements for each module
3. "message_structure": Input/output message structure details
4. "customizations": Specific customizations needed for template

Focus on:
- Node names from MessageFlow that need ESQL modules
- Business requirements from confluence/PDF content
- Database operations and procedures mentioned
- Transformation logic requirements
- Custom XPath expressions needed

Return valid JSON only:"""

    def _get_esql_generation_prompt(self, module_requirements: Dict, template_info: Dict) -> str:
        """
        Create ESQL generation prompt for specific module
        """
        module_name = module_requirements.get('name', 'UnknownModule')
        if module_name.lower().endswith('.esql'):
            module_name = module_name[:-5]

        purpose = module_requirements.get('purpose', 'Message processing')
        
        prompt = f"""Generate a complete ESQL module for IBM ACE:

## MODULE SPECIFICATION:
- **Name**: {module_name}
- **Purpose**: {purpose}
- **Type**: {module_requirements.get('type', 'COMPUTE')}

## BUSINESS REQUIREMENTS:
{json.dumps(module_requirements.get('business_logic', {}), indent=2)}

- Focus on message processing and validation
- Database operations will be added manually by developers
- No automatic database procedure generation

- Focus only on ESQL compute logic  
- No XSL transformations (handled separately)
- No inline XML transformations

## TEMPLATE CUSTOMIZATIONS:
{json.dumps(module_requirements.get('customizations', {}), indent=2)}

## GENERATION REQUIREMENTS:

### Must Include Exactly:
1. CREATE COMPUTE MODULE {module_name}
2. CREATE FUNCTION Main() RETURNS BOOLEAN
3. All required DECLARE statements for episInfo, sourceInfo, targetInfo, integrationInfo, dataInfo
4. Customized sourceInfo/targetInfo assignments based on message structure
5. Customized dataInfo assignments based on business requirements
6. RETURN TRUE;
7. CREATE PROCEDURE CopyMessageHeaders() - exactly as specified
8. CREATE PROCEDURE CopyEntireMessage() - exactly as specified
9. END MODULE;

### Customization Rules:
- Update dataInfo.mainIdentifier XPath based on main business entity
- Populate customReference1-4 with database procedures if needed
- Populate customProperty1-4 with transformation descriptions if needed
- Remove unused sourceInfo/targetInfo assignments if not required

### Critical Rules:
- InputRoot is READ-ONLY
- Use OutputRoot for modifications
- No comments starting with "--"
- No hardcoded values - make context-appropriate

Generate the complete ESQL module:"""
        
        return prompt

    def analyze_requirements_with_chunking(self, input_data: Dict) -> Dict:
        """
        Analyze requirements using chunked approach - 100% LLM
        """
        print("ðŸ” Starting LLM-based requirements analysis with chunking...")
        
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
        
        # Analyze each chunk with LLM
        for i, chunk in enumerate(data_chunks):
            print(f"ðŸ“Š Analyzing chunk {i + 1}/{len(data_chunks)} with LLM...")
            
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
                
                print(f"âœ… Chunk {i + 1} analysis complete")
                
            except Exception as e:
                print(f"âš ï¸ Chunk {i + 1} analysis failed: {e}")
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
        
        print(f"ðŸŽ¯ Analysis complete: {len(unique_modules)} unique ESQL modules identified")
        return combined_requirements

    def _llm_analyze_chunk(self, chunk_data: Dict, chunk_index: int) -> Dict:
        """
        LLM analysis of a single chunk
        """
        if not self.groq_client:
            raise Exception("LLM client not available")
        
        try:
            prompt = self._get_chunk_analysis_prompt(chunk_data, chunk_index)
            
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=3000
            )
            print(f"ðŸ” DEBUG: LLM call completed in method: {__name__}")
            print(f"ðŸ” DEBUG: token_tracker in session_state: {'token_tracker' in st.session_state}")
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
                    print(f"ðŸ“Š esql_generator/llm_call_detected: {response.usage.total_tokens} tokens")
                else:
                    print("ðŸ” DEBUG: Token tracking skipped - conditions not met")
            except Exception as e:
                print(f"ðŸ” DEBUG: Token tracking error: {e}")
            
            
            self.llm_calls_count += 1
            raw_response = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                return json.loads(raw_response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    print(f"âš ï¸ Could not parse JSON from chunk {chunk_index}")
                    return {}
                    
        except Exception as e:
            print(f"âŒ LLM chunk analysis failed: {e}")
            return {}

    def generate_esql_modules(self, requirements: Dict, template_info: Dict) -> List[Dict]:
        """
        Generate ESQL modules - 100% LLM based
        """
        print("ðŸ­ Starting LLM-based ESQL module generation...")
        
        modules = requirements.get('esql_modules', [])
        if not modules:
            print("âš ï¸ No ESQL modules identified in requirements")
            return []
        
        generated_modules = []
        
        for module_req in modules:
            module_name = module_req.get('name', 'UnknownModule')
            if module_name.lower().endswith('.esql'):
                module_name = module_name[:-5]

            print(f"âš¡ Generating {module_name}.esql with LLM...")
            
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
                
                print(f"âœ… {module_name}.esql generated successfully")
                
            except Exception as e:
                print(f"âŒ Failed to generate {module_name}.esql: {e}")
                continue
        
        print(f"ðŸŽ‰ Generated {len(generated_modules)} ESQL modules via LLM")
        return generated_modules

    def _enrich_module_requirements(self, module_req: Dict, global_requirements: Dict) -> Dict:
        """
        Enrich module requirements with global context
        """
        enriched = module_req.copy()
        
        # Add global business logic
        enriched['business_logic'] = {
            **global_requirements.get('business_logic', {}),
            **module_req.get('business_logic', {})
        }
        
        # Add relevant database operations
        enriched['database_operations'] = [
            op for op in global_requirements.get('database_operations', [])
            if module_req.get('name', '') in op.get('related_modules', [module_req.get('name', '')])
        ]
        
        # Add relevant transformations
        enriched['transformations'] = [
            trans for trans in global_requirements.get('transformations', [])
            if module_req.get('name', '') in trans.get('related_modules', [module_req.get('name', '')])
        ]
        
        # Add message structure
        enriched['message_structure'] = global_requirements.get('message_structure', {})
        
        # Add customizations
        enriched['customizations'] = global_requirements.get('customizations', {})
        
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
            print(f"ðŸ” DEBUG: LLM call completed in method: {__name__}")
            print(f"ðŸ” DEBUG: token_tracker in session_state: {'token_tracker' in st.session_state}")
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
                    print(f"ðŸ“Š esql_generator/llm_call_detected: {response.usage.total_tokens} tokens")
                else:
                    print("ðŸ” DEBUG: Token tracking skipped - conditions not met")
            except Exception as e:
                print(f"ðŸ” DEBUG: Token tracking error: {e}")


            self.llm_calls_count += 1
            esql_content = response.choices[0].message.content.strip()
            
            # Remove any markdown code blocks if present
            esql_content = re.sub(r'```esql\n?', '', esql_content)
            esql_content = re.sub(r'```\n?', '', esql_content)
            
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
1. âœ… Contains CREATE COMPUTE MODULE {module_name}
2. âœ… Contains CREATE FUNCTION Main() RETURNS BOOLEAN
3. âœ… Contains CREATE PROCEDURE CopyMessageHeaders() exactly as required
4. âœ… Contains CREATE PROCEDURE CopyEntireMessage() exactly as required
5. âœ… Contains END MODULE;
6. âœ… No comments starting with "--"
7. âœ… InputRoot treated as READ-ONLY
8. âœ… OutputRoot used for modifications
9. âœ… All required DECLARE statements present
10. âœ… RETURN TRUE; statement present

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
            print(f"ðŸ” DEBUG: LLM call completed in method: {__name__}")
            print(f"ðŸ” DEBUG: token_tracker in session_state: {'token_tracker' in st.session_state}")
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
                    print(f"ðŸ“Š esql_generator/llm_call_detected: {response.usage.total_tokens} tokens")
                else:
                    print("ðŸ” DEBUG: Token tracking skipped - conditions not met")
            except Exception as e:
                print(f"ðŸ” DEBUG: Token tracking error: {e}")
            
            self.llm_calls_count += 1
            enhanced_content = response.choices[0].message.content.strip()
            
            # Remove any markdown if present
            enhanced_content = re.sub(r'```esql\n?', '', enhanced_content)
            enhanced_content = re.sub(r'```\n?', '', enhanced_content)
            
            return enhanced_content
            
        except Exception as e:
            print(f"âš ï¸ LLM validation failed for {module_name}: {e}")
            return esql_content  # Return original if validation fails



    def generate_esql_files(self, vector_content: str, esql_template: Dict, 
                        msgflow_content: Dict, json_mappings: Dict, output_dir: str = None) -> Dict:
        """
        Generate ESQL files using Vector DB content and 100% LLM processing
        STANDALONE METHOD: Individual error isolation, no helper methods, no fallbacks
        Saves successful modules immediately, continues processing if some modules fail
        """
        print("ðŸš€ Starting Vector DB + LLM ESQL generation...")
        
        # Set output_dir at the beginning
        self.output_dir = output_dir or 'output'
        
        if not self.groq_client:
            raise Exception("LLM client not available - Vector DB processing requires LLM")
        
        # Initialize tracking arrays
        successful_modules = []
        failed_modules = []
        
        try:
            # Step 1: Load supporting files for LLM analysis
            print("ðŸ“ Loading supporting files for LLM analysis...")
            
            # Load ESQL template
            esql_template_content = ""
            if esql_template.get('path') and os.path.exists(esql_template['path']):
                with open(esql_template['path'], 'r', encoding='utf-8') as f:
                    esql_template_content = f.read()
                print(f"  âœ… ESQL Template loaded: {len(esql_template_content)} characters")
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
                print(f"  âœ… MessageFlow auto-discovered: {msgflow_file_found} ({len(msgflow_content_text)} characters)")
            else:
                print(f"  âš ï¸ No .msgflow files found in output directory")
                print(f"  ðŸ”„ Continuing with Vector DB content only...")
                msgflow_content_text = "" 
            


            
            # Load JSON mappings
            json_mappings_data = {}
            if json_mappings.get('path') and os.path.exists(json_mappings['path']):
                with open(json_mappings['path'], 'r', encoding='utf-8') as f:
                    json_mappings_data = json.load(f)
                print(f"  âœ… JSON mappings loaded: {len(json_mappings_data)} components")
            else:
                raise Exception(f"JSON mappings not found: {json_mappings.get('path', 'No path provided')}")
            
            
            
            # Extract ESQL requirements from MessageFlow compute expressions via LLM
            print("ðŸ” LLM extracting ESQL requirements from MessageFlow compute expressions...")
            
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
                    print(f"ðŸ“Š esql_generator/esql_module_generation: {generation_response.usage.total_tokens} tokens | ${generation_response.usage.total_tokens * 0.0008:.4f} | {self.groq_model}")
            except Exception as e:
                print(f"âš ï¸ Token tracking failed: {e}")
            
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
            
            print(f"  âœ… LLM extracted {len(msgflow_esql_modules)} ESQL modules from MessageFlow")
            
            # Step 2: Comprehensive LLM Analysis combining ALL sources
            print("ðŸ§  LLM Analysis: Extracting ESQL requirements from ALL sources...")


            
            analysis_prompt = f"""Use MessageFlow as the ONLY authoritative source for ESQL modules. Enhance with business context but do NOT create additional modules:

            ## BUSINESS REQUIREMENTS (CONTEXT ONLY - DO NOT CREATE NEW MODULES FROM THIS):
            {vector_content[:2000]}

            ## COMPONENT MAPPINGS (REFERENCE ONLY - DO NOT CREATE NEW MODULES FROM THIS):
            {json.dumps(json_mappings_data.get('component_mappings', [])[:3], indent=2)}

            ## MESSAGEFLOW COMPUTE EXPRESSIONS (AUTHORITATIVE SOURCE - CREATE ESQL FOR THESE ONLY):
            {json.dumps(msgflow_esql_modules, indent=2)}

            ## MESSAGEFLOW STRUCTURE:
            {msgflow_content_text[:5000]}

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

            Return valid JSON only:"""

            # LLM call for comprehensive requirements analysis
            analysis_response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": "Extract ESQL requirements from all business sources and return valid JSON only."},
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
                    print(f"ðŸ“Š esql_generator/esql_module_generation: {generation_response.usage.total_tokens} tokens | ${generation_response.usage.total_tokens * 0.0008:.4f} | {self.groq_model}")
            except Exception as e:
                print(f"âš ï¸ Token tracking failed: {e}")
            
            self.llm_calls_count += 1
            
            # Parse comprehensive analysis
            analysis_content = analysis_response.choices[0].message.content.strip()
            json_start = analysis_content.find('{')
            json_end = analysis_content.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise Exception("LLM failed to generate valid JSON requirements analysis")
            
            try:
                requirements = json.loads(analysis_content[json_start:json_end])
            except json.JSONDecodeError as e:
                raise Exception(f"LLM generated invalid JSON for requirements: {str(e)}")
            
            total_modules = len(requirements.get('esql_modules', []))
            component_modules = len([m for m in requirements.get('esql_modules', []) if m.get('source') == 'component_mapping'])
            msgflow_modules = len([m for m in requirements.get('esql_modules', []) if m.get('source') == 'messageflow_compute'])
            
            print(f"  âœ… Requirements analysis completed: {total_modules} modules identified")
            print(f"    ðŸ“Š Component mappings: {component_modules} modules")
            print(f"    ðŸ“Š MessageFlow compute: {msgflow_modules} modules")
            
            if total_modules == 0:
                raise Exception("No ESQL modules identified from Vector DB + MessageFlow analysis")
            
            # Step 3: Generate ESQL modules with individual error isolation
            print("ðŸ”§ LLM Generation: Creating ESQL modules...")
            
            esql_modules = requirements.get('esql_modules', [])
            
            # Process each module individually with error isolation
            for i, module_req in enumerate(esql_modules):
                module_name = module_req.get('name', f'ESQLModule_{i+1}')
                
                try:
                    print(f"  ðŸŽ¯ Generating {module_name}.esql with LLM...")
                    
                    # Generate comprehensive generation prompt
                    generation_prompt = f"""âš ï¸ CRITICAL: FOLLOW DATA TYPE RESTRICTIONS EXACTLY âš ï¸

            ## MANDATORY DATA TYPES - NO EXCEPTIONS:
            âœ… ONLY USE: BOOLEAN, INTEGER, DECIMAL, FLOAT, CHARACTER, BIT, BLOB, DATE, TIME, TIMESTAMP, REFERENCE, ROW
            âŒ NEVER USE: XML, RECORD, STRING, VARCHAR, JSON, Database

            ## XML PROCESSING EXAMPLES:
            âŒ WRONG: DECLARE xmlData XML;
            âœ… CORRECT: DECLARE xmlRef REFERENCE TO InputRoot.XMLNSC;

            âŒ WRONG: DECLARE msg RECORD;  
            âœ… CORRECT: DECLARE msgRef REFERENCE TO InputRoot;

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

    ## VARIABLE NAMING:
    Customize declare variable names based on business context (episInfo could become documentInfo, orderInfo, shipmentInfo, etc.)

    Generate ONLY the complete ESQL module:"""

                    # LLM call for ESQL generation with strict format
                    generation_response = self.groq_client.chat.completions.create(
                        model=self.groq_model,
                        messages=[
                            {"role": "system", "content": "Generate complete IBM ACE ESQL modules with exact format compliance. Follow all format requirements precisely."},
                            {"role": "user", "content": generation_prompt}
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
                            print(f"ðŸ“Š esql_generator/esql_module_generation: {generation_response.usage.total_tokens} tokens | ${generation_response.usage.total_tokens * 0.0008:.4f} | {self.groq_model}")
                    except Exception as e:
                        print(f"âš ï¸ Token tracking failed: {e}")
                    
                    self.llm_calls_count += 1
                    
                    esql_content = generation_response.choices[0].message.content.strip()
                    
                    # Clean LLM output - remove code markers if present
                    esql_content = re.sub(r'```[\w]*\n?', '', esql_content)
                    esql_content = re.sub(r'\n?```\s*$', '', esql_content)
                    esql_content = self._clean_esql_content(esql_content)
                    
                    # Validate ESQL format compliance (strict validation maintained)
                    if not esql_content.startswith('CREATE COMPUTE MODULE'):
                        raise Exception(f"ESQL format violation: {module_name} must start with 'CREATE COMPUTE MODULE'")
                    
                    if not esql_content.rstrip().endswith('END MODULE;'):
                        raise Exception(f"ESQL format violation: {module_name} must end with 'END MODULE;'")
                    
                    if '@' in esql_content:
                        raise Exception(f"ESQL format violation: {module_name} contains forbidden '@' symbols")

                    # ENHANCED: Apply constraint validation
                    constraint_result = self.validate_esql_constraints(esql_content)
                    if not constraint_result['valid']:
                        print(f"    âŒ Constraint violations for {module_name}: {constraint_result['errors']}")
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
                        'validation_status': 'FORMAT_VALIDATED'
                    })
                    
                    print(f"    âœ… {esql_filename} generated successfully ({len(esql_content)} characters)")
                    
                except Exception as module_error:
                    # Individual module failed - log and continue with next module
                    error_message = str(module_error)
                    print(f"ðŸ’¥ ESQL generation failed: ESQL format violation: {module_name} must start with 'CREATE COMPUTE MODULE'")
                    
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
                print(f"âœ… Successful modules: {successful_count}/{total_attempted}")
                print(f"ðŸ“Š Total files generated: {successful_count}")
                print(f"âŒ Failed modules: {failed_count}")
                for failed in failed_modules:
                    print(f"  - {failed['name']}: {failed['error_message']}")
            else:
                # Complete success
                print(f"ðŸŽ‰ ESQL generation complete: {successful_count} modules generated")
            
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
                'generation_method': '100% LLM Based with Individual Error Isolation',
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
                print(f"ðŸ’¥ ESQL generation failed: {str(e)}")
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
        if re.search(r'CALL\s+\w+\s*\(', esql_content, re.IGNORECASE):
            validation['errors'].append("Direct procedure calls are forbidden")
            validation['valid'] = False
        
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
        print("âŒ Vector DB pipeline not available")
        print("ðŸ’¡ Run this through main.py with Vector DB setup")
        print("ðŸ“ Steps: 1) Upload PDF in Agent 1, 2) Setup Vector Knowledge Base, 3) Run ESQL generation")
        return
    
    print("ðŸš€ Starting Vector DB ESQL generation test...")
    
    # Get Vector DB content for ESQL generation
    vector_content = st.session_state.vector_pipeline.search_engine.get_agent_content("esql_generator")
    
    if not vector_content:
        print("âŒ No Vector DB content found for 'esql_generator'")
        print("ðŸ’¡ Ensure Vector Knowledge Base contains ESQL-related content")
        return
    
    print(f"ðŸ“Š Vector DB content retrieved: {len(vector_content)} characters")
    
    # Generate ESQL using Vector DB content
    result = generator.generate_esql_files(
        vector_content=vector_content,                      # âœ… Vector DB content
        esql_template={'path': 'ESQL_Template_Updated.ESQL'},  # âœ… Direct path for testing
        msgflow_content={'path': 'output/generated_messageflow.msgflow'},  # âœ… Direct path
        json_mappings={'path': 'component_mapping.json'}    # âœ… Direct path for testing
    )
    
    print(f"\nðŸŽ¯ ESQL Generation Results:")
    print(f"âœ… Status: Success")
    print(f"ðŸ“Š Modules Generated: {result['total_modules']}")
    print(f"ðŸ§  LLM Calls: {result['llm_calls_made']}")
    print(f"ðŸŽ¯ Generation Method: {result['generation_method']}")
    print(f"âš¡ Token Management: {result['token_management']}")
    
    # Display individual modules
    if result.get('generated_modules'):
        print(f"\nðŸ“ Generated ESQL Modules:")
        for i, module in enumerate(result['generated_modules'], 1):
            print(f"  {i}. {module['filename']} - {module['purpose']} ({module['content_length']} chars)")
    
    # Display processing summary
    if result.get('processing_summary'):
        summary = result['processing_summary']
        print(f"\nðŸ“ˆ Processing Summary:")
        print(f"  â€¢ Vector content: {summary.get('vector_content_length', 0)} characters")
        print(f"  â€¢ ESQL modules identified: {summary.get('esql_modules_identified', 0)}")
        print(f"  â€¢ Business logic rules: {summary.get('business_logic_rules', 0)}")
        print(f"  â€¢ Database operations: {summary.get('database_operations', 0)}")
        print(f"  â€¢ Transformations: {summary.get('transformations', 0)}")


if __name__ == "__main__":
    main()