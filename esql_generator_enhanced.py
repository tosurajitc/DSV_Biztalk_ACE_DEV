"""
Enhanced ESQL Generator - 100% LLM Based with Token Management & ESQL Constraints
NO HARDCODED FALLBACKS - Pure LLM Generation Only
ENHANCED with strict ESQL data type and syntax constraints
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
    100% LLM-based ESQL generator with intelligent token management and enhanced constraints
    CRITICAL: NO hardcoded ESQL generation - pure LLM only
    ENHANCED: Strict ESQL constraint validation and enforcement
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
        self.avg_chars_per_token = 3.2  
        self.max_total_tokens = 32768
        
        # Initialize attributes
        self.output_dir = None
        self.llm_calls_count = 0
        self.esql_modules = []
        self.processing_results = {}
        
        # ENHANCED: Valid ESQL data types
        self.valid_data_types = {
            'BOOLEAN', 'INTEGER', 'DECIMAL', 'FLOAT', 'CHARACTER',
            'BIT', 'BLOB', 'DATE', 'TIME', 'TIMESTAMP', 'REFERENCE', 'ROW'
        }
        
        # ENHANCED: Forbidden patterns
        self.forbidden_patterns = {
            'varchar': r'\bVARCHAR\b',
            'database_declare': r'DECLARE\s+\w+\s+Database\b',
            'procedure_call': r'CALL\s+\w+\s*\(',
            'at_symbol': r'@',
            'esql_line': r'^esql\s',
            'comment_line': r'^--\s'
        }
        
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        return int(len(text) / self.avg_chars_per_token)
    
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
        
        print(f"📄 Created {len(chunks)} chunks for processing")
        return chunks
    
    def _get_system_prompt(self) -> str:
        """
        ENHANCED system prompt with strict ESQL constraints
        """
        return """You are an expert IBM ACE ESQL developer specializing in generating production-ready ESQL modules. You MUST follow these CRITICAL requirements exactly:

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

### 3. FORBIDDEN ELEMENTS:
- NO comments starting with "--" (e.g., "-- Declare variables")
- NO lines starting with "esql"
- NO "@" symbols anywhere
- NO code block markers
- NO custom SQLEXCEPTION handlers in procedures

## 🚫 ENHANCED ESQL CONSTRAINTS (CRITICAL):

### 4. DATA TYPE RESTRICTIONS (STRICT ENFORCEMENT):
- **VARCHAR is COMPLETELY FORBIDDEN** - use CHARACTER instead
- **ONLY these data types are valid:**
  * BOOLEAN
  * INTEGER  
  * DECIMAL
  * FLOAT
  * CHARACTER (never VARCHAR)
  * BIT
  * BLOB
  * DATE
  * TIME
  * TIMESTAMP
  * REFERENCE
  * ROW
- **ANY other data type will cause compilation failure**

### 5. DECLARE RESTRICTIONS (CRITICAL):
- **NEVER declare Database references:** 
  * DECLARE db Database; ❌ FORBIDDEN
  * DECLARE conn Database; ❌ FORBIDDEN
- **ONLY declare local variables and references:**
  * DECLARE customVar CHARACTER; ✅ CORRECT
  * DECLARE counter INTEGER; ✅ CORRECT
  * DECLARE ref REFERENCE TO OutputRoot; ✅ CORRECT

### 6. PROCEDURE CALL RESTRICTIONS (CRITICAL):
- **NO direct procedure calls in ESQL files:**
  * CALL ExternalProcedure(); ❌ FORBIDDEN
  * CALL DatabaseStoredProc(); ❌ FORBIDDEN
- **Use built-in ESQL functions ONLY:**
  * COALESCE(), CAST(), LENGTH() ✅ CORRECT
  * Built-in message manipulation functions ✅ CORRECT

### 7. CONSTRAINT EXAMPLES:

#### ✅ CORRECT USAGE:
```
DECLARE customerName CHARACTER;
DECLARE orderCount INTEGER;
DECLARE isProcessed BOOLEAN;
DECLARE orderRef REFERENCE TO InputRoot.XMLNSC.Order;
SET customerName = COALESCE(InputRoot.XMLNSC.Customer.Name, '');
```

#### ❌ FORBIDDEN USAGE:
```
DECLARE customerName VARCHAR(100);     -- FORBIDDEN: Use CHARACTER
DECLARE dbConn Database;               -- FORBIDDEN: No Database declares
CALL ProcessOrder(orderData);          -- FORBIDDEN: No procedure calls
DECLARE status STRING;                 -- FORBIDDEN: Invalid data type
```

### 8. TEMPLATE STRUCTURE (MUST BE CUSTOMIZED BASED ON BUSINESS FLOW):
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
- **ENFORCE ALL DATA TYPE CONSTRAINTS STRICTLY**

## CRITICAL: 100% LLM GENERATION WITH CONSTRAINTS
- NO hardcoded ESQL patterns
- NO template substitution fallbacks
- Generate everything through LLM reasoning
- Adapt to specific business requirements dynamically
- **NEVER violate data type or constraint rules**"""

    def validate_esql_constraints(self, esql_content: str) -> Dict[str, Any]:
        """
        ENHANCED: Validate strict ESQL constraints
        """
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'constraint_violations': []
        }
        
        # 1. Check for forbidden VARCHAR usage
        if re.search(self.forbidden_patterns['varchar'], esql_content, re.IGNORECASE):
            validation['errors'].append("CONSTRAINT VIOLATION: VARCHAR is forbidden - use CHARACTER instead")
            validation['constraint_violations'].append("varchar_usage")
            validation['valid'] = False
        
        # 2. Check for Database DECLARE statements
        database_declares = re.findall(self.forbidden_patterns['database_declare'], esql_content, re.IGNORECASE)
        if database_declares:
            validation['errors'].append("CONSTRAINT VIOLATION: DECLARE with Database is forbidden")
            validation['constraint_violations'].append("database_declare")
            validation['valid'] = False
        
        # 3. Check for direct procedure calls
        procedure_calls = re.findall(self.forbidden_patterns['procedure_call'], esql_content, re.IGNORECASE)
        if procedure_calls:
            validation['errors'].append("CONSTRAINT VIOLATION: Direct procedure calls are forbidden in ESQL files")
            validation['constraint_violations'].append("procedure_calls")
            validation['valid'] = False
        
        # 4. Check for invalid data types in DECLARE statements
        declare_pattern = r'DECLARE\s+\w+\s+(\w+)'
        declared_types = re.findall(declare_pattern, esql_content, re.IGNORECASE)
        
        for data_type in declared_types:
            if data_type.upper() not in self.valid_data_types:
                validation['errors'].append(f"CONSTRAINT VIOLATION: Invalid data type '{data_type}' - only {self.valid_data_types} are allowed")
                validation['constraint_violations'].append("invalid_data_type")
                validation['valid'] = False
        
        # 5. Check for @ symbols
        if re.search(self.forbidden_patterns['at_symbol'], esql_content):
            validation['errors'].append("CONSTRAINT VIOLATION: '@' symbols are forbidden")
            validation['constraint_violations'].append("at_symbol")
            validation['valid'] = False
        
        # 6. Check for lines starting with "esql"
        if re.search(self.forbidden_patterns['esql_line'], esql_content, re.MULTILINE):
            validation['errors'].append("CONSTRAINT VIOLATION: Lines starting with 'esql' are forbidden")
            validation['constraint_violations'].append("esql_line")
            validation['valid'] = False
        
        # 7. Check for comment lines starting with "--"
        if re.search(self.forbidden_patterns['comment_line'], esql_content, re.MULTILINE):
            validation['errors'].append("CONSTRAINT VIOLATION: Comment lines starting with '--' are forbidden")
            validation['constraint_violations'].append("comment_line")
            validation['valid'] = False
        
        return validation

    def validate_esql_syntax(self, esql_content: str) -> Dict:
        """
        ENHANCED: Validate ESQL syntax including new constraints
        """
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Basic syntax validation (existing)
        required_elements = [
            'CREATE COMPUTE MODULE',
            'CREATE FUNCTION Main()',
            'RETURNS BOOLEAN',
            'BEGIN',
            'END;',
            'END MODULE;'
        ]
        
        for element in required_elements:
            if element not in esql_content:
                validation['errors'].append(f"Missing required element: {element}")
                validation['valid'] = False
        
        # ENHANCED: Apply constraint validation
        constraint_validation = self.validate_esql_constraints(esql_content)
        
        # Merge constraint validation results
        validation['errors'].extend(constraint_validation['errors'])
        validation['warnings'].extend(constraint_validation['warnings'])
        
        if not constraint_validation['valid']:
            validation['valid'] = False
            validation['constraint_violations'] = constraint_validation['constraint_violations']
        
        return validation

    def _clean_esql_content(self, esql_content: str) -> str:
        """Clean and format ESQL content"""
        # Remove any potential code block markers
        esql_content = re.sub(r'```[\w]*\n?', '', esql_content)
        esql_content = re.sub(r'\n?```\s*$', '', esql_content)
        
        # Remove any trailing whitespace
        lines = [line.rstrip() for line in esql_content.split('\n')]
        esql_content = '\n'.join(lines)
        
        return esql_content.strip()

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
        Create ESQL generation prompt for specific module with enhanced constraints
        """
        module_name = module_requirements.get('name', 'UnknownModule')
        if module_name.lower().endswith('.esql'):
            module_name = module_name[:-5]

        purpose = module_requirements.get('purpose', 'Message processing')
        
        prompt = f"""Generate a complete ESQL module for IBM ACE with STRICT constraint compliance:

## MODULE SPECIFICATION:
- **Name**: {module_name}
- **Purpose**: {purpose}
- **Type**: {module_requirements.get('type', 'COMPUTE')}

## BUSINESS REQUIREMENTS:
{json.dumps(module_requirements.get('business_logic', {}), indent=2)}

## TEMPLATE CUSTOMIZATIONS:
{json.dumps(module_requirements.get('customizations', {}), indent=2)}

## CRITICAL CONSTRAINT REMINDERS:
- Use CHARACTER instead of VARCHAR
- Only valid data types: BOOLEAN, INTEGER, DECIMAL, FLOAT, CHARACTER, BIT, BLOB, DATE, TIME, TIMESTAMP, REFERENCE, ROW
- NO Database declarations
- NO direct procedure calls
- NO @ symbols, NO lines starting with 'esql', NO -- comments

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

Generate the complete, constraint-compliant ESQL module:"""
        
        return prompt

    def generate_esql_files(self, input_data: Dict, output_directory: str) -> Dict[str, Any]:
        """
        ENHANCED: Generate ESQL files with constraint validation
        """
        self.output_dir = output_directory
        
        print("🚀 Starting enhanced ESQL generation with constraint validation...")
        
        try:
            # Chunk input data for token management
            data_chunks = self.chunk_input_data(input_data)
            
            # Process chunks to extract requirements
            all_requirements = []
            
            for i, chunk in enumerate(data_chunks):
                print(f"📊 Processing chunk {i+1}/{len(data_chunks)}...")
                
                analysis_prompt = self._get_chunk_analysis_prompt(chunk, i)
                
                analysis_response = self.groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=[
                        {"role": "system", "content": "Extract ESQL requirements from business sources and return valid JSON only."},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=4000
                )
                
                self.llm_calls_count += 1
                
                # Parse requirements
                analysis_content = analysis_response.choices[0].message.content.strip()
                json_start = analysis_content.find('{')
                json_end = analysis_content.rfind('}') + 1
                
                if json_start != -1 and json_end > 0:
                    try:
                        chunk_requirements = json.loads(analysis_content[json_start:json_end])
                        all_requirements.extend(chunk_requirements.get('esql_modules', []))
                    except json.JSONDecodeError as e:
                        print(f"⚠️  JSON parsing error for chunk {i+1}: {str(e)}")
                        continue
            
            # Generate ESQL modules with constraint validation
            generated_files = []
            failed_modules = []
            
            for module_req in all_requirements:
                module_name = module_req.get('name', f'Module_{len(generated_files)+1}')
                
                try:
                    print(f"⚡ Generating {module_name} with constraint validation...")
                    
                    # Generate ESQL content
                    generation_prompt = self._get_esql_generation_prompt(module_req, {})
                    
                    generation_response = self.groq_client.chat.completions.create(
                        model=self.groq_model,
                        messages=[
                            {"role": "system", "content": self._get_system_prompt()},
                            {"role": "user", "content": generation_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=4000
                    )
                    
                    self.llm_calls_count += 1
                    
                    esql_content = generation_response.choices[0].message.content.strip()
                    esql_content = self._clean_esql_content(esql_content)
                    
                    # ENHANCED: Validate constraints
                    validation_result = self.validate_esql_syntax(esql_content)
                    
                    if not validation_result['valid']:
                        print(f"❌ Constraint validation failed for {module_name}")
                        print(f"   Errors: {validation_result['errors']}")
                        
                        # Try regeneration with enhanced guidance (optional)
                        if hasattr(validation_result, 'constraint_violations'):
                            print(f"   Constraint violations: {validation_result['constraint_violations']}")
                        
                        failed_modules.append({
                            'name': module_name,
                            'errors': validation_result['errors'],
                            'violations': validation_result.get('constraint_violations', [])
                        })
                        continue
                    
                    # Save valid ESQL file
                    if not module_name.endswith('.esql'):
                        module_name += '.esql'
                    
                    file_path = os.path.join(output_directory, module_name)
                    os.makedirs(output_directory, exist_ok=True)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(esql_content)
                    
                    generated_files.append({
                        'name': module_name,
                        'path': file_path,
                        'validation': validation_result
                    })
                    
                    print(f"✅ {module_name} generated and validated successfully")
                    
                except Exception as e:
                    print(f"❌ Error generating {module_name}: {str(e)}")
                    failed_modules.append({
                        'name': module_name,
                        'error': str(e)
                    })
            
            # Return comprehensive results
            return {
                'success': len(generated_files) > 0,
                'generated_files': generated_files,
                'failed_modules': failed_modules,
                'total_modules': len(all_requirements),
                'llm_calls': self.llm_calls_count,
                'constraint_enforcement': True,
                'summary': {
                    'generated': len(generated_files),
                    'failed': len(failed_modules),
                    'success_rate': len(generated_files) / len(all_requirements) * 100 if all_requirements else 0
                }
            }
            
        except Exception as e:
            print(f"💥 Critical error in ESQL generation: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'generated_files': [],
                'failed_modules': [],
                'constraint_enforcement': True
            }

    def get_generation_summary(self) -> Dict[str, Any]:
        """Get summary of generation process"""
        return {
            'llm_calls_made': self.llm_calls_count,
            'modules_generated': len(self.esql_modules),
            'processing_results': self.processing_results,
            'constraint_enforcement_enabled': True,
            'valid_data_types': list(self.valid_data_types),
            'forbidden_patterns': list(self.forbidden_patterns.keys())
        }