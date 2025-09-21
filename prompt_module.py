#!/usr/bin/env python3
"""
Enhanced Prompt Module for ACE Generation
Updated to handle MessageFlow templates and enforce ESQL coding standards
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from dotenv import load_dotenv
load_dotenv()

class PromptModule:
    """Enhanced prompt generation for ACE components with strict ESQL rules"""
    
    def __init__(self):
        self.system_context = """You are an expert IBM ACE (App Connect Enterprise) developer with deep knowledge of:
- MessageFlow XML structure and node configurations
- ESQL programming best practices and syntax
- IBM ACE runtime behavior and performance optimization
- BizTalk to ACE migration patterns
- Database integration and message transformation patterns"""

    def get_system_context(self) -> str:
        """Get the system context for LLM prompts"""
        return self.system_context


def get_msgflow_generation_prompt(flow_name: str, project_name: str, 
                                input_queue: str, output_service: str, error_queue: str,
                                esql_modules: List[Dict], msgflow_template: str = None,
                                confluence_spec: str = None, components_info: List[Dict] = None) -> str:
    """
    Generate comprehensive MessageFlow creation prompt using provided template
    
    Args:
        flow_name: Name of the message flow
        project_name: Name of the ACE project
        input_queue: Input queue name
        output_service: Output service endpoint
        error_queue: Error queue name
        esql_modules: List of ESQL modules to integrate
        msgflow_template: MessageFlow XML template to use as base
        confluence_spec: Business specification document
        components_info: BizTalk component information
    """
    
    prompt = f"""# IBM ACE MessageFlow Generation Task

You are an expert IBM ACE developer. Generate a production-ready MessageFlow XML based on the provided template and specifications.

## TEMPLATE PROVIDED:
```xml
{msgflow_template if msgflow_template else "No template provided - create from scratch"}
```

## BUSINESS REQUIREMENTS (from Confluence):
```
{confluence_spec[:1500] if confluence_spec else "No specification provided"}
```

## COMPONENT MAPPING CONTEXT:
"""
    
    if components_info:
        for comp in components_info[:5]:  # Limit to first 5 components
            prompt += f"""
- **{comp.get('biztalk_component', 'Unknown')}**: {comp.get('component_type', 'Unknown')} → {comp.get('ace_library', 'Unknown')}"""
    
    prompt += f"""

## MESSAGEFLOW SPECIFICATIONS:
- **Flow Name**: {flow_name}
- **Project**: {project_name}
- **Input Queue**: {input_queue}
- **Output Service**: {output_service}
- **Error Queue**: {error_queue}

## ESQL MODULES TO INTEGRATE:
"""
    
    for module in esql_modules:
        prompt += f"""
- **{module.get('name', 'Unknown')}**: {module.get('purpose', 'Processing')}"""
    
    prompt += f"""

## GENERATION REQUIREMENTS:

### 1. XML Structure Requirements (CRITICAL):
- Use the provided template as the EXACT foundation structure
- Maintain ALL namespace declarations (xmlns) exactly as in template
- Preserve node positioning and connection patterns from template
- Update ONLY node names, properties, and connections based on requirements
- DO NOT modify the root element structure or namespace URIs

### 2. Node Configuration:
- Configure MQInput node for queue: {input_queue}
- Configure HTTPRequest/SOAPRequest for service: {output_service}
- Configure MQOutput node for error queue: {error_queue}
- Add Compute nodes for each ESQL module integration
- Include proper error handling with TryCatch nodes
- Use node IDs following template pattern (e.g., FCMComposite_1_1, FCMComposite_1_2)

### 3. Flow Logic Requirements:
- Implement message validation at entry point
- Add database enrichment nodes based on specification
- Include transformation logic based on BizTalk component mapping
- Implement proper error handling and logging
- Add audit trail and monitoring capabilities
- Connect all nodes following template terminal patterns

### 4. Production Standards:
- Include comprehensive error handling
- Add proper message correlation
- Implement timeout and retry mechanisms  
- Include security and authentication nodes if required
- Follow IBM ACE best practices for performance

### 5. MANDATORY XML Structure Elements:
- Ensure <propertyOrganizer/> is added AFTER </composition> closing tag
- Ensure <stickyBoard/> is added AFTER </composition> closing tag
- These elements are REQUIRED for proper IBM ACE Toolkit compatibility

### 6. MANDATORY Graphic Resources (BEFORE <composition>):
Add these graphic resource declarations BEFORE the <composition> element:
```xml
<colorGraphic16 xmi:type="utility:GIFFileGraphic" resourceName="platform:/plugin/{project_name}/icons/full/obj16/{flow_name}.gif"/>
<colorGraphic32 xmi:type="utility:GIFFileGraphic" resourceName="platform:/plugin/{project_name}/icons/full/obj16/{flow_name}.gif"/>
```

### 7. XML Validation Rules:
- Ensure ALL opening tags have matching closing tags
- Verify all node references in connections exist
- Check that all xmi:id values are unique
- Validate that terminal references match actual terminal names
- Ensure proper XML escaping for special characters

### 8. Connection Requirements:
- Every node MUST have proper input/output terminal connections
- Follow the connection pattern: sourceNode.terminalName to targetNode.terminalName
- Ensure all connections reference valid xmi:id values
- Include failure terminals for error handling paths

## CRITICAL ERROR PREVENTION:
- DO NOT create nodes with types not present in the template
- DO NOT modify namespace URLs (xmlns values)
- DO NOT change the basic FCMComposite structure
- DO NOT add unsupported properties to nodes
- DO NOT create orphaned nodes without connections

## OUTPUT REQUIREMENTS:
- Generate ONLY the complete MessageFlow XML content
- Do not include explanations or comments outside the XML
- Ensure valid XML syntax with proper namespaces
- Test that all node references are properly connected
- Include proper XML declaration at the top
- End with proper closing tags for all elements

## XML OUTPUT FORMAT:
Provide the complete, production-ready MessageFlow XML below:

```xml"""
    
    return prompt


def get_esql_creation_prompt(module_name: str, purpose: str, business_context: str = None,
                           additional_context: Dict[str, Any] = None) -> str:
    """
    Generate ESQL module creation prompt with Event Data Usage compliance
    
    Args:
        module_name: Name of the ESQL module
        purpose: Purpose/functionality of the module
        business_context: Business requirements context
        additional_context: Additional context information
        node_type: Type of node (StartEvent, AfterTransformation, or other)
    """
    
    # Detect if this is an event capture stage from existing parameters
    name_lower = module_name.lower()
    purpose_lower = purpose.lower() if purpose else ""
    context_str = str(additional_context).lower() if additional_context else ""
    
    # Event capture indicators
    event_indicators = [
        'inputeventmessage', 'startevent', 'aftertransformation', 
        'failureeventmessage', 'errorevent', 'event_capture'
    ]
    
    # Check if this is an event capture stage
    is_event_capture = any(indicator in name_lower or indicator in purpose_lower or indicator in context_str 
                          for indicator in event_indicators)
    
    prompt = f"""# IBM ACE ESQL Module Generation

Create a production-ready ESQL module for IBM ACE: {module_name}

## BUSINESS CONTEXT:
```
{business_context[:2500] if business_context else "Standard transformation module"}
```
"""
    
    if additional_context:
        prompt += f"""
## MIGRATION CONTEXT:
- **Source BizTalk Component**: {additional_context.get('biztalk_source', 'Unknown')}
- **Component Type**: {additional_context.get('component_type', 'Unknown')}
- **Target ACE Library**: {additional_context.get('target_ace_library', 'Unknown')}
"""

    # Add Event Data Usage compliance rules for event capture stages
    if is_event_capture:
        prompt += f"""

## EVENT DATA USAGE COMPLIANCE (MANDATORY)

### ALLOWED IN EVENT CAPTURE STAGES:
1. **Metadata Extraction ONLY** - for production support team
2. **Source/Target Information** - from message Headers for tracking
3. **Technical Metadata** - messageType, dataFormat, timestamps
4. **Standard IBM ACE Infrastructure** - CopyEntireMessage(), CopyMessageHeaders()
5. **Error Information** - for failure handling and debugging

### STRICTLY PROHIBITED IN EVENT CAPTURE STAGES:
1. **Business Data Extraction** - NO extraction from message payload
2. **Business Transformation** - NO modification of business content
3. **Database Enrichment** - NO stored procedure calls for business logic
4. **Custom Business Fields** - NO customReference/customProperty with business values
5. **Message Content Changes** - NO alteration of the actual message

### COMPLIANT EVENT DATA TEMPLATE:
```
CREATE COMPUTE MODULE {module_name}
    CREATE FUNCTION Main() RETURNS BOOLEAN
    BEGIN
        -- METADATA CAPTURE ONLY
        DECLARE episInfo        REFERENCE TO Environment.variables.EventData.episInfo;
        DECLARE sourceInfo      REFERENCE TO Environment.variables.EventData.sourceInfo;
        DECLARE targetInfo      REFERENCE TO Environment.variables.EventData.targetInfo;
        DECLARE integrationInfo REFERENCE TO Environment.variables.EventData.integrationInfo;
        DECLARE dataInfo        REFERENCE TO Environment.variables.EventData.dataInfo;
        
        -- SOURCE/TARGET METADATA (for production support)
        SET sourceInfo.srcAppIdentifier = InputRoot.XMLNSC.[<].*:Header.*:Source.*:Identifier;
        SET sourceInfo.srcEnterpriseCode = InputRoot.XMLNSC.[<].*:Header.*:Source.*:EnterpriseCode;
        -- ... other Header metadata fields
        
        -- TECHNICAL METADATA ONLY
        SET dataInfo.messageType = InputRoot.XMLNSC.[<].*:Header.*:MessageType;
        SET dataInfo.dataFormat = 'XML';
        SET dataInfo.batch = false;
        
        -- ❌ DO NOT INCLUDE BUSINESS DATA:
        -- SET dataInfo.mainIdentifier = InputRoot.XMLNSC.[<].*:ShipmentInstruction.*:ShipmentDetails.*:ShipmentId;
        -- SET dataInfo.customReference1 = [business_value];
        
        -- STANDARD MESSAGE PROCESSING
        SET OutputRoot = NULL;
        CALL CopyEntireMessage();
        
        RETURN TRUE;
    END;
    
    -- MANDATORY: Standard IBM ACE Infrastructure
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
```

### VALIDATION CHECKLIST FOR EVENT CAPTURE:
- [ ] NO business data extraction from message payload
- [ ] NO database calls for business enrichment  
- [ ] NO custom business logic in event capture
- [ ] ONLY metadata and technical information captured
- [ ] Standard IBM ACE infrastructure included
"""
    
    else:
        # For non-event capture stages (transformation, enrichment, etc.)
        prompt += f"""

## BUSINESS TRANSFORMATION MODULE

### ALLOWED IN TRANSFORMATION STAGES:
1. **Business Logic Processing** - data transformation and enrichment
2. **Database Operations** - stored procedures for business data
3. **Message Transformation** - business content modification
4. **Custom Business Fields** - business-specific reference/property fields
5. **Error Handling** - business validation and error processing

### TRANSFORMATION TEMPLATE STRUCTURE:
```
CREATE COMPUTE MODULE {module_name}
    CREATE FUNCTION Main() RETURNS BOOLEAN
    BEGIN
        -- Business logic variables
        DECLARE businessData REFERENCE TO InputRoot.XMLNSC;
        DECLARE enrichedData REFERENCE TO OutputRoot.XMLNSC;
        
        -- Copy input to output first
        SET OutputRoot = InputRoot;
        
        -- Business transformation logic here
        -- Database lookups for enrichment
        -- Data validation and processing
        -- Custom business logic
        
        RETURN TRUE;
    END;
    
    -- MANDATORY: Standard IBM ACE Infrastructure
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
```
"""

    prompt += f"""

### REQUIRED ESQL STRUCTURE (ALL MODULES):

#### Complete Module Template:
```
CREATE COMPUTE MODULE {module_name}
    CREATE FUNCTION Main() RETURNS BOOLEAN
    BEGIN
        -- Your business logic or event data capture here
        -- (Follows compliance rules based on node type)
        
        RETURN TRUE;
    END;
    
    -- MANDATORY: These procedures must be included exactly as shown
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
```

#### Additional Requirements for Failure/Error Modules:
```
-- Add this function for error handling modules:
CREATE FUNCTION GetFaultDetailAsString(IN fault REFERENCE) RETURNS CHARACTER
BEGIN
    DECLARE str CHARACTER '';
    IF EXISTS(fault.*:detail.*:ExceptionDetail[]) THEN
        DECLARE exc REFERENCE TO fault.*:detail.*:ExceptionDetail[1];
        WHILE EXISTS(exc.*:Type[]) DO
            IF LENGTH(str) > 0 THEN
                SET str = str || ' ';
            END IF;
            SET str = str || COALESCE(exc.*:Type, '') || ': ' || COALESCE(exc.*:Message, '');
            MOVE exc TO exc.*:InnerException;
        END WHILE;
    END IF;
    RETURN str;
END;
```

#### Format Requirements:
- Module name: CREATE COMPUTE MODULE {module_name}
- NO .esql extension in module name
- NO "@" symbols anywhere
- NO lines starting with "esql"
- NO code block markers

#### Data Type Restrictions:
- ONLY use: BOOLEAN, INTEGER, DECIMAL, FLOAT, CHARACTER, BIT, BLOB, DATE, TIME, TIMESTAMP, REFERENCE, ROW
- NEVER use: XML, RECORD, STRING, VARCHAR (use REFERENCE TO InputRoot.XMLNSC for XML)

### CRITICAL REQUIREMENTS:
1. **Event Capture Compliance**: Business logic prohibited in event capture modules
2. **Complete Infrastructure**: All modules MUST include the exact procedures shown above
3. **Module Name Format**: CREATE COMPUTE MODULE {module_name} (NO .esql extension)
4. **Data Types**: ONLY use approved types (REFERENCE, CHARACTER, INTEGER, etc.)
5. **No Forbidden Elements**: No "@" symbols, no "esql" line prefixes, no code markers

## OUTPUT REQUIREMENTS:
Generate a complete, working ESQL module that:
- Includes ALL mandatory procedures exactly as shown above
- Follows Event Data compliance rules if this is an event capture stage
- Uses proper ESQL syntax and approved data types
- Contains NO explanatory text outside the ESQL code

Generate the complete ESQL module below:

"""
    
    return prompt


def get_xsl_consolidation_prompt(xsl_transform, biztalk_analysis, business_context) -> str:
    """
    Generate XSL transformation creation prompt with business context
    
    Args:
        xsl_transform: XSL transformation information/requirements
        biztalk_analysis: BizTalk project analysis data
        business_context: Business requirements context
        
    Returns:
        str: Comprehensive XSL generation prompt
    """
    
    # Extract transformation name and details
    transform_name = "Unknown_Transform"
    transform_purpose = "XSL transformation"
    
    if isinstance(xsl_transform, dict):
        transform_name = xsl_transform.get('name', transform_name)
        transform_purpose = xsl_transform.get('purpose', transform_purpose)
    elif isinstance(xsl_transform, str):
        transform_name = xsl_transform
    
    # Extract BizTalk context
    project_name = "ACE_Migration_Project"
    source_schemas = []
    target_schemas = []
    
    if isinstance(biztalk_analysis, dict):
        project_name = biztalk_analysis.get('project_name', project_name)
        
        # Extract schema information
        if 'data_transformations' in biztalk_analysis:
            for transform in biztalk_analysis['data_transformations']:
                if 'source_schema' in transform:
                    source_schemas.append(transform['source_schema'])
                if 'target_schema' in transform:
                    target_schemas.append(transform['target_schema'])
        
        # Extract map information
        if 'btm_maps' in biztalk_analysis:
            for btm_map in biztalk_analysis['btm_maps']:
                if 'source_schema' in btm_map:
                    source_schemas.append(btm_map['source_schema'])
                if 'target_schema' in btm_map:
                    target_schemas.append(btm_map['target_schema'])
    
    # Format business context
    business_requirements = business_context if isinstance(business_context, str) else "Standard XSL transformation"
    
    prompt = f"""# IBM ACE XSL Transformation Generation

Create a production-ready XSL transformation for IBM ACE with the following specifications:

## TRANSFORMATION DETAILS:
- **Transform Name**: {transform_name}
- **Purpose**: {transform_purpose}
- **Project**: {project_name}
- **Generated**: {datetime.now().isoformat()}

## BUSINESS CONTEXT:
```
{business_requirements[:1000]}
```

## BIZTALK MIGRATION CONTEXT:
- **Source Schemas**: {', '.join(source_schemas[:5]) if source_schemas else 'Not specified'}
- **Target Schemas**: {', '.join(target_schemas[:5]) if target_schemas else 'Not specified'}
- **Migration Type**: BizTalk Map to ACE XSL Transformation

## XSL TRANSFORMATION REQUIREMENTS:

### 🎯 TRANSFORMATION STANDARDS:
1. **Create production-ready XSL** that transforms source XML to target XML format
2. **Include proper XML declarations** and namespace handling
3. **Implement field mappings** based on BizTalk map analysis
4. **Add business rule logic** for data validation and transformation
5. **Include error handling** for missing or invalid data
6. **Follow ACE XSL best practices** for performance and maintainability

### 📋 MANDATORY XSL STRUCTURE:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    
    <!-- Include template matching and transformation logic -->
    <xsl:template match="/">
        <!-- Root template for document transformation -->
    </xsl:template>
    
    <!-- Include specific field mapping templates -->
    <!-- Include business rule implementations -->
    <!-- Include error handling templates -->
    
</xsl:stylesheet>
```

### 🔧 TRANSFORMATION LOGIC:
- **Field Mappings**: Transform source fields to target format
- **Data Validation**: Check required fields and data formats  
- **Business Rules**: Apply transformation logic and calculations
- **Namespace Handling**: Manage XML namespaces correctly
- **Error Management**: Handle missing data and validation failures

### 🚀 PERFORMANCE REQUIREMENTS:
- **Efficient Templates**: Use efficient XPath expressions
- **Memory Optimization**: Avoid recursive templates where possible
- **Processing Speed**: Optimize for large message volumes
- **Resource Management**: Minimize memory footprint

### 🎨 OUTPUT REQUIREMENTS:
- **Complete XSL File**: Generate full transformation stylesheet
- **Valid XML Syntax**: Ensure well-formed XSL document
- **Production Ready**: Include all necessary templates and logic
- **Commented Code**: Add clear comments explaining transformation logic
- **Error Resilient**: Handle edge cases and data anomalies

## CRITICAL XSL CODING RULES:

### ✅ REQUIRED ELEMENTS:
1. **Proper XML declaration** with UTF-8 encoding
2. **XSL namespace declaration** (xmlns:xsl="http://www.w3.org/1999/XSL/Transform")
3. **Root template match="/"** for document processing
4. **Field-specific templates** for complex transformations
5. **Error handling templates** for data validation

### 🚫 FORBIDDEN ELEMENTS:
1. **No hardcoded values** - use parameterized transformations
2. **No infinite recursion** - ensure template termination
3. **No missing namespaces** - include all required namespace declarations
4. **No unclosed tags** - ensure proper XML structure
5. **No performance bottlenecks** - avoid inefficient XPath expressions

## BUSINESS TRANSFORMATION LOGIC:
Based on the business context and BizTalk analysis, implement the specific transformation logic required for this module.

Generate the complete, production-ready XSL transformation below:

```xml"""
    
    return prompt

def get_enhancement_prompt(original_esql: str, enhancement_type: str, context: Dict = None) -> str:
    """Generate prompt for enhancing existing ESQL code"""
    
    prompt = f"""# ESQL Module Enhancement Task

Enhance the following ESQL module with {enhancement_type} capabilities:

## ORIGINAL ESQL CODE:
```
{original_esql}
```

## ENHANCEMENT TYPE: {enhancement_type}

## CRITICAL ESQL RULES - MUST FOLLOW:
1. **NEVER start any line with "esql"**
2. **NEVER use "@" symbol anywhere**
3. **NEVER include code block markers**
4. **Generate pure ESQL code only**

## ENHANCEMENT REQUIREMENTS:
"""
    
    if enhancement_type == "database_operations":
        prompt += """
### Database Enhancement:
- Add robust database lookup procedures
- Implement connection pooling and error handling
- Add retry logic for transient failures
- Use parameterized queries for security
- Include audit logging for all database operations

### Required Database Pattern:
```
DECLARE EXIT HANDLER FOR SQLEXCEPTION
BEGIN
    SET Environment.Variables.LastSQLError = SQLERRORTEXT;
    PROPAGATE TO TERMINAL 'failure';
END;

SET DATABASE.CompanyCode = InputRoot.XMLNSC.Header.CompanyCode;
SET CompanyName = THE(SELECT ITEM C.CompanyName FROM Database.Companies AS C WHERE C.CompanyCode = DATABASE.CompanyCode);
```
"""
    
    elif enhancement_type == "business_logic":
        prompt += """
### Business Logic Enhancement:
- Implement complex business rules and validations
- Add decision trees for routing logic
- Include data enrichment and transformation
- Add proper field validation and error messages
"""
    
    elif enhancement_type == "error_handling":
        prompt += """
### Error Handling Enhancement:
- Implement comprehensive try/catch patterns
- Add specific error codes and messages
- Include proper error logging and auditing
- Add error recovery mechanisms
"""
    
    prompt += f"""

## CONTEXT INFORMATION:
{json.dumps(context, indent=2) if context else "No additional context provided"}

## OUTPUT REQUIREMENTS:
- Enhance the original ESQL with the requested capabilities
- Maintain all existing functionality
- Follow IBM ACE best practices
- Generate complete, production-ready ESQL module
- No code block markers or language identifiers

## ENHANCED ESQL MODULE:
"""
    
    return prompt


def get_quality_analysis_prompt(esql_code: str) -> str:
    """Generate prompt for ESQL quality analysis"""
    
    return f"""# ESQL Code Quality Analysis

Analyze the following ESQL code for quality, performance, and best practices:

## ESQL CODE TO ANALYZE:
```
{esql_code}
```

## ANALYSIS CRITERIA:
1. **Code Quality**: Syntax, structure, readability
2. **Performance**: Efficiency, optimization opportunities
3. **Best Practices**: IBM ACE standards compliance
4. **Error Handling**: Robustness and error coverage
5. **Security**: Input validation and SQL injection protection
6. **Maintainability**: Code organization and documentation

## FORBIDDEN ELEMENTS CHECK:
- Verify no lines start with "esql"
- Verify no "@" symbols are used
- Verify proper ESQL syntax throughout

## PROVIDE ANALYSIS:
1. **Quality Score**: Rate from 1-100
2. **Issues Found**: List any problems or violations
3. **Improvements**: Specific recommendations
4. **Best Practices**: Adherence to IBM ACE standards

## ANALYSIS RESULT:
"""


# Utility functions for prompt management
def save_prompt_to_file(prompt: str, filename: str) -> str:
    """Save generated prompt to file for reference"""
    import os
    os.makedirs("prompts", exist_ok=True)
    filepath = os.path.join("prompts", filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(prompt)
    return filepath


def load_template(template_name: str) -> str:
    """Load prompt template from file"""
    try:
        with open(f"templates/{template_name}.txt", 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"Template {template_name} not found"


def get_functional_documentation_prompt(biztalk_analysis: Dict, confluence_analysis: Dict = None) -> str:
    """
    Generate prompt for creating functional requirements documentation
    
    Args:
        biztalk_analysis: Analysis results from BizTalk scanning
        confluence_analysis: Optional confluence specification analysis
    """
    
    prompt = f"""# Functional Requirements Document Generation

You are a business analyst expert in BizTalk to IBM ACE migration. Generate a comprehensive functional requirements document.

## BIZTALK ANALYSIS DATA:
```json
{json.dumps(biztalk_analysis, indent=2)[:2000]}
```

## BUSINESS SPECIFICATION:
```
{json.dumps(confluence_analysis, indent=2)[:1500] if confluence_analysis else "No business specification provided"}
```

## DOCUMENT REQUIREMENTS:

### 1. Executive Summary
- Purpose and scope of the migration
- High-level business benefits
- Key stakeholders and impact areas

### 2. Current State Analysis
- BizTalk architecture overview
- Component inventory and dependencies
- Integration points and data flows
- Business processes supported

### 3. Future State Design
- IBM ACE target architecture
- Component mapping strategy
- New capabilities and improvements
- Integration patterns in ACE

### 4. Functional Requirements
- Business process requirements
- Data transformation requirements
- Integration requirements
- Performance and scalability requirements
- Security and compliance requirements

### 5. Migration Strategy
- Component migration approach
- Testing strategy
- Deployment approach
- Risk mitigation
- Timeline and milestones

### 6. Technical Specifications
- ACE component specifications
- ESQL module requirements
- Message flow definitions
- Schema and data model requirements

## OUTPUT FORMAT:
Generate a comprehensive markdown document with:
- Clear section headers using # and ##
- Bullet points for lists and requirements
- Tables for component mappings where appropriate
- Professional business language
- Technical accuracy for IBM ACE
- Actionable recommendations

## FUNCTIONAL REQUIREMENTS DOCUMENT:
"""
    
    return prompt


# Configuration for different prompt types
PROMPT_CONFIG = {
    "msgflow": {
        "max_template_size": 5000,
        "max_spec_size": 2000,
        "include_business_context": True
    },
    "esql": {
        "max_context_size": 1500,
        "enforce_coding_rules": True,
        "include_error_handling": True
    },
    "enhancement": {
        "preserve_existing_logic": True,
        "add_production_features": True
    },
    "documentation": {
        "max_analysis_size": 2000,
        "include_technical_details": True,
        "business_focus": True
    }
}