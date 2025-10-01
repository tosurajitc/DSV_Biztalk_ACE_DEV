#!/usr/bin/env python3
"""
XSL Generator v2.0 - Hybrid Template + LLM Approach
Pattern: Template provides structure, LLM fills business logic blocks
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from groq import Groq
import streamlit as st

class XSLGenerator:
    """
    Hybrid XSL Generator
    Template (85%) + LLM-generated business logic blocks (15%)
    """
    
    def __init__(self, groq_api_key: str = None):
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY required")
        
        self.llm = Groq(api_key=self.groq_api_key)
        self.groq_model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
        
        # Template paths
        self.template_dir = Path(__file__).parent / 'templates'
        self.xsl_template_path = self.template_dir / 'xsl_template_with_placeholders.xml'
        self.contract_path = self.template_dir / 'enrichment_attribute_contract.json'
        
        self.llm_calls = 0
    
    def generate_xsl_transformations(self, 
                                    vector_content: str,
                                    component_mapping_json_path: str,
                                    output_dir: str,
                                    flow_name: str) -> Dict[str, Any]:
        """
        Generate XSL using template + LLM business logic insertion
        """
        print(f"🎨 Starting XSL generation for: {flow_name}")
        print("📋 Method: Template + LLM business logic blocks")
        
        try:
            # Step 1: Load template
            print("\n📄 Step 1: Loading XSL template...")
            with open(self.xsl_template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            print(f"   ✅ Template loaded: {len(template_content)} characters")
            
            # Step 2: Load contract
            print("\n📋 Step 2: Loading enrichment contract...")
            with open(self.contract_path, 'r', encoding='utf-8') as f:
                contract = json.load(f)
            print(f"   ✅ Contract loaded")
            
            # Step 3: LLM analyzes business requirements
            print("\n🧠 Step 3: LLM analyzing business requirements...")
            business_analysis = self._llm_analyze_business_requirements(
                vector_content, 
                component_mapping_json_path,
                contract
            )
            print(f"   ✅ Analysis complete")
            
            # Step 4: LLM generates placeholder content
            print("\n⚡ Step 4: LLM generating business logic blocks...")
            placeholder_content = self._llm_generate_placeholder_blocks(
                business_analysis,
                flow_name
            )
            print(f"   ✅ Generated {len(placeholder_content)} blocks")
            
            # Step 5: Insert LLM content into template
            print("\n🔧 Step 5: Assembling final XSL...")
            final_xsl = self._insert_placeholders(template_content, placeholder_content)
            print(f"   ✅ Final XSL: {len(final_xsl)} characters")
            
            # Step 6: Write output
            print("\n💾 Step 6: Writing XSL file...")
            transforms_dir = Path(output_dir) / 'transforms'
            transforms_dir.mkdir(parents=True, exist_ok=True)
            output_file = transforms_dir / f"{flow_name}.xsl"
            
            with open(output_file, 'w', encoding='utf-16') as f:
                f.write(final_xsl)
            print(f"   ✅ File written: {output_file}")
            
            return {
                'status': 'success',
                'xsl_transformations_generated': 1,
                'xsl_files': [str(output_file)],
                'output_directory': str(transforms_dir),
                'llm_analysis_calls': 1,
                'llm_generation_calls': self.llm_calls - 1,
                'processing_metadata': {
                    'flow_name': flow_name,
                    'generation_method': 'template_plus_llm',
                    'template_size': len(template_content),
                    'final_size': len(final_xsl),
                    'placeholders_filled': len(placeholder_content),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            print(f"❌ XSL generation failed: {str(e)}")
            return {
                'status': 'failed',
                'xsl_transformations_generated': 0,
                'xsl_files': [],
                'llm_analysis_calls': 0,
                'llm_generation_calls': 0,
                'processing_metadata': {'error': str(e)}
            }
    
    def _llm_analyze_business_requirements(self, vector_content: str, 
                                          json_path: str, 
                                          contract: dict) -> dict:
        """LLM analyzes Vector DB + JSON to extract business requirements"""
        
        # Load JSON mappings
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        system_prompt = """You are analyzing business requirements for XSL transformation generation.
Extract ONLY the information needed to determine:
1. Which enrichment attributes are used (from contract)
2. Priority order for entity key selection
3. Conditional logic requirements
4. Company context requirements

Return valid JSON only."""

        user_prompt = f"""Analyze business requirements:

## VECTOR DB CONTENT:
{vector_content[:3000]}

## COMPONENT MAPPINGS:
{json.dumps(json_data, indent=2)[:2000]}

## AVAILABLE ENRICHMENT ATTRIBUTES:
{json.dumps(contract['esql_contract']['required_attributes'], indent=2)}

Extract and return JSON with:
{{
  "enrichment_attributes_used": ["@EE_ShipmentId_by_SSN", "@EE_ShipmentId_by_HouseBill", ...],
  "entity_key_priority": ["SSN_lookup", "HouseBill_lookup", "direct_reference"],
  "company_context_required": true/false,
  "additional_context_needed": ["MasterBill"] or []
}}"""

        response = self.llm.chat.completions.create(
            model=self.groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        self.llm_calls += 1
        
        if 'token_tracker' in st.session_state and hasattr(response, 'usage'):
            st.session_state.token_tracker.manual_track(
                agent="xsl_generator",
                operation="business_analysis",
                model=self.groq_model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens
            )
        
        content = response.choices[0].message.content.strip()
        # Extract JSON from markdown blocks if present
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()
        
        return json.loads(content)
    
    def _llm_generate_placeholder_blocks(self, analysis: dict, flow_name: str) -> dict:
        """LLM generates XSL blocks for each placeholder"""
        
        placeholders = {}
        
        # Generate ENTITY_KEY_LOGIC (most critical)
        placeholders['PLACEHOLDER_ENTITY_KEY_LOGIC'] = self._generate_entity_key_block(analysis)
        
        # Generate COMPANY_CODE_VARIABLE
        placeholders['PLACEHOLDER_COMPANY_CODE_VARIABLE'] = self._generate_company_code_variable(analysis)
        
        # Generate RECIPIENT_ID_VARIABLE
        placeholders['PLACEHOLDER_RECIPIENT_ID_VARIABLE'] = self._generate_recipient_id_variable()
        
        # Generate COMPANY_CONTEXT
        if analysis.get('company_context_required'):
            placeholders['PLACEHOLDER_COMPANY_CONTEXT'] = self._generate_company_context(analysis)
        else:
            placeholders['PLACEHOLDER_COMPANY_CONTEXT'] = ''
        
        # Generate ADDITIONAL_CONTEXT
        if analysis.get('additional_context_needed'):
            placeholders['PLACEHOLDER_ADDITIONAL_CONTEXT'] = self._generate_additional_context(analysis)
        else:
            placeholders['PLACEHOLDER_ADDITIONAL_CONTEXT'] = ''
        
        return placeholders
    


    
    def _generate_entity_key_block(self, analysis: dict) -> str:
        """Generate the critical entity key selection logic with strict contract enforcement"""
        
        system_prompt = """You are generating XSL conditional logic for entity key selection.

    CRITICAL RULES:
    1. Use ONLY the exact attribute names from the contract (e.g., @EE_ShipmentId_by_SSN)
    2. Entity type checks MUST precede attribute checks
    3. Follow the proven developer pattern structure exactly
    4. NO invented attribute names - only use contract-defined attributes
    5. Use xsl:choose with proper when/otherwise structure
    6. All attribute checks use: normalize-space(@AttributeName) != ''

    Return ONLY valid XSL code, no markdown, no explanations."""

        user_prompt = f"""Generate entity key selection XSL block following this EXACT pattern:

    ## CONTRACT-DEFINED ATTRIBUTES (USE THESE ONLY):
    - @EE_ShipmentId_by_SSN (from sp_Shipment_GetIdBySSN stored procedure)
    - @EE_ShipmentId_by_HouseBill (from proc_Shipment_GetIdByHouseBill stored procedure)
    - @CW1BrokerageId (from brokerage lookup)

    ## DEVELOPER PATTERN TO FOLLOW:
    <xsl:choose>
    <!-- Pattern 1: Check for MasterBill (special case) -->
    <xsl:when test="s0:Document/s0:EntityReference/s0:Reference[@Type='MasterBill']">
        <!-- Empty - MasterBill uses ContextCollection instead -->
    </xsl:when>
    
    <!-- Pattern 2: Shipment/Booking via SSN -->
    <xsl:when test="(s0:Document/s0:EntityReference/s0:Type/text()='SHP' or s0:Document/s0:EntityReference/s0:Type/text()='QBK') 
                    and s0:Document/s0:EntityReference/s0:Reference[@Type='SSN'] 
                    and normalize-space(@EE_ShipmentId_by_SSN) != ''">
        <xsl:value-of select="@EE_ShipmentId_by_SSN"/>
    </xsl:when>
    
    <!-- Pattern 3: Shipment/Booking via HouseBill -->
    <xsl:when test="(s0:Document/s0:EntityReference/s0:Type/text()='SHP' or s0:Document/s0:EntityReference/s0:Type/text()='QBK') 
                    and s0:Document/s0:EntityReference/s0:Reference[@Type='HouseBill'] 
                    and normalize-space(@EE_ShipmentId_by_HouseBill) != ''">
        <xsl:value-of select="@EE_ShipmentId_by_HouseBill"/>
    </xsl:when>
    
    <!-- Pattern 4: Brokerage ID (if applicable) -->
    <xsl:when test="@CW1BrokerageId != ''">
        <xsl:value-of select="@CW1BrokerageId"/>
    </xsl:when>
    
    <!-- Pattern 5: Default - use first reference directly -->
    <xsl:otherwise>
        <xsl:value-of select="s0:Document/s0:EntityReference/s0:Reference[1]/text()"/>
    </xsl:otherwise>
    </xsl:choose>

    ## YOUR TASK:
    Based on the business analysis below, generate the entity key block using this EXACT pattern.
    Only include the patterns that are relevant based on the enrichment attributes identified.

    Business Analysis:
    - Enrichment attributes used: {analysis.get('enrichment_attributes_used', [])}
    - Priority order: {analysis.get('entity_key_priority', [])}

    RULES:
    1. If @EE_ShipmentId_by_SSN in attributes → include Pattern 2
    2. If @EE_ShipmentId_by_HouseBill in attributes → include Pattern 3
    3. If @CW1BrokerageId in attributes → include Pattern 4
    4. ALWAYS include Pattern 1 (MasterBill check) and Pattern 5 (default)
    5. Keep entity type checks: (s0:Document/s0:EntityReference/s0:Type/text()='SHP' or ...)
    6. Keep reference type checks: s0:Document/s0:EntityReference/s0:Reference[@Type='SSN']
    7. NEVER invent new attribute names like @SSN_lookup or @HouseBill_lookup

    Generate the complete xsl:choose block now."""

        response = self.llm.chat.completions.create(
            model=self.groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=1500
        )
        
        self.llm_calls += 1
        
        if 'token_tracker' in st.session_state and hasattr(response, 'usage'):
            st.session_state.token_tracker.manual_track(
                agent="xsl_generator",
                operation="entity_key_generation",
                model=self.groq_model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens
            )
        
        content = response.choices[0].message.content.strip()
        
        # Clean markdown blocks
        content = re.sub(r'```xml\n?', '', content)
        content = re.sub(r'```xsl\n?', '', content)
        content = re.sub(r'```\n?', '', content)
        
        # Validate output contains required patterns
        validation_errors = []
        
        if '@EE_ShipmentId_by_SSN' in str(analysis.get('enrichment_attributes_used', [])):
            if '@EE_ShipmentId_by_SSN' not in content:
                validation_errors.append("Missing required attribute: @EE_ShipmentId_by_SSN")
        
        if '@EE_ShipmentId_by_HouseBill' in str(analysis.get('enrichment_attributes_used', [])):
            if '@EE_ShipmentId_by_HouseBill' not in content:
                validation_errors.append("Missing required attribute: @EE_ShipmentId_by_HouseBill")
        
        # Check for invented attributes (common LLM mistake)
        invented_attrs = ['@SSN_lookup', '@HouseBill_lookup', '@direct_reference', '@EntityKey']
        for attr in invented_attrs:
            if attr in content:
                validation_errors.append(f"LLM invented non-contract attribute: {attr}")
        
        # REMOVED: fallback call
        # NEW: Raise exception if validation fails
        if validation_errors:
            error_msg = "\n".join(validation_errors)
            print(f"  ❌ LLM validation failed:")
            for error in validation_errors:
                print(f"     - {error}")
            
            # Log the bad output for debugging
            print(f"  📋 LLM generated content (INVALID):")
            print(content[:500])
            
            raise Exception(
                f"XSL entity key generation validation failed:\n{error_msg}\n"
                f"LLM must use only contract-defined attributes. "
                f"Check Vector DB business analysis for correct attribute identification."
            )
        
        return content.strip()


    
    def _generate_company_code_variable(self, analysis: dict) -> str:
        """Generate CompanyCode variable block"""
        return """<xsl:variable name="CompanyCode">
      <xsl:choose>
        <xsl:when test="normalize-space(@CompanyCode) != ''">
          <xsl:value-of select="@CompanyCode"/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:value-of select="concat(s0:Header/s0:Target/s0:CountryCode,'1')"/>
        </xsl:otherwise>
      </xsl:choose>
    </xsl:variable>"""
    
    def _generate_recipient_id_variable(self) -> str:
        """Generate eAdapterRecipientId variable"""
        return """<xsl:variable name="eAdapterRecipientId">
      <xsl:choose>
        <xsl:when test="s0:Header/@eAdapterRecipientId != ''">
          <xsl:value-of select="s0:Header/@eAdapterRecipientId"/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:choose>
            <xsl:when test="ext:GetESBEnvironment()='Production'">
              <xsl:text>DFDXXXPRD</xsl:text>
            </xsl:when>
            <xsl:otherwise>
              <xsl:text>DFDXXXDQA</xsl:text>
            </xsl:otherwise>
          </xsl:choose>
        </xsl:otherwise>
      </xsl:choose>
    </xsl:variable>"""
    
    def _generate_company_context(self, analysis: dict) -> str:
        """Generate company context conditional"""
        return """<xsl:if test="(s0:Document/s0:EntityReference/s0:Type='INV' or $var_SourceApplicationCode='DocPackApp') 
                      and (normalize-space(s0:Header/s0:Target/s0:CompanyCode)!='')">
      <Company>
        <Code>
          <xsl:value-of select="s0:Header/s0:Target/s0:CompanyCode" />
        </Code>
      </Company>
    </xsl:if>"""
    
    def _generate_additional_context(self, analysis: dict) -> str:
        """Generate additional context for special cases"""
        if 'MasterBill' in analysis.get('additional_context_needed', []):
            return """<xsl:if test="s0:Document/s0:EntityReference/s0:Reference[@Type ='MasterBill']">
      <ContextCollection>
        <Context>
          <Type>MBOLNumber</Type>
          <Value>
            <xsl:value-of select="s0:Document/s0:EntityReference/s0:Reference[@Type ='MasterBill']/text()"/>
          </Value>
        </Context>
        <Context>
          <Type>MAWBNumber</Type>
          <Value>
            <xsl:value-of select="s0:Document/s0:EntityReference/s0:Reference[@Type ='MasterBill']/text()"/>
          </Value>
        </Context>
      </ContextCollection>
    </xsl:if>"""
        return ''
    
    def _insert_placeholders(self, template: str, placeholders: dict) -> str:
        """Insert LLM-generated content into template placeholders"""
        result = template
        for placeholder, content in placeholders.items():
            result = result.replace(f"{{{placeholder}}}", content)
        
        # Remove any remaining unfilled placeholders
        result = re.sub(r'\{PLACEHOLDER_[A-Z_]+\}', '', result)
        
        return result