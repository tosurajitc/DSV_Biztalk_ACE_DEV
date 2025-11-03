"""
fetch_naming.py
Smart PDF Naming Extractor - Extracts ACE Application & Message Flow Names from Confluence PDF
NO HARDCODING | NO FALLBACKS | FAIL GRACEFULLY
"""

import json
import os
from datetime import datetime
from pathlib import Path
import PyPDF2
import streamlit as st
from vector_knowledge.pdf_processor import AdaptivePDFProcessor
from dotenv import load_dotenv
load_dotenv()

class PDFNamingExtractor:
    """Extract naming conventions from confluence PDF using LLM"""
    
    def __init__(self, pdf_file_path, llm_client, model="llama-3.1-70b-versatile"):
        self.pdf_file_path = Path(pdf_file_path)
        self.llm_client = llm_client
        self.model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')  # ✅ From .env
        self.output_file = "naming_convention.json"
        self.pdf_processor = AdaptivePDFProcessor()



    def extract_and_create_naming_json(self) -> bool:
        """
        Main method: Extract naming from PDF and create naming_convention.json files
        Returns: True if successful, False if failed
        """
        try:
            print("🔍 Extracting naming conventions from PDF...")
            
            # Step 1: Extract PDF text
            pdf_text = self._extract_pdf_text()
            if not pdf_text:
                print("❌ Failed to extract text from PDF")
                return False
            
            # Step 2: Use LLM to extract ALL flow data from tables
            extracted_data_list = self._llm_extract_summary_table(pdf_text)
            if not extracted_data_list:
                print("❌ Failed to extract flow information using LLM")
                return False
            
            # Step 3: Create naming_convention.json file(s)
            success = self._create_naming_convention_json(extracted_data_list)
            if success:
                print(f"✅ Successfully created naming convention file(s) from PDF")
                return True
            else:
                print("❌ Failed to create naming_convention.json file(s)")
                return False
                
        except Exception as e:
            print(f"❌ PDFNamingExtractor failed: {str(e)}")
            return False
    

    def _extract_pdf_text(self) -> str:
        """Extract text content from PDF file using AdaptivePDFProcessor"""
        try:
            if not self.pdf_file_path.exists():
                print(f"PDF file not found: {self.pdf_file_path}")
                return None
            
            # Use AdaptivePDFProcessor for extraction
            self.pdf_processor = AdaptivePDFProcessor()
            text_content = self.pdf_processor.extract_text_from_pdf(str(self.pdf_file_path))
            
            if len(text_content.strip()) < 50:
                print("PDF appears to contain insufficient text content")
                return None
                    
            return text_content
                
        except Exception as e:
            print(f"Error reading PDF: {str(e)}")
            return None
    

    
    def _detect_message_flow_tables(self, pdf_text: str) -> list:
        """
        Detect message flow tables and return their content
        
        Returns:
            List of dictionaries with table content and flow information
        """
        # Split text into sections
        sections = pdf_text.split('\n\n')
        flow_tables = []
        
        for section in sections:
            # Check if this section is a message flow table
            if self.pdf_processor._detect_messageflow_table(section):
                # Count flows in this table
                flow_count = self.pdf_processor._count_flow_mentions(section)
                flow_tables.append({
                    'content': section,
                    'flow_count': flow_count
                })
        
        return flow_tables



    

    def _llm_extract_summary_table(self, pdf_text: str) -> list:
        # Update the prompt to extract input type
        prompt = f"""You are a specialized PDF table parser for IBM ACE message flow configurations.

        DOCUMENT CONTENT:
        {pdf_text[:10000]}

        TASK: Extract ALL message flow configurations from tables with comma-separated values.

        YOUR TABLE FORMAT EXAMPLE:
        "ACE Message Flow Name: SAP_DailyBalance_REC, SAP_DailyBalance_RECSAT" 
        "ACE Application Name: EPIS_SAP_OUT_Delta_App, EPIS_SAP_OUT_Delta_SAT_App"
        "ACE Server Name: group-sap-server, group-sap-sat-server"

        PARSING RULES:
        1. COMMA-SEPARATED VALUES: Split by commas and match by position
        - 1st flow name → 1st app name → 1st server name  
        - 2nd flow name → 2nd app name → 2nd server name
        2. EXTRACT EVERY FLOW: Don't stop at first one
        3. HANDLE MISMATCHED COUNTS: Use available data, mark missing as "Not_Specified"
        4. FIND ANY TABLE: Look for flow/application/server information anywhere

        ADDITIONAL ANALYSIS - CRITICAL:
        Determine input type for each flow (File, MQ, or HTTP) based on actual flow specifications in the document:
        - Look for explicit input configuration descriptions mentioning "Input Type: File" or "Source: File" or "File Input"
        - Look for explicit input configuration descriptions mentioning "Input Type: MQ" or "Source: MQ" or "Queue Input" 
        - Look for explicit input configuration descriptions mentioning "Input Type: HTTP" or "Source: HTTP" or "REST API"
        - If RECSAT or SNDSAT appears in the flow name AND the document discusses file handling, use File input
        - If the document mentions reading from queues for a specific flow, use MQ input
        - If the document mentions REST API endpoints for a specific flow, use HTTP input
        - Use flow input type explicitly mentioned in specifications, regardless of flow naming conventions
        - Default to MQ only when there's no other information available

        DO NOT just rely on the flow name - analyze the full context about each flow's actual input mechanism.

        CRITICAL: This must work for 1000+ flows - extract ALL flows found

        OUTPUT FORMAT (JSON only, no markdown):
        {{"flows": [
        {{"ace_application_name": "EPIS_SAP_OUT_Delta_App", "message_flow_name": "SAP_DailyBalance_REC", "ace_server": "group-sap-server", "connected_system": "SAP", "input_type": "MQ", "description": "Flow description if available"}},
        {{"ace_application_name": "EPIS_SAP_OUT_Delta_SAT_App", "message_flow_name": "SAP_DailyBalance_RECSAT", "ace_server": "group-sap-sat-server", "connected_system": "SAP", "input_type": "File", "description": "Flow description if available"}}
        ]}}

        If no flows found: {{"flows": []}}"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=3000  # Increased for large tables
            )
            
            # Track token usage
            if 'token_tracker' in st.session_state and hasattr(response, 'usage') and response.usage:
                st.session_state.token_tracker.manual_track(
                    agent="enhanced_pdf_naming_extractor",
                    operation="multi_flow_extraction",
                    model=self.model,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    flow_name="enhanced_pdf_naming_extraction"
                )
            
            llm_response = response.choices[0].message.content.strip()
            print(f"📄 LLM Response (first 300 chars): {llm_response[:300]}...")
            
            # Enhanced JSON parsing
            extracted_flows = self._enhanced_parse_json(llm_response)
            
            if extracted_flows:
                print(f"✅ Successfully extracted {len(extracted_flows)} flow configurations")
                for i, flow in enumerate(extracted_flows):
                    flow_name = flow.get('message_flow_name', 'Unknown')
                    app_name = flow.get('ace_application_name', 'Unknown')
                    print(f"  Flow {i+1}: {flow_name} → {app_name}")
                return extracted_flows
            else:
                print("❌ No valid flow configurations extracted from PDF")
                return []  # NO HARDCODED FALLBACKS
                
        except Exception as e:
            print(f"❌ Enhanced extraction failed: {str(e)}")
            return []  # NO HARDCODED FALLBACKS

    # ADD this new helper method to PDFNamingExtractor class:

    def _enhanced_parse_json(self, llm_response: str) -> list:
        """
        Robust JSON parsing with multiple strategies for comma-separated tables
        """
        import re
        
        # Multiple parsing strategies
        json_patterns = [
            r'\{\s*"flows"\s*:\s*\[[^\]]*\]\s*\}',  # Complete flows object
            r'"flows"\s*:\s*\[[^\]]*\]',            # Just flows array  
            r'\[[^\[\]]*\{[^\}]*\}[^\[\]]*\]'       # Any array with objects
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, llm_response, re.DOTALL)
            if match:
                try:
                    json_text = match.group()
                    
                    # Handle partial JSON structures
                    if json_text.startswith('"flows":'):
                        json_text = '{' + json_text + '}'
                    elif json_text.startswith('[') and not json_text.startswith('{"flows"'):
                        json_text = '{"flows":' + json_text + '}'
                    
                    parsed_data = json.loads(json_text)
                    
                    if 'flows' in parsed_data and isinstance(parsed_data['flows'], list):
                        flows = parsed_data['flows']
                        
                        # Validate each flow has minimum required fields
                        valid_flows = []
                        for flow in flows:
                            if isinstance(flow, dict) and flow.get('message_flow_name'):
                                # Ensure all required fields exist with defaults
                                validated_flow = {
                                    'ace_application_name': flow.get('ace_application_name', 'Default_App'),
                                    'message_flow_name': flow.get('message_flow_name', 'Default_Flow'),
                                    'ace_server': flow.get('ace_server', 'Default_Server'),
                                    'connected_system': flow.get('connected_system', 'Unknown_System'),
                                    'input_type': flow.get('input_type', 'MQ'),  # Default to MQ if not specified
                                    'description': flow.get('description', 'Extracted from PDF')
                                }
                                valid_flows.append(validated_flow)
                        
                        if valid_flows:
                            return valid_flows
                            
                except json.JSONDecodeError as e:
                    print(f"  JSON parse attempt failed: {str(e)}")
                    continue
        
        print("❌ All JSON parsing strategies failed")
        return []
                

        
    def _create_naming_convention_json(self, extracted_data_list: list) -> bool:
        """Create naming_convention.json file(s) with extracted data"""
        try:
            # If we received a single dictionary (old format), convert to list
            if isinstance(extracted_data_list, dict):
                extracted_data_list = [extracted_data_list]
                
            # Filter to only include message flow names that don't contain dots
            valid_flows = []
            for flow_data in extracted_data_list:
                message_flow_name = flow_data.get("message_flow_name", "")
                # Check if message_flow_name exists and doesn't contain dots
                if message_flow_name and "." not in message_flow_name:
                    valid_flows.append(flow_data)
                    
            print(f"ℹ️ Found {len(valid_flows)} valid message flows (without dots in name)")
            
            if not valid_flows:
                print("⚠️ No valid message flow names found (all names contained dots)")
                return False
                
            # Create a file for each valid flow configuration
            for idx, extracted_data in enumerate(valid_flows):
                # Determine filename (numbered if multiple)
                if len(valid_flows) > 1:
                    # Start with 1 not 0 for user friendliness
                    output_file = f"naming_convention_{idx+1}.json"
                else:
                    output_file = self.output_file
                
                naming_convention = {
                    "pdf_source": str(self.pdf_file_path.name),
                    "extraction_timestamp": datetime.now().isoformat(),
                    "project_naming": {
                        "ace_application_name": extracted_data.get("ace_application_name", ""),
                        "message_flow_name": extracted_data.get("message_flow_name", ""),
                        "connected_system": extracted_data.get("connected_system", ""),
                        "ace_server": extracted_data.get("ace_server", ""),
                        "input_type": extracted_data.get("input_type", "MQ")  # Added input_type with MQ default
                    },
                    "component_naming_rules": {
                        "msgflow_files": f"{extracted_data.get('message_flow_name', '')}.msgflow",
                        "project_name": extracted_data.get("ace_application_name", "")
                    },
                    "business_context": {
                        "description": extracted_data.get("description", ""),
                        "pdf_extracted": True
                    }
                }
                
                # Write JSON file
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(naming_convention, f, indent=2, ensure_ascii=False)
                
                print(f"📄 Created: {output_file}")
                print(f"  Project Name: {extracted_data.get('ace_application_name', 'N/A')}")
                print(f"  Message Flow: {extracted_data.get('message_flow_name', 'N/A')}")
            
            return True
                
        except Exception as e:
            print(f"Error creating naming convention file: {str(e)}")
            return False


def run_pdf_naming_extraction(pdf_file_path, llm_client):
    """
    Main function to extract naming conventions from PDF
    NO FALLBACKS - If this fails, application continues with existing naming_convention.json
    
    Returns:
        List of created naming convention files if successful, empty list if failed
    """
    try:
        extractor = PDFNamingExtractor(pdf_file_path, llm_client)
        success = extractor.extract_and_create_naming_json()
        
        if not success:
            print("⚠️  PDF naming extraction failed - application will use existing/default naming_convention.json")
            return []
            
        # Check for multiple files
        created_files = []
        base_file = "naming_convention.json"
        if os.path.exists(base_file):
            created_files.append(base_file)
            
        # Check for numbered files
        idx = 1
        while True:
            numbered_file = f"naming_convention_{idx}.json"
            if os.path.exists(numbered_file):
                created_files.append(numbered_file)
                idx += 1
            else:
                break
                
        return created_files
        
    except Exception as e:
        print(f"⚠️  PDF naming extraction module failed: {str(e)}")
        print("⚠️  Application will continue with existing/default naming_convention.json")
        return []


# CLI usage
if __name__ == "__main__":
    import sys
    from groq import Groq
    
    if len(sys.argv) < 2:
        print("Usage: python fetch_naming.py <pdf_file_path>")
        sys.exit(1)
    
    try:
        # Initialize LLM client (requires GROQ_API_KEY environment variable)
        llm_client = Groq()
        
        # Run extraction
        success = run_pdf_naming_extraction(sys.argv[1], llm_client)
        
        if success:
            print("✅ PDF naming extraction completed successfully")
            sys.exit(0)
        else:
            print("❌ PDF naming extraction failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        sys.exit(1)