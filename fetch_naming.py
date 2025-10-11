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
from vector_knowledge.pdf_processor import PDFProcessor
from dotenv import load_dotenv
load_dotenv()

class PDFNamingExtractor:
    """Extract naming conventions from confluence PDF using LLM"""
    
    def __init__(self, pdf_file_path, llm_client, model="llama-3.1-70b-versatile"):
        self.pdf_file_path = Path(pdf_file_path)
        self.llm_client = llm_client
        self.model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')  # ✅ From .env
        self.output_file = "naming_convention.json"
        self.pdf_processor = PDFProcessor()



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
        """Extract text content from PDF file using PDFProcessor"""
        try:
            if not self.pdf_file_path.exists():
                print(f"PDF file not found: {self.pdf_file_path}")
                return None
            
            # Use PDFProcessor for extraction
            self.pdf_processor = PDFProcessor()
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
        """
        Use LLM to extract data from all message flow tables in the PDF
        
        Returns:
            List of extracted flow configurations
        """
        # First detect tables
        flow_tables = self._detect_message_flow_tables(pdf_text)
        
        if not flow_tables:
            # Fall back to scanning whole document if no tables detected
            prompt = f"""Extract ALL message flow information from this PDF document. Look for multiple tables or sections with these exact labels and extract each set of values:

    DOCUMENT CONTENT:
    {pdf_text[:5000]}

    Find and extract ALL instances of:
    1. "ACE Application(s):" followed by the application name
    2. "Message Flow Name(s):" followed by the flow name  
    3. "Connected System(s):" followed by the system name
    4. "ACE Server:" followed by the server name
    5. "Description:" followed by the description

    Return ONLY JSON with an array of flow configurations (no markdown, no extra text):
    {{"flows": [
    {{"ace_application_name": "extracted_app_name_1", "message_flow_name": "extracted_flow_name_1", "connected_system": "extracted_system_1", "ace_server": "extracted_server_1", "description": "extracted_description_1"}},
    {{"ace_application_name": "extracted_app_name_2", "message_flow_name": "extracted_flow_name_2", "connected_system": "extracted_system_2", "ace_server": "extracted_server_2", "description": "extracted_description_2"}}
    ]}}"""

            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500
            )
            
            # Track token usage
            if 'token_tracker' in st.session_state and hasattr(response, 'usage') and response.usage:
                st.session_state.token_tracker.manual_track(
                    agent="pdf_naming_extractor",
                    operation="pdf_summary_extraction",
                    model=self.model,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    flow_name="pdf_naming_extraction"
                )
            
            llm_response = response.choices[0].message.content.strip()
            
            # Parse JSON from LLM response
            import re
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                extracted_data = json.loads(json_match.group())
                if 'flows' in extracted_data and len(extracted_data['flows']) > 0:
                    return extracted_data['flows']
            
            # If no valid JSON found, return a single default entry
            return [{
                'ace_application_name': '',
                'message_flow_name': 'Default_Flow',
                'connected_system': '',
                'ace_server': '',
                'description': ''
            }]
        
        # Process each detected table
        all_configurations = []
        for idx, table in enumerate(flow_tables):
            table_content = table['content']
            
            prompt = f"""Extract information from this message flow table. Look for these exact labels and extract the values:

    TABLE CONTENT:
    {table_content}

    Find and extract:
    1. "ACE Application(s):" followed by the application name
    2. "Message Flow Name(s):" followed by the flow name  
    3. "Connected System(s):" followed by the system name
    4. "ACE Server:" followed by the server name
    5. "Description:" followed by the description

    Return only this JSON (no markdown, no extra text):
    {{"ace_application_name": "extracted_app_name", "message_flow_name": "extracted_flow_name", "connected_system": "extracted_system", "ace_server": "extracted_server", "description": "extracted_description"}}"""

            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            # Track token usage
            if 'token_tracker' in st.session_state and hasattr(response, 'usage') and response.usage:
                st.session_state.token_tracker.manual_track(
                    agent="pdf_naming_extractor",
                    operation=f"pdf_summary_extraction_table_{idx}",
                    model=self.model,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    flow_name="pdf_naming_extraction"
                )
            
            llm_response = response.choices[0].message.content.strip()
            
            # Parse JSON from LLM response
            import re
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                extracted_data = json.loads(json_match.group())
                all_configurations.append(extracted_data)
        
        return all_configurations
            

    
    def _create_naming_convention_json(self, extracted_data_list: list) -> bool:
        """Create naming_convention.json file(s) with extracted data"""
        try:
            # If we received a single dictionary (old format), convert to list
            if isinstance(extracted_data_list, dict):
                extracted_data_list = [extracted_data_list]
                
            # Create a file for each configuration
            for idx, extracted_data in enumerate(extracted_data_list):
                # Determine filename (numbered if multiple)
                if len(extracted_data_list) > 1:
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
                        "ace_server": extracted_data.get("ace_server", "")
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