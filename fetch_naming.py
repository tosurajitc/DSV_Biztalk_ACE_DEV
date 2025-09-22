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
from dotenv import load_dotenv
load_dotenv()

class PDFNamingExtractor:
    """Extract naming conventions from confluence PDF using LLM"""
    
    def __init__(self, pdf_file_path, llm_client, model="llama-3.1-70b-versatile"):
        self.pdf_file_path = Path(pdf_file_path)
        self.llm_client = llm_client
        self.model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')  # ✅ From .env
        self.output_file = "naming_convention.json"
        
    def extract_and_create_naming_json(self) -> bool:
        """
        Main method: Extract naming from PDF and create naming_convention.json
        Returns: True if successful, False if failed
        """
        try:
            print("🔍 Extracting naming conventions from PDF...")
            
            # Step 1: Extract PDF text
            pdf_text = self._extract_pdf_text()
            if not pdf_text:
                print("❌ Failed to extract text from PDF")
                return False
            
            # Step 2: Use LLM to extract Summary table data
            extracted_data = self._llm_extract_summary_table(pdf_text)
            if not extracted_data:
                print("❌ Failed to extract summary table data using LLM")
                return False
            
            # Step 3: Create naming_convention.json
            success = self._create_naming_convention_json(extracted_data)
            if success:
                print("✅ Successfully created naming_convention.json from PDF")
                return True
            else:
                print("❌ Failed to create naming_convention.json file")
                return False
                
        except Exception as e:
            print(f"❌ PDFNamingExtractor failed: {str(e)}")
            return False
    
    def _extract_pdf_text(self) -> str:
        """Extract text content from PDF file"""
        try:
            if not self.pdf_file_path.exists():
                print(f"PDF file not found: {self.pdf_file_path}")
                return None
            
            with open(self.pdf_file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = ""
                
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"
                
                if len(text_content.strip()) < 50:
                    print("PDF appears to contain insufficient text content")
                    return None
                    
                return text_content
                
        except Exception as e:
            print(f"Error reading PDF: {str(e)}")
            return None
    
    def _llm_extract_summary_table(self, pdf_text: str) -> dict:
        """Use LLM to extract Summary table data from PDF text"""
        try:
            prompt = f"""Extract information from this PDF document. Look for a table with these exact labels and extract the values:

    DOCUMENT CONTENT:
    {pdf_text[:3000]}

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
                    agent="pdf_naming_extractor",  # ✅ Consistent name
                    operation="pdf_summary_extraction",  # ✅ Correct operation name
                    model=self.model,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    flow_name="pdf_naming_extraction"  # ✅ Specific flow name
                )
            
            llm_response = response.choices[0].message.content.strip()
            
            # Parse JSON from LLM response
            import re
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                extracted_data = json.loads(json_match.group())
            else:
                print("No JSON found in LLM response")
                return None
            
            # Validate required fields exist
            # More flexible validation - accept if we get at least one useful field
            required_fields = ['ace_application_name', 'message_flow_name']
            if any(field in extracted_data and extracted_data[field] and extracted_data[field].strip() for field in required_fields):
                return extracted_data
            else:
                print("LLM extraction missing required fields")
                return None
                
        except json.JSONDecodeError:
            print("LLM response is not valid JSON")
            return None
        except Exception as e:
            print(f"LLM extraction error: {str(e)}")
            return None
    
    def _create_naming_convention_json(self, extracted_data: dict) -> bool:
        """Create naming_convention.json file with extracted data"""
        try:
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
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(naming_convention, f, indent=2, ensure_ascii=False)
            
            print(f"📄 Created: {self.output_file}")
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
    """
    try:
        extractor = PDFNamingExtractor(pdf_file_path, llm_client)
        success = extractor.extract_and_create_naming_json()
        
        if not success:
            print("⚠️  PDF naming extraction failed - application will use existing/default naming_convention.json")
            
        return success
        
    except Exception as e:
        print(f"⚠️  PDF naming extraction module failed: {str(e)}")
        print("⚠️  Application will continue with existing/default naming_convention.json")
        return False


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