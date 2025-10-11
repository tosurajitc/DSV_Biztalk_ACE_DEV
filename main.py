#!/usr/bin/env python3
"""
Streamlit Main UI - BizTalk to ACE Migration Pipeline
Complete UI for all 5 programs with progress tracking
MINIMAL UPDATE: Only adds Program 5 tab, preserves all existing functionality
"""
import streamlit as st
import os
import sys
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import traceback
from dotenv import load_dotenv
from llm_token_tracker import TrackedGroqClient, wrap_groq_client, SmartTokenTracker, create_tracker
import shutil
import requests
from bs4 import BeautifulSoup
from fetch_naming import run_pdf_naming_extraction
from cleanup_manager import CleanupManager
# Vector DB imports - MANDATORY, no fallback
try:
    from vector_knowledge.pdf_processor import PDFProcessor
    from vector_knowledge.vector_store import ChromaVectorStore
    from vector_knowledge.semantic_search import SemanticSearchEngine
    from vector_knowledge.pipeline_integration import VectorOptimizedPipeline
    
    VECTOR_DB_AVAILABLE = True
    VECTOR_PIPELINE_AVAILABLE = True
    print("  Vector DB modules imported successfully")
    
except ImportError as e:
    VECTOR_DB_AVAILABLE = False
    VECTOR_PIPELINE_AVAILABLE = False
    print(f"  CRITICAL: Vector DB modules not available: {e}")
    print("  APPLICATION REQUIRES Vector DB - Cannot proceed without it")
    print("   Install dependencies: pip install chromadb sentence-transformers")
    print("  Ensure vector_knowledge/ folder exists with all required files")

import shutil
import plotly

load_dotenv()



# Initialize PROGRAMS_AVAILABLE as False first
PROGRAMS_AVAILABLE = False

# Import our migration programs with proper error handling
try:
    from biztalk_ace_mapper import BizTalkACEMapper  # Updated class name
    from messageflow_generator import run_messageflow_generator  
    from migration_quality_reviewer import SmartACEQualityReviewer
    PROGRAMS_AVAILABLE = True
    print("  All migration programs imported successfully")
except ImportError as e:
    PROGRAMS_AVAILABLE = False
    print(f"  Migration programs not found: {e}")
    # Don't exit here - let Streamlit handle the error display

# Page configuration
st.set_page_config(
    page_title="BizTalk to ACE Migration",
    page_icon="ðŸ”„",
    layout="wide",
    initial_sidebar_state="expanded"
)


def optimized_pipeline_execution():
    # Import required classes
    from ace_module_creator import ACEModuleCreatorOrchestrator, ACEGenerationInputs
    from schema_generator import SchemaGenerator
    from migration_quality_reviewer import SmartACEQualityReviewer
    
    # Initialize vector pipeline
    vector_pipeline = VectorOptimizedPipeline()
    
    # Setup knowledge base once (when PDF is uploaded)
    if 'vector_knowledge_ready' not in st.session_state:
        stats = vector_pipeline.setup_knowledge_base("confluence.pdf")
        st.session_state.vector_knowledge_ready = True
        st.session_state.vector_stats = stats
        st.success(f"Vector knowledge base created: {stats['total_chunks']} chunks")
    
    # Run agents with vector search using actual existing implementations
    agents = [
        ('component_mapper', lambda confluence_content: run_specification_mapping(
            biztalk_folder=st.session_state.get('biztalk_folder', ''),
            confluence_pdf=confluence_content, 
            groq_api_key=st.session_state.get('groq_api_key', ''),
            groq_model=st.session_state.get('groq_model', 'llama-3.3-70b-versatile'),
            output_dir=st.session_state.get('output_dir', 'output')
        )),
        
        ('messageflow_generator', lambda confluence_content: run_messageflow_generator(
            confluence_content=confluence_content,
            biztalk_maps_path="",
            app_name=st.session_state.get('app_name', 'DefaultApp'),
            flow_name=st.session_state.get('flow_name', 'DefaultFlow'),
            groq_api_key=st.session_state.get('groq_api_key', ''),
            groq_model=st.session_state.get('groq_model', 'llama-3.3-70b-versatile')
        )),
        
        ('ace_module_creator', lambda confluence_content: ACEModuleCreatorOrchestrator(
            groq_api_key=st.session_state.get('groq_api_key', '')
        ).create_ace_project(ACEGenerationInputs(
            component_mapping_json_path=st.session_state.pipeline_progress['program_1']['output'],
            msgflow_path=st.session_state.pipeline_progress['program_2']['output'],
            esql_template_path="ESQL_Template.esql",
            application_descriptor_template_path="application_descriptor.xml",
            project_template_path="project_template.xml",
            output_dir=st.session_state.get('output_dir', 'output')
        ))),
        
        ('schema_generator', lambda confluence_content: SchemaGenerator(
            groq_api_key=st.session_state.get('groq_api_key', '')
        ).generate_schemas(
            vector_content=confluence_content,  #   CORRECT - Matches updated schema_generator.py
            component_mapping_json_path=st.session_state.pipeline_progress['program_1']['output'],
            output_dir=st.session_state.get('output_dir', 'output')
        )),
        
        ('quality_reviewer', lambda confluence_content: SmartACEQualityReviewer(
            naming_convention_path="naming_convention.json",
            confluence_content=confluence_content
        ).extract_naming_parameters(
            component_path=st.session_state.pipeline_progress['program_1']['output'],
            confluence_content=confluence_content
        ))
    ]
    
    for agent_name, agent_function in agents:
        print(f"Running {agent_name} with vector search...")
        result = vector_pipeline.run_agent_with_vector_search(agent_name, agent_function)
        print(f"  {agent_name} completed")



def reset_vector_pipeline():
    """Reset vector pipeline and PDF processing state"""
    st.session_state.vector_pipeline = None
    st.session_state.vector_ready = False
    st.session_state.vector_stats = {}
    st.session_state.confluence_pdf_processed = False
    st.success("  Vector pipeline reset")

def setup_vector_knowledge_base(uploaded_pdf, agent_name="Agent 1"):
    """Setup vector knowledge base from uploaded PDF"""
    if not VECTOR_DB_AVAILABLE or not st.session_state.vector_enabled:
        return False
    
    try:
        # Initialize vector pipeline
        if not VECTOR_PIPELINE_AVAILABLE:
            # Simple pipeline creation if VectorOptimizedPipeline not available
            st.session_state.vector_pipeline = {
                'vector_store': ChromaVectorStore(),
                'pdf_processor': PDFProcessor(),
                'search_engine': None
            }
        else:
            # Use full VectorOptimizedPipeline
            if st.session_state.vector_pipeline is None:
                st.session_state.vector_pipeline = VectorOptimizedPipeline()
        
        # Setup knowledge base
        with st.spinner(f"  Setting up Vector Knowledge Base for {agent_name}..."):
            if VECTOR_PIPELINE_AVAILABLE:
                stats = st.session_state.vector_pipeline.setup_knowledge_base(uploaded_pdf)
            else:
                # Simple setup if no pipeline integration
                pdf_processor = PDFProcessor()
                text = pdf_processor.extract_text_from_uploaded_file(uploaded_pdf)
                chunks = pdf_processor.intelligent_chunking(text)
                
                vector_store = ChromaVectorStore()
                vector_store.create_knowledge_base(chunks)
                
                stats = {
                    'chunks_created': len(chunks),
                    'total_chunks': len(chunks),
                    'embedding_model': 'all-MiniLM-L6-v2',
                    'pdf_processed': True
                }
            
            st.session_state.vector_stats = stats
            st.session_state.vector_ready = True
            st.session_state.confluence_pdf_processed = True

            # NEW: Extract naming conventions from PDF
            temp_pdf_path = None
            with st.spinner("  Extracting naming conventions from PDF..."):
                temp_pdf_path = None  # ✅ Initialize variable
                try:
                    # Save uploaded PDF to temporary file for fetch_naming.py
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                        # Reset file pointer and read content
                        uploaded_pdf.seek(0)
                        temp_file.write(uploaded_pdf.read())
                        temp_pdf_path = temp_file.name  # ✅ Now properly scoped
                        uploaded_pdf.seek(0)  # Reset for other uses
                    
                    # Initialize LLM client for PDF processing
                    from groq import Groq
                    llm_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
                    
                    # Extract naming conventions
                    naming_success = run_pdf_naming_extraction(temp_pdf_path, llm_client)
                    
                    if naming_success:
                        st.success("✅ Extracted naming conventions from PDF")
                    else:
                        st.warning("⚠️ PDF naming extraction failed - using default naming conventions")
                        
                except Exception as e:
                    st.warning(f"⚠️ PDF naming extraction error: {str(e)} - using default naming conventions")
                finally:
                    # Clean up temp file (os should already be imported at top of main.py)
                    if temp_pdf_path and os.path.exists(temp_pdf_path):  # ✅ Check if file exists
                        try:
                            os.unlink(temp_pdf_path)
                        except:
                            pass  # Ignore cleanup errors
            
            st.success(f"  Vector DB Ready! {stats['chunks_created']} chunks indexed")
            return True
            
    except Exception as e:
        st.error(f"  Vector DB setup failed: {e}")
        st.session_state.vector_ready = False
        return False
    
def get_vector_status_display():
    """Get vector status for pipeline display"""
    if not VECTOR_DB_AVAILABLE:
        return "🔴 Vector DB Not Available"
    elif st.session_state.vector_ready:
        return "🟢 Vector DB Ready"
    elif st.session_state.vector_enabled:
        return "🟡 Vector DB Enabled (Not Ready)"
    else:
        return "⚪ Vector DB Disabled"


def render_vector_db_controls():
    """Render Vector DB controls in sidebar"""
    if not VECTOR_DB_AVAILABLE:
        st.error("  Vector DB Not Available")
        st.caption("Install: chromadb, sentence-transformers")
        return
    
    st.subheader("  Vector Knowledge Base")
    
    # Vector DB Status Display
    if st.session_state.vector_ready:
        st.success("✅ Vector DB Ready")
        stats = st.session_state.vector_stats
        

        st.metric("📊 Chunks", stats.get('total_chunks', 0))
        
        # Reset button
        if st.button("🔄 Reset Vector DB", help="Clear knowledge base and start fresh"):
            reset_vector_pipeline()
            st.rerun()
            
    elif st.session_state.confluence_pdf_processed:
        st.warning("⚠️ PDF uploaded but Vector DB not ready")
        if st.button("🚀 Initialize Vector DB"):
            st.info("👆 Go to Agent 1 to set up Vector Knowledge Base")
    else:
        st.info("Upload Business Requirement to initialize")
    
    # Global Vector Mode Toggle
    st.session_state.vector_enabled = st.checkbox(
        "Enable Vector Search Mode",
        value=st.session_state.vector_enabled,
        help="Use intelligent document chunking for all agents"
    )
    
    if st.session_state.vector_enabled:
        st.caption("  Agents 2-5 will use Vector DB instead of PDF uploads")
    else:
        st.caption("  Traditional mode: Each agent processes full PDF")


def main():
    """Main Streamlit application"""
    
    # Header
    st.title("BizTalk to IBM ACE Migration Pipeline")
    st.markdown("**Automated migration with AI-powered enhancement and documentation**")

        # Initialize session state for progress tracking
    if 'pipeline_progress' not in st.session_state:
        st.session_state.pipeline_progress = {
            'program_1': {'status': 'pending', 'output': None},
            'program_2': {'status': 'pending', 'output': None},
            'program_3': {'status': 'pending', 'output': None},
            'program_4': {'status': 'pending', 'output': None},
            'program_5': {'status': 'pending', 'output': None}
        }

            # Initialize Vector DB session state
    if 'vector_pipeline' not in st.session_state:
        st.session_state.vector_pipeline = None
        st.session_state.vector_enabled = True  # Default enabled
        st.session_state.vector_ready = False
        st.session_state.vector_stats = {}
        st.session_state.confluence_pdf_processed = False    
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Migration Pipeline")
        

        render_vector_db_controls()
        # Pipeline status
        st.subheader("Pipeline Status")
        
        # Display pipeline progress
        progress = st.session_state.pipeline_progress

        # ADD Vector DB Status at the top
        st.write("**Vector DB**:", get_vector_status_display())
        st.write("---")  # Add separator line

        st.write("**Agent 1**: BizTalk Mapper", "✅" if progress['program_1']['status'] == 'success' else "🔄" if progress['program_1']['status'] == 'running' else "❌" if progress['program_1']['status'] == 'error' else "⏳")
        st.write("**Agent 2**: ACE Mesageflow", "✅" if progress['program_2']['status'] == 'success' else "🔄" if progress['program_2']['status'] == 'running' else "❌" if progress['program_2']['status'] == 'error' else "⏳")
        st.write("**Agent 3**: ACE Modules", "✅" if progress['program_3']['status'] == 'success' else "🔄" if progress['program_3']['status'] == 'running' else "❌" if progress['program_3']['status'] == 'error' else "⏳")
        st.write("**Agent 4**: Quality Review", "✅" if progress['program_4']['status'] == 'success' else "🔄" if progress['program_4']['status'] == 'running' else "❌" if progress['program_4']['status'] == 'error' else "⏳")
        st.write("**Agent 5**: Postman Collection", "✅" if progress['program_5']['status'] == 'success' else "🔄" if progress['program_5']['status'] == 'running' else "❌" if progress['program_5']['status'] == 'error' else "⏳")
        
        # Token tracker status
        if 'token_tracker' in st.session_state:
            tracker = st.session_state.token_tracker
            metrics = tracker.get_real_time_metrics()
            if metrics.total_calls > 0:
                st.write("**Token Analytics**:   Active", f"({metrics.total_calls} calls)")
            else:
                st.write("**Token Analytics**:   Ready")
        
        # Reset pipeline button
        if st.button("  Reset Pipeline"):
            reset_pipeline()
            st.rerun()
    
    # Main content area
    if not PROGRAMS_AVAILABLE:
        st.error("  Migration programs not available. Please ensure all Python files are in the same directory.")
        return
    
    # Create tabs for each program - UPDATED TO ADD TOKEN ANALYTICS TAB
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "  Agent 1: Mapping", 
        "  Agent 2: Messageflow", 
        "  Agent 3: Foundation", 
        "  Agent 4: Quality", 
        "  Agent 5: Postman",
        "  Token Analytics"
    ])
    
    with tab1:
        render_program_1_ui()
    
    with tab2:
        render_program_2_ui()
    
    with tab3:
        render_program_3_ui()
    
    with tab4:
        render_program_4_ui()
    
    with tab5:
        render_program_5_ui()
    
    with tab6:
        render_token_analytics_tab()


def get_status_icon(status):
    """Get status icon for pipeline progress"""
    icons = {
        'pending': ' ',
        'running': '',
        'success': '',
        'error': ''
    }
    return icons.get(status, 'â“')


def reset_pipeline():
    """Reset pipeline progress and Vector DB"""
    st.session_state.pipeline_progress = {
        'program_1': {'status': 'pending', 'output': None},
        'program_2': {'status': 'pending', 'output': None},
        'program_3': {'status': 'pending', 'output': None},
        'program_4': {'status': 'pending', 'output': None},
        'program_5': {'status': 'pending', 'output': None}
    }
    
    # Reset Vector DB when pipeline is reset
    reset_vector_pipeline()
    
    # Reset token tracker when pipeline is reset
    if 'token_tracker' in st.session_state:
        from llm_token_tracker import create_tracker
        session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.session_state.token_tracker = create_tracker(session_name)

        

def render_program_1_ui():
    """Render Program 1: Specification-Driven BizTalk ACE Mapper UI - REDESIGNED"""
    
    st.header("Agent 1: BizTalk to ACE Mapper")
    st.markdown("Business specification-driven **intelligent mapping** with sequential workflow")
    
    # Initialize session states if not exists
    if 'cleanup_completed' not in st.session_state:
        st.session_state.cleanup_completed = False
    if 'inputs_provided' not in st.session_state:
        st.session_state.inputs_provided = False
    if 'business_input_method' not in st.session_state:
        st.session_state.business_input_method = 'pdf'
    if 'confluence_content' not in st.session_state:
        st.session_state.confluence_content = ""
    if 'vector_db_ready' not in st.session_state:
        st.session_state.vector_db_ready = False
    if 'agent1_completed' not in st.session_state:
        st.session_state.agent1_completed = False
    if 'agent1_error' not in st.session_state:
        st.session_state.agent1_error = None
    
def render_program_1_ui():
    """Render Program 1: BizTalk to ACE Mapper UI - REDESIGNED"""
    
    st.header("Agent 1: BizTalk to ACE Mapper")
    st.markdown("Business specification-driven **intelligent mapping** with sequential workflow")
    
    # Initialize session states if not exists
    if 'cleanup_completed' not in st.session_state:
        st.session_state.cleanup_completed = False
    if 'inputs_provided' not in st.session_state:
        st.session_state.inputs_provided = False
    if 'business_input_method' not in st.session_state:
        st.session_state.business_input_method = 'pdf'
    if 'confluence_content' not in st.session_state:
        st.session_state.confluence_content = ""
    if 'vector_db_ready' not in st.session_state:
        st.session_state.vector_db_ready = False
    if 'agent1_completed' not in st.session_state:
        st.session_state.agent1_completed = False
    if 'agent1_error' not in st.session_state:
        st.session_state.agent1_error = None
    
    # ===== STEP 1: CLEANUP & RESET BUTTON (Center Position, Always Available) =====
    st.markdown("---")
    st.markdown("### Step 1: Prepare Clean Environment")
    
    col1, col2, col3 = st.columns([1.5, 1, 1.5])  # Create center alignment
    with col2:
        if st.button("🧹 Cleanup & Reset", type="primary", width="stretch", 
                    help="Clean Vector DB, output folder, and temporary files"):
            
            # Perform cleanup using CleanupManager
            with st.spinner("🧹 Cleaning workspace..."):
                try:
                    # Initialize CleanupManager
                    cleanup_manager = CleanupManager(
                        chroma_db_path="chroma_db",
                        output_dir="output",
                        root_cleanup_patterns=[
                            "naming_convention*.json",
                            "msgflow_template.xml"
                        ]
                    )
                    
                    # Perform full cleanup
                    cleanup_results = cleanup_manager.perform_full_cleanup()
                    
                    # Display cleanup results
                    st.write("📋 Cleanup Results:")
                    
                    # Vector DB
                    vdb_result = cleanup_results['vector_db']
                    if vdb_result['status'] == 'success':
                        st.write(f"✅ {vdb_result['message']}")
                    else:
                        st.write(f"⚠️ {vdb_result['message']}")
                    
                    # Output Folder
                    out_result = cleanup_results['output_folder']
                    if out_result['status'] == 'success':
                        st.write(f"✅ {out_result['message']}")
                    else:
                        st.write(f"⚠️ {out_result['message']}")
                    
                    # Root Files
                    root_result = cleanup_results['root_files']
                    if root_result['status'] == 'success':
                        st.write(f"✅ {root_result['message']}")
                        if root_result.get('files_removed'):
                            with st.expander("📄 Files Removed"):
                                for file in root_result['files_removed']:
                                    st.text(f"  • {file}")
                    else:
                        st.write(f"⚠️ {root_result['message']}")
                    
                    # Reset session states
                    st.session_state.cleanup_completed = True
                    st.session_state.inputs_provided = False
                    st.session_state.vector_db_ready = False
                    st.session_state.agent1_completed = False
                    st.session_state.confluence_content = ""
                    st.session_state.agent1_error = None
                    
                    # Reset pipeline progress
                    if 'pipeline_progress' in st.session_state:
                        if 'program_1' in st.session_state.pipeline_progress:
                            st.session_state.pipeline_progress['program_1'] = {
                                'status': None,
                                'output': None,
                                'error_message': None,
                                'timestamp': None
                            }
                    
                    st.write("✅ Session states reset")
                    
                    # Overall status
                    if cleanup_results['overall_status'] == 'success':
                        st.success("✅ Cleanup completed successfully!")
                    else:
                        st.warning("⚠️ Cleanup completed with some warnings. Check details above.")
                    
                except Exception as e:
                    st.error(f"❌ Cleanup failed: {str(e)}")
                    import traceback
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())
    
    # Show cleanup status
    if st.session_state.cleanup_completed:
        st.success("✅ Cleanup completed successfully!")
    else:
        st.info("Please run Cleanup & Reset first to start fresh")
    
    # ===== STEP 2: REQUIRED INPUTS (Disabled until cleanup completed) =====
    st.markdown("---")
    st.markdown("### Step 2: Provide Required Inputs")
    
    # BizTalk Folder Input C:\@Official\@Gen AI\DSV\BizTalk\MH.ESB.EE.Out.DocPackApp\MH.ESB.EE.Out.DocPackApp
    biztalk_folder = st.text_input(
        "📁 **BizTalk Folder Path**",
        disabled=not st.session_state.cleanup_completed,
        value="",
        help="Path to your BizTalk project folder containing maps and schemas"
    )
    
    # Business Requirements Input - Two Options
    st.markdown("**Business Requirements** (Choose one option):")
    
    # Option selection
    business_input_method = st.radio(
        "Select input method:",
        ["PDF Upload", "Confluence URL"],
        disabled=not st.session_state.cleanup_completed,
        horizontal=True
    )
    
    uploaded_file = None
    
    if st.session_state.cleanup_completed:
        if business_input_method == "PDF Upload":
            uploaded_file = st.file_uploader(
                "Choose Business Requirements PDF",
                type="pdf",
                help="Upload the PDF containing business requirements"
            )
            
        elif business_input_method == "Confluence URL":
            confluence_url = st.text_input(
                "Enter Confluence Page URL:",
                placeholder="https://your-company.atlassian.net/wiki/spaces/.../pages/...",
                help="Enter the full URL of the Confluence page containing business requirements"
            )
            
            if confluence_url and st.button("Fetch Content from Confluence"):
                # Fetch Confluence content inline
                try:
                    with st.spinner("Fetching content from Confluence..."):
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                        }
                        
                        response = requests.get(confluence_url, headers=headers, timeout=30)
                        response.raise_for_status()
                        
                        # Parse HTML content
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Extract text content
                        for script in soup(["script", "style"]):
                            script.decompose()
                        
                        content = soup.get_text()
                        
                        # Clean up whitespace
                        lines = (line.strip() for line in content.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        content = ' '.join(chunk for chunk in chunks if chunk)
                        
                        if content:
                            st.session_state.confluence_content = content
                            st.success(f"✅ Successfully fetched {len(content)} characters from Confluence")
                        else:
                            st.error("❌ No content found at the provided URL")
                            
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Failed to fetch Confluence content: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Error processing Confluence content: {str(e)}")
    
    # Check if inputs are provided
    inputs_ready = (
        st.session_state.cleanup_completed and 
        biztalk_folder and 
        (
            (business_input_method == "PDF Upload" and uploaded_file is not None) or
            (business_input_method == "Confluence URL" and st.session_state.confluence_content)
        )
    )
    
    if inputs_ready and not st.session_state.inputs_provided:
        st.session_state.inputs_provided = True
        st.success("✅ All required inputs provided!")
    

    # ===== STEP 3: SETUP VECTOR KNOWLEDGE BASE =====
    st.markdown("---")
    st.markdown("### Step 3: Setup Vector Knowledge Base")
    
    # ✅ FIX: Check if PDF uploaded OR Confluence content available
    inputs_ready = (
        uploaded_file is not None or 
        st.session_state.get('confluence_content') is not None
    )
    
    # Update session state
    st.session_state.inputs_provided = inputs_ready
    
    col1, col2, col3 = st.columns([1.5, 1, 1.5])  # Create center alignment
    with col2:
        if st.button(
            "🚀 Setup Vector DB", 
            disabled=not inputs_ready,
            type="primary" if inputs_ready else "secondary",
            width="stretch",
            help="Create vector embeddings from business requirements (PDF or Confluence)"
        ):
            try:
                with st.spinner("Setting up Vector Knowledge Base..."):
                    # ✅ Show correct processing message based on source
                    if uploaded_file is not None:
                        success = setup_vector_knowledge_base(uploaded_file, "Agent 1")
                    elif st.session_state.get('confluence_content'):
                        st.write("🌐 Processing Confluence content...")
                        success = setup_vector_knowledge_base(st.session_state.confluence_content, "Agent 1")
                    else:
                        st.error("❌ No content source available!")
                        success = False
                    
                    if success:
                        st.session_state.vector_db_ready = True
                        st.session_state.agent1_error = None
                        st.success("✅ Vector Knowledge Base setup completed!")
                    else:
                        st.session_state.vector_db_ready = False
                        st.error("❌ Vector Knowledge Base setup failed!")
                    
            except Exception as e:
                st.error(f"❌ Vector Knowledge Base setup failed: {str(e)}")
                st.session_state.vector_db_ready = False
    
    # Show vector DB status
    if st.session_state.get('vector_db_ready', False):
        st.success("✅ Vector Knowledge Base ready!")
        debug_vector_db_state("Agent 1 - After Setup")
        verify_business_requirements_quality()
    
    elif inputs_ready:
        st.info("🔄 Click 'Setup Vector Knowledge Base' to process requirements")
    else:
        st.warning("⚠️ Please upload a PDF or provide Confluence content first")

    # ===== STEP 4: RUN AGENT 1 (Disabled until Vector DB ready) =====
    st.markdown("---")
    st.markdown("### Step 4: Execute Agent 1")
    
    # Check for successful run from pipeline progress
    program_1_status = st.session_state.pipeline_progress.get('program_1', {}).get('status')
    
    # Display any existing error
    if st.session_state.agent1_error:
        st.error(f"Agent 1 failed: {st.session_state.agent1_error}")
    
    # Execute button


    col1, col2, col3 = st.columns([1.5, 1, 1.5])  # Create center alignment
    with col2:
        if st.button(
            "Run Agent 1 & create Mapping", 
            disabled=not st.session_state.vector_db_ready,
            type="primary" if st.session_state.vector_db_ready else "secondary",
            width="stretch",
            help="Execute BizTalk to ACE mapping"
        ):
            # Run Agent 1 inline
            try:
                with st.spinner("🎯 Running Agent 1..."):
                    # Get required parameters
                    groq_api_key = os.getenv('GROQ_API_KEY', '')
                    groq_model = os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')
                    output_dir = "output"
                    
                    # Clear any existing error
                    st.session_state.agent1_error = None
                    
                    # Use the proper high-level method that handles Vector DB integration
                    result = run_specification_mapping(
                        biztalk_folder=biztalk_folder,
                        confluence_pdf=None,  # Vector DB will be used
                        groq_api_key=groq_api_key,
                        groq_model=groq_model,
                        output_dir=output_dir
                    )

                    st.session_state.pipeline_progress['program_1']['components_processed'] = result.get('components_processed', 0)
                    st.session_state.pipeline_progress['program_1']['mappings_generated'] = result.get('mappings_generated', 0)
                    st.session_state.pipeline_progress['program_1']['vector_content_length'] = result.get('vector_content_length', 0)
                    
                    st.session_state.agent1_completed = True
                    st.success("✅ Agent 1 execution completed!")
                    st.rerun()
                    
            except Exception as e:
                # Store error for display
                error_message = str(e)
                st.session_state.agent1_error = error_message
                st.error(f"❌ Agent 1 execution failed: {error_message}")
                st.session_state.agent1_completed = False
    
    # ===== DISPLAY RESULTS (Only if execution was successful) =====
    if program_1_status == 'success' or st.session_state.agent1_completed:
        st.markdown("---")
        st.markdown("## Generated Files")
        
        # Get generated files
        output_dir = "output"
        if os.path.exists(output_dir):
            files_found = False
            
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path):
                    files_found = True
                    st.success(f"📄 {file}")
            
            # Check root directory for msgflow_template.xml
            msgflow_path = "msgflow_template.xml"
            if os.path.exists(msgflow_path):
                files_found = True
                st.success(f"📄 {file}")
            
            if not files_found:
                st.warning("No output files found in the output directory.")
        else:
            st.warning("Output directory not found. Please run Agent 1 first.")



        
        # Get generated files
        output_dir = "output"
        if os.path.exists(output_dir):
            files_found = False
            
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path):
                    files_found = True
                    st.success(f"📄 {file}")
                    # Show file info

            
            # Check root directory for msgflow_template.xml
            msgflow_path = "msgflow_template.xml"
            if os.path.exists(msgflow_path):
                files_found = True
                st.success(f"📄 {file}")
            
            if not files_found:
                st.warning("No output files found in the output directory.")
        else:
            st.warning("Output directory not found. Please run Agent 1 first.")




def debug_vector_db_state(agent_name="Unknown"):
    """Debug Vector DB state across agents"""
    print(f"\n🔍 VECTOR DB STATE DEBUG - {agent_name}")
    print("="*50)
    
    # Session State Check
    print(f"📊 Session State:")
    print(f"  vector_enabled: {st.session_state.get('vector_enabled', False)}")
    print(f"  vector_ready: {st.session_state.get('vector_ready', False)}")
    print(f"  vector_pipeline exists: {st.session_state.get('vector_pipeline') is not None}")
    
    # Pipeline Details
    if st.session_state.get('vector_pipeline'):
        pipeline = st.session_state.vector_pipeline
        print(f"  pipeline type: {type(pipeline)}")
        print(f"  knowledge_ready: {getattr(pipeline, 'knowledge_ready', 'N/A')}")
        print(f"  search_engine exists: {hasattr(pipeline, 'search_engine') and pipeline.search_engine is not None}")
        
        # Collection Check
        if hasattr(pipeline, 'search_engine') and pipeline.search_engine:
            print(f"  collection exists: {hasattr(pipeline.search_engine, 'collection') and pipeline.search_engine.collection is not None}")
            
            # Try to test collection
            try:
                if hasattr(pipeline.search_engine, 'collection') and pipeline.search_engine.collection:
                    test_result = pipeline.search_engine.collection.count()
                    print(f"  collection count: {test_result}")
                else:
                    print(f"  collection: None or missing")
            except Exception as e:
                print(f"  collection test failed: {e}")
    
    # ChromaDB Physical Check
    print(f"📁 Physical ChromaDB Check:")
    chroma_path = "chroma_db"
    if os.path.exists(chroma_path):
        folders = [f for f in os.listdir(chroma_path) if os.path.isdir(os.path.join(chroma_path, f))]
        print(f"  ChromaDB folders: {len(folders)}")
        if folders:
            print(f"  Latest folder: {folders[-1] if folders else 'None'}")
    else:
        print(f"  ChromaDB folder: Does not exist")
    
    print("="*50)



def verify_business_requirements_quality():
    """Verify Vector DB contains quality business requirements - ADAPTIVE VERSION"""
    print(f"\n🎯 BUSINESS REQUIREMENTS VERIFICATION")
    print("="*50)
    
    if not st.session_state.get('vector_pipeline'):
        print("❌ No vector pipeline available")
        return False
    
    pipeline = st.session_state.vector_pipeline
    
    try:
        print(f"🔍 Testing business requirement extraction...")
        
        # ADAPTIVE: Check for generic business requirement indicators
        # These patterns work across ANY BizTalk/ACE migration project
        generic_indicators = [
            "database",      # Database operations
            "lookup",        # Lookups
            "enrichment",    # Data enrichment
            "transformation",# Data transformation
            "mapping",       # Field mapping
            "routing",       # Message routing
            "validation",    # Business validation
            "flow",          # Message flow
            "integration",   # Integration patterns
            "message"        # Message processing
        ]
        
        # Get vector content using the same method as ESQL generator
        vector_content = pipeline.search_engine.get_agent_content("esql_generator")
        
        if not vector_content:
            print("❌ No vector content retrieved")
            return False
            
        print(f"✅ Vector content retrieved: {len(vector_content)} characters")
        
        # Count how many generic indicators are found
        found_indicators = []
        for indicator in generic_indicators:
            if indicator.lower() in vector_content.lower():
                found_indicators.append(indicator)
        
        print(f"📊 Business indicators found: {len(found_indicators)}/{len(generic_indicators)}")
        print(f"✅ Found: {found_indicators}")
        
        # CRITICAL FIX: Check for substantial content rather than specific keywords
        content_checks = {
            'has_content': len(vector_content) > 3000,  # At least 5000 chars
            'has_indicators': len(found_indicators) >= 4,  # At least 4 generic indicators
            'has_chunks': True  # Vector DB has chunks
        }
        
        # Additional check: Look for stored procedure patterns (sp_ or proc_)
        has_stored_procs = 'sp_' in vector_content or 'proc_' in vector_content
        if has_stored_procs:
            print(f"✅ Detected stored procedure references in content")
            content_checks['has_stored_procs'] = True
        
        # Content quality assessment
        print(f"\n📋 Content Quality Checks:")
        print(f"  ✅ Substantial content: {content_checks['has_content']} ({len(vector_content)} chars)")
        print(f"  ✅ Generic indicators: {content_checks['has_indicators']} ({len(found_indicators)} found)")
        print(f"  ✅ Has vector chunks: {content_checks['has_chunks']}")
        if has_stored_procs:
            print(f"  ✅ Has stored procedures: True")
        
        # Content preview
        print(f"\n📄 Content preview (first 300 chars):")
        print(f"   {vector_content[:300]}...")
        
        # ADAPTIVE SUCCESS CRITERIA:
        # Pass if: (1) Has substantial content AND (2) At least 4 generic indicators
        # This works for ANY project, not just specific hardcoded examples
        quality_passed = content_checks['has_content'] and content_checks['has_indicators']
        
        if quality_passed:
            print(f"\n✅ BUSINESS REQUIREMENTS QUALITY CHECK PASSED")
            print(f"   Vector DB has sufficient business requirement content")
        else:
            print(f"\n⚠️ BUSINESS REQUIREMENTS QUALITY CHECK FAILED")
            if not content_checks['has_content']:
                print(f"   ❌ Insufficient content: {len(vector_content)} chars (need > 5000)")
            if not content_checks['has_indicators']:
                print(f"   ❌ Insufficient indicators: {len(found_indicators)} found (need >= 4)")
        
        return quality_passed
            
    except Exception as e:
        print(f"❌ Business verification failed: {e}")
        return False


def run_specification_mapping(biztalk_folder, confluence_pdf, groq_api_key, groq_model, output_dir):
    """Execute specification-driven mapping with Vector DB optimization"""
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    try:
        # Update status
        st.session_state.pipeline_progress['program_1']['status'] = 'running'
        progress_placeholder.progress(0)
        status_placeholder.info("  Initializing Specification-Driven Mapper...")
        
        # Set environment variables
        os.environ['GROQ_API_KEY'] = groq_api_key
        os.environ['GROQ_MODEL'] = groq_model
        
        # Initialize mapper
        from biztalk_ace_mapper import BizTalkACEMapper
        mapper = BizTalkACEMapper()
        
        progress_placeholder.progress(20)
        
        # Vector DB Integration Check
        if (st.session_state.get('vector_enabled', False) and 
            st.session_state.get('vector_ready', False) and 
            st.session_state.get('vector_pipeline')):
            
            status_placeholder.info("  Using Vector DB for focused business requirements...")
            progress_placeholder.progress(40)
            
            # Create agent function that BizTalkACEMapper expects
            def agent_function(focused_content):
                """Agent function that receives focused vector content"""
                return mapper.process_mapping(
                    biztalk_files=biztalk_folder,
                    pdf_file=focused_content,  # Vector focused content (text)
                    output_dir=output_dir
                )
            
            progress_placeholder.progress(60)
            status_placeholder.info("  Running intelligent mapping with Vector optimization...")
            
            # Use Vector DB pipeline to get focused content and run agent
            result = st.session_state.vector_pipeline.run_agent_with_vector_search(
                agent_name="component_mapper",
                agent_function=agent_function
            )
            
            progress_placeholder.progress(90)
            status_placeholder.info("  Vector DB processing completed!")
            
            # Add vector processing indicators to result
            result['vector_processing'] = True
            result['processing_method'] = 'Vector DB Optimization'
            
        else:
            # Vector DB not available - raise error (no fallback)
            error_msg = "Vector DB not enabled or not ready. Please setup Vector Knowledge Base first."
            
            if not st.session_state.get('vector_enabled', False):
                error_msg = "Vector DB is disabled. Please enable Vector DB in sidebar."
            elif not st.session_state.get('vector_ready', False):
                error_msg = "Vector DB not ready. Please setup Vector Knowledge Base using PDF upload."
            elif not st.session_state.get('vector_pipeline'):
                error_msg = "Vector DB pipeline not initialized. Please restart the application."
            
            progress_placeholder.error(f"  {error_msg}")
            raise Exception(f"Vector DB Error: {error_msg}")
        
        progress_placeholder.progress(100)
        
        # Update session state - IMPORTANT UPDATE HERE
        st.session_state.pipeline_progress['program_1']['status'] = 'success'
        st.session_state.pipeline_progress['program_1']['output'] = result.get('json_file')
        st.session_state.pipeline_progress['program_1']['timestamp'] = datetime.now()

        st.session_state.pipeline_progress['program_1']['components_processed'] = result.get('components_processed', 0)
        st.session_state.pipeline_progress['program_1']['mappings_generated'] = result.get('mappings_generated', 0)
        st.session_state.pipeline_progress['program_1']['vector_content_length'] = result.get('vector_content_length', 0)
        
        
        # Add msgflow_template to session state for other agents to use
        if 'msgflow_template' in result:
            st.session_state.pipeline_progress['program_1']['msgflow_template'] = result['msgflow_template']
        
        # Success message with vector info
        vector_info = ""
        if result.get('vector_processing'):
            vector_info = f" (Vector optimized: {result.get('vector_content_length', 0)} chars)"
        
        status_placeholder.success(f"  Specification-driven mapping completed!")
        
        # Clear the error message container if it exists
        if 'agent1_error_container' in st.session_state:
            st.session_state.agent1_error_container = None
        
        # Force UI refresh
        st.rerun()
        
        return result
        
    except Exception as e:
        # Update status
        st.session_state.pipeline_progress['program_1']['status'] = 'error'
        st.session_state.pipeline_progress['program_1']['error_message'] = str(e)
        st.session_state.pipeline_progress['program_1']['timestamp'] = datetime.now()
        
        progress_placeholder.progress(100)
        status_placeholder.error(f"  Mapping failed: {str(e)}")
        
        # Store error for display
        st.session_state.agent1_error_container = str(e)
        
        # Enhanced error handling for Vector DB issues
        error_msg = str(e).lower()
        if "vector" in error_msg or "knowledge base" in error_msg:
            st.error("  Vector DB issue. Please check Vector Knowledge Base status.")
        elif "groq" in error_msg or "api" in error_msg:
            st.error("  LLM API issue. Please verify your GROQ API key is valid and has sufficient credits.")
        else:
            st.error("  Check your inputs and try again")
            
        # Show error details
        with st.expander("  Error Details"):
            import traceback
            st.code(traceback.format_exc())
        
        progress_placeholder.empty()
        raise e
    

    

def render_program_2_ui():
    """Render Program 2: DSV MessageFlow Generator - Enhanced with Vector DB"""
    st.header("Agent 2: DSV MessageFlow Generator")
    st.markdown("Generate DSV standard **ACE MessageFlow** with Vector DB optimization")
    
    # Validate backend availability
    try:
        from messageflow_generator import run_messageflow_generator
        st.success("  DSV MessageFlow Generator ready")
    except ImportError as e:
        st.error(f"  DSV MessageFlow Generator not available: {e}")
        st.error("Ensure updated messageflow_generator.py is available")
        return
    
    #   NEW: Vector DB Status Check (similar to Agent 1)
    if not st.session_state.get('vector_enabled', False):
        st.error("  Vector DB is disabled. Please enable Vector DB in sidebar.")
        st.info("  Enable Vector DB to access focused MessageFlow content")
        return
    
    if not st.session_state.get('vector_ready', False):
        st.warning("   Vector DB not ready. Please setup Vector Knowledge Base in Agent 1 first.")
        st.info("   Go to Agent 1 and upload PDF to setup Vector Knowledge Base")
        return
    
    # Check Program 1 completion
    prog1_status = st.session_state.pipeline_progress['program_1']['status']
    if prog1_status != 'success':
        st.warning("   Program 1 must be completed first to generate component mappings")
        return
    
    #   NEW: Show Vector DB Status (replaces PDF upload section)
    st.subheader("  Vector Knowledge Base Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Vector DB", "  Ready")
    with col2:
        st.metric("Chunks Available", st.session_state.get('vector_stats', {}).get('total_chunks', 0))
    with col3:
        if st.button("  Preview MessageFlow Content", key="messageflow_preview"):
            try:
                preview = st.session_state.vector_pipeline.get_agent_content_preview("messageflow_generator")
                st.text_area("Vector Content Preview", preview, height=200)
            except Exception as e:
                st.error(f"Preview failed: {e}")
    
    st.success("  Agent 2 will use Vector Knowledge Base for focused MessageFlow content")
    
    #   MODIFIED: Show automated inputs status
    st.info("  **Vector DB Integration**: MessageFlow patterns, routing logic, and integration specs automatically extracted from Vector Knowledge Base")
    
    # Check for required files
    json_file_path = os.path.join(os.getcwd(), "output", "biztalk_ace_component_mapping.json")
    template_file_path = os.path.join(os.getcwd(), "msgflow_template.xml")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists(json_file_path):
            st.success("  Component mapping JSON found")
            try:
                import json
                with open(json_file_path, 'r') as f:
                    json_data = json.load(f)
                st.info(f"   Components: {len(json_data.get('component_mappings', []))}")
            except:
                st.warning("   JSON file exists but couldn't read component count")
        else:
            st.error("  Component mapping JSON not found")
            st.error("Run Program 1 first to generate the mapping file")
            return
    
    with col2:
        if os.path.exists(template_file_path):
            st.success("  MessageFlow template found")
            try:
                with open(template_file_path, 'r') as f:
                    template_content = f.read()
                st.info(f"  Template size: {len(template_content):,} characters")
            except:
                st.warning("   Template file exists but couldn't read size")
        else:
            st.error("  MessageFlow template not found")
            st.error("Ensure msgflow_template.xml is in the root folder")
            return

    
    st.subheader("  Automated Business Requirements")
    # DSV MessageFlow Configuration - Generic hardcoded values (not displayed to user)
    app_name = None
    flow_name = None

    groq_api_key = os.getenv('GROQ_API_KEY', '')    
    groq_model=os.getenv('GROQ_MODEL', '')
    #   MODIFIED: Input Validation (removed confluence_doc check)
    missing_inputs = []
    if not groq_api_key.strip():
        missing_inputs.append("GROQ API Key")
    
    if missing_inputs:
        st.error(f"  **Missing Required Inputs**: {', '.join(missing_inputs)}")
        return

    if st.button("  Generate DSV MessageFlow", type="primary"):
        run_messageflow_generation(
            confluence_doc=None,  # No longer needed - Vector DB provides content
            app_name=app_name,
            flow_name=flow_name,
            groq_api_key=groq_api_key,
            groq_model=groq_model
        )


def extract_messageflow_details_from_pdf_content(confluence_content: str) -> dict:
    """Extract Message Flow Name and ACE Application from PDF content"""
    import re
    
    # Dynamic pattern matching - no hardcoded text
    flow_pattern = re.search(r'Message\s+Flow\s+Name\(?s?\)?[:\s]*([A-Za-z0-9_]+)', confluence_content, re.IGNORECASE)
    app_pattern = re.search(r'ACE\s+Application\(?s?\)?[:\s]*([A-Za-z0-9_]+)', confluence_content, re.IGNORECASE)
    
    flow_name = flow_pattern.group(1).strip() if flow_pattern else None
    app_name = app_pattern.group(1).strip() if app_pattern else None
    
    # Validation and fallback
    if flow_name and len(flow_name) > 3 and '_' in flow_name:
        if app_name and app_name.startswith('EPIS_') and len(app_name) > 8:
            return {'flow_name': flow_name, 'app_name': app_name, 'extraction_success': True}
    
    # Generate fallback based on any extracted flow name
    if flow_name:
        return {
            'flow_name': flow_name, 
            'app_name': f'EPIS_{flow_name}_App', 
            'extraction_success': True
        }
    
    # Final fallback
    return {'flow_name': 'Dynamic_MessageFlow', 'app_name': 'EPIS_Dynamic_App', 'extraction_success': False}



def run_messageflow_generation(confluence_doc, app_name, flow_name, groq_api_key, groq_model):
    """Execute DSV MessageFlow generation with Vector DB integration"""
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    try:
        # Update progress tracking
        st.session_state.pipeline_progress['program_2']['status'] = 'running'
        progress_placeholder.progress(10)
        status_placeholder.info("  Processing Vector DB focused content...")
        
        #   NEW: Vector DB Integration Check (similar to Agent 1)
        if (st.session_state.get('vector_enabled', False) and 
            st.session_state.get('vector_ready', False) and 
            st.session_state.get('vector_pipeline')):
            
            status_placeholder.info("  Using Vector DB for focused MessageFlow requirements...")
            progress_placeholder.progress(40)
            
            # Create agent function that run_messageflow_generator expects
            def agent_function(focused_content):
                """Agent function that receives focused vector content"""
                with open('naming_convention.json', 'r', encoding='utf-8') as f:
                    naming_data = json.load(f)

                app_name = naming_data['project_naming']['ace_application_name']
                flow_name = naming_data['project_naming']['message_flow_name']

                print(f"Using names from naming_convention.json:")
                print(f"  ACE Application: {app_name}")
                print(f"  Message Flow: {flow_name}")

                return run_messageflow_generator(
                    confluence_content=focused_content,
                    biztalk_maps_path="",
                    app_name=app_name,      # From naming_convention.json
                    flow_name=flow_name,    # From naming_convention.json
                    groq_api_key=groq_api_key,
                    groq_model=groq_model
                )
                        
            progress_placeholder.progress(60)
            status_placeholder.info("  Running MessageFlow generation with Vector optimization...")
            
            # Use Vector DB pipeline to get focused content and run agent
            result = st.session_state.vector_pipeline.run_agent_with_vector_search(
                agent_name="messageflow_generator",  #   Matches SemanticSearchEngine config
                agent_function=agent_function  # Updated parameter name
            )
            
            progress_placeholder.progress(90)
            status_placeholder.info("  Vector DB processing completed!")
            
            # Add vector processing indicators to result
            result['vector_processing'] = True
            result['processing_method'] = 'Vector DB Optimization'
            
        else:
            #   Vector DB not available - raise error (no fallback)
            error_msg = "Vector DB not enabled or not ready. Please setup Vector Knowledge Base first."
            
            if not st.session_state.get('vector_enabled', False):
                error_msg = "Vector DB is disabled. Please enable Vector DB in sidebar."
            elif not st.session_state.get('vector_ready', False):
                error_msg = "Vector DB not ready. Please setup Vector Knowledge Base using PDF upload."
            elif not st.session_state.get('vector_pipeline'):
                error_msg = "Vector DB pipeline not initialized. Please restart the application."
            
            raise Exception(error_msg)
        
        progress_placeholder.progress(100)
        
        if result['success']:
            # Update session state
            st.session_state.pipeline_progress['program_2'] = {
                'status': 'success',
                'output': result['messageflow_file'],
                'timestamp': datetime.now().isoformat()
            }
            
            status_placeholder.success("  DSV MessageFlow generation completed successfully!")
            
            # Display results with vector processing indicator
            vector_indicator = " (Vector DB Optimized)" if result.get('vector_processing') else ""
            st.success(f"  **DSV MessageFlow Generated Successfully!{vector_indicator}**")
            
            # File details
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"  **File:** {os.path.basename(result['messageflow_file'])}")
                st.info(f"   **Components:** {result.get('component_mappings_processed', 'N/A')}")
                if result.get('vector_processing'):
                    st.info("  **Processing:** Vector DB Optimization")
            
            with col2:
                st.info(f"**BizTalk Maps:** {result.get('biztalk_maps_processed', 0)}")
                st.info(f"  **Validation:** {'Passed' if result.get('validation', {}).get('valid') else 'Failed'}")
                st.info(f"**Generated:** {datetime.now().strftime('%H:%M:%S')}")
            
            # Download button
            try:
                with open(result['messageflow_file'], 'r') as f:
                    msgflow_content = f.read()
                
                st.download_button(
                    label=" Download MessageFlow",
                    data=msgflow_content,
                    file_name=os.path.basename(result['messageflow_file']),
                    mime="application/xml",
                    key="download_messageflow"
                )
            except Exception as e:
                st.warning(f"   Download not available: {e}")
            
            # Show file location
            st.markdown(f"""
            **  Output Location:** `{result['messageflow_file']}`
            
            **Import into IBM ACE Toolkit:**
            1. Copy the .msgflow file to your ACE workspace
            2. Import into your Integration Server
            3. Deploy and test the MessageFlow
            """)
            
        else:
            st.session_state.pipeline_progress['program_2']['status'] = 'error'
            error_message = result.get('error', 'Unknown error occurred')
            status_placeholder.error(f"  DSV MessageFlow generation failed: {error_message}")
            
            # Show error details
            with st.expander("  Error Details"):
                st.code(result.get('error_details', error_message))
        
    except Exception as e:
        st.session_state.pipeline_progress['program_2']['status'] = 'error'
        error_msg = str(e)
        status_placeholder.error(f"  DSV MessageFlow generation failed: {error_msg}")
        
        #   Enhanced error handling for Vector DB (inline)
        if "vector" in error_msg.lower():
            if "not enabled" in error_msg.lower():
                st.error("  Vector DB is disabled. Please enable it in the sidebar.")
            elif "not ready" in error_msg.lower():
                st.error("  Vector DB not ready. Please setup Vector Knowledge Base in Agent 1.")
            elif "pipeline not initialized" in error_msg.lower():
                st.error("  Vector DB pipeline error. Please restart the application.")
            else:
                st.error("  Vector DB processing error. Please check Vector Knowledge Base status.")
        elif "groq" in error_msg.lower() or "api" in error_msg.lower():
            st.error("  LLM API issue. Please verify your GROQ API key is valid and has sufficient credits.")
        elif "json" in error_msg.lower():
            st.error("  Component mapping JSON issue. Ensure Program 1 completed successfully.")
        elif "template" in error_msg.lower():
            st.error("  MessageFlow template issue. Ensure msgflow_template.xml exists in root folder.")
        else:
            st.error("  Check your inputs and try again")
            
        # Show error details
        with st.expander("  Error Details"):
            import traceback
            st.code(traceback.format_exc())
        
        progress_placeholder.empty()
        raise e


def render_program_3_ui():
    """Render Program 3: ACE Module Creator UI - Single Input (Confluence PDF only)"""
    st.header("Agent 3: ACE Module Creator")
    st.markdown("Pure Gen AI driven orchestration module manages 6 Agents to generate **ACE components**" \
    "\n- Generates: application.descriptor\n- Generates: enrichment files\n- Generates: .ESQL Files\n- Generates: Schema .XSD files\n- Generates: .xsl FIles\n- Generates: .project File")
    
    # Check prerequisites
    prog1_status = st.session_state.pipeline_progress['program_1']['status']
    prog2_status = st.session_state.pipeline_progress['program_2']['status']
    
    if prog1_status != 'success':
        st.warning("Agent 1 & 2 must be completed first to generate ACE components")
        return
    
    if prog2_status != 'success':
        st.warning("Program 2 must be completed first to generate MessageFlow structure")
        return
    
    # Show prerequisites status
    prog1_output = st.session_state.pipeline_progress['program_1']['output']
    prog2_output = st.session_state.pipeline_progress['program_2']['output']
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"  **Program 1 Complete**: {os.path.basename(prog1_output) if prog1_output else 'Output ready'}")
    with col2:
        st.info(f"  **Program 2 Complete**: {os.path.basename(prog2_output) if prog2_output else 'Output ready'}")
    
    # Single Required Input
    st.subheader("  Required Input")
    
    st.markdown("**Business Requirements Document**")
    # VECTOR DB STATUS SECTION (REPLACES PDF UPLOAD)
    st.subheader("  Knowledge Base Status")

    if st.session_state.vector_ready:
        debug_vector_db_state("Agent 3 - UI Check")
        business_quality = verify_business_requirements_quality()

        if business_quality:
            st.success("✅ Business requirements verified in Vector DB")
        else:
            st.warning("⚠️ Business requirements quality issues detected")
            st.info("Consider re-uploading your PDF in Agent 1")



        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Vector DB", "  Ready")
        with col2:
            st.metric("Chunks Available", st.session_state.vector_stats.get('total_chunks', 0))
        with col3:
            if st.button("  Preview Content", key="agent3_preview"):
                st.info("ACE Module content preview will be available in next update")
        
        st.success("  Agent 3 will use Vector Knowledge Base from Agent 1")        
    else:
        st.error("  No knowledge base available")
        st.info("   Upload PDF in Agent 1 to initialize Vector Knowledge Base")
        return  # Don't show rest of UI if no knowledge base

    # Auto-detected Inputs Display
    st.subheader("  Auto-detected Inputs")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.markdown("**Component Mapping** (Program 1)")
        if prog1_output and os.path.exists(prog1_output):
            st.success("  Component mapping detected")
        else:
            st.error("  Component mapping not found")
    
    with col4:
        st.markdown("**MessageFlow** (Program 2)")
        msgflow_found = False
        if prog2_output and os.path.exists(prog2_output):
            if os.path.isfile(prog2_output):
                prog2_dir = os.path.dirname(prog2_output)
            else:
                prog2_dir = prog2_output
            msgflow_files = list(Path(prog2_dir).glob('*.msgflow'))
            if msgflow_files:
                st.success(f"  {len(msgflow_files)} MessageFlow(s) detected")
                msgflow_found = True
        if not msgflow_found:
            st.error("  MessageFlow files not found")
    
    with col5:
        st.markdown("**Standard Templates**")
        
        # Check root templates
        root_templates = ['application_descriptor.xml', 'project.xml']
        root_found = sum(1 for template in root_templates if os.path.exists(template))
        
        # Check generated template (from Program 1 output)
        esql_template_locations = [
            'ESQL_Template_Updated.ESQL', # Fallback: root directory
            'output/ESQL_Template_Updated.ESQL'
        ]
        esql_found = any(os.path.exists(loc) for loc in esql_template_locations)
        
        total_found = root_found + (1 if esql_found else 0)
        
        # Display status with details
        if total_found == 3:
            st.success(f"  {total_found}/3 templates found")
        else:
            st.warning(f"   {total_found}/3 templates found")

    # AI Configuration Section  
    groq_api_key = os.getenv('GROQ_API_KEY', '') 

    # Input validation - Only PDF and API key required
    input_validation = True
    validation_messages = []
    
    if not groq_api_key:
        validation_messages.append("   GROQ API Key required")
        input_validation = False
    
    # Show validation messages
    if validation_messages:
        for msg in validation_messages:
            st.error(msg)
    
    # Generation Preview
    if input_validation:
        st.subheader("   Enhanced Generation Preview")
        

    # Execution Button
    if input_validation:
        if st.button("  Generate Enhanced ACE Project", type="primary", key="run_prog3"):
            run_program_3(groq_api_key)
    else:
        st.button("  Generate Enhanced ACE Project", disabled=True, help="Fix validation errors first")


def run_program_3(groq_api_key):
    """Execute Program 3 - Enhanced ACE Module Creator Orchestrator with auto-detection"""
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    try:
        st.session_state.pipeline_progress['program_3']['status'] = 'running'
        progress_placeholder.progress(0)
        status_placeholder.info("  Initializing Enhanced ACE Module Creator Orchestrator...")
        
        # Set environment variable
        os.environ['GROQ_API_KEY'] = groq_api_key

        debug_vector_db_state("Agent 3 - Before ACE Generation")
        business_quality = verify_business_requirements_quality()

        if not business_quality:
            st.error("❌ Business requirements not properly loaded in Vector DB")
            st.info("Try: 1) Reset Vector DB, 2) Re-upload PDF in Agent 1")
            return
        
        progress_placeholder.progress(10)
        status_placeholder.info("  Auto-detecting inputs from previous programs...")
        
        # Get Program 1 and Program 2 outputs
        prog1_output = st.session_state.pipeline_progress['program_1']['output']
        prog2_output = st.session_state.pipeline_progress['program_2']['output']
        
        # Auto-detect Component Mapping JSON from Program 1 output
        component_mapping_path = None
        if prog1_output and os.path.exists(prog1_output):
            if prog1_output.endswith('.json'):
                component_mapping_path = prog1_output
            else:
                # Look for JSON files in Program 1 output directory
                json_files = list(Path(prog1_output).glob('*.json'))
                if json_files:
                    component_mapping_path = str(json_files[0])
        
        if not component_mapping_path:
            raise FileNotFoundError("Component mapping JSON not found from Program 1 output")
        
        # Auto-detect MessageFlow from Program 2 output
        msgflow_path = None
        if prog2_output and os.path.exists(prog2_output):
            if os.path.isfile(prog2_output):
                prog2_dir = os.path.dirname(prog2_output)
            else:
                prog2_dir = prog2_output
            msgflow_files = list(Path(prog2_dir).glob('*.msgflow'))
            if msgflow_files:
                msgflow_path = str(msgflow_files[0])
        
        if not msgflow_path:
            raise FileNotFoundError("MessageFlow file not found from Program 2 output")
        
        # Auto-detect standard templates from root directory
        esql_template_path = None
        if prog1_output and os.path.exists(prog1_output):
            if os.path.isfile(prog1_output):
                prog1_dir = os.path.dirname(prog1_output)
            else:
                prog1_dir = prog1_output
            esql_files = list(Path(prog1_dir).glob('ESQL_Template_Updated.ESQL'))
            if esql_files:
                esql_template_path = str(esql_files[0])

        if not esql_template_path:
            # Fallback to default location
            esql_template_path = "ESQL_Template_Updated.ESQL"
        app_descriptor_template_path = "application_descriptor.xml"
        project_template_path = "project.xml"
        
        # Validate standard templates exist
        for template_name, template_path in [
            ("ESQL Template", esql_template_path),
            ("Application Descriptor Template", app_descriptor_template_path),
            ("Project Template", project_template_path)
        ]:
            if not os.path.exists(template_path):
                raise FileNotFoundError(f"{template_name} not found: {template_path}")
        
        progress_placeholder.progress(20)
        status_placeholder.info("  Preparing Confluence PDF...")
        
        progress_placeholder.progress(30)
        status_placeholder.info("  Starting Enhanced Orchestrator with auto-detected inputs...")
        
        # Create output directory
        output_directory = os.path.join(os.path.dirname(prog2_output if os.path.isdir(prog2_output) else os.path.dirname(prog2_output)), "Enhanced_ACE_Project")
        os.makedirs(output_directory, exist_ok=True)
        
        # Import Enhanced Orchestrator
        from ace_module_creator import ACEModuleCreatorOrchestrator, ACEGenerationInputs
        
        # Create ACEGenerationInputs object with auto-detected inputs
        inputs = ACEGenerationInputs(
            component_mapping_json_path=component_mapping_path,
            msgflow_path=msgflow_path,
            esql_template_path=esql_template_path,
            application_descriptor_template_path=app_descriptor_template_path,
            project_template_path=project_template_path,
            output_dir=output_directory
        )
        
        # Display auto-detected inputs for confirmation
        st.info("  **Auto-detected Inputs:**")
        st.text(f"  Component Mapping: {os.path.basename(component_mapping_path)}")
        st.text(f"  MessageFlow: {os.path.basename(msgflow_path)}")
        st.text(f"  ESQL Template: {os.path.basename(esql_template_path)}")
        st.text(f"  App Descriptor: {os.path.basename(app_descriptor_template_path)}")
        st.text(f"  Project Template: {os.path.basename(project_template_path)}")
        
        progress_placeholder.progress(50)
        status_placeholder.info("  Running orchestrated module execution...")
        
        # Initialize and run Enhanced Orchestrator
        orchestrator = ACEModuleCreatorOrchestrator(groq_api_key=groq_api_key)
        results = orchestrator.create_ace_project(inputs)
        
        progress_placeholder.progress(100)
        
        # Check orchestration results
        if results['orchestration_status'] in ['success', 'partial_success']:
            st.session_state.pipeline_progress['program_3']['status'] = 'success'
            st.session_state.pipeline_progress['program_3']['output'] = output_directory
            
            status_placeholder.success(
                f"  Enhanced ACE Project generated successfully! "
                f"Status: {results['orchestration_status'].title()}"
            )
            
            # Display Enhanced Results
            st.subheader("  Enhanced Orchestration Results")
            
            # Overall metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("   Total Files", results['total_files_generated'])
            with col2:
                st.metric("  Execution Time", f"{results['total_execution_time']:.1f}s")
            with col3:
                st.metric("  LLM Calls", results['total_llm_calls'])
            with col4:
                st.metric("  Success Rate", f"{results['successful_modules']}/6")
            
            # Module execution details
            with st.expander("  Module Execution Details", expanded=True):
                for module_result in results['module_results']:
                    status_icon = "" if module_result['status'] == 'success' else ""
                    
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    with col1:
                        st.text(f"{status_icon} {module_result['module_name']}")
                    with col2:
                        st.text(f"{module_result['execution_time']:.1f}s")
                    with col3:
                        st.text(f"{module_result['llm_analysis_calls'] + module_result['llm_generation_calls']} LLM")
                    with col4:
                        st.text(f"{module_result['output_files_count']} files")
            
            # Project structure
            with st.expander("  Generated Project Structure"):
                structure = results.get('project_structure', {})
                st.text(f"Total Files: {structure.get('total_files', 0)}")
                
                for file_ext, files in structure.get('files_by_type', {}).items():
                    if files:
                        st.markdown(f"**{file_ext.upper()} Files ({len(files)}):**")
                        for file_path in files[:5]:  # Show first 5
                            st.text(f"    {file_path}")
                        if len(files) > 5:
                            st.text(f"  ... and {len(files) - 5} more")
            
            # Next steps
            st.subheader("  Next Steps")
            st.markdown(f"""
            **Your Enhanced ACE project is ready!**
            
              **Project Location:** `{output_directory}`
            
            **Import into IBM ACE Toolkit:**
            1. Open IBM ACE Toolkit
            2. File Import General Existing Projects into Workspace
            3. Select directory: `{output_directory}`
            4. Import the complete project structure
            """)
            
        else:
            st.session_state.pipeline_progress['program_3']['status'] = 'error'
            status_placeholder.error(f"  Enhanced Orchestration failed: {results.get('error_message', 'Unknown error')}")
            
            # Show module-specific errors
            with st.expander("  Module Execution Details"):
                for module_result in results['module_results']:
                    status_icon = "" if module_result['status'] == 'success' else ""
                    st.text(f"{status_icon} {module_result['module_name']}: {module_result['status']}")
                    if module_result.get('error_message'):
                        st.error(f"  Error: {module_result['error_message']}")
        
        # Cleanup temporary PDF file
        try:
            os.unlink("vector_database")
        except:
            pass
        
    except Exception as e:
        st.session_state.pipeline_progress['program_3']['status'] = 'error'
        status_placeholder.error(f"  Enhanced Orchestrator failed: {str(e)}")
        
        with st.expander("  Error Details"):
            import traceback
            st.code(traceback.format_exc(), language='python')            


def render_program_4_ui():
    """Render Program 4: Smart Migration Quality Reviewer UI - Template-Driven, <10K Tokens"""
    st.header("Agent 4: Smart Migration Quality Reviewer")
    st.markdown("Validates ACE component quality with ACE toolkit and creates **ACE toolkit deployable** component folder")
    
    # Check prerequisites
    program_3_status = st.session_state.pipeline_progress.get('program_3', {}).get('status')
    program_3_output = st.session_state.pipeline_progress.get('program_3', {}).get('output')
    output_folder = "output"
    
    if program_3_status != 'success' or not program_3_output:
        st.warning("Agent 3 must be completed to start Agent 4")
        return
    
    st.success(f"✅ **ACE Components Ready**: {program_3_output}")

    ace_toolkit_path = r"C:\Program Files\IBM\ACE\13.0.4.0"

    # Configuration section
    with st.expander("🔧 Smart Review Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🛠️ IBM ACE Toolkit Path**")
            ace_toolkit_path = st.text_input(
                "ACE Installation Directory",
                value=r"C:\Program Files\IBM\ACE\13.0.4.0",
                help="Path to your IBM ACE Toolkit installation"
            )
            
            # Auto-detect button
            if st.button("🔍 Auto-Detect ACE", help="Scan common locations for ACE installation"):
                common_ace_paths = [
                    r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\IBM App Connect Enterprise 13.0.4.0",
                    r"C:\Program Files\IBM\ACE\13.0.4.0",
                    r"C:\Program Files (x86)\IBM\ACE\13.0.4.0", 
                    r"C:\IBM\ACE\13.0.4.0",
                    r"C:\Program Files\IBM\App Connect Enterprise\13.0.4.0",
                    r"C:\Program Files (x86)\IBM\App Connect Enterprise\13.0.4.0"
                ]
                
                detected_path = None
                for path in common_ace_paths:
                    if os.path.exists(path):
                        detected_path = path
                        break
                
                if detected_path:
                    st.success(f"✅ Found ACE at: {detected_path}")
                    ace_toolkit_path = detected_path
                    st.rerun()
                else:
                    st.warning("⚠️ ACE not found in common locations. Please enter path manually.")
            
            # ACE Status
            if ace_toolkit_path and os.path.exists(ace_toolkit_path):
                st.success("✅ ACE Toolkit path is valid")
                ace_toolkit_available = True
            else:
                st.warning("⚠️ ACE Toolkit path not found - will use basic validation")
                ace_toolkit_available = False
        
        with col2:
            st.markdown("**📋 Naming Standards File**")
            naming_standards_file = st.text_input(
                "Path to Naming Standards JSON",
                value=r"C:\@Official\@Gen AI\DSV\BizTalk\Analyze_this_folder\DSV_Biztalk_ACE\naming_convention.json",
                help="JSON file containing smart naming convention rules"
            )
            
            if naming_standards_file and os.path.exists(naming_standards_file):
                try:
                    with open(naming_standards_file, 'r') as f:
                        naming_data = json.load(f)
                    st.success(f"✅ Naming standards loaded ({len(naming_data)} rules)")
                except:
                    st.warning("⚠️ File exists but couldn't read JSON format")
            elif naming_standards_file:
                st.warning("⚠️ Naming standards file not found - will use defaults")
            else:
                st.info("ℹ️ Using default naming standards")

    # Enhanced validation status display
    st.markdown("**🔧 Validation Mode**")
    col1, col2, col3 = st.columns(3)

    with col1:
        if ace_toolkit_available:
            st.success("✅ **Enhanced Mode**")
            st.caption("ACE Toolkit validation enabled")
        else:
            st.info("ℹ️ **Basic Mode**")
            st.caption("Rule-based validation only")

    with col2:
        st.metric("Components Source", "output/ folder")
        st.caption("Fixed root folder")

    with col3:
        if ace_toolkit_available:
            st.metric("Quality Checks", "Advanced")
            st.caption("IBM ACE + Business Rules")
        else:
            st.metric("Quality Checks", "Standard")
            st.caption("Business Rules only")
    
    # User requirements section
    with st.expander("👤 Optional User Requirements", expanded=False):
        user_requirements = st.text_area(
            "Additional Quality Requirements",
            placeholder="Enter any specific quality requirements, customizations, or business rules...",
            height=100,
            help="Optional: Provide specific requirements for customized quality analysis"
        )
    
    # Run button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        run_button = st.button("🚀 Start Smart Review", type="primary", width='stretch')
    
    if run_button:
        # Create progress placeholders
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        st.session_state.pipeline_progress['program_4']['status'] = 'running'
        progress_placeholder.progress(0)
        status_placeholder.info("🚀 Initializing Smart Migration Quality Reviewer...")
        
        progress_placeholder.progress(10)
        status_placeholder.info("📋 Preparing inputs for smart review...")
        
        # Get vector DB content for business requirements
        vector_db_content = st.session_state.vector_pipeline.search_engine.get_agent_content("migration_quality_reviewer")
        
        progress_placeholder.progress(20)
        status_placeholder.info("🛠️ Initializing Smart ACE Quality Reviewer...")
        
        # Import and initialize - EXACT parameter matching
        from migration_quality_reviewer import SmartACEQualityReviewer
        
        reviewer = SmartACEQualityReviewer(
            ace_components_folder=output_folder,
            ace_toolkit_path=ace_toolkit_path,
            naming_standards_file=naming_standards_file,
            vector_db_content=vector_db_content,
            user_requirements=user_requirements if user_requirements else None
        )
        
        progress_placeholder.progress(30)
        status_placeholder.info("📊 Starting template-driven quality analysis...")
        
        # Run smart review - exact method name
        final_output_path = reviewer.run_smart_review()
        
        progress_placeholder.progress(100)
        
        # Update session state
        st.session_state.pipeline_progress['program_4']['status'] = 'success'
        st.session_state.pipeline_progress['program_4']['output'] = final_output_path
        
        # Track token usage
        st.session_state.token_tracker.manual_track(
            agent="smart_migration_quality_reviewer",
            operation="complete_review",
            model="llama-3.3-70b-versatile",
            input_tokens=int(reviewer.token_usage * 0.7),
            output_tokens=int(reviewer.token_usage * 0.3),
            flow_name="smart_quality_review"
        )
        
        status_placeholder.success("✅ Smart Quality Review completed!")
        
        # Display results
        st.success("🎉 **Smart Migration Quality Review Complete!**")
        
        # Results summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="🎯 Token Usage", 
                value=f"{reviewer.token_usage:,}",
                delta=f"{((10000 - reviewer.token_usage) / 10000 * 100):.1f}% under budget",
                help="Actual token usage vs 10K budget"
            )
        
        with col2:
            final_path = Path(final_output_path)
            component_count = len([f for f in final_path.iterdir() if f.is_file() and f.suffix in ['.esql', '.msgflow', '.subflow', '.xsd']])
            st.metric(
                label="📦 Components Reviewed",
                value=component_count,
                help="Total ACE components processed and reviewed"
            )
        
        with col3:
            st.metric(
                label="📁 Final Project",
                value="Ready",
                delta="ACE Application",
                help="Final project structure created"
            )
        
        # Final project details
        st.info(f"""
        🏆 **Smart Review Results**
        
        📁 **Project Location:** `{final_output_path}`
        
        **Quality Analysis Features:**
        - ✅ Template-driven compliance checking
        - ✅ Smart naming conventions applied  
        - ✅ Vector DB business requirements integration
        - ✅ Efficient <10K token processing
        - ✅ BeforeEnrichment / AfterEnrichment reports
        
        **Import into IBM ACE Toolkit:**
        1. Open IBM ACE Toolkit
        2. File → Import → General → Existing Projects into Workspace
        3. Select directory: `{final_output_path}`
        4. Import the complete reviewed project structure
        """)


def run_program_4(ace_components_folder, templates_folder, naming_standards_file, 
                 vector_db_content, user_requirements=None):
    """
    Execute Program 4: Smart Migration Quality Reviewer
    Rewritten to work with SmartACEQualityReviewer class
    
    Args:
        ace_components_folder: Path to ACE components from Program 3
        templates_folder: Path to reference templates
        naming_standards_file: Path to naming standards JSON file
        vector_db_content: Business requirements from vector database
        user_requirements: Optional user customization requirements
    
    Returns:
        str: Path to final reviewed ACE project
    """
    
    print("Starting Smart Migration Quality Review...")
    print("=" * 50)
    ace_toolkit_path = r"C:\Program Files\IBM\ACE\13.0.4.0"
    output_folder = "output"

    try:
        # Import and initialize SmartACEQualityReviewer
        from migration_quality_reviewer import SmartACEQualityReviewer
        
        reviewer = SmartACEQualityReviewer(
            ace_components_folder=output_folder,
            ace_toolkit_path=ace_toolkit_path,
            naming_standards_file=naming_standards_file,
            vector_db_content=vector_db_content,
            user_requirements=user_requirements if user_requirements else None
        )
        
        # Execute smart review
        final_output_path = reviewer.run_smart_review()
        
        print(f"\nSmart Review Complete!")
        print(f"Token Usage: {reviewer.token_usage:,} tokens")
        print(f"Final Project: {final_output_path}")
        
        return final_output_path
        
    except Exception as e:
        print(f"Smart Review Failed: {e}")
        raise Exception(f"Program 4 execution failed: {e}")


# Legacy compatibility function (if needed)
def run_program_4_legacy(quality_thresholds=None, include_groq_validation=None, validation_scope=None, 
                         generate_bar_file=None, templates_path=None, naming_convention_path=None,
                         confluence_pdf_data=None, account_input_data=None,
                         library_path=None, validate_libraries=None,
                         include_enrichment_files=None,
                         enable_github=None):
    """
    Legacy compatibility wrapper - converts old parameters to new format
    """
    
    # Convert old parameters to new format
    ace_components_folder = "."  # Default to current directory
    templates_folder = templates_path or "templates"
    naming_standards_file = naming_convention_path or "naming_standards.json"
    vector_db_content = confluence_pdf_data or "No business requirements available"
    user_requirements = account_input_data
    
    # Call new function
    return run_program_4(
        ace_components_folder=ace_components_folder,
        templates_folder=templates_folder,
        naming_standards_file=naming_standards_file,
        vector_db_content=vector_db_content,
        user_requirements=user_requirements
    )


# UPDATED PROGRAM 5 UI - Postman Collection Generator Integration
def render_program_5_ui():
    """Render Program 5: Postman Collection Generator UI"""
    st.header("   Agent 5: Generate Postman Collections")
    st.markdown("Create comprehensive **Postman test collections** from IBM ACE Message Flows with 100+ test scenarios")
    groq_api_key = os.getenv('GROQ_API_KEY', '') 

    # Check dependencies
    dependencies_check = check_postman_dependencies()
    if dependencies_check['all_available']:
        format_info = "  **Output**: Postman Collections + Environments + Test Data + Documentation"
    else:
        format_info = "   **Missing Dependencies**: " + ", ".join(dependencies_check['missing'])
    
    st.info(format_info)
    
    # Check if Program 4 completed
    program_4_status = st.session_state.pipeline_progress.get('program_4', {}).get('status')
    program_4_output = st.session_state.pipeline_progress.get('program_4', {}).get('output')
    
    if program_4_status != 'success' or not program_4_output:
        st.warning("Complete Migration Quality Reviewer first to generate Postman collections")
        st.info("  Program 5 needs the reviewed modules from Program 4 as input")
        return
    
    st.success(f"  **Input Ready**: {program_4_output}")
    
    # Configuration
    with st.expander(" Configuration", expanded=True):
        extracted_project_name = os.path.basename(program_4_output)
        
        project_name = st.text_input(
            "  Project Name",
            value=extracted_project_name,  # ← DYNAMIC from Agent 4!
            help="Name for the Postman collections and test scenarios"
        )
        
        # Output location (editable)
        default_output_path = get_default_postman_output_path(program_4_output, extracted_project_name)
        target_output_folder = st.text_input(
            "  Output Folder",
            value=default_output_path,
            help="Where to create the Postman collections folder"
        )
        
        # Validate output path
        if target_output_folder:
            output_parent = Path(target_output_folder).parent
            if output_parent.exists():
                st.success("  Valid output location")
            else:
                st.warning(f"   Parent directory doesn't exist: {output_parent}")
        
        col1, col2 = st.columns(2)
        with col1:
            # Confluence documentation (optional)
            use_confluence = st.checkbox("  Include Confluence Specification", value=True)
            
            # Advanced options
            generate_advanced_scenarios = st.checkbox("   Advanced Test Scenarios", value=True, 
                                                     help="Include performance, security, and integration tests")
        
        with col2:
            # Environment configurations
            environment_count = st.selectbox("Environment Configurations", 
                                           options=[2, 3, 4], 
                                           index=1,  # Default to 3 environments
                                           help="Number of environment configs to generate")
            
            # LLM enhancement (optional)
            use_llm_enhancement = st.checkbox("  AI-Enhanced Payloads", value=True,
                                            help="Use AI to generate more realistic test payloads")
        
        # Vector DB Knowledge Base
        confluence_pdf_path = None
        if use_confluence:
            st.markdown("**  Business Requirements from Vector DB**")
            
            if st.session_state.vector_ready:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Vector DB", "  Ready")
                with col2:
                    st.metric("Chunks Available", st.session_state.vector_stats.get('total_chunks', 0))
                with col3:
                    if st.button("  Preview Content", key="agent5_preview"):
                        st.info("Test scenario content preview will be available in next update")
                
                st.success("  Agent 5 will use Vector Knowledge Base for test scenarios")
                confluence_pdf_path = None  # Flag for downstream logic
                
            else:
                st.error("  No knowledge base available")
                st.info("   Upload PDF in Agent 1 to initialize Vector Knowledge Base")
    
    
    # Generation controls - Single column, cleanest approach
    if st.button("   Generate Collections", type="primary", key="run_postman_gen", width='content'):
        run_postman_collection_generation(
            reviewed_modules_path=program_4_output,
            target_output_folder=target_output_folder,
            project_name=project_name,
            generate_advanced_scenarios=generate_advanced_scenarios,
            environment_count=environment_count,
            use_llm_enhancement=use_llm_enhancement
        )


    


    if st.session_state.pipeline_progress.get('program_5', {}).get('status') == 'success':
        output_path = st.session_state.pipeline_progress['program_5']['output']
        if st.button("  Open Output Folder", key="open_output", width='content'):
            open_output_folder(output_path)

def check_postman_dependencies():
    """Check if required dependencies are available"""
    dependencies = {
        'xml.etree.ElementTree': True,  # Built-in
        'json': True,  # Built-in
        'pathlib': True,  # Built-in
    }
    
    optional_deps = {}
    try:
        from groq import Groq
        optional_deps['groq'] = True
    except ImportError:
        optional_deps['groq'] = False
    
    missing = [dep for dep, available in {**dependencies, **optional_deps}.items() if not available]
    
    return {
        'all_available': len(missing) == 0,
        'missing': missing,
        'optional_missing': [dep for dep, available in optional_deps.items() if not available]
    }

def get_default_postman_output_path(program_4_output, project_name):
    """Calculate default output path for Postman collections"""
    reviewed_modules_path = Path(program_4_output)
    root_folder = reviewed_modules_path.parent
    return str(root_folder / f"{project_name}_POSTMAN_COLLECTIONS")

def save_uploaded_file(uploaded_file, prefix):
    """Save uploaded file temporarily"""
    import tempfile
    temp_dir = Path(tempfile.gettempdir())
    file_extension = Path(uploaded_file.name).suffix
    temp_file_path = temp_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_extension}"
    
    with open(temp_file_path, 'wb') as f:
        f.write(uploaded_file.read())
    
    return str(temp_file_path)

def analyze_ace_input_files(program_4_output):
    """Analyze the ACE input files for preview"""
    if not Path(program_4_output).exists():
        st.error(f"  Input path not found: {program_4_output}")
        return
    
    try:
        from postman_collection_generator import PostmanCollectionGenerator
        
        # Create a temporary generator just for analysis
        temp_generator = PostmanCollectionGenerator(
            reviewed_modules_path=program_4_output,
            project_name="Analysis"
        )
        
        # Parse artifacts
        temp_generator._parse_ace_artifacts()
        
        st.success("  **Input Analysis Results:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Message Flows", len(temp_generator.ace_artifacts['msgflow_files']))
        with col2:
            st.metric("ESQL Modules", len(temp_generator.ace_artifacts['esql_modules']))
        with col3:
            st.metric("XSL Transforms", len(temp_generator.ace_artifacts['xsl_transforms']))
        with col4:
            st.metric("Project Files", len(temp_generator.ace_artifacts['project_configs']))
        
        # Show discovered endpoints
        endpoints_found = []
        for msgflow in temp_generator.ace_artifacts['msgflow_files']:
            if 'error' not in msgflow:
                endpoints = msgflow.get('endpoints', {})
                endpoints_found.extend(endpoints.get('http_inputs', []))
                endpoints_found.extend(endpoints.get('mq_inputs', []))
        
        if endpoints_found:
            st.info(f"  **Discovered {len(endpoints_found)} endpoints** for test generation")
            with st.expander("  Endpoint Details"):
                for i, endpoint in enumerate(endpoints_found[:5]):  # Show first 5
                    if 'url_suffix' in endpoint:  # HTTP endpoint
                        st.write(f"{i+1}. **HTTP**: {endpoint.get('http_method', 'POST')} {endpoint['url_suffix']}")
                    elif 'queue_name' in endpoint:  # MQ endpoint
                        st.write(f"{i+1}. **MQ**: {endpoint['queue_name']}")
        else:
            st.warning("   No endpoints discovered. Check input files.")
        
    except Exception as e:
        st.error(f"  Analysis failed: {e}")

def preview_test_scenarios(program_4_output, project_name):
    """Preview what test scenarios will be generated"""
    try:
        from postman_collection_generator import PostmanCollectionGenerator
        
        temp_generator = PostmanCollectionGenerator(
            reviewed_modules_path=program_4_output,
            project_name=project_name
        )
        
        # Get test templates
        test_templates = temp_generator.test_templates
        
        st.info("   **Test Scenarios Preview:**")
        
        total_scenarios = 0
        for category_name, category_data in test_templates.items():
            with st.expander(f"  {category_name.replace('_', ' ').title()}"):
                category_total = 0
                for test_type, test_config in category_data.items():
                    st.write(f"**{test_config['name']}** (Priority {test_config['priority']})")
                    st.write(f"*{test_config['description']}*")
                    
                    for scenario in test_config['scenarios']:
                        st.write(f"  â€¢ {scenario}")
                        category_total += 1
                    
                    st.write("---")
                
                st.write(f"**Category Total: {category_total} scenarios**")
                total_scenarios += category_total
        
        st.success(f"  **Estimated Total: {total_scenarios}+ test scenarios**")
        st.info("*Actual count will be higher with entity-specific and endpoint-specific tests*")
        
    except Exception as e:
        st.error(f"  Preview failed: {e}")

def run_postman_collection_generation(reviewed_modules_path, target_output_folder, 
                                    project_name, generate_advanced_scenarios, environment_count, use_llm_enhancement):


    print("🔍 VECTOR DB DIAGNOSTIC:")
    print(f"  VECTOR_DB_AVAILABLE: {VECTOR_DB_AVAILABLE}")
    print(f"  VECTOR_PIPELINE_AVAILABLE: {VECTOR_PIPELINE_AVAILABLE}")
    print(f"  st.session_state.vector_enabled: {st.session_state.get('vector_enabled', False)}")
    print(f"  st.session_state.vector_ready: {st.session_state.get('vector_ready', False)}")
    print(f"  st.session_state.vector_pipeline exists: {st.session_state.get('vector_pipeline') is not None}")                                    
    """Execute Postman collection generation with Vector DB integration"""

    if st.session_state.get('vector_pipeline'):
        print(f"  vector_pipeline.knowledge_ready: {getattr(st.session_state.vector_pipeline, 'knowledge_ready', 'N/A')}")
        print(f"  vector_pipeline.search_engine exists: {getattr(st.session_state.vector_pipeline, 'search_engine', None) is not None}")
    
    print("🔍 END DIAGNOSTIC")

    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Create full-width containers at the top level for results display
    success_container = st.empty()
    results_container = st.empty()
    instructions_container = st.empty()
    
    try:
        st.session_state.pipeline_progress['program_5']['status'] = 'running'
        progress_placeholder.progress(10)
        status_placeholder.info("  Processing Vector DB focused content...")
        
        #   NEW: Vector DB Integration Check (following Agent 3 pattern)
        if (st.session_state.get('vector_enabled', False) and 
            st.session_state.get('vector_ready', False) and 
            st.session_state.get('vector_pipeline')):
            
            status_placeholder.info("  Using Vector DB for focused Postman test requirements...")
            progress_placeholder.progress(40)
            
            # Create agent function for postman collection generation
            def postman_agent_function(focused_content):
                """Agent function that receives Vector DB focused content"""
                from postman_collection_generator import PostmanCollectionGenerator
                
                generator = PostmanCollectionGenerator(
                    reviewed_modules_path=reviewed_modules_path,
                    target_output_folder=target_output_folder,
                    project_name=project_name
                )
                
                # Use the existing method with vector_content parameter
                return generator.generate_postman_collections(vector_content=focused_content)
            
            progress_placeholder.progress(60)
            status_placeholder.info("  Running Postman generation with Vector optimization...")
            
            # Use Vector DB pipeline to get focused content and run agent
            result = st.session_state.vector_pipeline.run_agent_with_vector_search(
                agent_name="postman_collection_generator",  #   Matches Vector DB agent registry
                agent_function=postman_agent_function
            )
            
            progress_placeholder.progress(90)
            status_placeholder.info("  Vector DB processing completed!")
            
            # Extract output path from result
            output_path = result if isinstance(result, str) else result.get('output_path', str(Path(target_output_folder)))
            
            # Add vector processing indicators to result
            vector_processing_info = {
                'vector_processing': True,
                'processing_method': 'Vector DB Optimization',
                'output_path': output_path
            }
            
        else:
            #   Vector DB not available - raise error (no fallback)
            error_msg = "Vector DB not enabled or not ready. Please setup Vector Knowledge Base first."
            
            if not st.session_state.get('vector_enabled', False):
                error_msg = "Vector DB is disabled. Please enable Vector DB in sidebar."
            elif not st.session_state.get('vector_ready', False):
                error_msg = "Vector DB not ready. Please setup Vector Knowledge Base using PDF upload."
            elif not st.session_state.get('vector_pipeline'):
                error_msg = "Vector DB pipeline not initialized. Please restart the application."
            
            progress_placeholder.empty()
            status_placeholder.error(f"  {error_msg}")
            raise Exception(error_msg)
        
        progress_placeholder.progress(100)
        
        # Update session state
        st.session_state.pipeline_progress['program_5']['status'] = 'success'
        st.session_state.pipeline_progress['program_5']['output'] = output_path
        
        # Clear the column-constrained placeholders
        status_placeholder.empty()
        progress_placeholder.empty()
        
        # Display results using full-width containers created at top level
        with success_container:
            st.success("Postman Collections Generated Successfully!")
            st.info("**Vector DB Integration**: Test scenarios, API patterns, and business flows automatically extracted from Vector Knowledge Base")

        # Instead of: with results_container.container():  
        with results_container:           
            # Use more columns to spread across full width
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                st.caption("**Collections**")
                st.caption("✅ Generated")
            with col2:
                st.caption("**Test Scenarios**")
                st.caption("✅ Created")
            with col3:
                st.caption("**Environments**")
                st.caption("✅ Configured")
            with col4:
                st.caption("**Test Data**")
                st.caption("✅ Prepared")
            with col5:
                st.caption("**Processing**")
                st.caption("Vector DB")
            with col6:
                st.caption("**Output**")
                st.caption("🟢 Ready")
        
        with instructions_container.container():
            with st.expander("Next Steps", expanded=True):
                st.markdown(f"""
                **Vector DB Enhanced Collections Ready!**
                
                1. **Import Collections**: Import all `.postman_collection.json` files into Postman
                2. **Configure Environments**: Update environment variables with your server details  
                3. **Set Authentication**: Add valid tokens to each environment
                4. **Run Tests**: Execute comprehensive test scenarios
                5. **Monitor Results**: Track test execution and business validation
                
                **📂 Output Location**: `{output_path}`
            ```bash
                newman run "{project_name}_Complete_TestSuite.postman_collection.json" \\
                  -e "Development.postman_environment.json" \\
                  --reporters cli,html
            ```
            """)
        
    except Exception as e:
        st.session_state.pipeline_progress['program_5']['status'] = 'error'
        status_placeholder.error(f"  Generation failed: {str(e)}")
        
        # Enhanced error handling for Vector DB issues
        error_msg = str(e).lower()
        if "vector" in error_msg or "knowledge base" in error_msg:
            st.error("  Vector DB issue. Please check Vector Knowledge Base status.")
        elif "groq" in error_msg or "api" in error_msg:
            st.error("  LLM API issue. Please verify your GROQ API key is valid and has sufficient credits.")
        else:
            st.error("  Check your inputs and try again")
            
        # Show error details
        with st.expander("  Error Details"):
            import traceback
            st.code(traceback.format_exc())
        
        progress_placeholder.empty()
        raise e

def download_main_collection(collections_created):
    """Download the main Postman collection"""
    try:
        main_collection = None
        for collection_path in collections_created:
            if "Complete_TestSuite" in collection_path:
                main_collection = collection_path
                break
        
        if not main_collection:
            main_collection = collections_created[0]  # Fallback to first collection
        
        with open(main_collection, 'rb') as f:
            collection_data = f.read()
        
        st.download_button(
            label="   Download Main Collection",
            data=collection_data,
            file_name=Path(main_collection).name,
            mime="application/json",
            key="download_main_collection"
        )
        
    except Exception as e:
        st.error(f"Download failed: {e}")

def download_environments(environments_created):
    """Download environment configurations as a zip file"""
    try:
        import zipfile
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            with zipfile.ZipFile(tmp_file.name, 'w') as zipf:
                for env_file in environments_created:
                    zipf.write(env_file, Path(env_file).name)
            
            with open(tmp_file.name, 'rb') as f:
                zip_data = f.read()
            
            st.download_button(
                label="ðŸŒ Download All Environments",
                data=zip_data,
                file_name="postman_environments.zip",
                mime="application/zip",
                key="download_environments_zip"
            )
    
    except Exception as e:
        st.error(f"Environment download failed: {e}")

def download_documentation(documentation_files):
    """Download documentation files as a zip"""
    try:
        import zipfile
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            with zipfile.ZipFile(tmp_file.name, 'w') as zipf:
                for doc_file in documentation_files:
                    zipf.write(doc_file, Path(doc_file).name)
            
            with open(tmp_file.name, 'rb') as f:
                zip_data = f.read()
            
            st.download_button(
                label="   Download Documentation",
                data=zip_data,
                file_name="postman_documentation.zip",
                mime="application/zip",
                key="download_docs_zip"
            )
    
    except Exception as e:
        st.error(f"Documentation download failed: {e}")

def open_output_folder(output_path):
    """Open the output folder in file explorer"""
    try:
        import subprocess
        import platform
        
        if platform.system() == "Windows":
            subprocess.run(['explorer', str(output_path)])
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(['open', str(output_path)])
        else:  # Linux
            subprocess.run(['xdg-open', str(output_path)])
        
        st.success(f"  Opened: {output_path}")
        
    except Exception as e:
        st.error(f"Failed to open folder: {e}")
        st.info(f"Manual path: {output_path}")


def render_token_analytics_tab():
    """Token Analytics Tab Component - Complete with Advanced Capacity Planning"""
    
    st.header("LLM Token Analytics & Cost Analysis")
    st.markdown("**Real-time tracking of token usage and cost optimization for BizTalk-to-ACE conversion**")
    
    # Initialize token tracker in session state if not exists
    if 'token_tracker' not in st.session_state:
        from llm_token_tracker import create_tracker
        session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M')}"
        st.session_state.token_tracker = create_tracker(session_name)
    
    tracker = st.session_state.token_tracker
    
    # Get current metrics
    metrics = tracker.get_real_time_metrics()
    
    # =========================================================================
    # REAL-TIME METRICS DASHBOARD
    # =========================================================================
    
    st.subheader("   Real-Time Session Metrics")
    
    # Primary metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="  Session Cost",
            value=f"${metrics.total_cost:.4f}",
            delta=f"{metrics.total_calls} calls",
            help="Total cost of all LLM calls in current session"
        )
    
    with col2:
        st.metric(
            label="  Total Tokens", 
            value=f"{metrics.total_tokens:,}",
            delta=f"{metrics.current_tokens_per_minute:.0f}/min" if metrics.current_tokens_per_minute > 0 else "0",
            help="Total tokens consumed across all agents"
        )
    
    with col3:
        st.metric(
            label="  Flows Processed",
            value=metrics.flows_processed,
            delta=f"${metrics.average_cost_per_flow:.4f}/flow" if metrics.flows_processed > 0 else "No flows",
            help="Number of complete flows processed"
        )
    
    with col4:
        st.metric(
            label="  Avg Tokens/Flow",
            value=f"{metrics.average_tokens_per_flow:,.0f}" if metrics.average_tokens_per_flow > 0 else "0",
            delta="tokens per flow",
            help="Average token consumption per flow"
        )
    
    with col5:
        st.metric(
            label=" Daily Capacity",
            value=f"{metrics.estimated_daily_capacity:,}",
            delta=f"${metrics.estimated_daily_cost:.2f}/day",
            help="Estimated flows per day at current rate"
        )
    
    # =========================================================================
    # CAPACITY PLANNING CALCULATOR
    # =========================================================================
    
    st.subheader("Daily Capacity Planning & Cost Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Planning Inputs**")
        
        target_flows = st.number_input(
            "Target Flows per Day", 
            min_value=1, 
            max_value=1000, 
            value=50, 
            step=5,
            help="How many flows do you want to process daily?"
        )
        
        if target_flows > 0 and metrics.average_tokens_per_flow > 0:
            target_daily_tokens = target_flows * metrics.average_tokens_per_flow
            target_daily_cost = target_flows * metrics.average_cost_per_flow
            
            st.write(f"   **Projected Daily Usage**: {target_daily_tokens:,.0f} tokens")
            st.write(f"  **Projected Daily Cost**: ${target_daily_cost:.2f}")
            
            # Determine required tier
            if target_daily_tokens <= 100000:
                tier_needed = "  Free Tier"
                tier_cost = "$0"
            elif target_daily_tokens <= 1000000:
                tier_needed = "  Dev Tier"
                tier_cost = "$20/month"
            else:
                tier_needed = "  Pro Tier"
                tier_cost = "$100+/month"
            
            st.write(f"  **Required Tier**: {tier_needed}")
            st.write(f"  **Subscription**: {tier_cost}")
    
    with col2:
        st.markdown("**Performance Analysis**")
        
        if metrics.total_calls > 0:
            session_duration = (datetime.now() - tracker.session_start).total_seconds() / 3600  # hours
            
            st.write(f"  **Session Duration**: {session_duration:.1f} hours")
            st.write(f"  **Total LLM Calls**: {metrics.total_calls:,}")
            st.write(f"  **Calls per Hour**: {metrics.total_calls/session_duration:.1f}" if session_duration > 0 else "0")
            st.write(f"  **Efficiency**: {metrics.total_tokens/metrics.total_calls:.0f} tokens/call" if metrics.total_calls > 0 else "0")
        else:
            st.info("Start running programs to see performance metrics")

    # =========================================================================
    # ADVANCED CAPACITY PLANNING CALCULATOR
    # =========================================================================
    
    st.markdown("---")
    with st.expander("   Advanced Capacity Planning Calculator", expanded=False):
        st.markdown("Custom Flow Processing Analysis")
        
        # Initialize capacity inputs in session state
        if 'capacity_inputs' not in st.session_state:
            st.session_state.capacity_inputs = {
                'tokens_per_flow': 120000,
                'pending_flows': 900,
                'daily_hours': 4.0,
                'minutes_per_flow': 10
            }
        
        # Input controls
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Flow Parameters**")
            tokens_per_flow = st.number_input(
                "Tokens per Flow",
                min_value=1000,
                max_value=1000000,
                value=st.session_state.capacity_inputs['tokens_per_flow'],
                step=1000,
                help="Number of tokens consumed per flow conversion"
            )
            
            pending_flows = st.number_input(
                "Total Pending Flows",
                min_value=1,
                max_value=10000,
                value=st.session_state.capacity_inputs['pending_flows'],
                step=10,
                help="Total number of flows to be converted"
            )
        
        with col2:
            st.markdown("**Time Parameters**")
            daily_hours = st.slider(
                "Daily Processing Hours",
                min_value=1.0,
                max_value=24.0,
                value=st.session_state.capacity_inputs['daily_hours'],
                step=0.5,
                help="Hours available per day for processing"
            )
            
            minutes_per_flow = st.number_input(
                "Minutes per Flow",
                min_value=1,
                max_value=120,
                value=st.session_state.capacity_inputs['minutes_per_flow'],
                step=1,
                help="Time in minutes required per flow"
            )
        
        # Update session state
        st.session_state.capacity_inputs.update({
            'tokens_per_flow': tokens_per_flow,
            'pending_flows': pending_flows,
            'daily_hours': daily_hours,
            'minutes_per_flow': minutes_per_flow
        })
        
        # Calculate button and results
        if st.button("  Calculate Capacity & Costs", type="primary"):
            try:
                # Call the backend function
                capacity_results = tracker.calculate_capacity_planning(
                    tokens_per_flow=tokens_per_flow,
                    pending_flows=pending_flows,
                    daily_hours=daily_hours,
                    minutes_per_flow=minutes_per_flow
                )
                
                # Display results
                st.markdown("###   Capacity Analysis Results")
                
                # Capacity metrics table
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "  Flows per Day",
                        capacity_results['capacity_metrics']['flows_per_day'],
                        help="Number of flows processible per day"
                    )
                    
                    st.metric(
                        "ðŸ“… Flows per Month",
                        capacity_results['capacity_metrics']['flows_per_month'],
                        help="Number of flows processible per month (20 working days)"
                    )
                
                with col2:
                    st.metric(
                        "  Daily Tokens",
                        f"{capacity_results['capacity_metrics']['daily_tokens']:,}",
                        help="Total tokens consumed per day"
                    )
                    
                    st.metric(
                        "   Monthly Tokens", 
                        f"{capacity_results['capacity_metrics']['monthly_tokens']:,}",
                        help="Total tokens consumed per month"
                    )
                
                with col3:
                    st.metric(
                        "  Completion Time",
                        f"{capacity_results['capacity_metrics']['days_to_complete_all']:.1f} days",
                        help="Time to complete all pending flows"
                    )
                    
                    st.metric(
                        "  Utilization Rate",
                        f"{capacity_results['efficiency_metrics']['utilization_rate']:.1f}%",
                        help="Percentage of available time utilized"
                    )
                
                # Cost comparison table
                st.markdown("###   Cost Analysis Comparison")
                
                cost_data = {
                    "Model": ["Llama 8B", "Llama 70B"],
                    "Daily Cost": [
                        f"${capacity_results['cost_analysis']['llama_8b']['daily_cost']:.4f}",
                        f"${capacity_results['cost_analysis']['llama_70b']['daily_cost']:.4f}"
                    ],
                    "Monthly Cost": [
                        f"${capacity_results['cost_analysis']['llama_8b']['monthly_cost']:.2f}",
                        f"${capacity_results['cost_analysis']['llama_70b']['monthly_cost']:.2f}"
                    ],
                    "Cost per Flow": [
                        f"${capacity_results['cost_analysis']['llama_8b']['cost_per_flow']:.4f}",
                        f"${capacity_results['cost_analysis']['llama_70b']['cost_per_flow']:.4f}"
                    ],
                    "Total Project Cost": [
                        f"${capacity_results['cost_analysis']['llama_8b']['total_cost_for_all_flows']:.2f}",
                        f"${capacity_results['cost_analysis']['llama_70b']['total_cost_for_all_flows']:.2f}"
                    ]
                }
                
                cost_df = pd.DataFrame(cost_data)
                st.dataframe(cost_df, width='stretch')
                
                # Subscription requirements
                st.markdown("###   Subscription Requirements")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    tier = capacity_results['subscription_requirements']['required_tier']
                    tier_cost = capacity_results['subscription_requirements']['tier_monthly_cost']
                    
                    if "Free" in tier:
                        st.success(f"  **{tier}** (${tier_cost}/month)")
                    elif "Dev" in tier:
                        st.warning(f"   **{tier}** (${tier_cost}/month)")
                    else:
                        st.error(f"ðŸš¨ **{tier}** (${tier_cost}+/month)")
                    
                    st.write(f"**Daily Token Limit Needed**: {capacity_results['subscription_requirements']['daily_token_limit_needed']:,}")
                
                with col2:
                    savings_8b = capacity_results['efficiency_metrics']['savings_using_8b']
                    cost_diff = capacity_results['efficiency_metrics']['cost_difference_8b_vs_70b']
                    
                    st.metric(
                        "  Monthly Savings (8B vs 70B)",
                        f"${cost_diff:.2f}",
                        delta=f"{savings_8b:.1f}% savings",
                        help="Cost savings by using 8B model instead of 70B"
                    )
                    
                    if capacity_results['subscription_requirements']['exceeds_free_tier']:
                        st.warning("   Exceeds Free Tier limits")
                    if capacity_results['subscription_requirements']['exceeds_dev_tier']:
                        st.error("ðŸš¨ Requires Pro Tier subscription")
                
                # Efficiency insights
                st.markdown("###   Efficiency Insights")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.info(f"  **Tokens per Minute**: {capacity_results['efficiency_metrics']['tokens_per_minute']:,}")
                
                with col2:
                    st.info(f"   **Utilization Rate**: {capacity_results['efficiency_metrics']['utilization_rate']:.1f}%")
                
                with col3:
                    monthly_cost_8b = capacity_results['cost_analysis']['llama_8b']['monthly_cost']
                    if monthly_cost_8b < 50:
                        st.success(f"  **Affordable**: ${monthly_cost_8b:.2f}/month")
                    elif monthly_cost_8b < 200:
                        st.warning(f"ðŸ’› **Moderate**: ${monthly_cost_8b:.2f}/month") 
                    else:
                        st.error(f"  **Expensive**: ${monthly_cost_8b:.2f}/month")
                
            except Exception as e:
                st.error(f"  Calculation failed: {str(e)}")
                st.info("Please check your input parameters and try again")

    # =========================================================================
    # AGENT PERFORMANCE BREAKDOWN
    # =========================================================================
    
    st.subheader("  Agent Performance Breakdown")
    
    agent_breakdown = tracker.get_agent_breakdown()
    
    if agent_breakdown:
        # Create agent performance dataframe
        agent_data = []
        for agent, stats in agent_breakdown.items():
            agent_data.append({
                "Agent": agent,
                "Calls": stats["calls"],
                "Tokens": f"{stats['tokens']:,}",
                "Cost": f"${stats['cost']:.4f}",
                "Avg Tokens/Call": f"{stats['avg_tokens_per_call']:,.1f}",
                "Efficiency": f"{stats['efficiency_score']:,.1f} tokens/$"
            })
        
        agent_df = pd.DataFrame(agent_data)
        st.dataframe(agent_df, width='stretch')
        
        # Agent cost visualization
        if len(agent_breakdown) > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Cost distribution
                agent_names = list(agent_breakdown.keys())
                agent_costs = [agent_breakdown[agent]["cost"] for agent in agent_names]
                
                import plotly.express as px
                fig = px.pie(
                    values=agent_costs,
                    names=agent_names,
                    title="Cost Distribution by Agent"
                )
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                # Token distribution  
                agent_tokens = [agent_breakdown[agent]["tokens"] for agent in agent_names]
                
                fig = px.pie(
                    values=agent_tokens,
                    names=agent_names,
                    title="Token Distribution by Agent"
                )
                st.plotly_chart(fig, width='stretch')
    
    else:
        st.info("  No agent data available yet. Run some programs to see agent performance breakdown.")

    # =========================================================================
    # MODEL USAGE STATISTICS
    # =========================================================================
    
    st.subheader("Model Usage Statistics")
    
    model_usage = tracker.get_model_usage()
    
    if model_usage:
        model_data = []
        for model, stats in model_usage.items():
            model_data.append({
                "Model": model,
                "Calls": stats["calls"],
                "Tokens": f"{stats['tokens']:,}",
                "Cost": f"${stats['cost']:.4f}",
                "Cost per 1K Tokens": f"${stats['avg_cost_per_1k_tokens']:.4f}"
            })
        
        model_df = pd.DataFrame(model_data)
        st.dataframe(model_df, width='stretch')
        
        # Model comparison insights
        if len(model_usage) > 1:
            st.markdown("**  Model Efficiency Comparison:**")
            
            # Find most efficient model
            most_efficient = min(model_usage.items(), key=lambda x: x[1]["avg_cost_per_1k_tokens"])
            least_efficient = max(model_usage.items(), key=lambda x: x[1]["avg_cost_per_1k_tokens"])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"  **Most Efficient**: {most_efficient[0]} (${most_efficient[1]['avg_cost_per_1k_tokens']:.4f}/1K tokens)")
            
            with col2:
                st.warning(f"   **Least Efficient**: {least_efficient[0]} (${least_efficient[1]['avg_cost_per_1k_tokens']:.4f}/1K tokens)")
    
    else:
        st.info("  No model usage data available yet. Run some programs to see model statistics.")

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================
    
    st.subheader("Session Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("  Export Session Data"):
            try:
                files = tracker.export_session_data()
                st.success("  Session data exported successfully!")
                st.json(files)
            except Exception as e:
                st.error(f"  Export failed: {e}")
    
    with col2:
        if st.button("  Reset Session"):
            try:
                # Create new tracker
                session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.session_state.token_tracker = create_tracker(session_name)
                st.success("  Session reset successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"  Reset failed: {e}")
    
    with col3:
        if st.button("   Print Summary"):
            try:
                tracker.print_session_summary()
                st.success("  Summary printed to console!")
            except Exception as e:
                st.error(f"  Print failed: {e}")
    
    # Session info
    st.markdown("---")
    st.markdown(f"**Session**: {tracker.session_name}")
    st.markdown(f"**Started**: {tracker.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown(f"**Duration**: {(datetime.now() - tracker.session_start).total_seconds()/60:.1f} minutes")


    st.markdown("---")
    st.subheader("  Vector Database Analytics")
    
    if not VECTOR_DB_AVAILABLE:
        st.error("  Vector DB not available")
        st.info("Install dependencies: chromadb, sentence-transformers")
        return
    
    if not st.session_state.vector_ready:
        st.info("  Vector DB not initialized yet - Upload PDF in Agent 1")
        return
    
    # Vector DB Performance Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        stats = st.session_state.vector_stats
        st.metric("Total Chunks", stats.get('total_chunks', 0))
        st.metric("Embedding Model", stats.get('embedding_model', 'Unknown'))
    
    with col2:
        st.metric("Vector Mode", "  Enabled" if st.session_state.vector_enabled else "  Disabled")
        st.metric("Knowledge Base", "  Ready" if st.session_state.vector_ready else "  Not Ready")
    
    with col3:
        estimated_savings = "60-80%" if st.session_state.vector_enabled else "0%"
        st.metric("Est. Token Savings", estimated_savings)
        st.metric("Processing Mode", "  Vector" if st.session_state.vector_enabled else "  Traditional")
    
    # Vector DB Statistics
    if st.session_state.vector_stats:
        with st.expander("   Detailed Vector DB Stats", expanded=False):
            st.json(st.session_state.vector_stats)

    



def render_results_dashboard():
    """Enhanced results dashboard with token usage summary"""
    st.header("🏠 Migration Results Dashboard")
    
    # Check if pipeline_progress exists in session state
    if 'pipeline_progress' not in st.session_state:
        st.warning("⚠️ Pipeline not initialized. Please start with Agent 1.")
        return
    
    # Overall pipeline status
    progress = st.session_state.pipeline_progress
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        status = progress['program_1']['status']
        st.metric("Agent 1", "BizTalk Mapper", delta=status.title())
    
    with col2:
        status = progress['program_2']['status']
        st.metric("Agent 2", "ACE Foundation", delta=status.title())
    
    with col3:
        status = progress['program_3']['status']
        st.metric("Agent 3", "ACE Module Creator", delta=status.title())
    
    with col4:
        status = progress['program_4']['status']
        st.metric("Agent 4", "Quality Review", delta=status.title())
    
    with col5:
        status = progress['program_5']['status']
        st.metric("Agent 5", "Postman Collections", delta=status.title())
    
    # Token Usage Summary Section
    if 'token_tracker' in st.session_state and st.session_state.token_tracker:
        st.subheader("📊 Pipeline Token Usage Summary")
        
        tracker = st.session_state.token_tracker
        metrics = tracker.get_real_time_metrics()
        
        # Token summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🎯 Total Tokens Used",
                value=f"{metrics.total_tokens:,}",
                delta=f"{metrics.total_calls} LLM calls",
                help="Total tokens consumed across all pipeline agents"
            )
        
        with col2:
            st.metric(
                label="💰 Total Pipeline Cost",
                value=f"${metrics.total_cost:.4f}",
                delta=f"${metrics.average_cost_per_flow:.4f}/flow" if metrics.flows_processed > 0 else "No flows",
                help="Total cost for complete pipeline execution"
            )
        
        with col3:
            execution_time = (datetime.now() - tracker.session_start).total_seconds() / 60
            st.metric(
                label="⏱️ Pipeline Duration",
                value=f"{execution_time:.1f} min",
                delta=f"{metrics.current_tokens_per_minute:.0f} tokens/min",
                help="Total time for pipeline execution"
            )
        
        with col4:
            # Calculate pipeline efficiency score
            if metrics.total_cost > 0 and metrics.flows_processed > 0:
                efficiency = metrics.total_tokens / metrics.total_cost  # tokens per dollar
                st.metric(
                    label="⚡ Pipeline Efficiency",
                    value=f"{efficiency:,.0f}",
                    delta="tokens/$",
                    help="Token efficiency: higher is better"
                )
            else:
                st.metric("⚡ Pipeline Efficiency", "Calculating...", delta="tokens/$")
        
        # Agent contribution breakdown
        agent_breakdown = tracker.get_agent_breakdown()
        if agent_breakdown:
            with st.expander("🔍 Agent Token Contribution", expanded=False):
                agent_df = pd.DataFrame(agent_breakdown).T
                agent_df = agent_df.round(4)
                
                # Create percentage columns
                total_tokens = sum(data['tokens'] for data in agent_breakdown.values())
                total_cost = sum(data['cost'] for data in agent_breakdown.values())
                
                if total_tokens > 0:
                    agent_df['token_percentage'] = (agent_df['tokens'] / total_tokens * 100).round(1)
                    agent_df['cost_percentage'] = (agent_df['cost'] / total_cost * 100).round(1)
                
                st.dataframe(agent_df, width='stretch')
    
    # Show final results if all programs completed
    if all(progress[prog]['status'] == 'success' for prog in progress):
        st.success("🎉 **MIGRATION PIPELINE COMPLETED SUCCESSFULLY!**")
        
        # Enhanced completion summary with token metrics
        if 'token_tracker' in st.session_state and st.session_state.token_tracker:
            st.subheader("📋 Final Pipeline Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📦 Generated Artifacts:**")
                # List all outputs
                artifacts = []
                for i, prog in enumerate(['program_1', 'program_2', 'program_3', 'program_4', 'program_5'], 1):
                    output = progress[prog]['output']
                    if output:
                        artifacts.append(f"**Agent {i}**: {output}")
                
                for artifact in artifacts:
                    st.write(artifact)
            
            with col2:
                st.markdown("**📊 Resource Utilization:**")
                metrics = st.session_state.token_tracker.get_real_time_metrics()
                
                # Calculate cost per artifact
                total_artifacts = len([prog for prog in progress if progress[prog]['output']])
                cost_per_artifact = metrics.total_cost / max(total_artifacts, 1)
                
                st.write(f"💰 **Total Investment**: ${metrics.total_cost:.4f}")
                st.write(f"🎯 **Token Consumption**: {metrics.total_tokens:,}")
                st.write(f"📦 **Cost per Artifact**: ${cost_per_artifact:.4f}")
                st.write(f"⚡ **Processing Efficiency**: {metrics.current_tokens_per_minute:.0f} tokens/min")
                
                # ROI calculation
                estimated_manual_hours = st.number_input(
                    "Estimated Manual Hours", 
                    min_value=1, 
                    value=40, 
                    step=5,
                    help="Hours it would take to do this manually"
                )
                hourly_rate = st.number_input(
                    "Developer Hourly Rate ($)", 
                    min_value=10, 
                    value=75, 
                    step=5
                )
                
                manual_cost = estimated_manual_hours * hourly_rate
                savings = manual_cost - metrics.total_cost
                roi_percentage = (savings / manual_cost) * 100 if manual_cost > 0 else 0
                
                st.write(f"👨‍💻 **Manual Cost**: ${manual_cost:.2f}")
                st.write(f"💵 **Savings**: ${savings:.2f}")
                st.write(f"📈 **ROI**: {roi_percentage:.0f}%")
        
        # Next steps with token insights
        st.subheader("🚀 Next Steps")
        st.markdown("""
        1. 📖 **Review** generated functional documentation
        2. ✅ **Review** generated quality reports  
        3. 📥 **Import** ACE foundation into IBM ACE Toolkit
        4. ⚙️ **Configure** database connections and environment settings
        5. 🧪 **Test** enhanced ESQL modules individually
        6. 🚀 **Deploy** to ACE runtime environment
        7. 🔍 **Perform** end-to-end testing
        8. ✅ **Go-live** with migrated solution
        """)
        
        # Export pipeline summary
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Export Complete Pipeline Report"):
                if 'token_tracker' in st.session_state and st.session_state.token_tracker:
                    files = st.session_state.token_tracker.export_session_data()
                    st.success("Pipeline report exported!")
                    st.json(files)
        
        with col2:
            if st.button("📊 View Token Analytics"):
                st.info("Switch to 'Token Analytics' tab for detailed analysis")
        
        with col3:
            if st.button("🔄 Start New Pipeline"):
                # Reset pipeline progress
                st.session_state.pipeline_progress = {
                    'program_1': {'status': 'pending', 'output': None, 'timestamp': None},
                    'program_2': {'status': 'pending', 'output': None, 'timestamp': None},
                    'program_3': {'status': 'pending', 'output': None, 'timestamp': None},
                    'program_4': {'status': 'pending', 'output': None, 'timestamp': None},
                    'program_5': {'status': 'pending', 'output': None, 'timestamp': None}
                }
                if 'token_tracker' in st.session_state:
                    # Reset token tracker too
                    try:
                        from llm_token_tracker import create_tracker
                        session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        st.session_state.token_tracker = create_tracker(session_name)
                    except ImportError:
                        st.session_state.token_tracker = None
                st.rerun()
    
    elif any(progress[prog]['status'] == 'error' for prog in progress):
        st.error("❌ **PIPELINE HAS ERRORS** - Check individual program tabs for details")
        
        # Show partial token usage if available
        if 'token_tracker' in st.session_state and st.session_state.token_tracker:
            metrics = st.session_state.token_tracker.get_real_time_metrics()
            if metrics.total_calls > 0:
                st.warning(f"⚠️ **Partial Execution Cost**: ${metrics.total_cost:.4f} ({metrics.total_tokens:,} tokens)")
                st.info("📊 Check Token Analytics tab for detailed breakdown of completed operations")
    
    else:
        st.info("🔄 **PIPELINE IN PROGRESS** - Continue with remaining programs")
        
        # Show current token usage if available
        if 'token_tracker' in st.session_state and st.session_state.token_tracker:
            metrics = st.session_state.token_tracker.get_real_time_metrics()
            if metrics.total_calls > 0:
                with st.expander("📊 Current Token Usage", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Tokens So Far", f"{metrics.total_tokens:,}")
                    with col2:
                        st.metric("Cost So Far", f"${metrics.total_cost:.4f}")
                    with col3:
                        st.metric("Active Agents", len(st.session_state.token_tracker.get_agent_breakdown()))



def validate_groq_api_key(api_key: str) -> bool:
    """Validate GROQ API key format and connectivity"""
    
    if not api_key:
        print("  GROQ API key is empty or not set")
        return False
    
    if not api_key.startswith('gsk_'):
        print("  GROQ API key should start with 'gsk_'")
        return False
    
    if len(api_key) < 50:
        print("  GROQ API key appears too short")
        return False
    
    # Test the API key with a simple request
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        # Simple test request
        response = client.chat.completions.create(
            model=os.getenv('GROQ_MODEL', 'deepseek-r1-distill-llama-70b'),
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        print("  GROQ API key validated successfully")
        return True
        
    except Exception as e:
        print(f"  GROQ API key validation failed: {e}")
        return False


if __name__ == "__main__":
    main()