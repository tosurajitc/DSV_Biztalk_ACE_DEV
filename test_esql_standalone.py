#!/usr/bin/env python3
"""
Standalone ESQL Generator Test
Purpose: Test ESQL generation outside of Streamlit orchestrator context
"""

import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path so we can import modules
sys.path.append(os.getcwd())

try:
    from esql_generator import ESQLGenerator
    from llm_token_tracker import create_tracker
    print("Successfully imported modules")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def create_mock_data():
    """Create mock data for testing"""
    
    mock_vector_content = """
    Business Requirements for CW1 Document Processing:
    
    1. Message Flow: CW1_IN_Document_SND
    - Generic off-ramp to upload documents into CargoWiseOne
    - Uses universal event message format via eAdapter
    
    2. Processing Steps:
    - Receive CDM Document message from local queue
    - Transform customer format to CDM Document format
    - Enrich with database lookups (company codes, shipment validation)
    - Convert to CW1 universal event format
    - Route to CargoWise One via eAdapter
    
    3. Database Operations:
    - Company code lookup based on customer reference
    - Shipment validation using housebill number
    - IsPublished flag lookup based on document type
    
    4. Error Handling:
    - Route failed messages to error queue
    - Log transformation failures
    - Maintain message correlation IDs
    """
    
    mock_esql_template = {
        'path': 'ESQL_Template_Updated.ESQL'
    }
    
    mock_msgflow_content = {
        'path': 'output/MessageFlow_AGENT2_App_Name/AGENT2_Message_Flow.msgflow',
        'nodes': [
            {'name': 'DocPackApp', 'type': 'Compute', 'function': 'Document validation and packaging'},
            {'name': 'AzureBlob_To_CDM_Document', 'type': 'Compute', 'function': 'Transform Azure blob to CDM format'},
            {'name': 'CompanyCodeLookup', 'type': 'Compute', 'function': 'Database lookup for company codes'},
            {'name': 'ShipmentValidation', 'type': 'Compute', 'function': 'Validate shipment information'},
            {'name': 'CW1Transform', 'type': 'Compute', 'function': 'Transform to CW1 universal event format'}
        ],
        'connections': [
            {'from': 'DocPackApp', 'to': 'AzureBlob_To_CDM_Document'},
            {'from': 'AzureBlob_To_CDM_Document', 'to': 'CompanyCodeLookup'},
            {'from': 'CompanyCodeLookup', 'to': 'ShipmentValidation'},
            {'from': 'ShipmentValidation', 'to': 'CW1Transform'}
        ]
    }
    
    mock_json_mappings = {
        'path': 'output/biztalk_ace_component_mapping.json'  # ✅ FIX THIS LINE
    }
    
    return mock_vector_content, mock_esql_template, mock_msgflow_content, mock_json_mappings

def test_token_tracking_manually():
    """Test token tracking functionality manually"""
    print("\nTesting Token Tracking Functionality...")
    
    try:
        # Create a token tracker manually
        tracker = create_tracker("standalone_test_session")
        print("Token tracker created successfully")
        
        # Test manual tracking
        tracker.manual_track(
            agent="test_esql_generator",
            operation="test_generation",
            model="llama-3.3-70b-versatile",
            input_tokens=1500,
            output_tokens=800,
            flow_name="standalone_test"
        )
        print("Manual token tracking successful")
        
        # Get breakdown
        breakdown = tracker.get_agent_breakdown()
        print(f"Agent breakdown retrieved: {len(breakdown)} agents")
        
        for agent, stats in breakdown.items():
            print(f"   {agent}: {stats['calls']} calls, {stats['tokens']} tokens, ${stats['cost']:.4f}")
        
        return tracker
        
    except Exception as e:
        print(f"Token tracking test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_esql_generation_with_debug():
    """Test ESQL generation with enhanced debugging"""
    print("\nTesting ESQL Generation with Debug Output...")
    
    # Create test data
    vector_content, esql_template, msgflow_content, json_mappings = create_mock_data()
    
    try:
        # Initialize ESQL generator
        generator = ESQLGenerator()
        print("ESQL Generator initialized")
        
        # Create output directory
        output_dir = f"standalone_test_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory created: {output_dir}")
        
        # Test the generation
        print("\nStarting ESQL generation...")
        result = generator.generate_esql_files(
            vector_content=vector_content,
            esql_template=esql_template,
            msgflow_content=msgflow_content,
            json_mappings=json_mappings,
            output_dir=output_dir
        )
        
        print(f"\nGeneration Results:")
        print(f"   Status: {result.get('status', 'Unknown')}")
        print(f"   Modules Generated: {result.get('total_modules', 0)}")
        print(f"   LLM Calls Made: {result.get('llm_calls_made', 0)}")
        print(f"   Generation Method: {result.get('generation_method', 'Unknown')}")
        
        if result.get('generated_modules'):
            print(f"\nGenerated Files:")
            for i, module in enumerate(result['generated_modules'], 1):
                print(f"   {i}. {module.get('name', 'Unknown')}: {len(module.get('content', ''))} characters")
        
        return result
        
    except Exception as e:
        print(f"ESQL generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function"""
    print("=" * 60)
    print("STANDALONE ESQL GENERATOR TEST")
    print("=" * 60)
    
    # Check environment
    print("\nEnvironment Check:")
    groq_key = os.getenv('GROQ_API_KEY')
    if groq_key:
        print(f"GROQ_API_KEY found: {groq_key[:10]}...")
    else:
        print("GROQ_API_KEY not found in environment")
        return
    
    groq_model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
    print(f"GROQ_MODEL: {groq_model}")
    
    # Check if ESQL template file exists
    template_file = "ESQL_Template_Updated.ESQL"
    if os.path.exists(template_file):
        print(f"ESQL Template found: {template_file}")
    else:
        print(f"ESQL Template missing: {template_file}")
        print("   Please ensure ESQL_Template_Updated.ESQL exists in the root directory")
        return
    
    # Test 1: Token tracking functionality
    tracker = test_token_tracking_manually()
    
    # Test 2: ESQL generation
    result = test_esql_generation_with_debug()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if tracker:
        print("Token tracking: WORKING")
    else:
        print("Token tracking: FAILED")
    
    if result and result.get('status') == 'success':
        print("ESQL generation: WORKING")
    else:
        print("ESQL generation: FAILED")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()