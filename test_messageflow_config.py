#!/usr/bin/env python3
"""
Test Script for Enrichment Generator v2.0
Validates detection, template loading, and conditional generation
"""

import sys
import os
sys.path.insert(0, '/mnt/project')

from enrichment_generator import EnrichmentGenerator

def test_enrichment_detection():
    """Test 1: Enrichment flag detection"""
    print("\n" + "="*60)
    print("TEST 1: Enrichment Flag Detection")
    print("="*60)
    
    generator = EnrichmentGenerator(groq_api_key="test_key")
    
    # Test with sample messageflow
    msgflow_path = "/mnt/project/messageflow_template_sample.xml"
    
    if os.path.exists(msgflow_path):
        try:
            has_before, has_after = generator._detect_enrichment_flags(msgflow_path)
            print(f"✅ Detection successful:")
            print(f"   Before Enrichment: {has_before}")
            print(f"   After Enrichment: {has_after}")
            return True
        except Exception as e:
            print(f"❌ Detection failed: {str(e)}")
            return False
    else:
        print(f"⚠️  Messageflow file not found: {msgflow_path}")
        return False

def test_template_loading():
    """Test 2: Template loading"""
    print("\n" + "="*60)
    print("TEST 2: Template Loading")
    print("="*60)
    
    generator = EnrichmentGenerator(groq_api_key="test_key")
    
    templates_found = []
    
    # Test Before template
    try:
        before_template = generator._load_enrichment_template('before')
        print(f"✅ Before template loaded successfully")
        print(f"   Structure: {list(before_template.keys())}")
        templates_found.append('before')
    except Exception as e:
        print(f"❌ Before template loading failed: {str(e)}")
    
    # Test After template
    try:
        after_template = generator._load_enrichment_template('after')
        print(f"✅ After template loaded successfully")
        print(f"   Structure: {list(after_template.keys())}")
        templates_found.append('after')
    except Exception as e:
        print(f"❌ After template loading failed: {str(e)}")
    
    return len(templates_found) == 2

def test_cleanup():
    """Test 3: Cleanup functionality"""
    print("\n" + "="*60)
    print("TEST 3: Cleanup Functionality")
    print("="*60)
    
    import tempfile
    import shutil
    
    generator = EnrichmentGenerator(groq_api_key="test_key")
    
    # Create temporary test directory
    test_dir = tempfile.mkdtemp()
    enrichment_dir = os.path.join(test_dir, 'enrichment')
    os.makedirs(enrichment_dir, exist_ok=True)
    
    # Create dummy file
    dummy_file = os.path.join(enrichment_dir, 'test.json')
    with open(dummy_file, 'w') as f:
        f.write('{"test": "data"}')
    
    print(f"   Created test enrichment folder: {enrichment_dir}")
    
    # Test cleanup
    try:
        generator._cleanup_enrichment_folder(test_dir)
        
        if not os.path.exists(enrichment_dir):
            print(f"✅ Cleanup successful - enrichment folder removed")
            shutil.rmtree(test_dir)
            return True
        else:
            print(f"❌ Cleanup failed - enrichment folder still exists")
            shutil.rmtree(test_dir)
            return False
    except Exception as e:
        print(f"❌ Cleanup error: {str(e)}")
        shutil.rmtree(test_dir)
        return False

def validate_implementation():
    """Validate key implementation aspects"""
    print("\n" + "="*60)
    print("VALIDATION: Implementation Check")
    print("="*60)
    
    generator = EnrichmentGenerator(groq_api_key="test_key")
    
    # Check required methods exist
    required_methods = [
        '_detect_enrichment_flags',
        '_cleanup_enrichment_folder',
        '_load_enrichment_template',
        '_llm_fill_enrichment_template',
        'generate_enrichment_files'
    ]
    
    all_present = True
    for method_name in required_methods:
        if hasattr(generator, method_name):
            print(f"✅ Method exists: {method_name}")
        else:
            print(f"❌ Method missing: {method_name}")
            all_present = False
    
    return all_present

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("ENRICHMENT GENERATOR v2.0 - TEST SUITE")
    print("="*60)
    
    results = {
        'Detection': test_enrichment_detection(),
        'Template Loading': test_template_loading(),
        'Cleanup': test_cleanup(),
        'Implementation': validate_implementation()
    }
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED")
    else:
        print("⚠️  SOME TESTS FAILED - Review output above")
    print("="*60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())