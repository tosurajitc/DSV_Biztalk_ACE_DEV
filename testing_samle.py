#!/usr/bin/env python3
"""
BizTalk Vector Processor - Usage Example & Test Script
Demonstrates how to use the BizTalkVectorProcessor to extract schema content

Author: ACE Migration Team
"""

import os
from pathlib import Path
from biztalk_vector_processor import BizTalkVectorProcessor


def test_biztalk_processor(folder_path: str):
    """
    Test the BizTalk Vector Processor with a given folder
    
    Args:
        folder_path: Path to BizTalk project folder
    """
    print("="*70)
    print(" BizTalk Vector Processor - Test Script")
    print("="*70)
    print()
    
    # Create processor instance
    processor = BizTalkVectorProcessor()
    
    print(f"✅ Processor initialized")
    print(f"   Supported file types: {processor.supported_extensions}")
    print()
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder does not exist: {folder_path}")
        return
    
    # Scan and extract
    print(f"🔍 Scanning folder: {folder_path}")
    print()
    
    chunks = processor.scan_and_extract(folder_path, max_depth=4)
    
    # Display results
    if chunks:
        print()
        print("="*70)
        print(f" RESULTS - {len(chunks)} schema files processed")
        print("="*70)
        print()
        
        # Show first chunk as example
        print("📄 Example Output (First Schema):")
        print("-"*70)
        print(chunks[0]['content'])
        print("-"*70)
        print()
        
        print("📊 Metadata:")
        print(f"   Source: {chunks[0]['metadata']['source']}")
        print(f"   Type: {chunks[0]['metadata']['type']}")
        print(f"   Elements: {chunks[0]['metadata']['element_count']}")
        print(f"   Namespace: {chunks[0]['metadata']['namespace']}")
        print()
        
        # Show statistics
        stats = processor.get_statistics()
        print("📈 Processing Statistics:")
        print(f"   Total chunks created: {stats['total_chunks']}")
        print(f"   Files processed: {stats['processed_files']}")
        print(f"   File types: {stats['file_types']}")
        print()
        
        print("📁 Processed Files:")
        for i, file in enumerate(stats['files'], 1):
            print(f"   {i}. {file}")
        
        print()
        print("="*70)
        print("✅ Test Completed Successfully!")
        print("="*70)
        
    else:
        print()
        print("⚠️ No schema files found in folder")
        print()


def demo_vector_db_integration():
    """
    Demonstrate how to integrate with Vector DB
    """
    print("\n" + "="*70)
    print(" Integration with Vector DB - Example")
    print("="*70)
    print()
    
    code_example = '''
# Example: Integrate BizTalk schemas into Vector DB

from biztalk_vector_processor import BizTalkVectorProcessor
from vector_store import ChromaVectorStore

# 1. Process BizTalk folder
processor = BizTalkVectorProcessor()
biztalk_chunks = processor.scan_and_extract("C:/BizTalk/Project")

# 2. Process PDF/Confluence (existing code)
pdf_chunks = process_pdf_file("requirements.pdf")

# 3. Combine chunks
all_chunks = pdf_chunks + biztalk_chunks

# 4. Create Vector DB
vector_store = ChromaVectorStore()
vector_store.create_knowledge_base(all_chunks)

print(f"✅ Vector DB created with {len(all_chunks)} total chunks")
print(f"   - Business requirements: {len(pdf_chunks)}")
print(f"   - BizTalk schemas: {len(biztalk_chunks)}")
'''
    
    print(code_example)
    print("="*70)
    print()


if __name__ == "__main__":
    print()
    
    # Test folder path - UPDATE THIS to your BizTalk project folder
    test_folder = "C:\@Official\@Gen AI\DSV\BizTalk\Analyze_this_folder\MH.ESB.EE.BR_PaymentSlips\MH.ESB.EE.BR_PaymentSlips"  # CHANGE THIS
    
    # Check if test folder exists
    if os.path.exists(test_folder):
        # Run test
        test_biztalk_processor(test_folder)
    else:
        print("📌 To test the processor:")
        print(f"   1. Update 'test_folder' variable in this script")
        print(f"   2. Point it to your BizTalk project folder")
        print(f"   3. Run: python test_biztalk_processor.py")
        print()
        print(f"   Current test folder (not found): {test_folder}")
        print()
    
    # Show integration example
    demo_vector_db_integration()
    
    print("✅ For more information, see: BIZTALK_INTEGRATION_GUIDE.md")
    print()