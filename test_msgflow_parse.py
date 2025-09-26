# test_msgflow_parse.py
from pathlib import Path
import xml.etree.ElementTree as ET
import re

def test_parse_msgflow(msgflow_file_path):
    """Test the msgflow parsing logic in isolation"""
    print(f"Testing: {msgflow_file_path}")
    print(f"File exists: {Path(msgflow_file_path).exists()}")
    
    compute_references = {}
    
    try:
        # Parse the XML
        tree = ET.parse(msgflow_file_path)
        root = tree.getroot()
        print(f"XML parsed successfully. Root tag: {root.tag}")
        
        # Count all elements
        all_elements = list(root.iter())
        print(f"Total XML elements found: {len(all_elements)}")
        
        # Look for computeExpression attributes
        elements_with_compute = []
        for element in root.iter():
            if 'computeExpression' in element.attrib:
                elements_with_compute.append(element)
                compute_expr = element.attrib['computeExpression']
                print(f"Found computeExpression: '{compute_expr}'")
                
                # Get node name from translation
                node_name = "Unknown"
                for child in element:
                    if 'translation' in child.tag:
                        node_name = child.attrib.get('string', 'Unknown')
                        break
                
                # Test NEW parsing logic
                if compute_expr and compute_expr.strip():
                    esql_module = compute_expr.strip()
                    esql_filename = f"{esql_module}.esql"
                    compute_references[esql_filename] = node_name
                    print(f"  -> Mapped: {esql_filename} -> {node_name}")
        
        print(f"\nTotal computeExpression attributes found: {len(elements_with_compute)}")
        print(f"Final compute_references mapping: {compute_references}")
        
        return compute_references
        
    except Exception as e:
        print(f"Error: {e}")
        return {}

if __name__ == "__main__":
    # Test with your actual msgflow file
    msgflow_path = "C:\@Official\@Gen AI\DSV\BizTalk\Analyze_this_folder\DSV_Biztalk_ACE\output\MessageFlow_AGENT2_App_Name/AGENT2_Message_Flow.msgflow"  # UPDATE THIS PATH
    
    result = test_parse_msgflow(msgflow_path)
    print(f"\nResult: {len(result)} references extracted")