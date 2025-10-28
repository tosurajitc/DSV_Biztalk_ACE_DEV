#!/usr/bin/env python3
"""
Test script for verifying _extract_node_configuration functionality
"""

from typing import Dict, Optional
import json
import sys

def test_extract_node_configuration(business_reqs: Dict, flow_name: str) -> Dict:
    """
    Test version of _extract_node_configuration to verify it works correctly
    
    Args:
        business_reqs: Business requirements data
        flow_name: Name of the flow
        
    Returns:
        Node configuration dictionary
    """
    # Initialize with all nodes disabled
    node_config = {
        'needs_http_input': False,
        'needs_http_reply': False,
        'needs_mq_input': False,
        'needs_mq_output': False,
        'needs_file_input': False,
        'needs_file_output': False,
        'needs_soap_request': False,
        'needs_xsl_transform': False,
        'needs_before_enrichment': False,
        'needs_after_enrichment': False,
        'needs_gzip_compression': False,
        'needs_routing': False,
        'http_config': {},
        'soap_config': {}
    }
    
    try:
        print(f"Analyzing business requirements for flow: {flow_name}")
        
        # 1. Determine input method from business requirements
        input_methods = business_reqs.get('integration_flows', {}).get('input_methods', [])
        input_protocol = None
        
        print(f"Input methods found: {input_methods}")
        
        # Check for specific input protocols in input_methods
        for input_method in input_methods:
            if isinstance(input_method, dict):
                method = input_method.get('method', '').upper()
                protocol = input_method.get('protocol', '').upper()
                
                print(f"Checking input method: method={method}, protocol={protocol}")
                
                if protocol == 'HTTP' or method == 'HTTP' or method == 'REST':
                    node_config['needs_http_input'] = True
                    node_config['needs_http_reply'] = True
                    input_protocol = 'HTTP'
                    print("  - Detected HTTP input protocol")
                elif protocol == 'MQ' or method == 'MQ' or method == 'QUEUE':
                    node_config['needs_mq_input'] = True
                    input_protocol = 'MQ'
                    print("  - Detected MQ input protocol")
                elif protocol == 'FILE' or method == 'FILE':
                    node_config['needs_file_input'] = True
                    input_protocol = 'FILE'
                    print("  - Detected File input protocol")
        
        # 2. Check flow patterns for additional protocol hints
        flow_patterns = business_reqs.get('technical_specs', {}).get('message_flow_patterns', [])
        print(f"Flow patterns found: {flow_patterns}")
        
        for pattern in flow_patterns:
            if isinstance(pattern, str):
                if 'HTTP' in pattern.upper() or 'REST' in pattern.upper() or 'WEB' in pattern.upper():
                    node_config['needs_http_input'] = True
                    node_config['needs_http_reply'] = True
                    input_protocol = 'HTTP'
                    print("  - Detected HTTP input from flow pattern")
                elif 'MQ' in pattern.upper() or 'QUEUE' in pattern.upper() or 'JMS' in pattern.upper():
                    node_config['needs_mq_input'] = True 
                    input_protocol = 'MQ'
                    print("  - Detected MQ input from flow pattern")
                elif 'FILE' in pattern.upper() or 'DIRECTORY' in pattern.upper():
                    node_config['needs_file_input'] = True
                    input_protocol = 'FILE'
                    print("  - Detected File input from flow pattern")
        
        # 3. Check for XSL Transform indicators
        for section in ['technical_specs', 'integration_flows']:
            section_data = business_reqs.get(section, {})
            print(f"Checking section '{section}' for transformation indicators")
            
            # Convert to string for pattern matching
            section_str = str(section_data).lower()
            
            if 'transform' in section_str or 'xslt' in section_str or 'stylesheet' in section_str or 'mapping' in section_str:
                node_config['needs_xsl_transform'] = True
                print("  - Detected XSL Transform requirement")
        
        # 4. Check for Enrichment indicators
        enrichment_str = str(business_reqs).lower()
        if 'enrich before' in enrichment_str or 'before enrich' in enrichment_str:
            node_config['needs_before_enrichment'] = True
            print("  - Detected Before Enrichment requirement")
        
        if 'enrich after' in enrichment_str or 'after enrich' in enrichment_str:
            node_config['needs_after_enrichment'] = True
            print("  - Detected After Enrichment requirement")
        
        # 5. Check for SOAP indicators
        soap_str = str(business_reqs).lower()
        if 'soap' in soap_str or 'web service' in soap_str:
            node_config['needs_soap_request'] = True
            print("  - Detected SOAP Request requirement")
        
        # 6. Check for Compression indicators
        compression_str = str(business_reqs).lower()
        if 'compress' in compression_str or 'gzip' in compression_str:
            node_config['needs_gzip_compression'] = True
            print("  - Detected GZip Compression requirement")
        
        # 7. Check for Routing indicators
        routing_str = str(business_reqs).lower()
        if 'route' in routing_str or 'branch' in routing_str or 'decision' in routing_str:
            node_config['needs_routing'] = True
            print("  - Detected Routing requirement")
        
        # 8. Determine output method based on input method and explicit requirements
        output_methods = business_reqs.get('integration_flows', {}).get('output_systems', [])
        output_protocol = None
        
        print(f"Output methods found: {output_methods}")
        
        # Check for explicit output protocol specifications
        for output in output_methods:
            if isinstance(output, dict):
                method = output.get('method', '').upper()
                protocol = output.get('protocol', '').upper()
                
                print(f"Checking output method: method={method}, protocol={protocol}")
                
                if protocol == 'HTTP' or method == 'HTTP' or method == 'REST':
                    output_protocol = 'HTTP'
                    print("  - Detected HTTP output protocol")
                elif protocol == 'MQ' or method == 'MQ' or method == 'QUEUE':
                    output_protocol = 'MQ'
                    print("  - Detected MQ output protocol")
                elif protocol == 'FILE' or method == 'FILE':
                    output_protocol = 'FILE'
                    print("  - Detected File output protocol")
        
        # If no explicit output protocol, use the same as input for consistency
        if not output_protocol and input_protocol:
            output_protocol = input_protocol
            print(f"  - Using input protocol '{input_protocol}' for output")
        
        # Set output nodes based on determined output protocol
        if output_protocol == 'HTTP':
            node_config['needs_http_reply'] = True
            node_config['needs_mq_output'] = False
            node_config['needs_file_output'] = False
        elif output_protocol == 'MQ':
            node_config['needs_mq_output'] = True
            node_config['needs_http_reply'] = False
            node_config['needs_file_output'] = False
        elif output_protocol == 'FILE':
            node_config['needs_file_output'] = True
            node_config['needs_mq_output'] = False
            node_config['needs_http_reply'] = False
        
        # 9. Configure HTTP URL if needed
        if node_config['needs_http_input']:
            node_config['http_config'] = {
                'url_path': f'/services/{flow_name}',
                'http_method': 'POST'
            }
        
        # 10. Verify at least one input method is set
        if not (node_config['needs_http_input'] or node_config['needs_mq_input'] or node_config['needs_file_input']):
            print("⚠️ WARNING: No input method detected! Defaulting to MQ input.")
            node_config['needs_mq_input'] = True
            node_config['needs_mq_output'] = True
        
        return node_config
        
    except Exception as e:
        print(f"❌ ERROR in extraction: {str(e)}")
        # Return default configuration with basic nodes enabled
        return {
            'needs_http_input': False,
            'needs_http_reply': False,
            'needs_mq_input': True,  # Default to MQ
            'needs_mq_output': True,
            'needs_file_input': False,
            'needs_file_output': False,
            'needs_soap_request': False,
            'needs_xsl_transform': False,
            'needs_before_enrichment': False,
            'needs_after_enrichment': False,
            'needs_gzip_compression': False,
            'needs_routing': False
        }

def main():
    # Test with sample business requirements
    sample_business_reqs = {
        "technical_specs": {
            "message_flow_patterns": [
                "HTTP REST API with transformation",
                "MQ queue processing with SOAP integration"
            ],
            "routing_logic": ["Method-based routing for different service operations"],
            "data_transformation_points": ["XML to JSON transformation using XSLT"],
            "error_handling_patterns": ["Standard error flow with message logging"],
            "performance_requirements": ["Low latency processing"]
        },
        "integration_flows": {
            "input_systems": ["SAP ECC", "CRM"],
            "input_methods": [
                {"method": "MQ", "protocol": "MQ", "details": "IBM MQ queue input"},
                {"method": "HTTP", "protocol": "HTTP", "details": "REST API endpoint"}
            ],
            "output_systems": [
                {"system": "Edicom", "protocol": "SOAP", "details": "Web Service integration"}
            ],
            "integration_patterns": ["Transform and Enrich before sending to service"]
        }
    }
    
    # Test with different flow names
    test_flows = ["HTTPFlow", "MQFlow", "FileFlow", "SAP_Data_Edicom_RTS"]
    
    for flow in test_flows:
        print("\n" + "="*50)
        print(f"TESTING FLOW: {flow}")
        print("="*50)
        
        config = test_extract_node_configuration(sample_business_reqs, flow)
        
        print("\nEXTRACTED CONFIGURATION:")
        print("-"*30)
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    # Test with minimal business requirements
    minimal_reqs = {
        "technical_specs": {
            "message_flow_patterns": []
        },
        "integration_flows": {}
    }
    
    print("\n" + "="*50)
    print(f"TESTING WITH MINIMAL REQUIREMENTS")
    print("="*50)
    
    config = test_extract_node_configuration(minimal_reqs, "MinimalFlow")
    
    print("\nEXTRACTED CONFIGURATION:")
    print("-"*30)
    for key, value in config.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()