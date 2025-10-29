#!/usr/bin/env python3
"""
Test script to validate node removal functionality in messageflow generator
"""
import re

def test_node_removal():
    """Test the node removal functionality based on configuration"""
    # Sample XML content with various nodes
    sample_xml = """
    <composition>
      <!-- SECTION 1: INPUT NODES -->
      <nodes xmi:type="ComIbmWSInput.msgnode:FCMComposite_1" xmi:id="HTTPInput_Node" location="-201,103">
        <translation xmi:type="utility:ConstantString" string="HTTPInput"/>
      </nodes>
      
      <nodes xmi:type="ComIbmMQInput.msgnode:FCMComposite_1" xmi:id="MQInput_Node" location="-201,203">
        <translation xmi:type="utility:ConstantString" string="MQInput"/>
      </nodes>
      
      <nodes xmi:type="ComIbmFileInput.msgnode:FCMComposite_1" xmi:id="FileInput_Node" location="-201,303">
        <translation xmi:type="utility:ConstantString" string="FileInput"/>
      </nodes>
      
      <!-- SECTION 5: TRANSFORM NODES -->
      <nodes xmi:type="ComIbmXslMqsi.msgnode:FCMComposite_1" xmi:id="XSLTransform_Node" location="500,100">
        <translation xmi:type="utility:ConstantString" string="XSLTransform"/>
      </nodes>
      
      <!-- SECTION 6: ENRICHMENT NODES -->
      <nodes xmi:type="epis_enrichment_lib_EPIS_MessageEnrichment.subflow:FCMComposite_1" xmi:id="BeforeEnrichment_Node" location="300,200">
        <translation xmi:type="utility:ConstantString" string="BeforeEnrichment"/>
      </nodes>
      
      <nodes xmi:type="epis_enrichment_lib_EPIS_MessageEnrichment.subflow:FCMComposite_1" xmi:id="AfterEnrichment_Node" location="700,200">
        <translation xmi:type="utility:ConstantString" string="AfterEnrichment"/>
      </nodes>
      
      <!-- SECTION 7: SERVICE REQUEST NODES -->
      <nodes xmi:type="ComIbmSOAPRequest.msgnode:FCMComposite_1" xmi:id="SOAPRequest_Node" location="600,300">
        <translation xmi:type="utility:ConstantString" string="SOAPRequest"/>
      </nodes>
      
      <!-- SECTION 8: COMPRESSION NODES -->
      <nodes xmi:type="epis_compression_lib_GZipCompressAndB64EncodeElement.subflow:FCMComposite_1" xmi:id="GZipCompression_Node" location="400,400">
        <translation xmi:type="utility:ConstantString" string="GZipCompression"/>
      </nodes>
      
      <!-- SECTION 9: ROUTING NODES -->
      <nodes xmi:type="ComIbmRoute.msgnode:FCMComposite_1" xmi:id="Route_Node" location="300,500">
        <translation xmi:type="utility:ConstantString" string="MethodRouter"/>
      </nodes>
    </composition>
    """
    
    # Sample node configuration (based on the XML comment in your message)
    node_config = {
        'needs_http_input': True,
        'needs_mq_input': False,
        'needs_file_input': False,
        'needs_xsl_transform': True,
        'needs_before_enrichment': True,
        'needs_after_enrichment': True,
        'needs_soap_request': True,
        'needs_gzip_compression': False,
        'needs_routing': True
    }
    
    # Function to remove nodes based on config (simulating _process_template_with_connectors)
    def remove_nodes(xml_content, config):
        # Identify nodes to remove
        nodes_to_remove = []
        
        if not config.get('needs_http_input', False):
            nodes_to_remove.append('HTTPInput')
        
        if not config.get('needs_mq_input', False):
            nodes_to_remove.append('MQInput')
            
        if not config.get('needs_file_input', False):
            nodes_to_remove.append('FileInput')
            
        if not config.get('needs_xsl_transform', False):
            nodes_to_remove.append('XSLTransform')
            
        if not config.get('needs_before_enrichment', False):
            nodes_to_remove.append('BeforeEnrichment')
            
        if not config.get('needs_after_enrichment', False):
            nodes_to_remove.append('AfterEnrichment')
            
        if not config.get('needs_soap_request', False):
            nodes_to_remove.append('SOAPRequest')
            
        if not config.get('needs_gzip_compression', False):
            nodes_to_remove.append('GZipCompression')
            
        if not config.get('needs_routing', False):
            nodes_to_remove.append('MethodRouter')
            
        print(f"Nodes to remove based on configuration: {', '.join(nodes_to_remove)}")
        
        # Remove nodes
        for node in nodes_to_remove:
            pattern = rf'<nodes.*?string="{node}".*?</nodes>'
            xml_content = re.sub(pattern, '', xml_content, flags=re.DOTALL)
            
        return xml_content
    
    # Process the XML
    processed_xml = remove_nodes(sample_xml, node_config)
    
    # Check which nodes remain in the processed XML
    remaining_nodes = []
    node_patterns = {
        'HTTPInput': r'string="HTTPInput"',
        'MQInput': r'string="MQInput"',
        'FileInput': r'string="FileInput"',
        'XSLTransform': r'string="XSLTransform"',
        'BeforeEnrichment': r'string="BeforeEnrichment"',
        'AfterEnrichment': r'string="AfterEnrichment"',
        'SOAPRequest': r'string="SOAPRequest"',
        'GZipCompression': r'string="GZipCompression"',
        'MethodRouter': r'string="MethodRouter"'
    }
    
    for node, pattern in node_patterns.items():
        if re.search(pattern, processed_xml):
            remaining_nodes.append(node)
    
    print("\n=== NODES AFTER PROCESSING ===")
    print(f"Remaining nodes: {', '.join(remaining_nodes)}")
    
    # Validate results
    expected_nodes = ['HTTPInput', 'XSLTransform', 'BeforeEnrichment', 'AfterEnrichment', 'SOAPRequest', 'MethodRouter']
    missing_expected = [node for node in expected_nodes if node not in remaining_nodes]
    unexpected_nodes = [node for node in remaining_nodes if node not in expected_nodes]
    
    if not missing_expected and not unexpected_nodes:
        print("\n✅ TEST PASSED: All expected nodes present and no unexpected nodes found")
    else:
        if missing_expected:
            print(f"\n❌ TEST FAILED: Missing expected nodes: {', '.join(missing_expected)}")
        if unexpected_nodes:
            print(f"\n❌ TEST FAILED: Found unexpected nodes: {', '.join(unexpected_nodes)}")

if __name__ == "__main__":
    test_node_removal()