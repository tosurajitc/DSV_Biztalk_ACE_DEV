"""
additional_ACE_components.py
Dynamically generates additional ACE components based on messageflow analysis
"""

import os
import xml.etree.ElementTree as ET
from typing import List, Dict, Any

class AdditionalComponentsGenerator:
    """
    Analyzes messageflow XML and generates additional components 
    beyond the standard 6 ESQL files
    """
    
    def __init__(self):
        self.standard_modules = [
            'InputEventMessage',
            'Compute',
            'AfterEnrichment',
            'OutputEventMessage',
            'AfterEventMsg',
            'Failure'
        ]
    
    def analyze_and_generate(self, msgflow_path: str, output_dir: str) -> Dict[str, List[str]]:
        """
        Analyze messageflow and generate additional components
        
        Returns:
            Dict with keys: 'esql', 'subflows', 'wsdl', etc.
        """
        print("🔍 Analyzing messageflow for additional components...")
        
        tree = ET.parse(msgflow_path)
        root = tree.getroot()
        
        additional_components = {
            'esql': [],
            'subflows': [],
            'wsdl': [],
            'additional_xsl': []
        }
        
        # Find all compute nodes
        for node in root.findall(".//{*}nodes[@xmi:type='ComIbmCompute.msgnode:FCMComposite_1']"):
            compute_expr = node.get('computeExpression', '')
            if compute_expr and 'esql://routine/#' in compute_expr:
                module_name = compute_expr.split('#')[1].split('.')[0]
                
                # Check if this is NOT a standard module
                if not any(std in module_name for std in self.standard_modules):
                    print(f"   ➕ Found additional ESQL module: {module_name}")
                    additional_components['esql'].append(module_name)
        
        # Find SOAP nodes that need WSDL
        for node in root.findall(".//{*}nodes[@xmi:type='ComIbmSOAPRequest.msgnode:FCMComposite_1']"):
            wsdl_file = node.get('wsdlFileName', '')
            if wsdl_file:
                print(f"   ➕ Found WSDL requirement: {wsdl_file}")
                additional_components['wsdl'].append(wsdl_file)
        
        # Find additional XSL transforms
        for node in root.findall(".//{*}nodes[@xmi:type='ComIbmXslMqsi.msgnode:FCMComposite_1']"):
            xsl_file = node.get('stylesheetName', '')
            if xsl_file and xsl_file != 'DefaultTransform.xsl':
                print(f"   ➕ Found additional XSL: {xsl_file}")
                additional_components['additional_xsl'].append(xsl_file)
        
        # Generate the additional components
        self._generate_additional_esql(additional_components['esql'], output_dir)
        self._generate_placeholder_wsdl(additional_components['wsdl'], output_dir)
        
        return additional_components
    
    def _generate_additional_esql(self, module_names: List[str], output_dir: str):
        """Generate additional ESQL modules (e.g., SOAPPreparation)"""
        esql_dir = os.path.join(output_dir, 'esql')
        os.makedirs(esql_dir, exist_ok=True)
        
        for module_name in module_names:
            esql_content = f"""/*
================================================================================
ADDITIONAL MODULE: {module_name}
Generated dynamically based on messageflow requirements
================================================================================
*/

BROKER SCHEMA {module_name.split('_')[0]}_{module_name.split('_')[1]}_{module_name.split('_')[2]}_{module_name.split('_')[3]}

CREATE COMPUTE MODULE {module_name}
    CREATE FUNCTION Main() RETURNS BOOLEAN
    BEGIN
        -- Additional processing logic
        -- This module was generated because it's referenced in the messageflow
        -- but not part of the standard 6 modules
        
        SET OutputRoot = InputRoot;
        
        RETURN TRUE;
    END;

    CREATE PROCEDURE CopyMessageHeaders() BEGIN
        DECLARE I INTEGER 1;
        DECLARE J INTEGER;
        SET J = CARDINALITY(InputRoot.*[]);
        WHILE I < J DO
            SET OutputRoot.*[I] = InputRoot.*[I];
            SET I = I + 1;
        END WHILE;
    END;

    CREATE PROCEDURE CopyEntireMessage() BEGIN
        SET OutputRoot = InputRoot;
    END;

END MODULE;
"""
            filepath = os.path.join(esql_dir, f"{module_name}.esql")
            with open(filepath, 'w') as f:
                f.write(esql_content)
            print(f"   ✅ Generated additional ESQL: {module_name}.esql")
    
    def _generate_placeholder_wsdl(self, wsdl_files: List[str], output_dir: str):
        """Generate placeholder WSDL files"""
        wsdl_dir = os.path.join(output_dir, 'wsdl')
        os.makedirs(wsdl_dir, exist_ok=True)
        
        for wsdl_file in wsdl_files:
            filepath = os.path.join(wsdl_dir, wsdl_file)
            # Create placeholder - user must replace with actual WSDL
            with open(filepath, 'w') as f:
                f.write(f"<!-- PLACEHOLDER: Replace with actual WSDL for {wsdl_file} -->")
            print(f"   ⚠️  Generated WSDL placeholder: {wsdl_file} (must be replaced)")


def main():
    """Test harness"""
    generator = AdditionalComponentsGenerator()
    results = generator.analyze_and_generate(
        msgflow_path="output/CW1_IN_Document_SND.msgflow",
        output_dir="output"
    )
    print(f"\n📊 Additional components generated: {results}")


if __name__ == "__main__":
    main()