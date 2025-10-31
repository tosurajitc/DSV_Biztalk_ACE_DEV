#!/usr/bin/env python3
"""
BizTalk Vector Processor
Extracts business-meaningful content from BizTalk files for Vector DB

Author: ACE Migration Team
Phase 1: Schema (.xsd) file processing
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET

class BizTalkVectorProcessor:
    """Extract business content from BizTalk files for Vector DB embedding"""
    
    def __init__(self):
        self.supported_extensions = {'.xsd'}  # Phase 1: Schemas only
        self.processed_files = []
        self.chunks = []
        
        # XML namespaces commonly used in BizTalk schemas
        self.namespaces = {
            'xs': 'http://www.w3.org/2001/XMLSchema',
            'xsd': 'http://www.w3.org/2001/XMLSchema'
        }
    
    def scan_and_extract(self, biztalk_folder: str, max_depth: int = 4) -> List[Dict]:
        """
        Main entry point - scan folder and extract business content
        
        Args:
            biztalk_folder: Path to BizTalk project folder
            max_depth: Maximum folder depth to scan (default: 4 levels)
            
        Returns:
            List of chunks ready for Vector DB:
            [
                {
                    'content': 'Business description text',
                    'metadata': {
                        'source': 'CustomerOrder.xsd',
                        'type': 'schema',
                        'file_path': '/path/to/file'
                    }
                },
                ...
            ]
        """
        print(f"\n{'='*60}")
        print(f"🔍 BizTalk Vector Processor - Starting Scan")
        print(f"{'='*60}")
        print(f"📁 Folder: {biztalk_folder}")
        print(f"📊 Max Depth: {max_depth} levels")
        print(f"🎯 Target Files: {', '.join(self.supported_extensions)}")
        print()
        
        self.chunks = []
        self.processed_files = []
        
        # Validate folder exists
        folder_path = Path(biztalk_folder)
        if not folder_path.exists():
            print(f"❌ Error: Folder does not exist: {biztalk_folder}")
            return []
        
        # Scan folder recursively
        schema_files = self._scan_folder(folder_path, max_depth)
        
        print(f"✅ Found {len(schema_files)} schema files")
        print()
        
        # Process each schema file
        for file_path in schema_files:
            try:
                chunk = self._extract_schema_content(file_path)
                if chunk:
                    self.chunks.append(chunk)
                    self.processed_files.append(str(file_path))
                    print(f"  ✅ {file_path.name}")
            except Exception as e:
                print(f"  ⚠️ {file_path.name}: {str(e)}")
                continue
        
        print()
        print(f"{'='*60}")
        print(f"📦 Processing Complete")
        print(f"{'='*60}")
        print(f"✅ Successfully processed: {len(self.chunks)} files")
        print(f"📝 Total chunks created: {len(self.chunks)}")
        print()
        
        return self.chunks
    
    def _scan_folder(self, folder_path: Path, max_depth: int) -> List[Path]:
        """
        Recursively scan folder for BizTalk files
        
        Args:
            folder_path: Starting folder
            max_depth: Maximum depth to traverse
            
        Returns:
            List of file paths
        """
        schema_files = []
        
        for root, dirs, files in os.walk(folder_path):
            # Calculate current depth
            relative_path = Path(root).relative_to(folder_path)
            current_depth = len(relative_path.parts)
            
            # Stop if max depth exceeded
            if current_depth >= max_depth:
                dirs.clear()  # Don't descend further
                continue
            
            # Find schema files in current directory
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in self.supported_extensions:
                    schema_files.append(file_path)
        
        return sorted(schema_files)
    
    def _extract_schema_content(self, file_path: Path) -> Optional[Dict]:
        """
        Extract business content from XSD schema file
        
        Args:
            file_path: Path to .xsd file
            
        Returns:
            Chunk dictionary with content and metadata
        """
        try:
            # Parse XML
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Extract schema information
            schema_name = file_path.stem
            namespace = root.get('targetNamespace', 'Not specified')
            
            # Extract elements
            elements = self._extract_elements(root)
            
            # Extract documentation
            documentation = self._extract_documentation(root)
            
            # Generate business-readable description
            content = self._generate_schema_description(
                schema_name=schema_name,
                file_name=file_path.name,
                namespace=namespace,
                elements=elements,
                documentation=documentation
            )
            
            # Create chunk
            chunk = {
                'content': content,
                'metadata': {
                    'source': file_path.name,
                    'type': 'schema',
                    'file_path': str(file_path),
                    'schema_name': schema_name,
                    'element_count': len(elements),
                    'namespace': namespace
                }
            }
            
            return chunk
            
        except Exception as e:
            raise Exception(f"Failed to parse schema: {str(e)}")
    
    def _extract_elements(self, root: ET.Element) -> List[Dict]:
        """
        Extract element definitions from schema
        
        Args:
            root: XML root element
            
        Returns:
            List of element dictionaries
        """
        elements = []
        
        # Try multiple namespace combinations
        ns_variations = [
            {'xs': 'http://www.w3.org/2001/XMLSchema'},
            {'xsd': 'http://www.w3.org/2001/XMLSchema'},
            {},  # No namespace
        ]
        
        for ns in ns_variations:
            # Find all element definitions
            if ns:
                element_tags = root.findall('.//xs:element', ns) or root.findall('.//xsd:element', ns)
            else:
                # Try without namespace
                element_tags = root.findall('.//{http://www.w3.org/2001/XMLSchema}element')
                if not element_tags:
                    element_tags = [elem for elem in root.iter() if elem.tag.endswith('element')]
            
            if element_tags:
                for elem in element_tags:
                    element_info = self._parse_element(elem, ns)
                    if element_info:
                        elements.append(element_info)
                break  # Found elements, stop trying
        
        return elements
    
    def _parse_element(self, elem: ET.Element, ns: Dict) -> Optional[Dict]:
        """
        Parse individual element definition
        
        Args:
            elem: Element XML node
            ns: Namespace dictionary
            
        Returns:
            Element info dictionary
        """
        name = elem.get('name')
        if not name:
            return None
        
        # Get type
        elem_type = elem.get('type', 'complex')
        if ':' in elem_type:
            elem_type = elem_type.split(':')[1]  # Remove namespace prefix
        
        # Get constraints
        min_occurs = elem.get('minOccurs', '1')
        max_occurs = elem.get('maxOccurs', '1')
        
        # Determine if required
        is_required = min_occurs != '0'
        is_array = max_occurs == 'unbounded' or (max_occurs != '1' and max_occurs.isdigit() and int(max_occurs) > 1)
        
        # Get documentation
        doc = None
        if ns:
            annotation = elem.find('xs:annotation/xs:documentation', ns) or elem.find('xsd:annotation/xsd:documentation', ns)
            if annotation is not None and annotation.text:
                doc = annotation.text.strip()
        
        return {
            'name': name,
            'type': elem_type,
            'required': is_required,
            'array': is_array,
            'documentation': doc
        }
    
    def _extract_documentation(self, root: ET.Element) -> Optional[str]:
        """
        Extract schema-level documentation
        
        Args:
            root: XML root element
            
        Returns:
            Documentation text or None
        """
        # Try different namespace variations
        for ns_prefix in ['xs', 'xsd']:
            ns = {ns_prefix: 'http://www.w3.org/2001/XMLSchema'}
            doc_elem = root.find(f'{ns_prefix}:annotation/{ns_prefix}:documentation', ns)
            if doc_elem is not None and doc_elem.text:
                return doc_elem.text.strip()
        
        # Try without namespace
        for elem in root.iter():
            if elem.tag.endswith('documentation') and elem.text:
                return elem.text.strip()
        
        return None
    
    def _generate_schema_description(self, schema_name: str, file_name: str, 
                                    namespace: str, elements: List[Dict], 
                                    documentation: Optional[str]) -> str:
        """
        Generate human-readable business description of schema
        
        Args:
            schema_name: Name of schema
            file_name: File name
            namespace: Target namespace
            elements: List of element definitions
            documentation: Schema documentation
            
        Returns:
            Business-readable text description
        """
        lines = []
        
        # Header
        lines.append(f"Schema: {schema_name}")
        lines.append(f"File: {file_name}")
        lines.append(f"Type: XML Schema Definition (XSD)")
        lines.append("")
        
        # Documentation
        if documentation:
            lines.append(f"Purpose: {documentation}")
            lines.append("")
        
        # Elements summary
        if elements:
            lines.append(f"Data Structure ({len(elements)} elements):")
            lines.append("")
            
            for elem in elements[:20]:  # Limit to first 20 elements
                # Build element description
                desc_parts = [f"- {elem['name']}"]
                
                # Add type
                desc_parts.append(f"({elem['type']})")
                
                # Add constraints
                constraints = []
                if elem['required']:
                    constraints.append("required")
                else:
                    constraints.append("optional")
                
                if elem['array']:
                    constraints.append("array/multiple values")
                
                if constraints:
                    desc_parts.append(f"[{', '.join(constraints)}]")
                
                # Add documentation if available
                if elem['documentation']:
                    desc_parts.append(f": {elem['documentation']}")
                
                lines.append(" ".join(desc_parts))
            
            if len(elements) > 20:
                lines.append(f"... and {len(elements) - 20} more elements")
            
            lines.append("")
        
        # Technical details
        lines.append("Technical Information:")
        lines.append(f"- Namespace: {namespace}")
        lines.append(f"- Element Count: {len(elements)}")
        lines.append("")
        
        # Usage context
        lines.append("Usage Context:")
        lines.append(f"This schema defines the structure for {schema_name} messages used in BizTalk integration. ")
        lines.append(f"The schema contains {len(elements)} data elements that specify the format and constraints ")
        lines.append("for messages exchanged between systems.")
        
        return "\n".join(lines)
    
    def get_statistics(self) -> Dict:
        """
        Get processing statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            'total_chunks': len(self.chunks),
            'processed_files': len(self.processed_files),
            'file_types': list(self.supported_extensions),
            'files': [Path(f).name for f in self.processed_files]
        }


# Test function
if __name__ == "__main__":
    # Example usage
    processor = BizTalkVectorProcessor()
    
    # Test with a folder path
    test_folder = "/path/to/biztalk/project"
    
    if os.path.exists(test_folder):
        chunks = processor.scan_and_extract(test_folder)
        
        print("\n=== Sample Output ===")
        if chunks:
            print(chunks[0]['content'])
            print("\nMetadata:", chunks[0]['metadata'])
        
        stats = processor.get_statistics()
        print("\n=== Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
    else:
        print(f"Test folder not found: {test_folder}")
        print("Please update test_folder path to test the processor")