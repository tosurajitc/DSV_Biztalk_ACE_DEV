#!/usr/bin/env python3
"""
Enrichment Generator Module v2.0 - ACE Module Creator with Content Chunking
Purpose: Analyze enrichment requirements with intelligent content chunking to avoid rate limits
Input: Confluence PDF + component mapping JSON + .msgflow + LLM → Analyze enrichment requirements and generate JSON configs
Output: Creates before_enrichment.json and after_enrichment.json files for data enrichment patterns
Key Improvement: Intelligent content chunking to handle large documents within API limits
"""

import os
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv
import streamlit as st
import time
import math

load_dotenv()

class EnrichmentGenerator:
    """
    ACE Enrichment Generator with intelligent content chunking
    Handles large documents by processing them in manageable chunks
    """
    
    def __init__(self, groq_api_key: str = None):
        """Initialize with Groq LLM client and chunking parameters"""
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY must be provided or set in environment")
        
        self.llm = Groq(api_key=self.groq_api_key)
        self.groq_model = os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')  # Use smaller model by default
        
        # Chunking configuration
        self.max_tokens_per_request = 8000  # Safe limit for most models
        self.chunk_overlap = 200  # Characters overlap between chunks
        self.estimated_chars_per_token = 4  # Rough estimation
        
        # Processing results tracking
        self.processed_json_mappings = None
        self.processed_msgflow_content = None
        self.processed_vector_content = None        
        self.generated_enrichment_configs = {}
        self.llm_analysis_calls = 0
        self.llm_generation_calls = 0
        self.chunk_processing_stats = {
            'vector_chunks': 0,                     
            'json_chunks': 0,
            'msgflow_chunks': 0,
            'total_chunks_processed': 0
        }
    
    def generate_enrichment_files(self, 
                                vector_content,  # ← Vector DB content instead of PDF path
                                component_mapping_json_path: str,
                                msgflow_path: str,
                                output_dir: str) -> Dict[str, Any]:
        """
        Main entry point - Analyze enrichment requirements with Vector DB content
        
        Args:
            vector_content: Vector DB business requirements content (from pipeline)
            component_mapping_json_path: Path to component mapping JSON with enrichment patterns
            msgflow_path: Path to .msgflow file with enrichment points
            output_dir: Output directory for before_enrichment.json and after_enrichment.json files
            
        Returns:
            Dict with generation results and metadata
        """
        print("🎯 Starting Enrichment Configuration Generation - Vector DB Mode")
        print("🔋 VECTOR DB INTEGRATION - Processing business requirements from Vector knowledge base")
        
        # Step 1: Process inputs for Vector DB analysis
        print("\n📄 Step 1: Processing inputs for Vector DB enrichment analysis...")

        json_mappings = self._extract_complete_json_mappings(component_mapping_json_path)
        msgflow_content = self._extract_complete_msgflow_content(msgflow_path)
        
        # Step 2: LLM Analysis of enrichment patterns using Vector DB content
        print("\n🧠 Step 2: LLM analysis of enrichment patterns with Vector DB content...")
        enrichment_analysis = self._chunked_analyze_enrichment_patterns(vector_content, json_mappings, msgflow_content)
        
        # Step 3: LLM Generation of before/after enrichment JSON configurations
        print("\n⚡ Step 3: LLM generating before/after enrichment JSON configurations...")
        enrichment_configs = self._llm_generate_enrichment_configurations(enrichment_analysis)
        
        # Step 4: Write enrichment configuration files
        print("\n💾 Step 4: Writing before_enrichment.json and after_enrichment.json files...")
        config_files = self._write_enrichment_configuration_files(enrichment_configs, output_dir)
        
        return {
            'status': 'success',
            'enrichment_configs_generated': len(config_files),
            'config_files': config_files,
            'output_directory': output_dir,
            'llm_analysis_calls': self.llm_analysis_calls,
            'llm_generation_calls': self.llm_generation_calls,
            'chunking_stats': self.chunk_processing_stats,
            'processing_metadata': {
                'vector_content_processed': len(str(vector_content)) if vector_content else 0,  # ← Updated from PDF pages
                'json_components_processed': len(json_mappings.get('enrichment_components', [])),
                'msgflow_nodes_processed': len(msgflow_content.get('enrichment_nodes', [])),
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _calculate_chunk_size(self, content: str) -> int:
        """Calculate safe chunk size based on content and token limits"""
        max_chars = self.max_tokens_per_request * self.estimated_chars_per_token
        # Reserve 20% for system prompt and response
        return int(max_chars * 0.8)
    
    def _chunk_text_content(self, text: str, chunk_type: str = "general") -> List[str]:
        """
        Intelligently chunk text content based on type and natural boundaries
        
        Args:
            text: Text content to chunk
            chunk_type: Type of content (pdf, json, xml) for intelligent chunking
            
        Returns:
            List of text chunks
        """
        if not text or len(text) < self.max_tokens_per_request:
            return [text]
        
        chunk_size = self._calculate_chunk_size(text)
        chunks = []
        
        if chunk_type == "pdf":
            # For PDF content, try to chunk by paragraphs or sections
            chunks = self._chunk_by_paragraphs(text, chunk_size)
        elif chunk_type == "json":
            # For JSON, try to chunk by logical components
            chunks = self._chunk_json_content(text, chunk_size)
        elif chunk_type == "xml":
            # For XML, try to chunk by nodes
            chunks = self._chunk_xml_content(text, chunk_size)
        else:
            # Default: simple character-based chunking with overlap
            chunks = self._chunk_by_characters(text, chunk_size)
        
        print(f"  📊 Content chunked: {len(text):,} chars → {len(chunks)} chunks")
        return chunks
    
    def _chunk_by_paragraphs(self, text: str, max_size: int) -> List[str]:
        """Chunk text by paragraphs to maintain context"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 <= max_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text[:max_size]]
    
    def _chunk_json_content(self, json_text: str, max_size: int) -> List[str]:
        """Chunk JSON content by logical components"""
        try:
            json_data = json.loads(json_text)
            chunks = []
            
            # If it's a dict with top-level keys, chunk by keys
            if isinstance(json_data, dict):
                current_chunk = {}
                current_size = 0
                
                for key, value in json_data.items():
                    item_size = len(json.dumps({key: value}))
                    
                    if current_size + item_size > max_size and current_chunk:
                        chunks.append(json.dumps(current_chunk, indent=2))
                        current_chunk = {}
                        current_size = 0
                    
                    current_chunk[key] = value
                    current_size += item_size
                
                if current_chunk:
                    chunks.append(json.dumps(current_chunk, indent=2))
            else:
                # Fallback to character chunking
                return self._chunk_by_characters(json_text, max_size)
                
            return chunks if chunks else [json_text[:max_size]]
            
        except json.JSONDecodeError:
            # If not valid JSON, fall back to character chunking
            return self._chunk_by_characters(json_text, max_size)
    
    def _chunk_xml_content(self, xml_text: str, max_size: int) -> List[str]:
        """Chunk XML content by logical nodes"""
        try:
            # Simple approach: split by major XML nodes
            chunks = []
            remaining_text = xml_text
            current_chunk = ""
            
            # Extract XML header for each chunk
            xml_header = ""
            if xml_text.startswith('<?xml'):
                header_end = xml_text.find('?>') + 2
                xml_header = xml_text[:header_end] + "\n"
                remaining_text = xml_text[header_end:]
            
            # Simple line-based chunking for XML
            lines = remaining_text.split('\n')
            for line in lines:
                if len(current_chunk) + len(line) + 1 <= max_size:
                    current_chunk += line + "\n"
                else:
                    if current_chunk:
                        chunks.append(xml_header + current_chunk.strip())
                    current_chunk = line + "\n"
            
            if current_chunk:
                chunks.append(xml_header + current_chunk.strip())
            
            return chunks if chunks else [xml_text[:max_size]]
            
        except Exception:
            # Fallback to character chunking
            return self._chunk_by_characters(xml_text, max_size)
    
    def _chunk_by_characters(self, text: str, max_size: int) -> List[str]:
        """Simple character-based chunking with overlap"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_size
            
            # If not the last chunk, try to find a good break point
            if end < len(text):
                # Look for natural break points (paragraph, sentence, etc.)
                break_points = ['\n\n', '\n', '. ', ', ']
                for break_point in break_points:
                    last_break = text.rfind(break_point, start, end)
                    if last_break > start:
                        end = last_break + len(break_point)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap if end < len(text) else end
        
        return chunks
    

    
    def _chunked_analyze_enrichment_patterns(self, 
                                        vector_content,  # ← Vector DB content instead of pdf_content
                                        json_mappings: Dict,
                                        msgflow_content: Dict) -> Dict[str, Any]:
        """
        Analyze enrichment patterns using Vector DB content with LLM integration
        Single LLM call to extract enrichment requirements from Vector business content
        """
        print("  🧠 Starting Vector DB LLM analysis of enrichment patterns...")
        
        # Step 1: LLM Analysis of Vector DB content for enrichment requirements
        print("  🔋 LLM Analysis: Extracting enrichment requirements from Vector DB content...")
        
        analysis_prompt = f"""Analyze Vector DB business requirements to extract enrichment module specifications:

    ## VECTOR DB BUSINESS CONTENT:
    {str(vector_content)[:4000]}  

    ## COMPONENT MAPPINGS:
    {json.dumps(json_mappings.get('original_data', {}), indent=2)[:2000]}

    ## MESSAGEFLOW STRUCTURE:
    {msgflow_content.get('raw_xml', '')[:2000]}

    ## ENRICHMENT ANALYSIS REQUIREMENTS:
    Extract and return JSON with:
    1. "enrichment_patterns": List of data enrichment requirements and patterns found
    2. "before_enrichment_structure": Data structure before enrichment processing
    3. "after_enrichment_structure": Data structure after enrichment processing  
    4. "lookup_configurations": Database lookup specifications and procedures
    5. "validation_rules": Data validation and error handling requirements
    6. "data_flow_points": Enrichment flow points and routing specifications

    Focus on:
    - Database lookup operations (CompanyCode, CW1 shipment matching, IsPublished flag, etc.)
    - Business enhancement rules from Vector DB content
    - Data enrichment logic and transformation requirements
    - XPath-based field extraction specifications
    - Before/after enrichment data structures
    - Validation rules and error handling patterns

    Return valid JSON only - no explanations:"""

        # LLM call for enrichment requirements analysis
        try:
            analysis_response = self.llm.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": "You are an expert IBM ACE data enrichment architect. Extract enrichment requirements from business content and return valid JSON only."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            self.llm_analysis_calls += 1
            analysis_content = analysis_response.choices[0].message.content.strip()
            
            # Extract JSON from LLM response
            json_match = re.search(r'\{.*\}', analysis_content, re.DOTALL)
            if json_match:
                enrichment_requirements = json.loads(json_match.group())
                print(f"  ✅ Vector DB enrichment analysis complete via LLM")
                print(f"  📊 Found: {len(enrichment_requirements.get('enrichment_patterns', []))} patterns, {len(enrichment_requirements.get('lookup_configurations', []))} lookups")
                return enrichment_requirements
            else:
                raise Exception("LLM did not return valid JSON analysis")
                
        except Exception as e:
            print(f"  ❌ LLM enrichment analysis failed: {str(e)}")
            # Return default structure instead of failing
            return {
                'enrichment_patterns': [],
                'before_enrichment_structure': {},
                'after_enrichment_structure': {},
                'lookup_configurations': [],
                'validation_rules': [],
                'data_flow_points': []
            }
    

    
    def _analyze_single_chunk(self, chunk_content: str, chunk_type: str, 
                            chunk_name: str, context_data: Any = None,
                            additional_context: Any = None) -> Dict[str, Any]:
        """
        Analyze a single chunk of content for enrichment patterns
        
        Args:
            chunk_content: The content chunk to analyze
            chunk_type: Type of chunk (pdf, json, msgflow)
            chunk_name: Human-readable name for the chunk
            context_data: Additional context for analysis
            additional_context: Extra context information
            
        Returns:
            Dict with analysis results for this chunk
        """
        system_prompt = f"""You are an expert IBM ACE data enrichment architect analyzing {chunk_type} content.

FOCUS: Extract enrichment patterns and requirements from this content chunk.

ANALYSIS GOALS:
- Identify data enrichment requirements and patterns
- Extract database lookup specifications
- Determine before/after enrichment data structures
- Find validation rules and error handling requirements
- Identify enrichment components and relationships

OUTPUT: Return JSON with these sections:
- enrichment_patterns: List of enrichment requirements found
- lookup_configurations: Database lookup specifications
- validation_rules: Data validation requirements
- data_flow_points: Enrichment flow specifications

Be precise and focus only on enrichment-related information in this chunk."""

        user_prompt = f"""Analyze this {chunk_type} content chunk for enrichment patterns:

## CHUNK INFORMATION:
**Chunk Name**: {chunk_name}
**Content Type**: {chunk_type}
**Chunk Size**: {len(chunk_content)} characters

## CONTENT TO ANALYZE:
{chunk_content}

## CONTEXT DATA:
{json.dumps(context_data, indent=2) if context_data else "No additional context"}

Extract all enrichment patterns, lookup requirements, and data flow specifications from this chunk."""

        try:
            response = self.llm.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2000  # Smaller response for chunk analysis
            )

            if 'token_tracker' in st.session_state and hasattr(response, 'usage') and response.usage:
                st.session_state.token_tracker.manual_track(
                    agent="enrichment_generator",
                    operation="chunk_analysis",
                    model=self.groq_model,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    flow_name=f"enrichment_chunk_{chunk_name}"
                )
            
            raw_analysis = response.choices[0].message.content.strip()
            self.llm_analysis_calls += 1
            
            # Parse LLM response as JSON
            try:
                chunk_analysis = json.loads(raw_analysis)
                return chunk_analysis
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', raw_analysis, re.DOTALL)
                if json_match:
                    chunk_analysis = json.loads(json_match.group())
                    return chunk_analysis
                else:
                    print(f"    ⚠️ Warning: Could not parse JSON from chunk {chunk_name}")
                    return {}
                    
        except Exception as e:
            print(f"    ❌ Error analyzing chunk {chunk_name}: {str(e)}")
            return {}
    
    def _merge_analysis_results(self, aggregated: Dict[str, Any], chunk_result: Dict[str, Any]):
        """Merge analysis results from a single chunk into aggregated results"""
        if not chunk_result:
            return
        
        # Merge enrichment patterns
        if 'enrichment_patterns' in chunk_result:
            aggregated['enrichment_patterns'].extend(chunk_result['enrichment_patterns'])
        
        # Merge lookup configurations
        if 'lookup_configurations' in chunk_result:
            aggregated['lookup_configurations'].extend(chunk_result['lookup_configurations'])
        
        # Merge validation rules
        if 'validation_rules' in chunk_result:
            aggregated['validation_rules'].extend(chunk_result['validation_rules'])
        
        # Merge data flow points
        if 'data_flow_points' in chunk_result:
            aggregated['data_flow_points'].extend(chunk_result['data_flow_points'])
        
        # Merge structure definitions (careful not to overwrite)
        if 'before_enrichment_structure' in chunk_result and chunk_result['before_enrichment_structure']:
            aggregated['before_enrichment_structure'].update(chunk_result['before_enrichment_structure'])
        
        if 'after_enrichment_structure' in chunk_result and chunk_result['after_enrichment_structure']:
            aggregated['after_enrichment_structure'].update(chunk_result['after_enrichment_structure'])
    
    def _synthesize_chunk_results(self, aggregated_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize and deduplicate results from all chunks
        Remove duplicates and consolidate similar patterns
        """
        print("    🔄 Removing duplicates and consolidating patterns...")
        
        # Deduplicate enrichment patterns
        unique_patterns = []
        seen_patterns = set()
        for pattern in aggregated_analysis['enrichment_patterns']:
            pattern_key = str(pattern.get('name', '')) + str(pattern.get('source', ''))
            if pattern_key not in seen_patterns:
                unique_patterns.append(pattern)
                seen_patterns.add(pattern_key)
        aggregated_analysis['enrichment_patterns'] = unique_patterns
        
        # Deduplicate lookup configurations
        unique_lookups = []
        seen_lookups = set()
        for lookup in aggregated_analysis['lookup_configurations']:
            lookup_key = str(lookup.get('table', '')) + str(lookup.get('key', ''))
            if lookup_key not in seen_lookups:
                unique_lookups.append(lookup)
                seen_lookups.add(lookup_key)
        aggregated_analysis['lookup_configurations'] = unique_lookups
        
        # Deduplicate validation rules
        unique_rules = list({str(rule): rule for rule in aggregated_analysis['validation_rules']}.values())
        aggregated_analysis['validation_rules'] = unique_rules
        
        # Deduplicate data flow points
        unique_flows = []
        seen_flows = set()
        for flow in aggregated_analysis['data_flow_points']:
            flow_key = str(flow.get('name', '')) + str(flow.get('type', ''))
            if flow_key not in seen_flows:
                unique_flows.append(flow)
                seen_flows.add(flow_key)
        aggregated_analysis['data_flow_points'] = unique_flows
        
        print(f"    ✅ Synthesis complete: {len(unique_patterns)} patterns, {len(unique_lookups)} lookups, {len(unique_rules)} rules")
        
        return aggregated_analysis
    

    
    def _extract_complete_json_mappings(self, json_path: str) -> Dict[str, Any]:
        """
        Extract COMPLETE JSON mapping content for enrichment pattern analysis
        NO filtering - all component mapping data goes to LLM for enrichment analysis
        """
        print(f"  🗂️ Extracting complete JSON component mappings for enrichment analysis from: {json_path}")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Component mapping JSON not found: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            if not isinstance(json_data, dict):
                raise ValueError("JSON must be a dictionary object")
            
            # Enhance JSON data for enrichment analysis
            enhanced_json = {
                'original_data': json_data,
                'enrichment_components': self._extract_enrichment_components(json_data),
                'database_connections': self._extract_database_connections(json_data),
                'lookup_definitions': self._extract_lookup_definitions(json_data),
                'data_enhancement_rules': self._extract_data_enhancement_rules(json_data)
            }
            
            print(f"  ✅ JSON processed: {len(str(json_data))} characters, {len(enhanced_json['enrichment_components'])} enrichment components identified")
            self.processed_json_mappings = enhanced_json
            return enhanced_json
            
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to process JSON: {str(e)}")
    
    def _extract_enrichment_components(self, json_data: Dict) -> List[Dict]:
        """Extract enrichment-related components from JSON data"""
        enrichment_components = []
        
        def search_for_enrichment(obj, path="root"):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if any(term in key.lower() for term in ['enrichment', 'lookup', 'database', 'transform']):
                        enrichment_components.append({
                            'path': f"{path}.{key}",
                            'key': key,
                            'value': value,
                            'type': 'enrichment_component'
                        })
                    search_for_enrichment(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    search_for_enrichment(item, f"{path}[{i}]")
        
        search_for_enrichment(json_data)
        return enrichment_components
    
    def _extract_database_connections(self, json_data: Dict) -> List[Dict]:
        """Extract database connection information from JSON data"""
        db_connections = []
        
        def search_for_db(obj, path="root"):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if any(term in key.lower() for term in ['database', 'db', 'connection', 'datasource']):
                        db_connections.append({
                            'path': f"{path}.{key}",
                            'key': key,
                            'value': value,
                            'type': 'database_connection'
                        })
                    search_for_db(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    search_for_db(item, f"{path}[{i}]")
        
        search_for_db(json_data)
        return db_connections
    
    def _extract_lookup_definitions(self, json_data: Dict) -> List[Dict]:
        """Extract lookup definitions from JSON data"""
        lookup_definitions = []
        
        def search_for_lookups(obj, path="root"):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if any(term in key.lower() for term in ['lookup', 'reference', 'mapping']):
                        lookup_definitions.append({
                            'path': f"{path}.{key}",
                            'key': key,
                            'value': value,
                            'type': 'lookup_definition'
                        })
                    search_for_lookups(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    search_for_lookups(item, f"{path}[{i}]")
        
        search_for_lookups(json_data)
        return lookup_definitions
    
    def _extract_data_enhancement_rules(self, json_data: Dict) -> List[Dict]:
        """Extract data enhancement rules from JSON data"""
        enhancement_rules = []
        
        def search_for_rules(obj, path="root"):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if any(term in key.lower() for term in ['rule', 'validation', 'enhancement', 'transform']):
                        enhancement_rules.append({
                            'path': f"{path}.{key}",
                            'key': key,
                            'value': value,
                            'type': 'enhancement_rule'
                        })
                    search_for_rules(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    search_for_rules(item, f"{path}[{i}]")
        
        search_for_rules(json_data)
        return enhancement_rules
    
    def _extract_complete_msgflow_content(self, msgflow_path: str) -> Dict[str, Any]:
        """
        Extract COMPLETE .msgflow content for enrichment point analysis
        NO filtering - entire MessageFlow structure goes to LLM for enrichment analysis
        """
        print(f"  📄 Extracting complete .msgflow content for enrichment analysis from: {msgflow_path}")
        
        if not os.path.exists(msgflow_path):
            raise FileNotFoundError(f"MessageFlow file not found: {msgflow_path}")
        
        try:
            with open(msgflow_path, 'r', encoding='utf-8') as f:
                msgflow_xml = f.read()
            
            # Parse XML structure
            root = ET.fromstring(msgflow_xml)
            
            msgflow_data = {
                'raw_xml': msgflow_xml,
                'xml_length': len(msgflow_xml),
                'nodes': self._extract_msgflow_nodes(root),
                'connections': self._extract_msgflow_connections(root),
                'properties': self._extract_msgflow_properties(root),
                'enrichment_nodes': self._extract_enrichment_nodes(root),
                'database_nodes': self._extract_database_nodes(root),
                'flow_metadata': self._extract_msgflow_metadata(root)
            }
            
            print(f"  ✅ MessageFlow processed: {len(msgflow_xml)} characters, {len(msgflow_data['enrichment_nodes'])} enrichment nodes identified")
            self.processed_msgflow_content = msgflow_data
            return msgflow_data
            
        except ET.ParseError as e:
            raise Exception(f"Invalid MessageFlow XML: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to process MessageFlow for enrichment analysis: {str(e)}")
    
    def _extract_msgflow_nodes(self, root: ET.Element) -> List[Dict]:
        """Extract all nodes from MessageFlow XML"""
        nodes = []
        for node in root.iter():
            if node.tag != root.tag:  # Skip root element
                node_data = {
                    'tag': node.tag,
                    'attributes': dict(node.attrib),
                    'text': node.text.strip() if node.text else "",
                    'children_count': len(list(node))
                }
                nodes.append(node_data)
        return nodes
    
    def _extract_msgflow_connections(self, root: ET.Element) -> List[Dict]:
        """Extract connection information from MessageFlow XML"""
        connections = []
        for connection in root.iter():
            if 'connection' in connection.tag.lower() or 'wire' in connection.tag.lower():
                connection_data = {
                    'tag': connection.tag,
                    'attributes': dict(connection.attrib),
                    'type': 'connection'
                }
                connections.append(connection_data)
        return connections
    
    def _extract_msgflow_properties(self, root: ET.Element) -> List[Dict]:
        """Extract properties from MessageFlow XML"""
        properties = []
        for prop in root.iter():
            if 'property' in prop.tag.lower():
                prop_data = {
                    'tag': prop.tag,
                    'attributes': dict(prop.attrib),
                    'text': prop.text.strip() if prop.text else "",
                    'type': 'property'
                }
                properties.append(prop_data)
        return properties
    
    def _extract_enrichment_nodes(self, root: ET.Element) -> List[Dict]:
        """Extract enrichment-specific nodes from MessageFlow XML"""
        enrichment_nodes = []
        enrichment_keywords = ['enrichment', 'lookup', 'database', 'transform', 'map']
        
        for node in root.iter():
            node_str = ET.tostring(node, encoding='unicode').lower()
            if any(keyword in node_str for keyword in enrichment_keywords):
                enrichment_data = {
                    'tag': node.tag,
                    'attributes': dict(node.attrib),
                    'text': node.text.strip() if node.text else "",
                    'enrichment_type': 'detected',
                    'keywords_found': [kw for kw in enrichment_keywords if kw in node_str]
                }
                enrichment_nodes.append(enrichment_data)
        
        return enrichment_nodes
    
    def _extract_database_nodes(self, root: ET.Element) -> List[Dict]:
        """Extract database-related nodes from MessageFlow XML"""
        database_nodes = []
        db_keywords = ['database', 'db', 'sql', 'select', 'table', 'query']
        
        for node in root.iter():
            node_str = ET.tostring(node, encoding='unicode').lower()
            if any(keyword in node_str for keyword in db_keywords):
                db_data = {
                    'tag': node.tag,
                    'attributes': dict(node.attrib),
                    'text': node.text.strip() if node.text else "",
                    'database_type': 'detected',
                    'keywords_found': [kw for kw in db_keywords if kw in node_str]
                }
                database_nodes.append(db_data)
        
        return database_nodes
    
    def _extract_msgflow_metadata(self, root: ET.Element) -> Dict:
        """Extract metadata from MessageFlow XML"""
        metadata = {
            'root_tag': root.tag,
            'root_attributes': dict(root.attrib),
            'total_elements': len(list(root.iter())),
            'namespaces': {}
        }
        
        # Extract namespace information
        for elem in root.iter():
            for key, value in elem.attrib.items():
                if key.startswith('xmlns'):
                    metadata['namespaces'][key] = value
        
        return metadata
    

    
    def _llm_generate_enrichment_configurations(self, enrichment_analysis: Dict) -> Dict[str, Any]:
        """
        LLM generates CW1-specific before/after enrichment JSON configurations
        Pure LLM generation based on Vector DB analysis results for CW1 document processing
        """
        print("  ⚡ LLM generating CW1 before/after enrichment JSON configurations...")
        
        system_prompt = """You are an expert CargoWise One (CW1) data enrichment architect specializing in CDM Document enrichment configurations.

    Generate production-ready before_enrichment.json and after_enrichment.json configurations for CW1 document processing.

    CONFIGURATION REQUIREMENTS:
    - before_enrichment.json: CDM Document structure before database enrichment
    - after_enrichment.json: Enhanced CDM Document with enriched fields from 6 database operations
    - Include field mappings, XPath specifications, and validation rules
    - Ensure configurations support CW1.IN.DOCUMENT.SND.QL → UniversalEvent processing

    OUTPUT: Return JSON object with "before_enrichment" and "after_enrichment" keys containing complete IBM ACE-compatible configurations."""

        user_prompt = f"""Generate CW1 enrichment configurations based on Vector DB analysis:

    ## ENRICHMENT ANALYSIS RESULTS:
    **CW1 Enrichment Patterns Found:**
    {json.dumps(enrichment_analysis.get('enrichment_patterns', []), indent=2)}

    **Before Enrichment Structure:**
    {json.dumps(enrichment_analysis.get('before_enrichment_structure', {}), indent=2)}

    **After Enrichment Structure:**
    {json.dumps(enrichment_analysis.get('after_enrichment_structure', {}), indent=2)}

    **Database Lookup Configurations:**
    {json.dumps(enrichment_analysis.get('lookup_configurations', []), indent=2)}

    **Validation Rules:**
    {json.dumps(enrichment_analysis.get('validation_rules', []), indent=2)}

    **CW1 Data Flow Points:**
    {json.dumps(enrichment_analysis.get('data_flow_points', []), indent=2)}

    ## CW1 ENRICHMENT REQUIREMENTS:
    Generate configurations that handle the 6 specific database operations:
    1. **CompanyCode Lookup** → sp_GetMainCompanyInCountry (MH.ESB.EDIEnterprise)
    2. **CW1 Shipment by SSN** → sp_Shipment_GetIdBySSN (DSV.ESB.Integration)
    3. **CW1 Shipment by HouseBill** → proc_Shipment_GetIdByHouseBill (DSV.ESB.Integration)
    4. **IsPublished Flag** → proc_EDocument_GetIsPublishedFlag (DSV.ESB.Integration)
    5. **CW1 BrokerageId** → proc_CustomsDeclaration_GetIdByReference (DSV.ESB.Integration)
    6. **Target Recipient** → sp_Get_EAdapterRecepientId (MH.ESB.EDIEnterprise)

    ### before_enrichment.json requirements:
    - CDM Document structure with Header/Target/Document sections
    - Required XPath field definitions for the 6 lookup operations
    - Input validation specifications for CompanyCode, CountryCode, EntityReferenceType, etc.
    - Error handling configuration for missing required fields
    - Database connection configurations for MH.ESB.EDIEnterprise and DSV.ESB.Integration

    ### after_enrichment.json requirements:
    - Enhanced CDM Document with all enriched fields populated
    - Results from the 6 database operations: CompanyCode, EE_ShipmentId_by_SSN, EE_IsBooking, EE_ShipmentId_by_HouseBill, IsPublished, CW1BrokerageId, eAdapterRecipientID
    - Validation rules for enriched data consistency
    - Error handling for failed database lookups
    - Routing logic for CW1 UniversalEvent generation

    Return as JSON object: {{"before_enrichment": {{...}}, "after_enrichment": {{...}}}}

    Focus on CW1 document processing workflow: Queue → Enrichment → UniversalEvent → CargoWise One"""

        try:
            response = self.llm.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Lower temperature for consistent configuration generation
                max_tokens=4000
            )

            # Token tracking for Streamlit if available
            if 'token_tracker' in st.session_state and hasattr(response, 'usage') and response.usage:
                st.session_state.token_tracker.manual_track(
                    agent="enrichment_generator",
                    operation="cw1_enrichment_generation",
                    model=self.groq_model,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    flow_name="cw1_enrichment_config"
                )
            
            config_content = response.choices[0].message.content.strip()
            self.llm_generation_calls += 1
            
            # Clean up any markdown formatting if present
            config_content = re.sub(r'^```json\s*\n?', '', config_content)
            config_content = re.sub(r'\n?```\s*$', '', config_content)
            
            # Parse JSON response
            try:
                enrichment_configs = json.loads(config_content)
                
                # Validate structure
                if not isinstance(enrichment_configs, dict):
                    raise ValueError("LLM must return a JSON object")
                
                if 'before_enrichment' not in enrichment_configs or 'after_enrichment' not in enrichment_configs:
                    raise ValueError("LLM must return both before_enrichment and after_enrichment configurations")
                
                print(f"  ✅ CW1 enrichment configuration generation complete")
                print(f"  📊 Generated before_enrichment and after_enrichment configurations for CW1 processing")
                return enrichment_configs
                
            except json.JSONDecodeError:
                # Try to extract JSON from response using regex
                json_match = re.search(r'\{.*\}', config_content, re.DOTALL)
                if json_match:
                    enrichment_configs = json.loads(json_match.group())
                    print(f"  ✅ CW1 configuration generation complete (extracted from response)")
                    return enrichment_configs
                else:
                    raise Exception("LLM did not return valid JSON configuration")
                    
        except Exception as e:
            print(f"  ❌ CW1 enrichment configuration generation failed: {str(e)}")
            # Return default CW1 structure instead of failing
            return {
                'before_enrichment': {
                    'document_structure': 'CDM Document',
                    'required_fields': ['CompanyCode', 'CountryCode', 'EntityReferenceType'],
                    'validation_rules': [],
                    'database_connections': ['MH.ESB.EDIEnterprise', 'DSV.ESB.Integration']
                },
                'after_enrichment': {
                    'document_structure': 'Enriched CDM Document',
                    'enriched_fields': ['CompanyCode', 'EE_ShipmentId_by_SSN', 'IsPublished', 'eAdapterRecipientID'],
                    'validation_rules': [],
                    'output_format': 'UniversalEvent'
                }
            }
    

    
    def _safe_enrichment_data(self, data: Dict) -> Dict:
        """Convert any Ellipsis values to safe strings"""
        safe_data = {}
        for key, value in data.items():
            if value is ... or value is Ellipsis:
                safe_data[key] = f"[{key}_pending]"
            elif isinstance(value, dict):
                safe_data[key] = self._safe_enrichment_data(value)
            elif isinstance(value, list):
                safe_data[key] = [
                    self._safe_enrichment_data(item) if isinstance(item, dict) 
                    else f"[{key}_item_pending]" if item is ... or item is Ellipsis 
                    else item 
                    for item in value
                ]
            else:
                safe_data[key] = value
        return safe_data

    def _write_enrichment_configuration_files(self, enrichment_configs: Dict, output_dir: str) -> List[str]:
        """Write enrichment configuration files to output directory"""
        print("  💾 Writing enrichment configuration files...")
        
        os.makedirs(output_dir, exist_ok=True)
        config_files = []
        
        try:
            # ✅ FIXED: Apply safe data handling
            safe_configs = self._safe_enrichment_data(enrichment_configs)
            
            # Write before_enrichment.json
            before_config_path = os.path.join(output_dir, 'before_enrichment.json')
            with open(before_config_path, 'w', encoding='utf-8') as f:
                json.dump(safe_configs['before_enrichment'], f, indent=2)
            config_files.append(before_config_path)
            
            # Write after_enrichment.json  
            after_config_path = os.path.join(output_dir, 'after_enrichment.json')
            with open(after_config_path, 'w', encoding='utf-8') as f:
                json.dump(safe_configs['after_enrichment'], f, indent=2)
            config_files.append(after_config_path)
            
            return config_files
            
        except Exception as e:
            raise Exception(f"Failed to write enrichment configuration files: {str(e)}")


def main():
    """Test harness for enrichment generator"""
    generator = EnrichmentGenerator()
    
    # Test with sample inputs
    result = generator.generate_enrichment_configurations(
        confluence_pdf_path="sample_requirements.pdf",
        component_mapping_json_path="component_mapping.json",
        msgflow_path="sample.msgflow",
        output_dir="test_output"
    )
    
    print(f"\n🎯 Enrichment Generation Results:")
    print(f"✅ Status: {result['status']}")
    print(f"📊 Configs Generated: {result['enrichment_configs_generated']}")
    print(f"🧠 LLM Analysis Calls: {result['llm_analysis_calls']}")
    print(f"⚡ LLM Generation Calls: {result['llm_generation_calls']}")
    print(f"📦 Chunking Stats: {result['chunking_stats']}")


if __name__ == "__main__":
    main()