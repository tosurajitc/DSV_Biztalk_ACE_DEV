"""
Smart MessageFlow Corrector Module
==================================
Discovers ACE applications via naming_convention files and corrects messageflows
to sync with actual components and configuration flags.

Author: AI Assistant
Date: 2025-10-30
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SmartMessageFlowCorrector:
    """Smart corrector that syncs messageflows with actual ACE components"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.corrected_count = 0
        self.error_count = 0
        
    def run(self) -> Dict:
        """Main execution method"""
        logger.info("🚀 Starting Smart MessageFlow Correction")
        logger.info(f"Project Root: {self.project_root.absolute()}")
        
        try:
            # Discover applications via naming conventions
            naming_files = self.discover_naming_conventions()
            logger.info(f"📋 Found {len(naming_files)} naming convention files")
            
            if not naming_files:
                logger.warning("⚠️ No naming_convention*.json files found")
                return {'status': 'error', 'message': 'No naming convention files found'}
            
            # Process each application
            for naming_file in naming_files:
                try:
                    app_name = naming_file['ace_application_name']
                    logger.info(f"\n🔧 Processing Application: {app_name}")
                    
                    app_folder = self.project_root / app_name
                    if not app_folder.exists():
                        logger.error(f"❌ Application folder not found: {app_folder}")
                        self.error_count += 1
                        continue
                    
                    # Correct the messageflow
                    self.correct_application_messageflow(app_folder)
                    self.corrected_count += 1
                    
                except Exception as e:
                    logger.error(f"❌ Error processing {naming_file.get('ace_application_name', 'unknown')}: {e}")
                    self.error_count += 1
            
            # Summary
            logger.info(f"\n✅ Correction Complete!")
            logger.info(f"   Corrected: {self.corrected_count}")
            logger.info(f"   Errors: {self.error_count}")
            
            return {
                'status': 'success',
                'corrected': self.corrected_count,
                'errors': self.error_count
            }
            
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def discover_naming_conventions(self) -> List[Dict]:
        """Discover all naming_convention*.json files and extract ace_application_name"""
        naming_files = []
        
        # Find all naming convention files
        pattern = "naming_convention*.json"
        for file_path in self.project_root.glob(pattern):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check for ace_application_name in nested structure
                ace_app_name = None
                if 'project_naming' in data and 'ace_application_name' in data['project_naming']:
                    ace_app_name = data['project_naming']['ace_application_name']
                elif 'ace_application_name' in data:
                    ace_app_name = data['ace_application_name']
                
                if ace_app_name:
                    naming_files.append({
                        'file': file_path.name,
                        'ace_application_name': ace_app_name,
                        'full_data': data
                    })
                    logger.info(f"📄 {file_path.name} → {ace_app_name}")
                else:
                    logger.warning(f"⚠️ No ace_application_name in {file_path.name}")
                    
            except Exception as e:
                logger.error(f"❌ Error reading {file_path.name}: {e}")
        
        return naming_files
    
    def correct_application_messageflow(self, app_folder: Path):
        """Correct messageflow for a single ACE application"""
        # Find the messageflow file
        msgflow_files = list(app_folder.glob("*.msgflow"))
        if not msgflow_files:
            logger.error(f"❌ No .msgflow file found in {app_folder}")
            return
        
        if len(msgflow_files) > 1:
            logger.warning(f"⚠️ Multiple .msgflow files found, using first: {msgflow_files[0].name}")
        
        msgflow_path = msgflow_files[0]
        logger.info(f"📝 Messageflow: {msgflow_path.name}")
        
        # Scan ACE components
        components = self.scan_ace_components(app_folder)
        logger.info(f"🔍 Components found: {self.format_components(components)}")
        
        # Parse configuration from messageflow
        config = self.parse_node_configuration(msgflow_path)
        logger.info(f"⚙️ Configuration: {self.format_config(config)}")
        
        # Build corrected messageflow
        corrected_content = self.build_corrected_messageflow(msgflow_path, components, config)
        
        # Save corrected messageflow
        with open(msgflow_path, 'w', encoding='utf-8') as f:
            f.write(corrected_content)
        
        logger.info(f"✅ Corrected: {msgflow_path.name}")
    
    def scan_ace_components(self, app_folder: Path) -> Dict:
        """Scan actual ACE components in application folder"""
        components = {
            'esql_files': [],
            'xsl_files': [],
            'schemas': [],
            'has_transforms': False,
            'has_schemas': False
        }
        
        # Scan ESQL files
        for esql_file in app_folder.rglob("*.esql"):
            purpose = self.determine_esql_purpose(esql_file.stem)
            components['esql_files'].append({
                'name': esql_file.name,
                'path': esql_file,
                'purpose': purpose,
                'module': self.extract_esql_module(esql_file)
            })
        
        # Scan XSL files
        transforms_folder = app_folder / "transforms"
        if transforms_folder.exists():
            components['has_transforms'] = True
            for xsl_file in transforms_folder.glob("*.xsl"):
                components['xsl_files'].append({
                    'name': xsl_file.name,
                    'path': xsl_file
                })
        
        # Scan schemas
        schemas_folder = app_folder / "schemas"
        if schemas_folder.exists():
            components['has_schemas'] = True
            for schema_file in schemas_folder.rglob("*.xsd"):
                components['schemas'].append({
                    'name': schema_file.name,
                    'path': schema_file
                })
        
        return components
    
    def determine_esql_purpose(self, filename: str) -> str:
        """Determine ESQL file purpose from filename"""
        filename_lower = filename.lower()
        
        if "inputeventmessage" in filename_lower:
            return "input_event"
        elif "outputeventmessage" in filename_lower:
            return "output_event"
        elif "aftereventmsg" in filename_lower or "afterevent" in filename_lower:
            return "after_event"
        elif "failure" in filename_lower:
            return "failure"
        elif "beforeenrichment" in filename_lower:
            return "before_enrichment"
        elif "afterenrichment" in filename_lower:
            return "after_enrichment"
        elif "compute" in filename_lower and "before" not in filename_lower and "after" not in filename_lower:
            return "compute"
        else:
            return "unknown"
    
    def extract_esql_module(self, esql_file: Path) -> str:
        """Extract ESQL module name from file content"""
        try:
            with open(esql_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for CREATE FUNCTION or CREATE PROCEDURE patterns
            patterns = [
                r'CREATE\s+FUNCTION\s+(\w+)',
                r'CREATE\s+PROCEDURE\s+(\w+)',
                r'CREATE\s+(?:COMPUTE|DATABASE|FILTER)\s+MODULE\s+(\w+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    return match.group(1)
            
            # Fallback: use filename without extension
            return esql_file.stem
            
        except Exception as e:
            logger.debug(f"Could not extract module from {esql_file.name}: {e}")
            return esql_file.stem
    
    def parse_node_configuration(self, msgflow_path: Path) -> Dict:
        """Parse node configuration flags from messageflow comments"""
        config = {
            'http_input': False,
            'mq_input': False,
            'file_input': False,
            'xsl_transform': False,
            'before_enrichment': False,
            'after_enrichment': False,
            'soap_request': False,
            'gzip_compression': False,
            'routing': False
        }
        
        try:
            with open(msgflow_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract configuration from comments
            patterns = {
                'http_input': r'<!-- HTTP Input: (True|False) -->',
                'mq_input': r'<!-- MQ Input: (True|False) -->',
                'file_input': r'<!-- File Input: (True|False) -->',
                'xsl_transform': r'<!-- XSL Transform: (True|False) -->',
                'before_enrichment': r'<!-- Before Enrichment: (True|False) -->',
                'after_enrichment': r'<!-- After Enrichment: (True|False) -->',
                'soap_request': r'<!-- SOAP Request: (True|False) -->',
                'gzip_compression': r'<!-- GZip Compression: (True|False) -->',
                'routing': r'<!-- Routing: (True|False) -->'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    config[key] = match.group(1) == 'True'
            
        except Exception as e:
            logger.error(f"Error parsing configuration from {msgflow_path.name}: {e}")
        
        return config
    
    def filter_unused_properties(self, content: str, config: Dict) -> str:
        """Remove unused property templates based on configuration flags"""
        
        filtered_content = content
        
        # Remove HTTP properties if HTTP Input: False
        if not config.get('http_input', False):
            start_marker = '<!-- HTTP INPUT PROPERTIES -->'
            end_marker = '<!-- MQ INPUT PROPERTIES -->'
            start_pos = filtered_content.find(start_marker)
            if start_pos != -1:
                end_pos = filtered_content.find(end_marker)
                if end_pos != -1:
                    filtered_content = filtered_content[:start_pos] + filtered_content[end_pos:]
        
        # Remove MQ properties if MQ Input: False  
        if not config.get('mq_input', False):
            start_marker = '<!-- MQ INPUT PROPERTIES -->'
            end_marker = '<!-- FILE INPUT PROPERTIES -->'
            start_pos = filtered_content.find(start_marker)
            if start_pos != -1:
                end_pos = filtered_content.find(end_marker)
                if end_pos != -1:
                    filtered_content = filtered_content[:start_pos] + filtered_content[end_pos:]
        
        # Remove SOAP properties if SOAP Request: False
        if not config.get('soap_request', False):
            start_marker = '<!-- SOAP PROPERTIES -->'
            end_marker = '<!-- HTTP REQUEST PROPERTIES -->'
            start_pos = filtered_content.find(start_marker)
            if start_pos != -1:
                end_pos = filtered_content.find(end_marker)
                if end_pos != -1:
                    filtered_content = filtered_content[:start_pos] + filtered_content[end_pos:]
        
        # Remove BEFORE ENRICHMENT properties if Before Enrichment: False
        if not config.get('before_enrichment', False):
            start_marker = '<!-- BEFORE ENRICHMENT PROPERTIES -->'
            end_marker = '<!-- AFTER ENRICHMENT PROPERTIES -->'
            start_pos = filtered_content.find(start_marker)
            if start_pos != -1:
                end_pos = filtered_content.find(end_marker)
                if end_pos != -1:
                    filtered_content = filtered_content[:start_pos] + filtered_content[end_pos:]
        
        # Remove AFTER ENRICHMENT properties if After Enrichment: False  
        if not config.get('after_enrichment', False):
            start_marker = '<!-- AFTER ENRICHMENT PROPERTIES -->'
            end_marker = '<!-- SOAP PROPERTIES -->'
            start_pos = filtered_content.find(start_marker)
            if start_pos != -1:
                end_pos = filtered_content.find(end_marker)
                if end_pos != -1:
                    filtered_content = filtered_content[:start_pos] + filtered_content[end_pos:]
        
        return filtered_content
    
    def cleanup_orphaned_attribute_links(self, content: str, config: Dict) -> str:
        """Remove orphaned attribute links that reference non-existent nodes"""
        
        filtered_content = content
        
        # Remove MQ Input attribute links if MQ Input is disabled
        if not config.get('mq_input', False):
            # Find all attributeLinks elements with FCMComposite_1_7 and collect their positions
            import re
            
            # Find all attributeLinks elements that contain FCMComposite_1_7
            matches = []
            pattern = r'<attributeLinks[^>]*>.*?</attributeLinks>'
            
            for match in re.finditer(pattern, filtered_content, re.DOTALL):
                if 'FCMComposite_1_7' in match.group():
                    matches.append((match.start(), match.end()))
            
            # Remove matches in reverse order to maintain correct indices
            for start, end in reversed(matches):
                filtered_content = filtered_content[:start] + filtered_content[end:]
        
        return filtered_content
    
    def build_corrected_messageflow(self, msgflow_path: Path, components: Dict, config: Dict) -> str:
        """Build corrected messageflow content maintaining template format"""
        flow_name = msgflow_path.stem
        
        # Read original content
        with open(msgflow_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Extract and preserve header section (up to <composition>)
        header_end = original_content.find('<composition>')
        if header_end == -1:
            raise ValueError("Cannot find <composition> section in messageflow")
        
        header_section = original_content[:header_end]
        
        # Filter unused properties from header section
        header_section = self.filter_unused_properties(header_section, config)
        
        # Clean up orphaned attribute links
        header_section = self.cleanup_orphaned_attribute_links(header_section, config)
        
        # Build nodes and connections
        nodes_xml = self.build_nodes_section(components, config, flow_name)
        connections_xml = self.build_connections_section(components, config)
        
        # Build composition section
        composition_section = f'''<composition>
{nodes_xml}
{connections_xml}
</composition>'''
        
        # Extract and preserve footer section
        footer_start = original_content.find('<propertyOrganizer>')
        if footer_start == -1:
            footer_start = original_content.find('<stickyBoard/>')
        
        if footer_start != -1:
            footer_section = original_content[footer_start:]
        else:
            footer_section = "\n    <stickyBoard/>\n  </eClassifiers>\n</ecore:EPackage>"
        
        # Combine all sections
        return header_section + composition_section + "\n\n    " + footer_section
    
    def build_nodes_section(self, components: Dict, config: Dict, flow_name: str) -> str:
        """Build the nodes section based on components and configuration"""
        nodes = []
        x_pos = 50
        y_main = 100
        
        # 1. Input Node (based on config)
        if config['file_input']:
            nodes.append(self.create_file_input_node(x_pos, y_main))
            x_pos += 150
        elif config['mq_input']:
            nodes.append(self.create_mq_input_node(x_pos, y_main))
            x_pos += 150
        elif config['http_input']:
            nodes.append(self.create_http_input_node(x_pos, y_main))
            x_pos += 150
        
        # 2. Event Processing Nodes (based on ESQL files)
        input_event_esql = next((esql for esql in components['esql_files'] if esql['purpose'] == 'input_event'), None)
        if input_event_esql:
            nodes.append(self.create_compute_node('InputEvent', input_event_esql, 'InputEventMessage', x_pos, y_main))
            x_pos += 150
        
        # 3. Enrichment Nodes (based on config)
        if config['before_enrichment']:
            nodes.append(self.create_enrichment_node('BeforeEnrich', 'BeforeEnrichment', x_pos, y_main))
            x_pos += 150
        
        # 4. Main Compute Node
        compute_esql = next((esql for esql in components['esql_files'] if esql['purpose'] == 'compute'), None)
        if compute_esql:
            nodes.append(self.create_compute_node('Compute', compute_esql, 'Compute', x_pos, y_main))
            x_pos += 150
        
        # 5. Routing Structure (based on config and XSL files)
        if config['routing'] and len(components['xsl_files']) > 1:
            # Router
            nodes.append(self.create_router_node(x_pos, y_main))
            x_pos += 150
            
            # Labels, XSL transforms, and RouteToLabel for each method
            y_offset = 50
            for idx, xsl in enumerate(components['xsl_files']):
                method_name = self.extract_method_name(xsl['name'], flow_name)
                branch_y = y_main + (idx - len(components['xsl_files'])//2) * y_offset
                
                # Label
                nodes.append(self.create_label_node(f"Label_{idx+1}", method_name, x_pos, branch_y))
                
                # XSL Transform
                nodes.append(self.create_xsl_node(f"XSL_{idx+1}", xsl['name'], method_name, x_pos + 150, branch_y))
                
                # RouteToLabel back to merge
                nodes.append(self.create_route_to_label_node(f"RouteToLabel_{idx+1}", "MergeLabel", f"RouteToLabel_{method_name}", x_pos + 300, branch_y))
            
            # MergeLabel
            x_pos += 450
            nodes.append(self.create_label_node("MergeLabel", "MergeLabel", x_pos, y_main))
            x_pos += 150
            
        elif config['xsl_transform'] and components['xsl_files']:
            # Single XSL transform (no routing)
            xsl = components['xsl_files'][0]
            nodes.append(self.create_xsl_node('XSL', xsl['name'], 'XSLTransform', x_pos, y_main))
            x_pos += 150
        
        # 6. After Enrichment
        if config['after_enrichment']:
            nodes.append(self.create_enrichment_node('AfterEnrich', 'AfterEnrichment', x_pos, y_main))
            x_pos += 150
        
        # 7. Output Event Message (always required)
        output_event_esql = next((esql for esql in components['esql_files'] if esql['purpose'] == 'output_event'), None)
        if output_event_esql:
            nodes.append(self.create_compute_node('OutputEvent', output_event_esql, 'OutputEventMessage', x_pos, y_main))
            x_pos += 150
        
        # 8. Output Terminal Node
        if config['file_input']:
            nodes.append(self.create_file_output_node(x_pos, y_main))
            x_pos += 150
        elif config['mq_input']:
            nodes.append(self.create_mq_output_node(x_pos, y_main))
            x_pos += 150
        elif config['http_input']:
            nodes.append(self.create_http_reply_node(x_pos, y_main))
            x_pos += 150
        
        # 9. Mandatory Nodes (always required)
        # After Event Message
        after_event_esql = next((esql for esql in components['esql_files'] if esql['purpose'] == 'after_event'), None)
        if after_event_esql:
            nodes.append(self.create_compute_node('AfterEvent', after_event_esql, 'AfterEventMessage', x_pos, y_main))
        else:
            nodes.append(self.create_default_compute_node('AfterEvent', 'AfterEventMessage', x_pos, y_main))
        
        # Failure Handler
        failure_esql = next((esql for esql in components['esql_files'] if esql['purpose'] == 'failure'), None)
        if failure_esql:
            nodes.append(self.create_compute_node('FailureHandler', failure_esql, 'FailureHandler', x_pos, y_main + 200))
        else:
            nodes.append(self.create_default_compute_node('FailureHandler', 'FailureHandler', x_pos, y_main + 200))
        
        return '\n\n      '.join(nodes)
    
    def build_connections_section(self, components: Dict, config: Dict) -> str:
        """Build the connections section"""
        connections = []
        conn_counter = 1
        
        # Main flow connections
        if config['file_input']:
            # FileInput → InputEvent
            connections.append(self.create_connection(f"FCMConnection_{conn_counter}", 
                "FCMComposite_1_FileInput", "FCMComposite_1_InputEvent"))
            conn_counter += 1
            
            # InputEvent → Compute  
            connections.append(self.create_connection(f"FCMConnection_{conn_counter}", 
                "FCMComposite_1_InputEvent", "FCMComposite_1_Compute"))
            conn_counter += 1
            
            # Compute → Router (if routing) or XSL (if single transform)
            if config['routing'] and len(components['xsl_files']) > 1:
                connections.append(self.create_connection(f"FCMConnection_{conn_counter}", 
                    "FCMComposite_1_Compute", "FCMComposite_1_Router"))
                conn_counter += 1
                
                # Router → Labels → XSL → RouteToLabel for each branch
                for idx in range(len(components['xsl_files'])):
                    # Router → Label
                    connections.append(self.create_connection(f"FCMConnection_{conn_counter}", 
                        "FCMComposite_1_Router", f"FCMComposite_1_Label_{idx+1}"))
                    conn_counter += 1
                    
                    # Label → XSL
                    connections.append(self.create_connection(f"FCMConnection_{conn_counter}", 
                        f"FCMComposite_1_Label_{idx+1}", f"FCMComposite_1_XSL_{idx+1}"))
                    conn_counter += 1
                    
                    # XSL → RouteToLabel
                    connections.append(self.create_connection(f"FCMConnection_{conn_counter}", 
                        f"FCMComposite_1_XSL_{idx+1}", f"FCMComposite_1_RouteToLabel_{idx+1}"))
                    conn_counter += 1
                
                # MergeLabel → OutputEvent
                connections.append(self.create_connection(f"FCMConnection_{conn_counter}", 
                    "FCMComposite_1_MergeLabel", "FCMComposite_1_OutputEvent"))
                conn_counter += 1
                
            elif config['xsl_transform'] and components['xsl_files']:
                # Single XSL: Compute → XSL → OutputEvent
                connections.append(self.create_connection(f"FCMConnection_{conn_counter}", 
                    "FCMComposite_1_Compute", "FCMComposite_1_XSL"))
                conn_counter += 1
                
                connections.append(self.create_connection(f"FCMConnection_{conn_counter}", 
                    "FCMComposite_1_XSL", "FCMComposite_1_OutputEvent"))
                conn_counter += 1
            else:
                # No transforms: Compute → OutputEvent
                connections.append(self.create_connection(f"FCMConnection_{conn_counter}", 
                    "FCMComposite_1_Compute", "FCMComposite_1_OutputEvent"))
                conn_counter += 1
            
            # OutputEvent → FileOutput
            connections.append(self.create_connection(f"FCMConnection_{conn_counter}", 
                "FCMComposite_1_OutputEvent", "FCMComposite_1_FileOutput"))
            conn_counter += 1
            
            # FileOutput → AfterEvent (always required)
            connections.append(self.create_connection(f"FCMConnection_{conn_counter}", 
                "FCMComposite_1_FileOutput", "FCMComposite_1_AfterEvent"))
            conn_counter += 1
        
        # Error handling connections (always required)
        error_sources = ["FCMComposite_1_InputEvent", "FCMComposite_1_Compute", "FCMComposite_1_OutputEvent"]
        if config['xsl_transform']:
            if config['routing']:
                for idx in range(len(components['xsl_files'])):
                    error_sources.append(f"FCMComposite_1_XSL_{idx+1}")
            else:
                error_sources.append("FCMComposite_1_XSL")
        
        # Always create failure handler connections
        for source in error_sources:
            connections.append(self.create_error_connection(f"FCMConnection_Error_{conn_counter}", 
                source, "FCMComposite_1_FailureHandler"))
            conn_counter += 1
        
        return '\n\n        '.join(connections)
    
    def determine_node_sequence(self, components: Dict, config: Dict) -> List[str]:
        """Determine the correct sequence of nodes for connections"""
        sequence = []
        
        # Input node
        if config['file_input']:
            sequence.append('FCMComposite_1_FileInput')
        elif config['mq_input']:
            sequence.append('FCMComposite_1_MQInput')
        elif config['http_input']:
            sequence.append('FCMComposite_1_HTTPInput')
        
        # Processing sequence
        if any(esql['purpose'] == 'input_event' for esql in components['esql_files']):
            sequence.append('FCMComposite_1_InputEvent')
        
        if config['before_enrichment']:
            sequence.append('FCMComposite_1_BeforeEnrich')
        
        if any(esql['purpose'] == 'compute' for esql in components['esql_files']):
            sequence.append('FCMComposite_1_Compute')
        
        if config['routing'] and len(components['xsl_files']) > 1:
            sequence.append('FCMComposite_1_Router')
            # Routing branches handled separately
        elif config['xsl_transform']:
            sequence.append('FCMComposite_1_XSL')
        
        if config['after_enrichment']:
            sequence.append('FCMComposite_1_AfterEnrich')
        
        if any(esql['purpose'] == 'output_event' for esql in components['esql_files']):
            sequence.append('FCMComposite_1_OutputEvent')
        
        # Output node
        if config['file_input']:
            sequence.append('FCMComposite_1_FileOutput')
        elif config['mq_input']:
            sequence.append('FCMComposite_1_MQOutput')
        elif config['http_input']:
            sequence.append('FCMComposite_1_HTTPReply')
        
        if any(esql['purpose'] == 'after_event' for esql in components['esql_files']):
            sequence.append('FCMComposite_1_AfterEvent')
        
        return sequence
    
    def build_error_connections(self, components: Dict) -> List[str]:
        """Build error handling connections"""
        error_connections = []
        
        # Find failure handler
        failure_handler = None
        for esql in components['esql_files']:
            if esql['purpose'] == 'failure':
                failure_handler = 'FCMComposite_1_FailureHandler'
                break
        
        if failure_handler:
            # Connect common nodes to failure handler
            error_sources = [
                'FCMComposite_1_InputEvent',
                'FCMComposite_1_Compute',
                'FCMComposite_1_XSL',
                'FCMComposite_1_OutputEvent'
            ]
            
            for idx, source in enumerate(error_sources):
                conn_id = f"FCMConnection_Error_{idx+1}"
                error_connections.append(self.create_error_connection(conn_id, source, failure_handler))
        
        return error_connections
    
    # Node creation methods (maintaining template format)
    
    def create_file_input_node(self, x: int, y: int) -> str:
        return '''<!-- File Input Node -->
      <nodes xmi:type="ComIbmFileInput.msgnode:FCMComposite_1" 
            xmi:id="FCMComposite_1_FileInput" 
            location="50,100"
            inputDirectory="/var/mqsi/input"
            filenamePattern="*.xml"
            messageDomainProperty="XMLNSC">
        <translation xmi:type="utility:ConstantString" string="FileInput"/>
      </nodes>'''
    
    def create_mq_input_node(self, x: int, y: int) -> str:
        return '''<!-- MQ Input Node -->
      <nodes xmi:type="epis_common_flows_lib_MQInput.subflow:FCMComposite_1" 
            xmi:id="FCMComposite_1_MQInput" 
            location="50,100">
        <translation xmi:type="utility:ConstantString" string="MQInput"/>
      </nodes>'''
    
    def create_http_input_node(self, x: int, y: int) -> str:
        return f'''<!-- HTTP Input Node -->
      <nodes xmi:type="ComIbmWSInput.msgnode:FCMComposite_1" 
            xmi:id="FCMComposite_1_HTTPInput" 
            location="{x},{y}"
            URLSpecifier="/service">
        <translation xmi:type="utility:ConstantString" string="HTTPInput"/>
      </nodes>'''
    
    def create_compute_node(self, node_id: str, esql: Dict, display_name: str, x: int, y: int) -> str:
        module = esql.get('module', '')
        esql_file = esql.get('name', '')
        function_name = esql_file.replace('.esql', '')
        
        # Extract flow name from function name (remove suffix after last underscore)
        if '_' in function_name:
            flow_name = '_'.join(function_name.split('_')[:-1])
        else:
            flow_name = function_name
        
        if module:
            esql_expr = f"esql://routine/{flow_name}#{function_name}.Main"
        else:
            esql_expr = f"esql://routine/{flow_name}#{function_name}.Main"
        
        return f'''<!-- {display_name} Node -->
      <nodes xmi:type="ComIbmCompute.msgnode:FCMComposite_1" 
            xmi:id="FCMComposite_1_{node_id}" 
            location="{x},{y}" 
            computeExpression="{esql_expr}">
        <translation xmi:type="utility:ConstantString" string="{display_name}"/>
      </nodes>'''
    
    def create_router_node(self, x: int, y: int) -> str:
        return f'''<!-- Router Node -->
      <nodes xmi:type="ComIbmRoute.msgnode:FCMComposite_1" 
            xmi:id="FCMComposite_1_Router" 
            location="{x},{y}">
        <translation xmi:type="utility:ConstantString" string="MethodRouter"/>
      </nodes>'''
    
    def create_label_node(self, node_id: str, label_name: str, x: int, y: int) -> str:
        return f'''<!-- Label Node -->
      <nodes xmi:type="ComIbmLabel.msgnode:FCMComposite_1" 
            xmi:id="FCMComposite_1_{node_id}" 
            location="{x},{y}"
            labelName="{label_name}">
        <translation xmi:type="utility:ConstantString" string="{label_name}"/>
      </nodes>'''
    
    def create_xsl_node(self, node_id: str, stylesheet: str, display_name: str, x: int, y: int) -> str:
        return f'''<!-- XSL Transform Node -->
      <nodes xmi:type="ComIbmXslMqsi.msgnode:FCMComposite_1" 
            xmi:id="FCMComposite_1_{node_id}" 
            location="{x},{y}" 
            stylesheetName="{stylesheet}" 
            messageDomainProperty="XMLNSC">
        <translation xmi:type="utility:ConstantString" string="{display_name}"/>
      </nodes>'''
    
    def create_route_to_label_node(self, node_id: str, target_label: str, display_name: str, x: int, y: int) -> str:
        return f'''<!-- RouteToLabel Node -->
      <nodes xmi:type="ComIbmRouteToLabel.msgnode:FCMComposite_1" 
            xmi:id="FCMComposite_1_{node_id}" 
            location="{x},{y}"
            labelName="{target_label}">
        <translation xmi:type="utility:ConstantString" string="{display_name}"/>
      </nodes>'''
    
    def create_enrichment_node(self, node_id: str, display_name: str, x: int, y: int) -> str:
        return f'''<!-- {display_name} Node -->
      <nodes xmi:type="epis_enrichment_lib_EPIS_MessageEnrichment.subflow:FCMComposite_1" 
            xmi:id="FCMComposite_1_{node_id}" 
            location="{x},{y}" 
            inputDirectory="/var/mqsi/enrichment" 
            filenamePattern="EnrichmentConf.json">
        <translation xmi:type="utility:ConstantString" string="{display_name}"/>
      </nodes>'''
    
    def create_file_output_node(self, x: int, y: int) -> str:
        return f'''<!-- File Output Node -->
      <nodes xmi:type="ComIbmFileOutput.msgnode:FCMComposite_1" 
            xmi:id="FCMComposite_1_FileOutput" 
            location="{x},{y}"
            outputDirectory="/var/mqsi/output"
            outputFilename="output_%d%t.xml">
        <translation xmi:type="utility:ConstantString" string="FileOutput"/>
      </nodes>'''
    
    def create_mq_output_node(self, x: int, y: int) -> str:
        return f'''<!-- MQ Output Node -->
      <nodes xmi:type="epis_common_flows_lib_MQOutput.subflow:FCMComposite_1" 
            xmi:id="FCMComposite_1_MQOutput" 
            location="{x},{y}">
        <translation xmi:type="utility:ConstantString" string="MQOutput"/>
      </nodes>'''
    
    def create_http_reply_node(self, x: int, y: int) -> str:
        return f'''<!-- HTTP Reply Node -->
      <nodes xmi:type="ComIbmWSReply.msgnode:FCMComposite_1" 
            xmi:id="FCMComposite_1_HTTPReply" 
            location="{x},{y}">
        <translation xmi:type="utility:ConstantString" string="HTTPReply"/>
      </nodes>'''
    
    def create_default_compute_node(self, node_id: str, display_name: str, x: int, y: int) -> str:
        """Create compute node with default ESQL expression when no ESQL file exists"""
        esql_expr = "esql://routine/#Main"
        
        return f'''<!-- {display_name} Node -->
      <nodes xmi:type="ComIbmCompute.msgnode:FCMComposite_1" 
            xmi:id="FCMComposite_1_{node_id}" 
            location="{x},{y}" 
            computeExpression="{esql_expr}">
        <translation xmi:type="utility:ConstantString" string="{display_name}"/>
      </nodes>'''
    
    def create_connection(self, conn_id: str, source_node: str, target_node: str) -> str:
        return f'''<connections xmi:type="eflow:FCMConnection" xmi:id="{conn_id}" 
                    targetNode="{target_node}" sourceNode="{source_node}" 
                    sourceTerminalName="OutTerminal.out" targetTerminalName="InTerminal.in"/>'''
    
    def create_error_connection(self, conn_id: str, source_node: str, target_node: str) -> str:
        return f'''<connections xmi:type="eflow:FCMConnection" xmi:id="{conn_id}" 
                    targetNode="{target_node}" sourceNode="{source_node}" 
                    sourceTerminalName="OutTerminal.failure" targetTerminalName="InTerminal.in"/>'''
    
    def extract_method_name(self, xsl_filename: str, flow_name: str) -> str:
        """Extract method name from XSL filename with comprehensive fallback logic"""
        original_name = xsl_filename.replace('.xsl', '')
        name = original_name
        
        # Remove flow name prefix if present
        if name.startswith(flow_name):
            name = name[len(flow_name):].lstrip('_')
        
        # Pattern 1: Standard _To_ convention (Source_To_Target)
        if '_To_' in name:
            return name.split('_To_')[0]
        
        # Pattern 2: Has content after prefix removal
        if name and name != original_name:
            return name
        
        # Pattern 3: Extract meaningful part from original filename
        # Look for patterns like: Schema names, Transform types, etc.
        meaningful_parts = []
        for part in original_name.split('_'):
            if part not in ['Transform', 'Mapping', 'Convert', flow_name]:
                meaningful_parts.append(part)
        
        if meaningful_parts:
            return '_'.join(meaningful_parts[:2])  # Take first 2 meaningful parts
        
        # Pattern 4: Use filename without extension as last resort
        if original_name != flow_name:
            return original_name
        
        # Final fallback for base files
        return "BaseTransform"
    
    # Utility methods for logging
    
    def format_components(self, components: Dict) -> str:
        """Format components for logging"""
        parts = []
        if components['esql_files']:
            parts.append(f"ESQL({len(components['esql_files'])})")
        if components['xsl_files']:
            parts.append(f"XSL({len(components['xsl_files'])})")
        if components['has_schemas']:
            parts.append("Schemas")
        return ", ".join(parts) if parts else "None"
    
    def format_config(self, config: Dict) -> str:
        """Format configuration for logging"""
        active = [key.replace('_', ' ').title() for key, value in config.items() if value]
        return ", ".join(active) if active else "None active"


def main():
    """Main entry point"""
    corrector = SmartMessageFlowCorrector()
    result = corrector.run()
    return result


if __name__ == "__main__":
    main()