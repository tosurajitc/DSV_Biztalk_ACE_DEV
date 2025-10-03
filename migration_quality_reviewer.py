#!/usr/bin/env python3
"""
migration_quality_reviewer.py - SIMPLIFIED VERSION
Purpose: Create ACE Toolkit compatible folder structure with correct naming conventions
Input: Components from Agent 3 + naming_convention.json (from fetch_naming.py)
Output: Final ACE project with proper names
NO QUALITY REVIEW - Just naming and structure
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import streamlit as st
from llm_json_parser import parse_llm_json, LLMJSONParser

class SmartACEQualityReviewer:
    """
    Simplified ACE project creator - ONLY handles naming and folder structure
    Maintains same interface as original for compatibility with main.py
    """
    
    def __init__(self, 
                ace_components_folder: str,
                naming_standards_file: str,
                vector_db_content: str,
                user_requirements: Optional[str] = None,
                ace_toolkit_path: Optional[str] = None):
        
        # Store parameters (maintain compatibility)
        self.ace_components_folder = Path(ace_components_folder)
        self.naming_standards_file = Path(naming_standards_file)
        self.vector_db_content = vector_db_content
        self.user_requirements = user_requirements
        self.ace_toolkit_path = ace_toolkit_path
        
        # Component discovery
        self.discovered_components = []
        self.naming_convention = {}
        self.token_usage = 0  # For compatibility with main.py
        
        # Error tracking
        self.validation_errors = []
        self.component_errors = {}
        self.missing_files = []
        
        # ACE component types
        self.ace_component_types = {
            '.esql': 'compute_modules',
            '.msgflow': 'message_flows', 
            '.subflow': 'sub_flows',
            '.xsd': 'schemas',
            '.xml': 'configurations',
            '.map': 'mappings',
            '.xsl': 'transforms'
        }
    

    def _consolidate_enrichment_files(self, project_dir: Path, enrichment_files: List[Path]) -> bool:
        """
        Main method: Consolidate 5 separate enrichment files into beforeenrichment.json and afterenrichment.json
        Returns: True if successful, False if failed
        """
        print(f"\n🔄 Consolidating enrichment configurations...")
        
        try:
            # Step 1: Read all individual enrichment files
            individual_configs = self._read_individual_enrichment_files(enrichment_files)
            
            if not individual_configs:
                print(f"  ⚠️ No valid enrichment configurations found")
                return False
            
            # Step 2: Create before enrichment config (basic/empty state)
            before_config = self._create_before_enrichment_config(individual_configs)
            
            # Step 3: Create after enrichment config (consolidated all configs)
            after_config = self._create_after_enrichment_config(individual_configs)
            
            # Step 4: Write the consolidated files
            self._write_consolidated_enrichment_files(project_dir, before_config, after_config)
            
            print(f"  📄 Created: BeforeEnrichmentConf.json and AfterEnrichmentConf.json")
            return True
            
        except Exception as e:
            print(f"  ❌ Enrichment consolidation failed: {e}")
            self.validation_errors.append(f"Enrichment consolidation failed: {e}")
            return False

    def _read_individual_enrichment_files(self, enrichment_files: List[Path]) -> List[Dict]:
        """Read and parse all individual enrichment JSON files"""
        print(f"  📖 Reading {len(enrichment_files)} individual enrichment files...")
        
        individual_configs = []
        
        for enrichment_file in enrichment_files:
            try:
                print(f"    🔍 Processing: {enrichment_file.name}")
                
                with open(enrichment_file, 'r', encoding='utf-8') as f:
                    file_content = json.load(f)
                
                # Extract the enrichment configuration
                if isinstance(file_content, dict):
                    # Add metadata about source file
                    config_with_metadata = {
                        'source_file': enrichment_file.name,
                        'config': file_content
                    }
                    individual_configs.append(config_with_metadata)
                    print(f"      ✅ Loaded config from {enrichment_file.name}")
                else:
                    print(f"      ⚠️ Invalid JSON structure in {enrichment_file.name}")
                    
            except json.JSONDecodeError as e:
                print(f"      ❌ JSON parse error in {enrichment_file.name}: {e}")
                self.validation_errors.append(f"JSON parse error: {enrichment_file.name}")
            except Exception as e:
                print(f"      ❌ Error reading {enrichment_file.name}: {e}")
                self.validation_errors.append(f"File read error: {enrichment_file.name}")
        
        print(f"    📊 Successfully loaded {len(individual_configs)} enrichment configurations")
        return individual_configs
    


    def _create_before_enrichment_config(self, individual_configs: List[Dict]) -> Dict:
        """
        Create beforeenrichment.json - same configs as after but with empty Dest arrays
        Shows the state BEFORE database enrichment is applied
        """
        import copy
        
        before_config = {
            "EnrichConfigs": {
                "MsgEnrich": []
            }
        }
        
        # Copy all configs but clear Dest arrays to show "before enrichment" state
        for config_item in individual_configs:
            config_data = config_item.get('config', {})
            
            # Check if config has EnrichConfigs structure
            if 'EnrichConfigs' in config_data and 'MsgEnrich' in config_data['EnrichConfigs']:
                for enrich_entry in config_data['EnrichConfigs']['MsgEnrich']:
                    # Deep copy to avoid modifying original
                    before_entry = copy.deepcopy(enrich_entry)
                    
                    # Clear Dest array to show "not yet enriched" state
                    if 'Dest' in before_entry:
                        before_entry['Dest'] = []
                    
                    # Remove internal tracking fields
                    if '_source_file' in before_entry:
                        del before_entry['_source_file']
                    
                    before_config['EnrichConfigs']['MsgEnrich'].append(before_entry)
        
        print(f"    Created before enrichment config with {len(before_config['EnrichConfigs']['MsgEnrich'])} entries (Dest arrays empty)")
        return before_config



    def _create_after_enrichment_config(self, individual_configs: List[Dict]) -> Dict:
        """
        Consolidate all individual configs into single afterenrichment.json
        Shows the state AFTER database enrichment is applied (with Dest populated)
        """
        import copy
        
        consolidated_config = {
            "EnrichConfigs": {
                "MsgEnrich": []
            }
        }
        
        # Extract and consolidate all MsgEnrich configurations
        for config_item in individual_configs:
            config_data = config_item.get('config', {})
            
            # Check if config has EnrichConfigs structure
            if 'EnrichConfigs' in config_data and 'MsgEnrich' in config_data['EnrichConfigs']:
                for enrich_entry in config_data['EnrichConfigs']['MsgEnrich']:
                    # Deep copy to preserve Dest arrays (showing enriched state)
                    after_entry = copy.deepcopy(enrich_entry)
                    
                    # Remove internal tracking fields
                    if '_source_file' in after_entry:
                        del after_entry['_source_file']
                    
                    consolidated_config['EnrichConfigs']['MsgEnrich'].append(after_entry)
        
        print(f"    Consolidated {len(consolidated_config['EnrichConfigs']['MsgEnrich'])} configs (Dest arrays populated)")
        return consolidated_config
    


    def _write_consolidated_enrichment_files(self, project_dir: Path, before_config: Dict, after_config: Dict):
        """Write the final BeforeEnrichmentConf.json and AfterEnrichmentConf.json files"""
        enrichment_dir = project_dir / "enrichment"
        
        try:
            # Write BeforeEnrichmentConf.json
            before_file = enrichment_dir / "BeforeEnrichmentConf.json"
            with open(before_file, 'w', encoding='utf-8') as f:
                json.dump(before_config, f, indent=2, ensure_ascii=False)
            print(f"    📄 Created: BeforeEnrichmentConf.json")
            
            # Write AfterEnrichmentConf.json  
            after_file = enrichment_dir / "AfterEnrichmentConf.json"
            with open(after_file, 'w', encoding='utf-8') as f:
                json.dump(after_config, f, indent=2, ensure_ascii=False)
            print(f"    📄 Created: AfterEnrichmentConf.json")
            
            # Summary
            before_count = len(before_config['EnrichConfigs']['MsgEnrich'])
            after_count = len(after_config['EnrichConfigs']['MsgEnrich'])
            print(f"    📊 Before: {before_count} configs, After: {after_count} configs")
            
        except Exception as e:
            print(f"    ❌ Error writing consolidated files: {e}")
            self.validation_errors.append(f"File write error: {e}")
            raise


    def run_smart_review(self) -> str:
        """
        Main method - Create ACE project with correct naming conventions and error checking
        Returns: Path to final ACE project directory
        """
        print("🏗️ Creating ACE Project Structure with Correct Naming...")
        print("=" * 55)
        
        try:
            # Step 1: Load naming convention from PDF extraction
            self.naming_convention = self._load_naming_convention()
            
            # Step 2: Discover existing ACE components from Agent 3
            self.discovered_components = self._discover_ace_components()
            print(f"Found {len(self.discovered_components)} ACE components")
            
            # Step 3: Validate components and check for errors
            self._validate_components()
            
            # Step 4: Create final project structure with correct names
            final_project_path = self._create_ace_project_structure()
            self.validate_and_fix_messageflow_components(Path(final_project_path))
            
            # Step 5: Generate comprehensive project files
            self._generate_complete_project_files(Path(final_project_path))
            
            # Step 6: Report results
            self._report_final_results(final_project_path)
            
            return final_project_path
            
        except Exception as e:
            print(f"❌ ACE project creation failed: {str(e)}")
            # Create error directory for debugging
            error_dir = self.ace_components_folder.parent / f"ERROR_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(error_dir, exist_ok=True)
            self._create_error_report(error_dir, str(e))
            return str(error_dir)
    
    def _load_naming_convention(self) -> Dict:
        """Load naming convention from fetch_naming.py output"""
        naming_file = Path("naming_convention.json")
        
        try:
            if naming_file.exists():
                with open(naming_file, 'r', encoding='utf-8') as f:
                    naming_data = json.load(f)
                
                # Extract the key naming information
                project_naming = naming_data.get('project_naming', {})
                ace_app_name = project_naming.get('ace_application_name', '')
                message_flow_name = project_naming.get('message_flow_name', '')
                
                print(f"📄 Loaded naming convention from PDF:")
                print(f"  Project: {ace_app_name}")
                print(f"  Message Flow: {message_flow_name}")
                
                return naming_data
            else:
                print("⚠️ naming_convention.json not found - using fallback naming")
                return self._create_fallback_naming()
                
        except Exception as e:
            print(f"⚠️ Error reading naming_convention.json: {e}")
            return self._create_fallback_naming()
    
    def _create_fallback_naming(self) -> Dict:
        """Create fallback naming if PDF extraction failed"""
        timestamp = datetime.now().strftime("%m%d_%H%M")
        fallback_data = {
            "project_naming": {
                "ace_application_name": f"EPIS_MIGRATION_App_{timestamp}",
                "message_flow_name": f"MIGRATION_Flow_{timestamp}",
                "connected_system": "Unknown_System",
                "description": "Migrated ACE Application"
            },
            "component_naming_rules": {
                "msgflow_files": f"MIGRATION_Flow_{timestamp}.msgflow",
                "project_name": f"EPIS_MIGRATION_App_{timestamp}"
            }
        }
        print("🔄 Using fallback naming convention")
        return fallback_data
    
    def _discover_ace_components(self) -> List[Dict]:
        """Discover ACE components from Agent 3 output with error checking"""
        print("\n🔍 Discovering ACE components from Agent 3...")
        
        components = []
        
        if not self.ace_components_folder.exists():
            error_msg = f"Components folder not found: {self.ace_components_folder}"
            print(f"❌ {error_msg}")
            self.validation_errors.append(error_msg)
            return components
        
        # Scan for ACE component files
        for root, dirs, files in os.walk(self.ace_components_folder):
            for file in files:
                file_path = Path(root) / file
                file_extension = file_path.suffix.lower()
                
                # Check if it's an ACE component
                if file_extension in self.ace_component_types:

                    if 'template' in file.lower() or 'template_updated' in file.lower():
                        print(f"    🚫 Skipping template file: {file}")
                        continue


                    try:
                        # Validate file accessibility
                        file_size = file_path.stat().st_size if file_path.exists() else 0
                        
                        # Check for empty files
                        if file_size == 0:
                            self.validation_errors.append(f"Empty file detected: {file_path.name}")
                        
                        # Try to read file to validate it's not corrupted
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content_preview = f.read(100)  # Read first 100 chars
                        
                        component = {
                            'name': file_path.stem,
                            'full_name': file_path.name,
                            'original_path': str(file_path),
                            'extension': file_extension,
                            'component_type': self.ace_component_types[file_extension],
                            'file_size': file_size,
                            'content_preview': content_preview,
                            'readable': True
                        }
                        components.append(component)
                        print(f"  ✅ {component['full_name']} ({component['component_type']}) - {file_size} bytes")
                        
                    except Exception as e:
                        error_msg = f"Error reading {file_path.name}: {str(e)}"
                        print(f"  ❌ {error_msg}")
                        self.component_errors[file_path.name] = str(e)
                        
                        # Still add component but mark as unreadable
                        component = {
                            'name': file_path.stem,
                            'full_name': file_path.name,
                            'original_path': str(file_path),
                            'extension': file_extension,
                            'component_type': self.ace_component_types[file_extension],
                            'file_size': 0,
                            'content_preview': 'ERROR: File not readable',
                            'readable': False
                        }
                        components.append(component)
        
        # Check for minimum required components
        msgflow_count = len([c for c in components if c['extension'] == '.msgflow'])
        if msgflow_count == 0:
            self.validation_errors.append("No message flow (.msgflow) files found")
        
        # Skip enrichment JSON files (keep them as-is)
        enrichment_files = list(Path(root).glob('**/*.json')) if components else []
        if enrichment_files:
            print(f"  📁 Enrichment files found: {len(enrichment_files)} (will be preserved)")
        
        return components
    

    
    def _create_ace_project_structure(self) -> str:
        """Create final ACE project structure with organized subdirectories"""
        print("\n🗃️ Creating ACE Project Structure with Organized Folders...")
        compute_mapping = {} 
        # Get dynamic names from naming convention
        project_naming = self.naming_convention.get('project_naming', {})
        project_name = project_naming.get('ace_application_name', 'EPIS_MIGRATION_App')
        message_flow_base = project_naming.get('message_flow_name', 'MIGRATION_Flow')
        
        
        # Create main project directory (DYNAMIC name)
        project_dir = self.ace_components_folder.parent / f"{project_name}"
        
        try:
            # STEP 2: Create main project directory
            os.makedirs(project_dir, exist_ok=True)
            print(f"📁 Created main project directory: {project_name}")
            
            # STEP 3: Create organized subdirectories
            # Dynamic ESQL subdirectory name
            esql_dir = project_dir / message_flow_base
            # Hardcoded subdirectory names
            schemas_dir = project_dir / "schemas"
            transforms_dir = project_dir / "transforms"
            enrichment_dir = project_dir / "enrichment"
            
            # Create all subdirectories
            subdirs_created = []
            for subdir_name, subdir_path in [
                (message_flow_base, esql_dir),      # Dynamic ESQL folder
                ("schemas", schemas_dir),           # Hardcoded
                ("transforms", transforms_dir),     # Hardcoded  
                ("enrichment", enrichment_dir)      # Hardcoded
            ]:
                os.makedirs(subdir_path, exist_ok=True)
                subdirs_created.append(subdir_name)
                print(f"  📂 Created subdirectory: {subdir_name}/")
            
            print(f"📊 Created {len(subdirs_created)} organized subdirectories")
            

            # STEP 4: Copy and organize components by file type
            components_copied = 0
            file_routing_summary = {
                'esql_files': 0,
                'msgflow_files': 0,  # 🆕 ADD THIS LINE
                'xsd_files': 0, 
                'xsl_files': 0,
                'other_files': 0
            }

            for component in self.discovered_components:
                old_path = Path(component['original_path'])
                
                # ✅ Keep original filename for ESQL and XSL files
                if component['extension'] in ['.esql', '.xsl']:
                    new_name = component['full_name']  # Keep original filename - NO renaming
                else:
                    new_name = self._generate_component_name(component, message_flow_base)
                
                # 🗂️ ROUTE FILES TO APPROPRIATE SUBDIRECTORIES
                if component['extension'] == '.esql':
                    # Use original filename (no renaming)
                    original_filename = component['full_name']  # Keep original name
                    new_path = esql_dir / original_filename
                    file_routing_summary['esql_files'] += 1
                    destination = f"{message_flow_base}/"
                    
                    try:
                        # Simple file copy - no content changes, no name changes
                        shutil.copy2(old_path, new_path)
                        components_copied += 1
                        print(f"  ✅ {component['full_name']} → {destination}{original_filename}")
                    except Exception as e:
                        print(f"  ❌ Failed to copy {component['full_name']}: {e}")
                
                # 🆕 NEW SECTION: ADD THIS ENTIRE BLOCK RIGHT HERE
                elif component['extension'] == '.msgflow':
                    new_path = project_dir / new_name
                    file_routing_summary['msgflow_files'] += 1
                    destination = "root/"
                    
                    try:
                        # 🆕 UPDATED LOGIC FOR MSGFLOW FILES - Update computeExpression references
                        print(f"    🔧 Processing MSGFLOW: {component['full_name']} → {new_name}")
                        
                        # Use NEW helper method to update msgflow computeExpression references
                        updated_content = self._update_msgflow_references(old_path, compute_mapping, message_flow_base)
                        final_content = self._fix_msgflow_gif_references(updated_content, message_flow_base)
                        
                        # Write updated content to new file
                        with open(new_path, 'w', encoding='utf-8') as f:
                            f.write(final_content)
                        
                        components_copied += 1
                        print(f"  ✅ {component['full_name']} → {destination}{new_name} (references updated)")
                        
                    except Exception as e:
                        print(f"  ❌ Failed to update MSGFLOW {component['full_name']}: {e}")
                        # Fallback: copy original file without content updates
                        try:
                            shutil.copy2(old_path, new_path)
                            components_copied += 1
                            print(f"  ⚠️ {component['full_name']} → {destination}{new_name} (fallback copy)")
                        except Exception as fallback_error:
                            print(f"  ❌ Fallback copy also failed: {fallback_error}")
                # 🆕 END OF NEW SECTION
                    
                elif component['extension'] == '.xsd':
                    new_path = schemas_dir / new_name
                    file_routing_summary['xsd_files'] += 1
                    destination = "schemas/"
                    
                    try:
                        # Regular file copy for XSD files
                        shutil.copy2(old_path, new_path)
                        components_copied += 1
                        print(f"  ✅ {component['full_name']} → {destination}{new_name}")
                    except Exception as e:
                        print(f"  ❌ Failed to copy {component['full_name']}: {e}")
                                    

                elif component['extension'] == '.xsl':
                    # ✅ Keep original XSL filename - no renaming
                    original_filename = component['full_name']  # Use original name
                    new_path = transforms_dir / original_filename  # Copy with original name
                    file_routing_summary['xsl_files'] += 1
                    destination = "transforms/"
                    
                    try:
                        # Simple file copy - NO renaming, NO content changes
                        shutil.copy2(old_path, new_path)
                        components_copied += 1
                        print(f"  ✅ {component['full_name']} → {destination}{original_filename}")
                    except Exception as e:
                        print(f"  ❌ Failed to copy {component['full_name']}: {e}")
                    
                else:
                    # Other files go to root directory
                    new_path = project_dir / new_name
                    file_routing_summary['other_files'] += 1
                    destination = "root/"
                    
                    try:
                        # Regular file copy for other files
                        shutil.copy2(old_path, new_path)
                        components_copied += 1
                        print(f"  ✅ {component['full_name']} → {destination}{new_name}")
                    except Exception as e:
                        print(f"  ❌ Failed to copy {component['full_name']}: {e}")
            
            # STEP 5: Handle enrichment files (preserve existing logic)
            print(f"\n📄 Processing enrichment files...")
            self._copy_enrichment_files(project_dir)
            
            # STEP 6: Generate complete project files (root level)
            print(f"📋 Generating ACE project files...")
            self._generate_complete_project_files(project_dir)
            
            # STEP 7: Final summary with organized structure info
            print(f"\n📊 Project Creation Summary:")
            print(f"  📦 Project: {project_name}")
            print(f"  📁 ESQL Directory: {message_flow_base}/ ({file_routing_summary['esql_files']} files)")
            print(f"  📊 Message Flow Files: ({file_routing_summary['msgflow_files']} files)")  
            print(f"  📁 Schemas Directory: schemas/ ({file_routing_summary['xsd_files']} files)")
            print(f"  📁 Transforms Directory: transforms/ ({file_routing_summary['xsl_files']} files)")
            print(f"  📁 Enrichment Directory: enrichment/ (consolidated JSON files)")
            print(f"  📄 Root Files: {file_routing_summary['other_files']} + project files")
            print(f"  🔗 Message Flow Mapping: {len(compute_mapping)} compute references analyzed")
            
            if self.validation_errors:
                print(f"  ⚠️ Issues: {len(self.validation_errors)} validation warnings")
            
            print(f"\n✅ Organized ACE project structure created successfully!")
            return str(project_dir)
            
        except Exception as e:
            print(f"❌ Error creating project structure: {e}")
            self.validation_errors.append(f"Project structure creation failed: {e}")
            raise Exception(f"Failed to create ACE project structure: {e}")
    



    def _generate_component_name(self, component: Dict, message_flow_base: str) -> str:
        """Generate correct component name based on ACE naming conventions"""
        extension = component['extension']
        original_name = component['full_name']
        
        # Apply naming convention based on component type
        if extension == '.msgflow':
            # Message flows: Use exact message flow name from PDF
            return f"{message_flow_base}.msgflow"
            
        
        elif extension == '.xsl':
            return original_name
        
        elif extension == '.xsd':
            # Schema files: {MESSAGE_FLOW_BASE}_{PURPOSE}Schema.xsd
            if 'input' in original_name.lower():
                return f"{message_flow_base}_InputSchema.xsd"
            elif 'output' in original_name.lower():
                return f"{message_flow_base}_OutputSchema.xsd"
            elif 'error' in original_name.lower():
                return f"{message_flow_base}_ErrorSchema.xsd"
            else:
                base_name = component['name'].replace('Schema', '').replace('_Schema', '')
                return f"{message_flow_base}_{base_name}Schema.xsd"
        
        else:
            # Other files: keep original name or apply simple prefix
            return original_name
        


    
    def _copy_enrichment_files(self, project_dir: Path):
        """Process enrichment configurations - consolidate into before/after files only"""
        print(f"\n📄 Processing enrichment files...")
        
        # Look for .json files in the Enhanced_ACE_Project subfolder (where Agent3 outputs them)
        enhanced_project_folder = self.ace_components_folder / "Enhanced_ACE_Project"
        
        # First try the Enhanced_ACE_Project subfolder
        if enhanced_project_folder.exists():
            json_files = list(enhanced_project_folder.glob("*.json"))
            print(f"  📁 Searching in: {enhanced_project_folder}")
        else:
            # Fallback to original folder if Enhanced_ACE_Project doesn't exist
            json_files = list(self.ace_components_folder.glob("*.json"))
            print(f"  📁 Searching in: {self.ace_components_folder}")
        
        if not json_files:
            print(f"  ℹ️ No JSON files found for enrichment")
            return
        
        # Filter out the component mapping file - only process enrichment files
        enrichment_files = []
        for json_file in json_files:
            if json_file.name != "biztalk_ace_component_mapping.json":
                enrichment_files.append(json_file)
            else:
                print(f"  📋 Skipping mapping file: {json_file.name}")
        
        if not enrichment_files:
            print(f"  ℹ️ No enrichment files found (only mapping file present)")
            return
        
        print(f"  📋 Found {len(enrichment_files)} enrichment files to consolidate")
        for ef in enrichment_files:
            print(f"    • {ef.name}")
        
        # 🆕 UPDATED: Work with existing enrichment directory created by _create_ace_project_structure
        # The enrichment directory is already created by the new structure, so we just verify it exists
        enrichment_dest = project_dir / "enrichment"
        
        if not enrichment_dest.exists():
            # This shouldn't happen with the new structure, but create as fallback
            print(f"  ⚠️ Warning: enrichment directory not found, creating it...")
            os.makedirs(enrichment_dest, exist_ok=True)
        else:
            print(f"  ✅ Using existing enrichment directory: enrichment/")
        
        try:
            # 🆕 CONSOLIDATED APPROACH - Only create before/after files in the existing directory
            print(f"  📄 Consolidating enrichment configurations...")
            consolidation_success = self._consolidate_enrichment_files(project_dir, enrichment_files)
            
            if consolidation_success:
                print(f"  ✅ Enrichment consolidation completed successfully")
                print(f"  📁 Final enrichment folder contains:")
                print(f"    • beforeenrichment.json (empty template)")
                print(f"    • afterenrichment.json (consolidated configurations)")
            else:
                print(f"  ⚠️ Enrichment consolidation had issues - check logs")
            
        except Exception as e:
            print(f"  ❌ Error processing enrichment directory: {e}")
            self.validation_errors.append(f"Enrichment directory processing failed: {e}")

            
            

    def _update_msgflow_references(self, msgflow_file_path: Path, compute_mapping: Dict[str, str], message_flow_base: str) -> str:
        """
        Update .msgflow file references - simplified version
        Only updates namespace and project references since ESQL files keep original names
        """
        try:
            # Read original msgflow content
            with open(msgflow_file_path, 'r', encoding='utf-8') as f:
                msgflow_content = f.read()
            
            print(f"    🔧 Updating .msgflow references (simplified)...")
            
            import re
            updated_content = msgflow_content
            total_replacements = 0
            
            # Extract old msgflow name from nsURI
            msgflow_name_match = re.search(r'nsURI="([^"]+)\.msgflow"', msgflow_content)
            old_msgflow_name = msgflow_name_match.group(1) if msgflow_name_match else "AGENT2_Message_Flow"
            project_name = f"EPIS_{message_flow_base}_App"
            
            print(f"      📝 {old_msgflow_name} → {message_flow_base}")
            print(f"      📝 Project: {project_name}")
            
            # 1. Update namespace references
            namespace_patterns = [
                (f'nsURI="{old_msgflow_name}.msgflow"', f'nsURI="{message_flow_base}.msgflow"'),
                (f'nsPrefix="{old_msgflow_name}.msgflow"', f'nsPrefix="{message_flow_base}.msgflow"')
            ]
            
            for old_pattern, new_pattern in namespace_patterns:
                if old_pattern in updated_content:
                    updated_content = updated_content.replace(old_pattern, new_pattern)
                    total_replacements += 1
                    print(f"      ✅ Namespace: {old_pattern} → {new_pattern}")
            
            # 2. Update project references (simple patterns only)
            project_patterns = [
                ('AGENT2_App_Name', project_name),
                (f'platform:/plugin/AGENT2_App_Name/', f'platform:/plugin/{project_name}/'),
                (f'pluginId="AGENT2_App_Name"', f'pluginId="{project_name}"')
            ]
            
            for old_pattern, new_pattern in project_patterns:
                if old_pattern in updated_content:
                    updated_content = updated_content.replace(old_pattern, new_pattern)
                    total_replacements += 1
                    print(f"      ✅ Project: {old_pattern} → {new_pattern}")
            
            # 3. Update GIF references
            gif_pattern = f'{old_msgflow_name}.gif'
            new_gif = f'{message_flow_base}.gif'
            if gif_pattern in updated_content:
                updated_content = updated_content.replace(gif_pattern, new_gif)
                total_replacements += 1
                print(f"      ✅ GIF: {gif_pattern} → {new_gif}")
            
            # 4. Skip computeExpression updates (ESQL files keep original names)
            print(f"      ℹ️ Skipped computeExpression updates (ESQL files not renamed)")
            
            # Summary
            if total_replacements > 0:
                print(f"    🎯 Total updates: {total_replacements}")
            else:
                print(f"    ℹ️ No references found to update")
            
            return updated_content
            
        except UnicodeDecodeError:
            # Try alternative encoding
            try:
                with open(msgflow_file_path, 'r', encoding='cp1252') as f:
                    return self._update_msgflow_references(msgflow_file_path, compute_mapping, message_flow_base)
            except Exception:
                print(f"      ❌ Encoding error in {msgflow_file_path.name}")
                return msgflow_content
                
        except Exception as e:
            print(f"      ❌ Error updating {msgflow_file_path.name}: {e}")
            # Return original content as fallback
            try:
                with open(msgflow_file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except:
                return ""



    
    def _generate_project_file(self, project_dir: Path, project_name: str):
        """Generate .project file for ACE Toolkit"""
        project_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<projectDescription>
	<name>{project_name}</name>
	<comment></comment>
	<projects>
		<project>EPIS_CommonUtils_Lib</project>
		<project>EPIS_Consumer_Lib_v2</project>
		<project>EPIS_BlobStorage_Lib</project>
		<project>EPIS_MessageEnrichment_StaticLib</project>
		<project>EPIS_CommonFlows_Lib</project>
        <project>EPIS_CargoWiseOne_eAdapter_Lib</project>
        <project>EPIS_CargoWiseOne_Schemas_Lib</project>
	</projects>
	<buildSpec>
		<buildCommand>
			<name>com.ibm.etools.mft.applib.applibbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.applib.applibresourcevalidator</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.connector.policy.ui.PolicyBuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.applib.mbprojectbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.msg.validation.dfdl.mlibdfdlbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.flow.adapters.adapterbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.flow.sca.scabuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.msg.validation.dfdl.mbprojectresourcesbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.esql.lang.esqllangbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.map.builder.mslmappingbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.flow.msgflowxsltbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.flow.msgflowbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.decision.service.ui.decisionservicerulebuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.pattern.capture.PatternBuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.json.builder.JSONBuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.restapi.ui.restApiDefinitionsBuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.policy.ui.policybuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.msg.assembly.messageAssemblyBuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.msg.validation.dfdl.dfdlqnamevalidator</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.bar.ext.barbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.unittest.ui.TestCaseBuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
	</buildSpec>
	<natures>
		<nature>com.ibm.etools.msgbroker.tooling.applicationNature</nature>
		<nature>com.ibm.etools.msgbroker.tooling.messageBrokerProjectNature</nature>
	</natures>
</projectDescription>"""
        
        with open(project_dir / ".project", 'w', encoding='utf-8') as f:
            f.write(project_content)
        print(f"  📄 Generated: .project")
    
    def _generate_application_descriptor(self, project_dir: Path, project_name: str):
        """Generate application.descriptor file"""
        # Get description from naming convention
        description = self.naming_convention.get('project_naming', {}).get('description', 'Migrated ACE Application')
        
        descriptor_content = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<ns2:appDescriptor xmlns:ns2="http://com.ibm.etools.mft.descriptor.base">
    <ns2:application>
        <ns2:name>{project_name}</ns2:name>
        <ns2:description>{description}</ns2:description>
        <ns2:version>1.0.0</ns2:version>
    </ns2:application>
</ns2:appDescriptor>"""
        
        with open(project_dir / "application.descriptor", 'w', encoding='utf-8') as f:
            f.write(descriptor_content)
        print(f"  📄 Generated: application.descriptor")

    def _validate_components(self):
        """Validate discovered components for common issues"""
        print("\n🔍 Validating ACE components...")
        
        if not self.discovered_components:
            self.validation_errors.append("No ACE components found for validation")
            return
        
        # Check for required component types
        component_types_found = set(c['extension'] for c in self.discovered_components)
        
        if '.msgflow' not in component_types_found:
            self.validation_errors.append("Missing message flow (.msgflow) files - Required for ACE application")
        
        # Check for unreadable components
        unreadable_components = [c for c in self.discovered_components if not c.get('readable', True)]
        if unreadable_components:
            for comp in unreadable_components:
                self.validation_errors.append(f"Unreadable component: {comp['full_name']}")
        
        # Check for suspicious file sizes
        for component in self.discovered_components:
            if component['file_size'] == 0:
                self.validation_errors.append(f"Empty component file: {component['full_name']}")
            elif component['file_size'] > 10 * 1024 * 1024:  # 10MB
                self.validation_errors.append(f"Unusually large component: {component['full_name']} ({component['file_size']} bytes)")
        
        # Report validation results
        if self.validation_errors:
            print(f"  ⚠️ Validation issues found: {len(self.validation_errors)}")
            for error in self.validation_errors[:5]:  # Show first 5 errors
                print(f"    • {error}")
            if len(self.validation_errors) > 5:
                print(f"    ... and {len(self.validation_errors) - 5} more issues")
        else:
            print(f"  ✅ All components validated successfully")
    
    def _generate_complete_project_files(self, project_dir: Path):
        """Generate comprehensive ACE project files"""
        project_naming = self.naming_convention.get('project_naming', {})
        project_name = project_naming.get('ace_application_name', 'EPIS_MIGRATION_App')
        
        print(f"\n📄 Generating ACE project files...")
        
        # Generate enhanced .project file
        self._generate_enhanced_project_file(project_dir, project_name)
        
        # Generate enhanced application.descriptor
        self._generate_enhanced_application_descriptor(project_dir, project_name)
        
        # Generate project properties file
        self._generate_project_properties(project_dir, project_name)
    
    def _generate_enhanced_project_file(self, project_dir: Path, project_name: str):
        """Generate enhanced .project file with all required ACE configurations"""
        
        # Count different component types for project configuration
        msgflow_count = len([c for c in self.discovered_components if c['extension'] == '.msgflow'])
        esql_count = len([c for c in self.discovered_components if c['extension'] == '.esql'])
        
        project_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<projectDescription>
	<name>{project_name}</name>
	<comment></comment>
	<projects>
		<project>EPIS_CommonUtils_Lib</project>
		<project>EPIS_Consumer_Lib_v2</project>
		<project>EPIS_BlobStorage_Lib</project>
		<project>EPIS_MessageEnrichment_StaticLib</project>
		<project>EPIS_CommonFlows_Lib</project>
        <project>EPIS_CargoWiseOne_eAdapter_Lib</project>
        <project>EPIS_CargoWiseOne_Schemas_Lib</project>
	</projects>
	<buildSpec>
		<buildCommand>
			<name>com.ibm.etools.mft.applib.applibbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.applib.applibresourcevalidator</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.connector.policy.ui.PolicyBuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.applib.mbprojectbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.msg.validation.dfdl.mlibdfdlbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.flow.adapters.adapterbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.flow.sca.scabuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.msg.validation.dfdl.mbprojectresourcesbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.esql.lang.esqllangbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.map.builder.mslmappingbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.flow.msgflowxsltbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.flow.msgflowbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.decision.service.ui.decisionservicerulebuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.pattern.capture.PatternBuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.json.builder.JSONBuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.restapi.ui.restApiDefinitionsBuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.policy.ui.policybuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.msg.assembly.messageAssemblyBuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.msg.validation.dfdl.dfdlqnamevalidator</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.bar.ext.barbuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
		<buildCommand>
			<name>com.ibm.etools.mft.unittest.ui.TestCaseBuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
	</buildSpec>
	<natures>
		<nature>com.ibm.etools.msgbroker.tooling.applicationNature</nature>
		<nature>com.ibm.etools.msgbroker.tooling.messageBrokerProjectNature</nature>
	</natures>
</projectDescription>"""
        
        with open(project_dir / ".project", 'w', encoding='utf-8') as f:
            f.write(project_content)
        print(f"  ✅ Enhanced .project file generated")
    
    def _generate_enhanced_application_descriptor(self, project_dir: Path, project_name: str):
        """Generate enhanced application.descriptor with metadata"""
        project_naming = self.naming_convention.get('project_naming', {})
        
        description = project_naming.get('description', 'Migrated ACE Application')
        connected_system = project_naming.get('connected_system', 'Unknown System')
        
        descriptor_content = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<ns2:appDescriptor xmlns:ns2="http://com.ibm.etools.mft.descriptor.base">
    <ns2:application>
        <ns2:name>{project_name}</ns2:name>
        <ns2:description>{description}</ns2:description>
        <ns2:version>1.0.0</ns2:version>
        <ns2:additionalInformation>Connected System: {connected_system}</ns2:additionalInformation>
        <ns2:additionalInformation>Migration Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</ns2:additionalInformation>
        <ns2:additionalInformation>Components: {len(self.discovered_components)} ACE artifacts</ns2:additionalInformation>
    </ns2:application>
</ns2:appDescriptor>"""
        
        with open(project_dir / "application.descriptor", 'w', encoding='utf-8') as f:
            f.write(descriptor_content)
        print(f"  ✅ Enhanced application.descriptor generated")
    
    def _generate_project_properties(self, project_dir: Path, project_name: str):
        """Generate project.properties file for ACE deployment"""
        properties_content = f"""# ACE Application Properties
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

application.name={project_name}
application.version=1.0.0
application.description=Migrated from BizTalk
deployment.target=integration_server
"""
        
        with open(project_dir / "project.properties", 'w', encoding='utf-8') as f:
            f.write(properties_content)
        print(f"  📄 project.properties generated")
    
    def _report_final_results(self, final_project_path: str):
        """Generate comprehensive final report"""
        print(f"\n📊 Final Project Report:")
        print(f"  📁 Project: {Path(final_project_path).name}")
        print(f"  📄 Components: {len(self.discovered_components)} total")
        
        # Component breakdown
        component_counts = {}
        for comp in self.discovered_components:
            comp_type = comp['component_type']
            component_counts[comp_type] = component_counts.get(comp_type, 0) + 1
        
        for comp_type, count in component_counts.items():
            print(f"    • {comp_type}: {count}")
        
        # Error summary
        if self.validation_errors:
            print(f"  ⚠️ Validation Issues: {len(self.validation_errors)}")
        else:
            print(f"  ✅ Validation: All components valid")
        
        if self.component_errors:
            print(f"  ❌ Component Errors: {len(self.component_errors)}")
        else:
            print(f"  ✅ Component Processing: All successful")
        
        print(f"  🎯 ACE Toolkit Ready: Yes")
        print(f"  📍 Import Path: {final_project_path}")
    
    def _create_error_report(self, error_dir: Path, main_error: str):
        """Create detailed error report for debugging"""
        error_report = {
            "timestamp": datetime.now().isoformat(),
            "main_error": main_error,
            "validation_errors": self.validation_errors,
            "component_errors": self.component_errors,
            "discovered_components": len(self.discovered_components),
            "components_folder": str(self.ace_components_folder),
            "naming_convention_loaded": bool(self.naming_convention)
        }
        
        with open(error_dir / "error_report.json", 'w', encoding='utf-8') as f:
            json.dump(error_report, f, indent=2)
        
        print(f"  📄 Error report saved: {error_dir}/error_report.json")


    def _fix_msgflow_gif_references(self, msgflow_content: str, message_flow_base: str) -> str:
        """
        Simple post-processing to fix remaining GIF filename references
        """
        # Only fix the GIF filenames - everything else already updated
        updated_content = msgflow_content.replace(
            'AGENT2_Message_Flow.gif', 
            f'{message_flow_base}.gif'
        )
        return updated_content




    def validate_and_fix_messageflow_components(self, project_dir: Path) -> Dict:
        """
        Validate messageflow components and generate missing files.
        """
        import xml.etree.ElementTree as ET
        from groq import Groq
        import json
        
        print("\n🔍 Validating MessageFlow Components...")
        print("=" * 70)
        
        groq_model = os.getenv('GROQ_MODEL')
        if not groq_model:
            print("  ❌ ERROR: GROQ_MODEL environment variable not set")
            return {'total_found': 0, 'total_missing': 0, 'total_generated': 0, 'missing_files': []}
        
        groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        results = {'total_found': 0, 'total_missing': 0, 'total_generated': 0, 'missing_files': []}
        
        # Inline method for WSDL generation only
        def generate_wsdl_from_vector_db(vector_content: str, soap_context: dict, wsdl_name: str) -> str:
            """
            Generate WSDL by extracting structured data from Vector DB.
            Only called when WSDL file is missing.
            """
            # Step 1: Extract structured data from Vector DB
            extraction_prompt = f"""Extract WSDL specifications from these business requirements and service details.

            BUSINESS REQUIREMENTS:
            {vector_content}

            SERVICE DETAILS:
            - URL: {soap_context.get('service_url', 'http://tempuri.org')}
            - Operation: {soap_context.get('operation', 'ProcessMessage')}
            - Binding: {soap_context.get('binding', 'BasicHttpBinding')}
            - Port Type: {soap_context.get('port_type', 'ServicePort')}

            EXTRACTION RULES:
            1. Extract actual values from business requirements - DO NOT use placeholders
            2. For namespace: Use the service URL domain as namespace (e.g., http://eadapterqa.dsv.com/)
            3. For service_name: Extract from URL path or use file basename
            4. For operation_name: Use the actual operation from Service Details
            5. For fields: Extract actual message structure from business requirements
            6. Use standard XSD types: xsd:string, xsd:int, xsd:boolean, xsd:dateTime
            7. If a field is not explicitly marked optional, set required=true

            Return ONLY this JSON structure with ACTUAL extracted values:
            {{
            "namespace": "extracted namespace URI",
            "service_name": "extracted service name",
            "port_type": "extracted port type",
            "operation_name": "extracted operation",
            "request_fields": [
                {{"name": "actual_field_name", "type": "xsd:type", "required": true}}
            ],
            "response_fields": [
                {{"name": "actual_field_name", "type": "xsd:type"}}
            ]
            }}

            DO NOT include markdown, explanations, or code syntax. Return ONLY the JSON object."""

            try:
                extract_response = groq_client.chat.completions.create(
                    model=groq_model,
                    messages=[
                        {"role": "system", "content": "Extract structured WSDL data from requirements. Return only valid JSON."},
                        {"role": "user", "content": extraction_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )
                
                raw_response = extract_response.choices[0].message.content.strip()
                
                # Use robust LLM JSON parser
                parse_result = parse_llm_json(raw_response, debug=False)
                
                if parse_result.success:
                    wsdl_data = parse_result.data
                    print(f"       ✅ WSDL data parsed using {parse_result.method_used}")
                else:
                    print(f"       ⚠️ Data extraction failed: {parse_result.error_message}")
                    print(f"       📄 Raw response preview: {raw_response[:200]}...")
                    return None
                
            except Exception as e:
                print(f"       ❌ WSDL generation error: {e}")
                return None
            
            # Step 2: Build WSDL template with extracted data
            wsdl_template = f"""<?xml version="1.0" encoding="UTF-8"?>
    <wsdl:definitions 
        xmlns:wsdl="http://schemas.xmlsoap.org/wsdl/" 
        xmlns:xsd="http://www.w3.org/2001/XMLSchema" 
        xmlns:soap="http://schemas.xmlsoap.org/wsdl/soap/" 
        xmlns:tns="{wsdl_data.get('namespace', soap_context.get('service_url', 'http://tempuri.org'))}"
        targetNamespace="{wsdl_data.get('namespace', soap_context.get('service_url', 'http://tempuri.org'))}">
        
        <wsdl:types>
            <xsd:schema targetNamespace="{wsdl_data.get('namespace', soap_context.get('service_url', 'http://tempuri.org'))}">
                
                <xsd:element name="{wsdl_data.get('operation_name', 'Request')}">
                    <xsd:complexType>
                        <xsd:sequence>"""
            
            # Add request fields
            for field in wsdl_data.get('request_fields', []):
                min_occurs = ' minOccurs="0"' if not field.get('required', True) else ''
                wsdl_template += f"""
                            <xsd:element name="{field['name']}" type="{field['type']}"{min_occurs}/>"""
            
            # ✅ FIXED: Use f-string for response element section
            wsdl_template += f"""
                        </xsd:sequence>
                    </xsd:complexType>
                </xsd:element>
                
                <xsd:element name="{wsdl_data.get('operation_name', 'Response')}Response">
                    <xsd:complexType>
                        <xsd:sequence>"""
            
            # Add response fields
            for field in wsdl_data.get('response_fields', []):
                wsdl_template += f"""
                            <xsd:element name="{field['name']}" type="{field['type']}" minOccurs="0"/>"""
            
            wsdl_template += f"""
                        </xsd:sequence>
                    </xsd:complexType>
                </xsd:element>
                
            </xsd:schema>
        </wsdl:types>
        
        <wsdl:message name="{wsdl_data.get('operation_name', 'Request')}Message">
            <wsdl:part name="parameters" element="tns:{wsdl_data.get('operation_name', 'Request')}"/>
        </wsdl:message>
        
        <wsdl:message name="{wsdl_data.get('operation_name', 'Response')}ResponseMessage">
            <wsdl:part name="parameters" element="tns:{wsdl_data.get('operation_name', 'Response')}Response"/>
        </wsdl:message>
        
        <wsdl:portType name="{wsdl_data.get('port_type', soap_context.get('port_type', 'ServicePort'))}">
            <wsdl:operation name="{wsdl_data.get('operation_name', 'ProcessMessage')}">
                <wsdl:input message="tns:{wsdl_data.get('operation_name', 'Request')}Message"/>
                <wsdl:output message="tns:{wsdl_data.get('operation_name', 'Response')}ResponseMessage"/>
            </wsdl:operation>
        </wsdl:portType>
        
        <wsdl:binding name="{soap_context.get('binding', 'BasicHttpBinding')}" type="tns:{wsdl_data.get('port_type', soap_context.get('port_type', 'ServicePort'))}">
            <soap:binding style="document" transport="http://schemas.xmlsoap.org/soap/http"/>
            <wsdl:operation name="{wsdl_data.get('operation_name', 'ProcessMessage')}">
                <soap:operation soapAction="{wsdl_data.get('operation_name', 'ProcessMessage')}"/>
                <wsdl:input>
                    <soap:body use="literal"/>
                </wsdl:input>
                <wsdl:output>
                    <soap:body use="literal"/>
                </wsdl:output>
            </wsdl:operation>
        </wsdl:binding>
        
        <wsdl:service name="{wsdl_data.get('service_name', 'Service')}">
            <wsdl:port name="{soap_context.get('port', 'ServicePort')}" binding="tns:{soap_context.get('binding', 'BasicHttpBinding')}">
                <soap:address location="{soap_context.get('service_url')}"/>
            </wsdl:port>
        </wsdl:service>
        
    </wsdl:definitions>"""
            
            return wsdl_template        
        # ... (existing directory discovery code) ...
        existing_subdirs = [d for d in project_dir.iterdir() if d.is_dir()]
        
        esql_subdir = None
        for subdir in existing_subdirs:
            if subdir.name not in ['schemas', 'transforms', 'enrichment', '.git']:
                esql_subdir = subdir
                break
        
        transforms_dir = project_dir / 'transforms'
        
        if not esql_subdir or not esql_subdir.exists():
            print("  ❌ ERROR: ESQL subdirectory not found")
            return results
        
        if not transforms_dir.exists():
            print("  ❌ ERROR: transforms directory not found")
            return results
        
        component_mapping = {}
        json_path = Path('output') / 'biztalk_ace_component_mapping.json'
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                component_mapping = json.load(f)
        
        try:
            msgflow_files = list(project_dir.glob('**/*.msgflow'))
            if not msgflow_files:
                print("  ⚠️  No messageflow files found")
                return results
            
            print(f"\n📋 Checking {len(msgflow_files)} messageflow file(s)\n")
            
            for msgflow_file in msgflow_files:
                print(f"  📄 Processing: {msgflow_file.name}")
                
                with open(msgflow_file, 'r', encoding='utf-8') as f:
                    msgflow_content = f.read()
                
                tree = ET.parse(msgflow_file)
                root = tree.getroot()
                
                flow_name = msgflow_file.stem
                flow_integration_details = {}
                
                for mapping in component_mapping.get('component_mappings', []):
                    if flow_name.lower() in str(mapping).lower():
                        flow_integration_details = mapping.get('ace_components', {}).get('integration_details', {})
                        break
                
                # Build dynamic search query
                search_keywords = []
                for node in root.iter():
                    if node.tag.endswith('nodes') or node.tag == 'nodes':
                        service_url = node.get('webServiceURL', '')
                        if service_url:
                            domain_parts = service_url.replace('https://', '').replace('http://', '').split('/')
                            if domain_parts:
                                search_keywords.append(domain_parts[0].split('.')[0])
                        operation = node.get('selectedOperation', '')
                        if operation:
                            search_keywords.append(operation)
                
                flow_tokens = flow_name.replace('_', ' ').split()
                search_keywords.extend(flow_tokens[:3])
                search_keywords.extend(['SOAP', 'service', 'integration', 'message'])
                
                dynamic_query = ' '.join(set(search_keywords))
                
                # Retrieve Vector DB content
                vector_content = ""
                # ✅ CORRECT CODE - USE THIS
                try:
                    if hasattr(st.session_state, 'vector_pipeline') and st.session_state.vector_pipeline:
                        if hasattr(st.session_state.vector_pipeline, 'search_engine') and st.session_state.vector_pipeline.search_engine:
                            vector_content = st.session_state.vector_pipeline.search_engine.get_agent_content(
                                agent_name='migration_quality_reviewer',
                                top_k=8
                            )
                            print(f"  ✅ Vector DB: {len(vector_content)} characters retrieved")
                        else:
                            print(f"  ⚠️ Vector DB search_engine not initialized")
                            vector_content = ""
                except Exception as e:
                    print(f"  ❌ Vector DB failed: {e}")

                
                components_to_check = []
                
                # Extract components
                for node in root.iter():
                    if node.tag.endswith('nodes') or node.tag == 'nodes':
                        compute_expr = node.get('computeExpression', '')
                        if 'esql://routine/#' in compute_expr:
                            module_name = compute_expr.split('#')[1].replace('.Main', '')
                            components_to_check.append(('ESQL', f"{module_name}.esql", module_name))
                        
                        xsl_name = node.get('stylesheetName', '')
                        if xsl_name:
                            components_to_check.append(('XSL', xsl_name, None))
                        
                        wsdl_name = node.get('wsdlFileName', '')
                        if wsdl_name:
                            soap_context = {
                                'service_url': node.get('webServiceURL', ''),
                                'operation': node.get('selectedOperation', ''),
                                'port_type': node.get('selectedPortType', ''),
                                'binding': node.get('selectedBinding', ''),
                                'port': node.get('selectedPort', ''),
                            }
                            components_to_check.append(('WSDL', wsdl_name, soap_context))
                
                # Validate each component
                for comp_data in components_to_check:
                    comp_type = comp_data[0]
                    comp_name = comp_data[1]
                    comp_context = comp_data[2]
                    
                    if comp_type == 'ESQL':
                        file_path = esql_subdir / comp_name
                    elif comp_type == 'XSL':
                        file_path = transforms_dir / comp_name
                    elif comp_type == 'WSDL':
                        file_path = project_dir / comp_name
                    
                    print(f"    🔍 {comp_name}")
                    
                    if file_path.exists():
                        results['total_found'] += 1
                        print(f"       ✅ EXISTS")
                    else:
                        results['total_missing'] += 1
                        results['missing_files'].append(comp_name)
                        print(f"       ❌ MISSING")
                        
                        if not file_path.parent.exists():
                            print(f"       ⚠️ Cannot generate - directory missing")
                            continue
                        
                        if not vector_content:
                            print(f"       ⚠️ SKIPPED - No Vector DB content")
                            continue
                        
                        # CONDITIONAL: Use special WSDL method if it's a WSDL file
                        if comp_type == 'WSDL':
                            print(f"       🤖 Generating WSDL with business data mapping...")
                            
                            wsdl_content = generate_wsdl_from_vector_db(
                                vector_content=vector_content,
                                soap_context=comp_context,
                                wsdl_name=comp_name
                            )
                            
                            if wsdl_content:
                                with open(file_path, 'w', encoding='utf-8') as f:
                                    f.write(wsdl_content)
                                results['total_generated'] += 1
                                print(f"       ✅ Generated with business data")
                            else:
                                print(f"       ❌ Generation failed")
                        
                        # For ESQL/XSL: Simple LLM generation (no special template)
                        else:
                            print(f"       🤖 Generating {comp_type}...")
                            
                            if comp_type == 'ESQL':
                                prompt = f"""Generate IBM ACE ESQL: {comp_name}
    Flow: {flow_name}
    Business Requirements: {vector_content}
    Return ONLY ESQL code."""
                            else:
                                prompt = f"""Generate XSLT 1.0: {comp_name}
    Flow: {flow_name}
    Business Requirements: {vector_content}
    Return ONLY XSLT code."""
                            
                            try:
                                response = groq_client.chat.completions.create(
                                    model=groq_model,
                                    messages=[
                                        {"role": "system", "content": "Generate code from requirements. No placeholders."},
                                        {"role": "user", "content": prompt}
                                    ],
                                    temperature=0.3,
                                    max_tokens=4000
                                )
                                
                                generated = response.choices[0].message.content.strip()
                                
                                if '```' in generated:
                                    parts = generated.split('```')
                                    for part in parts:
                                        if any(x in part.lower() for x in ['esql', 'xslt']):
                                            generated = '\n'.join(part.split('\n')[1:])
                                            break
                                    else:
                                        generated = parts[1] if len(parts) > 1 else generated
                                
                                with open(file_path, 'w', encoding='utf-8') as f:
                                    f.write(generated)
                                
                                results['total_generated'] += 1
                                print(f"       ✅ Generated")
                                
                            except Exception as e:
                                print(f"       ❌ Failed: {e}")
                
                print()
            
            print("=" * 70)
            print("📊 VALIDATION SUMMARY")
            print("=" * 70)
            print(f"  ✅ Found:      {results['total_found']}")
            print(f"  ❌ Missing:    {results['total_missing']}")
            print(f"  🤖 Generated:  {results['total_generated']}")
            print("=" * 70)
            
            return results
            
        except Exception as e:
            print(f"  ❌ Validation failed: {e}")
            return results
    



# Compatibility functions for main.py (unchanged interface)
def main():
    """Command line interface (for testing)"""
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python migration_quality_reviewer.py <ace_components_folder> <naming_standards_file> <vector_db_content>")
        return 1
    
    try:
        reviewer = SmartACEQualityReviewer(
            ace_components_folder=sys.argv[1],
            naming_standards_file=sys.argv[2], 
            vector_db_content=sys.argv[3]
        )
        
        final_project_path = reviewer.run_smart_review()
        print(f"\n✅ ACE Project Created: {final_project_path}")
        return 0
        
    except Exception as e:
        print(f"❌ Project creation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())