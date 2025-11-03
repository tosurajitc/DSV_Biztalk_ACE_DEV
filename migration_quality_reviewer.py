#!/usr/bin/env python3
"""
Migration Quality Reviewer - ACE Project Structure Finalizer
Creates production-ready IBM ACE project folders with proper naming conventions
and strictly filters out unwanted ACE components
"""

import os
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Set


class MigrationQualityReviewer:
    """Smart and efficient ACE project structure finalizer with strict component filtering"""
    
    def __init__(self, source_dir: str = "output/multiple"):
        """Initialize the migration reviewer"""
        self.source_dir = Path(source_dir)
        self.root_dir = Path.cwd()  # Project root directory
        
    def process_all_projects(self) -> List[str]:
        """Process all projects in the source directory"""
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")
        
        created_projects = []
        
        # Find all project folders
        project_folders = []
        for root, dirs, files in os.walk(self.source_dir):
            if "naming_convention.json" in files:
                project_folders.append(Path(root))
        
        if not project_folders:
            raise ValueError(f"No project folders found in {self.source_dir}")
        
        print(f"🔍 Found {len(project_folders)} projects to analyze")
        
        # Detect duplicate project names and assign versions
        project_name_mapping = self._detect_duplicate_app_names(project_folders)
        
        for project_folder in project_folders:
            try:
                # Get required components before migration
                msgflow_files = list(project_folder.glob("*.msgflow"))
                if not msgflow_files:
                    print(f"⚠️ Skipping {project_folder.name}: No messageflow file found")
                    continue
                    
                # Parse messageflow to get required components
                required_components = self._get_required_components(msgflow_files[0])
                if not required_components:
                    print(f"⚠️ Skipping {project_folder.name}: No components found in messageflow")
                    continue
                
                print(f"📋 Found required components in {project_folder.name}:")
                for component in required_components:
                    print(f"  - {component}")
                
                # Proceed with migration
                project_name = self._migrate_project(project_folder, project_name_mapping, required_components)
                if project_name:
                    created_projects.append(project_name)
                    print(f"✅ Migrated: {project_name}")
            except Exception as e:
                print(f"❌ Failed to process {project_folder.name}: {e}")
        
        return created_projects
    
    def _get_required_components(self, msgflow_file: Path) -> Set[str]:
        """
        Parse messageflow XML file to identify required ACE components
        
        Args:
            msgflow_file: Path to the messageflow file
            
        Returns:
            Set of required ACE component names
        """
        try:
            # Parse XML with all namespaces
            with open(msgflow_file, 'r') as f:
                content = f.read()
                
            # Print first 100 chars of file for debugging
            print(f"  Debug - Messageflow content preview: {content[:100]}...")
                
            root = ET.fromstring(content)
            
            # Extract all component types and names
            required_components = set()
            
            # 1. Check for node types in format "ComIbm*.msgnode:*"
            for elem in root.findall(".//*"):
                # Check element tag for node types
                tag = elem.tag
                if ":" in tag:
                    namespace, node_type = tag.split("}")[-1].split(":")
                    if namespace.startswith("ComIbm") and namespace.endswith(".msgnode"):
                        component_name = namespace
                        required_components.add(component_name)
                        print(f"    Found node type: {component_name}")
            
            # 2. Check for subflow references
            for elem in root.findall(".//*[@componentName]"):
                subflow_name = elem.get("componentName")
                if subflow_name:
                    required_components.add(subflow_name)
                    print(f"    Found subflow: {subflow_name}")
            
            # 3. Extract ESQL modules from computeExpression attributes
            for elem in root.findall(".//*[@computeExpression]"):
                expr = elem.get("computeExpression")
                if expr and expr.startswith("esql://"):
                    parts = expr.split("#")
                    if len(parts) > 1:
                        module_parts = parts[1].split(".")
                        if len(module_parts) > 0:
                            esql_module = module_parts[0]
                            required_components.add(esql_module)
                            print(f"    Found ESQL module: {esql_module}")
            
            return required_components
            
        except ET.ParseError as e:
            print(f"  ⚠️ XML parsing error in {msgflow_file}: {e}")
            # Fallback to simpler parsing method for non-standard XML
            return self._fallback_parse_messageflow(msgflow_file)
        except Exception as e:
            print(f"  ⚠️ Error extracting components from {msgflow_file}: {e}")
            return set()
    
    def _fallback_parse_messageflow(self, msgflow_file: Path) -> Set[str]:
        """Fallback method for parsing messageflow when XML parsing fails"""
        required_components = set()
        try:
            with open(msgflow_file, 'r') as f:
                content = f.read()
            
            # Look for ComIbm*.msgnode patterns
            import re
            node_matches = re.findall(r'ComIbm[A-Za-z0-9]+\.msgnode', content)
            for match in node_matches:
                required_components.add(match)
                print(f"    Found node (fallback): {match}")
                
            # Look for ESQL references
            esql_matches = re.findall(r'esql://routine/([A-Za-z0-9_]+)#', content)
            for match in esql_matches:
                if match:
                    required_components.add(match)
                    print(f"    Found ESQL (fallback): {match}")
                    
            # Look for subflow references
            subflow_matches = re.findall(r'componentName="([A-Za-z0-9_]+)"', content)
            for match in subflow_matches:
                if match:
                    required_components.add(match)
                    print(f"    Found subflow (fallback): {match}")
                    
            return required_components
            
        except Exception as e:
            print(f"  ⚠️ Even fallback parsing failed for {msgflow_file}: {e}")
            return set()

    def _detect_duplicate_app_names(self, project_folders: List[Path]) -> Dict[str, Dict]:
        """Detect duplicate project names and assign version numbers"""
        project_name_counts = {}
        project_name_mapping = {}
        
        # First pass: Count occurrences of each project_name
        for project_folder in project_folders:
            naming_path = project_folder / "naming_convention.json"
            if not naming_path.exists():
                continue
                
            try:
                with open(naming_path, 'r') as f:
                    naming_data = json.load(f)
                
                # Extract project_name from component_naming_rules
                component_rules = naming_data.get("component_naming_rules", {})
                project_name = component_rules.get("project_name", "")
                
                if project_name:
                    project_name_counts[project_name] = project_name_counts.get(project_name, 0) + 1
                    
            except Exception as e:
                print(f"⚠️ Warning: Could not read {naming_path}: {e}")
        
        # Second pass: Assign version numbers for duplicates
        version_counters = {}
        
        for project_folder in project_folders:
            naming_path = project_folder / "naming_convention.json"
            if not naming_path.exists():
                continue
                
            try:
                with open(naming_path, 'r') as f:
                    naming_data = json.load(f)
                
                component_rules = naming_data.get("component_naming_rules", {})
                project_name = component_rules.get("project_name", "")
                msgflow_files = component_rules.get("msgflow_files", "")
                message_flow_name = msgflow_files.replace(".msgflow", "") if msgflow_files else ""
                
                if project_name:
                    # If this project name appears multiple times, add version suffix
                    if project_name_counts[project_name] > 1:
                        version_counters[project_name] = version_counters.get(project_name, 0) + 1
                        versioned_name = f"{project_name}_V{version_counters[project_name]}"
                        
                        project_name_mapping[project_folder.name] = {
                            'original_project_name': project_name,
                            'versioned_project_name': versioned_name,
                            'version_number': version_counters[project_name],
                            'message_flow_name': message_flow_name,
                            'has_duplicates': True
                        }
                        
                        print(f"📋 Duplicate detected: {project_name} → {versioned_name} (Flow: {message_flow_name})")
                    else:
                        # Single occurrence, no versioning needed
                        project_name_mapping[project_folder.name] = {
                            'original_project_name': project_name,
                            'versioned_project_name': project_name,
                            'version_number': 1,
                            'message_flow_name': message_flow_name,
                            'has_duplicates': False
                        }
                        
            except Exception as e:
                print(f"⚠️ Warning: Could not process {naming_path}: {e}")
        
        return project_name_mapping 

    def _migrate_project(self, source_project_dir: Path, app_name_mapping: Dict[str, Dict], 
                         required_components: Set[str]) -> str:
        """
        Migrate a single project, strictly filtering components
        
        Args:
            source_project_dir: Source project directory
            app_name_mapping: Mapping of folder names to versioning info
            required_components: Set of required component names
            
        Returns:
            Final project name or None if project should be skipped
        """
        # Read naming convention
        naming_path = source_project_dir / "naming_convention.json"
        if not naming_path.exists():
            raise FileNotFoundError(f"naming_convention.json not found in {source_project_dir}")
        
        with open(naming_path, 'r') as f:
            naming_data = json.load(f)
        
        # Get versioning info for this project
        folder_name = source_project_dir.name
        versioning_info = app_name_mapping.get(folder_name, {}) if app_name_mapping else {}
        
        # Use versioned project name if available, otherwise extract from naming data
        if versioning_info and 'versioned_project_name' in versioning_info:
            project_name = versioning_info['versioned_project_name']
            msgflow_name = versioning_info['message_flow_name']
            
            if versioning_info.get('has_duplicates', False):
                print(f"📁 Using versioned name: {project_name} (Flow: {msgflow_name})")
            else:
                print(f"📁 Using project name: {project_name} (Flow: {msgflow_name})")
        else:
            # Fallback to original extraction method
            project_name, msgflow_name = self._extract_naming_params(naming_data)
        
        # Create final project directory
        final_project_dir = self.root_dir / project_name
        
        if final_project_dir.exists():
            print(f"⚠️ Removing existing project: {project_name}")
            shutil.rmtree(final_project_dir)
        
        final_project_dir.mkdir(parents=True)
        
        # Now strictly copy only required components
        self._strictly_copy_required_files(source_project_dir, final_project_dir, required_components, msgflow_name)
        
        # Make sure ESQL folder is renamed
        self._rename_esql_folder(final_project_dir, msgflow_name)
        
        # Update project file
        self._update_project_file(final_project_dir, project_name)
        
        # IMPORTANT: Explicitly remove naming_convention.json if it was copied
        naming_file = final_project_dir / "naming_convention.json"
        if naming_file.exists():
            os.remove(naming_file)
            print(f"  🗑️ Removed naming_convention.json from final output")
        
        # Verify the removal worked
        if naming_file.exists():
            print(f"  ⚠️ WARNING: Failed to remove naming_convention.json!")
        else:
            print(f"  ✅ Confirmed naming_convention.json is removed")
        
        return project_name
    
    def _extract_naming_params(self, naming_data: Dict) -> Tuple[str, str]:
        """Extract project_name and msgflow_name from naming convention"""
        try:
            # Try new nested format first
            component_rules = naming_data.get("component_naming_rules", {})
            if component_rules:
                project_name = component_rules.get("project_name")
                msgflow_files = component_rules.get("msgflow_files")
            else:
                # Fallback to old flat format
                project_name = naming_data.get("project_name")
                msgflow_files = naming_data.get("msgflow_files")
            
            if not project_name:
                raise ValueError("project_name not found in naming_convention.json (checked both component_naming_rules and root level)")
            if not msgflow_files:
                raise ValueError("msgflow_files not found in naming_convention.json (checked both component_naming_rules and root level)")
            
            # Remove .msgflow extension for folder naming
            msgflow_name = msgflow_files.replace(".msgflow", "")
            
            print(f"  📋 Extracted: project_name='{project_name}', msgflow_name='{msgflow_name}'")
            return project_name, msgflow_name
            
        except Exception as e:
            raise ValueError(f"Invalid naming_convention.json format: {e}")
        

        
    
    def _strictly_copy_required_files(self, source_dir: Path, target_dir: Path, 
                                    required_components: Set[str], msgflow_name: str) -> None:
        """
        Strictly copy only the required files based on messageflow analysis
        
        Args:
            source_dir: Source directory
            target_dir: Target directory
            required_components: Set of required component names
            msgflow_name: Name of the messageflow (without extension)
        """
        print(f"  🔍 Strictly copying only required components")
        
        # 1. Always copy the messageflow file itself
        msgflow_files = list(source_dir.glob("*.msgflow"))
        if msgflow_files:
            msgflow_file = msgflow_files[0]
            shutil.copy2(msgflow_file, target_dir / msgflow_file.name)
            print(f"  📄 Copied messageflow: {msgflow_file.name}")
        
        # 2. Always copy .project file (needed for ACE Toolkit)
        project_file = source_dir / ".project"
        if project_file.exists():
            shutil.copy2(project_file, target_dir / ".project")
            print(f"  📄 Copied .project file")
        
        # 3. Always copy application.descriptor (needed for ACE Toolkit)
        app_desc = source_dir / "application.descriptor"
        if app_desc.exists():
            shutil.copy2(app_desc, target_dir / "application.descriptor")
            print(f"  📄 Copied application.descriptor")
        
        # 4. Filter ESQL files - only copy modules referenced in messageflow
        esql_dir = source_dir / "esql"
        if esql_dir.exists():
            target_esql = target_dir / "esql"
            os.makedirs(target_esql, exist_ok=True)
            
            # Check each ESQL file
            esql_files = list(esql_dir.glob("**/*.esql"))
            copied_esql = False
            
            for esql_file in esql_files:
                # Get module name from file name
                module_name = esql_file.stem
                
                # Extract parent folder name if in subdirectory
                relative_path = esql_file.relative_to(esql_dir)
                parent_folder = relative_path.parts[0] if len(relative_path.parts) > 1 else None
                
                # Check if this module or its parent folder is required
                is_required = (
                    module_name in required_components or 
                    (parent_folder and parent_folder in required_components) or
                    msgflow_name == module_name or
                    (parent_folder and msgflow_name == parent_folder)
                )
                
                if is_required:
                    # Preserve the directory structure when copying
                    target_file = target_esql / relative_path
                    os.makedirs(target_file.parent, exist_ok=True)
                    shutil.copy2(esql_file, target_file)
                    copied_esql = True
                    print(f"  📄 Copied required ESQL: {relative_path}")
                else:
                    print(f"  🗑️ Skipped unreferenced ESQL: {relative_path}")
            
            # If no ESQL files were copied, remove the empty directory
            if not copied_esql:
                shutil.rmtree(target_esql)
                print(f"  🗑️ Removed empty esql directory")
        
        # 5. Copy subflows directory - ADDED THIS PART
        subflows_dir = source_dir / "subflows"
        if subflows_dir.exists() and any(subflows_dir.iterdir()):
            target_subflows = target_dir / "subflows"
            shutil.copytree(subflows_dir, target_subflows)
            print(f"  📁 Copied subflows directory")
        
        # 6. Handle schemas directory - ACE might need these for validation
        schemas_dir = source_dir / "schemas"
        if schemas_dir.exists() and list(schemas_dir.glob("**/*.xsd")):
            target_schemas = target_dir / "schemas"
            shutil.copytree(schemas_dir, target_schemas)
            print(f"  📁 Copied schemas directory (with XSD files)")
        
        # 7. Handle transforms directory - only copy if XSL files exist
        transforms_dir = source_dir / "transforms"
        if transforms_dir.exists() and list(transforms_dir.glob("**/*.xsl")):
            target_transforms = target_dir / "transforms"
            shutil.copytree(transforms_dir, target_transforms)
            print(f"  📁 Copied transforms directory (with XSL files)")
        
        # 8. Handle enrichment directory - only copy if files exist
        enrichment_dir = source_dir / "enrichment"
        if enrichment_dir.exists() and any(enrichment_dir.iterdir()):
            target_enrichment = target_dir / "enrichment"
            shutil.copytree(enrichment_dir, target_enrichment)
            print(f"  📁 Copied enrichment directory (with content)")


    
    def _rename_esql_folder(self, project_dir: Path, msgflow_name: str) -> None:
        """Rename esql folder to msgflow_name"""
        esql_dir = project_dir / "esql"
        
        if esql_dir.exists():
            target_dir = project_dir / msgflow_name
            
            # Make sure target doesn't already exist
            if target_dir.exists():
                print(f"  ⚠️ Target directory {msgflow_name}/ already exists, can't rename esql/")
                return
                
            try:
                esql_dir.rename(target_dir)
                print(f"  📁 Renamed: esql → {msgflow_name}")
            except Exception as e:
                print(f"  ⚠️ Error renaming esql folder: {e}")
    
    def _update_project_file(self, project_dir: Path, project_name: str) -> None:
        """Update .project file with correct project name"""
        project_file = project_dir / ".project"
        
        if not project_file.exists():
            print(f"  ⚠️  .project file not found")
            return
        
        try:
            # Parse XML
            tree = ET.parse(project_file)
            root = tree.getroot()
            
            # Find and update name element
            name_element = root.find("name")
            if name_element is not None:
                old_name = name_element.text
                name_element.text = project_name
                
                # Write updated XML
                tree.write(project_file, encoding='utf-8', xml_declaration=True)
                print(f"  📝 Updated .project: {old_name} → {project_name}")
            else:
                print(f"  ⚠️  <name> element not found in .project")
                
        except ET.ParseError as e:
            print(f"  ❌ Failed to parse .project file: {e}")
    
    def cleanup_source(self) -> None:
        """Remove source directory after successful migration"""
        if self.source_dir.exists():
            shutil.rmtree(self.source_dir)
            print(f"🧹 Cleaned up source directory: {self.source_dir}")


def main():
    """Main execution function"""
    try:
        print("🚀 Starting Migration Quality Review...")
        print("=" * 50)
        
        # Initialize reviewer
        reviewer = MigrationQualityReviewer()
        
        # Process all projects
        created_projects = reviewer.process_all_projects()
        
        print("\n" + "=" * 50)
        print("🎉 Migration Complete!")
        print(f"✅ Created {len(created_projects)} project(s):")
        
        for project in created_projects:
            print(f"  📁 {project}/")
        
        print("\n💡 Import into IBM ACE Toolkit:")
        print("  1. File → Import → General → Existing Projects")
        print("  2. Select root directory")
        print("  3. Choose projects to import")
        
        # Optional cleanup
        cleanup_choice = input("\n🗑️  Remove source folders? (y/N): ").lower()
        if cleanup_choice == 'y':
            reviewer.cleanup_source()
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return 1
    
    return 0


# Backward compatibility alias for main.py
class SmartACEQualityReviewer:
    """Backward compatibility wrapper for main.py integration"""
    
    def __init__(self, ace_components_folder=None, ace_toolkit_path=None, 
                 naming_standards_file=None, vector_db_content=None, 
                 user_requirements=None, **kwargs):
        """Initialize with old parameter names for compatibility"""
        # Smart path detection
        if ace_components_folder:
            source_path = Path(ace_components_folder)
            # If pointing to output/, look for output/multiple/
            if source_path.name == "output" and (source_path / "multiple").exists():
                source_dir = str(source_path / "multiple")
            else:
                source_dir = ace_components_folder
        else:
            source_dir = "output/multiple"
            
        print(f"🎯 Initializing with source directory: {source_dir}")
        self.reviewer = MigrationQualityReviewer(source_dir)
        self.token_usage = 0  # For main.py compatibility
    
    def run_smart_review(self):
        """Run the migration process and return output path"""
        print(f"🔍 SmartACEQualityReviewer: Looking for projects in {self.reviewer.source_dir}")
        
        # Check if source directory exists
        if not self.reviewer.source_dir.exists():
            raise Exception(f"Source directory not found: {self.reviewer.source_dir}")
        
        # List contents for debugging
        contents = list(self.reviewer.source_dir.iterdir()) if self.reviewer.source_dir.exists() else []
        print(f"📁 Found {len(contents)} items in source directory:")
        for item in contents:
            print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
        
        created_projects = self.reviewer.process_all_projects()
        
        if created_projects:
            # Return the first created project path for main.py compatibility
            final_path = str(Path.cwd() / created_projects[0])
            print(f"✅ Migration successful: {final_path}")
            return final_path
        else:
            raise Exception("No projects were successfully migrated")


if __name__ == "__main__":
    exit(main())