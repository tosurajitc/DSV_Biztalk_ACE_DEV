#!/usr/bin/env python3
"""
Migration Quality Reviewer - ACE Project Structure Finalizer
Creates production-ready IBM ACE project folders with proper naming conventions
"""

import os
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple


class MigrationQualityReviewer:
    """Smart and efficient ACE project structure finalizer"""
    
    def __init__(self, source_dir: str = "output/multiple"):
        """
        Initialize the migration reviewer
        
        Args:
            source_dir: Source directory containing generated ACE projects
        """
        self.source_dir = Path(source_dir)
        self.root_dir = Path.cwd()  # Project root directory
        
    def process_all_projects(self) -> List[str]:
        """
        Process all projects in the source directory
        
        Returns:
            List of created project folder names
        """
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")
        
        created_projects = []
        
        # Find all project folders
        project_folders = [d for d in self.source_dir.iterdir() if d.is_dir()]
        
        if not project_folders:
            raise ValueError(f"No project folders found in {self.source_dir}")
        
        print(f"🔍 Found {len(project_folders)} projects to migrate")
        
        for project_folder in project_folders:
            try:
                project_name = self._migrate_project(project_folder)
                created_projects.append(project_name)
                print(f"✅ Migrated: {project_name}")
            except Exception as e:
                print(f"❌ Failed to migrate {project_folder.name}: {e}")
        
        return created_projects
    
    def _migrate_project(self, source_project_dir: Path) -> str:
        """
        Migrate a single project to production-ready structure
        
        Args:
            source_project_dir: Source project directory
            
        Returns:
            Final project name
        """
        # Read naming convention
        naming_path = source_project_dir / "naming_convention.json"
        if not naming_path.exists():
            raise FileNotFoundError(f"naming_convention.json not found in {source_project_dir}")
        
        with open(naming_path, 'r') as f:
            naming_data = json.load(f)
        
        # Extract key parameters
        project_name, msgflow_name = self._extract_naming_params(naming_data)
        
        # Create final project directory
        final_project_dir = self.root_dir / project_name
        
        if final_project_dir.exists():
            print(f"⚠️  Removing existing project: {project_name}")
            shutil.rmtree(final_project_dir)
        
        final_project_dir.mkdir(parents=True)
        
        # Copy and organize components
        self._copy_project_files(source_project_dir, final_project_dir)
        self._rename_esql_folder(final_project_dir, msgflow_name)
        self._update_project_file(final_project_dir, project_name)
        
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
    
    def _copy_project_files(self, source_dir: Path, target_dir: Path) -> None:
        """Copy all project files to target directory"""
        
        # Files to copy directly
        direct_files = [
            "naming_convention.json",
            "application.descriptor",
            ".project"
        ]
        
        # Copy direct files
        for file_name in direct_files:
            source_file = source_dir / file_name
            if source_file.exists():
                shutil.copy2(source_file, target_dir / file_name)
        
        # Copy msgflow file
        msgflow_files = list(source_dir.glob("*.msgflow"))
        if msgflow_files:
            shutil.copy2(msgflow_files[0], target_dir / msgflow_files[0].name)
        
        # Copy directories
        directories_to_copy = ["schemas", "transforms", "enrichment", "esql"]
        
        for dir_name in directories_to_copy:
            source_subdir = source_dir / dir_name
            if source_subdir.exists():
                target_subdir = target_dir / dir_name
                shutil.copytree(source_subdir, target_subdir)
    
    def _rename_esql_folder(self, project_dir: Path, msgflow_name: str) -> None:
        """Rename esql folder to msgflow_name"""
        esql_dir = project_dir / "esql"
        
        if esql_dir.exists():
            target_dir = project_dir / msgflow_name
            esql_dir.rename(target_dir)
            print(f"  📁 Renamed: esql → {msgflow_name}")
    
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
            raise Exception("No projects were successfully migrated - check if naming_convention.json exists in project folders")


if __name__ == "__main__":
    exit(main())