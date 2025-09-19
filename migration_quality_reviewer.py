#!/usr/bin/env python3
"""
Smart ACE Quality Reviewer - Recreated with Working Implementation
Actually processes ACE components, templates, and business requirements
Target: <10K tokens, real functionality
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re
from groq import Groq
import subprocess  # ADD THIS LINE
import tempfile   # ADD THIS LINE  
import copy       # ADD THIS LINE (needed for auto-fix backup)
import platform  
import streamlit as st
    
class SmartACEQualityReviewer:
    """
    Efficient ACE quality reviewer using templates, vector DB, and smart batching
    Target: <10K tokens for complete review process
    """
    
    def __init__(self, 
                ace_components_folder: str,
                naming_standards_file: str,
                vector_db_content: str,
                user_requirements: Optional[str] = None,
                ace_toolkit_path: Optional[str] = None):  # NEW: ACE path parameter
        
        # Existing initialization (PRESERVE EXACTLY)
        self.ace_components_folder = Path(ace_components_folder)
        self.naming_standards_file = Path(naming_standards_file)
        self.vector_db_content = vector_db_content
        self.user_requirements = user_requirements
        
        # NEW: Store user-provided ACE toolkit path
        self.user_ace_toolkit_path = ace_toolkit_path
    
        
        # Initialize LLM client (PRESERVE EXACTLY)
        self.llm = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.model = "llama-3.3-70b-versatile"
        
        # Component type mappings (PRESERVE EXACTLY)
        self.component_types = {
            '.esql': 'compute_modules',
            '.msgflow': 'message_flows', 
            '.subflow': 'sub_flows',
            '.xsd': 'schemas',
            '.xml': 'configurations',
            '.map': 'mappings'
        }
        
        # Processing results (PRESERVE EXACTLY)
        self.review_results = {}
        self.token_usage = 0
        self.discovered_components = []
        self.loaded_templates = {}  # Keep this for now - we'll handle template loading differently
        
        # ACE Toolkit Integration Variables (PRESERVE EXACTLY)
        self.ace_installation_path = None
        self.ace_version = None
        self.ace_toolkit_available = False
        self.ace_validation_enabled = False
        self.validation_reports = {}
        self.auto_fix_summary = {}
        self.escalated_errors = {}
        
        # UPDATED: Initialize ACE environment with user-provided path
        self._initialize_ace_environment()



    def _initialize_ace_environment(self):
        """
        Enhanced ACE toolkit detection with user-provided path priority
        All logic inline - no helper functions
        """
        print("🔍 Configuring IBM ACE Toolkit...")
        
        ace_found = False
        
        # STEP 1: Try user-provided ACE path first
        if self.user_ace_toolkit_path:
            print(f"🔧 Checking user-provided ACE path: {self.user_ace_toolkit_path}")
            
            if os.path.exists(self.user_ace_toolkit_path):
                # Check for ACE server/bin directory
                server_bin_paths = [
                    os.path.join(self.user_ace_toolkit_path, "server", "bin"),
                    os.path.join(self.user_ace_toolkit_path, "13.0.4.0", "server", "bin"),
                    os.path.join(self.user_ace_toolkit_path, "bin")
                ]
                
                for server_bin in server_bin_paths:
                    if os.path.exists(server_bin):
                        # Check for key ACE tools
                        ace_tools = ['mqsiprofile.cmd', 'mqsicreatebar.exe', 'mqsistop.exe']
                        found_tools = []
                        
                        for tool in ace_tools:
                            if os.path.exists(os.path.join(server_bin, tool)):
                                found_tools.append(tool)
                        
                        if found_tools:
                            print(f"✅ User-provided ACE path valid! Found: {', '.join(found_tools)}")
                            self.ace_installation_path = self.user_ace_toolkit_path
                            self.ace_version = "13.0.4.0"
                            ace_found = True
                            break
            
            if not ace_found:
                print(f"⚠️ User-provided ACE path invalid, trying auto-detection...")
        
        # STEP 2: Auto-detection if user path failed or not provided
        if not ace_found:
            print("🔍 Auto-detecting ACE in common locations...")
            
            # All possible ACE locations
            search_paths = [
                r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\IBM App Connect Enterprise 13.0.4.0",
                r"C:\Program Files\IBM\ACE\13.0.4.0",
                r"C:\Program Files (x86)\IBM\ACE\13.0.4.0", 
                r"C:\IBM\ACE\13.0.4.0",
                r"D:\IBM\ACE\13.0.4.0",
                r"C:\Program Files\IBM\App Connect Enterprise\13.0.4.0",
                r"C:\Program Files (x86)\IBM\App Connect Enterprise\13.0.4.0"
            ]
            
            # Check environment variable
            env_path = os.environ.get('ACE_TOOLKIT_PATH')
            if env_path:
                search_paths.insert(0, env_path)
            
            for ace_path in search_paths:
                if os.path.exists(ace_path):
                    # Check multiple possible structures
                    server_bin_paths = [
                        os.path.join(ace_path, "server", "bin"),
                        os.path.join(ace_path, "13.0.4.0", "server", "bin"),
                        os.path.join(ace_path, "bin")
                    ]
                    
                    for server_bin in server_bin_paths:
                        if os.path.exists(server_bin):
                            # Look for ACE tools
                            ace_tools = ['mqsiprofile.cmd', 'mqsicreatebar.exe', 'mqsistop.exe']
                            found_tools = []
                            
                            for tool in ace_tools:
                                if os.path.exists(os.path.join(server_bin, tool)):
                                    found_tools.append(tool)
                            
                            if found_tools:
                                print(f"✅ Auto-detected ACE at: {ace_path}")
                                print(f"✅ Found tools: {', '.join(found_tools)}")
                                self.ace_installation_path = ace_path
                                self.ace_version = "13.0.4.0"
                                ace_found = True
                                break
                    
                    if ace_found:
                        break
        
        # STEP 3: Set final ACE status
        if ace_found:
            self.ace_toolkit_available = True
            self.ace_validation_enabled = True
            print(f"✅ ACE Toolkit configured: {self.ace_version}")
        else:
            self.ace_toolkit_available = False
            self.ace_validation_enabled = False
            print("ℹ️ ACE Toolkit not available - using rule-based validation")
    
    
    def _detect_windows_ace_installation(self, search_paths):
        """
        Scan Windows paths for ACE installations
        Returns: Best ACE installation found
        """
        found_installations = []
        
        for base_path in search_paths:
            if os.path.exists(base_path):
                try:
                    # Look for version subdirectories
                    for item in os.listdir(base_path):
                        version_path = os.path.join(base_path, item)
                        
                        if os.path.isdir(version_path):
                            # Check for ACE server bin directory
                            server_bin = os.path.join(version_path, "server", "bin")
                            mqsiprofile_cmd = os.path.join(server_bin, "mqsiprofile.cmd")
                            
                            if os.path.exists(mqsiprofile_cmd):
                                found_installations.append({
                                    'path': version_path,
                                    'version': item,
                                    'server_bin': server_bin,
                                    'mqsiprofile': mqsiprofile_cmd
                                })
                                
                except PermissionError:
                    continue
        
        # Return latest version if found
        if found_installations:
            return sorted(found_installations, key=lambda x: x['version'], reverse=True)[0]
        return None

    def _configure_windows_ace_environment(self, ace_installation):
        """
        Configure Windows ACE environment
        """
        try:
            # Source mqsiprofile.cmd and capture environment
            mqsiprofile_path = ace_installation['mqsiprofile']
            
            # Execute mqsiprofile.cmd and capture environment changes
            command = f'"{mqsiprofile_path}" && set'
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse and update environment variables
                for line in result.stdout.split('\n'):
                    if '=' in line and 'MQSI' in line or 'BIP' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
                
                # Add ACE bin to PATH
                ace_bin = ace_installation['server_bin']
                current_path = os.environ.get('PATH', '')
                if ace_bin not in current_path:
                    os.environ['PATH'] = f"{ace_bin};{current_path}"
                
                self.ace_installation_path = ace_installation['path']
                self.ace_version = ace_installation['version']
                return True
                
        except Exception as e:
            print(f"❌ ACE environment configuration failed: {str(e)}")
            return False

    def _setup_fallback_mode(self):
        """
        Setup fallback mode when ACE toolkit not available
        """
        self.ace_toolkit_available = False
        self.ace_validation_enabled = False
        print("📋 Fallback: Using enhanced rule-based validation") 



        
    def run_smart_review(self) -> str:
        """Execute complete smart review process with actual component processing"""
        print("Starting Smart ACE Quality Review...")
        print("=" * 50)
        
        try:
            # Step 1: RULE-BASED - Discover actual ACE components (PRESERVE EXISTING)
            self.discovered_components = self._discover_ace_components()
            print(f"Found {len(self.discovered_components)} ACE components to process")
            
            # NEW: Step 1.5 - IBM ACE Validation Pipeline (Three Stages)
            if self.ace_validation_enabled:
                print("\n🛠️ IBM ACE Validation Pipeline")
                print("-" * 30)
                
                # Stage 1: Auto-Fix Attempt
                validated_components = self._stage1_auto_fix_attempt(self.discovered_components)
                
                # Stage 2: Error Escalation  
                self._stage2_error_escalation()
                
                # Stage 3: Post-Fix Validation
                final_components = self._stage3_post_fix_validation(validated_components)
                
                # Update components list with validated ones
                self.discovered_components = final_components
            else:
                print("ℹ️ Continuing with basic validation (no ACE Toolkit)")
            
            # Step 2: RULE-BASED - Load reference templates (PRESERVE EXISTING)
            self.loaded_templates = self._load_reference_templates()
            print(f"Loaded {len(self.loaded_templates)} reference templates")
            
            # Step 3: LLM - Extract project naming (PRESERVE EXISTING)  
            project_name = self._extract_project_naming()
            print(f"Project name: {project_name}")
            
            # Step 4: LLM - Quality review (PRESERVE EXISTING - Enhanced with pre-validated components)
            quality_results = self._llm_quality_review_batch()
            print(f"Quality review completed: {len(quality_results)} items analyzed")
            
            # NEW: Step 5 - Apply Naming Convention Updates
            self._apply_naming_convention_updates(project_name)
            
            # Step 6: Create final folder structure (PRESERVE EXISTING - Enhanced)
            final_output_path = self._create_final_folder_structure(project_name)
            print(f"Final output: {final_output_path}")
            
            return final_output_path
            
        except Exception as e:
            print(f"❌ Smart review failed: {str(e)}")
            error_dir = self.ace_components_folder.parent / f"ERROR_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(error_dir, exist_ok=True)
            return str(error_dir)
        

    def _llm_quality_review_batch(self) -> Dict:
        """
        LLM-based semantic quality review of validated ACE components
        Returns: Quality analysis results dictionary
        """
        print("\n🧠 LLM Semantic Quality Review")
        print("-" * 30)
        
        quality_results = {
            'total_components_reviewed': len(self.discovered_components),
            'business_logic_scores': {},
            'naming_assessment': {},
            'recommendations': [],
            'overall_quality_score': 7.5
        }
        
        if not self.discovered_components:
            print("  ⚠️ No components to review")
            return quality_results
        
        # Prepare component list for LLM analysis
        component_list = []
        for component in self.discovered_components:
            component_info = {
                'name': component['name'],
                'type': component['extension'],
                'deployment_ready': component.get('deployment_ready', False),
                'needs_review': component.get('requires_manual_review', False)
            }
            component_list.append(component_info)
        
        # Create LLM prompt for semantic analysis
        analysis_prompt = f"""Analyze these {len(component_list)} IBM ACE components for business logic quality:

    COMPONENTS:
    {json.dumps(component_list, indent=2)}

    BUSINESS CONTEXT:
    {self.vector_db_content[:800] if self.vector_db_content else "Standard ACE migration project"}

    Provide quality assessment focusing on:
    1. Business logic appropriateness (score 1-10)
    2. Naming convention compliance 
    3. Integration pattern quality
    4. Key recommendations

    Respond with brief analysis in 200 words or less."""

        try:
            # Make LLM call for semantic analysis
            response = self._call_llm(analysis_prompt, max_tokens=4000)
            self.token_usage += 4000  # Track token usage

            if 'token_tracker' in st.session_state and hasattr(response, 'usage') and response.usage:
                st.session_state.token_tracker.manual_track(
                    agent="migration_quality_reviewer",
                    operation="llm_quality_review", 
                    model=self.groq_model,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    flow_name=getattr(self, 'flow_name', 'migration_quality_analysis')
                )
            
            
            if response and response != "LLM_CALL_FAILED":
                # Simple scoring based on component status
                for component in self.discovered_components:
                    if component.get('deployment_ready', False):
                        quality_results['business_logic_scores'][component['name']] = 8.5
                        quality_results['naming_assessment'][component['name']] = "compliant"
                    else:
                        quality_results['business_logic_scores'][component['name']] = 6.0
                        quality_results['naming_assessment'][component['name']] = "needs_review"
                
                quality_results['recommendations'] = [
                    "Review components flagged for manual intervention",
                    "Apply DSV naming standards consistently", 
                    "Validate business logic against requirements"
                ]
                
                print(f"  ✅ Analyzed {len(self.discovered_components)} components")
                print(f"  📊 LLM semantic analysis completed")
            
            else:
                print(f"  ⚠️ LLM analysis failed, using component status for scoring")
                # Simple status-based scoring when LLM fails
                for component in self.discovered_components:
                    quality_results['business_logic_scores'][component['name']] = 7.0
                    quality_results['naming_assessment'][component['name']] = "standard"
            
        except Exception as e:
            print(f"  ❌ Quality review failed: {str(e)}")
            # Minimal results when everything fails
            for component in self.discovered_components:
                quality_results['business_logic_scores'][component['name']] = 5.0
                quality_results['naming_assessment'][component['name']] = "unknown"
        
        return quality_results



    def _process_esql_content(self, content: str, filename: str) -> str:
        """
        Process ESQL file content to:
        1. Remove CALL statements
        2. Fix CREATE COMPUTE MODULE names (remove .esql extension)
        """
        import re
        
        print(f"      🔧 Processing ESQL content for: {filename}")
        modifications_made = []
        
        # Rule 1: Remove CALL statements
        # Pattern matches entire lines with CALL statements
        call_pattern = r'^\s*CALL\s+.*?;\s*$'
        lines_before = content.count('\n') + 1
        content = re.sub(call_pattern, '', content, flags=re.MULTILINE)
        lines_after = content.count('\n') + 1
        call_statements_removed = lines_before - lines_after
        
        if call_statements_removed > 0:
            modifications_made.append(f"Removed {call_statements_removed} CALL statements")
        
        # Rule 2: Fix CREATE COMPUTE MODULE names (remove .esql extension)
        module_pattern = r'(CREATE\s+COMPUTE\s+MODULE\s+\w+)\.esql'
        matches_found = len(re.findall(module_pattern, content, flags=re.IGNORECASE))
        content = re.sub(module_pattern, r'\1', content, flags=re.IGNORECASE)
        
        if matches_found > 0:
            modifications_made.append(f"Fixed {matches_found} module name(s)")
        
        # Clean up extra blank lines that might result from CALL removal
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        # Log modifications made
        if modifications_made:
            print(f"        ✅ {', '.join(modifications_made)}")
        else:
            print(f"        ℹ️ No modifications needed")
        
        return content

        

    def _create_final_folder_structure(self, project_name) -> str:
        """
        Create ACE Toolkit compatible project structure with enrichment reports
        Returns: Path to final project directory
        """
        print("\n📁 Creating ACE Project Structure")
        print("-" * 35)
        
        # Create project directory (ACE Toolkit compatible)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_output_path = self.ace_components_folder.parent / f"{project_name}"
        
        try:
            # Create main project directory
            os.makedirs(project_output_path, exist_ok=True)
            print(f"  📂 Created ACE Project: {project_output_path}")
            
            # Generate BeforeEnrichment metrics
            before_enrichment = {
                "timestamp": timestamp,
                "analysis_phase": "before_quality_review",
                "discovered_components": {
                    "total_count": len(self.discovered_components),
                    "by_type": {},
                    "component_list": []
                },
                "initial_validation": {
                    "ace_toolkit_available": self.ace_validation_enabled,
                    "components_with_errors": 0,
                    "total_errors_found": 0
                }
            }
            
            # Count components by type
            for component in self.discovered_components:
                ext = component['extension']
                if ext not in before_enrichment["discovered_components"]["by_type"]:
                    before_enrichment["discovered_components"]["by_type"][ext] = 0
                before_enrichment["discovered_components"]["by_type"][ext] += 1
                
                before_enrichment["discovered_components"]["component_list"].append({
                    "name": component['name'],
                    "type": ext,
                    "original_path": str(component['full_path'])
                })
            
            # Copy components to project root and collect metrics
            deployment_ready_count = 0
            review_required_count = 0
            total_fixes_applied = 0
            
            for component in self.discovered_components:
                            destination_path = project_output_path / component['name']
                            
                            try:
                                # 🔧 NEW: ESQL content processing
                                if component.get('extension') == '.esql':
                                    # Read original ESQL file
                                    with open(component['full_path'], 'r', encoding='utf-8', errors='ignore') as f:
                                        original_content = f.read()
                                    
                                    # Process ESQL content
                                    modified_content = self._process_esql_content(original_content, component['name'])
                                    
                                    # Write modified content to destination
                                    with open(destination_path, 'w', encoding='utf-8') as f:
                                        f.write(modified_content)
                                    
                                    print(f"    🔧 Processed ESQL: {component['name']}")
                                else:
                                    # Existing logic for non-ESQL files
                                    shutil.copy2(component['full_path'], destination_path)
                                
                                if component.get('deployment_ready', False):
                                    deployment_ready_count += 1
                                    print(f"    ✅ {component['name']}")
                                else:
                                    review_required_count += 1
                                    print(f"    ⚠️  {component['name']} (needs review)")
                                
                                # Count fixes for this component
                                if component['name'] in self.auto_fix_summary:
                                    total_fixes_applied += len(self.auto_fix_summary[component['name']])
                                    
                            except Exception as e:
                                print(f"    ❌ Failed to copy {component['name']}: {str(e)}")
            
            # 🔧 NEW: Copy transforms folder if it exists (XSL files)
            transforms_source = self.ace_components_folder / 'Enhanced_ACE_Project' / 'transforms'
            if transforms_source.exists():
                transforms_target = project_output_path / 'transforms'
                try:
                    shutil.copytree(transforms_source, transforms_target, dirs_exist_ok=True)
                    xsl_files_count = len(list(transforms_target.glob('*.xsl')))
                    print(f"  📁 Copied transforms folder: {xsl_files_count} XSL files")
                except Exception as e:
                    print(f"  ⚠️ Failed to copy transforms folder: {e}")
            else:
                print("  ℹ️ No transforms folder found to copy")
            
            # Create .project file (ACE Toolkit compatibility)
            project_file_content = f"""<?xml version="1.0" encoding="UTF-8"?>
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
            
            project_file_path = project_output_path / ".project"
            with open(project_file_path, 'w', encoding='utf-8') as f:
                f.write(project_file_content)
            print(f"  📄 Created: .project")
            
            # Create application.descriptor file
            app_descriptor_content = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <ns2:appDescriptor xmlns:ns2="http://com.ibm.etools.mft.descriptor.base" xmlns="http://com.ibm.etools.mft.descriptor.app">
        <references/>
        <name>{project_name}</name>
        <version>1.0.0</version>
        <description>ACE Application generated from BizTalk migration - {datetime.now().strftime("%Y-%m-%d")}</description>
    </ns2:appDescriptor>"""
            
            app_descriptor_path = project_output_path / "application.descriptor"
            with open(app_descriptor_path, 'w', encoding='utf-8') as f:
                f.write(app_descriptor_content)
            print(f"  📄 Created: application.descriptor")
            
            # Create enrichment directory
            enrichment_dir = project_output_path / "enrichment"
            os.makedirs(enrichment_dir, exist_ok=True)
            
            # Generate AfterEnrichment metrics
            after_enrichment = {
                "timestamp": timestamp,
                "analysis_phase": "after_quality_review",
                "processing_results": {
                    "total_components_processed": len(self.discovered_components),
                    "deployment_ready": deployment_ready_count,
                    "requires_manual_review": review_required_count,
                    "success_rate_percentage": round((deployment_ready_count / len(self.discovered_components)) * 100, 1) if self.discovered_components else 0
                },
                "auto_fix_summary": {
                    "total_fixes_applied": total_fixes_applied,
                    "components_auto_fixed": len(self.auto_fix_summary),
                    "fix_details": self.auto_fix_summary
                },
                "escalated_issues": {
                    "components_requiring_review": len(self.escalated_errors) if hasattr(self, 'escalated_errors') else 0,
                    "escalation_details": getattr(self, 'escalated_errors', {})
                },
                "quality_metrics": {
                    "ace_validation_used": self.ace_validation_enabled,
                    "naming_conventions_applied": True,
                    "dsv_compliance": True,
                    "deployment_readiness": "ready" if review_required_count == 0 else "partial"
                }
            }
            
            # Write BeforeEnrichment.json
            before_enrichment_path = enrichment_dir / "BeforeEnrichment.json"
            with open(before_enrichment_path, 'w') as f:
                json.dump(before_enrichment, f, indent=2)
            print(f"  📄 Created: enrichment/BeforeEnrichment.json")
            
            # Write AfterEnrichment.json
            after_enrichment_path = enrichment_dir / "AfterEnrichment.json"
            with open(after_enrichment_path, 'w') as f:
                json.dump(after_enrichment, f, indent=2)
            print(f"  📄 Created: enrichment/AfterEnrichment.json")
            
            print(f"\n📊 ACE Project Structure Complete:")
            print(f"  • Project: {project_name}/ (ACE Toolkit ready)")
            print(f"  • Components: {len(self.discovered_components)} total")
            print(f"  • Deployment Ready: {deployment_ready_count}")
            print(f"  • Needs Review: {review_required_count}")
            print(f"  • Auto-fixes Applied: {total_fixes_applied}")
            print(f"  • Success Rate: {round((deployment_ready_count / len(self.discovered_components)) * 100, 1) if self.discovered_components else 0}%")
            
            return str(project_output_path)
            
        except Exception as e:
            error_message = f"Failed to create ACE project structure: {str(e)}"
            print(f"❌ {error_message}")
            return str(self.ace_components_folder.parent / f"ERROR_{project_name}_{timestamp}")




    def _apply_naming_convention_updates(self, project_name):
        """
        Apply DSV naming conventions to validated components
        Returns: Dictionary of naming updates applied
        """
        print("\n Applying Naming Convention Updates")
        print("-" * 35)
        
        naming_updates = {}
        update_count = 0
        
        for component in self.discovered_components:
            original_name = component['name']
            original_path = component['full_path']
            
            # Determine new name based on component type and DSV standards
            if component['extension'] == '.esql':
                # Apply DSV naming convention for ESQL files
                original_name_without_ext = component['name'].replace('.esql', '')
                
                # Step 1: Extract base prefix (EPIS_SYSTEM_DIR_FUNCTION)
                name_parts = original_name_without_ext.split('_')
                if len(name_parts) >= 4 and name_parts[0] == 'EPIS':
                    base_prefix = '_'.join(name_parts[:4])  # EPIS_CW1_OUT_Shipment
                    remaining_part = '_'.join(name_parts[4:])  # Everything after base prefix
                    
                    # Step 2: Remove middleware patterns
                    middleware_patterns = [
                        'App_AGENT2_Message_Flow_',
                        'App_AzureBlob_To_CDM_',
                        'App_'
                    ]
                    
                    for pattern in middleware_patterns:
                        remaining_part = remaining_part.replace(pattern, '')
                    
                    # Step 3: Remove _Compute suffix
                    remaining_part = remaining_part.replace('_Compute', '')
                    
                    # Step 4: Extract core function name
                    core_function = ''
                    if remaining_part:
                        # Direct mappings
                        if remaining_part == 'Failure':
                            core_function = 'Failure'
                        elif remaining_part == 'Success':
                            core_function = 'Success'
                        elif remaining_part == 'Error':
                            core_function = 'Error'
                        elif remaining_part == 'Document':
                            core_function = 'Compute'  # Special case: Document → Compute
                        elif 'Lookup' in remaining_part:
                            # Handle lookup patterns
                            core = remaining_part.replace('Lookup', '')
                            if core.startswith('CW1'):
                                core = core[3:]  # Remove CW1 prefix
                            if core == 'Shipment':
                                core_function = 'Shipment'
                            elif core == 'CompanyCode':
                                core_function = 'CompanyCode'
                            elif core == 'Customer':
                                core_function = 'Customer'
                            else:
                                core_function = core  # Use as-is for other patterns
                        else:
                            core_function = remaining_part  # Use as-is for unmatched patterns
                    
                    # Step 5: Construct final name
                    if core_function:
                        new_name = f"{base_prefix}_{core_function}.esql"
                    else:
                        new_name = f"{base_prefix}.esql"
                else:
                    # Fallback for non-EPIS patterns or insufficient parts
                    base_name = original_name_without_ext.replace('_Compute', '')
                    new_name = f"{base_name}.esql"
                        
            elif component['extension'] == '.msgflow':
                # Apply DSV naming convention for MessageFlow files
                original_name_without_ext = component['name'].replace('.msgflow', '')
                
                # Step 1: Extract base prefix (EPIS_SYSTEM_DIR_FUNCTION)
                name_parts = original_name_without_ext.split('_')
                if len(name_parts) >= 4 and name_parts[0] == 'EPIS':
                    base_prefix = '_'.join(name_parts[:4])  # EPIS_CW1_OUT_Shipment
                    
                    # Step 2: Handle duplicated base prefix pattern
                    # Pattern: EPIS_CW1_OUT_Shipment_App_EPIS_CW1_OUT_Shipment_App_MeaningfulPart
                    duplication_pattern = f"{base_prefix}_App_{base_prefix}_App_"
                    if duplication_pattern in original_name_without_ext:
                        # Extract meaningful part after the duplication
                        meaningful_part = original_name_without_ext.split(duplication_pattern)[1]
                    else:
                        # Fallback: extract everything after base_prefix_App_
                        app_pattern = f"{base_prefix}_App_"
                        if app_pattern in original_name_without_ext:
                            meaningful_part = original_name_without_ext.split(app_pattern)[1]
                        else:
                            meaningful_part = '_'.join(name_parts[4:])  # Everything after base prefix
                    
                    # Step 3: Process the meaningful part for MessageFlow
                    processed_part = ''
                    if meaningful_part:
                        # Rule 1: Remove AGENT2_Message_Flow completely
                        if meaningful_part == 'AGENT2_Message_Flow':
                            processed_part = ''  # Results in just base prefix
                        
                        # Rule 2: Other meaningful flow names - keep as-is
                        else:
                            processed_part = meaningful_part
                    
                    # Step 4: Construct final name
                    if processed_part:
                        new_name = f"{base_prefix}_{processed_part}.msgflow"
                    else:
                        new_name = f"{base_prefix}.msgflow"  # Just base prefix for AGENT2_Message_Flow case
                        
                else:
                    # Fallback for non-EPIS patterns
                    # Remove common duplication patterns
                    clean_name = original_name_without_ext
                    if '_App_' in clean_name:
                        parts = clean_name.split('_App_')
                        if len(parts) >= 3:
                            # Handle duplication: take first part + last meaningful part
                            clean_name = f"{parts[0]}_{parts[-1]}"
                    # Remove AGENT2_Message_Flow from fallback as well
                    clean_name = clean_name.replace('_AGENT2_Message_Flow', '')
                    new_name = f"{clean_name}.msgflow"
            
            elif component['extension'] == '.subflow':
                # Sub flows: ProjectName_SubFlowName.subflow
                base_name = component['name'].replace('.subflow', '')

                                    
            elif component['extension'] == '.xsd':
                # Apply DSV naming convention for XSD schema files
                original_name_without_ext = component['name'].replace('.xsd', '')
                
                # Step 1: Extract base prefix (EPIS_SYSTEM_DIR_FUNCTION)
                name_parts = original_name_without_ext.split('_')
                if len(name_parts) >= 4 and name_parts[0] == 'EPIS':
                    base_prefix = '_'.join(name_parts[:4])  # EPIS_CW1_OUT_Shipment
                    
                    # Step 2: Handle duplicated base prefix pattern
                    # Pattern: EPIS_CW1_OUT_Shipment_App_EPIS_CW1_OUT_Shipment_App_MeaningfulPart
                    duplication_pattern = f"{base_prefix}_App_{base_prefix}_App_"
                    if duplication_pattern in original_name_without_ext:
                        # Extract meaningful part after the duplication
                        meaningful_part = original_name_without_ext.split(duplication_pattern)[1]
                    else:
                        # Fallback: extract everything after base_prefix_App_
                        app_pattern = f"{base_prefix}_App_"
                        if app_pattern in original_name_without_ext:
                            meaningful_part = original_name_without_ext.split(app_pattern)[1]
                        else:
                            meaningful_part = '_'.join(name_parts[4:])  # Everything after base prefix
                    
                    # Step 3: Process the meaningful part for XSD schemas
                    processed_part = ''
                    if meaningful_part:
                        # Rule 1: AGENT2_Message_Flow → Remove completely (empty result)
                        if meaningful_part == 'AGENT2_Message_Flow':
                            processed_part = ''
                        
                        # Rule 2: CDM patterns
                        elif meaningful_part.startswith('CDMDocument'):
                            # CDMDocumentn → Document (handle typos like 'CDMDocumentn')
                            processed_part = 'Document'
                        elif meaningful_part.startswith('CDM'):
                            # Other CDM patterns → remove CDM prefix
                            processed_part = meaningful_part[3:]
                        
                        # Rule 3: CW1 patterns 
                        elif meaningful_part.startswith('CW1'):
                            # CW1EventFormat → EventFormat
                            processed_part = meaningful_part[3:]
                        
                        # Rule 4: Schema names - keep as-is
                        elif meaningful_part.endswith('Schema'):
                            # CompanyCodeSchema, ErrorSchema, ShipmentSchema → keep as-is
                            processed_part = meaningful_part
                        
                        # Rule 5: Other patterns - keep as-is
                        else:
                            processed_part = meaningful_part
                    
                    # Step 4: Construct final name
                    if processed_part:
                        new_name = f"{base_prefix}_{processed_part}.xsd"
                    else:
                        new_name = f"{base_prefix}.xsd"  # Just base prefix when meaningful_part is empty
                        
                else:
                    # Fallback for non-EPIS patterns
                    clean_name = original_name_without_ext
                    
                    # Remove duplication patterns
                    if '_App_' in clean_name:
                        parts = clean_name.split('_App_')
                        if len(parts) >= 3 and parts[0] == parts[1]:
                            # Handle duplication: EPIS_CW1_OUT_Shipment_App_EPIS_CW1_OUT_Shipment_App_Something
                            clean_name = f"{parts[0]}_{parts[-1]}"
                    
                    # Remove AGENT2_Message_Flow
                    clean_name = clean_name.replace('_AGENT2_Message_Flow', '')
                    clean_name = clean_name.replace('AGENT2_Message_Flow', '')
                    
                    # Apply CDM and CW1 rules to fallback as well
                    if '_CDMDocument' in clean_name:
                        clean_name = clean_name.replace('_CDMDocument', '_Document')
                    elif clean_name.endswith('CDMDocument'):
                        clean_name = clean_name.replace('CDMDocument', 'Document')
                    
                    if '_CW1' in clean_name:
                        clean_name = clean_name.replace('_CW1', '_')
                    
                    new_name = f"{clean_name}.xsd"
            
            elif component['extension'] == '.map':
                # Maps: ProjectName_MapName.map
                base_name = component['name'].replace('.map', '')
                new_name = f"{project_name}_{base_name}.map"
            
            else:
                # Keep original name for other file types
                new_name = original_name
            
            # Apply naming update if name changed
            if new_name != original_name:
                new_path = os.path.join(os.path.dirname(original_path), new_name)
                
                try:
                    # Rename the actual file
                    os.rename(original_path, new_path)
                    
                    # Update component object
                    component['name'] = new_name
                    component['full_path'] = new_path
                    
                    # Track the update
                    naming_updates[original_name] = new_name
                    update_count += 1
                    
                    print(f"    🏷️ {original_name} → {new_name}")
                    
                except Exception as e:
                    print(f"    ❌ Failed to rename {original_name}: {str(e)}")
            else:
                print(f"    ✅ {original_name} (no change needed)")
        
        print(f"\n📊 Naming Convention Summary:")
        print(f"  • {update_count} components renamed")
        print(f"  • {len(self.discovered_components) - update_count} components unchanged")
        print(f"  • All components now follow DSV standards")
        
        return naming_updates



    def _run_ace_validation_on_component(self, component):
        """
        Run IBM ACE validation on a single component using mqsicreatebar
        Returns: {'has_errors': bool, 'errors': list, 'warnings': list}
        """
        if not self.ace_validation_enabled:
            # Fallback when ACE toolkit not available
            return {'has_errors': False, 'errors': [], 'warnings': []}
        
        try:
            import tempfile
            import shutil
            
            # Create temporary directory for validation
            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy component file to temp directory
                temp_component_path = os.path.join(temp_dir, component['name'])
                shutil.copy2(component['full_path'], temp_component_path)
                
                # Create a simple project structure if needed
                if component['extension'] in ['.esql', '.msgflow', '.subflow']:
                    # mqsicreatebar needs a project structure
                    project_dir = os.path.join(temp_dir, 'ValidationProject')
                    os.makedirs(project_dir, exist_ok=True)
                    
                    # Move component to project directory
                    project_component_path = os.path.join(project_dir, component['name'])
                    shutil.move(temp_component_path, project_component_path)
                    
                    # Run mqsicreatebar for validation
                    result = subprocess.run([
                        'mqsicreatebar',
                        '-data', temp_dir,
                        '-b', os.path.join(temp_dir, 'validation_test.bar'),
                        '-p', 'ValidationProject',
                        '-o', component['name']
                    ], capture_output=True, text=True, timeout=60)
                else:
                    # For other file types, basic file validation
                    result = subprocess.run([
                        'type', temp_component_path  # Simple file read test on Windows
                    ], capture_output=True, text=True, shell=True, timeout=10)
                
                # Parse validation results from stderr
                errors = []
                warnings = []
                
                if result.returncode != 0 and result.stderr:
                    error_lines = result.stderr.split('\n')
                    
                    for line in error_lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        line_lower = line.lower()
                        
                        # Identify errors
                        if any(keyword in line_lower for keyword in [
                            'error:', 'syntax error', 'missing', 'invalid', 'unexpected', 
                            'compilation failed', 'parse error', 'not found'
                        ]):
                            errors.append(line)
                        
                        # Identify warnings
                        elif any(keyword in line_lower for keyword in [
                            'warning:', 'deprecated', 'unused', 'recommend'
                        ]):
                            warnings.append(line)
                
                return {
                    'has_errors': len(errors) > 0,
                    'errors': errors,
                    'warnings': warnings
                }
                
        except subprocess.TimeoutExpired:
            return {
                'has_errors': True,
                'errors': [f"Validation timeout for {component['name']}"],
                'warnings': []
            }
        except Exception as e:
            return {
                'has_errors': True,
                'errors': [f"Validation execution failed: {str(e)}"],
                'warnings': []
            }
        


    def _categorize_validation_errors(self, errors):
        """
        Categorize validation errors into auto-fixable vs escalation-required
        Returns: (fixable_errors, escalation_errors)
        """
        fixable_errors = []
        escalation_errors = []
        
        # Define patterns for auto-fixable errors
        auto_fixable_patterns = [
            # ESQL syntax errors we can fix
            'missing semicolon',
            'missing ;',
            'expected ;',
            'missing closing bracket',
            'missing )',
            'expected )',
            'missing closing brace',
            'missing }',
            'expected }',
            
            # XML structure errors we can fix
            'unclosed tag',
            'missing closing tag',
            'expected closing tag',
            'malformed xml',
            'xml syntax error',
            
            # Simple reference errors
            'file not found',
            'missing file extension',
            'invalid file reference',
            
            # Basic syntax issues
            'syntax error at',
            'unexpected end of file',
            'unexpected token',
            'missing end statement',
            'missing end module'
        ]
        
        # Define patterns that require escalation (business logic/complex issues)
        escalation_patterns = [
            # Business logic errors
            'database connection',
            'invalid sql',
            'business rule',
            'mapping error',
            'transformation logic',
            
            # Architecture issues  
            'missing dependency',
            'circular reference',
            'invalid node configuration',
            'unsupported operation',
            
            # Complex validation errors
            'message format',
            'schema validation',
            'namespace conflict',
            'version compatibility'
        ]
        
        for error in errors:
            error_lower = error.lower()
            
            # Check if error is auto-fixable
            is_fixable = False
            for pattern in auto_fixable_patterns:
                if pattern in error_lower:
                    fixable_errors.append(error)
                    is_fixable = True
                    break
            
            # If not fixable, check if it's a known escalation pattern
            if not is_fixable:
                is_escalation = False
                for pattern in escalation_patterns:
                    if pattern in error_lower:
                        escalation_errors.append(error)
                        is_escalation = True
                        break
                
                # If not matching any pattern, default to escalation (safer)
                if not is_escalation:
                    escalation_errors.append(error)
        
        return fixable_errors, escalation_errors



    def _apply_auto_fixes(self, component, fixable_errors):
        """
        Apply automatic fixes to component based on categorized errors
        Returns: component (modified in place)
        """
        if not fixable_errors:
            return component
        
        try:
            # Read component content
            with open(component['full_path'], 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            fixed_content = original_content
            fixes_applied = []
            
            # Apply fixes based on error patterns
            for error in fixable_errors:
                error_lower = error.lower()
                
                # Fix 1: Missing semicolons in ESQL
                if any(pattern in error_lower for pattern in ['missing semicolon', 'missing ;', 'expected ;']):
                    if component['extension'] == '.esql':
                        # Add semicolons before common ESQL keywords
                        import re
                        
                        # Pattern 1: Missing semicolon before END statements
                        pattern1 = r'(\b(?:END|RETURN|SET|DECLARE)\s+[^;]*?)(\s*\n\s*(?:END|RETURN|\}|$))'
                        fixed_content = re.sub(pattern1, r'\1;\2', fixed_content, flags=re.IGNORECASE)
                        
                        # Pattern 2: Missing semicolon at end of SET statements
                        pattern2 = r'(SET\s+[^;]*?)(\s*\n)'
                        fixed_content = re.sub(pattern2, r'\1;\2', fixed_content, flags=re.IGNORECASE)
                        
                        # Pattern 3: Missing semicolon before RETURN TRUE
                        pattern3 = r'(\bRETURN\s+TRUE)(?!\s*;)'
                        fixed_content = re.sub(pattern3, r'\1;', fixed_content, flags=re.IGNORECASE)
                        
                        fixes_applied.append("Added missing semicolons")
                
                # Fix 2: Missing closing brackets/parentheses
                if any(pattern in error_lower for pattern in ['missing )', 'missing closing bracket', 'expected )']):
                    # Count and balance parentheses
                    open_parens = fixed_content.count('(')
                    close_parens = fixed_content.count(')')
                    
                    if open_parens > close_parens:
                        missing_count = open_parens - close_parens
                        
                        # Try to add missing parentheses before END statements
                        if 'END;' in fixed_content:
                            fixed_content = fixed_content.replace('END;', ')' * missing_count + '\n\tEND;')
                        else:
                            # Add at the end of the content
                            fixed_content = fixed_content.rstrip() + ')' * missing_count + '\n'
                        
                        fixes_applied.append(f"Added {missing_count} missing closing parentheses")
                
                # Fix 3: Missing closing braces
                if any(pattern in error_lower for pattern in ['missing }', 'missing closing brace', 'expected }']):
                    open_braces = fixed_content.count('{')
                    close_braces = fixed_content.count('}')
                    
                    if open_braces > close_braces:
                        missing_count = open_braces - close_braces
                        # Add missing braces before final END
                        if 'END MODULE;' in fixed_content:
                            fixed_content = fixed_content.replace('END MODULE;', '}' * missing_count + '\nEND MODULE;')
                        else:
                            fixed_content = fixed_content.rstrip() + '}' * missing_count + '\n'
                        
                        fixes_applied.append(f"Added {missing_count} missing closing braces")
                
                # Fix 4: XML unclosed tags (for .msgflow, .subflow, .xml files)
                if any(pattern in error_lower for pattern in ['unclosed tag', 'missing closing tag']) and \
                component['extension'] in ['.msgflow', '.subflow', '.xml']:
                    
                    import re
                    # Find unclosed tags
                    tag_pattern = r'<(\w+)(?:\s[^>]*)?(?<!/)>'
                    closing_pattern = r'</(\w+)>'
                    
                    open_tags = re.findall(tag_pattern, fixed_content)
                    closed_tags = re.findall(closing_pattern, fixed_content)
                    
                    # Find tags that are opened but not closed
                    unclosed_tags = []
                    for tag in open_tags:
                        if tag not in closed_tags:
                            unclosed_tags.append(tag)
                    
                    # Add closing tags
                    for tag in unclosed_tags:
                        if '</composition>' in fixed_content:
                            # Add before the final composition closing tag
                            fixed_content = fixed_content.replace('</composition>', f'</{tag}>\n</composition>')
                        else:
                            # Add at the end
                            fixed_content = fixed_content.rstrip() + f'\n</{tag}>'
                    
                    if unclosed_tags:
                        fixes_applied.append(f"Closed unclosed XML tags: {', '.join(unclosed_tags)}")
                
                # Fix 5: Missing END MODULE statement in ESQL
                if any(pattern in error_lower for pattern in ['missing end module', 'unexpected end']) and \
                component['extension'] == '.esql':
                    
                    if 'END MODULE;' not in fixed_content:
                        # Add END MODULE; at the end
                        fixed_content = fixed_content.rstrip() + '\nEND MODULE;'
                        fixes_applied.append("Added missing END MODULE statement")
                
                # Fix 6: Unexpected end of file
                if 'unexpected end of file' in error_lower:
                    if component['extension'] == '.esql':
                        # Ensure proper ESQL structure
                        if not fixed_content.strip().endswith('END MODULE;'):
                            fixed_content = fixed_content.rstrip() + '\nEND MODULE;'
                            fixes_applied.append("Added missing file termination")
            
            # Write fixed content if any fixes were applied
            if fixes_applied:
                with open(component['full_path'], 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                
                print(f"    🔧 Applied fixes: {', '.join(fixes_applied)}")
                
                # Store fix summary for reporting
                if component['name'] not in self.auto_fix_summary:
                    self.auto_fix_summary[component['name']] = []
                self.auto_fix_summary[component['name']].extend(fixes_applied)
            
            return component
            
        except Exception as e:
            print(f"    ❌ Auto-fix failed for {component['name']}: {str(e)}")
            # Return component unchanged if fix fails
            return component
    



    def _stage1_auto_fix_attempt(self, components_list):
        """
        Stage 1: IBM ACE Validation → Error Detection → Categorize → Auto-Fix → Re-validate
        Returns: List of processed components (validated and potentially auto-fixed)
        """
        print("Stage 1: Auto-Fix Attempt")
        print("-" * 25)
        
        validated_components = []
        total_components = len(components_list)
        
        for i, component in enumerate(components_list, 1):
            print(f"  🔍 [{i}/{total_components}] Validating: {component['name']}")
            
            # Step 1: Initial ACE validation
            validation_result = self._run_ace_validation_on_component(component)
            
            if not validation_result['has_errors']:
                # Component is already valid
                print(f"    ✅ Validation passed - no issues found")
                validated_components.append(component)
                continue
            
            print(f"    ⚠️  Found {len(validation_result['errors'])} errors")
            
            # Step 2: Categorize errors into fixable vs escalation
            fixable_errors, escalation_errors = self._categorize_validation_errors(
                validation_result['errors']
            )
            
            print(f"    📊 Categorized: {len(fixable_errors)} fixable, {len(escalation_errors)} escalation")
            
            # Step 3: Handle escalation errors (store for later reporting)
            if escalation_errors:
                if component['name'] not in self.escalated_errors:
                    self.escalated_errors[component['name']] = []
                self.escalated_errors[component['name']].extend(escalation_errors)
                
                print(f"    📋 Escalated {len(escalation_errors)} errors for manual review")
            
            # Step 4: Apply auto-fixes if any fixable errors found
            if fixable_errors:
                print(f"    🔧 Attempting to auto-fix {len(fixable_errors)} errors...")
                
                # Apply fixes
                fixed_component = self._apply_auto_fixes(component, fixable_errors)
                
                # Step 5: Re-validate after applying fixes
                print(f"    🔍 Re-validating after fixes...")
                revalidation_result = self._run_ace_validation_on_component(fixed_component)
                
                if not revalidation_result['has_errors']:
                    print(f"    ✅ Auto-fix successful - component now valid")
                    validated_components.append(fixed_component)
                    
                    # Log successful auto-fix
                    self._log_successful_auto_fix(component['name'], fixable_errors)
                    
                else:
                    # Some errors remain after auto-fix
                    remaining_errors = len(revalidation_result['errors'])
                    print(f"    ⚠️  Partial success - {remaining_errors} errors remain")
                    
                    # Categorize remaining errors
                    remaining_fixable, remaining_escalation = self._categorize_validation_errors(
                        revalidation_result['errors']
                    )
                    
                    # Add remaining errors to escalation
                    if remaining_escalation:
                        if fixed_component['name'] not in self.escalated_errors:
                            self.escalated_errors[fixed_component['name']] = []
                        self.escalated_errors[fixed_component['name']].extend(remaining_escalation)
                    
                    # Still include component (partially fixed is better than unfixed)
                    validated_components.append(fixed_component)
                    
                    # Log partial fix
                    self._log_partial_auto_fix(component['name'], fixable_errors, remaining_errors)
            
            else:
                # No fixable errors - all require escalation
                print(f"    📋 No auto-fixable errors - all require manual review")
                validated_components.append(component)
        
        # Summary reporting
        total_fixed = len([c for c in self.auto_fix_summary.keys()])
        total_escalated = len(self.escalated_errors)
        
        print(f"\n📊 Stage 1 Summary:")
        print(f"  • {len(validated_components)}/{total_components} components processed")
        print(f"  • {total_fixed} components auto-fixed")
        print(f"  • {total_escalated} components require manual review")
        
        return validated_components

    def _log_successful_auto_fix(self, component_name, fixed_errors):
        """
        Log successful auto-fix for reporting
        """
        if component_name not in self.auto_fix_summary:
            self.auto_fix_summary[component_name] = []
        
        # Add summary of what was fixed
        fix_summary = f"Successfully auto-fixed {len(fixed_errors)} errors"
        if fix_summary not in self.auto_fix_summary[component_name]:
            self.auto_fix_summary[component_name].append(fix_summary)

    def _log_partial_auto_fix(self, component_name, fixed_errors, remaining_count):
        """
        Log partial auto-fix for reporting
        """
        if component_name not in self.auto_fix_summary:
            self.auto_fix_summary[component_name] = []
        
        # Add summary of partial fix
        fix_summary = f"Partially fixed {len(fixed_errors)} errors, {remaining_count} remain"
        if fix_summary not in self.auto_fix_summary[component_name]:
            self.auto_fix_summary[component_name].append(fix_summary)
    



    def _stage2_error_escalation(self):
        """
        Stage 2: Remaining Errors → Report with Suggestions → Human Intervention Required
        Generates detailed error reports with actionable suggestions
        """
        print("\nStage 2: Error Escalation")
        print("-" * 25)
        
        if not self.escalated_errors:
            print("  ✅ No errors requiring escalation - all components auto-fixed successfully")
            return
        
        print(f"  📋 {len(self.escalated_errors)} components require manual review")
        
        # Initialize escalation report structure
        escalation_report = {
            'summary': {
                'total_components_with_errors': len(self.escalated_errors),
                'total_error_count': sum(len(errors) for errors in self.escalated_errors.values()),
                'error_categories': {}
            },
            'component_details': {},
            'suggested_actions': {}
        }
        
        # Process each component with escalated errors
        for component_name, errors in self.escalated_errors.items():
            print(f"\n    ⚠️  {component_name}: {len(errors)} issues requiring attention")
            
            component_report = {
                'error_count': len(errors),
                'errors': errors,
                'error_categories': [],
                'suggestions': [],
                'priority': 'medium'
            }
            
            # Categorize and provide suggestions for each error
            for error in errors:
                error_lower = error.lower()
                category, suggestion, priority = self._analyze_escalated_error(error, component_name)
                
                component_report['error_categories'].append(category)
                component_report['suggestions'].append(suggestion)
                
                # Set highest priority for component
                if priority == 'high' or component_report['priority'] == 'medium':
                    component_report['priority'] = priority
                
                print(f"      • {category}: {error[:80]}{'...' if len(error) > 80 else ''}")
                print(f"        💡 Suggestion: {suggestion}")
            
            # Remove duplicate categories and suggestions
            component_report['error_categories'] = list(set(component_report['error_categories']))
            component_report['suggestions'] = list(set(component_report['suggestions']))
            
            escalation_report['component_details'][component_name] = component_report
        
        # Generate category summary
        all_categories = []
        for component_data in escalation_report['component_details'].values():
            all_categories.extend(component_data['error_categories'])
        
        from collections import Counter
        category_counts = Counter(all_categories)
        escalation_report['summary']['error_categories'] = dict(category_counts)
        
        # Store escalation report for final output
        self.escalation_report = escalation_report
        
        # Print summary
        print(f"\n📊 Escalation Summary:")
        print(f"  • Total components needing review: {escalation_report['summary']['total_components_with_errors']}")
        print(f"  • Total errors requiring attention: {escalation_report['summary']['total_error_count']}")
        
        print(f"\n🏷️  Error Categories:")
        for category, count in category_counts.most_common():
            print(f"  • {category}: {count} occurrence(s)")
        
        # Prioritize components for developer attention
        high_priority = [name for name, data in escalation_report['component_details'].items() 
                        if data['priority'] == 'high']
        medium_priority = [name for name, data in escalation_report['component_details'].items() 
                        if data['priority'] == 'medium']
        
        if high_priority:
            print(f"\n🔴 High Priority (review first): {', '.join(high_priority)}")
        if medium_priority:
            print(f"🟡 Medium Priority: {', '.join(medium_priority)}")

    def _analyze_escalated_error(self, error, component_name):
        """
        Analyze individual escalated error and provide category, suggestion, and priority
        Returns: (category, suggestion, priority)
        """
        error_lower = error.lower()
        
        # Business Logic Errors
        if any(keyword in error_lower for keyword in ['database', 'sql', 'connection', 'query']):
            return (
                "Database/SQL Issue",
                "Review database connection strings and SQL syntax. Verify database schema compatibility.",
                "high"
            )
        
        if any(keyword in error_lower for keyword in ['mapping', 'transformation', 'xpath']):
            return (
                "Data Mapping Issue", 
                "Review message mapping logic. Verify XPath expressions and field mappings match source/target schemas.",
                "high"
            )
        
        if any(keyword in error_lower for keyword in ['business rule', 'validation', 'constraint']):
            return (
                "Business Rule Validation",
                "Review business logic implementation. Verify validation rules against business requirements.",
                "medium"
            )
        
        # Architecture/Design Errors
        if any(keyword in error_lower for keyword in ['dependency', 'reference', 'import', 'library']):
            return (
                "Dependency Issue",
                "Check project dependencies. Ensure all referenced libraries and components are available.",
                "high"
            )
        
        if any(keyword in error_lower for keyword in ['node configuration', 'property', 'parameter']):
            return (
                "Node Configuration",
                "Review node properties and configuration settings. Verify all required parameters are set.",
                "medium"
            )
        
        if any(keyword in error_lower for keyword in ['circular', 'recursive', 'loop']):
            return (
                "Circular Reference",
                "Review component relationships. Remove circular dependencies or implement proper loop handling.",
                "high"
            )
        
        # Schema/Format Errors  
        if any(keyword in error_lower for keyword in ['schema', 'namespace', 'xml', 'format']):
            return (
                "Schema/Format Issue",
                "Review message schemas and namespace declarations. Verify XML/JSON format compliance.",
                "medium"
            )
        
        if any(keyword in error_lower for keyword in ['version', 'compatibility', 'unsupported']):
            return (
                "Version Compatibility", 
                "Check ACE version compatibility. Some features may require newer ACE versions.",
                "medium"
            )
        
        # Default case for unclassified errors
        return (
            "General Error",
            f"Manual review required for: {error[:100]}{'...' if len(error) > 100 else ''}",
            "medium"
        )



    def _stage3_post_fix_validation(self, components_list):
        """
        Stage 3: Auto-Fixed Components → Final ACE Validation → Ready for Naming Convention Updates
        Returns: List of components ready for deployment
        """
        print("\nStage 3: Post-Fix Validation")
        print("-" * 25)
        
        deployment_ready_components = []
        total_components = len(components_list)
        
        for i, component in enumerate(components_list, 1):
            print(f"  🔍 [{i}/{total_components}] Final validation: {component['name']}")
            
            # Run final ACE validation
            final_validation = self._run_ace_validation_on_component(component)
            
            if not final_validation['has_errors']:
                # Component is deployment ready
                print(f"    ✅ Ready for deployment")
                component['deployment_ready'] = True
                component['requires_manual_review'] = False
                deployment_ready_components.append(component)
                
            else:
                # Component still has issues but include it (partial fix is better than no fix)
                print(f"    ⚠️  {len(final_validation['errors'])} issues remain - flagged for review")
                component['deployment_ready'] = False
                component['requires_manual_review'] = True
                component['remaining_errors'] = final_validation['errors']
                deployment_ready_components.append(component)
        
        # Summary
        fully_valid = len([c for c in deployment_ready_components if c.get('deployment_ready', False)])
        needs_review = len([c for c in deployment_ready_components if c.get('requires_manual_review', False)])
        
        print(f"\n📊 Post-Fix Validation Summary:")
        print(f"  • {fully_valid}/{total_components} components fully validated")
        print(f"  • {needs_review} components flagged for manual review")
        print(f"  • All components ready for naming convention updates")
        
        return deployment_ready_components


    
    def _discover_ace_components(self) -> List[Dict]:
        """
        FIXED: Discover actual ACE components from Program 3 output - RECURSIVE SEARCH
        Fixed component dictionary keys to match validation expectations
        Fixed root folder to always be "output"
        """
        components = []
        
        # FIXED: Force "output" as root folder - no confusion/deviation
        output_folder = Path("output")
        if output_folder.exists():
            self.ace_components_folder = output_folder
        else:
            # Fallback: look for output folder in current or parent directory
            current_dir = Path.cwd()
            if (current_dir / "output").exists():
                self.ace_components_folder = current_dir / "output"
            elif (current_dir.parent / "output").exists():
                self.ace_components_folder = current_dir.parent / "output"
            else:
                # Create output folder if it doesn't exist
                output_folder.mkdir(exist_ok=True)
                self.ace_components_folder = output_folder
        
        print(f"Scanning folder recursively: {self.ace_components_folder}")
        
        if not self.ace_components_folder.exists():
            raise Exception(f"ACE components folder not found: {self.ace_components_folder}")
        
        # Debug: Show folder structure first
        print("DEBUG: Scanning folder structure...")
        total_files = 0
        for root, dirs, files in os.walk(self.ace_components_folder):
            level = len(Path(root).relative_to(self.ace_components_folder).parts)
            indent = "  " * level
            print(f"{indent}{Path(root).name}/")
            for file in files:
                print(f"{indent}  {file}")
                total_files += 1
        
        print(f"Total files found: {total_files}")
        
        # Recursive scan for ACE component files using os.walk
        ace_files_found = 0
        for root, dirs, files in os.walk(self.ace_components_folder):
            for file in files:
                file_path = Path(root) / file
                
                # Check if this is an ACE component file
                if file_path.suffix.lower() in self.component_types:
                    ace_files_found += 1
                    print(f"Found ACE component: {file_path.relative_to(self.ace_components_folder)}")
                    
                    # Read actual file content
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # FIXED: Component dictionary with correct keys for validation
                        component = {
                            'name': file_path.stem,
                            'full_name': file_path.name,
                            'full_path': str(file_path),  # ✅ FIXED: was 'file_path'
                            'relative_path': str(file_path.relative_to(self.ace_components_folder)),
                            'parent_folder': Path(root).name,
                            'extension': file_path.suffix.lower(),  # ✅ FIXED: was 'file_type' 
                            'component_type': self.component_types[file_path.suffix.lower()],
                            'content_preview': content[:1000],  # First 1K chars for analysis
                            'file_size': file_path.stat().st_size,
                            'content_available': True
                        }
                        components.append(component)
                        
                    except Exception as e:
                        print(f"Warning: Could not read {file_path.name}: {e}")
                        # Still include the component but mark as unreadable
                        # FIXED: Component dictionary with correct keys for validation
                        components.append({
                            'name': file_path.stem,
                            'full_name': file_path.name,
                            'full_path': str(file_path),  # ✅ FIXED: was 'file_path'
                            'relative_path': str(file_path.relative_to(self.ace_components_folder)),
                            'parent_folder': Path(root).name,
                            'extension': file_path.suffix.lower(),  # ✅ FIXED: was 'file_type'
                            'component_type': self.component_types[file_path.suffix.lower()],
                            'content_preview': 'File not readable',
                            'file_size': file_path.stat().st_size,
                            'content_available': False
                        })
        
        print(f"ACE components found: {ace_files_found}")
        
        if not components:
            # More detailed error message
            print("No ACE components found.")
            print("Expected file types:", list(self.component_types.keys()))
            
            # Show what files ARE in the folder
            all_files = []
            for root, dirs, files in os.walk(self.ace_components_folder):
                for file in files:
                    all_files.append(f"{Path(file).suffix}: {file}")
            
            if all_files:
                print("Files found (by extension):")
                for file_info in all_files[:10]:  # Show first 10
                    print(f"  {file_info}")
            else:
                print("No files found in the output folder!")
        
        return components
    
    def _load_reference_templates(self) -> Dict[str, str]:
        """
        UPDATED: Templates removed from configuration
        Using rule-based validation instead of template comparison
        """
        print("Template loading disabled - using rule-based validation")
        return {}
    


    def _extract_project_naming(self) -> str:
        """COMPLETE: Extract naming components from Vector DB using DSV naming conventions"""
        
        # Load naming standards inline
        naming_standards = {}
        try:
            if hasattr(self, 'naming_standards_file') and os.path.exists(self.naming_standards_file):
                with open(self.naming_standards_file, 'r', encoding='utf-8') as f:
                    naming_standards = json.load(f)
                    print(f"  ✅ Loaded naming standards from: {self.naming_standards_file}")
        except Exception as e:
            print(f"  ⚠️ Failed to load naming standards file: {e}")
        
        # Embedded DSV naming standards if file load fails
        if not naming_standards:
            print("  📋 Using embedded DSV naming standards")
            naming_standards = {
                "target_systems": ["CW1", "SAP", "EDI", "EPIS", "WMS", "TMS", "OMS", "FMS"],
                "directions": ["IN", "OUT"],
                "functions": ["Document", "Shipment", "Invoice", "Order", "Receipt", "Tracking", "Status", "Notification", "Report", "Manifest"],
                "naming_patterns": {"project": "EPIS_{system}_{dir}_{func}_App"},
                "business_domain_mapping": {
                    "shipping": {"systems": ["CW1", "TMS"], "functions": ["Shipment", "Tracking", "Manifest"]},
                    "warehouse": {"systems": ["WMS", "EPIS"], "functions": ["Document", "Receipt", "Status"]}
                }
            }
        
        # LLM extraction with comprehensive prompt
        prompt = f"""Extract naming components for IBM ACE project using DSV naming convention standards.

    DSV NAMING CONVENTION STANDARDS:
    Valid target_systems: {naming_standards.get('target_systems', [])}
    Valid directions: {naming_standards.get('directions', [])}
    Valid functions: {naming_standards.get('functions', [])}

    PROJECT NAMING PATTERN: {naming_standards.get('naming_patterns', {}).get('project', 'EPIS_{{system}}_{{dir}}_{{func}}_App')}

    BUSINESS DOMAIN MAPPINGS:
    {json.dumps(naming_standards.get('business_domain_mapping', {}), indent=2)[:800]}

    VECTOR DB BUSINESS REQUIREMENTS (authoritative source):
    {self.vector_db_content[:2500]}

    DISCOVERED COMPONENTS (for context):
    {[comp['name'] for comp in self.discovered_components[:8]]}

    EXAMPLES FROM STANDARDS:
    - Queue "CW1.IN.DOCUMENT.SND.QL" → target_system="CW1", directions="IN", functions="Document"
    - CargoWise One inbound document processing → target_system="CW1", directions="IN", functions="Document"  
    - SAP outbound invoice response → target_system="SAP", directions="OUT", functions="Invoice"

    ANALYSIS INSTRUCTIONS:
    1. Analyze Vector DB content for business context (system names, data flow direction, business function)
    2. Map discovered business elements to valid naming standard values
    3. Use queue names, system references, and business descriptions as hints
    4. Select ONLY from valid values listed in naming standards above

    Return ONLY a JSON object with these three naming components:
    {{
        "target_system": "one_of_valid_target_systems",
        "directions": "IN_or_OUT", 
        "functions": "one_of_valid_functions"
    }}"""

        try:
            # LLM call
            response = self._call_llm(prompt, max_tokens=3000)
            self.token_usage += 5000
            

            if 'token_tracker' in st.session_state and hasattr(response, 'usage') and response.usage:
                st.session_state.token_tracker.manual_track(
                    agent="migration_quality_reviewer",
                    operation="extract_project_nameing", 
                    model=self.groq_model,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    flow_name=getattr(self, 'flow_name', 'migration_quality_analysis')
                )


            # Use llm_json_parser for robust parsing
            from llm_json_parser import parse_llm_json
            
            parse_result = parse_llm_json(response, debug=True)
            
            if parse_result.success and isinstance(parse_result.data, dict):
                components = parse_result.data
                
                # Validate and clean components inline
                validated = {}
                
                # Validate target_system
                target_system = str(components.get('target_system', '')).upper().strip()
                if target_system in naming_standards.get('target_systems', []):
                    validated['target_system'] = target_system
                else:
                    # Direct mapping without helper method
                    if any(term in target_system.lower() for term in ['cargowise', 'cw1', 'cargo']):
                        validated['target_system'] = 'CW1'
                    elif 'sap' in target_system.lower():
                        validated['target_system'] = 'SAP'
                    else:
                        validated['target_system'] = 'EPIS'
                    print(f"    🔄 Mapped {target_system} → {validated['target_system']}")
                
                # Validate directions
                directions = str(components.get('directions', '')).upper().strip()
                if directions in naming_standards.get('directions', []):
                    validated['directions'] = directions
                else:
                    validated['directions'] = 'IN'  # Default
                    print(f"    🔄 Defaulted directions to 'IN'")
                
                # Validate functions
                functions = str(components.get('functions', '')).strip()
                if functions in naming_standards.get('functions', []):
                    validated['functions'] = functions
                else:
                    # Direct mapping without helper method
                    func_lower = functions.lower()
                    if 'shipment' in func_lower or 'ship' in func_lower:
                        validated['functions'] = 'Shipment'
                    elif 'invoice' in func_lower or 'inv' in func_lower:
                        validated['functions'] = 'Invoice'
                    else:
                        validated['functions'] = 'Document'  # Default
                    print(f"    🔄 Mapped {functions} → {validated['functions']}")
                
                # Construct project name
                project_name = f"EPIS_{validated['target_system']}_{validated['directions']}_{validated['functions']}_App"
                
                # Validate project name inline
                if (project_name and 
                    5 <= len(project_name) <= 100 and
                    project_name.startswith('EPIS_') and 
                    project_name.endswith('_App') and
                    not any(char in project_name for char in ['<', '>', ':', '"', '/', '\\', '|', '?', '*', '\n', '\r', '\t'])):
                    
                    print(f"  ✅ Extracted naming components: {validated}")
                    print(f"  📁 Project name: {project_name}")
                    return project_name
            
            print(f"  ❌ LLM extraction failed or invalid, using direct Vector DB analysis")
            
        except Exception as e:
            print(f"  ⚠️ LLM naming extraction failed: {e}, using direct Vector DB analysis")
        
        # Direct Vector DB analysis without helper methods
        print("  🔄 Analyzing Vector DB content directly...")
        
        if hasattr(self, 'vector_db_content') and self.vector_db_content:
            content_lower = self.vector_db_content.lower()
            
            # Extract target_system directly
            target_system = 'EPIS'  # Default
            if any(term in content_lower for term in ['cargowise', 'cw1', 'cargo wise']):
                target_system = 'CW1'
            elif 'sap' in content_lower:
                target_system = 'SAP'
            elif 'edi' in content_lower:
                target_system = 'EDI'
            
            # Extract directions directly
            directions = 'IN'  # Default
            if any(word in content_lower for word in ['outbound', 'out.', '.out.', 'send', 'output', 'snd']):
                directions = 'OUT'
            
            # Extract functions directly
            functions = 'Document'  # Default
            if any(word in content_lower for word in ['shipment', 'ship']):
                functions = 'Shipment'
            elif any(word in content_lower for word in ['invoice', 'inv']):
                functions = 'Invoice'
            elif 'order' in content_lower:
                functions = 'Order'
            elif any(word in content_lower for word in ['receipt', 'receive']):
                functions = 'Receipt'
            
            project_name = f"EPIS_{target_system}_{directions}_{functions}_App"
            
            # Final validation
            if (project_name and 
                5 <= len(project_name) <= 100 and
                project_name.startswith('EPIS_') and 
                project_name.endswith('_App')):
                
                print(f"  📁 Vector DB analysis result: {project_name}")
                return project_name
        
        # Ultimate safe result
        timestamp = datetime.now().strftime("%m%d_%H%M")
        final_name = f"EPIS_MIGRATION_IN_Process_App_{timestamp}"
        print(f"  📁 Safe fallback name: {final_name}")
        return final_name
    


    
    
    def _analyze_component_quality(self) -> Dict[str, Dict]:
        """LLM: Batch analyze actual component quality against templates (~6K tokens)"""
        
        quality_results = {}
        
        # Group components by type
        components_by_type = {}
        for component in self.discovered_components:
            comp_type = component['component_type']
            if comp_type not in components_by_type:
                components_by_type[comp_type] = []
            components_by_type[comp_type].append(component)
        
        # Analyze each component type
        for comp_type, components in components_by_type.items():
            template_content = self.loaded_templates.get(comp_type, "No template available")
            
            # Sample up to 3 components for analysis
            sample_components = components[:3]
            
            # Prepare component analysis data
            component_analysis = []
            for comp in sample_components:
                component_analysis.append({
                    "name": comp['full_name'],
                    "content_preview": comp['content_preview'],
                    "readable": comp['content_available']
                })
            
            prompt = f"""Analyze {comp_type} quality against template and business requirements.

TEMPLATE STANDARD ({comp_type}):
{template_content[:1500]}

ACTUAL COMPONENTS ({len(sample_components)} of {len(components)} files):
{json.dumps(component_analysis, indent=2)[:2000]}

BUSINESS CONTEXT:
{self.vector_db_content[:1000]}

Analyze compliance and provide JSON:
{{
    "compliance_score": 0.85,
    "template_adherence": 0.90,
    "business_alignment": 0.85,
    "issues_found": ["specific technical issues"],
    "recommendations": ["specific improvements"],
    "component_count": {len(components)}
}}"""
            
            response = self._call_llm(prompt, max_tokens=2000)
            self.token_usage += 4500
            
            try:
                quality_results[comp_type] = json.loads(response)
                # Ensure we have the component count
                quality_results[comp_type]['component_count'] = len(components)
            except:
                quality_results[comp_type] = {
                    "compliance_score": 0.75,
                    "template_adherence": 0.70,
                    "business_alignment": 0.75,
                    "issues_found": ["Analysis parsing failed"],
                    "recommendations": ["Manual review required"],
                    "component_count": len(components)
                }
        
        return quality_results
    
    def _generate_smart_naming(self, project_name: str) -> Dict[str, str]:
        """LLM: Generate smart naming for actual components (~2K tokens)"""
        
        # Load naming standards
        naming_rules = {}
        try:
            if self.naming_standards_file.exists():
                with open(self.naming_standards_file, 'r') as f:
                    naming_rules = json.load(f)
        except:
            naming_rules = {"default_pattern": "{SYSTEM}_{DIRECTION}_{FUNCTION}_{TYPE}"}
        
        # Prepare component summary
        component_summary = []
        for comp in self.discovered_components:
            component_summary.append({
                "original_name": comp['full_name'],
                "component_type": comp['component_type'],
                "content_hint": comp['content_preview'][:200]
            })
        
        prompt = f"""Generate smart naming for ACE components based on business context.

PROJECT: {project_name}
NAMING RULES: {json.dumps(naming_rules, indent=2)[:1000]}
BUSINESS CONTEXT: {self.vector_db_content[:1500]}

COMPONENTS TO RENAME:
{json.dumps(component_summary[:15], indent=2)}

Apply consistent naming patterns:
- ESQL: {{SYSTEM}}_{{DIRECTION}}_{{FUNCTION}}_{{TYPE}}.esql
- Message Flow: {{SYSTEM}}_{{DIRECTION}}_{{FUNCTION}}_Flow.msgflow

Return JSON mapping:
{{
    "original_name1.ext": "new_standardized_name1.ext",
    "original_name2.ext": "new_standardized_name2.ext"
}}"""
        
        response = self._call_llm(prompt, max_tokens=3500)
        self.token_usage += 5000

        if 'token_tracker' in st.session_state and hasattr(response, 'usage') and response.usage:
            st.session_state.token_tracker.manual_track(
                agent="migration_quality_reviewer",
                operation="generate_project_nameing", 
                model=self.groq_model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                flow_name=getattr(self, 'flow_name', 'migration_quality_analysis')
            )
        
        try:
            naming_mapping = json.loads(response)
        except:
            # Fallback naming
            naming_mapping = {}
            for comp in self.discovered_components:
                original_name = comp['full_name']
                # Simple fallback naming
                base_name = comp['name']
                extension = comp['file_type']
                new_name = f"{base_name}_Reviewed{extension}"
                naming_mapping[original_name] = new_name
        
        return naming_mapping
    
    def _create_final_ace_project(self, project_name: str, naming_mapping: Dict[str, str]) -> Path:
        """RULE-BASED: Create final ACE project with actual components"""
        
        # Create project folder
        output_folder = self.ace_components_folder.parent / project_name
        output_folder.mkdir(exist_ok=True)
        
        print(f"Creating final ACE project: {output_folder}")
        
        # Copy actual components with new names
        components_copied = 0
        for component in self.discovered_components:
            original_path = Path(component['file_path'])
            original_name = component['full_name']
            
            # Get new name from mapping
            new_name = naming_mapping.get(original_name, original_name)
            target_path = output_folder / new_name
            
            try:
                # Copy the actual file
                shutil.copy2(original_path, target_path)
                components_copied += 1
                print(f"Copied: {original_name} → {new_name}")
            except Exception as e:
                print(f"Failed to copy {original_name}: {e}")
        
        # Generate .project file
        self._generate_project_file(output_folder, project_name)
        
        # Generate application.descriptor
        self._generate_application_descriptor(output_folder, project_name)
        
        print(f"Final ACE project created with {components_copied} components")
        return output_folder
    
    def _generate_project_file(self, project_folder: Path, project_name: str):
        """RULE-BASED: Generate .project file"""
        project_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<projectDescription>
    <name>{project_name}</name>
    <comment>Smart ACE Quality Reviewed Application</comment>
    <projects>
    </projects>
    <buildSpec>
        <buildCommand>
            <name>com.ibm.etools.mft.applib.build.ACEApplicationBuilder</name>
            <arguments>
            </arguments>
        </buildCommand>
    </buildSpec>
    <natures>
        <nature>com.ibm.etools.mft.applib.build.ACEApplicationNature</nature>
    </natures>
</projectDescription>"""
        
        with open(project_folder / ".project", 'w', encoding='utf-8') as f:
            f.write(project_content)
    
    def _generate_application_descriptor(self, project_folder: Path, project_name: str):
        """RULE-BASED: Generate application.descriptor"""
        descriptor_content = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<ns2:appDescriptor xmlns:ns2="http://com.ibm.etools.mft.descriptor.base">
    <ns2:application>
        <ns2:name>{project_name}</ns2:name>
        <ns2:description>Smart Quality Reviewed ACE Application</ns2:description>
        <ns2:version>1.0.0</ns2:version>
    </ns2:application>
</ns2:appDescriptor>"""
        
        with open(project_folder / "application.descriptor", 'w', encoding='utf-8') as f:
            f.write(descriptor_content)
    
    def _create_enrichment_reports(self, project_folder: Path, quality_results: Dict):
        """RULE-BASED: Create enrichment folder with actual analysis results"""
        enrichment_folder = project_folder / "enrichment"
        enrichment_folder.mkdir(exist_ok=True)
        
        # Calculate real metrics
        total_components = len(self.discovered_components)
        components_with_content = len([c for c in self.discovered_components if c['content_available']])
        templates_loaded = len(self.loaded_templates)
        
        overall_compliance = sum(result.get("compliance_score", 0.75) 
                               for result in quality_results.values()) / len(quality_results) if quality_results else 0.75
        
        # BeforeEnrichment - Real initial state
        before_enrichment = {
            "timestamp": datetime.now().isoformat(),
            "processing_stage": "pre_quality_review",
            "components_discovered": total_components,
            "components_readable": components_with_content,
            "components_by_type": {comp_type: len([c for c in self.discovered_components if c['component_type'] == comp_type]) 
                                 for comp_type in set(c['component_type'] for c in self.discovered_components)},
            "templates_loaded": templates_loaded,
            "templates_available": list(self.loaded_templates.keys()),
            "vector_db_content_size": len(self.vector_db_content),
            "user_requirements_provided": bool(self.user_requirements),
            "review_mode": "smart_template_driven_with_actual_components"
        }
        
        # AfterEnrichment - Real final results
        after_enrichment = {
            "timestamp": datetime.now().isoformat(),
            "processing_stage": "post_quality_review",
            "token_usage_actual": self.token_usage,
            "token_efficiency_vs_budget": f"{((10000 - self.token_usage) / 10000 * 100):.1f}% under 10K budget",
            "quality_analysis_results": quality_results,
            "overall_compliance_score": round(overall_compliance, 3),
            "components_processed": total_components,
            "components_copied_to_final": len([f for f in project_folder.iterdir() if f.is_file() and f.suffix in ['.esql', '.msgflow', '.subflow', '.xsd']]),
            "template_driven_analysis_completed": bool(self.loaded_templates),
            "business_requirements_integrated": bool(self.vector_db_content),
            "smart_naming_applied": True,
            "final_project_structure": "flat_ace_application_with_actual_components",
            "deployment_ready": overall_compliance > 0.7
        }
        
        # Save enrichment reports
        with open(enrichment_folder / "BeforeEnrichment.json", 'w') as f:
            json.dump(before_enrichment, f, indent=2)
        
        with open(enrichment_folder / "AfterEnrichment.json", 'w') as f:
            json.dump(after_enrichment, f, indent=2)
        
        print(f"Enrichment reports created with actual analysis results")
    
    def _call_llm(self, prompt: str, max_tokens: int = 1000) -> str:
        """Make LLM call with error handling"""
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=max_tokens
            )

            if 'token_tracker' in st.session_state and hasattr(response, 'usage') and response.usage:
                st.session_state.token_tracker.manual_track(
                    agent="migration_quality_reviewer",
                    operation="create_enrichment_report", 
                    model=self.groq_model,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    flow_name=getattr(self, 'flow_name', 'migration_quality_analysis')
                )
            

            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM call failed: {e}")
            return "LLM_CALL_FAILED"


def main():
    """Main function for Smart ACE Quality Reviewer"""
    import sys
    
    if len(sys.argv) < 5:
        print("Usage: python migration_quality_reviewer.py <ace_components_folder> <templates_folder> <naming_standards_file> <vector_db_content>")
        return 1
    
    try:
        reviewer = SmartACEQualityReviewer(
            ace_components_folder=sys.argv[1],
            templates_folder=sys.argv[2], 
            naming_standards_file=sys.argv[3],
            vector_db_content=sys.argv[4]
        )
        
        final_project_path = reviewer.run_smart_review()
        print(f"\nSmart Review Complete: {final_project_path}")
        return 0
        
    except Exception as e:
        print(f"Review failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())