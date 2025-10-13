#!/usr/bin/env python3
"""
ACE Module Creator v3.0 - Clean Architecture
Pure orchestration with no hardcoded fallbacks or flow-specific logic
Designed for 1000+ flows with consistent execution pattern
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass
import streamlit as st
try:
    from schema_generator import SchemaGenerator
    from esql_generator import ESQLGenerator
    from xsl_generator import XSLGenerator
    from application_descriptor_generator import ApplicationDescriptorGenerator
    from enrichment_generator import EnrichmentGenerator
    from project_generator import ProjectGenerator
except ImportError as e:
    raise ImportError(f"Missing required generator modules: {e}")


@dataclass
class FlowContext:
    """Context for a single messageflow processing"""
    msgflow_path: str
    naming_convention_path: str  # Points to root folder naming file
    flow_name: str
    output_subdir: str


class ACEModuleCreator:
    """Clean orchestrator for ACE module generation"""
    
    def __init__(self, groq_api_key: str, output_dir: str = "output"):
        self.groq_api_key = groq_api_key
        self.output_dir = Path(output_dir)
        self.business_requirements_path = self.output_dir / "business_requirements.json"
        
        # Template paths (all in templates folder except ESQL)
        self.templates = {
            'esql': Path("templates/ESQL_Template_Updated.ESQL"),
            'app_descriptor': Path("templates/application_descriptor.xml"),
            'project': Path("templates/project_template.xml")
        }
        
        self.stats = {
            'flows_processed': 0,
            'flows_succeeded': 0,
            'flows_failed': 0,
            'total_llm_calls': 0,
            'start_time': None,
            'end_time': None
        }
        
    def execute(self) -> Dict:
        """Main execution entry point"""
        self.stats['start_time'] = datetime.now()
        
        print("🚀 ACE Module Creator v3.0")
        print("=" * 70)
        
        # Step 1: Validate prerequisites
        self._validate_prerequisites()
        
        # Step 2: Discover and map flows
        flow_contexts = self._discover_and_map_flows()
        
        if not flow_contexts:
            raise RuntimeError("No messageflows found in output directory")
        
        print(f"\n📊 Found {len(flow_contexts)} messageflow(s) to process")
        
        # Step 3: Process each flow independently
        results = []
        for idx, context in enumerate(flow_contexts, 1):
            print(f"\n{'='*70}")
            print(f"Processing Flow {idx}/{len(flow_contexts)}: {context.flow_name}")
            print(f"{'='*70}")
            
            result = self._process_single_flow(context)
            results.append(result)
            
            self.stats['flows_processed'] += 1
            if result['status'] == 'success':
                self.stats['flows_succeeded'] += 1
            else:
                self.stats['flows_failed'] += 1
        
        self.stats['end_time'] = datetime.now()
        
        # Step 4: Generate final report
        return self._generate_final_report(results)
    
    def _validate_prerequisites(self):
        """Validate all required files exist"""
        print("\n🔍 Validating prerequisites...")
        
        missing = []
        
        # Check output directory
        if not self.output_dir.exists():
            missing.append(f"Output directory: {self.output_dir}")
        
        # Check business requirements
        if not self.business_requirements_path.exists():
            missing.append(f"Business requirements: {self.business_requirements_path}")
        
        # Check templates
        for name, path in self.templates.items():
            if not path.exists():
                missing.append(f"{name} template: {path}")
        
        if missing:
            raise FileNotFoundError(f"Missing required files:\n" + "\n".join(f"  - {m}" for m in missing))
        
        print("✅ All prerequisites validated")
    
    def _discover_and_map_flows(self) -> List[FlowContext]:
        """Discover messageflows and map to naming conventions by flow name"""
        import shutil
        
        print("\n🔍 Discovering messageflows and naming conventions...")
        
        # Determine if single or multiple flow mode
        single_dir = self.output_dir / "single"
        multiple_dir = self.output_dir / "multiple"
        
        msgflow_files = []
        base_path = None
        
        if single_dir.exists():
            print("  📁 Mode: Single MessageFlow")
            base_path = single_dir
            msgflow_files = list(single_dir.glob("*/*.msgflow"))
        elif multiple_dir.exists():
            print("  📁 Mode: Multiple MessageFlows")
            base_path = multiple_dir
            msgflow_files = list(multiple_dir.glob("*/*.msgflow"))
        else:
            raise RuntimeError("Neither output/single/ nor output/multiple/ folder found")
        
        # Find all naming_convention*.json files in ROOT directory
        root_dir = Path(".")
        naming_files = sorted(root_dir.glob("naming_convention*.json"))
        
        print(f"  Found {len(msgflow_files)} messageflow(s)")
        print(f"  Found {len(naming_files)} naming convention(s) in root/")
        
        # Validation: counts must match
        if len(msgflow_files) != len(naming_files):
            raise ValueError(
                f"Mismatch: {len(msgflow_files)} messageflows but {len(naming_files)} naming conventions. "
                "Each messageflow requires exactly one naming convention file."
            )
        
        # Build mapping: flow_name → naming_convention_path
        naming_map = {}
        for naming_file in naming_files:
            with open(naming_file, 'r') as f:
                naming_data = json.load(f)
            
            # Extract flow name from naming convention
            flow_name = naming_data.get('project_naming', {}).get('message_flow_name')
            if not flow_name:
                raise ValueError(f"Missing 'message_flow_name' in {naming_file}")
            
            naming_map[flow_name] = str(naming_file)
            print(f"  Mapped: {flow_name} ← {naming_file.name}")
        
        # Create flow contexts by matching messageflow files with naming map
        contexts = []
        for msgflow_path in msgflow_files:
            flow_name = msgflow_path.stem
            
            # Find matching naming convention
            if flow_name not in naming_map:
                raise ValueError(
                    f"No naming convention found for messageflow: {flow_name}\n"
                    f"Available naming conventions: {list(naming_map.keys())}"
                )
            
            # Output subfolder already exists (created by Program 2)
            output_subdir = msgflow_path.parent
            
            # Copy naming convention to flow subfolder (rename to naming_convention.json)
            source_naming = Path(naming_map[flow_name])
            target_naming = output_subdir / "naming_convention.json"
            shutil.copy2(source_naming, target_naming)
            
            contexts.append(FlowContext(
                msgflow_path=str(msgflow_path),
                naming_convention_path=str(target_naming),  # Copied path in subfolder
                flow_name=flow_name,
                output_subdir=str(output_subdir)
            ))
            
            print(f"  ✅ Flow: {flow_name}")
            print(f"     Messageflow: {msgflow_path.name}")
            print(f"     Naming: {source_naming.name} → naming_convention.json")
            print(f"     Output: {output_subdir}")
        
        return contexts
    
    def _detect_required_modules(self, msgflow_path: str) -> Dict[str, bool]:
        """Analyze messageflow to determine which generators to run"""
        import xml.etree.ElementTree as ET
        
        print(f"  🔍 Analyzing messageflow nodes...")
        
        try:
            tree = ET.parse(msgflow_path)
            root = tree.getroot()
            
            # Convert to string for simple pattern matching
            msgflow_content = ET.tostring(root, encoding='unicode')
            
            required = {
                'schema': True,  # Always needed for message parsing
                'esql': False,
                'xsl': False,
                'enrichment': False,
                'app_descriptor': True,  # Always needed
                'project': True  # Always needed
            }
            
            # Check for ESQL compute nodes
            if 'ComIbmCompute.msgnode' in msgflow_content or 'computeExpression="esql://' in msgflow_content:
                required['esql'] = True
                print(f"    ✅ Compute nodes detected → ESQL generation required")
            
            # Check for XSL transform nodes
            if 'ComIbmXslMqsi.msgnode' in msgflow_content or 'stylesheetName=' in msgflow_content:
                required['xsl'] = True
                print(f"    ✅ XSL transform detected → XSL generation required")
            
            # Check for enrichment subflows
            if 'EPIS_MessageEnrichment.subflow' in msgflow_content or 'BeforeEnrichment' in msgflow_content or 'AfterEnrichment' in msgflow_content:
                required['enrichment'] = True
                print(f"    ✅ Enrichment subflows detected → Enrichment generation required")
            
            return required
            
        except Exception as e:
            print(f"    ⚠️ Node detection failed: {e}, running all generators")
            # Fallback: run everything
            return {
                'schema': True,
                'esql': True,
                'xsl': True,
                'enrichment': True,
                'app_descriptor': True,
                'project': True
            }
    
    def _process_single_flow(self, context: FlowContext) -> Dict:
        """Process one messageflow through required generators only"""
        start_time = time.time()
        
        results = {
            'flow_name': context.flow_name,
            'status': 'in_progress',
            'modules': {},
            'skipped_modules': [],
            'errors': [],
            'llm_calls': 0
        }
        
        try:
            # Detect which modules are needed based on messageflow nodes
            required_modules = self._detect_required_modules(context.msgflow_path)
            
            print(f"\n📋 Execution Plan:")
            for module, needed in required_modules.items():
                status = "✅ Required" if needed else "⏭️ Skipped"
                print(f"  {module.upper()}: {status}")
            
            # Module 1: Schema Generation (conditional)
            if required_modules['schema']:
                results['modules']['schema'] = self._run_schema_generator(context)
            else:
                results['skipped_modules'].append('schema')
            
            # Module 2: ESQL Generation (conditional)
            if required_modules['esql']:
                results['modules']['esql'] = self._run_esql_generator(context)
            else:
                results['skipped_modules'].append('esql')
            
            # Module 3: XSL Generation (conditional)
            if required_modules['xsl']:
                results['modules']['xsl'] = self._run_xsl_generator(context)
            else:
                results['skipped_modules'].append('xsl')
            
            # Module 4: Application Descriptor (conditional)
            if required_modules['app_descriptor']:
                results['modules']['app_descriptor'] = self._run_app_descriptor_generator(context)
            else:
                results['skipped_modules'].append('app_descriptor')
            
            # Module 5: Enrichment Configuration (conditional)
            if required_modules['enrichment']:
                results['modules']['enrichment'] = self._run_enrichment_generator(context)
            else:
                results['skipped_modules'].append('enrichment')
            
            # Module 6: Project File (conditional)
            if required_modules['project']:
                results['modules']['project'] = self._run_project_generator(context)
            else:
                results['skipped_modules'].append('project')
            
            # Count total LLM calls
            results['llm_calls'] = sum(
                m.get('llm_calls', 0) for m in results['modules'].values()
            )
            self.stats['total_llm_calls'] += results['llm_calls']
            
            results['status'] = 'success'
            
        except Exception as e:
            results['status'] = 'failed'
            results['errors'].append(str(e))
            print(f"❌ Flow processing failed: {e}")
        
        results['execution_time'] = time.time() - start_time
        return results
    


    
    def _run_schema_generator(self, context: FlowContext) -> Dict:
        """Execute schema generation with Vector DB integration"""
        print("\n📊 Step 1: Schema Generation")
        start = time.time()
        
        try:
            import streamlit as st
            
            # ✅ Check if Vector DB is available
            if (st.session_state.get('vector_enabled', False) and 
                st.session_state.get('vector_ready', False) and 
                st.session_state.get('vector_pipeline')):
                
                print("  🚀 Using Vector DB for schema-focused content...")
                
                # Create agent function for schema generation
                def schema_agent_function(focused_content):
                    """Agent function that receives Vector DB focused content"""
                    generator = SchemaGenerator(groq_api_key=self.groq_api_key)
                    return generator.generate_schemas(
                        vector_content=focused_content,  # ✅ Use Vector DB content
                        business_requirements_json_path=str(self.business_requirements_path),
                        output_dir=context.output_subdir
                    )
                
                # Run with Vector DB pipeline
                result = st.session_state.vector_pipeline.run_agent_with_vector_search(
                    agent_name="schema_generator",  # ← Vector search for schema content
                    agent_function=schema_agent_function
                )
                
                print("  ✅ Vector DB processing completed!")
                
            else:
                # Fallback: Read business requirements file as vector content
                print("  ⚠️  Vector DB not available, using file-based content...")
                
                with open(self.business_requirements_path, 'r') as f:
                    business_content = f.read()
                
                generator = SchemaGenerator(groq_api_key=self.groq_api_key)
                result = generator.generate_schemas(
                    vector_content=business_content,  # ✅ Use file content as fallback
                    business_requirements_json_path=str(self.business_requirements_path),
                    output_dir=context.output_subdir
                )
            
            print(f"  ✅ Generated {result.get('schemas_generated', 0)} schema files")
            
            return {
                'status': 'success',
                'files_generated': result.get('schemas_generated', 0),
                'llm_calls': result.get('llm_analysis_calls', 0) + result.get('llm_generation_calls', 0),
                'execution_time': time.time() - start
            }
            
        except Exception as e:
            print(f"  ❌ Schema generation failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'execution_time': time.time() - start
            }
        


    
    def _run_esql_generator(self, context: FlowContext) -> Dict:
        """Execute ESQL generation with Vector DB"""
        print("\n⚡ Step 2: ESQL Generation")
        start = time.time()
        
        try:
            import streamlit as st
            
            # ✅ Check if Vector DB is available
            if (st.session_state.get('vector_enabled', False) and 
                st.session_state.get('vector_ready', False) and 
                st.session_state.get('vector_pipeline')):
                
                print("  🚀 Using Vector DB for ESQL business logic...")
                
                # Create agent function
                def esql_agent_function(focused_content):
                    generator = ESQLGenerator(groq_api_key=self.groq_api_key)
                    return generator.generate_esql_files(
                        vector_content=focused_content,  # ✅ Use Vector DB content
                        esql_template={'path': str(self.templates['esql'])},
                        msgflow_content={'path': context.msgflow_path},
                        json_mappings={'path': str(self.business_requirements_path)},
                        output_dir=os.path.join(context.output_subdir, "esql")
                    )
                
                # Run with Vector DB
                result = st.session_state.vector_pipeline.run_agent_with_vector_search(
                    agent_name="esql_generator",
                    agent_function=esql_agent_function
                )
            else:
                # Fallback: Use business_requirements file content
                with open(self.business_requirements_path, 'r') as f:
                    business_content = f.read()
                
                generator = ESQLGenerator(groq_api_key=self.groq_api_key)
                result = generator.generate_esql_files(
                    vector_content=business_content,  # ✅ Use file content as fallback
                    esql_template={'path': str(self.templates['esql'])},
                    msgflow_content={'path': context.msgflow_path},
                    json_mappings={'path': str(self.business_requirements_path)},
                    output_dir=os.path.join(context.output_subdir, "esql") 
                )
            
            print(f"  ✅ Generated {result.get('successful', 0)} ESQL modules")
            
            return {
                'status': 'success',
                'files_generated': result.get('successful', 0),
                'llm_calls': result.get('llm_calls', 0),
                'execution_time': time.time() - start
            }
            
        except Exception as e:
            print(f"  ❌ ESQL generation failed: {e}")
            return {'status': 'failed', 'error': str(e), 'execution_time': time.time() - start}
        

    
    def _run_xsl_generator(self, context: FlowContext) -> Dict:
        """Execute XSL generation with Vector DB integration"""
        print("\n🎨 Step 3: XSL Generation")
        start = time.time()
        
        try:
            import streamlit as st
            
            # ✅ Check if Vector DB is available
            if (st.session_state.get('vector_enabled', False) and 
                st.session_state.get('vector_ready', False) and 
                st.session_state.get('vector_pipeline')):
                
                print("  🚀 Using Vector DB for XSL transformation-focused content...")
                
                # Create agent function for XSL generation
                def xsl_agent_function(focused_content):
                    """Agent function that receives Vector DB focused content for XSL processing"""
                    generator = XSLGenerator(groq_api_key=self.groq_api_key)
                    return generator.generate_xsl_transformations(
                        vector_content=focused_content,  # ✅ Use Vector DB content
                        business_requirements_json_path=str(self.business_requirements_path),
                        output_dir=context.output_subdir,
                        flow_name=context.flow_name
                    )
                
                # Run with Vector DB pipeline
                result = st.session_state.vector_pipeline.run_agent_with_vector_search(
                    agent_name="xsl_generator",  # ← Vector search for XSL transformation content
                    agent_function=xsl_agent_function
                )
                
                print("  ✅ Vector DB processing completed!")
                
            else:
                # Fallback: Read business requirements file as vector content
                print("  ⚠️  Vector DB not available, using file-based content...")
                
                with open(self.business_requirements_path, 'r') as f:
                    business_content = f.read()
                
                generator = XSLGenerator(groq_api_key=self.groq_api_key)
                result = generator.generate_xsl_transformations(
                    vector_content=business_content,  # ✅ Use file content as fallback
                    business_requirements_json_path=str(self.business_requirements_path),
                    output_dir=context.output_subdir,
                    flow_name=context.flow_name
                )
            
            print(f"  ✅ Generated {result.get('xsl_transformations_generated', 0)} XSL files")
            
            return {
                'status': 'success',
                'files_generated': result.get('xsl_transformations_generated', 0),
                'llm_calls': result.get('llm_analysis_calls', 0) + result.get('llm_generation_calls', 0),
                'execution_time': time.time() - start
            }
            
        except Exception as e:
            print(f"  ❌ XSL generation failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'execution_time': time.time() - start
            }
        


    
    def _run_app_descriptor_generator(self, context: FlowContext) -> Dict:
        """Execute application descriptor generation"""
        print("\n📋 Step 4: Application Descriptor Generation")
        start = time.time()
        
        try:
            generator = ApplicationDescriptorGenerator(groq_api_key=self.groq_api_key)
            
            result = generator.generate_application_descriptor(
                vector_content="",
                business_requirements_json_path=str(self.business_requirements_path),
                template_path=str(self.templates['app_descriptor']),
                output_dir=context.output_subdir
            )
            
            print(f"  ✅ Generated application descriptor")
            
            return {
                'status': 'success',
                'files_generated': 1,
                'llm_calls': result.get('llm_analysis_calls', 0) + result.get('llm_generation_calls', 0),
                'execution_time': time.time() - start
            }
            
        except Exception as e:
            print(f"  ❌ Application descriptor generation failed: {e}")
            return {'status': 'failed', 'error': str(e), 'execution_time': time.time() - start}
    



    def _run_enrichment_generator(self, context: FlowContext) -> Dict:
        """Execute enrichment configuration generation with Vector DB integration"""
        print("\n🔋 Step 5: Enrichment Configuration Generation")
        start = time.time()
        
        try:
            import streamlit as st
            
            # ✅ Check if Vector DB is available
            if (st.session_state.get('vector_enabled', False) and 
                st.session_state.get('vector_ready', False) and 
                st.session_state.get('vector_pipeline')):
                
                print("  🚀 Using Vector DB for enrichment-focused content...")
                
                # Create agent function for enrichment generation
                def enrichment_agent_function(focused_content):
                    """Agent function that receives Vector DB focused content for enrichment processing"""
                    
                    # Enhanced Vector search for enrichment-specific content
                    try:
                        collection = st.session_state.vector_pipeline.search_engine.collection
                        
                        enrichment_queries = [
                            "enrichment business logic validation rules",
                            "database operations lookup validation",
                            "CargoWise eAdapter service integration",
                            "business validation rules entity processing",
                            "stored procedure database lookup exec",
                            "database connection alias integration service"
                        ]
                        
                        enhanced_content = []
                        for query in enrichment_queries:
                            try:
                                results = collection.query(query_texts=[query], n_results=50, include=['documents'])
                                if 'documents' in results and results['documents']:
                                    for doc_list in results['documents']:
                                        if isinstance(doc_list, list):
                                            enhanced_content.extend(doc_list)
                            except Exception as e:
                                continue
                        
                        # Combine focused content with enhanced enrichment content
                        combined_content = focused_content
                        if enhanced_content:
                            unique_enhanced = list(set(enhanced_content))
                            enhanced_text = "\n\n---ENRICHMENT_LOGIC---\n\n".join(unique_enhanced)
                            combined_content = f"{focused_content}\n\n---ENHANCED_ENRICHMENT---\n\n{enhanced_text}"
                            
                    except Exception as e:
                        combined_content = focused_content
                    
                    # Generate enrichment files with enhanced content
                    generator = EnrichmentGenerator(groq_api_key=self.groq_api_key)
                    return generator.generate_enrichment_files(
                        vector_content=combined_content,  # ✅ Use Vector DB content
                        business_requirements_json_path=str(self.business_requirements_path),
                        msgflow_path=context.msgflow_path,
                        output_dir=os.path.join(context.output_subdir, "enrichment")
                    )
                
                # Run with Vector DB pipeline
                result = st.session_state.vector_pipeline.run_agent_with_vector_search(
                    agent_name="enrichment_generator",  # ← Vector search for enrichment content
                    agent_function=enrichment_agent_function
                )
                
                print("  ✅ Vector DB processing completed!")
                
            else:
                # Fallback: Read business requirements file as vector content
                print("  ⚠️  Vector DB not available, using file-based content...")
                
                with open(self.business_requirements_path, 'r') as f:
                    business_content = f.read()
                
                generator = EnrichmentGenerator(groq_api_key=self.groq_api_key)
                result = generator.generate_enrichment_files(
                    vector_content=business_content,  # ✅ Use file content as fallback
                    business_requirements_json_path=str(self.business_requirements_path),
                    msgflow_path=context.msgflow_path,
                    output_dir=os.path.join(context.output_subdir, "enrichment")
                )
            
            print(f"  ✅ Generated {result.get('enrichment_configs_generated', 0)} enrichment files")
            
            return {
                'status': 'success',
                'files_generated': result.get('enrichment_configs_generated', 0),
                'llm_calls': result.get('llm_analysis_calls', 0) + result.get('llm_generation_calls', 0),
                'execution_time': time.time() - start
            }
            
        except Exception as e:
            print(f"  ❌ Enrichment generation failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'execution_time': time.time() - start
            }
        


    
    def _run_project_generator(self, context: FlowContext) -> Dict:
        """Execute project file generation"""
        print("\n🗂️ Step 6: Project File Generation")
        start = time.time()
        
        try:
            generator = ProjectGenerator()
            
            result = generator.generate_project_file(
                vector_content="",
                template_path=str(self.templates['project']),
                business_requirements_json_path=str(self.business_requirements_path),
                output_dir=context.output_subdir
            )
            
            print(f"  ✅ Generated .project file")
            
            return {
                'status': 'success',
                'files_generated': 1,
                'llm_calls': 0,
                'execution_time': time.time() - start
            }
            
        except Exception as e:
            print(f"  ❌ Project generation failed: {e}")
            return {'status': 'failed', 'error': str(e), 'execution_time': time.time() - start}
        
        
    
    def _generate_final_report(self, results: List[Dict]) -> Dict:
        """Generate comprehensive execution report"""
        total_time = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        report = {
            'summary': {
                'total_flows': len(results),
                'succeeded': self.stats['flows_succeeded'],
                'failed': self.stats['flows_failed'],
                'success_rate': f"{(self.stats['flows_succeeded'] / len(results) * 100):.1f}%",
                'total_llm_calls': self.stats['total_llm_calls'],
                'total_time_seconds': round(total_time, 2)
            },
            'flow_results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        print("\n" + "="*70)
        print("🎉 ACE MODULE CREATION COMPLETE")
        print("="*70)
        print(f"✅ Succeeded: {self.stats['flows_succeeded']}/{len(results)}")
        print(f"❌ Failed: {self.stats['flows_failed']}/{len(results)}")
        print(f"🧠 Total LLM calls: {self.stats['total_llm_calls']}")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print("="*70)
        
        return report


def create_ace_project(groq_api_key: str, output_dir: str = "output") -> Dict:
    """
    Main entry point for integration with main.py
    Compatible with existing pipeline pattern
    
    Args:
        groq_api_key: Groq API key
        output_dir: Output directory containing messageflows and business_requirements.json
    
    Returns:
        Dict with orchestration results matching main.py expectations
    """
    try:
        creator = ACEModuleCreator(groq_api_key=groq_api_key, output_dir=output_dir)
        report = creator.execute()
        
        # Save report
        report_path = Path(output_dir) / "ace_generation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Return in format expected by main.py
        return {
            'orchestration_status': 'success' if report['summary']['failed'] == 0 else 'partial_success',
            'total_execution_time': report['summary']['total_time_seconds'],
            'successful_modules': report['summary']['succeeded'],
            'failed_modules': report['summary']['failed'],
            'total_files_generated': sum(
                sum(m.get('files_generated', 0) for m in flow.get('modules', {}).values())
                for flow in report['flow_results']
            ),
            'total_llm_calls': report['summary']['total_llm_calls'],
            'module_results': report['flow_results'],
            'output_directory': output_dir,
            'generation_timestamp': report['timestamp']
        }
        
    except Exception as e:
        return {
            'orchestration_status': 'failed',
            'error_message': str(e),
            'total_execution_time': 0,
            'successful_modules': 0,
            'failed_modules': 0,
            'output_directory': output_dir
        }


def main():
    """CLI entry point"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ace_module_creator.py <groq_api_key> [output_dir]")
        sys.exit(1)
    
    groq_api_key = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    
    result = create_ace_project(groq_api_key, output_dir)
    
    if result['orchestration_status'] == 'success':
        print(f"\n✅ Success! Report: {output_dir}/ace_generation_report.json")
        sys.exit(0)
    else:
        print(f"\n❌ Failed: {result.get('error_message', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()