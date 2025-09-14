#!/usr/bin/env python3
"""
Enhanced ACE Module Creator v2.0 (Orchestrator)
Purpose: Coordinates execution order and manages LLM integration for all modules
Coordination: Manages execution order → Schema → ESQL → XSL → Application Descriptor → Enrichment → Project
LLM Integration: Passes all inputs (PDF, JSON, msgflow, templates) to each module's LLM calls with no fallback logic
NO HARDCODED FALLBACKS - Pure AI-driven generation orchestration
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

try:
    from schema_generator import SchemaGenerator
    from esql_generator import ESQLGenerator
    from xsl_generator import XSLGenerator
    from application_descriptor_generator import ApplicationDescriptorGenerator
    from enrichment_generator import EnrichmentGenerator
    from project_generator import ProjectGenerator
except ImportError as e:
    print(f"❌ Missing generator modules: {e}")
    print("Ensure all generator modules are in the same directory")
    exit(1)

@dataclass
class ACEGenerationInputs:
    """Data class to hold all input files and paths"""
    component_mapping_json_path: str
    msgflow_path: str
    esql_template_path: str
    application_descriptor_template_path: str
    project_template_path: str
    output_dir: str

@dataclass
class ModuleExecutionResult:
    """Data class to hold module execution results"""
    module_name: str
    status: str
    execution_time: float
    llm_analysis_calls: int
    llm_generation_calls: int
    output_files: List[str]
    metadata: Dict[str, Any]
    error_message: Optional[str] = None

class ACEModuleCreatorOrchestrator:
    """
    Enhanced ACE Module Creator Orchestrator
    Coordinates all modules with pure LLM integration and no fallback logic
    """
    
    def __init__(self, groq_api_key: str = None):
        """Initialize orchestrator with API key"""
        self.groq_api_key = groq_api_key
        self.execution_results: List[ModuleExecutionResult] = []
        self.total_llm_calls = 0
        self.start_time = None
        self.end_time = None
        
    def create_ace_project(self, inputs: ACEGenerationInputs) -> Dict[str, Any]:
        """
        Main orchestration method - creates complete ACE project
        Execution Order: Schema → ESQL → XSL → Application Descriptor → Enrichment → Project
        
        Args:
            inputs: ACEGenerationInputs containing all required input files
            
        Returns:
            Dict with complete generation results and metadata
        """

        missing_files = []
        if not os.path.exists(inputs.component_mapping_json_path):
            missing_files.append(f"Component mapping: {inputs.component_mapping_json_path}")
        if not os.path.exists(inputs.msgflow_path):
            missing_files.append(f"MessageFlow: {inputs.msgflow_path}")
        
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")


        print("🎯 Starting Enhanced ACE Module Creator (Orchestrator)")
        print("📋 NO HARDCODED FALLBACKS - Pure AI-driven generation orchestration")
        print("🔄 Execution Order: Schema → ESQL → XSL → Application Descriptor → Enrichment → Project")
        
        self.start_time = datetime.now()
        
        # Validate all inputs before starting
        print("\n🔍 Step 0: Validating all inputs...")
        self._validate_inputs(inputs)
        
        # Create output directory structure
        os.makedirs(inputs.output_dir, exist_ok=True)
        
        try:
            # Execute modules in specified order with LLM integration
            print("\n🚀 Starting coordinated module execution...")
            
            # Step 1: Schema Generation
            schema_result = self._execute_schema_generation(inputs)
            self.execution_results.append(schema_result)
            
            # Step 2: ESQL Generation  
            esql_result = self._execute_esql_generation(inputs)
            self.execution_results.append(esql_result)
            
            # Step 3: XSL Generation
            xsl_result = self._execute_xsl_generation(inputs)
            self.execution_results.append(xsl_result)
            
            # Step 4: Application Descriptor Generation
            app_desc_result = self._execute_application_descriptor_generation(inputs)
            self.execution_results.append(app_desc_result)
            
            # Step 5: Enrichment Generation
            enrichment_result = self._execute_enrichment_generation(inputs)
            self.execution_results.append(enrichment_result)
            
            # Step 6: Project Generation (analyzes all generated components)
            project_result = self._execute_project_generation(inputs)
            self.execution_results.append(project_result)
            
            self.end_time = datetime.now()
            
            # Generate comprehensive results
            return self._generate_orchestration_results(inputs)
            
        except Exception as e:
            self.end_time = datetime.now()
            print(f"\n❌ Orchestration failed: {str(e)}")
            return self._generate_error_results(inputs, str(e))
    
    def _validate_inputs(self, inputs: ACEGenerationInputs) -> None:
        """
        Validate all required input files exist
        
        Note: Currently in Vector DB transition state:
        - Schema Generator (Step 1): Uses Vector DB ✅
        - ESQL Generator (Step 2): Uses Vector DB ✅  
        - XSL Generator (Step 3): Uses Vector DB ✅
        - Application Descriptor (Step 4): Uses Vector DB ✅
        - EEnrichment Generator (Step 5): Uses Vector DB ✅  
        - Project Generator (Step 6): Still uses PDF
        
        PDF validation maintained until all modules converted to Vector DB
        """
        required_files = {
            'Component Mapping JSON': inputs.component_mapping_json_path,
            'MessageFlow': inputs.msgflow_path,
            'ESQL Template': inputs.esql_template_path,
            'Application Descriptor Template': inputs.application_descriptor_template_path,
            'Project Template': inputs.project_template_path
        }
        
        missing_files = []
        for file_type, file_path in required_files.items():
            if not os.path.exists(file_path):
                missing_files.append(f"{file_type}: {file_path}")
        
        if missing_files:
            raise FileNotFoundError(f"Missing required input files:\n" + "\n".join(missing_files))
        
        # Additional Vector DB validation for schema generation
        import streamlit as st
        if not (st.session_state.get('vector_enabled', False) and 
                st.session_state.get('vector_ready', False) and 
                st.session_state.get('vector_pipeline')):
            raise Exception(
                "Vector DB Error: Vector Knowledge Base not ready. "
                "Schema Generator requires Vector DB for business requirement processing. "
                "Please setup Vector Knowledge Base in Agent 1."
            )
        
        print("  ✅ All input files validated successfully")
        print("  🚀 Vector DB validated for schema generation")


    
    def _execute_schema_generation(self, inputs: ACEGenerationInputs) -> ModuleExecutionResult:
        """Execute Schema Generation module with Vector DB integration"""
        print("\n📊 Step 1: Schema Generation")
        print("  🧠 Vector DB Processing: Schema-focused content extraction")
        
        start_time = time.time()
        
        try:
            # ✅ NEW: Vector DB Integration (following Agent 1/2 pattern)
            import streamlit as st
            
            if (st.session_state.get('vector_enabled', False) and 
                st.session_state.get('vector_ready', False) and 
                st.session_state.get('vector_pipeline')):
                
                print("  🚀 Using Vector DB for schema-focused content...")
                
                # Create agent function for schema generation
                def schema_agent_function(focused_content):
                    """Agent function that receives Vector DB focused content"""
                    generator = SchemaGenerator(groq_api_key=self.groq_api_key)
                    return generator.generate_schemas(
                        vector_content=focused_content,  # ← Vector DB content
                        component_mapping_json_path=inputs.component_mapping_json_path,  # ✅ EXISTS
                        output_dir=inputs.output_dir  # ✅ EXISTS
                    )
                
                # Use Vector DB pipeline to get focused content and run agent
                result = st.session_state.vector_pipeline.run_agent_with_vector_search(
                    agent_name="schema_generator",  # ← Vector search for schema content
                    agent_function=schema_agent_function
                )
                
                print("  ✅ Vector DB processing completed!")
                
            else:
                # ❌ Vector DB not available - raise error (no fallback)
                raise Exception("Vector DB not enabled or ready. Please setup Vector Knowledge Base first.")
                
            execution_time = time.time() - start_time
            self.total_llm_calls += result['llm_analysis_calls'] + result['llm_generation_calls']
            
            print(f"  ✅ Schema generation completed in {execution_time:.2f}s")
            print(f"  📊 Generated {result['schemas_generated']} schema files")
            
            return ModuleExecutionResult(
                module_name="Schema Generator",
                status="success",
                execution_time=execution_time,
                llm_analysis_calls=result['llm_analysis_calls'],
                llm_generation_calls=result['llm_generation_calls'],
                output_files=result['schema_files'],
                metadata=result['processing_metadata']
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"  ❌ Schema generation failed: {str(e)}")
            
            return ModuleExecutionResult(
                module_name="Schema Generator",
                status="failed",
                execution_time=execution_time,
                llm_analysis_calls=0,
                llm_generation_calls=0,
                output_files=[],
                metadata={},
                error_message=str(e)
            )
        
    def _execute_esql_generation(self, inputs: ACEGenerationInputs) -> ModuleExecutionResult:
        """Execute ESQL Generation module with Vector DB integration - NO FALLBACKS"""
        print("\n⚡ Step 2: ESQL Generation")
        print("  🧠 Vector DB Processing: ESQL-focused content extraction")
        
        start_time = time.time()
        
        try:
            # ✅ Vector DB Integration - Following exact schema_generator pattern
            import streamlit as st
            
            if (st.session_state.get('vector_enabled', False) and 
                st.session_state.get('vector_ready', False) and 
                st.session_state.get('vector_pipeline')):
                
                print("  🚀 Using Vector DB for ESQL-focused content...")
                print("  🔍 Vector search focus: ESQL transformation logic, business rules, database operations")
                
                # Create agent function for ESQL generation
                def esql_agent_function(focused_content):
                    """Agent function that receives Vector DB focused content for ESQL processing"""
                    from esql_generator import ESQLGenerator
                    
                    generator = ESQLGenerator(groq_api_key=self.groq_api_key)
                    return generator.generate_esql_files(
                        vector_content=focused_content,  # ← Vector DB content instead of PDF
                        esql_template={'path': inputs.esql_template_path},
                        msgflow_content={'path': inputs.msgflow_path}, 
                        json_mappings={'path': inputs.component_mapping_json_path}
                    )
                
                print("  🤖 Running LLM-based ESQL generation with Vector optimization...")
                
                # Use Vector DB pipeline to get focused content and run agent
                result = st.session_state.vector_pipeline.run_agent_with_vector_search(
                    agent_name="esql_generator",  # ← Vector search for ESQL-specific content
                    agent_function=esql_agent_function
                )
                
                print("  ✅ Vector DB processing completed!")
                print(f"  📊 ESQL modules analysis: {result.get('llm_calls_made', 'N/A')} LLM calls")
                print(f"  🎯 Generated modules: {result.get('total_modules', 'N/A')}")
                
            else:
                # ❌ Vector DB not available - raise error (NO FALLBACK)
                error_details = []
                if not st.session_state.get('vector_enabled', False):
                    error_details.append("Vector DB is disabled")
                if not st.session_state.get('vector_ready', False):
                    error_details.append("Vector Knowledge Base not ready")
                if not st.session_state.get('vector_pipeline'):
                    error_details.append("Vector pipeline not initialized")
                
                raise Exception(
                    f"Vector DB Error: {', '.join(error_details)}. "
                    "ESQL Generator requires Vector DB for business requirement processing. "
                    "Please setup Vector Knowledge Base in Agent 1."
                )
                
            execution_time = time.time() - start_time
            
            # Track LLM calls from ESQL generation
            llm_calls_made = result.get('llm_calls_made', 0)
            self.total_llm_calls += llm_calls_made
            
            print(f"  ✅ ESQL generation completed in {execution_time:.2f}s")
            print(f"  🔥 100% LLM-based generation: {result.get('generation_method', 'Unknown')}")
            
            # Prepare output files list
            output_files = []
            generated_modules = result.get('generated_modules', [])
            for module in generated_modules:
                if 'file_path' in module:
                    output_files.append(module['file_path'])
            
            return ModuleExecutionResult(
                module_name="ESQL Generator",
                status="success",
                execution_time=execution_time,
                llm_analysis_calls=llm_calls_made,  # All calls are combined for ESQL
                llm_generation_calls=0,  # Already counted above
                output_files=output_files,
                metadata={
                    'vector_processing': True,
                    'total_modules_generated': result.get('total_modules', 0),
                    'generation_method': result.get('generation_method', '100% LLM Based'),
                    'chunking_used': result.get('chunking_used', True),
                    'token_management': result.get('token_management', 'Active'),
                    'requirements_analysis': result.get('requirements_analysis', {}),
                    'timestamp': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"  ❌ ESQL generation failed: {str(e)}")
            
            return ModuleExecutionResult(
                module_name="ESQL Generator",
                status="failed",
                execution_time=execution_time,
                llm_analysis_calls=0,
                llm_generation_calls=0,
                output_files=[],
                metadata={
                    'vector_processing': False,
                    'error_type': 'Vector DB' if 'Vector DB' in str(e) else 'Processing',
                    'timestamp': datetime.now().isoformat()
                },
                error_message=str(e)
            )
    
    def _execute_xsl_generation(self, inputs: ACEGenerationInputs) -> ModuleExecutionResult:
        """Execute XSL Generation module with Vector DB integration - NO FALLBACKS"""
        print("\n🎨 Step 3: XSL Generation")
        print("  🧠 Vector DB Processing: XSL transformation-focused content extraction")
        
        start_time = time.time()
        
        try:
            # ✅ Vector DB Integration - Following exact schema_generator/esql_generator pattern
            import streamlit as st
            
            if (st.session_state.get('vector_enabled', False) and 
                st.session_state.get('vector_ready', False) and 
                st.session_state.get('vector_pipeline')):
                
                print("  🚀 Using Vector DB for XSL transformation-focused content...")
                print("  🔍 Vector search focus: XSL mappings, field transformations, stylesheet patterns")
                
                # Create agent function for XSL generation
                def xsl_agent_function(focused_content):
                    """Agent function that receives Vector DB focused content for XSL processing"""
                    from xsl_generator import XSLGenerator
                    
                    generator = XSLGenerator(groq_api_key=self.groq_api_key)
                    return generator.generate_xsl_transformations(
                        vector_content=focused_content,  # ← Vector DB content instead of PDF
                        component_mapping_json_path=inputs.component_mapping_json_path,
                        output_dir=inputs.output_dir
                    )
                
                print("  🤖 Running LLM-based XSL generation with Vector optimization...")
                
                # Use Vector DB pipeline to get focused content and run agent
                result = st.session_state.vector_pipeline.run_agent_with_vector_search(
                    agent_name="xsl_generator",  # ← Vector search for XSL-specific content
                    agent_function=xsl_agent_function
                )
                
                print("  ✅ Vector DB processing completed!")
                print(f"  📊 XSL transformations: {result.get('xsl_transformations_generated', 'N/A')}")
                print(f"  🧠 LLM calls: {result.get('llm_analysis_calls', 0)} + {result.get('llm_generation_calls', 0)}")
                
            else:
                # ❌ Vector DB not available - raise error (NO FALLBACK)
                error_details = []
                if not st.session_state.get('vector_enabled', False):
                    error_details.append("Vector DB is disabled")
                if not st.session_state.get('vector_ready', False):
                    error_details.append("Vector Knowledge Base not ready")
                if not st.session_state.get('vector_pipeline'):
                    error_details.append("Vector pipeline not initialized")
                
                raise Exception(
                    f"Vector DB Error: {', '.join(error_details)}. "
                    "XSL Generator requires Vector DB for business requirement processing. "
                    "Please setup Vector Knowledge Base in Agent 1."
                )
                
            execution_time = time.time() - start_time
            self.total_llm_calls += result['llm_analysis_calls'] + result['llm_generation_calls']
            
            print(f"  ✅ XSL generation completed in {execution_time:.2f}s")
            print(f"  📊 Generated {result['xsl_transformations_generated']} XSL transformation files")
            
            return ModuleExecutionResult(
                module_name="XSL Generator",
                status="success",
                execution_time=execution_time,
                llm_analysis_calls=result['llm_analysis_calls'],
                llm_generation_calls=result['llm_generation_calls'],
                output_files=result['xsl_files'],
                metadata=result['processing_metadata']
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"  ❌ XSL generation failed: {str(e)}")
            
            return ModuleExecutionResult(
                module_name="XSL Generator",
                status="failed",
                execution_time=execution_time,
                llm_analysis_calls=0,
                llm_generation_calls=0,
                output_files=[],
                metadata={},
                error_message=str(e)
            )
        
    
    def _execute_application_descriptor_generation(self, inputs: ACEGenerationInputs) -> ModuleExecutionResult:
        """Execute Application Descriptor Generation module with Vector DB integration"""
        print("\n📋 Step 4: Application Descriptor Generation")
        print("  🧠 Vector DB Processing: Library dependencies, configuration settings, shared libraries")
        
        start_time = time.time()
        
        try:
            # ✅ Vector DB Integration (following same pattern as other modules)
            import streamlit as st
            
            if (st.session_state.get('vector_enabled', False) and 
                st.session_state.get('vector_ready', False) and 
                st.session_state.get('vector_pipeline')):
                
                print("  🚀 Using Vector DB for application descriptor-focused content...")
                print("  🔍 Vector search focus: Library dependencies, configuration settings, shared libraries")
                print("  📋 DSV template: Account-specific standards and compliance structure")
                
                # Create agent function for application descriptor generation
                def application_descriptor_agent_function(focused_content):
                    """Agent function that receives Vector DB focused content for application descriptor processing"""
                    from application_descriptor_generator import ApplicationDescriptorGenerator
                    
                    generator = ApplicationDescriptorGenerator(groq_api_key=self.groq_api_key)
                    return generator.generate_application_descriptor(
                        vector_content=focused_content,  # ← Vector DB content from pipeline
                        template_path=inputs.application_descriptor_template_path,  # ← DSV template
                        component_mapping_json_path=inputs.component_mapping_json_path,  # ← Component mappings
                        output_dir=inputs.output_dir  # ← Output directory
                    )
                
                print("  🤖 Running LLM-based application descriptor generation with Vector optimization...")
                
                # Use Vector DB pipeline to get focused content and run agent
                result = st.session_state.vector_pipeline.run_agent_with_vector_search(
                    agent_name="application_descriptor_generator",  # ← Vector search for application descriptor-specific content
                    agent_function=application_descriptor_agent_function
                )
                
                print("  ✅ Vector DB processing completed!")
                print(f"  📊 Application descriptor: {result.get('application_descriptor_generated', 'N/A')}")
                print(f"  🧠 LLM calls: {result.get('llm_analysis_calls', 0)} + {result.get('llm_generation_calls', 0)}")
                
            else:
                # ❌ Vector DB not available - raise error (NO FALLBACK)
                error_details = []
                if not st.session_state.get('vector_enabled', False):
                    error_details.append("Vector DB is disabled")
                if not st.session_state.get('vector_ready', False):
                    error_details.append("Vector Knowledge Base not ready")
                if not st.session_state.get('vector_pipeline'):
                    error_details.append("Vector pipeline not initialized")
                
                raise Exception(
                    f"Vector DB Error: {', '.join(error_details)}. "
                    "Application Descriptor Generator requires Vector DB for business requirement processing. "
                    "Please setup Vector Knowledge Base in Agent 1."
                )
                
            execution_time = time.time() - start_time
            self.total_llm_calls += result['llm_analysis_calls'] + result['llm_generation_calls']
            
            print(f"  ✅ Application descriptor generation completed in {execution_time:.2f}s")
            print(f"  📊 Generated application.descriptor file")
            print(f"  🧠 LLM calls: {result['llm_analysis_calls']} analysis + {result['llm_generation_calls']} generation")
            
            return ModuleExecutionResult(
                module_name="Application Descriptor Generator",
                status="success",
                execution_time=execution_time,
                llm_analysis_calls=result['llm_analysis_calls'],
                llm_generation_calls=result['llm_generation_calls'],
                output_files=[result['descriptor_file']],
                metadata=result['processing_metadata']
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"  ❌ Application descriptor generation failed: {str(e)}")
            
            return ModuleExecutionResult(
                module_name="Application Descriptor Generator",
                status="failed",
                execution_time=execution_time,
                llm_analysis_calls=0,
                llm_generation_calls=0,
                output_files=[],
                metadata={},
                error_message=str(e)
            )
        

    
    def _execute_enrichment_generation(self, inputs: ACEGenerationInputs) -> ModuleExecutionResult:
        """Execute Enrichment Generation module with Vector DB integration - NO FALLBACKS"""
        print("\n🔋 Step 5: Enrichment Configuration Generation")
        print("  🧠 Vector DB Processing: CW1 enrichment-focused content extraction")
        
        start_time = time.time()
        
        try:
            # ✅ Vector DB Integration - Following exact esql_generator pattern
            import streamlit as st
            
            if (st.session_state.get('vector_enabled', False) and 
                st.session_state.get('vector_ready', False) and 
                st.session_state.get('vector_pipeline')):
                
                print("  🚀 Using Vector DB for CW1 enrichment-focused content...")
                print("  🔍 Vector search focus: CW1 database operations, enrichment logic, CargoWise One integration")
                
                # Create agent function for enrichment generation
                def enrichment_agent_function(focused_content):
                    """Agent function that receives Vector DB focused content for CW1 enrichment processing"""
                    from enrichment_generator import EnrichmentGenerator
                    
                    generator = EnrichmentGenerator(groq_api_key=self.groq_api_key)
                    return generator.generate_enrichment_files(
                        vector_content=focused_content,  # ← Vector DB content instead of PDF
                        component_mapping_json_path=inputs.component_mapping_json_path,
                        msgflow_path=inputs.msgflow_path,
                        output_dir=inputs.output_dir
                    )
                
                print("  🤖 Running LLM-based CW1 enrichment generation with Vector optimization...")
                
                # Use Vector DB pipeline to get focused content and run agent
                result = st.session_state.vector_pipeline.run_agent_with_vector_search(
                    agent_name="enrichment_generator",  # ← Vector search for enrichment-specific content
                    agent_function=enrichment_agent_function
                )
                
                print("  ✅ Vector DB processing completed!")
                print(f"  📊 CW1 enrichment analysis: {result.get('llm_analysis_calls', 'N/A')} LLM calls")
                print(f"  🎯 Generated configurations: {result.get('enrichment_configs_generated', 'N/A')}")
                
            else:
                # ❌ Vector DB not available - raise error (NO FALLBACK)
                error_details = []
                if not st.session_state.get('vector_enabled', False):
                    error_details.append("Vector DB is disabled")
                if not st.session_state.get('vector_ready', False):
                    error_details.append("Vector Knowledge Base not ready")
                if not st.session_state.get('vector_pipeline'):
                    error_details.append("Vector pipeline not initialized")
                
                raise Exception(
                    f"Vector DB Error: {', '.join(error_details)}. "
                    "Enrichment Generator requires Vector DB for business requirement processing. "
                    "Please setup Vector Knowledge Base in Agent 1."
                )
                
            execution_time = time.time() - start_time
            
            # Track LLM calls from enrichment generation
            llm_analysis_calls = result.get('llm_analysis_calls', 0)
            llm_generation_calls = result.get('llm_generation_calls', 0)
            self.total_llm_calls += llm_analysis_calls + llm_generation_calls
            
            print(f"  ✅ CW1 enrichment generation completed in {execution_time:.2f}s")
            print(f"  🔥 100% LLM-based generation with Vector DB optimization")
            print(f"  📊 Generated {result.get('enrichment_configs_generated', 0)} enrichment configuration files")
            print(f"  🧠 LLM calls: {llm_analysis_calls} analysis + {llm_generation_calls} generation")
            
            return ModuleExecutionResult(
                module_name="Enrichment Generator",
                status="success",
                execution_time=execution_time,
                llm_analysis_calls=llm_analysis_calls,
                llm_generation_calls=llm_generation_calls,
                output_files=result.get('config_files', []),
                metadata={
                    'vector_processing': True,
                    'enrichment_configs_generated': result.get('enrichment_configs_generated', 0),
                    'generation_method': '100% LLM Based with Vector DB',
                    'cw1_database_operations': 6,  # CompanyCode, SSN, HouseBill, IsPublished, BrokerageId, RecipientId
                    'database_aliases': ['MH.ESB.EDIEnterprise', 'DSV.ESB.Integration'],
                    'processing_flow': 'CW1.IN.DOCUMENT.SND.QL → Enrichment → UniversalEvent',
                    'timestamp': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"  ❌ CW1 enrichment generation failed: {str(e)}")
            
            return ModuleExecutionResult(
                module_name="Enrichment Generator",
                status="failed",
                execution_time=execution_time,
                llm_analysis_calls=0,
                llm_generation_calls=0,
                output_files=[],
                metadata={},
                error_message=str(e)
            )
    
    def _execute_project_generation(self, inputs: ACEGenerationInputs) -> ModuleExecutionResult:
        """Execute Project Generation module with Vector DB integration"""
        print("\n🗂️ Step 6: Project File Generation")
        print("  🧠 Vector DB Processing: Project architecture and build configuration analysis")
        
        start_time = time.time()
        
        try:
            # ✅ NEW: Vector DB Integration (following established pattern)
            import streamlit as st
            
            if (st.session_state.get('vector_enabled', False) and 
                st.session_state.get('vector_ready', False) and 
                st.session_state.get('vector_pipeline')):
                
                print("  🚀 Using Vector DB for project architecture analysis...")
                
                # Create agent function that receives Vector DB focused content for project generation
                def project_generator_agent_function(focused_content):
                    """Agent function that receives Vector DB focused content for project processing"""
                    
                    from project_generator import ProjectGenerator                    
                    generator = ProjectGenerator(groq_api_key=self.groq_api_key)
                    return generator.generate_project_file(
                        vector_content=focused_content,  # ✅ Vector DB content instead of PDF
                        template_path=inputs.project_template_path,
                        component_mapping_json_path=inputs.component_mapping_json_path,
                        output_dir=inputs.output_dir,
                        biztalk_folder=st.session_state.get('biztalk_folder'),
                        generated_components_dir=inputs.output_dir  # Analyze all generated components
                    )
                
                print("  🤖 Running LLM-based project generation with Vector optimization...")
                
                # Use Vector DB pipeline to get focused content and run agent
                result = st.session_state.vector_pipeline.run_agent_with_vector_search(
                    agent_name="project_generator",  # ✅ Vector search for project architecture content
                    agent_function=project_generator_agent_function
                )
                
                print("  ✅ Vector DB processing completed!")
                print(f"  📊 Project file: {result.get('project_file_generated', 'N/A')}")
                print(f"  🧠 LLM calls: {result.get('llm_analysis_calls', 0)} + {result.get('llm_generation_calls', 0)}")
                
            else:
                # ❌ Vector DB not available - raise error (NO FALLBACK)
                error_details = []
                if not st.session_state.get('vector_enabled', False):
                    error_details.append("Vector DB is disabled")
                if not st.session_state.get('vector_ready', False):
                    error_details.append("Vector Knowledge Base not ready")
                if not st.session_state.get('vector_pipeline'):
                    error_details.append("Vector pipeline not initialized")
                
                raise Exception(
                    f"Vector DB Error: {', '.join(error_details)}. "
                    "Project Generator requires Vector DB for business requirement processing. "
                    "Please setup Vector Knowledge Base in Agent 1."
                )
            
            execution_time = time.time() - start_time
            self.total_llm_calls += result['llm_analysis_calls'] + result['llm_generation_calls']
            
            print(f"  ✅ Project generation completed in {execution_time:.2f}s")
            print(f"  📊 Generated .project file for IBM ACE Toolkit")
            print(f"  🧠 LLM calls: {result['llm_analysis_calls']} analysis + {result['llm_generation_calls']} generation")
            
            return ModuleExecutionResult(
                module_name="Project Generator",
                status="success",
                execution_time=execution_time,
                llm_analysis_calls=result['llm_analysis_calls'],
                llm_generation_calls=result['llm_generation_calls'],
                output_files=[result['project_file_path']],
                metadata=result.get('processing_summary', {})
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"  ❌ Project generation failed: {str(e)}")
            
            return ModuleExecutionResult(
                module_name="Project Generator",
                status="failed",
                execution_time=execution_time,
                llm_analysis_calls=0,
                llm_generation_calls=0,
                output_files=[],
                metadata={},
                error_message=str(e)
            )
    

    
    def _generate_orchestration_results(self, inputs: ACEGenerationInputs) -> Dict[str, Any]:
        """Generate comprehensive orchestration results"""
        total_execution_time = (self.end_time - self.start_time).total_seconds()
        
        # Aggregate results
        total_files_generated = sum(len(result.output_files) for result in self.execution_results)
        successful_modules = [r for r in self.execution_results if r.status == "success"]
        failed_modules = [r for r in self.execution_results if r.status == "failed"]
        
        total_analysis_calls = sum(r.llm_analysis_calls for r in self.execution_results)
        total_generation_calls = sum(r.llm_generation_calls for r in self.execution_results)
        
        # Create final project structure summary
        project_structure = self._analyze_generated_project_structure(inputs.output_dir)
        
        print(f"\n🎉 ACE Module Creator Orchestration Complete!")
        print(f"⏱️ Total execution time: {total_execution_time:.2f} seconds")
        print(f"✅ Successful modules: {len(successful_modules)}/6")
        print(f"📊 Total files generated: {total_files_generated}")
        print(f"🧠 Total LLM calls: {total_analysis_calls} analysis + {total_generation_calls} generation = {self.total_llm_calls}")
        
        if failed_modules:
            print(f"❌ Failed modules: {len(failed_modules)}")
            for failed in failed_modules:
                print(f"  - {failed.module_name}: {failed.error_message}")
        
        return {
            'orchestration_status': 'success' if len(failed_modules) == 0 else 'partial_success',
            'total_execution_time': total_execution_time,
            'successful_modules': len(successful_modules),
            'failed_modules': len(failed_modules),
            'total_files_generated': total_files_generated,
            'total_llm_analysis_calls': total_analysis_calls,
            'total_llm_generation_calls': total_generation_calls,
            'total_llm_calls': self.total_llm_calls,
            'module_results': [
                {
                    'module_name': r.module_name,
                    'status': r.status,
                    'execution_time': r.execution_time,
                    'llm_analysis_calls': r.llm_analysis_calls,
                    'llm_generation_calls': r.llm_generation_calls,
                    'output_files_count': len(r.output_files),
                    'output_files': r.output_files,
                    'error_message': r.error_message
                }
                for r in self.execution_results
            ],
            'project_structure': project_structure,
            'output_directory': inputs.output_dir,
            'ace_project_ready': len(failed_modules) == 0,
            'generation_timestamp': self.end_time.isoformat()
        }
    
    def _generate_error_results(self, inputs: ACEGenerationInputs, error_message: str) -> Dict[str, Any]:
        """Generate error results for failed orchestration"""
        total_execution_time = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        
        return {
            'orchestration_status': 'failed',
            'error_message': error_message,
            'total_execution_time': total_execution_time,
            'successful_modules': len([r for r in self.execution_results if r.status == "success"]),
            'failed_modules': len([r for r in self.execution_results if r.status == "failed"]),
            'total_llm_calls': self.total_llm_calls,
            'module_results': [
                {
                    'module_name': r.module_name,
                    'status': r.status,
                    'execution_time': r.execution_time,
                    'error_message': r.error_message
                }
                for r in self.execution_results
            ],
            'output_directory': inputs.output_dir,
            'ace_project_ready': False,
            'generation_timestamp': self.end_time.isoformat() if self.end_time else datetime.now().isoformat()
        }
    
    def _analyze_generated_project_structure(self, output_dir: str) -> Dict[str, Any]:
        """Analyze the final generated project structure"""
        structure = {
            'directories': [],
            'files_by_type': {},
            'total_files': 0
        }
        
        try:
            for root, dirs, files in os.walk(output_dir):
                rel_root = os.path.relpath(root, output_dir)
                if rel_root != '.':
                    structure['directories'].append(rel_root)
                
                for file in files:
                    ext = Path(file).suffix.lower()
                    if ext not in structure['files_by_type']:
                        structure['files_by_type'][ext] = []
                    
                    rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                    structure['files_by_type'][ext].append(rel_path)
                    structure['total_files'] += 1
        
        except Exception as e:
            structure['error'] = f"Failed to analyze project structure: {str(e)}"
        
        return structure


def main():
    """Main entry point for ACE Module Creator Orchestrator with Vector DB Integration"""
    
    # Import Vector DB pipeline (following project pattern)
    try:
        from vector_knowledge.pipeline_integration import VectorOptimizedPipeline
    except ImportError as e:
        print(f"❌ Vector DB modules not available: {e}")
        return
    
    # Create ACE generation inputs with Vector DB approach
    inputs = ACEGenerationInputs(
        confluence_pdf_path="vector_database",  # Placeholder - Vector DB provides content
        component_mapping_json_path="component_mapping.json",  
        msgflow_path="sample.msgflow",
        esql_template_path="ESQL_Template_Updated.ESQL",
        application_descriptor_template_path="application_descriptor.xml",
        project_template_path="project.xml",
        output_dir="generated_ace_project"
    )
    
    # Create output directory
    os.makedirs(inputs.output_dir, exist_ok=True)
    
    # Get GROQ API key
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        print("❌ GROQ_API_KEY environment variable not set")
        return
    
    # Initialize orchestrator
    orchestrator = ACEModuleCreatorOrchestrator(groq_api_key=groq_api_key)
    results = orchestrator.create_ace_project(inputs)
    
    # Save results to file
    with open(os.path.join(inputs.output_dir, 'generation_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"📊 Results saved to: {inputs.output_dir}/generation_results.json")


if __name__ == "__main__":
    main()