import PyPDF2
import re
from typing import List, Dict, Tuple
from groq import Groq
from dotenv import load_dotenv
import os
from llm_json_parser import LLMJSONParser
from datetime import datetime
import streamlit as st
load_dotenv()

class PDFProcessingError(Exception):
    """Custom exception for PDF processing failures - explicit error reporting"""
    pass



class PDFProcessor:

    def __init__(self):
        # ✅ Keep existing field names for backward compatibility - NO LIMITS APPLIED
        self.chunk_size = None     # No limit - LLM decides all sizes
        self.overlap = None        # No limit - LLM decides overlap strategy
        
        # 🆕 LLM-powered settings - ZERO ARTIFICIAL LIMITS
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        if not self.groq_api_key:
            raise PDFProcessingError("GROQ_API_KEY must be set in environment variables")
        
        # Initialize Groq client for unlimited processing
        self._groq_client = Groq(api_key=self.groq_api_key)
        self.groq_model = os.getenv('GROQ_MODEL', 'llama-3.1-70b-versatile')  # Use larger model
        
        # Enhanced processing settings - NO SIZE RESTRICTIONS
        self._unlimited_processing = True    # Process ANY document size
        self._preserve_technical_specs = True  # Keep technical content together
        self._llm_full_control = True          # LLM decides everything
        
        # Processing tracking
        self._llm_analysis_calls = 0
        self._total_chunks_created = 0
        
        print("🚀 PDFProcessor initialized with LLM-powered chunking")
        print(f"📊 Model: {self.groq_model}")
        print("📏 Chunk sizes: UNLIMITED - LLM decides all boundaries")
        print("🎯 Processing: NO SIZE RESTRICTIONS - Full paid subscription capacity")



    def _analyze_document_with_llm(self, text: str) -> Dict:
        """
        Analyze entire document structure with LLM reasoning power
        Uses existing llm_json_parser.py for response parsing
        
        Args:
            text: Complete extracted PDF text (no size limits)
            
        Returns:
            Dict: Comprehensive document structure analysis
        """
        try:
            print(f"📋 Analyzing document structure with LLM...")
            print(f"📏 Document length: {len(text)} characters")
        
            json_parser = LLMJSONParser()
            
            # Direct comprehensive analysis prompt
            analysis_prompt = f"""
            RETURN ONLY VALID JSON. NO OTHER TEXT.
    ANALYZE THIS COMPLETE TECHNICAL DOCUMENT FOR INTELLIGENT CHUNKING:

    DOCUMENT CONTENT:
    {text}

    ANALYSIS REQUIREMENTS:
    1. DOCUMENT STRUCTURE: Identify all major sections, headers, technical specifications
    2. TECHNICAL COMPONENTS: Find database procedures, URLs, queue names, configuration parameters  
    3. CONTENT BOUNDARIES: Determine where technical specifications should stay together
    4. CHUNKING STRATEGY: Recommend how to split this for optimal AI agent processing

    RESPONSE FORMAT: JSON with structure analysis and chunking recommendations.

    JSON SCHEMA:
    {{
        "sections": ["list of major sections found"],
        "technical_components": ["database procedures", "URLs", "queue names", "config params"],
        "chunking_recommendations": [
            {{
                "section": "section name",
                "start_marker": "text pattern to identify start",
                "end_marker": "text pattern to identify end", 
                "keep_together": true/false,
                "reason": "why this chunking approach"
            }}
        ],
        "total_recommended_chunks": number
    }}
    RETURN ONLY VALID JSON. NO EXPLANATIONS OR OTHER TEXT.
    """
            
            # Send to LLM for analysis
            response = self._groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert technical document analyzer. Analyze the complete document structure and provide JSON recommendations for intelligent chunking that preserves technical specifications."
                    },
                    {
                        "role": "user", 
                        "content": analysis_prompt
                    }
                ],
                temperature=0.1,
                max_tokens=4000
            )

            if 'token_tracker' in st.session_state and hasattr(response, 'usage') and response.usage:
                st.session_state.token_tracker.manual_track(
                    agent="pdf_processor",
                    operation="document_analysis",
                    model=self.groq_model,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    flow_name="pdf_chunking"
                )
                        
            # Parse response using existing JSON parser
            parse_result = json_parser.parse(response.choices[0].message.content)
            if not parse_result.success:
                raise PDFProcessingError(f"Document analysis parsing failed: {parse_result.error_message}")
            analysis_result = parse_result.data
            
            print(f"✅ Document analysis completed")
            print(f"📊 Recommended chunks: {analysis_result.get('total_recommended_chunks', 'Unknown')}")
            
            return analysis_result
            
        except Exception as e:
            raise PDFProcessingError(f"LLM document analysis failed: {str(e)}")
        


    def _create_llm_chunk_plan(self, text: str, structure: Dict) -> List[Dict]:
        """
        Create intelligent chunking plan based on document structure analysis
        
        Args:
            text: Complete document text
            structure: Document structure analysis from _analyze_document_with_llm
            
        Returns:
            List[Dict]: Detailed chunking plan with boundaries and metadata
        """
        try:
            print(f"🧩 Creating LLM-powered chunk plan...")
            
            # Import existing JSON parser
            from llm_json_parser import LLMJSONParser
            json_parser = LLMJSONParser()
            
            # Direct chunking plan prompt with structure context
            chunk_plan_prompt = f"""
    CREATE INTELLIGENT CHUNKING PLAN FOR THIS TECHNICAL DOCUMENT:

    DOCUMENT STRUCTURE ANALYSIS:
    {structure}

    DOCUMENT TEXT:
    {text}

    CHUNKING REQUIREMENTS:
    1. PRESERVE TECHNICAL SPECIFICATIONS: Keep database procedures, URLs, queue names as complete units
    2. RESPECT LOGICAL BOUNDARIES: Split at natural section breaks, not mid-specification
    3. MAXIMIZE CONTEXT: Create chunks large enough for downstream AI agents to understand
    4. IDENTIFY BOUNDARIES: Specify exact text patterns where to split

    RESPONSE FORMAT: JSON array of chunk specifications with exact boundaries.

    JSON SCHEMA:
    [
        {{
            "chunk_id": 1,
            "start_text": "exact text pattern where chunk starts",
            "end_text": "exact text pattern where chunk ends", 
            "section_name": "section this chunk belongs to",
            "content_type": "header|process_diagram|technical_spec|mapping_logic|description",
            "preserve_complete": true/false,
            "estimated_length": approximate_character_count,
            "technical_components": ["list of technical elements in this chunk"],
            "reasoning": "why this chunking boundary makes sense"
        }}
    ]

    ANALYZE THE TEXT AND CREATE SPECIFIC CHUNK BOUNDARIES THAT PRESERVE TECHNICAL INTEGRITY.
    """
            
            # Send to LLM for chunk planning
            response = self._groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating intelligent document chunking plans. Analyze the text and structure to create specific chunk boundaries that preserve technical specifications and logical flow."
                    },
                    {
                        "role": "user",
                        "content": chunk_plan_prompt
                    }
                ],
                temperature=0.1,
                max_tokens=8000
            )

            if 'token_tracker' in st.session_state and hasattr(response, 'usage') and response.usage:
                st.session_state.token_tracker.manual_track(
                    agent="pdf_processor",
                    operation="document_analysis",
                    model=self.groq_model,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    flow_name="pdf_chunking"
                )
                
            
            # Parse chunking plan using existing JSON parser
            parse_result = json_parser.parse(response.choices[0].message.content)
            if not parse_result.success:
                raise PDFProcessingError(f"Chunk plan parsing failed: {parse_result.error_message}")
            chunk_plan = parse_result.data
            
            # Ensure chunk_plan is a list
            if isinstance(chunk_plan, dict):
                chunk_plan = chunk_plan.get('chunks', [chunk_plan])
            
            print(f"✅ Chunk plan created: {len(chunk_plan)} intelligent chunks planned")
            
            # Log technical components found
            total_tech_components = sum(len(chunk.get('technical_components', [])) for chunk in chunk_plan)
            print(f"🔧 Total technical components to preserve: {total_tech_components}")
            
            return chunk_plan
            
        except Exception as e:
            raise PDFProcessingError(f"LLM chunking plan creation failed: {str(e)}")
        


    
    def _execute_content_aware_chunking(self, text: str, chunk_plan: List[Dict]) -> List[Dict]:
        """
        Execute intelligent chunking plan to create final enhanced chunks
        
        Args:
            text: Complete document text
            chunk_plan: Chunking boundaries and metadata from _create_llm_chunk_plan
            
        Returns:
            List[Dict]: Final chunks in existing format for backward compatibility
        """
        try:
            print(f"✂️ Executing content-aware chunking...")
            print(f"📋 Processing {len(chunk_plan)} planned chunks")
            
            final_chunks = []
            text_length = len(text)
            
            for i, plan in enumerate(chunk_plan):
                chunk_id = plan.get('chunk_id', i + 1)
                
                # Find start boundary in text
                start_text = plan.get('start_text', '')
                end_text = plan.get('end_text', '')
                
                if start_text:
                    start_pos = text.find(start_text)
                    if start_pos == -1:
                        # Try partial match if exact not found
                        start_pos = text.find(start_text[:50])  # First 50 chars
                        if start_pos == -1:
                            start_pos = 0 if i == 0 else final_chunks[-1]['metadata']['end_position']
                else:
                    start_pos = 0 if i == 0 else final_chunks[-1]['metadata']['end_position']
                
                # Find end boundary in text
                if end_text and i < len(chunk_plan) - 1:  # Not the last chunk
                    end_pos = text.find(end_text, start_pos)
                    if end_pos == -1:
                        # Try partial match if exact not found
                        end_pos = text.find(end_text[:50], start_pos)
                        if end_pos == -1:
                            # Use next chunk's start as boundary
                            next_start = chunk_plan[i + 1].get('start_text', '')
                            if next_start:
                                end_pos = text.find(next_start, start_pos)
                                if end_pos == -1:
                                    end_pos = start_pos + plan.get('estimated_length', 5000)
                            else:
                                end_pos = start_pos + plan.get('estimated_length', 5000)
                    else:
                        end_pos = end_pos + len(end_text)  # Include end marker
                else:
                    # Last chunk - take everything remaining
                    end_pos = text_length
                
                # Ensure boundaries are valid
                if end_pos <= start_pos:
                    end_pos = min(start_pos + 1000, text_length)  # Minimum 1000 chars
                
                # Extract chunk content
                chunk_content = text[start_pos:end_pos].strip()
                
                # Skip empty chunks
                if not chunk_content:
                    continue
                
                # Create chunk in existing format
                chunk = {
                    'content': chunk_content,
                    'section': plan.get('section_name', f'Section_{chunk_id}'),
                    'chunk_id': len(final_chunks),  # Sequential numbering
                    'metadata': {
                        # ✅ Enhanced metadata while keeping compatibility
                        'content_type': plan.get('content_type', 'general'),
                        'technical_components': plan.get('technical_components', []),
                        'preserve_complete': plan.get('preserve_complete', False),
                        'chunking_reasoning': plan.get('reasoning', ''),
                        'start_position': start_pos,
                        'end_position': end_pos,
                        'chunk_length': len(chunk_content),
                        'llm_generated': True
                    }
                }
                
                final_chunks.append(chunk)
                final_chunks = self._serialize_all_chunks_metadata(final_chunks)
                # Log progress for important technical chunks
                tech_components = plan.get('technical_components', [])
                if tech_components:
                    print(f"🔧 Chunk {len(final_chunks)}: Preserved {len(tech_components)} technical components")
            
            print(f"✅ Content-aware chunking completed")
            print(f"📊 Created {len(final_chunks)} intelligent chunks")
            print(f"📏 Average chunk size: {sum(len(c['content']) for c in final_chunks) // len(final_chunks) if final_chunks else 0} characters")
            
            # Track total chunks created
            self._total_chunks_created = len(final_chunks)
            
            return self._serialize_all_chunks_metadata(final_chunks)
            
        except Exception as e:
            raise PDFProcessingError(f"Content-aware chunking execution failed: {str(e)}")



    def _serialize_all_chunks_metadata(self, chunks: List[Dict]) -> List[Dict]:
        """
        Smart, efficient serialization of complex metadata to ChromaDB-compatible primitives.
        Converts lists, dicts, and nested structures to strings while preserving searchability.
        
        Args:
            chunks: List of chunks with potentially complex metadata
            
        Returns:
            List of chunks with primitive-only metadata (str, int, float, bool, None)
        """
        import json
        
        for chunk in chunks:
            if 'metadata' not in chunk:
                continue
                
            metadata = chunk['metadata']
            serialized_metadata = {}
            
            for key, value in metadata.items():
                # Handle None values
                if value is None:
                    serialized_metadata[key] = None
                    
                # Handle primitive types (pass through unchanged)
                elif isinstance(value, (str, int, float, bool)):
                    serialized_metadata[key] = value
                    
                # Handle lists - convert to searchable string format
                elif isinstance(value, list):
                    if not value:  # Empty list
                        serialized_metadata[key] = ""
                    elif all(isinstance(item, (str, int, float)) for item in value):
                        # Simple list of primitives - comma-separated
                        serialized_metadata[key] = ",".join(str(item) for item in value)
                    else:
                        # Complex list - JSON string
                        serialized_metadata[key] = json.dumps(value)
                        
                # Handle dictionaries - convert to JSON string
                elif isinstance(value, dict):
                    if not value:  # Empty dict
                        serialized_metadata[key] = ""
                    else:
                        serialized_metadata[key] = json.dumps(value)
                        
                # Handle other iterables (sets, tuples) - convert to list then string
                elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                    try:
                        list_value = list(value)
                        if all(isinstance(item, (str, int, float)) for item in list_value):
                            serialized_metadata[key] = ",".join(str(item) for item in list_value)
                        else:
                            serialized_metadata[key] = json.dumps(list_value)
                    except (TypeError, ValueError):
                        serialized_metadata[key] = str(value)
                        
                # Handle any other type - convert to string
                else:
                    serialized_metadata[key] = str(value)
            
            # Replace original metadata with serialized version
            chunk['metadata'] = serialized_metadata
        
        return chunks



    def _get_timestamp(self) -> str:
        """Simple timestamp for metadata"""
        from datetime import datetime
        return datetime.now().isoformat()


    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    

    
    def intelligent_chunking(self, text: str) -> List[Dict]:
        """
        ✅ SAME signature - Enhanced LLM-powered internal logic 
        Split text into meaningful chunks using LLM analysis and reasoning
        
        Args:
            text: Complete document text (no size limits)
            
        Returns:
            List[Dict]: Enhanced chunks in existing format for backward compatibility
        """
        try:
            print(f"🚀 Starting LLM-powered intelligent chunking...")
            print(f"📏 Processing document: {len(text)} characters")
            
            # Validate input
            if not text or len(text.strip()) < 100:
                raise PDFProcessingError("Insufficient text content for processing")
            
            # 🆕 Step 1: LLM analyzes complete document structure
            print("📋 Step 1: Analyzing document structure...")
            structure_analysis = self._analyze_document_with_llm(text)
            
            # 🆕 Step 2: LLM creates intelligent chunk plan
            print("🧩 Step 2: Creating intelligent chunk plan...")
            chunk_plan = self._create_llm_chunk_plan(text, structure_analysis)
            
            # 🆕 Step 3: Execute content-aware chunking
            print("✂️ Step 3: Executing content-aware chunking...")
            enhanced_chunks = self._execute_content_aware_chunking(text, chunk_plan)
            
            # Validate results
            if not enhanced_chunks:
                raise PDFProcessingError("No chunks created - chunking logic failed")
            
            print(f"✅ LLM-powered chunking completed successfully!")
            print(f"📊 Total chunks created: {len(enhanced_chunks)}")
            
            # ✅ Return same format - existing modules see no difference
            return enhanced_chunks
            
        except Exception as e:
            # 🚨 EXPLICIT FAILURE - No silent fallback
            raise PDFProcessingError(f"Intelligent chunking failed: {str(e)}")
    


    
    def _split_by_sections(self, text: str) -> List[Tuple[str, str]]:
        """Split by document sections"""
        # Common section patterns in technical docs
        section_patterns = [
            r'(?i)^(Process Diagram|Functional Description|Technical Description|Requirements?|Implementation|Architecture|Component|System)',
            r'(?i)^(Mapping Logic|Data Flow|Message Flow|Schema|Transformation)',
            r'(?i)^(\d+\.|\•|\-)\s+',  # Numbered or bulleted lists
            r'\n\n+'  # Paragraph breaks
        ]
        
        sections = []
        current_section = "Introduction"
        current_content = ""
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a new section
            is_new_section = any(re.match(pattern, line) for pattern in section_patterns[:2])
            
            if is_new_section and current_content:
                sections.append((current_section, current_content))
                current_section = line[:50]  # Use first 50 chars as section name
                current_content = line + "\n"
            else:
                current_content += line + "\n"
        
        # Add final section
        if current_content:
            sections.append((current_section, current_content))
            
        return sections
    
    def _split_text_with_overlap(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text with overlapping windows"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk = ' '.join(chunk_words)
            if chunk.strip():
                chunks.append(chunk)
                
        return chunks
    
    def _extract_metadata(self, chunk: str, section: str) -> Dict:
        """Extract metadata for better search relevance"""
        chunk_lower = chunk.lower()
        
        # Get agent scores
        agent_scores = self._calculate_agent_scores(chunk_lower)
        
        # Build metadata with flattened agent scores
        metadata = {
            'section': section,
            'contains_technical': any(word in chunk_lower for word in ['esql', 'xsl', 'schema', 'mapping', 'transformation']),
            'contains_business': any(word in chunk_lower for word in ['requirement', 'business', 'process', 'rule']),
            'contains_architecture': any(word in chunk_lower for word in ['component', 'system', 'flow', 'integration']),
            'contains_data': any(word in chunk_lower for word in ['data', 'message', 'format', 'structure']),
            'length': len(chunk)
        }
        
        # Flatten agent relevance scores into separate scalar fields
        for agent_name, score in agent_scores.items():
            metadata[f'agent_relevance_{agent_name}'] = float(score)  # Ensure it's a float scalar
        
        return metadata
    
    def _calculate_agent_scores(self, chunk_lower: str) -> Dict[str, float]:
        """Pre-calculate relevance scores for each agent"""
        agent_keywords = {
            'component_mapper': ['biztalk', 'orchestration', 'port', 'pipeline', 'mapping', 'transform'],
            'messageflow_generator': ['flow', 'message', 'routing', 'endpoint', 'integration', 'connection'],
            'ace_module_creator': ['esql', 'xsl', 'transformation', 'module', 'logic', 'enrichment'],
            'schema_generator': ['schema', 'xsd', 'structure', 'element', 'validation', 'type'],
            'quality_reviewer': ['requirement', 'validation', 'criteria', 'quality', 'compliance'],
            'esql_generator': ['esql', 'transformation', 'logic', 'business', 'database', 'compute', 'procedure', 'lookup'],
            'postman_collection_generator': [
                'integration', 'interface', 'connector', 'adapter', 'bridge',
                'flow', 'message', 'document', 'event', 'transaction',
                'endpoint', 'service', 'api', 'queue', 'topic', 'channel',
                'http', 'soap', 'rest', 'mq', 'jms', 'tcp', 'ssl',
                'transformation', 'mapping', 'enrichment', 'validation',
                'lookup', 'database', 'stored', 'procedure', 'query',
                'test', 'scenario', 'sample', 'verification', 'validation',
                'error', 'exception', 'handling', 'monitoring'
            ],
            'project_generator': [
                'project', 'architecture', 'build', 'configuration', 'dependency', 'library',
                'ace', 'toolkit', 'eclipse', 'nature', 'builder', 'runtime', 'deployment',
                'shared', 'component', 'relationship', 'management', 'structure', 'template'
            ]
        }
        
        scores = {}
        for agent, keywords in agent_keywords.items():
            score = sum(chunk_lower.count(keyword) for keyword in keywords)
            scores[agent] = score / max(len(chunk_lower.split()), 1)  # Normalize by length
            
        return scores