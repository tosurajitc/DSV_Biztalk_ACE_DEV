import streamlit as st
from .pdf_processor import AdaptivePDFProcessor
from .vector_store import ChromaVectorStore
from .semantic_search import SemanticSearchEngine

class VectorOptimizedPipeline:
    """
    Main vector pipeline class that integrates with your existing application
    """
    
    def __init__(self):
        self.vector_store = None
        self.search_engine = None
        self.knowledge_ready = False
        self.pdf_processor = AdaptivePDFProcessor()
    


    def fix_collection_issue(self):
        """Fix the collection not being available issue"""
        try:
            if (hasattr(self, 'search_engine') and 
                self.search_engine and 
                not hasattr(self.search_engine, 'collection')):
                
                print("ðŸ”§ Fixing missing collection in search_engine")
                
                # Link the collection from vector_store to search_engine
                if (hasattr(self, 'vector_store') and 
                    self.vector_store and 
                    hasattr(self.vector_store, 'collection')):
                    
                    self.search_engine.collection = self.vector_store.collection
                    print("âœ… Collection linked successfully")
                    return True
                else:
                    print("âŒ No vector_store.collection found")
                    return False
        except Exception as e:
            print(f"Failed to fix collection: {e}")
            return False


    def setup_knowledge_base(self, uploaded_file):
        """
        Setup vector knowledge base from uploaded PDF file or content string
        
        Args:
            uploaded_file: Either a Streamlit UploadedFile, string content, or dictionary
                
        Returns:
            Dict with knowledge base statistics
        """
        print("🚀 Setting up vector knowledge base...")
        
        try:
            import tempfile
            import os
            
            # Handle different input types
            if isinstance(uploaded_file, str):
                # Process content string directly
                print("  Processing string content directly...")
                text = uploaded_file
                
                if not text.strip():
                    raise Exception("Empty content string provided")
                    
                # Create raw_content structure expected by the pipeline
                raw_content = {'text': text, 'tables': [], 'diagrams': [], 'metadata': {}}
                
            elif isinstance(uploaded_file, dict):
                # Process dictionary directly
                print("  Processing dictionary content directly...")
                if 'text' in uploaded_file:
                    text = uploaded_file['text']
                    raw_content = uploaded_file
                else:
                    text = str(uploaded_file)
                    raw_content = {'text': text, 'tables': [], 'diagrams': [], 'metadata': {}}
                    
            else:
                # Handle file-like object (e.g., Streamlit UploadedFile)
                print("  Processing uploaded file...")
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    # Write uploaded file content to temp file
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name
                
                # Reset uploaded file pointer in case it's needed elsewhere
                if hasattr(uploaded_file, 'seek'):
                    uploaded_file.seek(0)
                    
                try:
                    # Process PDF using AdaptivePDFProcessor
                    text = self.pdf_processor.extract_text_from_pdf(temp_file_path)
                    
                    if not text.strip():
                        raise Exception("Could not extract text from PDF")
                    
                    # Create raw_content structure expected by the pipeline
                    raw_content = {'text': text, 'tables': [], 'diagrams': [], 'metadata': {}}
                    
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass  # Ignore cleanup errors
            
            # Get chunks from the adaptive processing pipeline
            chunks = self.pdf_processor._create_adaptive_chunks(raw_content)

            if not chunks:
                raise Exception("No chunks created from content")

            # Check for the correct vector preparation method
            if hasattr(self.pdf_processor, '_prepare_adaptive_vector_chunks'):
                print("📊 Preparing chunks with adaptive metadata for vector store...")
                vector_ready_chunks = self.pdf_processor._prepare_adaptive_vector_chunks(chunks)
                
                # Count enhanced chunks
                enhanced_chunks = sum(1 for chunk in vector_ready_chunks 
                                if chunk.get('metadata', {}).get('section_type') == 'content')
                print(f"✅ Prepared {len(vector_ready_chunks)} chunks ({enhanced_chunks} with adaptive metadata)")
            else:
                print("⚠️ Adaptive vector preparation not available, using standard chunks")
                vector_ready_chunks = chunks

            # Create vector store with enhanced chunks
            self.vector_store = ChromaVectorStore()
            self.vector_store.create_knowledge_base(vector_ready_chunks)
            
            # Initialize search engine
            self.search_engine = SemanticSearchEngine(self.vector_store)
            
            self.knowledge_ready = True
            
            stats = self.vector_store.get_stats()
            stats['chunks_created'] = len(chunks)
            stats['vector_ready_chunks'] = len(vector_ready_chunks)
            stats['pdf_processed'] = True
            stats['adaptive_processing'] = True
            
            # Add adaptive processing statistics if available
            if hasattr(self.pdf_processor, 'processing_stats'):
                stats.update({
                    'business_blocks_found': self.pdf_processor.processing_stats.get('business_blocks_found', 0),
                    'patterns_discovered': self.pdf_processor.processing_stats.get('patterns_discovered', 0),
                    'diagrams_processed': self.pdf_processor.processing_stats.get('diagrams_processed', 0),
                    'tables_extracted': self.pdf_processor.processing_stats.get('tables_extracted', 0)
                })
            
            print(f"✅ Vector knowledge base ready! {len(vector_ready_chunks)} chunks indexed")
            return stats
                
        except Exception as e:
            print(f"❌ Vector knowledge base setup failed: {e}")
            import traceback
            traceback.print_exc()
            self.knowledge_ready = False
            raise e

        

    
        

    
    def run_agent_with_vector_search(self, agent_name: str, agent_function):
        """
        Run agent with vector search instead of full PDF
        
        Args:
            agent_name: Name of the agent (component_mapper, messageflow_generator, etc.)
            agent_function: Lambda function that takes confluence_content as input
            
        Returns:
            Result from agent function
        """
        if not self.knowledge_ready:
            raise Exception("Knowledge base not ready. Call setup_knowledge_base() first.")
        
        if not self.search_engine:
            raise Exception("Search engine not initialized.")
        
        # Get focused content for this specific agent
        focused_content = self.search_engine.get_agent_content(agent_name)
        
        # Get search summary for monitoring
        search_summary = self.search_engine.get_search_summary(agent_name)
        print(f"ðŸ“Š Search Summary for {agent_name}: {search_summary}")
        
        # Run the agent function with focused content
        try:
            result = agent_function(focused_content)
            return result
        except Exception as e:
            print(f"âŒ Agent {agent_name} failed with vector content: {e}")
            raise e
    
    def get_agent_content_preview(self, agent_name: str, max_chars: int = 500) -> str:
        """
        Get a preview of what content would be sent to an agent
        Useful for debugging and monitoring
        """
        if not self.knowledge_ready:
            return "Knowledge base not ready"
        
        focused_content = self.search_engine.get_agent_content(agent_name, top_k=2)
        
        if len(focused_content) > max_chars:
            return focused_content[:max_chars] + "..."
        
        return focused_content
    
    def get_knowledge_base_info(self) -> dict:
        """Get information about the current knowledge base"""
        if not self.knowledge_ready:
            return {"status": "not_ready"}
        
        stats = self.vector_store.get_stats()
        stats["status"] = "ready"
        stats["search_engine_ready"] = self.search_engine is not None
        
        return stats
    
    def reset_knowledge_base(self):
        """Reset the knowledge base (useful for switching PDFs)"""
        self.vector_store = None
        self.search_engine = None
        self.knowledge_ready = False
        print("ðŸ”„ Vector knowledge base reset")