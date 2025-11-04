import streamlit as st
import os
from typing import Dict, Any, Optional
from chunk_creator import create_chunks
from vector_store import VectorStore
from semantic_search import SemanticSearchEngine

class VectorOptimizedPipeline:
    """
    Main vector pipeline class that integrates with your existing application
    """
    
    def __init__(self):
        """Initialize the pipeline"""
        self.vector_store = None
        self.search_engine = None
        self.knowledge_ready = False
        self.BUSINESS_REQUIREMENT_DIR = "business_requirement"
        
        # Create business requirement directory if it doesn't exist
        os.makedirs(self.BUSINESS_REQUIREMENT_DIR, exist_ok=True)
    
    def setup_knowledge_base(self, content_source, biztalk_folder: Optional[str] = None):
        """
        Setup vector knowledge base from content source and optional BizTalk folder
        
        Args:
            content_source: PDF file or text content string (from Confluence)
            biztalk_folder: Optional path to BizTalk folder for component analysis
            
        Returns:
            Dict with knowledge base statistics
        """
        print("🚀 Setting up vector knowledge base...")
        
        try:
            # Initialize vector store
            self.vector_store = VectorStore(
                collection_name="business_requirements",
                persist_directory=os.path.join(self.BUSINESS_REQUIREMENT_DIR, "chroma_db")
            )
            
            # Check if consolidated_analysis.json exists (from BizTalk analyzer)
            consolidated_json_path = None
            if biztalk_folder:
                consolidated_json_path = os.path.join(self.BUSINESS_REQUIREMENT_DIR, "consolidated_analysis.json")
                if os.path.exists(consolidated_json_path):
                    print(f"📋 Found consolidated analysis from BizTalk folder")
            
            # Create chunks from content source and consolidated analysis
            debug_output_path = os.path.join(self.BUSINESS_REQUIREMENT_DIR, "debug_chunks.json")
            chunks = create_chunks(
                content_source=content_source,
                consolidated_json_path=consolidated_json_path,
                debug_output_path=debug_output_path
            )
            
            if not chunks:
                raise Exception("No chunks created from content source")
            
            # Add chunks to vector store
            success = self.vector_store.add_chunks(chunks)
            
            if not success:
                raise Exception("Failed to add chunks to vector store")
            
            # Initialize search engine
            self.search_engine = SemanticSearchEngine(self.vector_store)
            
            # Mark knowledge base as ready
            self.knowledge_ready = True
            
            # Get statistics
            stats = self.vector_store.get_stats()
            stats['chunks_created'] = len(chunks)
            stats['biztalk_processed'] = consolidated_json_path is not None
            stats['debug_output'] = debug_output_path
            
            print(f"✅ Vector knowledge base ready! {len(chunks)} chunks indexed")
            return stats
            
        except Exception as e:
            print(f"❌ Vector knowledge base setup failed: {e}")
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
        print(f"📊 Search Summary for {agent_name}: {search_summary}")
        
        # Run the agent function with focused content
        try:
            result = agent_function(focused_content)
            return result
        except Exception as e:
            print(f"❌ Agent {agent_name} failed with vector content: {e}")
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
    
    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """Get information about the current knowledge base"""
        if not self.knowledge_ready:
            return {"status": "not_ready"}
        
        stats = self.vector_store.get_stats()
        stats["status"] = "ready"
        stats["search_engine_ready"] = self.search_engine is not None
        
        return stats
    
    def reset_knowledge_base(self):
        """Reset the knowledge base (useful for switching PDFs)"""
        if self.vector_store:
            # Clear the vector store
            self.vector_store.clear()
        
        self.vector_store = None
        self.search_engine = None
        self.knowledge_ready = False
        print("🔄 Vector knowledge base reset")