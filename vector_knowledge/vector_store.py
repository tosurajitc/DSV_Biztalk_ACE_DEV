#!/usr/bin/env python3
"""
Enhanced Vector Store with Langchain Integration
===============================================

Provides vector database functionality using Langchain and ChromaDB,
optimized to work with chunk_creator.py outputs.

Features:
- Direct integration with langchain Document objects
- Support for different embedding models
- Enhanced semantic search capabilities
- Backward compatibility with existing code
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass

# Langchain imports
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VectorStore")

@dataclass
class SearchResult:
    """Standardized search result format"""
    content: str
    metadata: Dict[str, Any]
    score: float
    id: str
    source: str
    query: str

class EmbeddingProvider:
    """Manages different embedding providers and models"""
    
    @staticmethod
    def get_embedding_function(provider: str = "huggingface", model_name: str = "all-MiniLM-L6-v2") -> Embeddings:
        """Get embedding function based on provider and model"""
        if provider == "huggingface":
            # HuggingFace embeddings (default)
            return HuggingFaceEmbeddings(model_name=model_name)
        elif provider == "openai":
            # OpenAI embeddings if API key is available
            from langchain.embeddings import OpenAIEmbeddings
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                return OpenAIEmbeddings()
            else:
                logger.warning("OpenAI API key not found, falling back to HuggingFace")
                return HuggingFaceEmbeddings(model_name=model_name)
        elif provider == "default":
            # Fallback to Chroma's default embeddings (SentenceTransformers)
            # For compatibility with existing code
            try:
                from chromadb.utils import embedding_functions
                return embedding_functions.DefaultEmbeddingFunction()
            except ImportError:
                logger.warning("ChromaDB default embedding not available, using HuggingFace")
                return HuggingFaceEmbeddings(model_name=model_name)
        else:
            logger.warning(f"Unknown embedding provider '{provider}', using HuggingFace")
            return HuggingFaceEmbeddings(model_name=model_name)

class VectorStore:
    """Enhanced vector store with langchain integration"""
    
    def __init__(self, 
                 collection_name: str = "business_requirements",
                 persist_directory: str = "business_requirement/chroma_db",
                 embedding_provider: str = "huggingface",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
            embedding_provider: Provider for embeddings (huggingface, openai, default)
            embedding_model: Model name for embeddings
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Create directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize embedding function
        self.embedding_function = EmbeddingProvider.get_embedding_function(
            provider=embedding_provider,
            model_name=embedding_model
        )
        
        # Initialize vector store
        self._initialize_vector_store()
        
        logger.info(f"Vector store initialized with collection: {collection_name}")
    
    def _initialize_vector_store(self):
        """Initialize the vector store with langchain Chroma"""
        try:
            # Try to load existing database
            self.vector_db = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory
            )
            
            # Get collection stats
            self.collection_count = len(self.vector_db.get())
            logger.info(f"Loaded existing vector store with {self.collection_count} documents")
            
        except Exception as e:
            logger.warning(f"Could not load existing vector store: {str(e)}")
            # Create new database
            self.vector_db = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory
            )
            self.collection_count = 0
            logger.info("Created new vector store")
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add langchain Document objects to vector store
        
        Args:
            documents: List of langchain Document objects
            
        Returns:
            bool: Success status
        """
        try:
            self.vector_db.add_documents(documents)
            self.vector_db.persist()
            self.collection_count += len(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
            return True
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> bool:
        """
        Add texts with optional metadata to vector store
        
        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            bool: Success status
        """
        try:
            self.vector_db.add_texts(texts=texts, metadatas=metadatas)
            self.vector_db.persist()
            self.collection_count += len(texts)
            logger.info(f"Added {len(texts)} texts to vector store")
            return True
        except Exception as e:
            logger.error(f"Error adding texts: {str(e)}")
            return False
        
    
    def add_chunks(self, chunks):
        """Add chunks from chunk_creator.py to vector store"""
        try:
            # Convert chunks to langchain Document objects
            documents = []
            for chunk in chunks:
                # Extract required fields
                content = chunk.get('content', '')
                metadata = chunk.get('metadata', {})
                
                # CRITICAL FIX: Filter complex metadata to avoid Chroma errors
                filtered_metadata = {}
                for key, value in metadata.items():
                    # Only include simple types that Chroma supports
                    if isinstance(value, (str, int, float, bool, type(None))):
                        filtered_metadata[key] = value
                    # For lists of simple types, convert to string representation
                    elif isinstance(value, list) and all(isinstance(x, (str, int, float, bool)) for x in value):
                        filtered_metadata[key] = ", ".join(str(x) for x in value)
                
                # Add source and id to metadata if not present
                if 'source' not in filtered_metadata and 'source' in chunk:
                    filtered_metadata['source'] = chunk['source']
                if 'id' not in filtered_metadata and 'id' in chunk:
                    filtered_metadata['id'] = chunk['id']
                
                # Create Document object with filtered metadata
                doc = Document(page_content=content, metadata=filtered_metadata)
                documents.append(doc)
            
            # Add documents to vector store
            self.vector_db.add_documents(documents)
            self.vector_db.persist()
            self.collection_count += len(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding chunks: {str(e)}")
            return False
    
    def similarity_search(self, 
                          query: str, 
                          k: int = 5,
                          filter: Optional[Dict] = None) -> List[SearchResult]:
        """
        Perform similarity search with optional filters
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional filter criteria
            
        Returns:
            List[SearchResult]: Search results
        """
        try:
            # Use langchain similarity search
            docs = self.vector_db.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            
            # Convert to standardized format
            results = []
            for doc, score in docs:
                result = SearchResult(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=1.0 - score,  # Convert distance to similarity score
                    id=doc.metadata.get('id', ''),
                    source=doc.metadata.get('source', 'unknown'),
                    query=query
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def multi_query_search(self, 
                           queries: List[str], 
                           k: int = 5,
                           filter: Optional[Dict] = None,
                           deduplicate: bool = True) -> List[SearchResult]:
        """
        Perform search with multiple queries and combine results
        
        Args:
            queries: List of search queries
            k: Number of results to return per query
            filter: Optional filter criteria
            deduplicate: Whether to remove duplicate results
            
        Returns:
            List[SearchResult]: Combined search results
        """
        all_results = []
        
        for query in queries:
            results = self.similarity_search(query=query, k=k, filter=filter)
            all_results.extend(results)
        
        # Remove duplicates if requested
        if deduplicate:
            unique_results = self._remove_duplicates(all_results)
        else:
            unique_results = all_results
        
        # Sort by score
        sorted_results = sorted(unique_results, key=lambda x: x.score, reverse=True)
        
        # Limit to k results
        return sorted_results[:k]
    
    def agent_specific_search(self, 
                              agent_name: str, 
                              queries: List[str], 
                              k: int = 5) -> List[SearchResult]:
        """
        Perform agent-specific search with customized relevance scoring
        
        Args:
            agent_name: Name of the agent for specialized search
            queries: List of search queries
            k: Number of results to return
            
        Returns:
            List[SearchResult]: Search results optimized for specific agent
        """
        # Agent-specific keywords for relevance boosting
        agent_keywords = {
            'component_mapper': ['integration', 'component', 'mapping', 'biztalk', 'adapter', 'connector'],
            'schema_generator': ['schema', 'xsd', 'structure', 'element', 'validation', 'type'],
            'esql_generator': ['esql', 'transformation', 'logic', 'business', 'database', 'compute'],
            'messageflow_generator': ['flow', 'message', 'routing', 'endpoint', 'integration'],
            'postman_collection_generator': ['test', 'api', 'endpoint', 'validation', 'http', 'rest']
        }
        
        # Get keywords for this agent
        keywords = agent_keywords.get(agent_name, [])
        
        # Get raw results
        all_results = self.multi_query_search(queries=queries, k=k*2, deduplicate=True)
        
        # Apply agent-specific relevance scoring
        for result in all_results:
            content_lower = result.content.lower()
            
            # Calculate keyword matches in content
            content_score = sum(1 for keyword in keywords if keyword in content_lower) / max(len(keywords), 1)
            
            # Calculate keyword matches in metadata
            metadata_str = json.dumps(result.metadata).lower()
            metadata_score = sum(1 for keyword in keywords if keyword in metadata_str) / max(len(keywords), 1)
            
            # Apply agent-specific boost
            agent_relevance = (content_score * 0.6) + (metadata_score * 0.4)
            boosted_score = result.score + (agent_relevance * 0.3)
            
            # Update score with boosted value
            result.score = min(boosted_score, 1.0)  # Cap at 1.0
        
        # Sort by boosted score
        sorted_results = sorted(all_results, key=lambda x: x.score, reverse=True)
        
        return sorted_results[:k]
    
    def _remove_duplicates(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on content similarity"""
        unique_results = []
        seen_content = set()
        
        for result in results:
            # Use first 100 chars as hash for deduplication
            content_hash = hash(result.content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
                
        return unique_results
    
    def clear(self) -> bool:
        """Clear all documents from the vector store"""
        try:
            # Get Chroma collection
            collection = self.vector_db._collection
            
            # Get all IDs
            ids = collection.get()['ids']
            
            # Delete all documents
            if ids:
                collection.delete(ids=ids)
                self.vector_db.persist()
                self.collection_count = 0
                logger.info(f"Cleared {len(ids)} documents from vector store")
            
            return True
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        embedding_info = str(self.embedding_function.__class__.__name__)
        if hasattr(self.embedding_function, 'model_name'):
            embedding_info += f" ({self.embedding_function.model_name})"
        
        return {
            'collection_name': self.collection_name,
            'document_count': self.collection_count,
            'embedding_model': embedding_info,
            'persist_directory': self.persist_directory
        }

# For backward compatibility with existing code
class ChromaVectorStore(VectorStore):
    """Legacy class for backward compatibility"""
    
    def __init__(self, db_path: str = "./chroma_db"):
        """Initialize with legacy parameters"""
        collection_name = "confluence_knowledge"
        super().__init__(
            collection_name=collection_name,
            persist_directory=db_path,
            embedding_provider="default"
        )
    
    def create_knowledge_base(self, chunks: List[Dict]) -> None:
        """Legacy method to create knowledge base from chunks"""
        # Convert to new format if needed
        if chunks and 'chunk_id' in chunks[0]:
            # Legacy format
            for i, chunk in enumerate(chunks):
                if 'id' not in chunk:
                    chunk['id'] = f"chunk_{chunk.get('chunk_id', i)}"
        
        # Add chunks to vector store
        success = self.add_chunks(chunks)
        if success:
            logger.info(f"✅ Knowledge base created with {len(chunks)} chunks")
        else:
            logger.error("❌ Failed to create knowledge base")
    
    def search_for_agent(self, agent_name: str, queries: List[str], top_k: int = 5) -> List[Dict]:
        """Legacy method to search for agent-specific content"""
        results = self.agent_specific_search(
            agent_name=agent_name,
            queries=queries,
            k=top_k
        )
        
        # Convert to legacy format
        legacy_results = []
        for result in results:
            legacy_results.append({
                'content': result.content,
                'score': result.score,
                'metadata': result.metadata,
                'query': result.query
            })
        
        return legacy_results

# Example usage
if __name__ == "__main__":
    # Create vector store
    vector_store = VectorStore(
        collection_name="test_collection",
        persist_directory="business_requirement/chroma_db"
    )
    
    # Example data
    texts = [
        "The customer must be able to place an order",
        "The system shall validate credit card details",
        "Integration with payment gateway is required"
    ]
    
    metadatas = [
        {"source": "requirements", "type": "business_rule"},
        {"source": "requirements", "type": "validation_rule"},
        {"source": "requirements", "type": "integration"}
    ]
    
    # Add texts
    vector_store.add_texts(texts, metadatas)
    
    # Search
    results = vector_store.similarity_search("payment processing", k=2)
    for result in results:
        print(f"Score: {result.score:.2f}, Content: {result.content}")
    
    # Get stats
    stats = vector_store.get_stats()
    print(f"Stats: {stats}")