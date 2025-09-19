import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional
import json

class ChromaVectorStore:
    def __init__(self, db_path: str = "./chroma_db"):
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Explicitly use ChromaDB's default embedding function (no downloads)
        default_ef = embedding_functions.DefaultEmbeddingFunction()
        
        # Create or get collection with explicit embedding function
        self.collection = self.client.get_or_create_collection(
            name="confluence_knowledge",
            metadata={"description": "Confluence PDF knowledge base"},
            embedding_function=default_ef  # ✅ Explicitly use default (no downloads)
        )



    def search_for_agent_with_diagrams(self, agent_name: str, queries: List[str], top_k: int = 5) -> List[Dict]:
        """
        NEW: Enhanced search that prioritizes chunks with diagram data
        """
        all_results = []
        
        for query in queries:
            # Add diagram-specific terms to query
            enhanced_query = query + " technical diagram process flow architecture"
            
            results = self.collection.query(
                query_texts=[enhanced_query],
                n_results=top_k * 2  # Get more results for filtering
            )
            
            for i in range(len(results['documents'][0])):
                result_data = {
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'id': results['ids'][0][i],
                    'query': query  # Add query for traceability
                }
                
                # Boost score for diagram-containing chunks
                if result_data['metadata'].get('has_technical_diagrams'):
                    result_data['boosted_score'] = result_data['distance'] * 0.7  # Better score
                    result_data['has_technical_diagrams'] = True
                    result_data['score'] = 1.0 - result_data['boosted_score']  # Convert to similarity
                else:
                    result_data['boosted_score'] = result_data['distance']
                    result_data['has_technical_diagrams'] = False
                    result_data['score'] = 1.0 - result_data['distance']
                
                all_results.append(result_data)
        
        # Remove duplicates and sort by boosted score
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result['id'] not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result['id'])
        
        # Sort by score (higher is better)
        unique_results.sort(key=lambda x: x['score'], reverse=True)
        return unique_results[:top_k]



    def create_knowledge_base(self, chunks: List[Dict]) -> None:
        """Create vector knowledge base from PDF chunks"""
        print(f"Creating knowledge base from {len(chunks)} chunks...")
        
        # Prepare data for ChromaDB
        documents = [chunk['content'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        ids = [f"chunk_{chunk['chunk_id']}" for chunk in chunks]
        
        # Add to ChromaDB (embeddings generated with default function - no downloads)
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Knowledge base created with {len(chunks)} chunks")

        
        
    def search_for_agent(self, agent_name: str, queries: List[str], top_k: int = 5) -> List[Dict]:
        """Search for agent-specific content using semantic similarity and keyword matching"""
        all_results = []
        
        # Agent-specific keywords for relevance scoring
        agent_keywords = {
            'component_mapper': ['integration', 'component', 'mapping', 'biztalk', 'adapter', 'connector'],
            'schema_generator': ['schema', 'xsd', 'structure', 'element', 'validation', 'type'],
            'esql_generator': ['esql', 'transformation', 'logic', 'business', 'database', 'compute'],
            'messageflow_generator': ['flow', 'message', 'routing', 'endpoint', 'integration'],
            'postman_collection_generator': ['test', 'api', 'endpoint', 'validation', 'http', 'rest']
        }
        
        keywords = agent_keywords.get(agent_name, [])
        
        for query in queries:
            # ✅ Remove the problematic where clause - use pure semantic search
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k * 2
            )
            
            # Process results and calculate agent relevance directly
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                
                # Calculate agent relevance directly in this method
                content_lower = doc.lower()
                content_score = sum(1 for keyword in keywords if keyword in content_lower) / max(len(keywords), 1)
                
                tech_components = metadata.get('technical_components', '')
                tech_score = sum(1 for keyword in keywords if keyword in tech_components.lower()) / max(len(keywords), 1)
                
                content_type = metadata.get('content_type', '')
                type_bonus = 0.2 if content_type in ['technical_spec', 'mapping_logic', 'description'] else 0
                
                agent_relevance = (content_score * 0.6) + (tech_score * 0.3) + type_bonus
                
                # Only include results with some relevance
                if agent_relevance > 0.1:
                    boosted_score = (1 - distance) + (agent_relevance * 0.3)
                    
                    all_results.append({
                        'content': doc,
                        'score': boosted_score,
                        'metadata': metadata,
                        'query': query
                    })
        
        # Sort by boosted score and remove duplicates
        unique_results = self._remove_duplicates(all_results)
        return sorted(unique_results, key=lambda x: x['score'], reverse=True)[:top_k]
    

    
    def _remove_duplicates(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate chunks based on content similarity"""
        unique_results = []
        seen_content = set()
        
        for result in results:
            content_hash = hash(result['content'][:100])  # Use first 100 chars as hash
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
                
        return unique_results
    
    def get_stats(self) -> Dict:
        """Get knowledge base statistics"""
        count = self.collection.count()
        return {
            'total_chunks': count,
            'embedding_model': 'chromadb-default-local',
            'collection_name': self.collection.name
        }