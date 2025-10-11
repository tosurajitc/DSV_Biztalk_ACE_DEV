from typing import List, Dict
from .vector_store import ChromaVectorStore

class SemanticSearchEngine:
    def __init__(self, vector_store: ChromaVectorStore):
        self.vector_store = vector_store
        self.collection = None
        # Define agent-specific search queries
        self.agent_queries = {
            'component_mapper': [
                'BizTalk orchestration components and schemas with process diagrams',
                'mapping transformations and data conversion with technical diagrams', 
                'system architecture diagrams and component relationships',
                'integration flow diagrams and message routing'
            ],
            'messageflow_generator': [
                'message flow patterns and routing logic with flow diagrams',
                'system integration diagrams and connection points',
                'data flow diagrams between applications and services',
                'technical architecture diagrams and message processing flows'
            ],
            'esql_generator': [
                'database lookup diagrams and stored procedures',
                'data transformation logic with technical diagrams',
                'enrichment process diagrams and business rules',
                'technical specifications with database integration diagrams'
            ],
            'schema_generator': [
                'schema definitions with data structure diagrams', 
                'message format specifications and technical diagrams',
                'data model diagrams and schema relationships',
                'technical specifications with data flow diagrams'
            ],
            'xsl_generator': [
                'transformation mapping diagrams and XSL specifications',
                'data conversion diagrams and mapping logic',
                'technical diagrams with transformation processes',
                'message transformation flows and technical specifications'
            ],
            'project_generator': [
                'system architecture diagrams and project dependencies',
                'technical integration diagrams and component relationships',
                'project structure diagrams and module dependencies',
                'technical specifications with system integration diagrams'
            ],
            'application_descriptor_generator': [
                'application descriptor and library dependencies',
                'shared library configurations and deployment',
                'dependency management and component integration'
            ],
            'enrichment_generator': [
                'data enrichment and database lookups',
                'integration and business rule processing',
                'data transformation and enhancement patterns'
            ],
            'postman_collection_generator': [
                'api', 'endpoint', 'service', 'rest', 'http', 'soap',
                'test', 'testing', 'validation', 'verification', 'scenario',
                'request', 'response', 'authentication', 'authorization',
                'integration', 'interface', 'connector', 'adapter',
                'error', 'exception', 'handling', 'monitoring',
                'queue', 'topic', 'mq', 'jms', 'ssl', 'tcp'
            ],
            'migration_quality_reviewer': [
                'quality', 'compliance', 'standard', 'best', 'practice',
                'review', 'audit', 'assessment', 'validation', 'verification',
                'architecture', 'design', 'pattern', 'framework', 'structure',
                'performance', 'scalability', 'reliability', 'security',
                'integration', 'compatibility', 'dependency', 'component',
                'error', 'exception', 'handling', 'monitoring', 'logging',
                'testing', 'coverage', 'criteria', 'requirement', 'specification'
            ],
        }

    def ensure_collection_exists(self):
        """Ensure collection is properly initialized"""
        try:
            if self.collection is None:
                # Re-create collection if missing
                if hasattr(self, 'vector_store') and self.vector_store:
                    self.collection = self.vector_store.collection
                    
                    # Check if collection is still None after assignment
                    if self.collection is None:
                        print("🔧 Force creating collection in SemanticSearchEngine")
                        # Create a new vector store if needed
                        from .vector_store import ChromaVectorStore
                        if not hasattr(self, 'vector_store') or self.vector_store is None:
                            self.vector_store = ChromaVectorStore()
                        
                        # Create a new collection with an empty knowledge base if needed
                        if hasattr(self.vector_store, 'create_collection'):
                            self.vector_store.create_collection()
                        
                        # Get the new collection
                        self.collection = self.vector_store.collection
                else:
                    # Initialize vector store if missing
                    from .vector_store import ChromaVectorStore
                    self.vector_store = ChromaVectorStore()
                    self.collection = self.vector_store.collection
            
            return self.collection is not None
        
        except Exception as e:
            print(f"Collection creation failed: {e}")
            return False
    
    def get_agent_content(self, agent_name: str, top_k: int = 5) -> str:
        """
        ENHANCED: Get focused content for specific agent with diagram data integration
        """
        # Check collection exists first
        if not self.ensure_collection_exists():
            return f"No content available - collection not initialized"
            
        # Check if agent name is valid
        if agent_name not in self.agent_queries:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        queries = self.agent_queries[agent_name]
        
        # Search for relevant chunks with diagram priority
        results = self.vector_store.search_for_agent_with_diagrams(agent_name, queries, top_k)
        
        if not results:
            return f"No relevant content found for {agent_name}"
        
        # Assemble content with enhanced diagram extraction
        content_parts = []
        content_parts.append(f"=== FOCUSED CONTENT FOR {agent_name.upper()} ===\n")
        
        # Separate regular content and diagram-rich content
        diagram_sections = []
        regular_sections = []
        
        for i, result in enumerate(results, 1):
            content = result['content']
            metadata = result['metadata']
            
            has_technical_diagrams = metadata.get('has_technical_diagrams', False) or metadata.get('has_diagrams', False)
            
            section_info = {
                'index': i,
                'score': result['score'],
                'source': metadata.get('section', 'Unknown'),
                'query': result['query'],
                'content': content,
                'has_technical_diagrams': has_technical_diagrams,
                'diagram_count': metadata.get('diagram_count', 0)
            }
            
            if has_technical_diagrams:
                diagram_sections.append(section_info)
            else:
                regular_sections.append(section_info)
        
        # Prioritize diagram-rich content first
        all_sections = diagram_sections + regular_sections
        
        for section in all_sections:
            content_parts.append(f"--- Relevant Section {section['index']} (Score: {section['score']:.3f}) ---")
            content_parts.append(f"Source: {section['source']}")
            content_parts.append(f"Query: {section['query']}")
            
            # Add diagram indicator
            if section['has_technical_diagrams']:
                content_parts.append(f"🔍 CONTAINS TECHNICAL DIAGRAMS ({section['diagram_count']} diagrams)")
            
            content_parts.append(section['content'])
            content_parts.append("")  # Empty line for separation
        
        # Add diagram summary at the top if any diagrams found
        if diagram_sections:
            diagram_summary = f"\n🎯 DIAGRAM CONTENT AVAILABLE: {len(diagram_sections)} sections with technical diagrams\n"
            content_parts.insert(1, diagram_summary)
        
        focused_content = "\n".join(content_parts)
        
        print(f"✅ {agent_name}: Retrieved {len(results)} relevant chunks ({len(focused_content)} chars) - {len(diagram_sections)} with diagrams")
        return focused_content
    
    def get_search_summary(self, agent_name: str) -> Dict:
        """Get summary of search results for monitoring"""
        # Check collection exists first
        if not self.ensure_collection_exists():
            return {
                'agent': agent_name,
                'queries_used': 0,
                'chunks_found': 0,
                'avg_relevance': 0,
                'content_size_chars': 0,
                'sections_covered': []
            }
            
        queries = self.agent_queries[agent_name]
        results = self.vector_store.search_for_agent_with_diagrams(agent_name, queries, 3)
        
        return {
            'agent': agent_name,
            'queries_used': len(queries),
            'chunks_found': len(results),
            'avg_relevance': sum(r['score'] for r in results) / len(results) if results else 0,
            'content_size_chars': sum(len(r['content']) for r in results),
            'sections_covered': list(set(r['metadata'].get('section', 'Unknown') for r in results))
        }