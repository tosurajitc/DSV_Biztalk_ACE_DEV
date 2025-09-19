from typing import List, Dict
from .vector_store import ChromaVectorStore

class SemanticSearchEngine:
    def __init__(self, vector_store: ChromaVectorStore):
        self.vector_store = vector_store
        
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
            ]
        }







    
    def get_agent_content(self, agent_name: str, top_k: int = 5) -> str:
        """
        ENHANCED: Get focused content for specific agent with diagram data integration
        """
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