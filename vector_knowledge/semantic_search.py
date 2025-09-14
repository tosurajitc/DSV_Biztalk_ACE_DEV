from typing import List, Dict
from .vector_store import ChromaVectorStore

class SemanticSearchEngine:
    def __init__(self, vector_store: ChromaVectorStore):
        self.vector_store = vector_store
        
        # Define agent-specific search queries
# Replace the agent_queries dictionary in semantic_search.py with this updated version:

        self.agent_queries = {
            'component_mapper': [
                'BizTalk orchestration components and schemas',
                'mapping transformations and data conversion',
                'pipeline components and message processing'
            ],
            'messageflow_generator': [
                'message flow patterns and routing logic',
                'system integration and connection points',
                'data flow between applications and services'
            ],
            'ace_module_creator': [
                'ESQL transformation modules and logic',
                'XSL stylesheet mappings and conversions',
                'data enrichment and computation rules'
            ],
            'schema_generator': [
                'schema definitions and data structures',
                'message format specifications and validation',
                'data type definitions and constraints'
            ],
            'migration_quality_reviewer': [
                'business requirements and acceptance criteria',
                'quality standards and validation rules',
                'compliance requirements and testing criteria',
                'naming conventions and template compliance',
                'migration quality patterns and best practices'
            ],
            'esql_generator': [
                'ESQL transformation logic and business rules',
                'database operations and data processing',
                'compute node implementations and message routing'
            ],
            'xsl_generator': [
                'XSL stylesheet mappings and field transformations',
                'data conversion and transformation patterns',
                'XML processing and template transformations'
            ],
            'application_descriptor_generator': [
                'library dependencies and shared libraries',
                'application configuration settings and parameters',
                'runtime requirements and deployment configurations'
            ],
            'enrichment_generator': [
                'CW1 document enrichment and database lookup operations',
                'CargoWise One data enhancement and validation rules', 
                'database stored procedures and enrichment transformations',
                'CDM document processing and Universal Event generation',
                'CompanyCode lookup and target recipient enrichment logic',
                'CW1 shipment matching and IsPublished flag validation'
            ],
            'postman_collection_generator': [
                'CDM document validation testing and XML schema compliance testing',
                'CargoWise One CW1 integration testing and eAdapter service validation',
                'database enrichment testing and stored procedure validation scenarios',
                'XSL transformation testing and CDM to UniversalEvent conversion validation', 
                'MQ message processing testing and queue-based integration scenarios',
                'business logic validation testing and data enrichment rule verification',
                'error handling testing and exception scenario validation for document processing',
                'CompanyCode and shipment matching testing for CW1 business rules'
            ],
            'project_generator': [
                'application integration and system connectivity requirements',
                'enterprise service bus and message processing specifications',
                'document transformation and data enrichment operations', 
                'business process automation and workflow integration',
                'database operations and lookup procedures for data enrichment',
                'message routing and transformation between connected systems',
                'system dependencies and component integration architecture',
                'service connectivity and endpoint configuration requirements'
            ]
        }
    
    def get_agent_content(self, agent_name: str, top_k: int = 5) -> str:
        """Get focused content for specific agent"""
        if agent_name not in self.agent_queries:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        queries = self.agent_queries[agent_name]
        
        # Search for relevant chunks
        results = self.vector_store.search_for_agent(agent_name, queries, top_k)
        
        if not results:
            return f"No relevant content found for {agent_name}"
        
        # Assemble content with context
        content_parts = []
        content_parts.append(f"=== FOCUSED CONTENT FOR {agent_name.upper()} ===\n")
        
        for i, result in enumerate(results, 1):
            content_parts.append(f"--- Relevant Section {i} (Score: {result['score']:.3f}) ---")
            content_parts.append(f"Source: {result['metadata'].get('section', 'Unknown')}")
            content_parts.append(f"Query: {result['query']}")
            content_parts.append(result['content'])
            content_parts.append("")  # Empty line for separation
        
        focused_content = "\n".join(content_parts)
        
        print(f"✅ {agent_name}: Retrieved {len(results)} relevant chunks ({len(focused_content)} chars)")
        return focused_content
    
    def get_search_summary(self, agent_name: str) -> Dict:
        """Get summary of search results for monitoring"""
        queries = self.agent_queries[agent_name]
        results = self.vector_store.search_for_agent(agent_name, queries, 3)
        
        return {
            'agent': agent_name,
            'queries_used': len(queries),
            'chunks_found': len(results),
            'avg_relevance': sum(r['score'] for r in results) / len(results) if results else 0,
            'content_size_chars': sum(len(r['content']) for r in results),
            'sections_covered': list(set(r['metadata'].get('section', 'Unknown') for r in results))
        }