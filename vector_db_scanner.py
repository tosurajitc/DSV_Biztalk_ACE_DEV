#!/usr/bin/env python3
"""
Vector DB Content Scanner
========================
Scans ChromaDB content to verify PDF chunking and technical detail extraction
"""

import chromadb
import json
from pathlib import Path
from typing import Dict, List

class VectorDBScanner:
    def __init__(self, db_path: str = "./chroma_db"):
        """Initialize connection to ChromaDB"""
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"ChromaDB path not found: {db_path}")
        
        print(f"Connecting to ChromaDB at: {self.db_path}")
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        
        # Get the collection
        collections = self.client.list_collections()
        print(f"Found collections: {[c.name for c in collections]}")
        
        if not collections:
            raise Exception("No collections found in ChromaDB")
        
        self.collection = collections[0]  # Use first collection
        print(f"Using collection: {self.collection.name}")

    def get_basic_stats(self) -> Dict:
        """Get basic statistics about the Vector DB"""
        try:
            # Get all documents without filtering
            all_data = self.collection.get()
            
            total_chunks = len(all_data['documents'])
            
            # Calculate content statistics
            total_chars = sum(len(doc) for doc in all_data['documents'])
            avg_chunk_size = total_chars / total_chunks if total_chunks > 0 else 0
            
            # Check metadata fields
            metadata_keys = set()
            if all_data['metadatas']:
                for metadata in all_data['metadatas']:
                    if metadata:
                        metadata_keys.update(metadata.keys())
            
            return {
                'total_chunks': total_chunks,
                'total_characters': total_chars,
                'avg_chunk_size': int(avg_chunk_size),
                'metadata_fields': list(metadata_keys)
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}

    def search_for_terms(self, search_terms: List[str], max_results: int = 5) -> Dict:
        """Search for specific terms in the Vector DB"""
        results = {}
        
        for term in search_terms:
            try:
                query_result = self.collection.query(
                    query_texts=[term],
                    n_results=max_results
                )
                
                matches = []
                if query_result['documents'][0]:  # Check if results exist
                    for i, doc in enumerate(query_result['documents'][0]):
                        distance = query_result['distances'][0][i]
                        matches.append({
                            'content_preview': doc[:200] + "..." if len(doc) > 200 else doc,
                            'distance': distance,
                            'similarity_score': 1 - distance,  # Convert distance to similarity
                            'full_length': len(doc)
                        })
                
                results[term] = {
                    'matches_found': len(matches),
                    'best_matches': matches
                }
                
            except Exception as e:
                results[term] = {'error': str(e)}
        
        return results

    def get_sample_chunks(self, num_samples: int = 5) -> List[Dict]:
        """Get sample chunks to see content variety"""
        try:
            all_data = self.collection.get()
            
            samples = []
            num_samples = min(num_samples, len(all_data['documents']))
            
            for i in range(num_samples):
                doc = all_data['documents'][i]
                metadata = all_data['metadatas'][i] if all_data['metadatas'] else {}
                
                samples.append({
                    'chunk_id': i,
                    'content_length': len(doc),
                    'content_preview': doc[:300] + "..." if len(doc) > 300 else doc,
                    'metadata': metadata
                })
            
            return samples
        except Exception as e:
            print(f"Error getting samples: {e}")
            return []

    def analyze_technical_content(self) -> Dict:
        """Analyze if technical content from your PDF is present"""
        # Technical terms that should be in your PDF
        technical_terms = [
            "CW1.IN.DOCUMENT.SND.QL",
            "eadapterqa.dsv.com",
            "eadapter.dsv.com",
            "sp_GetMainCompanyInCountry",
            "MH.ESB.EDIEnterprise",
            "DSV.ESB.Integration",
            "CompanyCode",
            "proc_EDocument_GetIsPublishedFlag",
            "CargoWise One",
            "eAdapter service"
        ]
        
        print("Searching for technical terms from your PDF...")
        search_results = self.search_for_terms(technical_terms, max_results=3)
        
        # Analyze results
        found_terms = []
        missing_terms = []
        
        for term, result in search_results.items():
            if 'error' in result:
                missing_terms.append(f"{term} (error: {result['error']})")
            elif result['matches_found'] > 0:
                best_score = result['best_matches'][0]['similarity_score']
                found_terms.append(f"{term} (similarity: {best_score:.3f})")
            else:
                missing_terms.append(f"{term} (no matches)")
        
        return {
            'found_terms': found_terms,
            'missing_terms': missing_terms,
            'search_results': search_results
        }

    def full_scan_report(self) -> None:
        """Generate a comprehensive scan report"""
        print("\n" + "="*60)
        print("VECTOR DB CONTENT SCAN REPORT")
        print("="*60)
        
        # Basic statistics
        print("\n1. BASIC STATISTICS:")
        stats = self.get_basic_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Sample content
        print("\n2. SAMPLE CHUNKS:")
        samples = self.get_sample_chunks(3)
        for sample in samples:
            print(f"\n   Chunk {sample['chunk_id']}:")
            print(f"   Length: {sample['content_length']} characters")
            print(f"   Preview: {sample['content_preview'][:150]}...")
            if sample['metadata']:
                print(f"   Metadata: {sample['metadata']}")
        
        # Technical content analysis
        print("\n3. TECHNICAL CONTENT ANALYSIS:")
        analysis = self.analyze_technical_content()
        
        print(f"\n   FOUND TERMS ({len(analysis['found_terms'])}):")
        for term in analysis['found_terms']:
            print(f"   ✓ {term}")
        
        print(f"\n   MISSING TERMS ({len(analysis['missing_terms'])}):")
        for term in analysis['missing_terms']:
            print(f"   ✗ {term}")
        
        # Detailed results for key terms
        print("\n4. DETAILED SEARCH RESULTS:")
        key_terms = ["CW1.IN.DOCUMENT.SND.QL", "eadapter.dsv.com", "CompanyCode"]
        
        for term in key_terms:
            if term in analysis['search_results']:
                result = analysis['search_results'][term]
                print(f"\n   Term: {term}")
                if result.get('matches_found', 0) > 0:
                    best_match = result['best_matches'][0]
                    print(f"   Best match (score: {best_match['similarity_score']:.3f}):")
                    print(f"   Content: {best_match['content_preview']}")
                else:
                    print(f"   No matches found")

def main():
    """Main execution"""
    try:
        # Initialize scanner
        scanner = VectorDBScanner("./chroma_db")  # Adjust path if needed
        
        # Run full scan
        scanner.full_scan_report()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're running this from the correct directory")
        print("2. Check that ./chroma_db folder exists")
        print("3. Verify that Vector DB was properly created")

if __name__ == "__main__":
    main()