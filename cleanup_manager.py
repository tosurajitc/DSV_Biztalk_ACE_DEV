"""
Cleanup Manager Module
Handles cleanup of Vector DB, output folders, and temporary files
"""

import os
import shutil
from typing import Dict, List, Optional
from pathlib import Path


class CleanupManager:
    """Manages cleanup operations for the ACE migration project"""
    
    def __init__(self, 
                 chroma_db_path: str = "chroma_db",
                 output_dir: str = "output",
                 root_cleanup_patterns: Optional[List[str]] = None):
        """
        Initialize CleanupManager
        
        Args:
            chroma_db_path: Path to ChromaDB storage
            output_dir: Path to output directory
            root_cleanup_patterns: List of file patterns to clean from root (e.g., ["naming_convention*.json"])
        """
        self.chroma_db_path = chroma_db_path
        self.output_dir = output_dir
        self.root_cleanup_patterns = root_cleanup_patterns or [
            "naming_convention*.json",
            "msgflow_template.xml"
        ]
        self.cleanup_results = {
            'vector_db': {'status': 'pending', 'message': ''},
            'output_folder': {'status': 'pending', 'message': ''},
            'root_files': {'status': 'pending', 'message': '', 'files_removed': []},
            'overall_status': 'pending'
        }
    
    def cleanup_vector_db(self) -> Dict[str, str]:
        """
        Cleanup ChromaDB vector database
        
        Returns:
            Dict with status and message
        """
        try:
            import chromadb
            import time
            
            client = chromadb.PersistentClient(path=self.chroma_db_path)
            
            # List all collections
            collections = client.list_collections()
            
            if collections:
                for collection in collections:
                    try:
                        client.delete_collection(name=collection.name)
                        time.sleep(0.1)  # Small delay between deletions
                    except Exception as e:
                        pass  # Continue even if individual collection deletion fails
            
            self.cleanup_results['vector_db'] = {
                'status': 'success',
                'message': f'ChromaDB cleaned ({len(collections)} collections removed)'
            }
            
        except ImportError:
            self.cleanup_results['vector_db'] = {
                'status': 'warning',
                'message': 'ChromaDB not installed, skipping vector DB cleanup'
            }
        except Exception as e:
            self.cleanup_results['vector_db'] = {
                'status': 'warning',
                'message': f'ChromaDB cleanup issue: {str(e)}'
            }
        
        return self.cleanup_results['vector_db']
    
    def cleanup_output_folder(self) -> Dict[str, str]:
        """
        Cleanup output folder - remove all files and subdirectories
        
        Returns:
            Dict with status and message
        """
        try:
            if os.path.exists(self.output_dir):
                items_removed = 0
                
                for item in os.listdir(self.output_dir):
                    item_path = os.path.join(self.output_dir, item)
                    
                    try:
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                            items_removed += 1
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                            items_removed += 1
                    except Exception as e:
                        # Log but continue with other items
                        print(f"Warning: Could not remove {item_path}: {str(e)}")
                
                self.cleanup_results['output_folder'] = {
                    'status': 'success',
                    'message': f'Output folder cleaned ({items_removed} items removed)'
                }
            else:
                self.cleanup_results['output_folder'] = {
                    'status': 'success',
                    'message': 'Output folder does not exist (nothing to clean)'
                }
                
        except Exception as e:
            self.cleanup_results['output_folder'] = {
                'status': 'error',
                'message': f'Output folder cleanup failed: {str(e)}'
            }
        
        return self.cleanup_results['output_folder']
    
    def cleanup_root_files(self) -> Dict[str, any]:
        """
        Cleanup specific files from root directory based on patterns
        Removes: naming_convention*.json, msgflow_template.xml, etc.
        
        Returns:
            Dict with status, message, and list of files removed
        """
        files_removed = []
        
        try:
            root_path = Path(".")
            
            for pattern in self.root_cleanup_patterns:
                # Use glob to find matching files
                matching_files = list(root_path.glob(pattern))
                
                for file_path in matching_files:
                    if file_path.is_file():
                        try:
                            os.remove(file_path)
                            files_removed.append(str(file_path))
                        except Exception as e:
                            print(f"Warning: Could not remove {file_path}: {str(e)}")
            
            if files_removed:
                self.cleanup_results['root_files'] = {
                    'status': 'success',
                    'message': f'Root files cleaned ({len(files_removed)} files removed)',
                    'files_removed': files_removed
                }
            else:
                self.cleanup_results['root_files'] = {
                    'status': 'success',
                    'message': 'No matching root files found to clean',
                    'files_removed': []
                }
                
        except Exception as e:
            self.cleanup_results['root_files'] = {
                'status': 'error',
                'message': f'Root files cleanup failed: {str(e)}',
                'files_removed': files_removed
            }
        
        return self.cleanup_results['root_files']
    
    def perform_full_cleanup(self) -> Dict[str, any]:
        """
        Perform complete cleanup: Vector DB + Output Folder + Root Files
        
        Returns:
            Dict with comprehensive cleanup results
        """
        print("🧹 Starting full cleanup...")
        
        # 1. Vector DB Cleanup
        print("  Step 1/3: Cleaning Vector DB...")
        self.cleanup_vector_db()
        
        # 2. Output Folder Cleanup
        print("  Step 2/3: Cleaning output folder...")
        self.cleanup_output_folder()
        
        # 3. Root Files Cleanup
        print("  Step 3/3: Cleaning root files...")
        self.cleanup_root_files()
        
        # Determine overall status
        statuses = [
            self.cleanup_results['vector_db']['status'],
            self.cleanup_results['output_folder']['status'],
            self.cleanup_results['root_files']['status']
        ]
        
        if all(s == 'success' for s in statuses):
            self.cleanup_results['overall_status'] = 'success'
        elif any(s == 'error' for s in statuses):
            self.cleanup_results['overall_status'] = 'partial'
        else:
            self.cleanup_results['overall_status'] = 'success'
        
        print("✅ Cleanup completed!")
        return self.cleanup_results
    
    def get_cleanup_summary(self) -> str:
        """
        Get a human-readable summary of cleanup results
        
        Returns:
            Formatted string with cleanup summary
        """
        summary_lines = ["Cleanup Summary:", "=" * 50]
        
        # Vector DB
        vdb = self.cleanup_results['vector_db']
        summary_lines.append(f"Vector DB: {vdb['status'].upper()} - {vdb['message']}")
        
        # Output Folder
        out = self.cleanup_results['output_folder']
        summary_lines.append(f"Output Folder: {out['status'].upper()} - {out['message']}")
        
        # Root Files
        root = self.cleanup_results['root_files']
        summary_lines.append(f"Root Files: {root['status'].upper()} - {root['message']}")
        if root.get('files_removed'):
            summary_lines.append(f"  Files removed: {', '.join(root['files_removed'])}")
        
        summary_lines.append("=" * 50)
        summary_lines.append(f"Overall Status: {self.cleanup_results['overall_status'].upper()}")
        
        return "\n".join(summary_lines)


# Convenience function for quick cleanup
def perform_cleanup(chroma_db_path: str = "chroma_db",
                   output_dir: str = "output",
                   additional_patterns: Optional[List[str]] = None) -> Dict[str, any]:
    """
    Perform full cleanup with default or custom settings
    
    Args:
        chroma_db_path: Path to ChromaDB storage
        output_dir: Path to output directory
        additional_patterns: Additional file patterns to clean from root
    
    Returns:
        Dict with cleanup results
    """
    patterns = ["naming_convention*.json", "msgflow_template.xml"]
    if additional_patterns:
        patterns.extend(additional_patterns)
    
    manager = CleanupManager(
        chroma_db_path=chroma_db_path,
        output_dir=output_dir,
        root_cleanup_patterns=patterns
    )
    
    return manager.perform_full_cleanup()


if __name__ == "__main__":
    # Example usage
    manager = CleanupManager()
    results = manager.perform_full_cleanup()
    print("\n" + manager.get_cleanup_summary())