#!/usr/bin/env python3
"""
Enhanced Chunk Creator with OCR for Process Diagrams
====================================================

Creates semantically meaningful chunks from content sources with
special handling for process flow diagrams using OCR.
"""

import os
import re
import json
import logging
import tempfile
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

# Import PDF processing libraries
import PyPDF2
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import io

# Import OCR
import easyocr

# Import langchain for text chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ChunkCreator")

@dataclass
class DiagramInfo:
    """Information about an extracted diagram"""
    image_data: bytes  # Raw image data
    ocr_text: str      # Extracted text from OCR
    page_num: int      # Page number in PDF
    width: int         # Image width
    height: int        # Image height
    diagram_type: str  # Type of diagram (flow, architecture, etc.)
    confidence: float  # Confidence score

class ChunkCreator:
    """
    Enhanced chunk creator with diagram OCR capabilities
    """
    
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        """Initialize with configurable chunk parameters"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize OCR
        try:
            self.ocr_reader = easyocr.Reader(['en'])
            logger.info("✅ EasyOCR initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ EasyOCR initialization failed: {e}")
            self.ocr_reader = None
    
    def create_chunks(self, content_source, consolidated_json_path: Optional[str] = None) -> List[Dict]:
        """
        Create chunks from content source and optionally from consolidated analysis
        
        Args:
            content_source: PDF file object or text content string
            consolidated_json_path: Optional path to consolidated_analysis.json
            
        Returns:
            List of chunks ready for vector store
        """
        all_chunks = []
        
        # Process main content source (PDF or text)
        if isinstance(content_source, str):
            # Process text content (from Confluence)
            text_content = content_source
            content_type = "text"
            
            # Use langchain text splitter for chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            # Create chunks
            langchain_docs = text_splitter.create_documents([text_content])
            
            # Convert to our format
            for i, doc in enumerate(langchain_docs):
                all_chunks.append({
                    'id': f'text_chunk_{i}',
                    'content': doc.page_content,
                    'metadata': {
                        'section_type': 'confluence_content',
                        'char_count': len(doc.page_content),
                        'word_count': len(doc.page_content.split()),
                        'source': 'confluence'
                    },
                    'source': 'confluence',
                })
                
        else:
            # Process PDF file
            try:
                # Extract diagrams and text from PDF
                pdf_chunks, diagram_chunks = self._process_pdf_with_diagrams(content_source)
                
                # Add both text and diagram chunks
                all_chunks.extend(pdf_chunks)
                all_chunks.extend(diagram_chunks)
                
                logger.info(f"Created {len(pdf_chunks)} text chunks and {len(diagram_chunks)} diagram chunks from PDF")
                
            except Exception as e:
                logger.error(f"Error processing PDF: {str(e)}")
                raise
        
        # Process consolidated analysis if provided
        if consolidated_json_path and os.path.exists(consolidated_json_path):
            try:
                logger.info(f"Processing consolidated analysis from: {consolidated_json_path}")
                
                # Load and process consolidated analysis
                with open(consolidated_json_path, 'r', encoding='utf-8') as f:
                    consolidated_data = json.load(f)
                
                # Create chunks from consolidated analysis
                biztalk_chunks = self._process_consolidated_analysis(consolidated_data)
                all_chunks.extend(biztalk_chunks)
                
                logger.info(f"Added {len(biztalk_chunks)} chunks from BizTalk analysis")
                
            except Exception as e:
                logger.error(f"Error processing consolidated analysis: {str(e)}")
                # Don't fail completely if only the consolidated analysis fails
                logger.warning("Continuing with only PDF/text chunks")
        
        logger.info(f"Created a total of {len(all_chunks)} chunks")
        return all_chunks
    
    def _process_pdf_with_diagrams(self, pdf_file) -> tuple:
        """
        Process PDF with special handling for diagrams
        
        Args:
            pdf_file: PDF file object
            
        Returns:
            Tuple of (text_chunks, diagram_chunks)
        """
        # Create a temporary file to work with
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            if hasattr(pdf_file, 'read'):
                temp_file.write(pdf_file.read())
                pdf_file.seek(0)  # Reset file pointer for future use
            else:
                with open(pdf_file, 'rb') as f:
                    temp_file.write(f.read())
            temp_file_path = temp_file.name
        
        try:
            logger.info(f"Processing PDF file: {os.path.basename(temp_file_path)}")
            
            # Extract text using PyMuPDF
            text_content = ""
            doc = fitz.open(temp_file_path)
            
            # Extract diagrams
            diagrams = self._extract_diagrams_from_pdf(doc)
            logger.info(f"Extracted {len(diagrams)} diagrams from PDF")
            
            # Extract text (excluding diagram areas)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text_content += page.get_text() + "\n\n"
            
            # Close the document
            doc.close()
            
            # Create text chunks using langchain
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            # Split text into chunks
            langchain_docs = text_splitter.create_documents([text_content])
            
            # Convert to our format
            text_chunks = []
            for i, doc in enumerate(langchain_docs):
                text_chunks.append({
                    'id': f'pdf_text_chunk_{i}',
                    'content': doc.page_content,
                    'metadata': {
                        'section_type': 'pdf_text',
                        'char_count': len(doc.page_content),
                        'word_count': len(doc.page_content.split()),
                        'source': 'pdf'
                    },
                    'source': 'pdf',
                })
            
            # Create diagram chunks
            diagram_chunks = self._create_diagram_chunks(diagrams)
            
            return text_chunks, diagram_chunks
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
    
    def _extract_diagrams_from_pdf(self, pdf_document) -> List[DiagramInfo]:
        """
        Extract diagrams from PDF document
        
        Args:
            pdf_document: PyMuPDF document
            
        Returns:
            List of DiagramInfo objects
        """
        diagrams = []
        
        # Process each page
        for page_idx in range(len(pdf_document)):
            page = pdf_document[page_idx]
            
            # Get images from page
            image_list = page.get_images(full=True)
            
            # Process each image
            for img_idx, img_info in enumerate(image_list):
                try:
                    # Get image
                    xref = img_info[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert to numpy array for processing
                    try:
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        image_np = np.array(pil_image)
                        
                        # Process image only if it's large enough (likely a diagram, not an icon)
                        if pil_image.width > 200 and pil_image.height > 200:
                            # Determine if it's likely a diagram
                            is_diagram, diagram_type, confidence = self._is_diagram(image_np)
                            
                            if is_diagram:
                                # Perform OCR on the image
                                ocr_text = self._perform_ocr(image_np)
                                
                                # Create diagram info
                                diagram_info = DiagramInfo(
                                    image_data=image_bytes,
                                    ocr_text=ocr_text,
                                    page_num=page_idx + 1,
                                    width=pil_image.width,
                                    height=pil_image.height,
                                    diagram_type=diagram_type,
                                    confidence=confidence
                                )
                                
                                diagrams.append(diagram_info)
                    except Exception as e:
                        logger.warning(f"Error processing image: {e}")
                        continue
                        
                except Exception as e:
                    logger.warning(f"Error extracting image: {e}")
                    continue
        
        return diagrams
    
    def _is_diagram(self, image_np) -> tuple:
        """
        Determine if an image is a diagram
        
        Args:
            image_np: Numpy array of image
            
        Returns:
            Tuple of (is_diagram, diagram_type, confidence)
        """
        # Convert to grayscale
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        # Basic heuristics for diagrams:
        
        # 1. Edge detection - diagrams tend to have many edges
        edges = cv2.Canny(gray, 50, 150)
        edge_percentage = np.count_nonzero(edges) / edges.size
        
        # 2. Line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=50, maxLineGap=10)
        line_count = 0 if lines is None else len(lines)
        
        # 3. Check for uniform areas (diagrams often have large uniform areas)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        uniform_percentage = np.count_nonzero(thresh) / thresh.size
        
        # Calculate overall confidence
        # High edge percentage + high line count + moderate uniform areas = likely diagram
        edge_score = min(edge_percentage * 5, 1.0)  # Scale to 0-1
        line_score = min(line_count / 50, 1.0)  # Scale to 0-1
        uniform_score = min(max(0.2, uniform_percentage), 0.8)  # Scale to 0.2-0.8
        
        confidence = (edge_score * 0.4) + (line_score * 0.4) + (uniform_score * 0.2)
        
        # Determine diagram type based on features
        diagram_type = "unknown"
        if line_count > 30:
            diagram_type = "flow_diagram"
        elif edge_percentage > 0.1 and uniform_percentage > 0.6:
            diagram_type = "architecture_diagram"
        elif edge_percentage < 0.05:
            diagram_type = "screenshot"
        
        # Decision threshold
        is_diagram = confidence > 0.5
        
        return is_diagram, diagram_type, confidence
    
    def _perform_ocr(self, image_np) -> str:
        """
        Perform OCR on an image
        
        Args:
            image_np: Numpy array of image
            
        Returns:
            OCR text
        """
        if self.ocr_reader is None:
            return "OCR not available"
        
        try:
            # Perform OCR
            result = self.ocr_reader.readtext(image_np)
            
            # Extract text
            ocr_text = ""
            for detection in result:
                text = detection[1]
                ocr_text += text + " "
            
            return ocr_text.strip()
        except Exception as e:
            logger.warning(f"OCR error: {e}")
            return "OCR failed"
    
    def _create_diagram_chunks(self, diagrams: List[DiagramInfo]) -> List[Dict]:
        """
        Create chunks from diagrams
        
        Args:
            diagrams: List of DiagramInfo objects
            
        Returns:
            List of diagram chunks
        """
        diagram_chunks = []
        
        for i, diagram in enumerate(diagrams):
            # Create diagram chunk with OCR text
            chunk_id = f'diagram_chunk_{i}'
            
            # Add OCR text as content
            content = f"Diagram on page {diagram.page_num}: {diagram.diagram_type.replace('_', ' ').title()}\n\n"
            content += diagram.ocr_text
            
            # Create chunk
            chunk = {
                'id': chunk_id,
                'content': content,
                'metadata': {
                    'section_type': 'diagram',
                    'diagram_type': diagram.diagram_type,
                    'page_num': diagram.page_num,
                    'width': diagram.width,
                    'height': diagram.height,
                    'confidence': diagram.confidence,
                    'char_count': len(content),
                    'word_count': len(content.split()),
                    'source': 'pdf_diagram',
                    'has_diagram': True,
                    'has_technical_diagrams': True,
                    'diagram_count': 1
                },
                'source': 'pdf_diagram',
                # Don't include image_data in metadata, just OCR text
            }
            
            diagram_chunks.append(chunk)
        
        return diagram_chunks
    
    def _process_consolidated_analysis(self, consolidated_data: Dict) -> List[Dict]:
        """
        Process consolidated analysis data
        
        Args:
            consolidated_data: Dictionary with consolidated BizTalk analysis
            
        Returns:
            List of chunks from consolidated analysis
        """
        chunks = []
        chunk_id = 0
        
        # Process Business Entities
        if consolidated_data.get("business_entities"):
            business_entities = consolidated_data["business_entities"]
            if business_entities:
                content = "Business Entities:\n"
                for entity in business_entities:
                    content += f"- {entity}\n"
                
                # Create simpler metadata structure without nested dictionaries
                metadata = {
                    'section_type': 'business_entities',
                    'char_count': len(content),
                    'word_count': len(content.split()),
                    'source': 'consolidated_analysis'
                }
                
                chunks.append({
                    'id': f'biztalk_chunk_{chunk_id}',
                    'content': content,
                    'metadata': metadata,
                    'source': 'biztalk_analysis',
                })
                chunk_id += 1
        
        # Process Business Rules
        if consolidated_data.get("business_rules"):
            business_rules = consolidated_data["business_rules"]
            if business_rules:
                content = "Business Rules:\n"
                for rule in business_rules:
                    content += f"- {rule}\n"
                
                metadata = {
                    'section_type': 'business_rules',
                    'char_count': len(content),
                    'word_count': len(content.split()),
                    'source': 'consolidated_analysis'
                }
                
                chunks.append({
                    'id': f'biztalk_chunk_{chunk_id}',
                    'content': content,
                    'metadata': metadata,
                    'source': 'biztalk_analysis',
                })
                chunk_id += 1
        
        # Process Transformations
        if consolidated_data.get("transformations"):
            transformations = consolidated_data["transformations"]
            if transformations:
                content = "Transformations:\n"
                for transform in transformations:
                    content += f"- {transform}\n"
                
                metadata = {
                    'section_type': 'transformations',
                    'char_count': len(content),
                    'word_count': len(content.split()),
                    'source': 'consolidated_analysis'
                }
                
                chunks.append({
                    'id': f'biztalk_chunk_{chunk_id}',
                    'content': content,
                    'metadata': metadata,
                    'source': 'biztalk_analysis',
                })
                chunk_id += 1
        
        # Process Integration Points
        if consolidated_data.get("integration_points"):
            integration_points = consolidated_data["integration_points"]
            if integration_points:
                content = "Integration Points:\n"
                for point in integration_points:
                    content += f"- {point}\n"
                
                metadata = {
                    'section_type': 'integration_points',
                    'char_count': len(content),
                    'word_count': len(content.split()),
                    'source': 'consolidated_analysis'
                }
                
                chunks.append({
                    'id': f'biztalk_chunk_{chunk_id}',
                    'content': content,
                    'metadata': metadata,
                    'source': 'biztalk_analysis',
                })
                chunk_id += 1
        
        # Process Message Formats
        if consolidated_data.get("message_formats"):
            message_formats = consolidated_data["message_formats"]
            if message_formats:
                content = "Message Formats:\n"
                for format_name in message_formats:
                    content += f"- {format_name}\n"
                
                metadata = {
                    'section_type': 'message_formats',
                    'char_count': len(content),
                    'word_count': len(content.split()),
                    'source': 'consolidated_analysis'
                }
                
                chunks.append({
                    'id': f'biztalk_chunk_{chunk_id}',
                    'content': content,
                    'metadata': metadata,
                    'source': 'biztalk_analysis',
                })
                chunk_id += 1
        
        # Process Technical Components
        if consolidated_data.get("technical_components"):
            tech_components = consolidated_data["technical_components"]
            for comp_type, components in tech_components.items():
                if components:
                    content = f"{comp_type.title()} Components:\n"
                    for component in components:
                        if component.get("summary"):
                            content += f"- {component['name']}: {component['summary']}\n"
                        else:
                            content += f"- {component['name']}\n"
                    
                    metadata = {
                        'section_type': f'technical_components_{comp_type}',
                        'char_count': len(content),
                        'word_count': len(content.split()),
                        'source': 'consolidated_analysis'
                    }
                    
                    chunks.append({
                        'id': f'biztalk_chunk_{chunk_id}',
                        'content': content,
                        'metadata': metadata,
                        'source': 'biztalk_analysis',
                    })
                    chunk_id += 1
        
        return chunks
    
    def save_chunks_for_debug(self, chunks: List[Dict], output_path: str):
        """Save chunks to file for debugging"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved debug chunks to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving debug chunks: {str(e)}")
            return False
    
    def chunks_to_langchain_documents(self, chunks: List[Dict]) -> List[Document]:
        """Convert chunks to langchain Document objects for vector store"""
        docs = []
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            
            # Create simplified metadata (no nested dictionaries)
            simple_metadata = {}
            for key, value in metadata.items():
                # Only include simple types
                if isinstance(value, (str, int, float, bool, type(None))):
                    simple_metadata[key] = value
            
            # Add source and ID
            simple_metadata['source'] = chunk.get('source', 'unknown')
            simple_metadata['id'] = chunk.get('id', f"chunk_{len(docs)}")
            
            docs.append(Document(
                page_content=chunk['content'],
                metadata=simple_metadata
            ))
        return docs


# Convenience functions

def create_chunks(
    content_source: Any, 
    consolidated_json_path: Optional[str] = None,
    debug_output_path: Optional[str] = None,
    chunk_size: int = 300,
    chunk_overlap: int = 50
) -> List[Dict]:
    """
    Convenience function to create chunks from content source and consolidated analysis
    
    Args:
        content_source: PDF file object or text content string
        consolidated_json_path: Optional path to consolidated_analysis.json
        debug_output_path: Optional path to save debug chunks
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List of chunks ready for vector store
    """
    creator = ChunkCreator(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = creator.create_chunks(content_source, consolidated_json_path)
    
    # Save debug chunks if requested
    if debug_output_path:
        creator.save_chunks_for_debug(chunks, debug_output_path)
    
    return chunks

def create_langchain_documents(
    content_source: Any, 
    consolidated_json_path: Optional[str] = None,
    debug_output_path: Optional[str] = None,
    chunk_size: int = 300,
    chunk_overlap: int = 50
) -> List[Document]:
    """
    Create langchain Document objects from content source and consolidated analysis
    
    Args:
        content_source: PDF file object or text content string
        consolidated_json_path: Optional path to consolidated_analysis.json
        debug_output_path: Optional path to save debug chunks
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List of langchain Document objects ready for vector store
    """
    creator = ChunkCreator(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = creator.create_chunks(content_source, consolidated_json_path)
    
    # Save debug chunks if requested
    if debug_output_path:
        creator.save_chunks_for_debug(chunks, debug_output_path)
    
    return creator.chunks_to_langchain_documents(chunks)


# Main execution
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create chunks from content source")
    parser.add_argument("--pdf", help="Path to PDF file")
    parser.add_argument("--text", help="Text content")
    parser.add_argument("--consolidated", help="Path to consolidated_analysis.json")
    parser.add_argument("--output", default="chunks.json", help="Output path for debug chunks")
    parser.add_argument("--chunk-size", type=int, default=300, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap in characters")
    args = parser.parse_args()
    
    if args.pdf:
        with open(args.pdf, "rb") as f:
            chunks = create_chunks(
                content_source=f,
                consolidated_json_path=args.consolidated,
                debug_output_path=args.output,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            )
        print(f"Created {len(chunks)} chunks from {args.pdf}")
    elif args.text:
        chunks = create_chunks(
            content_source=args.text,
            consolidated_json_path=args.consolidated,
            debug_output_path=args.output,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        print(f"Created {len(chunks)} chunks from text")
    else:
        print("No input provided. Use --pdf or --text")