"""
User Upload Processor Component
Processes uploaded documents: extracts text, chunks, and creates embeddings.
"""

from typing import List, Dict, Any
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.entity.config_entity import UserUploadProcessorConfig
from AI_Lawyer.utils.logging_setup import logger


class UserUploadProcessor:
    """
    Processes user-uploaded documents for hybrid RAG.
    Chunks text and prepares documents for embedding and temporary FAISS index.
    """

    def __init__(self, config: UserUploadProcessorConfig = None):
        """
        Initialize UserUploadProcessor with configuration.
        
        Args:
            config: Optional UserUploadProcessorConfig. If None, loads from ConfigurationManager
        """
        if config is None:
            config_manager = ConfigurationManager()
            config = config_manager.get_user_upload_processor_config()
        
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            add_start_index=config.add_start_index
        )
        
        logger.info(
            f"✅ UserUploadProcessor initialized - "
            f"chunk_size={config.chunk_size}, "
            f"overlap={config.chunk_overlap}, "
            f"max_file_size={config.max_upload_size_mb}MB"
        )

    def process_single_file(
        self, 
        filename: str, 
        extracted_text: str,
        file_type: str = "uploaded"
    ) -> List[Document]:
        """
        Process a single file's extracted text into chunks.
        
        Args:
            filename: Name of the source file
            extracted_text: Raw text content extracted from file
            file_type: Type of source (uploaded, legal_db, etc.)
            
        Returns:
            List of LangChain Document objects with metadata
            
        Raises:
            ValueError: If text is empty
        """
        if not extracted_text or not extracted_text.strip():
            logger.warning(f"⚠️  File '{filename}' contains no text to process")
            return []
        
        try:
            # Split text into chunks
            chunks = self.splitter.split_text(extracted_text)
            
            # Create Document objects with metadata
            documents = []
            for idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": filename,
                        "chunk_index": idx,
                        "chunk_size": len(chunk),
                        "file_type": file_type,
                        "total_chunks": len(chunks)
                    }
                )
                documents.append(doc)
            
            logger.info(
                f"✓ Processed '{filename}': {len(documents)} chunks "
                f"({len(extracted_text)} chars total)"
            )
            return documents
            
        except Exception as e:
            logger.error(f"✗ Error processing file '{filename}': {e}")
            raise

    def process_uploaded_files(
        self, 
        files_dict: Dict[str, str]
    ) -> List[Document]:
        """
        Process multiple uploaded files into chunks.
        
        Args:
            files_dict: Dictionary mapping filename -> extracted_text
                       (from FileExtractor.extract_batch())
            
        Returns:
            List of all Document objects from all files
        """
        if not files_dict:
            logger.warning("⚠️  No files provided for processing")
            return []
        
        all_documents = []
        
        for filename, text in files_dict.items():
            try:
                docs = self.process_single_file(filename, text, file_type="user_upload")
                all_documents.extend(docs)
            except Exception as e:
                logger.error(f"✗ Failed to process '{filename}': {e}")
                # Continue processing other files
                continue
        
        logger.info(
            f"✅ Processed {len(files_dict)} files → "
            f"{len(all_documents)} total chunks"
        )
        return all_documents

    def validate_file_size(self, file_size_bytes: int) -> bool:
        """
        Validate if file size is within limits.
        
        Args:
            file_size_bytes: Size of file in bytes
            
        Returns:
            True if valid, False otherwise
        """
        max_bytes = self.config.max_upload_size_mb * 1024 * 1024
        
        if file_size_bytes > max_bytes:
            logger.warning(
                f"⚠️  File size {file_size_bytes / 1024 / 1024:.2f}MB "
                f"exceeds limit of {self.config.max_upload_size_mb}MB"
            )
            return False
        
        return True

    def get_chunk_statistics(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about processed documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary with chunk statistics
        """
        if not documents:
            return {
                "total_documents": 0,
                "total_characters": 0,
                "average_chunk_size": 0,
                "files_represented": 0
            }
        
        sources = set(doc.metadata.get("source", "unknown") for doc in documents)
        total_chars = sum(len(doc.page_content) for doc in documents)
        avg_size = total_chars / len(documents) if documents else 0
        
        stats = {
            "total_documents": len(documents),
            "total_characters": total_chars,
            "average_chunk_size": avg_size,
            "files_represented": len(sources),
            "files": list(sources)
        }
        
        logger.info(f"📊 Chunk Statistics: {stats}")
        return stats
