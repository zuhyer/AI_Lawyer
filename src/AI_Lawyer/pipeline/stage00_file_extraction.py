"""
Stage 0: File Extraction Pipeline
Extracts text from various file formats (PDF, DOCX, TXT, images with OCR)
using FileExtractorConfig from config.yaml
"""

from pathlib import Path
from typing import Dict, List, Any

from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.components.file_extractor import FileExtractor
from AI_Lawyer.utils.logging_setup import logger


STAGE_NAME = "File Extraction Pipeline"


class FileExtractionPipeline:
    """Orchestrates file extraction using configured FileExtractor."""

    def __init__(self):
        """Initialize with config from ConfigurationManager."""
        self.config_manager = ConfigurationManager()
        self.file_extractor_config = self.config_manager.get_file_extractor_config()
        self.file_extractor = FileExtractor(config=self.file_extractor_config)

    def extract_from_directory(self, directory_path: str) -> Dict[str, str]:
        """
        Extract text from all supported files in a directory.
        
        Args:
            directory_path: Path to directory containing files
            
        Returns:
            Dictionary mapping filename -> extracted text
        """
        dir_path = Path(directory_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        supported_exts = tuple(f".{ext}" for ext in self.file_extractor_config.supported_formats)
        files = [f for f in dir_path.rglob("*") if f.suffix.lower() in supported_exts]
        
        logger.info(f"Found {len(files)} supported files in {directory_path}")
        
        if not files:
            logger.warning(f"No supported files found in {directory_path}")
            return {}

        return self.file_extractor.extract_batch([str(f) for f in files])

    def extract_from_list(self, file_list: List[Any]) -> Dict[str, str]:
        """
        Extract text from list of files.
        
        Args:
            file_list: List of file paths, UploadFile objects, or dicts
            
        Returns:
            Dictionary mapping filename -> extracted text
        """
        return self.file_extractor.extract_batch(file_list)

    def extract_single(self, file_path: str, file_name: str = None) -> str:
        """
        Extract text from a single file.
        
        Args:
            file_path: Path to the file
            file_name: Optional filename (for format detection)
            
        Returns:
            Extracted text
        """
        return self.file_extractor.extract_from_file(file_path, file_name)


def start_file_extraction(directory_path: str = None, file_list: List[Any] = None) -> FileExtractionPipeline:
    """
    Start file extraction pipeline.
    
    Args:
        directory_path: Optional path to extract files from directory
        file_list: Optional list of files to extract
        
    Returns:
        FileExtractionPipeline instance with results
    """
    try:
        logger.info(f"Starting {STAGE_NAME}")
        
        pipeline = FileExtractionPipeline()
        
        if directory_path:
            results = pipeline.extract_from_directory(directory_path)
            logger.info(f"Extracted from directory: {len(results)} files")
        elif file_list:
            results = pipeline.extract_from_list(file_list)
            logger.info(f"Extracted from list: {len(results)} files")
        else:
            logger.warning("No directory or file list provided")
            results = {}
        
        logger.info(f"{STAGE_NAME} completed successfully!")
        return pipeline

    except Exception as e:
        logger.exception(f"{STAGE_NAME} failed: {e}")
        raise


if __name__ == "__main__":
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<")
        
        # Example: extract from a directory
        pdf_dir = "artifacts/data/pdfs/"
        if Path(pdf_dir).exists():
            pipeline = start_file_extraction(directory_path=pdf_dir)
            logger.info(f">>>> stage {STAGE_NAME} Completed <<<<")
        else:
            logger.warning(f"PDF directory not found: {pdf_dir}")
            logger.info("To test: provide a valid directory path")

    except Exception as e:
        logger.exception(e)
        raise
