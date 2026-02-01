"""
File Extraction Component
Provides a clean interface to FileExtractor configured from YAML/entity
"""

from typing import Dict, List, Any
from pathlib import Path

from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.components.file_extractor import FileExtractor
from AI_Lawyer.entity.config_entity import FileExtractorConfig
from AI_Lawyer.utils.logging_setup import logger


class FileExtractionComponent:
    """
    High-level component for file extraction.
    Integrates FileExtractor with ConfigurationManager for config-driven behavior.
    """

    def __init__(self, config: FileExtractorConfig = None):
        """
        Initialize with optional custom config (defaults to ConfigurationManager).
        
        Args:
            config: Optional FileExtractorConfig. If None, loads from ConfigurationManager
        """
        if config is None:
            config_manager = ConfigurationManager()
            config = config_manager.get_file_extractor_config()
        
        self.config = config
        self.extractor = FileExtractor(config=config)
        logger.info(f"FileExtractionComponent initialized with formats: {config.supported_formats}")

    def extract(self, file_path: str) -> str:
        """
        Extract text from a single file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Extracted text content
        """
        return self.extractor.extract_from_file(file_path)

    def extract_multiple(self, file_paths: List[str]) -> Dict[str, str]:
        """
        Extract text from multiple files.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Dictionary mapping filename -> text
        """
        return self.extractor.extract_batch(file_paths)

    def extract_from_uploads(self, upload_list: List[Any]) -> Dict[str, str]:
        """
        Extract text from uploaded files (FastAPI UploadFile objects).
        
        Args:
            upload_list: List of UploadFile-like objects
            
        Returns:
            Dictionary mapping filename -> text
        """
        return self.extractor.extract_batch(upload_list)

    def extract_directory(self, directory: str) -> Dict[str, str]:
        """
        Extract text from all supported files in a directory.
        
        Args:
            directory: Path to directory
            
        Returns:
            Dictionary mapping filename -> text
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Get supported file extensions
        supported_exts = tuple(f".{ext}" for ext in self.config.supported_formats)
        files = [str(f) for f in dir_path.rglob("*") if f.suffix.lower() in supported_exts]
        
        if not files:
            logger.warning(f"No supported files found in {directory}")
            return {}

        logger.info(f"Found {len(files)} files to extract from {directory}")
        return self.extract_multiple(files)

    def get_config(self) -> FileExtractorConfig:
        """Get the current configuration."""
        return self.config

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return self.config.supported_formats

    def is_ocr_enabled(self) -> bool:
        """Check if OCR is enabled."""
        return self.config.ocr_enabled
