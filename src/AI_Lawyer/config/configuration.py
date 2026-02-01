import os 
from pathlib import Path
from AI_Lawyer.utils.common import read_yaml, create_directories
from AI_Lawyer.utils.logging_setup import *
from AI_Lawyer.entity.config_entity import (
    DataConfig, 
    ChunkingConfig, 
    EmbeddingConfig, 
    LLMConfig, 
    FileExtractorConfig,
    UserUploadProcessorConfig
)
from AI_Lawyer.constants import *


class ConfigurationManager:
    """Configuration manager for AI Lawyer application."""

    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        """
        Initialize configuration manager.
        
        Args:
            config_filepath: Path to config.yaml
            params_filepath: Path to params.yaml
        """
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config['data']['root_dir']])

    def get_data_ingestion_config(self) -> DataConfig:
        """Get data ingestion configuration."""
        config = self.config['data']
        create_directories([config['pdf_directory']])
        
        data_config = DataConfig(   
            root_dir=Path(config['root_dir']),
            pdf_directory=Path(config['pdf_directory']),
            source_url=config['source_url']
        )
        return data_config

    def get_chunking_config(self) -> ChunkingConfig:
        """Get text chunking configuration."""
        config = self.params['chunkingparams']

        chunking_config = ChunkingConfig(
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap'],
            add_start_index=config['add_start_index']
        )
        return chunking_config
        
    def get_embeddings_config(self) -> EmbeddingConfig:
        """Get embeddings configuration."""
        config = self.config['embeddings']
        embedding_config = EmbeddingConfig(
            model=config['model'],
            vector_store=config['vector_store'],
            vector_store_path=config['vector_store_path'],
            api_key=config['api_key']
        )
        return embedding_config

    def get_llm_config(self) -> LLMConfig:
        """Get LLM configuration."""
        config = self.config['llm']
        llm_config = LLMConfig(
            provider=config['provider'],
            model=config['model'],
            api_key=config['api_key']
        )
        return llm_config

    def get_file_extractor_config(self) -> FileExtractorConfig:
        """Get file extraction configuration."""
        config = self.config['file_extraction']
        file_extractor_config = FileExtractorConfig(
            supported_formats=config['supported_formats'],
            ocr_enabled=config['ocr_enabled'],
            tesseract_path=config['tesseract_path'],
            ocr_language=config['ocr_language'],
            log_extraction_details=config['log_extraction_details'],
            batch_processing=config['batch_processing']
        )
        return file_extractor_config

    def get_user_upload_processor_config(self) -> UserUploadProcessorConfig:
        """Get user upload processor configuration."""
        config = self.config.get('user_upload_processor', {})
        params = self.params.get('user_upload_params', {})
        
        user_upload_config = UserUploadProcessorConfig(
            chunk_size=params.get('chunk_size', 1000),
            chunk_overlap=params.get('chunk_overlap', 200),
            add_start_index=params.get('add_start_index', True),
            max_upload_size_mb=config.get('max_upload_size_mb', 50),
            temp_index_ttl_seconds=config.get('temp_index_ttl_seconds', 3600)
        )
        return user_upload_config