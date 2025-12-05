import os 
from pathlib import Path
from AI_Lawyer.utils.common import read_yaml, create_directories
from AI_Lawyer.utils.logging_setup import *
from AI_Lawyer.entity.config_entity import DataConfig, ChunkingConfig, EmbeddingConfig, LLMConfig
from AI_Lawyer.constants import *

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        # ✅ This is now correctly placed inside __init__ and uses self
        create_directories([self.config['data']['root_dir']])

    def get_data_ingestion_config(self) -> DataConfig:
        config = self.config['data']

        create_directories([config['pdf_directory']])
        
        data_config = DataConfig(   
            root_dir=Path(config['root_dir']),
            pdf_directory=Path(config['pdf_directory']),
            source_url=config['source_url']  # ✅ This stays as a list
        )
        return data_config

    def get_chunking_config(self) -> ChunkingConfig:
        config = self.params['chunkingparams']

        chunking_config = ChunkingConfig(
            chunk_size = config['chunk_size'],
            chunk_overlap = config['chunk_overlap'],
            add_start_index = config['add_start_index']
        )

        return chunking_config
        
     
    def get_embeddings_config(self) -> EmbeddingConfig:
        config = self.config['embeddings']
        embedding_config = EmbeddingConfig(
            model = config['model'],
            vector_store = config['vector_store'],
            vector_store_path = config['vector_store_path'],
            api_key = config['api_key']
        )
        return embedding_config


    def get_llm_config(self) -> LLMConfig:
        config = self.config['llm']
        llm_config = LLMConfig(
            provider = config['provider'],
            model = config['model'],
            api_key = config['api_key']
        )
        return llm_config