from pathlib import Path
from AI_Lawyer.entity.config_entity import ChunkingConfig, DomainChunkingConfig, DataConfig
from AI_Lawyer.utils.logging_setup import logger
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Data_Loader:
    def __init__(self, config: DataConfig, domain: str = None, domain_config: DomainChunkingConfig = None):
        """
        Initialize Data_Loader.
        
        Args:
            config: DataConfig instance
            domain: Optional domain name (e.g., 'legal_templates_db')
            domain_config: Optional DomainChunkingConfig for domain-specific loading
        """
        self.config = config
        self.domain = domain
        self.domain_config = domain_config
        self.pdf_dir = Path(self.config.pdf_directory)

    def load_single_file(self, file_path: str):
        """Load a single PDF file."""
        documents = []
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                logger.error(f"File not found: {file_path}")
                return documents
            
            loader = PDFPlumberLoader(str(file_path_obj))
            docs = loader.load()
            documents.extend(docs)
            logger.info(f"Successfully loaded single file: {file_path}")
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
        return documents

    def load_pdfs(self):
        """Load all PDFs from directory (generic loader)."""
        documents = []

        # Iterate through all PDF files in directory
        for pdf_file in self.pdf_dir.glob("*.pdf"):
            try:
                loader = PDFPlumberLoader(str(pdf_file))
                docs = loader.load()
                documents.extend(docs)

                logger.info(f"Successfully loaded: {pdf_file}")

            except Exception as e:
                logger.error(f"Error loading file: {pdf_file} | Error: {e}")

        return documents

    def main(self):
        """Main method with domain-aware logic."""
        # If domain config specifies a single data source, load only that file
        if self.domain_config and self.domain_config.data_source:
            logger.info(f"Loading single file for domain '{self.domain}' from: {self.domain_config.data_source}")
            return self.load_single_file(self.domain_config.data_source)
        
        # Default: load all PDFs from directory
        return self.load_pdfs()
    

class Chunking_text:
    def __init__(self, config: ChunkingConfig):
        self.config = config


    def create_chunks(self,documents):

        try:
            tex_spillter = RecursiveCharacterTextSplitter(
            chunk_size = self.config.chunk_size,
            chunk_overlap = self.config.chunk_overlap,
            add_start_index = self.config.add_start_index)

            text_chunks = tex_spillter.split_documents(documents)

            return text_chunks
        
        except Exception as e:
            logger.error(f"Error while chunking documents: {e}")
            raise e
        
    def main(self, documents):
     """
      Main method for pipeline compatibility.
     """
     return self.create_chunks(documents)
    

        


        
