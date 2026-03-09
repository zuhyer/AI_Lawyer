from pathlib import Path
from AI_Lawyer.entity.config_entity import ChunkingConfig, DomainChunkingConfig, DataConfig
from AI_Lawyer.utils.logging_setup import logger
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Data_Loader:
    def __init__(self, config: DataConfig, domain: str = None, domain_config: DomainChunkingConfig = None):
        self.config = config
        self.domain = domain
        self.domain_config = domain_config
        self.pdf_dir = Path(self.config.pdf_directory)

    # =========================
    # Load Single File or Directory
    # =========================
    def load_single_file(self, file_path: str):

        documents = []

        try:
            file_path_obj = Path(file_path)

            if not file_path_obj.exists():
                logger.error(f"File/directory not found: {file_path}")
                return documents

            # Check if it's a directory (for templates)
            if file_path_obj.is_dir():
                logger.info(f"📁 Loading directory: {file_path}")
                # Load all supported files from directory
                for file_path in file_path_obj.rglob('*'):
                    if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.docx', '.txt']:
                        try:
                            loader = PDFPlumberLoader(str(file_path))
                            docs = loader.load()

                            # 🔥 Inject metadata
                            for doc in docs:
                                doc.metadata["domain"] = self.domain
                                doc.metadata["source_file"] = file_path.name

                            documents.extend(docs)
                            logger.info(f"✅ Loaded {file_path.name} | Domain={self.domain}")

                        except Exception as e:
                            logger.error(f"Error loading file {file_path}: {e}")
                return documents

            # Single file loading (existing logic)
            loader = PDFPlumberLoader(str(file_path_obj))
            docs = loader.load()

            # 🔥 Inject metadata
            for doc in docs:
                doc.metadata["domain"] = self.domain
                doc.metadata["source_file"] = file_path_obj.name

            documents.extend(docs)

            logger.info(
                f"✅ Loaded file | Domain={self.domain} | Pages={len(docs)}"
            )

        except Exception as e:
            logger.error(f"Error loading file/directory {file_path}: {e}")

        return documents

    # =========================
    # Load Directory PDFs
    # =========================
    def load_pdfs(self):

        documents = []

        for pdf_file in self.pdf_dir.glob("*.pdf"):

            try:
                loader = PDFPlumberLoader(str(pdf_file))
                docs = loader.load()

                # 🔥 Inject metadata
                for doc in docs:
                    doc.metadata["domain"] = self.domain
                    doc.metadata["source_file"] = pdf_file.name

                documents.extend(docs)

                logger.info(
                    f"✅ Loaded {pdf_file.name} | Domain={self.domain}"
                )

            except Exception as e:
                logger.error(
                    f"Error loading file: {pdf_file} | Error: {e}"
                )

        return documents

    # =========================
    # Main Loader Entry
    # =========================
    def main(self):

        if self.domain_config and self.domain_config.data_source:

            logger.info(
                f"📂 Domain '{self.domain}' → "
                f"{self.domain_config.data_source}"
            )

            return self.load_single_file(
                self.domain_config.data_source
            )

        return self.load_pdfs()


    

class Chunking_text:

    def __init__(self, config: ChunkingConfig,
                 domain_config: DomainChunkingConfig = None):

        self.config = config
        self.domain_config = domain_config

    # =========================
    # Create Chunks
    # =========================
    def create_chunks(self, documents):

        try:

            # 🔥 Template / full-doc bypass
            if self.domain_config and \
               self.domain_config.preserve_full_document:

                logger.info(
                    f"⚠️ Skipping chunking "
                    f"(Domain={self.domain_config.domain})"
                )

                return documents

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                add_start_index=self.config.add_start_index
            )

            chunks = splitter.split_documents(documents)

            logger.info(
                f"✂️ Chunks created: {len(chunks)}"
            )

            return chunks

        except Exception as e:
            logger.error(f"Chunking error: {e}")
            raise e

    # =========================
    def main(self, documents):
        return self.create_chunks(documents)



        


        
