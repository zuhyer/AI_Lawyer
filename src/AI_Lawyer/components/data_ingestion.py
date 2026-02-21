import requests
from pathlib import Path
from typing import List

from AI_Lawyer.entity.config_entity import DataConfig
from AI_Lawyer.utils.logging_setup import logger


class DataIngestion:
    """
    Data Ingestion Pipeline - Supports both URL-based and Local file modes.
    
    Modes:
    1. URL MODE (Legacy): Downloads files from URLs specified in config.yaml
    2. LOCAL DATA MODE: Scans local pdf_directory for supported file types
    
    The mode is automatically selected based on whether source_url is empty.
    """
    
    # Supported file formats for LOCAL DATA MODE
    SUPPORTED_FORMATS = {'.pdf', '.docx', '.txt'}
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.pdf_dir = Path(self.config.pdf_directory)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.file_paths: List[str] = []  # Collected file paths from either mode

    # ============================================================================
    # OLD URL MODE (Legacy) - Preserved for backward compatibility
    # ============================================================================
    
    def download_pdfs(self):
        """
        Downloads PDFs from URLs specified in config.source_url.
        
        Legacy method maintained for backward compatibility.
        Only executed when source_url is not empty.
        """
        if not self.config.source_url:
            logger.debug("No URLs configured. Skipping URL-based download.")
            return
            
        logger.info("🔗 URL MODE ACTIVATED: Downloading files from configured URLs...")
        
        for url in self.config.source_url:
            try:
                file_name = url.split("/")[-1]
                if "?" in file_name:
                    file_name = file_name.split("?")[0]  # Clean filename if there are query params

                save_path = self.pdf_dir / file_name

                # Check if already downloaded
                if save_path.exists():
                    logger.info(f"Already downloaded. Skipping: {save_path}")
                    self.file_paths.append(str(save_path))
                    continue

                response = requests.get(url, timeout=10)
                response.raise_for_status()  # Raises HTTPError for bad responses

                with open(save_path, 'wb') as f:
                    f.write(response.content)

                logger.info(f"✅ Successfully downloaded: {url} -> {save_path}")
                self.file_paths.append(str(save_path))

            except requests.exceptions.RequestException as e:
                logger.error(f"❌ FAILED to download {url}. Error: {e}")

    # ============================================================================
    # NEW LOCAL DATA MODE - Scans local directory for supported file types
    # ============================================================================
    
    def scan_local_files(self):
        """
        Scans the configured pdf_directory for supported file types.
        
        Supported formats: pdf, docx, txt
        
        Returns:
            List[str]: File paths of discovered files
        """
        logger.info("📁 LOCAL DATA MODE ACTIVATED: Scanning directory for local files...")
        logger.info(f"📍 Directory: {self.pdf_dir}")
        
        if not self.pdf_dir.exists():
            logger.warning(f"⚠️  Directory does not exist: {self.pdf_dir}")
            logger.info("Please ensure that artifacts/data/ contains your legal documents.")
            return
        
        discovered_files = []
        
        try:
            # Scan recursively for supported file types
            for file_path in self.pdf_dir.rglob('*'):
                if file_path.is_file():
                    file_suffix = file_path.suffix.lower()
                    
                    # Check if file extension is supported
                    if file_suffix in self.SUPPORTED_FORMATS:
                        discovered_files.append(str(file_path))
                        logger.info(f"✅ Found: {file_path.name}")
            
            if not discovered_files:
                logger.warning(
                    f"⚠️  No supported files found in {self.pdf_dir}\n"
                    f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
                )
            else:
                logger.info(f"📊 Total files discovered: {len(discovered_files)}")
            
            self.file_paths = discovered_files
            return discovered_files
            
        except Exception as e:
            logger.error(f"❌ Error scanning local files: {e}")
            return []

    # ============================================================================
    # MAIN ORCHESTRATION - Decides which mode to activate
    # ============================================================================
    
    def main(self):
        """
        Main orchestration method - Determines which ingestion mode to use.
        
        Decision logic:
        - If source_url is empty → LOCAL DATA MODE
        - If source_url has URLs → URL MODE
        """
        logger.info("=" * 70)
        logger.info("🚀 DATA INGESTION PIPELINE STARTED")
        logger.info("=" * 70)
        
        # Check if URLs are configured
        if self.config.source_url and len(self.config.source_url) > 0:
            logger.info("📡 Source URLs detected in configuration")
            self.download_pdfs()
        else:
            logger.info("📍 No source URLs detected - using LOCAL DATA MODE")
            self.scan_local_files()
        
        logger.info("=" * 70)
        logger.info(f"✅ DATA INGESTION COMPLETED - {len(self.file_paths)} files processed")
        logger.info("=" * 70)

