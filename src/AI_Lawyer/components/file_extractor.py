import logging
from pathlib import Path
from typing import List, Dict, Any

import pdfplumber
from docx import Document
from PIL import Image
import pytesseract

from AI_Lawyer.entity.config_entity import FileExtractorConfig
from AI_Lawyer.utils.logging_setup import logger


class FileExtractor:
    """
    Extracts text from multiple file formats: PDF, DOCX, TXT, and images (with OCR).
    Uses configuration from FileExtractorConfig entity.
    """

    def __init__(self, config: FileExtractorConfig):
        """
        Initialize FileExtractor with config.
        
        Args:
            config: FileExtractorConfig containing OCR settings and supported formats
        """
        self.config = config
        if config.tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = config.tesseract_path
        self.ocr_lang = config.ocr_language

    def extract_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            text_pages = []
            with pdfplumber.open(file_path) as pdf:
                for idx, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    text_pages.append(page_text)
                    if self.config.log_extraction_details:
                        logger.debug(f"Extracted page {idx + 1} from {Path(file_path).name}")
            
            result = "\n\n---PAGE---\n\n".join(text_pages)
            logger.info(f"✓ PDF extracted: {len(result)} chars from {Path(file_path).name}")
            return result
        except Exception as e:
            logger.exception(f"✗ Failed to extract PDF: {file_path}")
            raise

    def extract_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            doc = Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            result = "\n".join(paragraphs)
            logger.info(f"✓ DOCX extracted: {len(result)} chars from {Path(file_path).name}")
            return result
        except Exception as e:
            logger.exception(f"✗ Failed to extract DOCX: {file_path}")
            raise

    def extract_txt(self, file_path: str) -> str:
        """Extract text from TXT file with encoding fallback."""
        try:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    result = f.read()
            except UnicodeDecodeError:
                logger.debug(f"UTF-8 decode failed for {file_path}, trying latin-1")
                with open(file_path, "r", encoding="latin-1") as f:
                    result = f.read()
            logger.info(f"✓ TXT extracted: {len(result)} chars from {Path(file_path).name}")
            return result
        except Exception as e:
            logger.exception(f"✗ Failed to read TXT: {file_path}")
            raise

    def extract_image_ocr(self, file_path: str) -> str:
        """Extract text from image using OCR."""
        if not self.config.ocr_enabled:
            logger.warning(f"OCR disabled in config, skipping: {file_path}")
            return ""
        
        try:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img, lang=self.config.ocr_language)
            logger.info(f"✓ OCR extracted: {len(text)} chars from {Path(file_path).name}")
            return text
        except Exception as e:
            logger.exception(f"✗ OCR failed on: {file_path}")
            raise

    def extract_from_file(self, file_path: str, file_name: str = None) -> str:
        """
        Detect file type and extract text accordingly.
        
        Args:
            file_path: Path to the file
            file_name: Optional filename (used to detect format)
            
        Returns:
            Extracted text content
        """
        if not file_name:
            file_name = Path(file_path).name
        
        ext = Path(file_name).suffix.lower().lstrip('.')
        
        if self.config.log_extraction_details:
            logger.debug(f"Extracting {file_name} as {ext}")

        if ext == "pdf":
            return self.extract_pdf(file_path)
        elif ext in ("docx", "doc"):
            return self.extract_docx(file_path)
        elif ext == "txt":
            return self.extract_txt(file_path)
        elif ext in ("png", "jpg", "jpeg", "bmp", "tiff", "tif"):
            return self.extract_image_ocr(file_path)
        else:
            msg = f"Unsupported file format: .{ext}. Supported: {', '.join(self.config.supported_formats)}"
            logger.warning(msg)
            raise ValueError(msg)

    def extract_batch(self, file_list: List[Any]) -> Dict[str, str]:
        """
        Extract text from multiple files.
        
        Args:
            file_list: List of file paths, UploadFile objects, or dicts with 'path' key
            
        Returns:
            Dictionary mapping filename -> extracted text
            Errors stored as "ERROR: <exception message>"
        """
        if not self.config.batch_processing:
            logger.warning("Batch processing disabled in config")
            return {}
        
        results: Dict[str, str] = {}
        
        for item in file_list:
            try:
                # Handle string/Path
                if isinstance(item, (str, Path)):
                    path = str(item)
                    name = Path(path).name
                    text = self.extract_from_file(path, name)
                    results[name] = text
                    continue

                # Handle UploadFile-like objects (FastAPI, Starlette, etc.)
                filename = getattr(item, "filename", None)
                fileobj = getattr(item, "file", None)
                
                if filename and fileobj and hasattr(fileobj, "read"):
                    content = fileobj.read()
                    tmp_path = Path("/tmp") / filename
                    tmp_path.write_bytes(content)
                    text = self.extract_from_file(str(tmp_path), filename)
                    results[filename] = text
                    try:
                        tmp_path.unlink()
                    except Exception as cleanup_err:
                        logger.warning(f"Failed to cleanup temp file: {tmp_path}")
                    continue

                # Handle dict with 'path' key
                if isinstance(item, dict) and "path" in item:
                    path = item["path"]
                    name = Path(path).name
                    text = self.extract_from_file(path, name)
                    results[name] = text
                    continue

                logger.warning(f"Unsupported item type in batch: {type(item)}")
                
            except Exception as e:
                # Graceful error handling: store error message instead of failing
                name = None
                if hasattr(item, "filename"):
                    name = item.filename
                elif isinstance(item, (str, Path)):
                    name = Path(item).name
                else:
                    name = str(item)
                
                error_msg = f"ERROR: {str(e)}"
                results[name] = error_msg
                logger.error(f"Error processing {name}: {e}")
        
        logger.info(f"Batch extraction complete: {len(results)} files processed")
        return results

