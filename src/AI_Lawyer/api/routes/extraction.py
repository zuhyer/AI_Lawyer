"""File extraction endpoint."""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import List, Optional
import logging

from AI_Lawyer.api.models.requests import ExtractionRequest
from AI_Lawyer.api.models.responses import ExtractionResponse, ErrorResponse
from AI_Lawyer.components.extraction_component import FileExtractionComponent
from AI_Lawyer.utils.logging_setup import logger

router = APIRouter(prefix="/extraction", tags=["extraction"])

# Initialize extraction component
extraction_component = None


def get_extraction_component():
    """Lazy load extraction component."""
    global extraction_component
    if extraction_component is None:
        extraction_component = FileExtractionComponent()
    return extraction_component


@router.post("/extract", response_model=ExtractionResponse)
async def extract_files(
    files: List[UploadFile] = File(..., description="Files to extract text from")
):
    """
    Extract text from uploaded files.
    
    Supports: PDF, DOCX, TXT, PNG, JPG, JPEG, BMP, TIFF
    
    - **files**: List of files to extract (required)
    
    Returns extracted text mapped by filename.
    """
    try:
        if not files:
            raise ValueError("No files provided")
        
        logger.info(f"Extracting text from {len(files)} uploaded files")
        
        component = get_extraction_component()
        results = component.extract_from_uploads(files)
        
        # Separate successful extractions from errors
        data = {}
        errors = {}
        
        for filename, content in results.items():
            if isinstance(content, str) and content.startswith("ERROR:"):
                errors[filename] = content
            else:
                data[filename] = content
        
        success = len(errors) == 0
        message = f"Extracted from {len(data)} files" + (f" with {len(errors)} errors" if errors else "")
        
        return ExtractionResponse(
            success=success,
            message=message,
            data=data,
            errors=errors,
            file_count=len(files)
        )
        
    except Exception as e:
        logger.exception(f"File extraction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {str(e)}"
        )


@router.post("/extract-path", response_model=ExtractionResponse)
async def extract_from_path(request: ExtractionRequest):
    """
    Extract text from file(s) at specified path(s).
    
    Can extract from:
    - Single file: `file_path`
    - Multiple files: `file_paths` (list)
    - Directory: `directory_path`
    """
    try:
        component = get_extraction_component()
        
        if request.file_path:
            logger.info(f"Extracting from single file: {request.file_path}")
            text = component.extract(request.file_path)
            return ExtractionResponse(
                success=True,
                message="Successfully extracted",
                data={request.file_path: text},
                file_count=1
            )
        
        elif request.file_paths:
            logger.info(f"Extracting from {len(request.file_paths)} files")
            results = component.extract_multiple(request.file_paths)
            
            data = {}
            errors = {}
            for filename, content in results.items():
                if isinstance(content, str) and content.startswith("ERROR:"):
                    errors[filename] = content
                else:
                    data[filename] = content
            
            return ExtractionResponse(
                success=len(errors) == 0,
                message=f"Extracted from {len(data)} files" + (f" with {len(errors)} errors" if errors else ""),
                data=data,
                errors=errors,
                file_count=len(request.file_paths)
            )
        
        elif request.directory_path:
            logger.info(f"Extracting from directory: {request.directory_path}")
            results = component.extract_directory(request.directory_path)
            
            data = {}
            errors = {}
            for filename, content in results.items():
                if isinstance(content, str) and content.startswith("ERROR:"):
                    errors[filename] = content
                else:
                    data[filename] = content
            
            return ExtractionResponse(
                success=len(errors) == 0,
                message=f"Extracted from {len(data)} files" + (f" with {len(errors)} errors" if errors else ""),
                data=data,
                errors=errors,
                file_count=len(data) + len(errors)
            )
        
        else:
            raise ValueError("No file_path, file_paths, or directory_path provided")
            
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Path extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@router.get("/supported-formats", response_model=dict)
async def get_supported_formats():
    """Get list of supported file formats for extraction."""
    try:
        component = get_extraction_component()
        formats = component.get_supported_formats()
        ocr_enabled = component.is_ocr_enabled()
        
        return {
            "supported_formats": formats,
            "ocr_enabled": ocr_enabled,
            "total_formats": len(formats)
        }
    except Exception as e:
        logger.exception("Failed to get supported formats")
        raise HTTPException(status_code=500, detail=str(e))
