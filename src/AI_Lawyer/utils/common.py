import os 
import joblib
import yaml
import json 
import hashlib
from box.exceptions import BoxValueError
from typing import Any
from AI_Lawyer.utils.logging_setup import logger 
from ensure import ensure_annotations 
import base64
from box import ConfigBox
from pathlib import Path 

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads yaml file and returns its content as a ConfigBox.

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Raises:
        ValueError: If the YAML file is empty.
        Exception: For any other issues.

    Returns:
        ConfigBox: The parsed content of the YAML file.
    """
    try:
        with open(path_to_yaml, 'r', encoding='utf-8') as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"{path_to_yaml} : file loaded successfully")

            if not content:  # Check if content is empty
                raise ValueError("YAML file is empty")

            return ConfigBox(content)  # Return ConfigBox

    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directory : list ,  verbose = True ):
    """
    Args:
         path_to_directory (list) : list of path to directory 
    
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.

    """
    for path in path_to_directory:
        os.makedirs(path, exist_ok =True)  
        if verbose:
            logger.info(f"created directory at path: {path}")



@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")



@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())


@ensure_annotations
def compute_document_hash(text: str, file_path: str = None) -> str:
    """
    Compute hash of document content for deduplication.
    
    Args:
        text: Document text content
        file_path: Optional file path to include in hash
        
    Returns:
        Hexadecimal hash string
    """
    hash_input = text
    if file_path:
        hash_input = f"{file_path}|{text}"
    
    return hashlib.sha256(hash_input.encode()).hexdigest()


@ensure_annotations
def deduplicate_documents(documents: list) -> list:
    """
    Deduplicate documents based on content hash.
    Preserves order and keeps first occurrence.
    
    Args:
        documents: List of LangChain Document objects
        
    Returns:
        List of deduplicated documents
    """
    if not documents:
        return documents
    
    seen_hashes = set()
    unique_docs = []
    
    for doc in documents:
        # Extract text content
        text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        
        # Compute hash
        doc_hash = compute_document_hash(text)
        
        # Only add if not seen before
        if doc_hash not in seen_hashes:
            unique_docs.append(doc)
            seen_hashes.add(doc_hash)
        else:
            logger.debug(f"Duplicate document skipped (hash: {doc_hash[:8]}...)")
    
    if len(unique_docs) < len(documents):
        logger.info(f"Deduplicated: {len(documents)} → {len(unique_docs)} documents")
    
    return unique_docs


@ensure_annotations
def add_document_metadata(documents: list, domain: str, source_file: str = None) -> list:
    """
    Add or update metadata for documents (domain, source, etc.).
    Useful for template ingestion to preserve provenance.
    
    Args:
        documents: List of LangChain Document objects
        domain: Domain name (e.g., 'legal_templates_db')
        source_file: Optional source file path
        
    Returns:
        List of documents with updated metadata
    """
    for doc in documents:
        if not hasattr(doc, 'metadata'):
            doc.metadata = {}
        
        doc.metadata['domain'] = domain
        if source_file:
            doc.metadata['source_file'] = source_file
        doc.metadata['ingestion_type'] = 'full_document' if doc.metadata.get('preserve_full') else 'chunked'
    
    logger.info(f"Added metadata to {len(documents)} documents (domain={domain})")
    return documents 