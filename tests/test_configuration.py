import tempfile
import yaml
from pathlib import Path

from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.entity.config_entity import LLMConfig


def make_temp_config(tmp_path, llm_extra=None):
    base = {
        "data": {"root_dir": str(tmp_path), "pdf_directory": str(tmp_path), "source_url": []},
        "embeddings": {"model": "test", "vector_store": "faiss", "vector_store_path": "v"},
        "llm": {"provider": "none", "model": "m", "api_key": "k"},
        "file_extraction": {"supported_formats": [], "ocr_enabled": False, "tesseract_path": "", "ocr_language": "", "log_extraction_details": False, "batch_processing": False},
    }
    if llm_extra:
        base["llm"].update(llm_extra)
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(base, f)
    params = {"chunkingparams": {"chunk_size": 100, "chunk_overlap": 20, "add_start_index": True}}
    ppath = tmp_path / "params.yaml"
    with open(ppath, "w") as f:
        yaml.safe_dump(params, f)
    return str(path), str(ppath)


def test_llm_prompt_defaults(tmp_path):
    cfg_path, prm_path = make_temp_config(tmp_path)
    mgr = ConfigurationManager(config_filepath=cfg_path, params_filepath=prm_path)
    llm_cfg = mgr.get_llm_config()
    assert isinstance(llm_cfg, LLMConfig)
    assert llm_cfg.prompt_template == ""  # default blank


def test_llm_prompt_custom(tmp_path):
    custom = {"prompt_template": "Hello {question} - {context}"}
    cfg_path, prm_path = make_temp_config(tmp_path, llm_extra=custom)
    mgr = ConfigurationManager(config_filepath=cfg_path, params_filepath=prm_path)
    llm_cfg = mgr.get_llm_config()
    assert llm_cfg.prompt_template == custom["prompt_template"]
