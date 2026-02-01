#!/usr/bin/env python3
"""
Quick helper to validate the resolved Groq API key in `config/secret.yaml`.
This script will:
 - resolve the key referenced in `config/config.yaml`
 - print a masked view and basic checks (empty / HF token detection)
 - optionally attempt to instantiate `ChatGroq` to validate authentication (commented out by default)

Usage:
  python check_groq_key.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.utils.secret_loader import resolve_secret
from AI_Lawyer.utils.logging_setup import logger

try:
    logger.info("Resolving LLM API key from config...")
    cfg = ConfigurationManager()
    llm_cfg = cfg.get_llm_config()
    raw = llm_cfg.api_key

    if isinstance(raw, str) and raw.startswith("!secret"):
        key_name = raw.split()[1]
        logger.info(f"Secret reference found: {key_name}")
        resolved = resolve_secret(raw)
    else:
        key_name = None
        resolved = raw

    if not resolved:
        logger.error("No API key resolved. Check `config/secret.yaml` and ensure the key exists.")
        raise SystemExit(2)

    masked = (resolved[:6] + "...") if len(resolved) > 6 else resolved
    logger.info(f"Resolved key (masked): {masked}")

    if resolved.startswith("hf_"):
        logger.warning("The resolved token looks like a Hugging Face token (prefix 'hf_').\n"
                       "Groq requires its own API key — replace this value in `config/secret.yaml` with your Groq API key.")

    # Optional: attempt to instantiate ChatGroq to validate auth. Uncomment to run live check (will make network call).
    # from langchain_groq import ChatGroq
    # try:
    #     logger.info("Attempting to initialize ChatGroq to validate the key (live check)...")
    #     _ = ChatGroq(model=llm_cfg.model, groq_api_key=resolved)
    #     logger.info("ChatGroq instance created successfully. Key is likely valid.")
    # except Exception as e:
    #     logger.exception(f"ChatGroq initialization failed: {e}")
    #     raise

    logger.info("Check complete. If you see a warning about 'hf_' or the key is empty, update `config/secret.yaml`.")

except Exception as exc:
    logger.exception(f"Key check failed: {exc}")
    raise
