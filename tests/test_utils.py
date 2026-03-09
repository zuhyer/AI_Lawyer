import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "../src"))

from AI_Lawyer.utils.secret_loader import resolve_secret
from AI_Lawyer.utils import logging_setup


def test_resolve_secret(tmp_path, monkeypatch):
    # environment variable takes precedence
    monkeypatch.setenv("MYKEY", "value_env")
    assert resolve_secret("!secret MYKEY") == "value_env"
    # remove env, fallback to file
    monkeypatch.delenv("MYKEY", raising=False)
    secrets_file = tmp_path / "secret.yaml"
    secrets_file.write_text("MYKEY: value_file")
    assert resolve_secret("!secret MYKEY", secret_path=str(secrets_file)) == "value_file"
    # missing returns empty string
    assert resolve_secret("!secret NONEXIST", secret_path=str(secrets_file)) == ""


def test_logger_creation(monkeypatch):
    # set environment to json
    monkeypatch.setenv("LOG_FORMAT", "json")
    logger = logging_setup._build_logger("test")
    assert logger.name == "test"
    # verify handler exists
    assert any(isinstance(h, logging_setup.logging.StreamHandler) for h in logger.handlers)
