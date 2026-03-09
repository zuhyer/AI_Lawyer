import os
import yaml
from pathlib import Path
from AI_Lawyer.utils.logging_setup import logger


def resolve_secret(
    value: str,
    secret_path: str = "/workspaces/AI_Lawyer/config/secret.yaml",
) -> str:
    """Resolve a '!secret KEY_NAME' reference with priority order.

    Resolution order:
    1. If value does NOT start with '!secret ' — return as-is.
    2. Extract KEY_NAME from '!secret KEY_NAME'.
    3. Check os.environ[KEY_NAME] — return if set (non-empty).
    4. Fall back to secret.yaml file.
    5. Log error and return '' if neither found (never raises in prod).

    Args:
        value: Raw config value, possibly a '!secret ...' reference.
        secret_path: Path to secret.yaml (optional fallback).

    Returns:
        Resolved secret string, or '' if not found.
    """
    if not isinstance(value, str) or not value.startswith("!secret "):
        return value or ""

    key = value.split(maxsplit=1)[1].strip()

    # Priority 1: environment variable
    env_val = os.environ.get(key, "")
    if env_val:
        logger.debug(f"Secret '{key}' resolved from environment variable")
        return env_val

    # Priority 2: secret.yaml
    secret_file = Path(secret_path)
    if secret_file.exists():
        try:
            with open(secret_file, "r") as f:
                secrets = yaml.safe_load(f) or {}
            if key in secrets and secrets[key]:
                logger.debug(f"Secret '{key}' resolved from secret.yaml")
                return str(secrets[key])
        except Exception as e:
            logger.warning(f"Failed to read secret.yaml: {e}")

    # Neither found
    logger.error(
        f"Secret '{key}' not found in environment or secret.yaml. "
        f"Set the environment variable {key} in your .env file."
    )
    return ""
