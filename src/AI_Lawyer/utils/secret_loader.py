import yaml
from pathlib import Path
from AI_Lawyer.utils.logging_setup import logger


def resolve_secret(value: str, secret_path="/workspaces/AI_Lawyer/config/secret.yaml"):
    """
    Resolves values like "!secret KEY_NAME" from secret.yaml
    Example:
        api_key: "!secret Gemini_API_Key"
    """

    # Check if the value is a secret reference
    if isinstance(value, str) and value.startswith("!secret"):
        key = value.split()[1]  # extract KEY_NAME

        secret_file = Path(secret_path)

        if not secret_file.exists():
            raise FileNotFoundError(f"Secret file not found: {secret_file}")

        try:
            with open(secret_file, "r") as file:
                secrets = yaml.safe_load(file)

            if key not in secrets:
                raise KeyError(f"Secret key '{key}' not found in secret.yaml")

            return secrets[key]

        except Exception as e:
            logger.error(f"Error loading secret key '{key}': {e}")
            raise e

    # If value is normal (not secret reference)
    return value