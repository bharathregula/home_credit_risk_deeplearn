from home_credit_risk.config import settings
from home_credit_risk.logger import logger


def add_numbers(a: int, b: int) -> int:
    return a + b


def connect_to_service() -> str:
    """Example function using our new config."""
    # To use a SecretStr, you have to explicitly call .get_secret_value()
    key = settings.api_key.get_secret_value()
    logger.debug(f"Using API key: {key}")
    return f"Connecting to {settings.app_env} with {settings.max_retries} retries."
