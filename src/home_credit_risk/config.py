from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Defining the types gives you autocomplete and strict validation
    app_env: str = "production"  # Default value if not found in .env
    api_key: SecretStr  # hides key from accidental print statements
    max_retries: int = 3

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# Instantiate the settings once so they can be imported anywhere
settings = Settings()
