from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Defining the types gives you autocomplete and strict validation
    app_env: str = "production"  # Default value if not found in .env
    api_key: SecretStr  # hides key from accidental print statements
    max_retries: int = 3

    # Experiment tracking. The default is a local SQLite store in the repo root
    # (gitignored); point MLFLOW_TRACKING_URI at a server to share runs. MLflow 3
    # retired the plain-directory file store, so this must be a database URI.
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "home-credit-risk"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# Instantiate the settings once so they can be imported anywhere
settings = Settings()
