import logging
import sys

from home_credit_risk.config import settings


def setup_logging() -> logging.Logger:
    """Configure and return the root logger for the application."""

    # 1. Determine the log level based on your environment config
    # In production, you usually want INFO or WARNING. In dev, DEBUG.
    log_level = logging.DEBUG if settings.app_env == "development" else logging.INFO

    # 2. Define the log format
    # This includes timestamps, log level, the module name, and the actual message
    log_format = "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s"
    date_format = "%Y-%m-%m %H:%M:%S"

    # 3. Configure the root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            # Ensure logs stream cleanly to standard output (STDOUT)
            # This is critical for Docker containers and cloud providers like AWS/GCP
            logging.StreamHandler(sys.stdout)
        ],
    )

    # 4. Get a named logger for your specific package
    logger = logging.getLogger("home_credit_risk")
    return logger


# Initialize the logger instance to be imported across the app
logger = setup_logging()
