from home_credit_risk import build_status_message
from home_credit_risk.logger import logger


def main() -> None:
    message = build_status_message("home-credit-risk-deeplearn")
    logger.info(message)
    print(message)


if __name__ == "__main__":
    main()
