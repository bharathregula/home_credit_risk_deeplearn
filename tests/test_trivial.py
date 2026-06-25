from home_credit_risk import build_status_message
from home_credit_risk.trivial_test import add_numbers


def test_build_status_message() -> None:
    assert build_status_message("credit-risk") == "credit-risk is ready"


def test_add_numbers() -> None:
    assert add_numbers(1, 2) == 3
