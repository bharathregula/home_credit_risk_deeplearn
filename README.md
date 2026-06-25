# Home Credit Risk Deeplearn

This project is a small Python starter template for building and testing a credit-risk related package with linting, type checking, and automated tests.

## What you will find here

- A simple package structure under [src/home_credit_risk](src/home_credit_risk)
- Basic logging and configuration examples
- Automated tests with pytest
- Pre-commit checks for formatting, linting, and type safety

## Prerequisites

Make sure you have Python 3.14+ and uv installed.

- Install uv: https://docs.astral.sh/uv/getting-started/installation/
- Verify your Python version:

```bash
python --version
```

## First-time setup

From the project root, run:

```bash
uv sync --group dev
```

This creates the virtual environment and installs the project and development dependencies.

## Run the example app

```bash
uv run python src/home_credit_risk/main.py
```

You should see a message printed to the terminal.

## Run the tests

```bash
uv run pytest
```

## Run quality checks

```bash
uv run ruff check .
uv run ruff format --check .
unv run mypy src
```

## Pre-commit hooks

To run the local hooks before committing:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

## Project layout

```text
src/home_credit_risk/
├── __init__.py
├── config.py
├── logger.py
├── main.py
└── trivial_test.py

tests/
└── test_trivial.py
```

## Next steps

If you are new to the project, start by reading the package files in [src/home_credit_risk](src/home_credit_risk) and then add your own features and tests.
