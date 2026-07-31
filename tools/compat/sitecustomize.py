"""
Startup shim: restore ``importlib.abc.Traversable`` for Python 3.14.

``Traversable`` was deprecated in Python 3.12 and **removed** from ``importlib.abc`` in
3.14; it now lives only at ``importlib.resources.abc.Traversable``. mlflow 3.14.0 has
not caught up — ``mlflow/assistant/skill_installer.py`` still imports the old path, and
``mlflow.server.fastapi_app`` imports the assistant router unconditionally, so the
tracking **UI server** cannot start::

    mlflow ui → uvicorn loads mlflow.server.fastapi_app
              → mlflow.server.assistant.api
              → mlflow.assistant.skill_installer
              → from importlib.abc import Traversable   ImportError

Only the web server is affected; logging, ``best_runs()`` and the ``mlflow`` CLI
subcommands are pure client code and work fine without this.

Python imports ``sitecustomize`` automatically at interpreter startup for every process
that can see it on ``sys.path``, **including subprocesses** — which is why this is a
``sitecustomize`` module rather than a patch inside a launcher script. ``mlflow ui``
spawns the uvicorn server as a child process, so an in-process monkeypatch never reaches
the code that fails.

Usage (see README / CLAUDE.md)::

    PYTHONPATH=tools/compat uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

The ``hasattr`` guard makes this a no-op on Python ≤ 3.13 and the moment mlflow ships a
fix, so it is safe to leave in place; delete this directory once mlflow's minimum
supported version has the corrected import.
"""

import importlib.abc

if not hasattr(importlib.abc, "Traversable"):
    import importlib.resources.abc

    # Deliberate stdlib patch, narrowly scoped: re-expose a name the stdlib moved, so a
    # third-party import of the pre-3.14 path resolves instead of raising ImportError.
    importlib.abc.Traversable = importlib.resources.abc.Traversable  # type: ignore[attr-defined]
