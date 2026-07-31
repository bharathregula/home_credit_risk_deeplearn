"""
Pytest configuration.

**Why `OMP_NUM_THREADS` is pinned here.** LightGBM and PyTorch each ship their own
OpenMP runtime. When both are loaded into one process — which pytest does, since it
collects `test_gbt.py` and `test_nn.py` together — the two thread pools can collide and
the process dies with a segmentation fault inside a torch kernel (observed in
`BatchNorm1d` on macOS/arm64 with torch 2.13). Pinning to a single OpenMP thread avoids
it entirely.

This must run **before** either library is imported, which is why it lives in
`conftest.py` (pytest imports this first) rather than in a fixture or a test module.

It costs nothing here: every test in this suite runs on tiny synthetic fixtures where
thread-level parallelism is pure overhead. It does **not** affect real training runs,
which execute outside pytest and are free to use every core.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
