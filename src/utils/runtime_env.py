"""
Runtime configuration helpers for fair benchmarking.

Goal:
- Make it easy to run all pipelines under the same constraints (threads/device)
  without sprinkling ad-hoc environment logic across scripts.
"""

from __future__ import annotations

import os
from typing import Optional


def get_env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except Exception:
        return default


def get_benchmark_threads(default: Optional[int] = None) -> Optional[int]:
    """
    Threads used for fair benchmarking.
    Set by env var `MLFE_THREADS`.
    """
    return get_env_int("MLFE_THREADS", default=default)


def get_benchmark_device(default: str = "auto") -> str:
    """
    Device preference for fair benchmarking.
    Set by env var `MLFE_DEVICE` in {cpu,gpu,mps,auto}.
    """
    dev = (os.environ.get("MLFE_DEVICE") or default).strip().lower()
    if dev in {"cpu", "gpu", "mps", "auto"}:
        return dev
    return default


def configure_xgboost_booster_threads(booster) -> None:
    """
    Force XGBoost booster to use a fixed number of threads for inference.
    """
    threads = get_benchmark_threads()
    if threads is None or threads <= 0:
        return
    try:
        booster.set_param({"nthread": threads})
    except Exception:
        # Best-effort; keep default behavior if booster doesn't support it.
        return


def configure_tensorflow_runtime(tf) -> None:
    """
    Configure TensorFlow threads and device visibility.

    Notes:
    - Threads should be set as early as possible in the process.
    - On Apple Silicon, disabling GPU uses `tf.config.set_visible_devices([], "GPU")`.
    """
    threads = get_benchmark_threads()
    if threads is not None and threads > 0:
        try:
            tf.config.threading.set_intra_op_parallelism_threads(threads)
            tf.config.threading.set_inter_op_parallelism_threads(threads)
        except Exception:
            pass

    device = get_benchmark_device()
    if device == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
