"""Tests for the intra-op thread-cap helpers and the policy that Steps 1 and 3
cap intra-op threads while Step 2 (training) does not.

These are deliberately SDK-free: they import only bench_common (which never
imports hoops_ai / torch) and read the source of the two scripts that must not
be imported here, so `pytest tests` runs without a HOOPS AI license.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench_common import (  # noqa: E402
    INTRAOP_ENV_VARS,
    apply_intraop_cap,
    resolve_intraop_threads,
)


@pytest.fixture(autouse=True)
def _clear_intraop_env(monkeypatch):
    """Isolate every test from the real environment: neither the new nor the
    legacy env var leaks in, and none of the five thread vars start set."""
    monkeypatch.delenv("HOOPS_AI_INTRAOP_THREADS", raising=False)
    monkeypatch.delenv("BENCH_INTRAOP_THREADS", raising=False)
    for var in INTRAOP_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------
# resolve_intraop_threads
# --------------------------------------------------------------------------
def test_default_is_one():
    assert resolve_intraop_threads(None) == 1


def test_cli_wins_over_everything(monkeypatch):
    monkeypatch.setenv("HOOPS_AI_INTRAOP_THREADS", "4")
    monkeypatch.setenv("BENCH_INTRAOP_THREADS", "8")
    assert resolve_intraop_threads(2) == 2


def test_cli_zero_is_honoured_not_treated_as_unset(monkeypatch):
    # 0 means "do not cap"; it must win over the env vars and not fall through
    # to the default of 1.
    monkeypatch.setenv("HOOPS_AI_INTRAOP_THREADS", "4")
    assert resolve_intraop_threads(0) == 0


def test_primary_env_var(monkeypatch):
    monkeypatch.setenv("HOOPS_AI_INTRAOP_THREADS", "3")
    assert resolve_intraop_threads(None) == 3


def test_primary_env_var_wins_over_legacy(monkeypatch):
    monkeypatch.setenv("HOOPS_AI_INTRAOP_THREADS", "3")
    monkeypatch.setenv("BENCH_INTRAOP_THREADS", "9")
    assert resolve_intraop_threads(None) == 3


def test_legacy_env_var_still_supported(monkeypatch):
    monkeypatch.setenv("BENCH_INTRAOP_THREADS", "5")
    assert resolve_intraop_threads(None) == 5


def test_primary_env_zero_disables(monkeypatch):
    # "0" is a non-empty string, so it is read and honoured as uncapped (0),
    # not skipped as if unset.
    monkeypatch.setenv("HOOPS_AI_INTRAOP_THREADS", "0")
    assert resolve_intraop_threads(None) == 0


def test_negative_cli_is_hard_error():
    with pytest.raises(SystemExit):
        resolve_intraop_threads(-1)


def test_negative_env_is_hard_error(monkeypatch):
    monkeypatch.setenv("HOOPS_AI_INTRAOP_THREADS", "-2")
    with pytest.raises(SystemExit):
        resolve_intraop_threads(None)


def test_non_integer_env_is_hard_error(monkeypatch):
    monkeypatch.setenv("HOOPS_AI_INTRAOP_THREADS", "lots")
    with pytest.raises(SystemExit):
        resolve_intraop_threads(None)


# --------------------------------------------------------------------------
# apply_intraop_cap
# --------------------------------------------------------------------------
def test_apply_sets_all_five_vars(monkeypatch):
    apply_intraop_cap(1)
    for var in INTRAOP_ENV_VARS:
        assert os.environ.get(var) == "1"


def test_apply_uses_the_given_value(monkeypatch):
    apply_intraop_cap(2)
    for var in INTRAOP_ENV_VARS:
        assert os.environ.get(var) == "2"


def test_apply_zero_is_a_noop(monkeypatch):
    apply_intraop_cap(0)
    for var in INTRAOP_ENV_VARS:
        assert var not in os.environ


# --------------------------------------------------------------------------
# Policy tripwires (source-level, so no SDK import is needed)
# --------------------------------------------------------------------------
def test_step2_never_applies_the_cap():
    src = (REPO_ROOT / "bench_step2_training.py").read_text(encoding="utf-8")
    assert "apply_intraop_cap" not in src, (
        "Step 2 (training) must NOT cap intra-op threads: a single-process CPU "
        "training run should use all cores, or the CPU-vs-GPU comparison breaks.")


def test_run_pipeline_passes_cap_to_steps_1_and_3_but_not_step_2():
    src = (REPO_ROOT / "run_pipeline.py").read_text(encoding="utf-8")

    def block(start_marker: str, end_marker: str) -> str:
        start = src.index(start_marker)
        end = src.index(end_marker, start)
        return src[start:end]

    step1 = block("step1 = [", "run_step(\"step 1")
    step2 = block("step2 = [", "run_step(\"step 2")
    step3 = block("step3 = [", "run_step(\"step 3")

    assert "--intraop-threads" in step1
    assert "--intraop-threads" in step3
    assert "--intraop-threads" not in step2, (
        "run_pipeline must not pass --intraop-threads to step 2 training.")


def test_run_pipeline_intraop_default_is_none():
    # Default None means "do not override": each step then falls back to
    # HOOPS_AI_INTRAOP_THREADS or its own default of 1.
    src = (REPO_ROOT / "run_pipeline.py").read_text(encoding="utf-8")
    assert "\"--intraop-threads\", type=int, default=None" in src


def test_run_pipeline_only_passes_cap_when_explicitly_set():
    # The flag must be appended conditionally, so omitting it on the CLI omits
    # it from the step 1 / step 3 subprocess commands.
    src = (REPO_ROOT / "run_pipeline.py").read_text(encoding="utf-8")

    def block(start_marker: str, end_marker: str) -> str:
        start = src.index(start_marker)
        end = src.index(end_marker, start)
        return src[start:end]

    step1 = block("step1 = [", "run_step(\"step 1")
    step3 = block("step3 = [", "run_step(\"step 3")
    for blk in (step1, step3):
        assert "if args.intraop_threads is not None:" in blk, (
            "run_pipeline must append --intraop-threads only when it is set.")
