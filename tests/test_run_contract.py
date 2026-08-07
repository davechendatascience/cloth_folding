"""Each test replays a real failure from this project and checks the guard fires.

If a guard is ever loosened, the corresponding incident is free to recur, so
these are written as regression tests against history rather than as generic
unit tests.
"""

import math

import pytest

from lehome.real_damped_project.train.run_contract import (
    RunContract,
    Verdict,
    Watchdog,
)


def contract(**kw):
    base = dict(
        name="test",
        primary_metric="J",
        direction="minimize",
        baselines={"frozen": 10.0, "random": 12.0},
        must_beat_baseline="frozen",
        success_threshold=0.02,
        reachability_verified=True,
        log_is_unbuffered=True,
        natural_period=1.0,
        min_evals_before_verdict=6,
        patience_evals=5,
    )
    base.update(kw)
    return RunContract(**base)


# ------------------------------------------------------------------ preflight


def test_preflight_blocks_missing_baselines():
    """Three runs happened with no baseline; 'is it converging?' was unanswerable."""
    c = contract(baselines={}, must_beat_baseline="")
    assert any("baseline" in p for p in c.preflight())
    with pytest.raises(RuntimeError, match="not ready"):
        c.require_ready()


def test_preflight_blocks_unverified_reachability():
    """Three runs optimised an objective the oracle later proved unreachable."""
    c = contract(reachability_verified=False)
    assert any("reachability" in p for p in c.preflight())


def test_preflight_blocks_buffered_logs():
    """A 400-iteration run was piped through `tail` and never observable."""
    c = contract(log_is_unbuffered=False)
    assert any("unbuffered" in p for p in c.preflight())


def test_preflight_blocks_missing_success_threshold():
    c = contract(success_threshold=None)
    assert any("success_threshold" in p for p in c.preflight())


def test_preflight_passes_when_ready():
    contract().require_ready()


def test_must_beat_baseline_must_exist():
    assert any("must_beat_baseline" in p for p in contract(must_beat_baseline="oracle").preflight())


# ------------------------------------------------- anti-early-conclusion guards


def test_no_verdict_before_minimum_evals():
    """The GAE fix was called successful at 50 iterations. It diverged at 51-100."""
    w = Watchdog(contract(min_evals_before_verdict=6))
    for i in range(5):
        assert w.update({"J": 9.0 - i * 0.1}) is Verdict.PENDING
    assert w.update({"J": 8.4}) is Verdict.ON_TRACK


def test_trend_window_must_span_two_natural_periods():
    """A 6-iteration sample of a 6.25-iteration cycle looked like a clean trend."""
    with pytest.raises(ValueError, match="alias"):
        RunContract(name="x", primary_metric="J", natural_period=6.25, trend_window=6)


def test_trend_window_defaults_to_two_periods():
    c = RunContract(name="x", primary_metric="J", natural_period=6.25)
    assert c.trend_window >= 2 * 6.25


def test_trend_is_none_until_a_full_window_exists():
    w = Watchdog(contract(natural_period=3.0))
    for i in range(w.c.trend_window - 1):
        w.update({"J": 5.0})
    assert w.trend() is None
    w.update({"J": 5.0})
    assert w.trend() is not None


def test_periodic_signal_does_not_register_as_a_trend():
    """The actual aliasing incident: a sawtooth read as monotone improvement."""
    period = 6
    w = Watchdog(contract(natural_period=float(period), min_evals_before_verdict=4))
    for i in range(4 * period):
        w.update({"J": 5.0 + (i % period)})  # pure cycle, zero drift
    t = w.trend()
    assert abs(t) < 1.0, f"cycle leaked into trend estimate: {t}"


# ------------------------------------------------------------- failure detection


def test_nan_is_caught_immediately():
    w = Watchdog(contract())
    assert w.update({"J": float("nan")}) is Verdict.NAN
    assert w.alerts


def test_divergence_is_caught_without_waiting_for_min_evals():
    """Divergence is a failure, not a conclusion -- waiting wastes the run."""
    w = Watchdog(contract(divergence_factor=2.0))
    v = w.update({"J": 25.0})  # > 2x the frozen baseline of 10
    assert v is Verdict.DIVERGED
    assert len(w.values) == 1, "should not need min_evals to abort"


def test_the_actual_divergence_that_happened():
    """J went 2.7 -> 19.7 while frozen scored ~3.6. That must abort."""
    w = Watchdog(contract(baselines={"frozen": 3.6}, must_beat_baseline="frozen"))
    assert w.update({"J": 2.7}) is not Verdict.DIVERGED
    assert w.update({"J": 19.7}) is Verdict.DIVERGED


def test_plateau_after_patience():
    w = Watchdog(contract(min_evals_before_verdict=3, patience_evals=4))
    w.update({"J": 5.0})
    for _ in range(6):
        v = w.update({"J": 6.0})
    assert v is Verdict.PLATEAU


def test_success_is_reported_when_threshold_met():
    w = Watchdog(contract())
    assert w.update({"J": 0.01}) is Verdict.SUCCESS


def test_stall_detected_when_no_evals_arrive():
    """Two Isaac processes hung 26 minutes; silence looked like progress."""
    t = [1000.0]
    w = Watchdog(contract(stall_timeout_s=60.0), clock=lambda: t[0])
    w.update({"J": 5.0})
    assert w.check_stall() is None
    t[0] += 61.0
    assert w.check_stall() is Verdict.STALLED
    assert w.alerts


def test_missing_primary_metric_is_an_error_not_a_silent_skip():
    w = Watchdog(contract())
    with pytest.raises(KeyError, match="J"):
        w.update({"reward": 1.0})


# ------------------------------------------------------------------- integrity


def test_contract_digest_detects_post_hoc_edits(tmp_path):
    """Criteria must not be movable after the numbers are in."""
    import json

    p = tmp_path / "c.json"
    c = contract()
    c.save(p)
    RunContract.load(p)  # round-trips fine

    d = json.loads(p.read_text())
    d["success_threshold"] = 99.0  # "well, 99 counts as success too"
    p.write_text(json.dumps(d))
    with pytest.raises(RuntimeError, match="edited after being registered"):
        RunContract.load(p)


def test_maximize_direction_works():
    c = contract(direction="maximize", success_threshold=100.0,
                 baselines={"frozen": 10.0}, must_beat_baseline="frozen")
    w = Watchdog(c)
    assert w.update({"J": 200.0}) is Verdict.SUCCESS
    w2 = Watchdog(c)
    assert w2.update({"J": 1.0}) is Verdict.DIVERGED  # 2x worse than baseline


def test_report_is_honest_before_any_data():
    w = Watchdog(contract())
    r = w.report()
    assert "evals=0" in r and "n/a" in r
