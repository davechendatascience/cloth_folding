"""Pre-registered run criteria and a watchdog that enforces them.

Written because every significant misjudgement in this project came from the
same place -- concluding too early from too little data -- and reminders do not
fix that. Each guard below maps to a specific incident:

=========================  =====================================================
guard                      the failure it prevents
=========================  =====================================================
``min_evals_before_verdict``  A GAE fix was declared successful at 50 iterations;
                           it diverged over 51-100. Twice.
``natural_period``         A log sampled every 6 iterations when the episode
                           period was 6.25 produced pure aliasing, reported as
                           a "steady decrease".
``baselines``              Three training runs had no baseline, so "is it
                           converging?" was unanswerable when asked.
``reachability_verified``  Those same runs optimised an objective an oracle
                           later proved unreachable (best J 1.51 vs a 0.02
                           threshold). The check costs seconds.
``stall_timeout_s``        Two Isaac processes hung 26 minutes at ~600% CPU.
                           Silence looked identical to progress.
``log_is_unbuffered``      A 400-iteration run was piped through ``tail`` and
                           was unobservable for its entire duration.
=========================  =====================================================

The contract is written and frozen *before* a run starts, and hashed so an
edit mid-run is detectable. The watchdog then reports verdicts against it
rather than against whatever story fits the numbers.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Verdict(str, Enum):
    PENDING = "PENDING"          # not enough data to say anything
    ON_TRACK = "ON_TRACK"
    SUCCESS = "SUCCESS"
    PLATEAU = "PLATEAU"
    DIVERGED = "DIVERGED"
    NAN = "NAN"
    STALLED = "STALLED"


@dataclass
class RunContract:
    """Criteria fixed before a run begins."""

    name: str
    primary_metric: str
    direction: str = "minimize"

    baselines: Dict[str, float] = field(default_factory=dict)
    """What trivial policies score. A result is uninterpretable without these,
    and they must be measured, not guessed."""

    success_threshold: Optional[float] = None
    """Primary metric value that counts as success."""

    must_beat_baseline: str = ""
    """Name of the baseline the run must beat to be considered working at all."""

    reachability_verified: bool = False
    """Has some policy (oracle/scripted/demo) been shown to reach the success
    threshold in this environment? Refuse to spend days if not."""

    min_evals_before_verdict: int = 10
    """No verdict beyond PENDING before this many evaluations."""

    natural_period: float = 1.0
    """Length of the intrinsic cycle in eval units (e.g. episode length /
    steps-per-eval). Trend windows must span at least two of these."""

    trend_window: int = 0
    """Evaluations per trend estimate. 0 => derived from natural_period."""

    patience_evals: int = 20
    """Evaluations without improvement before declaring PLATEAU."""

    divergence_factor: float = 2.0
    """Abort if the metric becomes this much worse than the must-beat baseline."""

    stall_timeout_s: float = 1800.0
    """No new evaluation within this long => STALLED."""

    log_is_unbuffered: bool = False
    """Confirms the run writes progress unbuffered to a file that can be read
    while it runs. A pipe through `tail` does not count."""

    notes: str = ""

    def __post_init__(self) -> None:
        if self.direction not in ("minimize", "maximize"):
            raise ValueError("direction must be minimize|maximize")
        if self.trend_window == 0:
            # Two full periods, so a cycle cannot masquerade as a trend.
            self.trend_window = max(4, int(math.ceil(2 * self.natural_period)))
        if self.trend_window < 2 * self.natural_period:
            raise ValueError(
                f"trend_window={self.trend_window} spans < 2 natural periods "
                f"({self.natural_period}); a periodic signal would alias into a trend"
            )
        if self.min_evals_before_verdict < self.trend_window:
            self.min_evals_before_verdict = self.trend_window

    # ------------------------------------------------------------------ gating

    def preflight(self) -> List[str]:
        """Reasons this run should not start. Empty list means go."""
        problems = []
        if not self.baselines:
            problems.append("no baselines measured -- results would be uninterpretable")
        if self.must_beat_baseline and self.must_beat_baseline not in self.baselines:
            problems.append(f"must_beat_baseline {self.must_beat_baseline!r} is not in baselines")
        if not self.reachability_verified:
            problems.append(
                "reachability not verified -- no policy has been shown to reach "
                "success_threshold in this environment"
            )
        if not self.log_is_unbuffered:
            problems.append("log not confirmed unbuffered/readable during the run")
        if self.success_threshold is None:
            problems.append("no success_threshold: success would be a judgement call")
        return problems

    def require_ready(self) -> None:
        problems = self.preflight()
        if problems:
            raise RuntimeError(
                "run contract is not ready:\n  - " + "\n  - ".join(problems)
            )

    # ------------------------------------------------------------- serialisation

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()[:16]

    def save(self, path) -> None:
        d = asdict(self)
        d["_digest"] = self.digest()
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def load(cls, path) -> "RunContract":
        with open(path) as f:
            d = json.load(f)
        stored = d.pop("_digest", None)
        c = cls(**d)
        if stored is not None and stored != c.digest():
            raise RuntimeError(
                f"contract digest mismatch (file {stored}, recomputed {c.digest()}): "
                "the contract was edited after being registered"
            )
        return c


class Watchdog:
    """Applies a :class:`RunContract` to a stream of evaluations."""

    def __init__(self, contract: RunContract, clock=time.time) -> None:
        self.c = contract
        self._clock = clock
        self.values: List[float] = []
        self.best: Optional[float] = None
        self.best_eval = -1
        self.last_update = clock()
        self.alerts: List[str] = []

    # ------------------------------------------------------------------ helpers

    def _better(self, a: float, b: float) -> bool:
        return a < b if self.c.direction == "minimize" else a > b

    def _worse_than(self, v: float, ref: float, factor: float) -> bool:
        if self.c.direction == "minimize":
            return v > ref * factor if ref > 0 else v > ref + abs(ref) * factor
        return v < ref / factor if ref > 0 else v < ref - abs(ref) * factor

    # -------------------------------------------------------------------- update

    def update(self, metrics: Dict[str, float]) -> Verdict:
        """Ingest one evaluation and return the current verdict."""
        if self.c.primary_metric not in metrics:
            raise KeyError(
                f"primary metric {self.c.primary_metric!r} missing from {sorted(metrics)}"
            )
        v = float(metrics[self.c.primary_metric])
        self.last_update = self._clock()

        if math.isnan(v) or math.isinf(v):
            self.alerts.append(f"eval {len(self.values)}: primary metric is {v}")
            return Verdict.NAN

        self.values.append(v)
        if self.best is None or self._better(v, self.best):
            self.best, self.best_eval = v, len(self.values) - 1

        n = len(self.values)

        # Divergence is checked immediately: it is a failure, not a conclusion,
        # and waiting for min_evals would waste the very time this exists to save.
        ref_name = self.c.must_beat_baseline
        if ref_name and ref_name in self.c.baselines:
            ref = self.c.baselines[ref_name]
            if self._worse_than(v, ref, self.c.divergence_factor):
                self.alerts.append(
                    f"eval {n-1}: {self.c.primary_metric}={v:.4g} is >{self.c.divergence_factor}x "
                    f"worse than baseline {ref_name}={ref:.4g}"
                )
                return Verdict.DIVERGED

        if self.c.success_threshold is not None and self._better(
            v, self.c.success_threshold
        ):
            return Verdict.SUCCESS

        # Everything below is a *judgement*, so it waits for enough data.
        if n < self.c.min_evals_before_verdict:
            return Verdict.PENDING

        if n - 1 - self.best_eval >= self.c.patience_evals:
            return Verdict.PLATEAU

        return Verdict.ON_TRACK

    # -------------------------------------------------------------------- trend

    def trend(self) -> Optional[float]:
        """Mean change per eval over the last ``trend_window`` evaluations.

        ``None`` until a full window exists -- a window shorter than two
        natural periods cannot distinguish a trend from the cycle.
        """
        w = self.c.trend_window
        if len(self.values) < w:
            return None
        window = self.values[-w:]
        half = w // 2
        return (sum(window[half:]) / len(window[half:])) - (
            sum(window[:half]) / len(window[:half])
        )

    def check_stall(self) -> Optional[Verdict]:
        """Silence is not success: call periodically even when no eval arrives."""
        if self._clock() - self.last_update > self.c.stall_timeout_s:
            self.alerts.append(
                f"no evaluation for {self._clock() - self.last_update:.0f}s "
                f"(limit {self.c.stall_timeout_s:.0f}s)"
            )
            return Verdict.STALLED
        return None

    # ------------------------------------------------------------------- report

    def report(self) -> str:
        n = len(self.values)
        t = self.trend()
        beats = ""
        if self.c.must_beat_baseline in self.c.baselines and self.best is not None:
            ref = self.c.baselines[self.c.must_beat_baseline]
            beats = f"  beats_{self.c.must_beat_baseline}={self._better(self.best, ref)}"
        return (
            f"evals={n} best={self.best if self.best is not None else float('nan'):.4g} "
            f"@{self.best_eval} trend={'n/a' if t is None else f'{t:+.4g}'}{beats}"
        )
