"""Supervise a long run against its pre-registered contract.

Emits one line per *event* on stdout, so it can be attached to a Monitor and
surface problems as they happen rather than at the post-mortem.

Covers the failure paths, not just the happy one. The rule learned the hard
way: **silence is not success.** A run that has hung, died, or gone NaN looks
exactly like a healthy one if you only grep for progress markers. So this
alerts on:

  * process death (exit without a completion marker)
  * log staleness (no new metric line within the contract's stall timeout)
  * NaN / divergence past the baseline
  * plateau past patience
  * GPU idle while the process claims to be training

Usage::

    python scripts/watchdog.py \
        --log runs/ft/train.log \
        --contract runs/ft/contract.json \
        --pattern 'train_ppo' \
        --metric-regex 'J=\\s*(?P<J>[-\\d.eE+]+)'
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from lehome.real_damped_project.train.run_contract import (  # noqa: E402
    RunContract,
    Verdict,
    Watchdog,
)


def gpu_util() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return float("nan")


def process_alive(pattern: str) -> bool:
    try:
        r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=10)
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--contract", required=True)
    p.add_argument("--pattern", required=True, help="pgrep pattern for the training process")
    p.add_argument("--metric-regex", required=True,
                   help="regex with named groups; must capture the primary metric")
    p.add_argument("--poll-s", type=float, default=30.0)
    p.add_argument("--gpu-idle-threshold", type=float, default=5.0)
    p.add_argument("--gpu-idle-strikes", type=int, default=4)
    args = p.parse_args()

    contract = RunContract.load(args.contract)
    problems = contract.preflight()
    if problems:
        for s in problems:
            print(f"PREFLIGHT-FAIL {s}", flush=True)
        print("ABORT contract not ready; run should not have started", flush=True)
        return 2

    print(f"WATCHDOG start contract={contract.name} digest={contract.digest()} "
          f"metric={contract.primary_metric} must_beat={contract.must_beat_baseline}", flush=True)

    dog = Watchdog(contract)
    rx = re.compile(args.metric_regex)
    log = Path(args.log)
    pos = 0
    last_verdict = None
    gpu_idle_run = 0
    seen_any = False

    while True:
        # ---- new metric lines ------------------------------------------
        if log.exists():
            with log.open() as f:
                f.seek(pos)
                for line in f:
                    m = rx.search(line)
                    if not m:
                        continue
                    try:
                        metrics = {k: float(v) for k, v in m.groupdict().items() if v is not None}
                    except ValueError:
                        continue
                    if contract.primary_metric not in metrics:
                        continue
                    seen_any = True
                    v = dog.update(metrics)
                    if v in (Verdict.NAN, Verdict.DIVERGED, Verdict.SUCCESS, Verdict.PLATEAU):
                        print(f"ALERT {v.value} {dog.report()}", flush=True)
                        for a in dog.alerts[-2:]:
                            print(f"ALERT   {a}", flush=True)
                        if v in (Verdict.NAN, Verdict.DIVERGED):
                            print("ABORT recommend killing the run", flush=True)
                            return 3
                        if v is Verdict.SUCCESS:
                            return 0
                    elif v != last_verdict:
                        print(f"STATUS {v.value} {dog.report()}", flush=True)
                    last_verdict = v
                pos = f.tell()

        # ---- liveness --------------------------------------------------
        alive = process_alive(args.pattern)
        if not alive:
            print(f"ALERT process matching {args.pattern!r} is gone. {dog.report()}", flush=True)
            return 4

        if seen_any and dog.check_stall() is Verdict.STALLED:
            print(f"ALERT STALLED {dog.alerts[-1]}", flush=True)
            print("ABORT no progress within the contract's stall timeout", flush=True)
            return 5

        g = gpu_util()
        if g == g and g < args.gpu_idle_threshold:
            gpu_idle_run += 1
            if gpu_idle_run == args.gpu_idle_strikes:
                print(f"ALERT GPU idle ({g:.0f}%) for {gpu_idle_run} polls "
                      f"while the process is alive -- possible deadlock", flush=True)
        else:
            gpu_idle_run = 0

        time.sleep(args.poll_s)


if __name__ == "__main__":
    raise SystemExit(main())
