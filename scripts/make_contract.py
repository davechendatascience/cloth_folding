"""Build a RunContract from measured files, never from remembered numbers.

The contract's whole value is that its criteria were fixed before the run and
grounded in measurement. Typing the baselines by hand quietly reintroduces the
failure it exists to prevent -- a number recalled slightly wrong, or an
optimistic threshold chosen after glancing at early results.

So this reads them:

  * ``--eval`` : eval_bc_in_sim.py output -> frozen / random baselines, and the
    BC policy's own J for reference.
  * ``--reachability`` : check_reachability.py output -> whether some policy has
    been shown to reach the success threshold in this environment.

It refuses to emit a contract that would fail preflight, and it refuses to
claim reachability the evidence does not support.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from lehome.real_damped_project.train.run_contract import RunContract  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--eval", required=True, help="eval_bc_in_sim.py --out JSON")
    p.add_argument("--reachability", required=True, help="check_reachability.py --out JSON")
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="ppo-finetune-top-long")
    p.add_argument("--episode_steps", type=int, default=300)
    p.add_argument("--steps_per_eval", type=int, default=64)
    p.add_argument("--patience", type=int, default=25)
    args = p.parse_args()

    ev = json.loads(Path(args.eval).read_text())
    rc = json.loads(Path(args.reachability).read_text())
    summary = ev["summary"]

    baselines = {m: float(summary[m]["J_min"]) for m in summary if m != "bc"}
    if not baselines:
        print("ERROR: eval JSON has no non-BC baselines", file=sys.stderr)
        return 2

    bc_j = float(summary["bc"]["J_min"])
    frozen_j = baselines.get("frozen")
    print(f"measured: bc J_min={bc_j:.4f}  baselines={ {k: round(v,4) for k,v in baselines.items()} }")

    if frozen_j is not None and bc_j >= frozen_j:
        print(
            f"\nREFUSING: BC (J_min={bc_j:.4f}) does not beat frozen "
            f"({frozen_j:.4f}). Finetuning would refine a policy that does "
            f"nothing. Get more data (the other three garment types take the "
            f"set from 83k to 265k frames) before spending RL compute.",
            file=sys.stderr,
        )
        return 3

    reach = bool(rc.get("reachability_verified", False))

    contract = RunContract(
        name=args.name,
        primary_metric="J_mean",
        direction="minimize",
        baselines=baselines,
        must_beat_baseline="frozen" if "frozen" in baselines else sorted(baselines)[0],
        success_threshold=0.0,          # LeHome's own predicate: J == 0
        reachability_verified=reach,
        natural_period=args.episode_steps / args.steps_per_eval,
        patience_evals=args.patience,
        divergence_factor=2.0,
        stall_timeout_s=3600.0,
        log_is_unbuffered=True,
        notes=(
            f"BC J_min={bc_j:.4f}; reachability from {Path(args.reachability).name} "
            f"({rc.get('note','')})"
        ),
    )

    problems = contract.preflight()
    if problems:
        print("REFUSING: contract would fail preflight:", file=sys.stderr)
        for s in problems:
            print(f"  - {s}", file=sys.stderr)
        return 4

    contract.save(args.out)
    print(f"wrote {args.out}  digest={contract.digest()}")
    print(f"  must beat {contract.must_beat_baseline}={baselines[contract.must_beat_baseline]:.4f}")
    print(f"  natural_period={contract.natural_period:.2f} evals -> "
          f"trend_window={contract.trend_window}, "
          f"min_evals_before_verdict={contract.min_evals_before_verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
