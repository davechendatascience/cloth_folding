"""Graded fold quality from an official-eval log, instead of a binary rate.

`success` is one bit per episode, and at this stage of training it is 0 for
every episode -- which cannot distinguish "moved the garment most of the way"
from "did nothing at all". That is the same mistake as reading a training loss:
a measurement with no dynamic range in the regime you are actually in.

LeHome's checker already logs everything needed for a continuous score:

    dist(p[0], p[4]) = 26.72 <= 7.2  -> X
    dist(p[0], p[1]) = 16.59 >= 9.9  -> V

so each condition's *margin violation* is recoverable, and J is exactly the
repo's Lyapunov functional over those same five conditions:

    margin    = (d - t) for 'le',  (t - d) for 'ge'
    violation = max(margin, 0)                       # centimetres
    J         = sum(violations) / 10.0               # scale=10, equal weights

J == 0 is identical to LeHome's success predicate (verified over 300 configs),
so this is not a proxy for the metric -- it is the metric, before thresholding.

Reported per episode: the best (minimum) J reached, the final J, and the most
conditions simultaneously satisfied. Best-J says how close it ever came; final
J says whether it held the fold or undid it.

Caveat on absolute values: earlier J figures in this repo (frozen = 7.118) came
from a different harness with matched demo poses on one garment. These come from
the official protocol across 12 garments, so use them to compare checkpoints
against each other, not against that number.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict

import numpy as np

COND = re.compile(r"dist\(p\[(\d+)\], p\[(\d+)\]\)\s*=\s*([\d.]+)\s*(<=|>=)\s*([\d.]+)")
HEAD = re.compile(r"\[Success Check\] Garment type:")
EPIS = re.compile(r"Episode (\d+)/(\d+):.*Success=(True|False)")
GARM = re.compile(r"Evaluating:\s+(\S+)")


def main(path: str) -> None:
    garment, checks, cur = None, [], []
    episodes = []  # (garment, J_min, J_final, max_passed, success)

    for line in open(path, errors="ignore"):
        if HEAD.search(line):
            if cur:
                checks.append(cur)
            cur = []
            continue
        m = COND.search(line)
        if m:
            _, _, d, cmp_, t = m.groups()
            d, t = float(d), float(t)
            margin = (d - t) if cmp_ == "<=" else (t - d)
            cur.append(max(margin, 0.0))
            continue
        m = GARM.search(line)
        if m:
            garment = m.group(1)
            continue
        m = EPIS.search(line)
        if m:
            if cur:
                checks.append(cur)
                cur = []
            if checks:
                js = np.array([sum(c) / 10.0 for c in checks])
                passed = max(sum(1 for v in c if v <= 0) for c in checks)
                episodes.append((garment, float(js.min()), float(js[-1]),
                                 passed, m.group(3) == "True"))
            checks = []

    if not episodes:
        print("no episodes parsed")
        return

    def block(label, rows):
        if not rows:
            print(f"  {label:<8} (none)")
            return
        jmin = np.array([r[1] for r in rows])
        jend = np.array([r[2] for r in rows])
        pas = np.array([r[3] for r in rows])
        print(f"  {label:<8} n={len(rows):>2}  J_best {jmin.mean():6.3f} +- {jmin.std():5.3f}"
              f"  (min seen {jmin.min():6.3f})   J_final {jend.mean():6.3f}"
              f"   conds {pas.mean():.2f}/5 (best {pas.max()}/5)"
              f"   success {sum(r[4] for r in rows)}/{len(rows)}")

    print(f"{path.split('/')[-1]}   episodes={len(episodes)}")
    block("seen", [r for r in episodes if "Unseen" not in (r[0] or "")])
    block("unseen", [r for r in episodes if "Unseen" in (r[0] or "")])
    block("ALL", episodes)

    per = defaultdict(list)
    for r in episodes:
        per[r[0]].append(r[1])
    worst = sorted(per.items(), key=lambda kv: -np.mean(kv[1]))[:3]
    best = sorted(per.items(), key=lambda kv: np.mean(kv[1]))[:3]
    print("  best garments :", ", ".join(f"{k.replace('Top_Long_','')} {np.mean(v):.2f}" for k, v in best))
    print("  worst garments:", ", ".join(f"{k.replace('Top_Long_','')} {np.mean(v):.2f}" for k, v in worst))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
