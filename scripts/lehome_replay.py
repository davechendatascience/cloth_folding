"""Launcher for LeHome's `scripts.dataset_sim` (record / replay).

Same aarch64 problem as `scripts/eval.py`, same fix: `dataset_sim.py` forces the
multiprocessing start method to "spawn" before importing `isaaclab.app`, and
`isaacsim` starts a Process at import time, so under spawn the child re-imports
`isaacsim` and spawns again -- unbounded recursion on the first import. Keeping
`fork` avoids it. See `scripts/lehome_eval.py` for the full explanation.

Why replay matters: the SmolVLA checkpoint reproduces demonstration actions
offline to 0.0051 MSE, beating a persistence baseline 8.7x, yet is a no-op in
closed loop. Replaying the *recorded* actions through this same environment
separates the two remaining explanations -- if the demo actions fold the
garment here, the environment is sound and the fault is policy-in-the-loop; if
they do not, the environment or its initial state does not match what was
recorded.
"""

import multiprocessing

if __name__ == "__main__":
    multiprocessing.set_start_method = lambda *a, **k: None

    from scripts.dataset_sim import main

    main()
