"""Launch the damped-RL finetuning run inside a guarded Isaac app.

``train_finetune.main`` does not start Kit itself, and nothing from ``pxr``,
``omni``, ``isaaclab.envs`` or ``lehome`` is importable until the app is up --
running the module directly fails with ``ModuleNotFoundError: No module named
'pxr'`` at env construction. It also must not use a bare ``AppLauncher``: Kit
runs non-daemon threads and ignores SIGTERM, so an unguarded launch leaves a
300%-CPU orphan on any failure. ``isaac_app`` closes in a ``finally`` and hard
-exits if Kit refuses to unwind.

Every argument is forwarded to ``train_finetune``, so the contract, the
reinitialised actor and the damping gate are all set there.

This is the experiment the repository was built for and it has never been run.
The contract gates it before anything expensive happens, and the watchdog holds
a PENDING verdict until it has enough evaluations to distinguish a trend from
the natural oscillation period.
"""

from __future__ import annotations

import sys

from lehome.real_damped_project.tasks.isaac_app import isaac_app


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    # Pull the device out for the launcher without consuming it: train_finetune
    # needs to see --sim_device too.
    device = "cuda"
    for i, a in enumerate(argv):
        if a == "--sim_device" and i + 1 < len(argv):
            device = argv[i + 1]

    with isaac_app(headless=True, enable_cameras=True, device=device):
        # Imported inside the block: the module chain reaches Isaac Lab and
        # LeHome, which require a live Kit runtime.
        from lehome.real_damped_project.train.train_finetune import main as train_main
        try:
            return train_main(argv)
        except BaseException:
            # isaac_app tears down and hard-exits in its finally, which happens
            # BEFORE the interpreter prints an uncaught traceback -- the first
            # attempt exited 1 with no diagnostic at all. Print it here, while
            # we still can.
            import traceback
            traceback.print_exc()
            sys.stdout.flush(); sys.stderr.flush()
            raise


if __name__ == "__main__":
    raise SystemExit(main())
