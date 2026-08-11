"""Launcher for LeHome's official ``scripts.eval``. Their ``main()`` is called
unmodified -- this only fixes how the process is started.

``python -m scripts.eval`` cannot work on aarch64. Two things collide:

* ``scripts/eval.py`` forces the multiprocessing start method to ``"spawn"`` at
  import time, before it imports ``isaaclab.app``.
* ``isaacsim.__init__.bootstrap_kernel()`` calls ``aarch_preload_checking()``,
  which starts a ``multiprocessing.Process`` **at import time** to verify
  ``LD_PRELOAD``. It does this unconditionally on ARM, whether or not
  ``LD_PRELOAD`` is already correct.

Under ``spawn`` the child re-imports the ``__main__`` module, which re-enters
``from isaaclab.app import AppLauncher`` and tries to start that Process again
from inside a process that is still bootstrapping::

    RuntimeError: An attempt has been made to start a new process before the
    current process has finished its bootstrapping phase.

x86_64 never sees this: ``aarch_preload_checking`` returns immediately there, so
nothing is spawned during import.

This module's top level imports nothing but ``multiprocessing``, so when the
spawn child re-imports it as ``__main__`` the import is a no-op and the child
proceeds normally.

Run it from the lehome-challenge checkout (their code resolves ``Assets/`` and
the particle config relative to the working directory), with that checkout on
``PYTHONPATH`` so the ``scripts`` package is importable -- see
``scripts/run_lehome_eval.sh``.
"""

import multiprocessing

if __name__ == "__main__":
    # Keep the default `fork` start method by making their forced switch to
    # `spawn` a no-op.
    #
    # `aarch_preload_checking()` runs at `isaacsim` import and starts a Process
    # whose target lives in `isaacsim`. Under `spawn` the child unpickles that
    # target, which re-imports `isaacsim`, which runs `bootstrap_kernel()`,
    # which spawns again -- unbounded recursion, so the very first import
    # fails. Under `fork` the child inherits the module and the check returns.
    #
    # Pre-importing `isaacsim` here under `fork` also clears the recursion, but
    # it loads Kit before AppLauncher has configured the environment and the
    # extensions then fail with "module 'warp' has no attribute 'context'".
    # Suppressing the start-method change instead leaves their import order
    # exactly as written, which is the order known to work on this machine.
    multiprocessing.set_start_method = lambda *a, **k: None

    from scripts.eval import main

    # Repair LeHome's throttled success checker before anything imports it.
    import fix_success_checker

    fix_success_checker.install()

    # Optional vision ablation, controlled by LEHOME_ABLATE. Import order
    # matters: scripts.eval must be imported first so the stock policies are
    # registered, and install() then replaces the `lerobot` entry. With the
    # variable unset this is a no-op and the official path runs untouched.
    import ablate_vision

    ablate_vision.install()

    # Optional check-point position logging, controlled by LEHOME_LOG_CP.
    import log_checkpoints

    log_checkpoints.install()

    # Optional joint state/action logging, controlled by LEHOME_LOG_ACT.
    import log_actions

    log_actions.install()

    # Optional photometric matching, controlled by LEHOME_PHOTOMATCH. Installed
    # last so it wraps whatever select_action the other shims left in place.
    import photometric_match

    photometric_match.install()

    # Optional open-loop horizon override, controlled by LEHOME_N_ACTION_STEPS.
    import set_action_horizon

    set_action_horizon.install()

    # Optional external WebSocket policy, controlled by LEHOME_WS. Installed
    # last so it replaces the registry entry outright.
    import remote_ws_policy

    remote_ws_policy.install()

    # Optional recorded-action playback, controlled by LEHOME_REPLAY_EPISODE.
    import replay_policy

    replay_policy.install()

    main()
