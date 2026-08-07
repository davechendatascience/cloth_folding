"""Guaranteed-teardown launcher for Isaac Sim.

Kit runs non-daemon background threads. If the Python script raises, is
interrupted, or simply forgets ``simulation_app.close()``, **the process does
not exit** -- it sits at roughly 300% CPU indefinitely, and SIGTERM does not
reclaim it (``pkill`` is ignored; SIGKILL is required). Two such orphans
accumulated during development, one for 13 minutes, before being noticed.

That makes an unguarded ``AppLauncher`` unsafe in anything that can fail --
which is everything. Always launch through :func:`isaac_app`, which closes the
app in a ``finally`` and hard-exits if Kit still refuses to unwind.

Usage::

    with isaac_app(headless=True, device="cpu") as app:
        import lehome.tasks           # noqa: F401  (needs the kit runtime)
        ...

Nothing from ``pxr``, ``omni``, ``isaaclab.envs`` or ``lehome`` is importable
until the app is up, so those imports must happen *inside* the block.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from typing import Iterator


@contextlib.contextmanager
def isaac_app(
    headless: bool = True,
    enable_cameras: bool = True,
    device: str = "cpu",
    accept_eula: bool = True,
    hard_exit_on_hang: bool = True,
    close_timeout_s: float = 30.0,
) -> Iterator[object]:
    """Launch Isaac Sim, yield the app handle, and always tear it down.

    Args:
        headless: run without a GUI.
        enable_cameras: required for any task that reads camera sensors --
            LeHome's garment env does.
        device: ``"cpu"`` or ``"cuda"``. LeHome recommends ``cpu``; note
            ``docs/policy_eval.md`` frames that as a GUI-conflict workaround
            rather than a solver limitation.
        accept_eula: set ``OMNI_KIT_ACCEPT_EULA``. Without it Kit blocks on an
            interactive prompt and dies with "Unable to bootstrap inner kit
            kernel: EOF when reading a line" under any non-tty launch.
        hard_exit_on_hang: after ``close()``, bypass interpreter shutdown with
            ``os._exit``. Kit's threads can still stall a normal exit even
            after a clean close.
        close_timeout_s: watchdog deadline. ``app.close()`` can hang
            indefinitely, so a timer armed *before* the call force-exits if
            teardown does not finish in time.
    """
    if accept_eula:
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=headless, enable_cameras=enable_cameras, device=device)
    app = launcher.app
    failed = False
    try:
        yield app
    except BaseException:
        failed = True
        raise
    finally:
        code = 1 if failed else 0
        if hard_exit_on_hang:
            # `app.close()` can itself hang -- observed blocking for 26 minutes
            # at 154% CPU after an exception, so a plain
            # `close(); os._exit()` never reaches the exit. Arm the watchdog
            # *before* calling close so a stuck teardown cannot outlive it.
            def _force_exit() -> None:  # pragma: no cover - watchdog
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(code)

            watchdog = threading.Timer(close_timeout_s, _force_exit)
            watchdog.daemon = True
            watchdog.start()

        try:
            app.close()
        except Exception:  # pragma: no cover - teardown is best-effort
            pass
        if hard_exit_on_hang:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(code)


def preload_env() -> str:
    """The ``LD_PRELOAD`` value Isaac Sim needs on ARM64.

    Isaac Lab's installer prints this: torch's bundled libgomp must be loaded
    alongside the system one or Kit crashes on aarch64. The path it prints is
    stale after any torch reinstall, so resolve it at call time.
    """
    import torch

    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib", "libgomp.so.1")
    system = "/lib/aarch64-linux-gnu/libgomp.so.1"
    return ":".join(p for p in (system, torch_lib) if os.path.exists(p))
