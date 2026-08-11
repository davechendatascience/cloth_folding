"""Repair LeHome's throttled success checker. Active by default; LEHOME_FIX_CHECKER=0 disables.

`lehome/utils/success_checker_chanllege.py` decorates the success predicate:

    def step_interval(interval=50):
        def decorator(func):
            call_count = 0                       # closure state, created once
            def wrapper(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count % interval == 0:
                    return func(*args, **kwargs) # -> result dict
                else:
                    return False                 # -> bool

    @step_interval(interval=50)
    def success_checker_garment_fold(particle_object, garment_type): ...

Two defects:

1. **It returns `False` on 49 of 50 calls**, where callers expect the result
   dict. `_check_success` guards with `isinstance(result, dict)`, so those calls
   silently report "not successful" rather than "not evaluated".
2. **`call_count` never resets.** It lives in the decorator's closure, so it
   persists across episodes *and* across garment switches. Which step of each
   episode actually gets evaluated drifts arbitrarily through a run.

Together: a fold that completes and then relaxes within 50 steps can be missed
outright, and the phase at which any episode is sampled is effectively
arbitrary. This is why our episodes logged only ~12-36 checks over 600 steps at
inconsistent positions. The challenge winner hit the same problem and bypassed
it by calling `check_top_sleeve` / `check_pant_*` directly.

The repair unwraps the decorator and restores the real function, then optionally
re-throttles *correctly*: evaluate every `LEHOME_CHECK_INTERVAL` calls but return
the **cached last result** in between rather than `False`, with the counter and
cache reset per episode. Default interval is 1 (evaluate every call), which is
the honest setting; raise it only if the particle reads prove too slow.

Patching order matters: `garment_bi_v2` does `from ... import
success_checker_garment_fold`, which copies the reference at import time. This
must therefore run **before** `lehome.tasks.bedroom` is imported (it is -- our
launcher installs shims before calling their `main()`). The already-imported
binding is patched too, defensively.
"""

from __future__ import annotations

import os


def _unwrap(fn):
    """Recover the undecorated function from step_interval's closure."""
    for cell in (fn.__closure__ or ()):
        try:
            v = cell.cell_contents
        except ValueError:
            continue
        if callable(v) and getattr(v, "__name__", "") == "success_checker_garment_fold":
            return v
    return None


def install() -> bool:
    if os.environ.get("LEHOME_FIX_CHECKER", "1") == "0":
        return False

    import lehome.utils.success_checker_chanllege as sc

    wrapped = sc.success_checker_garment_fold
    real = _unwrap(wrapped)
    if real is None:
        print("[fix_success_checker] could not unwrap; leaving as-is", flush=True)
        return False

    interval = max(1, int(os.environ.get("LEHOME_CHECK_INTERVAL", "1")))
    state = {"n": 0, "last": None}

    def fixed(particle_object, garment_type: str):
        state["n"] += 1
        if interval == 1 or state["last"] is None or state["n"] % interval == 0:
            state["last"] = real(particle_object, garment_type)
        return state["last"]

    fixed.reset_counter = lambda: state.update(n=0, last=None)  # noqa: E731
    sc.success_checker_garment_fold = fixed

    # Patch any module that already copied the reference.
    import sys
    for name, mod in list(sys.modules.items()):
        if name.startswith("lehome.") and getattr(mod, "success_checker_garment_fold", None) is wrapped:
            setattr(mod, "success_checker_garment_fold", fixed)

    print(f"[fix_success_checker] ACTIVE - unthrottled (interval={interval}); "
          f"was evaluating 1 call in 50 with a counter that never reset", flush=True)
    return True
