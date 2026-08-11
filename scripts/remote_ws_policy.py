"""Drive LeHome's eval from Larchenko's winning policy server. LEHOME_WS=ws://...

Why: our SmolVLA scores 0/24 in closed loop while imitating demonstrations well
offline (0.0036 MSE, 12.3x better than persistence) and while demo replay folds
the garment in this very environment (J 7.27 -> 0.00). That isolates the fault to
policy-in-the-loop, but it cannot tell us whether our *harness* is subtly wrong.

Running the LeHome Challenge 2026 winner (1st of 62, 74.5% on long tops) through
the same harness settles it. If it folds, the harness is correct and the deficit
is ours. If it also scores zero, something in our evaluation is still broken and
every conclusion from it is suspect.

The server (`scripts/serve.py` in lehome_solution) speaks a stateless WebSocket
`infer_chunk` protocol and expects exactly LeHome's native observation keys, so
the env output forwards almost unchanged:

    -> {"type": "infer_chunk", "session_id": ..., "observation.state": [...],
        "observation.images.top_rgb": {"base64":..., "dtype":..., "shape":...},
        ..., "initial_actions": [...] | null}
    <- {"actions": [[12 floats] x N], "next_initial_actions": [...], ...}

Two pieces of episode state are the *client's* responsibility, per their design:

* **chunk caching** -- the server returns a chunk; we execute `execute_n` of it
  before asking again. Their served config is `execute=5`, i.e. re-plan every 5
  steps. (Ours ran at 50.)
* **rolling inpaint anchor** -- `next_initial_actions` from each response must be
  sent back as `initial_actions` with the next request so consecutive chunks stay
  aligned. Dropping this yields discontinuous motion at every chunk boundary.

Images are sent base64 raw rather than as JSON lists: three 480x640x3 uint8
frames per step is ~2.7 MB, and `.tolist()` on that is both enormous and slow.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict

import numpy as np

from scripts.eval_policy.base_policy import BasePolicy
from scripts.eval_policy.registry import PolicyRegistry


def _enc(a: np.ndarray) -> Dict[str, Any]:
    a = np.ascontiguousarray(a)
    return {"base64": base64.b64encode(a.tobytes()).decode("ascii"),
            "dtype": a.dtype.str, "shape": list(a.shape)}


class RemoteWSPolicy(BasePolicy):
    """Queries an external WebSocket policy server for action chunks."""

    def __init__(self, device: str = "cuda", **kw):
        super().__init__()
        from websockets.sync.client import connect  # lazy: only this policy needs it

        self.url = os.environ.get("LEHOME_WS", "ws://127.0.0.1:8000")
        self.execute_n = int(os.environ.get("LEHOME_WS_EXECUTE", "5"))
        self._connect = connect
        self.ws = connect(self.url, max_size=None, open_timeout=60)
        self.ws.send(json.dumps({"type": "ping"}))
        print(f"[remote_ws] connected {self.url} -> {self.ws.recv()} "
              f"(execute_n={self.execute_n})", flush=True)
        self.reset()

    def reset(self):
        self._queue: list[np.ndarray] = []
        self._initial = None
        self._session = f"s{np.random.randint(1 << 30)}"

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        if not self._queue:
            req: Dict[str, Any] = {"type": "infer_chunk", "session_id": self._session}
            for k, v in observation.items():
                if not k.startswith("observation."):
                    continue
                a = np.asarray(v)
                if "images" in k:
                    req[k] = _enc(a[..., :3])
                elif k == "observation.state":
                    req[k] = _enc(a.astype(np.float32).reshape(-1))
            req["initial_actions"] = (self._initial.tolist()
                                      if self._initial is not None else None)
            self.ws.send(json.dumps(req))
            resp = json.loads(self.ws.recv())
            chunk = np.asarray(resp["actions"], dtype=np.float32)
            nxt = resp.get("next_initial_actions")
            self._initial = np.asarray(nxt, dtype=np.float32) if nxt is not None else None
            self._queue = list(chunk[: self.execute_n])
        return np.asarray(self._queue.pop(0), dtype=np.float32).reshape(-1)


def install() -> bool:
    """Replace the registered `lerobot` policy so the official kwargs branch is kept."""
    if not os.environ.get("LEHOME_WS"):
        return False
    # evaluation.py builds kwargs by testing `policy_type == "lerobot"` exactly,
    # so registering a new name would send model_path instead of policy_path.
    # Swapping the class keeps their argument plumbing untouched; the extra
    # kwargs are absorbed by **kw.
    PolicyRegistry._registry["lerobot"] = RemoteWSPolicy
    print("[remote_ws] ACTIVE - 'lerobot' now RemoteWSPolicy", flush=True)
    return True
