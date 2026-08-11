"""Move the top camera along its view axis. LEHOME_CAM_BACK=<metres>.

Measured discrepancy: the recorded dataset shows the garment occupying ~20.5% of
frame area (three episodes: 20.5 / 20.5 / 24.5), our sim shows 37.1%, and the
masks confirm ours is the same garment *clipped* at the top and both sides rather
than a segmentation artefact. Area scales as 1/d^2, so the recording implies a
camera-to-garment distance near 0.81 m against our measured 0.547 m.

This shifts the camera **backwards along the garment->camera axis**, which
changes distance without changing where it points, so the existing orientation
stays valid. Positive values move it away.

Interception point is `scripts.utils.common.stabilize_garment_after_reset`: it
receives the live env and is called once per episode immediately after
`env.reset()`, which is the first moment both the camera and the settled garment
exist. Patching the env config instead would need `lehome.tasks.bedroom` imported
before the Isaac app is up, which it is not.

Note the caveat this experiment is testing against: the challenge winner used the
unmodified camera config and scored 74.5%, so if this fixes framing it means they
absorbed the same gap through augmentation (their solution jitters camera
position, rotation and focal length) rather than that the config is wrong.
"""

from __future__ import annotations

import os

import numpy as np


def install() -> bool:
    raw = os.environ.get("LEHOME_CAM_BACK", "").strip()
    if not raw:
        return False
    back = float(raw)

    import scripts.utils.common as common

    orig = common.stabilize_garment_after_reset
    done = {"v": False}

    def wrapped(env, args, num_steps: int = 20):
        out = orig(env, args, num_steps)
        try:
            cam = env.top_camera
            pos = cam.data.pos_w[0].detach().cpu().numpy().astype(float)
            quat = cam.data.quat_w_world[0].detach().cpu().numpy().astype(float)

            pts = np.asarray(env.object.get_current_mesh_points()[0],
                             dtype=float).reshape(-1, 3)
            centroid = pts.mean(axis=0)

            axis = pos - centroid
            d = float(np.linalg.norm(axis))
            axis = axis / max(d, 1e-9)
            new = pos + axis * back

            import torch
            cam.set_world_poses(
                torch.tensor(new, dtype=torch.float32, device=cam.device).unsqueeze(0),
                torch.tensor(quat, dtype=torch.float32, device=cam.device).unsqueeze(0),
                convention="world",
            )
            if not done["v"]:
                print(f"[shift_top_camera] distance {d:.3f} -> {d + back:.3f} m "
                      f"(+{back:.3f}); pos {np.round(pos,3).tolist()} -> "
                      f"{np.round(new,3).tolist()}", flush=True)
                done["v"] = True
        except Exception as e:
            print(f"[shift_top_camera] FAILED: {type(e).__name__}: {e}", flush=True)
        return out

    common.stabilize_garment_after_reset = wrapped
    print(f"[shift_top_camera] ACTIVE (+{back} m)", flush=True)
    return True
