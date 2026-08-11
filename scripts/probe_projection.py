"""Does our render match its own camera config? Pure projection arithmetic.

Four image-based scale estimates disagreed wildly (garment area 1.8x, gripper
separation 1.59x, gripper blob width 0.75x, table-texture match 0.55x) because
every feature I picked was confounded -- grippers clip at the frame edge, the
garment sits in a different configuration, marble veins are self-similar. This
avoids feature-picking entirely.

Method: take world points whose positions are known exactly (the two robot bases
at (-0.23, -0.25, 0.5) and (0.23, -0.25, 0.5), and the garment centroid measured
from the particle set), project them through the camera pose and intrinsics read
from the live sim, and print where they should land in pixels. Compare against
where those things actually appear in our rendered frame.

* If prediction matches our render -> our render is faithful to its config, so
  the config is not the bug and the recorded dataset is the outlier (it must have
  been captured with different camera parameters).
* If prediction does NOT match our render -> the render disagrees with its own
  config, which would be a genuine bug on this machine.

Isaac's "ros" camera convention: +X right, +Y down, +Z forward along the optical
axis. Intrinsics come from focal length and aperture:
    fx = W * focal / horizontal_aperture,  fy = H * focal / vertical_aperture
"""

from __future__ import annotations

import math
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("PYTORCH_JIT", "0")

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True, enable_cameras=True, device="cuda").app

EXIT = 0
try:
    import numpy as np  # noqa: E402
    import torch  # noqa: E402
    from lehome.real_damped_project.tasks.isaac_garment_backend import (  # noqa: E402
        IsaacGarmentCfg, IsaacGarmentBackend,
    )

    backend = IsaacGarmentBackend(IsaacGarmentCfg(
        garment_name="Top_Long_Seen_0", device="cuda", decimation=3, joint_damping={}))
    backend.reset_env_ids(torch.zeros(1, dtype=torch.long))
    for _ in range(30):
        backend.simulate()
    print("[proj] settled", flush=True)

    env = backend.env
    cam, cfg = env.top_camera, env.cfg.top_camera
    pos = cam.data.pos_w[0].detach().cpu().numpy().astype(float)
    # quat_w_world uses the USD/world convention; the pinhole maths below is
    # written for ROS (+X right, +Y down, +Z forward), so read the ROS-convention
    # orientation. Using quat_w_world here put the garment behind the camera.
    if hasattr(cam.data, "quat_w_ros"):
        quat = cam.data.quat_w_ros[0].detach().cpu().numpy().astype(float)
        conv = "ros"
    else:
        quat = cam.data.quat_w_world[0].detach().cpu().numpy().astype(float)
        conv = "world(fallback)"

    W, H = int(cfg.width), int(cfg.height)
    focal = float(cfg.spawn.focal_length)
    h_ap = float(cfg.spawn.horizontal_aperture)
    v_ap = h_ap * H / W
    fx, fy = W * focal / h_ap, H * focal / v_ap
    cx, cy = W / 2.0, H / 2.0
    print(f"[proj] camera pos {np.round(pos,4).tolist()}  quat[{conv}] {np.round(quat,4).tolist()}")
    print(f"[proj] intrinsics fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}  ({W}x{H})")

    def R_from_quat(q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])

    Rwc = R_from_quat(quat)          # camera -> world
    Rcw = Rwc.T                       # world -> camera

    def project(p_w, label):
        p_c = Rcw @ (np.asarray(p_w, float) - pos)
        if p_c[2] <= 1e-6:
            print(f"[proj] {label:22} BEHIND camera (z={p_c[2]:+.3f})")
            return
        u = fx * p_c[0] / p_c[2] + cx
        v = fy * p_c[1] / p_c[2] + cy
        inside = (0 <= u < W) and (0 <= v < H)
        print(f"[proj] {label:22} world {np.round(p_w,3).tolist()} -> "
              f"pixel ({u:7.1f},{v:7.1f})  depth {p_c[2]:.3f} m  "
              f"{'in frame' if inside else 'OUT OF FRAME'}")

    pts = backend.env.object.get_current_mesh_points()[0]
    pts = np.asarray(pts, float).reshape(-1, 3)
    project((-0.23, -0.25, 0.5), "left robot base")
    project((0.23, -0.25, 0.5), "right robot base")
    project(pts.mean(axis=0), "garment centroid")
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    project((lo[0], pts.mean(axis=0)[1], pts.mean(axis=0)[2]), "garment left extreme")
    project((hi[0], pts.mean(axis=0)[1], pts.mean(axis=0)[2]), "garment right extreme")
    print(f"[proj] garment bbox {np.round(hi - lo, 3).tolist()} m", flush=True)

    backend.close()
except Exception:
    import traceback
    traceback.print_exc()
    sys.stderr.flush()
    EXIT = 1
finally:
    import threading

    def _force(code=EXIT):
        sys.stdout.flush(); sys.stderr.flush(); os._exit(code)

    t = threading.Timer(30.0, _force); t.daemon = True; t.start()
    try:
        app.close()
    except Exception:
        pass
    _force()
