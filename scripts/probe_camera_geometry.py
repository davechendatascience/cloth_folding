"""Does the garment subtend the frame it should? Pure geometry, no rendering.

The sim's top camera shows the garment filling the view; the recorded dataset
shows it occupying roughly 40% with wide white margins. Everything needed to
settle that analytically is knowable:

  camera   focal_length 28.7 mm, horizontal_aperture 38.11 mm, 640x480,
           offset (0.245, -0.44, 0.56) from /World/Robot/Right_Robot/base
  garment  bbox 0.511 x 0.294 m, centroid ~(-0.062, +0.111, +0.534)  [measured]

The config gives the camera's offset from the robot base, not its world pose, so
the distance to the garment cannot be computed from the config alone -- this
reads `top_camera.data.pos_w` from the live sim instead.

If predicted coverage matches the dataset's ~40% then the camera is where it
should be and the framing difference is something else (garment scale, or the
reset leaving it bunched). If predicted coverage matches what we observe (~100%)
then the camera really is too close, and that alone would explain the missing
pixel motion: the parts of the scene that move are cropped out.

Run via the same launcher as the other Isaac scripts; needs no policy.
"""

from __future__ import annotations

import math
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("PYTORCH_JIT", "0")

from isaaclab.app import AppLauncher  # noqa: E402

launcher = AppLauncher(headless=True, enable_cameras=True, device="cuda")
app = launcher.app

EXIT = 0
try:
    import numpy as np  # noqa: E402
    import torch  # noqa: E402
    from lehome.real_damped_project.tasks.isaac_garment_backend import (  # noqa: E402
        IsaacGarmentCfg, IsaacGarmentBackend,
    )

    backend = IsaacGarmentBackend(IsaacGarmentCfg(
        garment_name="Top_Long_Seen_0", device="cuda", decimation=3, joint_damping={}))
    print("[probe] backend built", flush=True)
    backend.reset_env_ids(torch.zeros(1, dtype=torch.long))
    print("[probe] reset done", flush=True)
    for _ in range(30):          # let the garment settle, as the eval does
        backend.simulate()

    print("[probe] settled", flush=True)
    env = backend.env
    cam = env.top_camera
    cfg = env.cfg.top_camera

    pos = cam.data.pos_w[0].detach().cpu().numpy()
    print(f"\n[cam] world position      : {np.round(pos, 4).tolist()}", flush=True)
    try:
        q = np.round(cam.data.quat_w_world[0].detach().cpu().numpy(), 4).tolist()
        print(f"[cam] world quat (wxyz)   : {q}", flush=True)
    except Exception:
        pass

    obj = env.object
    pts = np.asarray(obj.get_current_mesh_points()[0], dtype=float).reshape(-1, 3)
    centroid = pts.mean(axis=0)
    bbox = np.ptp(pts, axis=0)
    print(f"[cloth] points {len(pts)}  centroid {np.round(centroid, 4).tolist()}  "
          f"bbox {np.round(bbox, 4).tolist()}", flush=True)

    focal = float(getattr(cfg.spawn, "focal_length", 28.7))
    h_ap = float(getattr(cfg.spawn, "horizontal_aperture", 38.11))
    W, H = int(cfg.width), int(cfg.height)
    v_ap = h_ap * H / W
    hfov = 2 * math.atan(h_ap / (2 * focal))
    vfov = 2 * math.atan(v_ap / (2 * focal))
    print(f"[optics] focal {focal} mm  h_aperture {h_ap} mm  {W}x{H}", flush=True)
    print(f"[optics] HFOV {math.degrees(hfov):.1f} deg   VFOV {math.degrees(vfov):.1f} deg", flush=True)

    d = float(np.linalg.norm(pos - centroid))
    print(f"\n[geom] camera-to-garment distance : {d:.4f} m", flush=True)
    view_w = 2 * d * math.tan(hfov / 2)
    view_h = 2 * d * math.tan(vfov / 2)
    print(f"[geom] visible extent at that range: {view_w:.3f} m wide x {view_h:.3f} m tall", flush=True)

    gw, gh = float(bbox[0]), float(bbox[1])
    fw, fh = gw / view_w, gh / view_h
    print(f"[geom] garment {gw:.3f} x {gh:.3f} m  ->  {100*fw:5.1f}% of width, "
          f"{100*fh:5.1f}% of height, ~{100*fw*fh:5.1f}% of frame area", flush=True)
    print("\n[reference] dataset ~40% of frame area; our render fills the view (~100%).",
          flush=True)
    for label, frac in (("dataset-like 40% area", 0.40), ("observed ~100% area", 1.00)):
        target = math.sqrt(frac)
        need = gw / (2 * math.tan(hfov / 2) * target)
        print(f"            {label:<22} would need distance {need:.3f} m", flush=True)

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
