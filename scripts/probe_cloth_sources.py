"""Compare the two cloth geometry sources. LEHOME_PROBE_CLOTH=1.

The success checker reads `particle_object.get_current_mesh_points()` and sees the
garment fold (J 7.15 -> 0.118 on a demo replay). The cameras render a garment
that does not move: over the same 315-step replay only 6.51% of pixels ever
change by >25, versus 53.85% in the recording, and frame 314 is visually
identical in configuration to frame 0.

Both cannot be reading the same geometry. `get_object_particle_position` already
hints at two representations by falling back from `get_current_mesh_points()` to
`_cloth_prim_view.get_world_positions()`. If the first moves while the second
stays put, then PhysX is advancing a particle state that is never written back
to the prim the renderer draws -- which would explain every symptom at once:
J moves, check-points move, the image does not, and any vision policy is blind.

Logs the centroid and spread of both sources each check so divergence is
obvious without dumping thousands of points.
"""

from __future__ import annotations

import os

import numpy as np


def install() -> bool:
    if not os.environ.get("LEHOME_PROBE_CLOTH"):
        return False

    import lehome.utils.success_checker_chanllege as sc

    orig = sc.get_object_particle_position
    n = {"i": 0}

    def wrapped(particle_object, index_list):
        out = orig(particle_object, index_list)
        n["i"] += 1
        if n["i"] % 20 == 1:
            def summarize(arr):
                a = np.asarray(arr, dtype=float).reshape(-1, 3)
                return (f"n={len(a)} centroid=({a[:, 0].mean():+.3f},"
                        f"{a[:, 1].mean():+.3f},{a[:, 2].mean():+.3f}) "
                        f"bbox={np.ptp(a, axis=0).round(3).tolist()}")

            msgs = []
            try:
                pts, *_ = particle_object.get_current_mesh_points()
                msgs.append("MESH  " + summarize(pts))
            except Exception as e:
                msgs.append(f"MESH  failed: {type(e).__name__}")
            try:
                wp = particle_object._cloth_prim_view.get_world_positions()
                msgs.append("VIEW  " + summarize(wp.squeeze(0).detach().cpu().numpy()))
            except Exception as e:
                msgs.append(f"VIEW  failed: {type(e).__name__}")
            print(f"[CLOTH {n['i']:04d}] " + " | ".join(msgs), flush=True)
        return out

    sc.get_object_particle_position = wrapped
    print("[probe_cloth_sources] ACTIVE", flush=True)
    return True
