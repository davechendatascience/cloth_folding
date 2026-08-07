"""Merge per-shard J labels into a single J.npy."""
import sys
from pathlib import Path
import numpy as np

cache = Path(sys.argv[1])
shards = sorted(cache.glob("J_shard*.npy"))
if not shards:
    raise SystemExit(f"no J_shard*.npy in {cache}")

out = None
for f in shards:
    a = np.load(f)
    out = a.copy() if out is None else np.where(np.isnan(out), a, out)
    print(f"  {f.name}: {int(np.isfinite(a).sum())} frames")

n_ok = int(np.isfinite(out).sum())
np.save(cache / "J.npy", out)
print(f"merged -> {cache/'J.npy'}: {n_ok}/{len(out)} frames labelled "
      f"({n_ok/len(out)*100:.1f}%)")
if n_ok:
    fin = out[np.isfinite(out)]
    print(f"  J range [{fin.min():.3f}, {fin.max():.3f}] mean {fin.mean():.3f}")
    print(f"  frames at J=0: {(fin<=0).sum()} ({(fin<=0).mean()*100:.1f}%)")
