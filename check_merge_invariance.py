import sys
import numpy as np
import h5py

I6 = np.array([1,0,0, 0,1,0], dtype=np.float32)

def stat(x):
    x = np.asarray(x)
    return dict(shape=x.shape,
                mean=float(x.mean()),
                std=float(x.std()),
                maxabs=float(np.max(np.abs(x))))

def maxdiff_to_I6(x):
    x = np.asarray(x).reshape(-1, 6).astype(np.float32)
    return float(np.max(np.abs(x - I6[None, :])))

path = sys.argv[1]

hits = []
with h5py.File(path, "r") as f:
    def visit(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        n = name.lower()
        # 你按自己文件命名习惯可加关键词
        if any(k in n for k in ["sbj_c", "second_sbj_c", "sbj_global", "second_sbj_global"]):
            x = obj[()]
            hits.append((name, x))
    f.visititems(visit)

if len(hits) == 0:
    print("No sbj_c / sbj_global datasets found in this hdf5.")
    print("=> 说明 samples 里可能根本没存这些参数；那就只能用 joints/root 去间接验证。")
    sys.exit(0)

for name, x in hits:
    n = name.lower()
    if "c" in n and x.shape[-1] == 3:
        s = stat(x)
        print(f"[{name}] {s}  (expect std≈0, maxabs≈0)")
    elif "global" in n and x.shape[-1] == 6:
        d = maxdiff_to_I6(x)
        print(f"[{name}] max|global-I6| = {d}  (expect ≈0)")
    else:
        print(f"[{name}] shape={np.asarray(x).shape} (unrecognized, but present)")
