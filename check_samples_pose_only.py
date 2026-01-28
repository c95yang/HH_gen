import sys, h5py, numpy as np

path = sys.argv[1]

def summarize_root(root):
    # root: (T,3)
    mean = root.mean(0)
    std  = root.std(0)
    rng  = root.max(0) - root.min(0)
    return mean, std, rng

all_std = []
all_rng = []

with h5py.File(path, "r") as f:
    joint_keys = []
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and name.lower().endswith(("_j", "j")):
            x = obj
            if len(x.shape) == 3 and x.shape[-1] == 3:
                joint_keys.append(name)
    f.visititems(visit)

    print("num joint datasets:", len(joint_keys))
    worst = None  # (score, name, std, rng)

    for k in joint_keys:
        J = f[k][()]            # (T,J,3)
        root = J[:, 0, :]       # (T,3)
        mean, std, rng = summarize_root(root)

        all_std.append(std)
        all_rng.append(rng)

        score = float(np.max(std))  # 最大轴向std作为“最坏”指标
        if worst is None or score > worst[0]:
            worst = (score, k, std, rng, mean)

    all_std = np.stack(all_std, 0)
    all_rng = np.stack(all_rng, 0)

    print("=== Aggregate over all seq ===")
    print("mean(std_xyz) =", all_std.mean(0))
    print("max(std_xyz)  =", all_std.max(0))
    print("mean(range_xyz) =", all_rng.mean(0))
    print("max(range_xyz)  =", all_rng.max(0))

    print("\n=== Worst seq ===")
    print("key:", worst[1])
    print("std_xyz :", worst[2])
    print("range_xyz:", worst[3])
    print("mean_xyz :", worst[4])
