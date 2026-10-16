import json, glob, os, numpy as np
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
KEYS = ["ex1", "ex2", "ex3", "ex4", "ex5", "ex6"]
SUBS = [("ex1", 0), ("ex2", 0), ("ex3", 0), ("ex4", 0), ("ex5", 0), ("ex5", 1), ("ex6", 0)]
SUBNAME = {("ex1", 0): "Example 1", ("ex2", 0): "Example 2", ("ex3", 0): "Example 3", ("ex4", 0): "Example 4",
           ("ex5", 0): "Example 5, ch. 1", ("ex5", 1): "Example 5, ch. 2", ("ex6", 0): "Example 6"}


def load(group):
    D = {}
    for f in sorted(glob.glob(os.path.join(RES, "raw", f"*_{group}.json"))):
        d = json.load(open(f)); D[(d["key"], d["seed"])] = d
    return D


def collect(D, arch, sub, field, mode="sp"):
    key, ch = sub; v = []
    for (k, s), d in sorted(D.items()):
        if k != key or arch not in d["archs"]: continue
        pc = d["archs"][arch]["per_channel"][ch]
        v.append(pc[mode][field] if field in pc[mode] else np.nan)
    return np.array(v, float)

if __name__ == "__main__":
    D = load("main")
    for mode in ("sp", "fr"):
        print("==", mode)
        for sub in SUBS:
            row = []
            for a in ["FNN", "ENN", "MRENN", "ENN-BPTT", "FNN-m", "ENN-m"]:
                v = collect(D, a, sub, "rmse", mode)
                row.append(f"{a}:{np.median(v):.3f}({v.mean():.3f}±{v.std(ddof=1):.3f})")
            print(SUBNAME[sub], " | ".join(row))
