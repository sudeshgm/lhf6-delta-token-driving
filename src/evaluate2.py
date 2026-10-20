"""Block 2 / Exp0 evaluation on features/full256 (resumable features; evaluates whatever logs exist).
Usage: python src/evaluate2.py features/full256 results/exp0"""
import json, os, sys, numpy as np
from sklearn.linear_model import RidgeCV, Ridge; from sklearn.model_selection import GroupKFold; from sklearn.metrics import r2_score
CL = 16; rng = np.random.default_rng(0)
ms = lambda a: np.concatenate([a.mean(0), a.std(0)])
def load(fd, clips):
    E, keep, tok = {}, [], []
    add = lambda k, v: E.setdefault(k, []).append(v)
    for c in clips:
        f = f"{fd}/{c['log']}.npz"
        if not os.path.exists(f): continue
        d = np.load(f); s, e = c["start"], c["start"] + CL; g = lambda k, n=0: d[k][s:e - n].astype(np.float32)
        v2, v3, z, zs, zB, rB, cC = g("v2"), g("v3"), g("z", 1), g("zshuf", 1), g("zB", 1), g("rB", 1), g("cC", 1)
        ego = np.concatenate([c["speed"], np.array(c["yaw_rate"]) * 10])
        add("meanpool_dinov2", v2.mean(0)); add("meanpool_dinov3", v3.mean(0)); add("delta_mean", z.mean(0)); add("delta_meanstd", ms(z))
        add("ctrl_middle_frame", v2[CL // 2]); add("ctrl_shuffled_delta", ms(zs)); add("ctrl_ego_speed_yaw", ego)
        add("B_deltatok_egocomp", ms(zB)); add("B_patch_residual", ms(rB)); add("C_sampson_moving", ms(cC))
        sp, yr = np.array(c["speed"]), np.array(c["yaw_rate"])
        tok.append(dict(raw=z, B=zB, B_resid=rB, C=cC, sp=(sp[:-1] + sp[1:]) / 2, yr=(yr[:-1] + yr[1:]) / 2, log=c["log"])); keep.append(c)
    return {k: np.stack(v) for k, v in E.items()}, keep, tok
def hits(X, tags, logs, k=5, mask=None):
    X = X - X.mean(0); X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8); S = X @ X.T
    S[logs[:, None] == logs[None, :]] = -np.inf
    if mask is not None: S[:, ~mask] = -np.inf
    nn = np.argsort(-S, 1)[:, :k]; return tags[nn] == tags[:, None]
def summarise(h, tags, logs, B=1000):
    """R@1, class-balanced R@1, R@5 with 95% cluster (log) bootstrap CIs."""
    def stat(idx):
        h1, h5, t = h[idx, 0], h[idx, :5].any(1), tags[idx]
        return np.array([h1.mean(), np.mean([h1[t == u].mean() for u in np.unique(t)]), h5.mean()])
    ul = np.unique(logs); by = {l: np.where(logs == l)[0] for l in ul}; pt = stat(np.arange(len(tags)))
    bs = np.stack([stat(np.concatenate([by[l] for l in rng.choice(ul, len(ul))])) for _ in range(B)])
    lo, hi = np.percentile(bs, [2.5, 97.5], 0)
    return {n: [round(float(pt[i]), 3), round(float(lo[i]), 3), round(float(hi[i]), 3)] for i, n in enumerate(["R@1", "balR@1", "R@5"])}
def residualise(X, Z, groups):
    """Cross-fitted (by log) ridge residual of each dim of X on ego profile Z."""
    out = np.zeros_like(X)
    for tr, te in GroupKFold(5).split(X, groups=groups):
        m = Ridge(alpha=1.0).fit(Z[tr], X[tr]); out[te] = X[te] - m.predict(Z[te])
    return out
def zblock(X): X = (X - X.mean(0)) / (X.std(0) + 1e-6); return X / np.sqrt(X.shape[1])
def probe(tok, key):
    X = np.concatenate([t[key] for t in tok]); G = np.concatenate([[t["log"]] * len(t[key]) for t in tok]); r = {}
    for tn in ["sp", "yr"]:
        y = np.concatenate([t[tn] for t in tok]); p = np.zeros_like(y)
        for tr, te in GroupKFold(5).split(X, y, G): p[te] = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(X[tr], y[tr]).predict(X[te])
        r[{"sp": "speed", "yr": "yaw_rate"}[tn]] = round(float(r2_score(y, p)), 3)
    return r
if __name__ == "__main__":
    R = os.path.join(os.path.dirname(__file__), ".."); fd, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    clips = json.load(open(f"{R}/results/exp0/clips_tags.json")); E, keep, tok = load(fd, clips)
    tags = np.array([c["tag"] for c in keep]); logs = np.array([c["log"] for c in keep])
    ego = E["ctrl_ego_speed_yaw"]
    E["partial_out_delta_meanstd"] = residualise(E["delta_meanstd"], ego, logs)
    E["concat_ego+delta"] = np.concatenate([zblock(ego), zblock(E["delta_meanstd"])], 1)
    E["partial_out_B_deltatok"] = residualise(E["B_deltatok_egocomp"], ego, logs)
    res = {"n_clips": len(keep), "n_logs": len(set(logs)), "tag_counts": {k: int(v) for k, v in zip(*np.unique(tags, return_counts=True))},
           "retrieval": {k: summarise(hits(v, tags, logs), tags, logs) for k, v in E.items()},
           "probe_R2": {k: probe(tok, k) for k in ["raw", "B", "B_resid", "C"]}}
    dyn = json.load(open(f"{R}/results/exp0/dyn_labels.json")); cross = np.array([dyn[c["clip"]]["cross"] for c in keep])
    msp = np.array([np.mean(c["speed"]) for c in keep]); myr = np.array([np.abs(c["yaw_rate"]).max() for c in keep])
    subsets = {"ego_stationary(mean v<0.5)": msp < 0.5, "fast_cruise(mean v>10, |yr|<0.05; no clip >15 m/s)": (msp > 10) & (myr < 0.05), "all": np.ones(len(keep), bool)}
    res["cross_traffic_retrieval"] = {}
    for sn, m in subsets.items():
        r = {"n": int(m.sum()), "frac_cross": round(float(cross[m].mean()), 3)}
        for k in ["meanpool_dinov3", "delta_meanstd", "B_deltatok_egocomp", "B_patch_residual", "C_sampson_moving", "ctrl_ego_speed_yaw"]:
            X, t, l = E[k][m], cross[m].astype(int), logs[m]; h = hits(X, t, l); r[k] = summarise(h, t, l, B=300)
        res["cross_traffic_retrieval"][sn] = r
    json.dump(res, open(f"{out}/metrics.json", "w"), indent=1); print(json.dumps(res))
