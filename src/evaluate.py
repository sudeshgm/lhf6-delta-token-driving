"""Clip embeddings, LOO cosine Recall@k on maneuver tags, controls, ridge probes.
Usage: python src/evaluate.py features/pilot results/e_minus_1/pilot"""
import json, os, sys, glob, numpy as np
from sklearn.linear_model import RidgeCV; from sklearn.model_selection import GroupKFold; from sklearn.metrics import r2_score
CL = 16
def clip_embeddings(feat_dir, clips):
    E = {k: [] for k in ["meanpool_dinov2", "meanpool_dinov3", "delta_mean", "delta_meanstd", "ctrl_middle_frame_dinov2", "ctrl_shuffled_delta_meanstd", "ctrl_ego_speed_yawrate"]}
    keep, tok = [], []
    for c in clips:
        f = f"{feat_dir}/{c['log']}.npz"
        if not os.path.exists(f): continue
        d = np.load(f); s = c["start"]; e = s + CL; v2, v3 = d["v2"][s:e].astype(np.float32), d["v3"][s:e].astype(np.float32)
        z = d["z"][s:e - 1].astype(np.float32)   # tokens for pairs (s,s+1)..(e-2,e-1): 15 tokens, all inside the clip
        zs = d["zshuf"][s:e - 1].astype(np.float32)
        E["meanpool_dinov2"].append(v2.mean(0)); E["meanpool_dinov3"].append(v3.mean(0))
        E["delta_mean"].append(z.mean(0)); E["delta_meanstd"].append(np.concatenate([z.mean(0), z.std(0)]))
        E["ctrl_middle_frame_dinov2"].append(v2[CL // 2]); E["ctrl_shuffled_delta_meanstd"].append(np.concatenate([zs.mean(0), zs.std(0)]))
        E["ctrl_ego_speed_yawrate"].append(np.concatenate([c["speed"], np.array(c["yaw_rate"]) * 10]))
        sp, yr = np.array(c["speed"]), np.array(c["yaw_rate"])
        tok.append((z, (sp[:-1] + sp[1:]) / 2, (yr[:-1] + yr[1:]) / 2, v3[1:] - v3[:-1], c["log"])); keep.append(c)
    return {k: np.stack(v) for k, v in E.items()}, keep, tok
def recall(X, tags, logs, ks=(1, 5)):
    X = X - X.mean(0); X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8); S = X @ X.T
    S[logs[:, None] == logs[None, :]] = -np.inf   # LOO: exclude self and same-log clips
    nn = np.argsort(-S, 1)[:, :max(ks)]; hit = tags[nn] == tags[:, None]; out = {}
    for k in ks:
        h = hit[:, :k].any(1); out[f"R@{k}"] = float(h.mean())
        out[f"macroR@{k}"] = float(np.mean([h[tags == t].mean() for t in np.unique(tags)]))
    return out
def probe(tok):
    Z = np.concatenate([t[0] for t in tok]); V = np.concatenate([t[1] for t in tok]); Y = np.concatenate([t[2] for t in tok])
    DV3 = np.concatenate([t[3] for t in tok]); G = np.concatenate([[t[4]] * len(t[0]) for t in tok]); res = {}
    for name, X in [("delta_token", Z), ("dinov3_framediff", DV3)]:
        for tn, y in [("speed", V), ("yaw_rate", Y)]:
            p = np.zeros_like(y)
            for tr, te in GroupKFold(5).split(X, y, G):
                m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(X[tr], y[tr]); p[te] = m.predict(X[te])
            res[f"{name}->{tn}"] = float(r2_score(y, p))
    return res
if __name__ == "__main__":
    R = os.path.join(os.path.dirname(__file__), ".."); fd, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    clips = json.load(open(f"{R}/results/e_minus_1/clips_tags.json")); E, keep, tok = clip_embeddings(fd, clips)
    tags = np.array([c["tag"] for c in keep]); logs = np.array([c["log"] for c in keep])
    vals, cnt = np.unique(tags, return_counts=True)
    res = {"n_clips": len(keep), "tag_counts": dict(zip(vals.tolist(), cnt.tolist())), "chance_R@1_majority": float(cnt.max() / cnt.sum()),
           "chance_R@1_random": float(((cnt / cnt.sum()) ** 2).sum()),
           "retrieval": {k: recall(v, tags, logs) for k, v in E.items()}, "probe_R2": probe(tok)}
    rc = np.concatenate([np.load(f)["rc"] for f in glob.glob(f"{fd}/*.npz")]); cc = np.concatenate([np.load(f)["cc"] for f in glob.glob(f"{fd}/*.npz")])
    res["deltatok_recon_cos"] = float(rc.mean()); res["copy_last_cos"] = float(cc.mean())
    json.dump(res, open(f"{out}/metrics.json", "w"), indent=1); print(json.dumps(res, indent=1))
