"""Exp1 ego-fixed evaluation. Scene labels (src/scene_labels.py), retrieval restricted to the same ego bin
(ego maneuver tag x speed tercile), same-log excluded; log-bootstrap CIs; logistic-probe AUROC (GroupKFold by log); kill rule.
Usage: python src/evaluate_egofixed.py <feature_dir> <out_dir>   (e.g. /workspace/lhf6_snap results/exp1_egofixed  or  features/full256 results/exp1_egofixed/full)"""
import json, os, sys, numpy as np
from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler; from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold; from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__)); from evaluate2 import load, residualise, zblock
rng = np.random.default_rng(0)
FEATS = ["meanpool_dinov2", "meanpool_dinov3", "ctrl_middle_frame", "ctrl_shuffled_delta", "ctrl_ego_speed_yaw", "delta_meanstd",
         "B_deltatok_egocomp", "B_patch_residual", "C_sampson_moving", "concat_ego+delta"]
DELTA = ["delta_meanstd", "B_deltatok_egocomp", "B_patch_residual", "C_sampson_moving"]
def top1(X, logs, bins):
    X = X - X.mean(0); X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8); S = X @ X.T
    S[(logs[:, None] == logs[None, :]) | (bins[:, None] != bins[None, :])] = -np.inf
    valid = np.isfinite(S.max(1)); return S.argmax(1), valid
def boot(hit, y, logs, B=1000):
    def st(i): h, t = hit[i], y[i]; return np.array([h.mean(), np.mean([h[t == u].mean() for u in np.unique(t)])])
    ul = np.unique(logs); by = {l: np.where(logs == l)[0] for l in ul}; pt = st(np.arange(len(y)))
    bs = np.stack([st(np.concatenate([by[l] for l in rng.choice(ul, len(ul))])) for _ in range(B)]); lo, hi = np.percentile(bs, [2.5, 97.5], 0)
    return {"R@1": [round(float(pt[0]), 3), round(float(lo[0]), 3), round(float(hi[0]), 3)], "balR@1": [round(float(pt[1]), 3), round(float(lo[1]), 3), round(float(hi[1]), 3)]}
def auroc(X, y, logs):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, logs): p[te] = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return round(float(roc_auc_score(y, p)), 3)
if __name__ == "__main__":
    R = os.path.join(os.path.dirname(__file__), ".."); fd, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    clips = json.load(open(f"{R}/results/exp0/clips_tags.json")); E, keep, _ = load(fd, clips)
    E["concat_ego+delta"] = np.concatenate([zblock(E["ctrl_ego_speed_yaw"]), zblock(E["delta_meanstd"])], 1)
    logs = np.array([c["log"] for c in keep]); msp = np.array([np.mean(c["speed"]) for c in keep])
    terc = np.digitize(msp, np.percentile(msp, [33.3, 66.7])); bins = np.array([f"{c['tag']}|{t}" for c, t in zip(keep, terc)])
    SL = json.load(open(f"{R}/results/exp1_egofixed/scene_labels.json")); labs = list(next(iter(SL.values())))
    res = {"n_clips": len(keep), "n_logs": len(set(logs)), "n_ego_bins": len(set(bins)), "prevalence": {}, "dropped": [], "retrieval": {}, "auroc": {}}
    for L in labs:
        y = np.array([SL[c["clip"]][L] for c in keep]).astype(int); res["prevalence"][L] = [int(y.sum()), round(float(y.mean()), 3)]
        if y.sum() < 15 or (len(y) - y.sum()) < 15: res["dropped"].append(L); continue
        res["retrieval"][L] = {}; res["auroc"][L] = {}
        for f in FEATS:
            nn, v = top1(E[f], logs, bins); hit = (y[nn] == y)[v]
            r = boot(hit.astype(float), y[v], logs[v]); r["n_queries"] = int(v.sum()); res["retrieval"][L][f] = r
            res["auroc"][L][f] = auroc(E[f], y, logs)
    # kill rule: a delta variant beats BOTH ego-only and meanpool-v3 by >=3 balR@1 points with non-overlapping CIs on >=2 labels
    wins = {}
    for d in DELTA:
        wins[d] = [L for L in res["retrieval"] if all(
            res["retrieval"][L][d]["balR@1"][0] - res["retrieval"][L][b]["balR@1"][0] >= .03 and res["retrieval"][L][d]["balR@1"][1] > res["retrieval"][L][b]["balR@1"][2]
            for b in ["ctrl_ego_speed_yaw", "meanpool_dinov3"])]
    res["kill_rule_wins"] = wins; res["premise_survives"] = any(len(v) >= 2 for v in wins.values())
    json.dump(res, open(f"{out}/metrics.json", "w"), indent=1); print(json.dumps({k: res[k] for k in ["n_clips", "prevalence", "dropped", "kill_rule_wins", "premise_survives"]}))
