"""Per-clip scene-dynamics labels from AV2 3D cuboid annotations (not ego motion):
  n_moving : distinct tracks moving > 1 m/s (city frame) inside the front-camera wedge (ego x>2, |y|<x, range<50 m)
  cross    : >=1 such track whose velocity is crossing (|cos(angle to ego heading)| < 0.5)"""
import json, os, glob, numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot
R = os.path.join(os.path.dirname(__file__), ".."); clips = json.load(open(f"{R}/results/exp0/clips_tags.json")); out = {}
for log in sorted({c["log"] for c in clips}):
    d = glob.glob(f"{R}/data/av2/*/{log}")[0]; a = pd.read_feather(f"{d}/annotations.feather"); p = pd.read_feather(f"{d}/city_SE3_egovehicle.feather").drop_duplicates("timestamp_ns").set_index("timestamp_ns")
    p = p.loc[p.index.intersection(a.timestamp_ns.unique())]; Rm = Rot.from_quat(p[["qx", "qy", "qz", "qw"]].values)
    pc = {t: (r, np.array([x, y, z])) for t, r, x, y, z in zip(p.index, Rm, p.tx_m, p.ty_m, p.tz_m)}
    a = a[a.timestamp_ns.isin(pc)].copy(); E = a[["tx_m", "ty_m", "tz_m"]].values
    W = np.stack([pc[t][0].apply(e) + pc[t][1] for t, e in zip(a.timestamp_ns, E)]); a["cx"], a["cy"] = W[:, 0], W[:, 1]
    a = a.sort_values(["track_uuid", "timestamp_ns"]); g = a.groupby("track_uuid")
    dt = g.timestamp_ns.diff() * 1e-9; a["vx"] = g.cx.diff() / dt; a["vy"] = g.cy.diff() / dt
    hd = {t: pc[t][0].apply([1, 0, 0])[:2] for t in pc}; H = np.stack([hd[t] for t in a.timestamp_ns])
    sp = np.hypot(a.vx, a.vy); a["moving"] = sp > 1.0; a["cross"] = a.moving & (np.abs((a.vx * H[:, 0] + a.vy * H[:, 1]) / (sp + 1e-9)) < 0.5)
    a["infov"] = (a.tx_m > 2) & (a.ty_m.abs() < a.tx_m) & (np.hypot(a.tx_m, a.ty_m) < 50)
    for c in [c for c in clips if c["log"] == log]:
        ts = [int(os.path.basename(f)[:-4]) for f in c["frames"]]; w = a[(a.timestamp_ns >= ts[0]) & (a.timestamp_ns <= ts[-1]) & a.infov]
        out[c["clip"]] = dict(n_moving=int(w[w.moving].track_uuid.nunique()), cross=bool(w[w.cross].track_uuid.nunique() > 0), n_cross=int(w[w.cross].track_uuid.nunique()))
json.dump(out, open(f"{R}/results/exp0/dyn_labels.json", "w")); v = list(out.values())
print(len(v), "cross frac", np.mean([x["cross"] for x in v]), "n_moving median", np.median([x["n_moving"] for x in v]))
