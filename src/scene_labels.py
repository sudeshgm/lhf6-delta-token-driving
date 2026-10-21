"""Ego-independent scene labels per clip from AV2 3D cuboid tracks (no HD map, no RefAV: ego lane approximated as
|lateral| < 1.8 m in the ego frame). Objects in city frame; velocities from 0.5 s-smoothed track positions.
  cross_traffic : vehicle moving >1 m/s with |cos(heading_vel, ego_heading)| < .5, in front wedge (x>2,|y|<x,r<50)
  vru_crossing  : pedestrian/cyclist/wheelchair moving >0.5 m/s, |cos|<.5, 2<x<30, |y|<10
  lead_braking  : vehicle in ego lane (|y|<1.8, 2<x<30) with longitudinal decel > 2 m/s^2 (speed >1 m/s before)
  cut_in        : vehicle moving >1 m/s seen at 1.8<|y|<5 then later |y|<1.2 with 2<x<40
  oncoming_turn : vehicle with velocity opposing ego (cos<-.7) at some time, then |Δheading| > 30 deg within the clip, 2<x<50, |y|<20"""
import json, os, glob, numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot
VEH = {"REGULAR_VEHICLE", "LARGE_VEHICLE", "BUS", "BOX_TRUCK", "TRUCK", "TRUCK_CAB", "VEHICULAR_TRAILER", "MOTORCYCLIST", "SCHOOL_BUS", "ARTICULATED_BUS", "RAILED_VEHICLE"}
VRU = {"PEDESTRIAN", "BICYCLIST", "WHEELCHAIR", "WHEELED_RIDER", "STROLLER"}
def log_tracks(d):
    a = pd.read_feather(f"{d}/annotations.feather"); p = pd.read_feather(f"{d}/city_SE3_egovehicle.feather").drop_duplicates("timestamp_ns").set_index("timestamp_ns")
    p = p.loc[p.index.intersection(a.timestamp_ns.unique())]; a = a[a.timestamp_ns.isin(p.index)].copy()
    Rm = Rot.from_quat(p.loc[a.timestamp_ns][["qx", "qy", "qz", "qw"]].values); T = p.loc[a.timestamp_ns][["tx_m", "ty_m"]].values
    W = Rm.apply(a[["tx_m", "ty_m", "tz_m"]].values); a["cx"], a["cy"] = W[:, 0] + T[:, 0], W[:, 1] + T[:, 1]
    H = Rm.apply(np.tile([1, 0, 0], (len(a), 1)))[:, :2]; a["hx"], a["hy"] = H[:, 0], H[:, 1]
    a = a.sort_values(["track_uuid", "timestamp_ns"]); out = []
    for _, g in a.groupby("track_uuid"):
        g = g.copy(); t = g.timestamp_ns.values * 1e-9
        if len(g) < 4: continue
        k = min(5, len(g)); sm = lambda v: np.convolve(np.pad(v, k // 2, mode="edge"), np.ones(k) / k, mode="valid")[:len(v)]
        vx, vy = np.gradient(sm(g.cx.values), t), np.gradient(sm(g.cy.values), t); sp = np.hypot(vx, vy)
        g["sp"], g["vx"], g["vy"] = sp, vx, vy; g["acc"] = np.gradient(sm(sp), t)
        g["cos"] = (vx * g.hx + vy * g.hy) / (sp + 1e-9); g["vh"] = np.unwrap(np.arctan2(vy, vx)); out.append(g)
    return pd.concat(out) if out else a.iloc[:0]
def clip_labels(g, t0, t1):
    w = g[(g.timestamp_ns >= t0) & (g.timestamp_ns <= t1)]; x, y = w.tx_m, w.ty_m; L = {}
    veh, vru = w.category.isin(VEH), w.category.isin(VRU)
    L["cross_traffic"] = bool((veh & (w.sp > 1) & (w.cos.abs() < .5) & (x > 2) & (y.abs() < x) & (np.hypot(x, y) < 50)).any())
    L["vru_crossing"] = bool((vru & (w.sp > .5) & (w.cos.abs() < .5) & (x > 2) & (x < 30) & (y.abs() < 10)).any())
    L["lead_braking"] = bool((veh & (y.abs() < 1.8) & (x > 2) & (x < 30) & (w.acc < -2) & (w.sp > 1)).any())
    ci = on = False
    for _, tr in w[veh].groupby("track_uuid"):
        tr = tr.sort_values("timestamp_ns"); ty, tx = tr.ty_m.values, tr.tx_m.values; mv = tr.sp.values > 1
        out_i = np.where((np.abs(ty) > 1.8) & (np.abs(ty) < 5) & (tx > 2) & (tx < 40) & mv)[0]; in_i = np.where((np.abs(ty) < 1.2) & (tx > 2) & (tx < 40) & mv)[0]
        if len(out_i) and len(in_i) and in_i.max() > out_i.min(): ci = True
        opp = np.where((tr.cos.values < -.7) & mv & (tx > 2) & (tx < 50) & (np.abs(ty) < 20))[0]
        if len(opp) and mv.sum() > 3 and np.abs(tr.vh.values[mv][-1] - tr.vh.values[opp[0]]) > np.radians(30): on = True
    L["cut_in"], L["oncoming_turn"] = ci, on
    return L
if __name__ == "__main__":
    R = os.path.join(os.path.dirname(__file__), ".."); clips = json.load(open(f"{R}/results/exp0/clips_tags.json")); out = {}
    for log in sorted({c["log"] for c in clips}):
        g = log_tracks(glob.glob(f"{R}/data/av2/*/{log}")[0])
        for c in [c for c in clips if c["log"] == log]:
            ts = [int(os.path.basename(f)[:-4]) for f in c["frames"]]; out[c["clip"]] = clip_labels(g, ts[0], ts[-1])
    json.dump(out, open(f"{R}/results/exp1_egofixed/scene_labels.json", "w"))
    print({k: sum(v[k] for v in out.values()) for k in next(iter(out.values()))}, len(out))
