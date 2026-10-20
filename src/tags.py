"""Clips + maneuver tags from AV2 ego poses (city_SE3_egovehicle.feather).
Clip = 16 consecutive 2 Hz frames (~7.5 s); 2 non-overlapping clips per log.
Per frame timestamp we interpolate ego pose -> speed (m/s, 2D), yaw (rad), yaw rate (rad/s)."""
import numpy as np, pandas as pd, glob, os, json
CLIP_LEN = 16
def yaw_from_q(qw, qx, qy, qz): return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
def ego_profile(pose_df, ts):
    pose_df = pose_df.sort_values("timestamp_ns").drop_duplicates("timestamp_ns")
    t0 = pose_df.timestamp_ns.values[0]
    t = (pose_df.timestamp_ns.values - t0) * 1e-9; tq = (np.asarray(ts) - t0) * 1e-9
    yaw = np.unwrap(yaw_from_q(pose_df.qw.values, pose_df.qx.values, pose_df.qy.values, pose_df.qz.values))
    g = np.arange(t[0], t[-1], 0.1)  # resample to 10 Hz grid, then differentiate
    gx, gy, gyaw = (np.interp(g, t, v) for v in (pose_df.tx_m.values, pose_df.ty_m.values, yaw))
    sm = lambda v: np.convolve(np.pad(v, 2, mode="edge"), np.ones(5) / 5, mode="valid")  # 0.5 s box filter
    sp = sm(np.hypot(np.gradient(gx, 0.1), np.gradient(gy, 0.1))); yr = sm(np.gradient(gyaw, 0.1))
    X, Y, YAW = (np.interp(tq, g, v) for v in (gx, gy, gyaw))
    return dict(x=X, y=Y, yaw=YAW, speed=np.interp(tq, g, sp), yaw_rate=np.interp(tq, g, yr))
# thresholds (documented in README)
STOP_V, MOVE_V, TURN_DEG, LC_HEAD_DEG, LC_LAT_M, DV = 0.5, 3.0, 30.0, 10.0, 2.0, 3.0
S_TEST, LC_YR = True, 0.03  # rad/s
def s_shape(yr, side):
    """Lane change = S-shaped yaw rate: a peak of the lane-change sign (>LC_YR) followed later by an opposite peak (>LC_YR).
    Road curves give one-signed yaw rate and fail this test."""
    yr = np.asarray(yr) * side; a = np.where(yr > LC_YR)[0]; b = np.where(yr < -LC_YR)[0]
    return len(a) > 0 and len(b) > 0 and b.max() > a.min()
def tag(p):
    v, yaw = p["speed"], p["yaw"]; dyaw = np.degrees(yaw[-1] - yaw[0])
    c, s = np.cos(yaw[0]), np.sin(yaw[0]); dx, dy = p["x"] - p["x"][0], p["y"] - p["y"][0]
    lon, lat = c * dx + s * dy, -s * dx + c * dy   # lateral offset in start frame (+ = left)
    # lateral offset w.r.t. a straight path along the chord heading removes gentle road curvature only partially; require small net heading
    if v.max() < STOP_V: return "stationary"
    if abs(dyaw) > TURN_DEG: return "turn_left" if dyaw > 0 else "turn_right"
    if v[:3].mean() < STOP_V and v[-3:].mean() > MOVE_V: return "start"
    if v[:3].mean() > MOVE_V and v[-3:].mean() < STOP_V: return "stop"
    # lane change: net heading small, but lateral offset relative to chord of a constant-curvature arc is large
    arc_lat = lon[-1] * np.tan(np.radians(dyaw) / 2)   # lateral expected from smooth arc with same net heading
    if abs(dyaw) < LC_HEAD_DEG and v.mean() > MOVE_V and abs(lat[-1] - arc_lat) > LC_LAT_M:
        side = 1 if lat[-1] - arc_lat > 0 else -1
        if not S_TEST or s_shape(p["yaw_rate"], side):
            return "lane_change_left" if side > 0 else "lane_change_right"
    if v[-3:].mean() - v[:3].mean() > DV: return "accelerate"
    if v[:3].mean() - v[-3:].mean() > DV: return "decelerate"
    return "cruise"
def build(root, out):
    clips = []
    for d in sorted(glob.glob(f"{root}/*/*/DONE")):
        d = os.path.dirname(d); sp, log = d.split("/")[-2:]
        fr = sorted(glob.glob(f"{d}/ring_front_center/*.jpg")); ts = [int(os.path.basename(f)[:-4]) for f in fr]
        p = ego_profile(pd.read_feather(f"{d}/city_SE3_egovehicle.feather"), ts)
        for ci in range(len(fr) // CLIP_LEN):
            s = slice(ci * CLIP_LEN, (ci + 1) * CLIP_LEN); q = {k: v[s] for k, v in p.items()}
            clips.append(dict(clip=f"{log}_{ci}", split=sp, log=log, start=ci * CLIP_LEN, frames=[os.path.relpath(f, root) for f in fr[s]],
                              tag=tag(q), speed=q["speed"].round(3).tolist(), yaw_rate=q["yaw_rate"].round(4).tolist(),
                              x=(q["x"] - q["x"][0]).round(2).tolist(), y=(q["y"] - q["y"][0]).round(2).tolist(), yaw=q["yaw"].round(4).tolist()))
    json.dump(clips, open(out, "w")); return clips
if __name__ == "__main__":
    import collections, sys
    R = os.path.join(os.path.dirname(__file__), "..")
    c = build(f"{R}/data/av2", f"{R}/results/exp0/clips_tags.json")  # E-1 used results/e_minus_1/clips_tags.json (no S-test)
    print(len(c), collections.Counter(x["tag"] for x in c))
