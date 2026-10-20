"""Epipolar ego-motion geometry for AV2 ring_front_center at our preprocessing (512-short-side jpg -> centre square -> SxS).
F from KNOWN ego pose + calibration (preferred); ORB + F-RANSAC fallback when calibration/pose is missing."""
import numpy as np, pandas as pd, cv2
from scipy.spatial.transform import Rotation as Rot
def se3(qw, qx, qy, qz, tx, ty, tz):
    T = np.eye(4); T[:3, :3] = Rot.from_quat([qx, qy, qz, qw]).as_matrix(); T[:3, 3] = [tx, ty, tz]; return T
def cam_model(d, S):
    i = pd.read_feather(f"{d}/intrinsics.feather").set_index("sensor_name").loc["ring_front_center"]
    e = pd.read_feather(f"{d}/egovehicle_SE3_sensor.feather").set_index("sensor_name").loc["ring_front_center"]
    W, H = i.width_px, i.height_px; side = min(W, H); s = S / side; ox, oy = (W - side) / 2, (H - side) / 2
    K = np.array([[i.fx_px * s, 0, (i.cx_px - ox) * s], [0, i.fy_px * s, (i.cy_px - oy) * s], [0, 0, 1]])  # radial distortion ignored
    return K, se3(e.qw, e.qx, e.qy, e.qz, e.tx_m, e.ty_m, e.tz_m)
def cam_poses(d, ts, ego_T_cam):
    p = pd.read_feather(f"{d}/city_SE3_egovehicle.feather").sort_values("timestamp_ns")
    idx = np.clip(np.searchsorted(p.timestamp_ns.values, ts), 0, len(p) - 1)
    return [se3(*p.iloc[j][["qw", "qx", "qy", "qz", "tx_m", "ty_m", "tz_m"]].values) @ ego_T_cam for j in idx]
def fundamental(K, c1, c2):
    """x2^T F x1 = 0 for pixel coords of frame 1 (t) and frame 2 (t+1). Returns F, |baseline| m."""
    T = np.linalg.inv(c2) @ c1; R, t = T[:3, :3], T[:3, 3]
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]]); Ki = np.linalg.inv(K)
    return Ki.T @ (tx @ R) @ Ki, float(np.linalg.norm(t))
def fundamental_orb(im1, im2):
    """Fallback: ORB matches + F-RANSAC (OpenCV)."""
    o = cv2.ORB_create(2000); k1, d1 = o.detectAndCompute(cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY), None); k2, d2 = o.detectAndCompute(cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY), None)
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    p1 = np.float32([k1[x.queryIdx].pt for x in m]); p2 = np.float32([k2[x.trainIdx].pt for x in m])
    F, _ = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 1.0, 0.999); return F
def sampson(F, x1, x2):
    x1h = np.c_[x1, np.ones(len(x1))]; x2h = np.c_[x2, np.ones(len(x2))]
    Fx1 = x1h @ F.T; Ftx2 = x2h @ F; num = np.sum(x2h * Fx1, 1) ** 2
    return num / (Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2 + 1e-12)
def patch_centers(S, P=16):
    g = (np.arange(S // P) + 0.5) * P; X, Y = np.meshgrid(g, g); return np.c_[X.ravel(), Y.ravel()]
def compensate(y1, y2, F, base, C, tol=12.0, samp_thr=100.0):
    """y1,y2: (N,D) L2-normalisable patch features of t, t+1.
    B: ego-predicted y2_hat[p] = y1[q*], q* = best cosine match among patches of t within `tol` px of the epipolar line F^T x2(p)
       (static-scene correspondence must lie on that line; depth unknown, so we search along it). Fallback: same position.
    C: unconstrained best match m(p); Sampson distance of (x1(m), x2(p)) w.r.t. ego F; patches with d > samp_thr px^2 (10 px; patch grid quantisation ~8 px) are
       inconsistent with ego motion => independently moving. Returns mean (y2-y1[m]) over those patches and their fraction."""
    n1 = y1 / np.linalg.norm(y1, axis=1, keepdims=True); n2 = y2 / np.linalg.norm(y2, axis=1, keepdims=True); Sim = n2 @ n1.T  # (p2, q1)
    N = len(C)
    if base < 0.05:  # ego (nearly) stationary: static scene does not move
        y2hat = y1.copy(); m = Sim.argmax(1); moving = (m != np.arange(N)) & (Sim[np.arange(N), m] > Sim[np.arange(N), np.arange(N)] + 0.05)
    else:
        L = np.c_[C, np.ones(N)] @ F  # epipolar lines in image t for each p2 : l = F^T x2
        dist = np.abs(np.c_[C, np.ones(N)] @ L.T).T / (np.linalg.norm(L[:, :2], axis=1, keepdims=True) + 1e-12)  # (p2, q1)
        cand = dist <= tol; Sm = np.where(cand, Sim, -np.inf); q = Sm.argmax(1); q = np.where(cand.any(1), q, np.arange(N))
        y2hat = y1[q]; m = Sim.argmax(1); moving = sampson(F, C[m], C) > samp_thr
    cC = (y2[moving] - y1[m[moving]]).mean(0) if moving.any() else np.zeros(y1.shape[1], np.float32)
    return y2hat, cC, float(moving.mean())
