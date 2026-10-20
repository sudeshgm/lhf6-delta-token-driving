"""Block 2 per-log extraction (CPU, 256px): v2, v3, raw z, zshuf, plus ego-compensated variants
  zB : DeltaTok enc(y_hat_{t+1}, y_{t+1}) where y_hat is the epipolar ego-predicted DINOv3 patch map (see egocomp.compensate)
  rB : patch-feature residual mean(y_{t+1} - y_hat_{t+1})  (DeltaTok-free version of B)
  cC : variant C, mean delta over high-Sampson (independently moving) patches; fC = fraction of such patches
Resumable: one npz per log. Usage: python src/extract2.py --out features/full256"""
import argparse, glob, os, sys, time, numpy as np, torch, cv2
sys.path.insert(0, os.path.dirname(__file__)); from encoders import DINOv2, DeltaTokEnc, prep; import egocomp as E
ap = argparse.ArgumentParser(); ap.add_argument("--size", type=int, default=256); ap.add_argument("--out", default="features/full256")
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1); ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args(); torch.set_num_threads(a.threads); R = os.path.join(os.path.dirname(__file__), ".."); os.makedirs(f"{R}/{a.out}", exist_ok=True)
v2 = DINOv2(); D = DeltaTokEnc(); m = D.m; bb = m.backbone.backbone; npre = m.backbone.num_prefix_tokens
logs = sorted(os.path.dirname(p) for p in glob.glob(f"{R}/data/av2/*/*/DONE"))[a.shard::a.nshard]; rng = np.random.default_rng(0); t0 = time.time()
C = E.patch_centers(a.size)
for i, d in enumerate(logs):
    o = f"{R}/{a.out}/{os.path.basename(d)}.npz"
    if os.path.exists(o): continue
    fr = sorted(glob.glob(f"{d}/ring_front_center/*.jpg")); ts = np.array([int(os.path.basename(f)[:-4]) for f in fr])
    ims = [cv2.imread(f)[:, :, ::-1].copy() for f in fr]
    try:
        K, eTc = E.cam_model(d, a.size); cp = E.cam_poses(d, ts, eTc); Fs = [E.fundamental(K, cp[t], cp[t + 1]) for t in range(len(fr) - 1)]; src = "pose"
    except Exception:
        sq = [cv2.resize(x[(x.shape[0] - x.shape[1]) // 2:][:x.shape[1]], (a.size, a.size)) for x in ims]
        Fs = [(E.fundamental_orb(sq[t], sq[t + 1]), 1.0) for t in range(len(fr) - 1)]; src = "orb"
    with torch.no_grad():
        f2 = torch.cat([v2(prep(ims[j:j + 16], 224)) for j in range(0, len(ims), 16)])
        x = prep(ims, a.size); y = bb(pixel_values=x).last_hidden_state[:, npre:]; rope = m.rope_embeddings(x[:1])
        z = m.encode(y[:-1], y[1:], rope)[:, 0]
        perm = rng.permutation(len(ims)); zs = m.encode(y[perm][:-1], y[perm][1:], rope)[:, 0]
        yn = y.numpy(); yh, cC, fC = [], [], []
        for t in range(len(fr) - 1):
            h, c, f = E.compensate(yn[t], yn[t + 1], Fs[t][0], Fs[t][1], C); yh.append(h); cC.append(c); fC.append(f)
        yh = torch.from_numpy(np.stack(yh)); zB = m.encode(yh, y[1:], rope)[:, 0]; rB = (y[1:] - yh).mean(1)
    np.savez(o, ts=ts, v2=f2.half().numpy(), v3=y.mean(1).half().numpy(), z=z.half().numpy(), zshuf=zs.half().numpy(),
             zB=zB.half().numpy(), rB=rB.half().numpy(), cC=np.stack(cC).astype(np.float16), fC=np.array(fC), base=np.array([b for _, b in Fs]), src=src)
    print(f"{i+1}/{len(logs)} {os.path.basename(d)} {src} {time.time()-t0:.0f}s", flush=True)
