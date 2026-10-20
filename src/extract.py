"""Per-log feature extraction. For each log (all 2 Hz front frames):
  v2   : DINOv2 ViT-B/14 [CLS, patch-mean] per frame (T,1536) @224
  v3   : DINOv3 ViT-B/16 patch-mean per frame (T,768)          @S
  z    : DeltaTok delta token z_t = enc(y_{t-1}, y_t), t=1..T-1 (T-1,768)
  zshuf: DeltaTok tokens over a random temporal permutation of the frames (control)
  rc/cc: recon cosine of decode(z) vs copy-last cosine (sanity)
Usage: python src/extract.py --logs N --size 256 --out features/pilot"""
import argparse, glob, os, sys, time, numpy as np, torch, cv2
sys.path.insert(0, os.path.dirname(__file__)); from encoders import DINOv2, DeltaTokEnc, prep
ap = argparse.ArgumentParser(); ap.add_argument("--logs", type=int, default=100); ap.add_argument("--size", type=int, default=256)
ap.add_argument("--out", default="features/pilot"); ap.add_argument("--fp16", action="store_true"); ap.add_argument("--device", default="cpu")
ap.add_argument("--per_frame_tokens", action="store_true", help="unused placeholder: per-frame tokens are always saved (no dense maps)")
a = ap.parse_args(); R = os.path.join(os.path.dirname(__file__), ".."); os.makedirs(f"{R}/{a.out}", exist_ok=True)
dev = torch.device(a.device); dt = torch.float16 if a.fp16 else torch.float32
v2 = DINOv2(); v2.m.to(dev, dt); D = DeltaTokEnc(); D.m.to(dev, dt)
logs = sorted(os.path.dirname(p) for p in glob.glob(f"{R}/data/av2/*/*/DONE"))
val = [l for l in logs if "/val/" in l]; tr = [l for l in logs if "/train/" in l]
logs = (val + tr)[:a.logs]; rng = np.random.default_rng(0); t0 = time.time()
for i, d in enumerate(logs):
    o = f"{R}/{a.out}/{os.path.basename(d)}.npz"
    if os.path.exists(o): continue
    fr = sorted(glob.glob(f"{d}/ring_front_center/*.jpg")); ims = [cv2.imread(f)[:, :, ::-1].copy() for f in fr]
    with torch.no_grad():
        f2 = torch.cat([v2(prep(ims[j:j + 16], 224).to(dev, dt)) for j in range(0, len(ims), 16)])
        f3, z, rc, cc = D(prep(ims, a.size).to(dev, dt))
        perm = rng.permutation(len(ims)); _, zs, _, _ = D(prep([ims[p] for p in perm], a.size).to(dev, dt))
    np.savez(o, ts=np.array([int(os.path.basename(f)[:-4]) for f in fr]), v2=f2.half().cpu().numpy(), v3=f3.half().cpu().numpy(),
             z=z.half().cpu().numpy(), zshuf=zs.half().cpu().numpy(), rc=rc.float().cpu().numpy(), cc=cc.float().cpu().numpy())
    print(f"{i+1}/{len(logs)} {os.path.basename(d)} T={len(fr)} {time.time()-t0:.0f}s", flush=True)
