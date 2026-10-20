"""Render 50 stratified clips (trajectory in start frame, speed, first/last frame) for manual tag checking."""
import json, random, collections, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, cv2, os
R = os.path.join(os.path.dirname(__file__), ".."); c = json.load(open(f"{R}/results/e_minus_1/clips_tags.json")); random.seed(0)
by = collections.defaultdict(list); [by[x["tag"]].append(x) for x in c]
sel = []; 
for t, v in sorted(by.items()): sel += random.sample(v, min(5, len(v)))
sel += random.sample([x for x in c if x not in sel], 50 - len(sel)); sel = sel[:50]
json.dump([x["clip"] for x in sel], open(f"{R}/results/e_minus_1/tagcheck_ids.json", "w"))
os.makedirs(f"{R}/results/e_minus_1/tagcheck", exist_ok=True)
for page in range(5):
    fig, ax = plt.subplots(10, 4, figsize=(14, 30))
    for r, x in enumerate(sel[page * 10:(page + 1) * 10]):
        yaw0 = x["yaw"][0]; X, Y = np.array(x["x"]), np.array(x["y"]); c0, s0 = np.cos(yaw0), np.sin(yaw0)
        lon, lat = c0 * X + s0 * Y, -s0 * X + c0 * Y
        ax[r, 0].plot(-lat, lon, ".-"); ax[r, 0].set_aspect("equal", "datalim"); ax[r, 0].set_title(f"#{page*10+r} {x['tag']}  (x=right,y=fwd)", fontsize=9)
        ax[r, 1].plot(x["speed"]); ax[r, 1].set_ylim(0, 16); ax[r, 1].set_title(f"speed m/s  dyaw={np.degrees(x['yaw'][-1]-x['yaw'][0]):.0f}deg", fontsize=9)
        for k, i in enumerate([0, -1]):
            im = cv2.imread(f"{R}/data/av2/{x['frames'][i]}")[:, :, ::-1]; ax[r, 2 + k].imshow(im[100:-150]); ax[r, 2 + k].axis("off")
    plt.tight_layout(); plt.savefig(f"{R}/results/e_minus_1/tagcheck/page{page}.jpg", dpi=45); plt.close()
