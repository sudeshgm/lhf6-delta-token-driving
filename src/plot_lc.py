"""Plot 20 random S-test lane-change clips (trajectory, yaw rate, first/mid/last frame) for manual check."""
import json, random, numpy as np, matplotlib, os; matplotlib.use("Agg"); import matplotlib.pyplot as plt, cv2
R = os.path.join(os.path.dirname(__file__), ".."); c = json.load(open(f"{R}/results/exp0/clips_tags.json")); random.seed(1)
sel = random.sample([x for x in c if x["tag"].startswith("lane_change")], 20); json.dump([x["clip"] for x in sel], open(f"{R}/results/exp0/lc_check_ids.json", "w"))
for pg in range(2):
    fig, ax = plt.subplots(10, 5, figsize=(17, 30))
    for r, x in enumerate(sel[pg * 10:pg * 10 + 10]):
        y0 = x["yaw"][0]; X, Y = np.array(x["x"]), np.array(x["y"]); lon, lat = np.cos(y0) * X + np.sin(y0) * Y, -np.sin(y0) * X + np.cos(y0) * Y
        ax[r, 0].plot(-lat, lon, ".-"); ax[r, 0].set_xlim(-8, 8); ax[r, 0].set_title(f"#{pg*10+r} {x['tag']} (x=right)", fontsize=9)
        ax[r, 1].plot(x["yaw_rate"]); ax[r, 1].axhline(0, c="k"); ax[r, 1].set_ylim(-0.2, 0.2); ax[r, 1].set_title("yaw rate rad/s", fontsize=9)
        for k, i in enumerate([0, 8, 15]):
            ax[r, 2 + k].imshow(cv2.imread(f"{R}/data/av2/{x['frames'][i]}")[150:-150, :, ::-1]); ax[r, 2 + k].axis("off")
    plt.tight_layout(); plt.savefig(f"{R}/results/exp0/lc_check_{pg}.jpg", dpi=50); plt.close()
