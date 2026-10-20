"""Unit test on the 20-clip set (10 logs in features/unit)."""
import os, sys, json, numpy as np; sys.path.insert(0, os.path.dirname(__file__)); from evaluate import clip_embeddings, recall, probe
R = os.path.join(os.path.dirname(__file__), "..")
def test_all():
    clips = json.load(open(f"{R}/results/e_minus_1/clips_tags.json")); E, keep, tok = clip_embeddings(f"{R}/features/unit", clips)
    assert len(keep) == 20
    assert E["delta_mean"].shape == (20, 768) and E["delta_meanstd"].shape == (20, 1536) and E["meanpool_dinov2"].shape == (20, 1536)
    assert all(t[0].shape == (15, 768) for t in tok) and all(np.isfinite(v).all() for v in E.values())
    tags = np.array([c["tag"] for c in keep]); logs = np.array([c["log"] for c in keep])
    # recall sanity: identical-tag duplicate embeddings in distinct logs must give R@1 = 1
    X = np.eye(4)[np.arange(20) % 4]; X = X + 1e-3 * np.random.default_rng(0).normal(size=X.shape)
    assert recall(X, np.arange(20) % 4, np.arange(20))["R@1"] == 1.0
    r = recall(E["delta_meanstd"], tags, logs); assert 0 <= r["R@1"] <= 1
    p = probe(tok); assert set(p) >= {"delta_token->speed"}
    d = np.load(f"{R}/features/unit/{keep[0]['log']}.npz"); assert d["rc"].mean() > d["cc"].mean()  # DeltaTok weights loaded correctly
    print("OK", r, p)
if __name__ == "__main__": test_all()
