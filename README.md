# LHF-6 — One token per frame: delta tokenisation as a driving scenario embedding

**Status:** not started · **Priority:** medium · **Budget:** 4–6 weeks

---

## The claim

*A Frame is Worth One Token* (Kerssies et al., CVPR 2026) introduces **DeltaTok**: encode the
*change* between consecutive frame features in a vision-foundation-model space as a **single
continuous token**, collapsing a 3D spatio-temporal grid to a 1D temporal sequence — a **1024×
token reduction at 512×512**. It reaches competitive dense forecasting on Cityscapes, KITTI and
VSPW with 35× fewer parameters and 2000× fewer FLOPs than Cosmos, by predicting only what changes
and leaving static background implicit.

Structurally, a sequence of delta tokens **is** the variable-length temporal embedding that
§7 of the gap report says nobody in CVPR 2026 produces. Kerssies et al. use it for generative
world modelling and test to roughly 10-frame horizons. This project asks whether it is also a
good *retrieval and clustering* representation for driving, at log length rather than clip length.

## Two claims

- **C1.** Delta-token sequences are a strong, extremely cheap representation for driving-scenario
  clustering and retrieval, beating mean-pooled frame features at a fraction of the storage.
- **C2.** They extend well past the ~10-frame horizon tested, and their per-frame granularity
  makes them naturally variable-length — you can index a 15 s AV2 log and a 20 min nuPlan log in
  the same space.

## Experiment −1 — three days

Compute delta tokens over 2,000 Argoverse 2 clips. Cluster. Compare against mean-pooled DINOv2
frame features on a maneuver-tag retrieval task (the same G-gallery construction used in
`../../vjepa_temporal/`).

**Stop rule.** If delta tokens do not beat mean pooling on maneuver retrieval, the "change is the
signal" premise does not hold for driving, where the dominant change is ego forward motion rather
than events. Try ego-motion-compensated deltas once, then stop.

That caveat is the real risk and it is worth stating plainly: in a driving video almost all
frame-to-frame change is caused by the ego vehicle moving forward past static structure. A naive
delta token may encode ego speed and nothing else. **Experiment 0 must test exactly this.**

## Relationship to P1

If it works, this is P1's visual front-end: delta tokens for appearance, trajectory tokens for
behaviour, one joint embedding.

---

See [PLAN.md](PLAN.md).

## Block 1 implementation (E-1)
- `src/download_av2.py`: AV2 Sensor front camera at 2 Hz plus ego poses (public S3, no auth)
- `src/tags.py`: clips (16 frames at 2 Hz) and maneuver tags from `city_SE3_egovehicle.feather`. The thresholds are stationary <0.5 m/s, turn |Δyaw|>30°, start/stop 0.5/3 m/s, lane change |Δyaw|<10° with lateral deviation >2 m from the arc, and accel/decel |Δv|>3 m/s. Everything else is cruise.
- `src/encoders.py`, `src/extract.py`: DINOv2-B mean-pool features and DeltaTok (DINOv3-B) delta tokens
- `src/evaluate.py`: LOO Recall@k, controls and ridge probes
- `src/test_pipeline.py`: the 20-clip unit test
- Results: `results/exp-1_block1_feasibility.md`
