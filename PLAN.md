# WORKING RESEARCH NOTE — LHF-6

## Delta tokens as a variable-length driving scenario representation

---

## 1. Claims

- **C1.** Delta-token sequences beat mean-pooled frame features for driving-scenario retrieval and
  clustering at a small fraction of the storage.
- **C2.** The representation is genuinely variable-length: quality does not depend on clip length,
  so 15 s and 20 min logs live in one index.
- **C3.** Ego-motion compensation is necessary. Raw delta tokens in driving encode ego speed; the
  useful signal appears only after the ego-induced component is removed.

C3 is the most likely actual finding and, if it holds, is the paper's method contribution.

## 2. Objects of study

- **Backbone:** a frozen VFM (DINOv2/DINOv3) as in the original; features per frame, delta encoded
  by a small learned tokeniser.
- **Data:** Argoverse 2 Sensor (15–20 s logs, maneuver-taggable), nuScenes, BDD100K, and nuPlan for
  the long-clip test.
- **Labels for evaluation:** RefAV prompts and maneuver tags. This is the cheapest source of
  scenario-level ground truth that exists.

## 3. Hypotheses

| # | Hypothesis | Falsified if |
|---|---|---|
| H1 | Delta tokens beat mean-pooled frame features on maneuver retrieval. | No improvement in Recall@1. **Exp −1 kill criterion.** |
| H2 | Raw deltas are dominated by ego motion. | A linear probe on the delta sequence predicts ego speed with R² < 0.5. |
| H3 | Ego-compensated deltas beat raw deltas. | Δ Recall@1 < 3 points. |
| H4 | Quality is invariant to clip length. | Recall@1 degrades > 10 % relative when moving from 15 s to 5 min clips at matched query difficulty. |
| H5 | Storage advantage is real at equal accuracy. | Not at least 10× smaller than the frame-feature baseline at matched Recall@1. |

## 4. Experiments, kill-first

### 4.1 Experiment −1 — three days
Delta tokens over 2,000 AV2 clips; maneuver-tag retrieval vs mean-pooled DINOv2.
**Stop rule:** no win, and no win after ego compensation → the premise fails for driving; stop.

### 4.2 Experiment 0 — the ego-motion confound, one week
This is the whole risk of the project, so it is the whole of Experiment 0.
1. Probe the delta sequence for ego speed and yaw rate (H2). If R² is high, the token is an
   odometer.
2. Build three variants: raw delta, delta after homography/epipolar ego compensation, and delta
   restricted to independently-moving regions. The epipolar formulation matters here for the same
   reason it does in `../../vjepa_temporal/`: forward translation over 3D structure produces
   parallax that no single homography explains, so RANSAC on a homography will label every
   building as a moving agent.
3. **Ego-stationary control set.** Clips where the ego is stopped at a light with cross traffic.
   Raw deltas should work well here and badly on highway cruising; if that pattern appears, C3 is
   confirmed cheaply.

### 4.3 Experiment 1 — retrieval and clustering (C1)
Retrieval by maneuver tag and by RefAV prompt; clustering purity against RefAV's atomic-function
activations. Baselines: mean-pooled frame features, first/middle/last frame only, hand-crafted
kinematics, and the frozen-CLIP baseline from `../p1_scenario_embedding/`.

### 4.4 Experiment 2 — length invariance (C2, H4)
Index 15 s AV2 logs and multi-minute nuPlan logs together. Query with a fixed-length clip and
measure whether long logs are systematically over- or under-retrieved — a well-known failure of
length-sensitive representations that no CVPR 2026 video paper reports because none of them
handles variable length.

### 4.5 Experiment 3 — storage/accuracy frontier (H5)
Bytes per second of video vs Recall@1, against frame-feature and keyframe baselines. This is the
plot that connects to the missing index literature (2 papers in 4,042 mention ANN indexing).

### 4.6 Negative controls
- **Single-frame control.** Middle frame only. Whatever it achieves is the static ceiling.
- **Shuffled-delta control.** Shuffle the delta sequence. If retrieval survives, order does not
  matter and the representation is a bag of changes.
- **Ego-speed-only control.** A 1-D feature: the ego speed profile. Embarrassing if competitive,
  and it may well be — say so if it is.

## 5. Risk register

| Risk | Reality | Mitigation |
|---|---|---|
| Ego motion dominates | The central risk. Almost all frame-to-frame change in driving is ego forward motion. | Exp 0, in full, before anything else. |
| Homography is the wrong ego model | Parallax from forward translation is not explained by a homography. | Use epipolar geometry (fundamental matrix + Sampson distance) as in the `vjepa_temporal` note, not a homography. |
| Tokeniser training | The original trains a tokeniser on top of frozen VFM features. | Keep it small; a T4 handles it. Report the training budget. |
| Long-clip feature extraction | Extracting VFM features for multi-minute nuPlan logs is the compute bottleneck. | Subsample frame rate for the long-clip experiment and report it; the length-invariance claim is about the representation, not the sampling. |

## 6. Failure modes

1. Delta tokens = odometer, and compensation does not fix it → stop; report the negative, which is
   directly useful to anyone about to apply DeltaTok to driving.
2. Ego-speed-only control is competitive → the retrieval task is too easy; harden it with RefAV
   prompts that hold ego behaviour fixed.
3. Length invariance fails → drop C2, keep C1 and C3.

## 7. Week plan

| Week | Work | Stop rule |
|---|---|---|
| 0 (3 days) | Exp −1 retrieval vs mean pooling. | No win after one compensation attempt → stop. |
| 1 | Exp 0 ego-motion confound in full; three delta variants; ego-stationary control. | Odometer, uncorrectable → stop. |
| 2–3 | Exp 1 retrieval/clustering; all negative controls. | Ego-speed control competitive → harden the task. |
| 4 | Exp 2 length invariance; Exp 3 storage frontier. | — |
| 5–6 | Writing; integration path into P1. | — |

## 8. Title directions

- *A Drive Is Worth One Token Per Frame*
- *Change Is Not Enough: Ego-Compensated Delta Tokens for Driving Video*

## 9. Non-goals

Generative world modelling — that is the original paper. Beating video-LLMs on VQA. Pixel-level
reconstruction quality.
