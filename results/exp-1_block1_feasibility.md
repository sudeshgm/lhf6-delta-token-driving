# LHF6 Block 1: feasibility + E-1 CPU pilot (2026-10-20 IST)

## Box
- 8 vCPU, 15 GB RAM (12 GB available), about 26 GB free disk on / after the data download. No GPU. venv: /workspace/venv_lhf6 (torch 2.10, transformers 5.0)
- DINOv2-S/14 @224, batch 16: 0.69 s/batch. DINOv2-B/14: 1.58 s/batch
- DeltaTok (DINOv3 ViT-B/16 + Kinetics tokenizer), 16 frames: 7.2 s @256, 31.7 s @512 (native)

## Weights
- Amazon-FAR/deltatok-kinetics: OK
- facebook/dinov3-vitb16-pretrain-lvd1689m: **gated, but the licence is already accepted on the box's HF login**, so it downloaded normally. No workaround was used.
- facebook/dinov2-base and facebook/dinov2-small: OK
- Sanity check: DeltaTok reconstruction cosine is 0.938, compared with 0.911 for copy-last, so the weights load correctly

## Data
- AV2 Sensor is public without auth: `s5cmd --no-sign-request ls s3://argoverse/datasets/av2/sensor/{train,val}/` (700 + 150 logs)
- Raw ring_front_center is about 100–118 MB/log (319 frames at 20 Hz). Every log has `city_SE3_egovehicle.feather` (about 2.6k poses over about 16 s).
- Downloaded **all 850 train+val logs**: ring_front_center at 2 Hz (every 10th frame, 32 frames/log), resized to a 512 short side as JPEG q90, plus the ego poses. Total **2.1 GB** in `data/av2/{split}/{log}/`. Recipe: `python src/download_av2.py`
- Clips: 16 frames (about 7.5 s) at 2 Hz, 2 per log, giving **1698 clips** (`results/e_minus_1/clips_tags.json`, with per-clip speed and yaw-rate profiles, xy and yaw)
- RefAV is public (github.com/CainanD/RefAV; a copy is at /workspace/refav). Its prompts are not used in E-1. The tags come from ego poses.

## Maneuver tags (src/tags.py)
Poses are resampled to 10 Hz with a 0.5 s smoothing filter. Rules are checked in this order:
1. stationary: max speed < 0.5 m/s
2. turn L/R: |Δyaw| > 30°
3. start / stop: first or last 1.5 s < 0.5 m/s and the other end > 3 m/s
4. lane change L/R: |Δyaw| < 10°, mean speed > 3, and the end lateral offset deviates from the constant-curvature arc by > 2 m
5. accelerate / decelerate: |Δv| > 3 m/s
6. cruise: everything else

Counts (1698 clips): cruise 857, decelerate 215, turn_right 130, accelerate 115, stop 113, turn_left 88, start 71, lane_change_left 46, lane_change_right 33, stationary 30.

**Hand-check of 50 stratified clips** (pose plot, speed curve, first and last frame; `results/e_minus_1/tagcheck/page*.jpg`):
- About 43/50 correct = **86%** overall
- Non-lane-change tags: about 39/40 (one cruise was borderline decelerate)
- Lane change: only about 4/10. Most false positives are gentle road curves.

Lane-change tags need a yaw-rate S-curve test or map lane IDs before they can be used in headline numbers.

## Pilot: 200 clips (100 logs, 10 of them val first), CPU, DeltaTok @256, DINOv2-B @224
- Extraction took 2014 s (about 20 s/log). Features are in `features/pilot/*.npz`, 24 MB.
- Recall is leave-one-out cosine, excluding clips from the same log.
- Chance: R@1 is 0.455 for the majority class and 0.25 for random-by-prior.

| embedding | R@1 | macro R@1 | R@5 |
|---|---|---|---|
| mean-pool DINOv2-B | 0.265 | 0.115 | 0.635 |
| mean-pool DINOv3-B | 0.250 | 0.103 | 0.670 |
| delta tokens, mean | 0.445 | 0.406 | 0.820 |
| **delta tokens, [mean,std]** | **0.470** | **0.417** | **0.850** |
| ctrl: middle frame DINOv2 | 0.205 | 0.096 | 0.650 |
| ctrl: shuffled-order deltas | 0.275 | 0.126 | 0.640 |
| ctrl: ego speed+yaw-rate profile (oracle; the tags come from it) | 0.715 | 0.603 | 0.875 |

Ridge probe on per-pair tokens, 5-fold GroupKFold by log:

| input | speed R² | yaw-rate R² |
|---|---|---|
| delta token | **0.78** | **0.87** |
| DINOv3 frame-difference | -0.03 | -0.00 |

Reading:
- **H1 (kill criterion) passes**: delta tokens beat mean-pool by about +20 points R@1 and about 4× on macro R@1. The shuffled-order control collapses back to the mean-pool level, so the temporal order carries the signal.
- **H2 holds**: ego speed R² is 0.78, above 0.5, so the deltas are dominated by ego motion. This supports the C3/Block-2 motivation.
- Caveats:
  - n = 200, so treat these as noisy.
  - Micro R@1 for delta tokens is only about equal to majority-class chance. Macro R@1 is the better metric.
  - The pilot ran at 256 px, not the native 512.

## Wall time
- Full 1698 clips (850 logs × 32 frames, about 27k frames) on CPU: about 4.7 h @256 (feasible), about 18 h @512 (impractical)
- DINOv2-S alone over 2000 × 30 frames: about 1.5 h on CPU
- **CPU-feasible: yes at 256 px; native 512 px is better done on GPU**

## GPU Bot command (full extraction, native 512, fp16, pooled + per-pair tokens, no dense maps)
```
git clone --recursive https://github.com/sudeshgm/lhf6-delta-token-driving && cd lhf6-delta-token-driving && pip install -r requirements.txt && pip install s5cmd && S5CMD=$(which s5cmd) python src/download_av2.py && python src/tags.py && python src/extract.py --logs 850 --size 512 --fp16 --device cuda --out features/full512 && python src/evaluate.py features/full512 results/e_minus_1/full512
```
- The command needs an HF login with the DINOv3 licence accepted.
- Outputs:
  - `data/av2` about 2.1 GB
  - `features/full512/{log}.npz`, 850 files, about 200 MB total (fp16: v2 32×1536, v3 32×768, z/zshuf 31×768)
  - `results/e_minus_1/full512/metrics.json`
- Expected time: about 1 h on one A100-class GPU.
