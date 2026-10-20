# Exp0 / Block 2: ego-motion compensation (CPU, 256 px)

**Status:** INTERIM, covering 414 clips from 207 logs. Full extraction (850 logs) is still running detached:
- `src/extract2.py`, 2 shards, resumable, one npz per log in `features/full256/`
- `/workspace/lhf6_finalize.sh` will write `results/exp0/full/metrics.json` and push automatically

**Protocol**
- Leave-one-out cosine retrieval, excluding neighbours from the same log
- 95% CIs from a cluster bootstrap over logs (1000 resamples)
- Tags come from `results/exp0/clips_tags.json`, which includes the S-test for lane changes

## Compensation method (epipolar, no homography)
- F is computed per pair from the **known AV2 ego pose + calibration**: E=[t]x R between camera poses, F = K^-T E K^-1. K is adjusted for our crop/resize; distortion is ignored. All logs used pose.
- An ORB + F-RANSAC fallback is implemented (`egocomp.fundamental_orb`) but was never needed.
- **B (DeltaTok):**
  - Ego-predicted map: for each DINOv3 patch p of frame t+1, take the best cosine-matching patch of frame t within 12 px of the epipolar line F^T x(p), giving ŷ_{t+1}. The epipolar line is used because depth is unknown, so the search runs along it.
  - Token: z_B = DeltaTok.enc(ŷ_{t+1}, y_{t+1}). DeltaTok accepts any pair of patch maps.
  - Also reported: the patch residual mean(y_{t+1} − ŷ_{t+1}) ("B_patch_residual").
- **C:** take the unconstrained best match per patch and keep patches whose Sampson distance w.r.t. the ego F is above 100 px² (independently moving). The embedding is the mean feature delta over those patches.
- Clip embedding = [mean, std] over the 15 pairs.

## Retrieval on maneuver tags (n=414; point estimate [95% CI])
| embedding | R@1 | balanced R@1 | R@5 |
|---|---|---|---|
| mean-pool DINOv2-B | .415 [.365,.464] | .139 [.108,.174] | .725 |
| mean-pool DINOv3-B | .374 [.326,.425] | .122 [.095,.153] | .729 |
| delta mean | .541 [.493,.592] | .340 [.297,.397] | .821 |
| delta [mean,std] | .565 [.512,.616] | .383 [.322,.461] | .829 |
| ctrl middle frame | .394 [.345,.440] | .128 [.101,.159] | .737 |
| ctrl shuffled deltas | .362 [.312,.413] | .135 [.106,.169] | .722 |
| ctrl ego speed+yaw (tags derive from this) | .742 [.693,.787] | .593 [.516,.652] | .923 |
| B DeltaTok ego-comp | .541 [.493,.589] | .343 [.274,.425] | .824 |
| B patch residual | .560 [.510,.609] | .358 [.305,.428] | .754 |
| C Sampson-moving | .478 [.435,.524] | .292 [.214,.338] | .751 |
| delta partialled-out on ego | .481 [.435,.527] | .254 [.194,.335] | .821 |
| [ego, delta] concat | .681 [.638,.725] | .537 [.453,.612] | .884 |

## H2 probe (ridge, GroupKFold by log; per-pair R²)
| tokens | speed | yaw rate |
|---|---|---|
| raw delta | 0.82 | 0.80 |
| B DeltaTok | 0.77 | 0.51 |
| B residual | 0.57 | 0.94 |
| C | 0.21 | 0.01 |

## Subsets: cross-traffic retrieval
Tags inside these subsets are too uniform, so the label here is "cross traffic present": at least one annotated track moving >1 m/s and crossing (|cos|<.5) in the front wedge, taken from AV2 cuboids. R@1 [CI]:

| subset | n | frac cross | meanpool v3 | delta raw | B DeltaTok | B resid | C | ego only |
|---|---|---|---|---|---|---|---|---|
| ego stationary (mean v<.5) | 26 | .65 | .42 | .69 [.52,.84] | .69 | .58 | .58 | .39 |
| fast cruise (v>10, low yaw)* | 39 | .26 | .67 | .62 | .59 | .69 | .54 | .67 |
| all | 414 | .55 | .53 | .57 | .57 | .50 | .59 [.54,.64] | .55 |

\*No AV2 clip exceeds 15 m/s, so the highway subset is empty and >10 m/s is used as a proxy.

## Lane-change S-test
- Rule: lateral rule + yaw rate that has a peak of the lane-change sign >0.03 rad/s followed by an opposite peak.
- Lane-change clips: 79 → 51.
- Manual check of 20 (`lc_check_*.jpg`): 16/20 = **80% precision**, up from about 40% without the S-test. The remaining errors are a driveway/turn and lateral shifts on bends.

## Key findings (interim)
1. Delta tokens beat mean-pooling: balanced R@1 .38 vs .14, with non-overlapping CIs. Shuffled order goes back to the mean-pool level. Ego kinematics alone are still stronger (.59), and adding delta to ego helps a little (.54 vs .59 balanced; R@1 .68 vs .74). Ego alone wins on these ego-derived tags.
2. H2 confirmed: raw delta tokens linearly encode ego speed (R² .82) and yaw rate (.80).
3. The epipolar compensation is only partial:
   - B removes yaw-rate information (.80 → .51) but keeps speed (.77). Searching along the epipolar line still leaves the scale/expansion from forward motion.
   - C removes almost all ego information (speed .21, yaw .01).
   - C's balanced R@1 on ego tags is lower (.29), as expected, since those tags are ego-defined.
4. **For scene dynamics (cross traffic), C is the best single embedding overall (R@1 .59 vs .53 for mean-pool)**, but the CIs overlap. With the ego stationary, raw and B tie at .69.
5. H3 (compensated > raw by ≥3 points on maneuver tags) does not hold on maneuver tags. That is by construction, because the tags are ego-defined. The relevant test is the scene-dynamics label, where C is +2 points over raw with overlapping CIs, so it is inconclusive at n=414.
