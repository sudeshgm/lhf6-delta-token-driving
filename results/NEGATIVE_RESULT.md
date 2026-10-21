# LHF6: negative result (stopped 2026-10-21)

**Verdict: KILL.** Delta tokens do not give a better *retrieval* representation of driving scenarios once ego motion is controlled for.

## Findings
- **H1, delta tokens beat mean-pool on maneuver retrieval: supported, but trivially.**
  - Exp0, n=760: [mean,std] delta reaches balanced R@1 .44 [.40,.49], against .20 [.16,.24] for mean-pool DINOv3.
  - Shuffled-order deltas fall back to .18, so the gain needs temporal order.
  - Ego speed+yaw alone does better (.65), and these tags are ego-defined.
- **H2, raw deltas are dominated by ego motion: supported.** A ridge probe on per-pair tokens gives ego speed R² .82 and yaw rate R² .84. The delta token behaves like an odometer.
- **H3, ego-compensated deltas beat raw by ≥3 points: unsupported.**
  - Compensation is epipolar, using known pose + calibration (B: epipolar-constrained feature prediction → DeltaTok; C: Sampson-moving patches).
  - B mostly removes yaw (R² .84 → .61) but not speed (.77). C removes both (.23 / .05).
  - Neither beats raw on maneuver tags or on cross-traffic labels by ≥3 points with non-overlapping CIs.
- **Ego-fixed kill test (Exp1), pre-registered: FAILED.**
  - Setup: 5 track-derived labels (cross traffic, pedestrian/cyclist crossing, lead braking, cut-in, oncoming turn). Retrieval is restricted to the same ego-maneuver × speed-tercile bin, excluding same-log clips.
  - No delta variant beats both ego-only and mean-pool v3 by ≥3 balanced R@1 with non-overlapping CIs on any label. The rule needed 2.
  - Closest: cut-in, where B scores .628 [.59,.67] against mean-pool .549 [.51,.59]; the intervals touch.
- **Probe note:** a supervised logistic probe (log-grouped CV) ranks B DeltaTok best on all 5 ego-fixed labels, with AUROC .69–.76 against .61–.67 for mean-pool. Ego-independent scene dynamics *are* linearly present in the delta tokens, but unsupervised cosine kNN does not recover them. A learned metric might; that falls outside this premise.

## Caveats
- Everything ran on CPU at 256 px, while DeltaTok's native resolution is 512 px. Lens distortion was ignored in F.
- No HD map and no RefAV: the ego lane is approximated as |lateral| < 1.8 m. The cut-in label is noisy, since ego lane changes can trigger it.
- Lane-change tag precision is about 80% (S-test, 16/20 hand-checked). Other ego tags are about 98%.
- Results come from a partial snapshot: n=760 clips / 380 logs out of 1698 / 850. A full-run re-evaluation is pending and will be auto-pushed to `results/exp0/full/` and `results/exp1_egofixed/full/` (with `VERDICT.txt`).

Details: `results/exp-1_block1_feasibility.md`, `results/exp0/`, `results/exp1_egofixed/`.
