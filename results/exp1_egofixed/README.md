# Exp1: ego-fixed scene-label retrieval (Block 3)

**Data:** 760 clips / 380 logs, from the partial_380 snapshot of features/full256 (256 px, CPU).

**Labels** (`src/scene_labels.py`):
- Built from AV2 cuboid tracks only. No ego tag is used, and **no HD map or RefAV** is used: RefAV needs the av2 map API and was not set up. The ego lane is approximated as |lateral| < 1.8 m.
- Definitions are in the script docstring.
- Caveat: cut_in is noisy (ego-frame lateral, no map; the ego vehicle's own lane changes can also trigger it).

**Retrieval protocol:**
- Candidates must be in the same ego bin: maneuver tag × mean-speed tercile, 24 bins.
- Same-log candidates are excluded.
- 95% CIs come from a 1000× bootstrap over logs.
- AUROC comes from logistic regression (C=.05, standardised) with GroupKFold(5) by log.

**Re-run on full features:** `python src/scene_labels.py && python src/evaluate_egofixed.py features/full256 results/exp1_egofixed/full`

## Prevalence (positives / 760)
- cross_traffic: 309 (.41)
- vru_crossing: 172 (.23)
- lead_braking: 78 (.10)
- cut_in: 187 (.25)
- oncoming_turn: 113 (.15)

No label was dropped; all have at least 15 positives.

## Balanced R@1 [95% CI], ego-bin matched
| feature | cross | vru | lead_brake | cut_in | oncoming_turn |
|---|---|---|---|---|---|
| meanpool v2 | .563 [.53,.60] | .538 [.50,.58] | .584 [.54,.63] | .561 [.52,.60] | .538 [.50,.57] |
| meanpool v3 | .561 [.53,.59] | .547 [.51,.59] | .574 [.53,.63] | .549 [.51,.59] | .550 [.51,.59] |
| middle frame | .537 | .538 | .593 | .557 | .501 |
| shuffled delta | .563 | .536 | .555 | .574 | .507 |
| ego only | .526 [.49,.56] | .537 [.50,.57] | .539 [.50,.58] | .559 [.52,.60] | .533 [.49,.57] |
| raw delta [mean,std] | .566 [.53,.60] | .548 [.51,.59] | .559 [.52,.60] | .623 [.58,.66] | .516 [.48,.56] |
| B DeltaTok | .554 [.52,.59] | .536 [.50,.58] | .567 [.52,.62] | **.628 [.59,.67]** | .556 [.51,.60] |
| B residual | .549 | .527 | .573 | .575 | .526 |
| C Sampson | .534 | .525 | .538 | .621 [.58,.66] | .531 |
| [ego, delta] | .551 | .554 | .594 | .587 | .556 |

## Probe AUROC (log-grouped CV)
| feature | cross | vru | lead_brake | cut_in | oncoming_turn |
|---|---|---|---|---|---|
| meanpool v2 | .664 | .675 | .650 | .616 | .637 |
| meanpool v3 | .631 | .647 | .674 | .649 | .609 |
| middle frame | .626 | .646 | .671 | .578 | .605 |
| shuffled delta | .603 | .684 | .638 | .609 | .589 |
| ego only | .607 | .641 | .571 | .637 | .568 |
| raw delta | .680 | .674 | .720 | .736 | .663 |
| **B DeltaTok** | **.691** | **.693** | **.760** | **.763** | **.694** |
| B residual | .678 | .587 | .709 | .706 | .599 |
| C Sampson | .588 | .633 | .597 | .695 | .633 |
| [ego, delta] | .685 | .677 | .720 | .737 | .663 |

## Kill-rule verdict (pre-registered)
**PREMISE FAILS.** No delta variant beats both ego-only and meanpool-DINOv3 by at least 3 balanced-R@1 points with non-overlapping CIs on even one label, let alone the two required.
- The closest case is cut_in. B scores .628 [.59,.67] and raw .623 [.58,.66], against meanpool v3 .549 [.51,.59] and ego .559 [.52,.60]. The point gains are +7–8 points, but the CIs touch or overlap.
- Everything else is within about ±3 points of chance-level .50–.59.

**Side note** (not part of the kill rule): in a supervised linear probe, B DeltaTok is the best feature on all 5 labels. Its AUROC is +.03 to +.11 above meanpool v3, for example lead_braking .76 vs .67 and cut_in .76 vs .65. So the delta tokens *contain* ego-independent dynamics, but unsupervised cosine retrieval does not surface them.
