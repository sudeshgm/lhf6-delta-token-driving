Partial Exp0 eval: snapshot of 380 completed logs (760 clips) taken 2026-10-21 ~08:50 IST while extraction continued. Files modified in the last minute were skipped and each npz was load-checked. Same protocol as interim (results/exp0/README.md). Lane-change tag precision is unchanged (S-test, 16/20 = 80%).
Changes vs the n=414 interim:
- C's cross-traffic lead on "all" disappeared (.543 vs meanpool .586 / raw .574).
- B now ties raw on maneuver tags.
- Partial-out costs less: balanced R@1 .369 vs .441 (interim .254 vs .383).
- Delta still beats mean-pool on balanced R@1 (.44 vs .20, CIs disjoint).
- Ego-stationary cross traffic: B .72, raw .67, meanpool .42, ego .33 (n=43).
