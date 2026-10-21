#!/bin/bash
# Follow-on to lhf6_finalize.sh: after Block 2 full eval (FINAL_DONE), run Exp1 ego-fixed on full features + verdict, commit, push.
cd /workspace/lhf6-delta-token-driving
while [ ! -f results/exp0/FINAL_DONE ] && pgrep -f lhf6_finalize.sh >/dev/null; do sleep 60; done
# if the first watcher died without finishing, run Block 2 full eval ourselves once extraction ends
while pgrep -f extract2.py >/dev/null; do sleep 60; done
[ -f results/exp0/full/metrics.json ] || /workspace/venv_lhf6/bin/python src/evaluate2.py features/full256 results/exp0/full > results/exp0/full_eval.log 2>&1
/workspace/venv_lhf6/bin/python src/scene_labels.py > results/exp1_egofixed/full_labels.log 2>&1 && \
/workspace/venv_lhf6/bin/python src/evaluate_egofixed.py features/full256 results/exp1_egofixed/full > results/exp1_egofixed/full_eval.log 2>&1
/workspace/venv_lhf6/bin/python - <<'PY'
import json
r = json.load(open("results/exp1_egofixed/full/metrics.json"))
w = r["kill_rule_wins"]; ok = any(len(v) >= 2 for v in w.values())
txt = (f"Exp1 ego-fixed kill rule on FULL features (n_clips={r['n_clips']}, n_logs={r['n_logs']})\n"
       "Rule: PASS if any delta variant (raw, B, B-residual, C) beats BOTH ego-only and meanpool-DINOv3 by >=3 balanced R@1 points "
       "with non-overlapping log-bootstrap CIs on >=2 labels.\n"
       f"Labels with wins per variant: {json.dumps(w)}\nVERDICT: {'PASS (premise survives)' if ok else 'FAIL (premise killed)'}\n")
open("results/exp1_egofixed/full/VERDICT.txt", "w").write(txt); print(txt)
PY
git pull -q --rebase; git add results/exp0 results/exp1_egofixed && git commit -qm "Full-run re-eval: Block 2 + Exp1 ego-fixed with VERDICT.txt (auto)" && git push -q
echo DONE > results/exp1_egofixed/FULL_DONE
