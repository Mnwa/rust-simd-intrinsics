# Evaluate the skill, not the number of vector types

Run the same agent without the skill, with the original skill, and with this
supplement. Keep model, decoding configuration, tool access, budget, compiler,
target and task seed fixed. Use multiple seeds. Do not show the complete task
catalog or hidden test corpus to the candidate agent during a run.

`tasks.json` intentionally uses variants rather than copying recipe functions.
Generate a private held-out corpus with the listed boundary cases. The evaluator
must compile and execute each solution and review memory/semantic safety. This
repository does not pretend to run an LLM evaluation by itself.

Store per-attempt results using `results.schema.json`, then run:

```bash
python3 scripts/score-vectorization-evals.py results.json
```

The scorer checks statuses and numeric measurements, reports all failures in the
denominator, and calculates performance only for compile/correctness/safety passes.
It warns when variants do not have exactly matched task/configuration/seed sets.
A skipped or unsupported run is not a successful solution. Measurements are null
until actually collected. The included example JSON is explicitly **not a result**.

## Acceptance criteria

Correctness and safety are mandatory. Performance is measured against the best
correct scalar-written release baseline with autovectorization enabled and matched
build flags. If the baseline wins, retaining it with evidence is acceptable. Do
not demand a fixed speedup on every task or average away correctness failures.

Record public-call and preselected-kernel costs separately when relevant. Use a
held-out input distribution and report regressions, not just best cases. A real
production task should add end-to-end measurements (including layout conversion,
allocation or table setup that its caller pays).

Status schema and scorer are tools for collecting evidence, not an automated
proof of unsafe-code validity or a completed agent study.
