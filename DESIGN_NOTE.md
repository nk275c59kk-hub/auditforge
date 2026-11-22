# AuditForge: design note

## Objective
Provide a deterministic, read-only, Gymnasium-style accounting audit benchmark with four injected anomaly classes and private scoring. The system is **not** an LLM-based auditor; the baseline is a deterministic rule-based agent.

## Architecture
- `generator/data_generator.py`: seed-based synthetic company with balanced journal transactions and invoices, payments, and monthly FX tables.
- `generator/error_generator.py`: copies clean data, injects one example of each anomaly, writes public evidence and private truth/provenance.
- `environment/auditor_env.py`: actual team-contributed environment. Restricts the agent to `list_records`, `get_record`, `get_document`, `calculator`, `submit`; enforces <=100 actions and keeps private sections out of observations.
- `evaluator/reward.py`: strict matching of type, affected IDs, currency, amount (minor units), and required evidence. One-to-one matching; F1-based reward and inspection penalty.
- `agents/baselines.py`: silent, noisy, duplicate-only, and public-evidence rule-based auditor.
- `build_submission.py`: deterministic train/dev episodes, public-only trajectories, score summaries.

## Data and threat model
The agent receives only the public directory. The evaluator alone reads private truth. Generated baseline trajectories are agent-visible observations only. Keep private truth outside any agent prompt, tool output, or shared public archive.

## Simplifications and limitations
- The FX corruption is checked against an independent approved rate table included in public policy; this is an intentionally simple benchmark mechanism, not a realistic multi-currency accounting conversion.
- Amounts in generator rows use whole currency units, while truth and submissions use minor units (x100). This is a demo convention and is not correct for all currencies (e.g., zero-decimal currencies). It should be revised for production.
- One corruption of each type is injected per episode; the baseline's high score on these seeds does **not** demonstrate generalization to different errors or real financial statements.
- Synthetic records are not regulatory-grade accounting data, and this package is not a financial compliance product.
- No external LLM is integrated. The baseline is a rule-based control for benchmarking and reproducibility.
- The public approved FX reference data makes FX anomalies easy to find by direct comparison. A harder benchmark would use independent dated source documents and ledger currency conversion.

## Reproducibility
From project root run `python main.py test`, `python main.py demo`, `python build_submission.py`. No dependencies beyond Python standard library. Test on multiple seeds. Never infer official challenge compliance from internal test scores.
