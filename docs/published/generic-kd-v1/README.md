# Korean generic KD v1 — public portable results

Start with the [completed Korean report](REPORT.md) and [evidence coverage](EVIDENCE.md). This is a curated metadata/aggregate publication, not a model-weight, corpus or operational-log release.

## Final heldout outcome

Negative CE→KD relative PPL change is better. Each row compares the same Student/tokenizer; do not rank families using token PPL.

| Family | Student seed | CE PPL | KD PPL | KD relative change |
|---|---:|---:|---:|---:|
| qwen35 | 42 | 14.862177175 | 14.823483215 | -0.260352% |
| qwen35 | 43 | 14.883492218 | 14.803128918 | -0.539949% |
| qwen35 | 44 | 14.898702584 | 14.823911765 | -0.501996% |
| gemma4 | 42 | 12.261580431 | 12.365465145 | +0.847238% |
| gemma4 | 43 | 12.278857508 | 12.353458288 | +0.607555% |
| gemma4 | 44 | 12.296883945 | 12.262601114 | -0.278793% |
| lfm25 | 42 | 17.591614757 | 17.574565690 | -0.096916% |
| lfm25 | 43 | 17.629134006 | 17.576804421 | -0.296836% |
| lfm25 | 44 | 17.544124248 | 17.557484472 | +0.076152% |

Qwen improves slightly in all three paired seeds; Gemma and Liquid are mixed. These small effects do not establish general KD superiority, statistical significance, practical benefit, grounded Korean QA ability, medical safety or service readiness. No confidence intervals can be reconstructed from these token-weighted aggregates.

## Actual selected settings and execution units

- Student CE/KD learning rates: Qwen `8e-5`, Gemma `1.5e-4`, Liquid `1.25e-4`. KD alpha is the **CE** weight: `0.7`, `0.7`, `0.9` respectively; KL weight is `1-alpha`. Selected KD uses forward KL, temperature `1.0`.
- The retained selected contract/plan projections, actual optimizer groups, selection steps, model revisions and initialization hashes are in [selected-metadata.json](../../../results/published/generic-kd-v1/selected-metadata.json), not inferred from smoke configs. AdamW betas `(0.9,0.999)`, eps `1e-8`, weight decay `0.01`; Teacher LR `2e-5`. LoRA rank8/alpha16/dropout0.05, seq512/batch1 are report-level settings, not newly checked weight properties. Scheduler evidence is unavailable in this projection.
- Budget: two search rounds/family, CE4/KD4 candidates/round, maximum 3,000 updates/arm. Validation every200, checkpoint every100, patience3, min_delta0.0001. Actual completed early stopping is not a failure or proof of convergence.
- [Reported run census](../../../results/published/generic-kd-v1/run-census.json): common Student60 = CE10/KD10 per family; formal Teacher3 separately. Historical pilot16 and small diagnostic10 are separate. This is not a complete historical execution census or 60 independent seeds. Student seed42 is reused selection; only seeds43/44 are additional fixed-setting pairs. One fixed Teacher/family conditions all Student seeds.

## Data and portable artifacts

[Canonical metadata manifest](../../../results/published/generic-kd-v1/cohort-manifest.json) is byte-identical to SHA-256 `53e4e887ec898c2cce985a8027f0dec33360761ba16f705a3c95d244887f904a`. It identifies FineWeb2 `kor_Hang`, pinned revision, deterministic split policy and counts: train6814, validation851, test851. It contains exclusion source IDs/text hashes, not corpus text. The referenced train/validation/test files were **not opened**, copied or rehashed for this publication. No claim is made that an exclusion hash alone proves the complete historical exposure record.

[Results README](../../../results/published/generic-kd-v1/README.md) links all24 original aggregate JSONs via the portable results index, all9 paired effects, provenance and the offline stdlib verifier. Keep the two publication directories in their repository-relative positions; no other repository files are required for verification. Archived source paths in provenance are identifiers, not downloadable or relocated payloads.
