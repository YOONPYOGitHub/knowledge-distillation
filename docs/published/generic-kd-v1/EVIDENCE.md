# Portable publication evidence coverage

## Scope and trust boundary

This curation freshly verifies **local retained bytes and arithmetic only**. It does not train models, score heldout again, load weights, contact cloud services, freshly authenticate historical Blob objects or prove all past nonaccess. The original Korean report body is preserved verbatim beneath an added publication header. Its remote verification and review statements remain historical statements, not newly exercised gates. Its scientific conclusions and limitations are unchanged.

The publication manifest pins the original report SHA `e9c17e82e4f9ad843881dbd394e997dad14f8fc23f49c1ec7021e92e384e3f30` and the new published SHA. It also records all source files used and explicit transformations. Sources present in the original publication manifest were compared to that retained manifest's byte/SHA pins. The original manifest is itself a retained provenance object, not an external signature.

## Portable coverage of historical report references

| Historical reference | Portable evidence / bounded disposition |
|---|---|
| `results.csv`, original24 final JSONs | [results index](../../../results/published/generic-kd-v1/results.csv), `published_path` resolves all24 byte-identical completed aggregate JSONs; original hashes are retained. |
| `paired_effects.csv` | [all9 pairs](../../../results/published/generic-kd-v1/paired_effects.csv), byte-identical original; verifier recomputes every paired effect and ordinary adaptation effect. |
| `aggregation-verification.json` | Historical acquisition/aggregation record is not republished. Current [verifier](../../../results/published/generic-kd-v1/verify_results.py) rechecks local result hashes, protocol IDs, CE sum/tokens, PPL and CSV values. |
| `selected-checkpoint-links.csv`, selected contract/detail projections | [Path-free joined metadata](../../../results/published/generic-kd-v1/selected-metadata.json) carries all21 models/revisions/contract/step/optimizer/settings and historical manifest/COMPLETE hashes. Three source projections agree on selection step/contract/settings. Absolute VM checkpoint paths are intentionally omitted, not relabeled as portable downloads. |
| `runs.csv`, compact report/identity projections, Student/Teacher partial CSVs | [Run census](../../../results/published/generic-kd-v1/run-census.json) is a projection of89 reported rows: commonStudent60, formalTeacher3, historicalPilot16, diagnostic10. Not exhaustive history, independent seeds or fresh optimizer/checkpoint authentication. Full operational rows are not shipped. |
| Canonical raw manifest | [Exact metadata manifest](../../../results/published/generic-kd-v1/cohort-manifest.json), canonical SHA `53e4e887ec898c2cce985a8027f0dec33360761ba16f705a3c95d244887f904a`; train6814/validation851/test851, FineWeb2 `kor_Hang`. Metadata only; referenced split files never opened. |
| Frozen prepared identities and full selection→cohort source chain | Prepared-data/cohort identity hashes from retained selected projections are preserved. Canonical metadata plus these hashes do not constitute fresh verification of all prepared payloads or complete historical exposure adjudication. |
| `FAILURES.md`, historical inventory908, transport chunks, live snapshots, review logs, closure/RUNBOOK and source capsules | Intentionally not copied. Report statements remain bounded historical claims. No new failure census, live observation, remote authentication or full-capsule proof is asserted. Historical bare filenames in the unchanged report body are not broken publication links to invented equivalents. |
| Original `manifest.json` and delivery ZIP | New [manifest](../../../results/published/generic-kd-v1/manifest.json) covers only this curated publication. Original ZIP and50-file operational archive are not redistributed. |

## Reproduction limits

- Aggregates contain only version/model ID/protocol SHA/status and `{ce_sum,tokens}`. No per-document/cluster losses, confidence intervals, input text, input IDs, gold answers or model predictions are available here. The nine effect directions are verified arithmetic, not significance tests.
- Protocol hashes are checked against the retained original CSV and published manifest pins. Full protocol/source execution is not revalidated simply because the identities match.
- Same raw document identities do not establish identical predicted-byte coverage across tokenizer families; do not make a cross-family PPL ranking.
- Teacher numerical_policy=null at one projected identity location is missing-field evidence only. Scheduler evidence is not present. Public model licensing/gating is not freshly reviewed and no weights are redistributed.
- No complete GPU-time/cost accounting, quantized device measurements, QA/grounding, privacy/safety or clinical/service-readiness certification is supplied.

Run the read-only CLI from [results README](../../../results/published/generic-kd-v1/README.md). Its exact inventory includes its own code and tests, excludes the manifest itself, and treats all `source_path` values strictly as provenance strings. No source/corpus path is followed.
