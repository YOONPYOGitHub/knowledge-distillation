# Portable generic KD results v1

- [Publication entry/report](../../../docs/published/generic-kd-v1/README.md)
- [Evidence and limitations](../../../docs/published/generic-kd-v1/EVIDENCE.md)
- [results.csv](results.csv): 24 result rows; `published_path` is relative to this directory and resolves every original aggregate JSON. All existing scientific columns are unchanged. `original_path`/`original_sha256` retain repository-relative archival provenance; they are not required to exist in a portable checkout.
- [paired_effects.csv](paired_effects.csv): original byte-identical 9 paired comparisons. `(KD/CE-1)*100`; ordinary adaptation `(1-CE/pretrained)*100`. Seed42 selection reuse is explicit.
- [selected-metadata.json](selected-metadata.json): path-free 21 selected contracts (Teacher3+Student18), models/revisions, steps, settings, optimizer groups and historical identity hashes; not checkpoint downloads or fresh payload authentication.
- [cohort-manifest.json](cohort-manifest.json): byte-identical canonical **metadata**, including split file hashes and historical exclusion IDs/hashes. No referenced corpus leaf is included or read by the verifier.
- [run-census.json](run-census.json): reported execution-count projection, not independent seeds or complete historical census.
- [manifest.json](manifest.json): exact published inventory with byte counts/SHA-256 and source→publication provenance, including original report hash. It excludes only itself. Source inventory hashes are historical provenance, not files the verifier reads.

## Offline verification

Python3 standard library only; from repository root (or an extracted bundle retaining these two directories):

```sh
python3 -B results/published/generic-kd-v1/verify_results.py
python3 -B results/published/generic-kd-v1/test_verify_results.py
```

The CLI reads only the two publication directories, forbids writes/network/subprocesses after imports, checks exact inventory/hash/bytes, completed aggregate schema, protocol identity, token-weighted CE and exp(CE), all9 effects and adaptation effects, selected metadata pairing and canonical metadata hash. It never follows original/source/corpus/checkpoint paths. Tests use clearly authored invalid fixtures and disposable publication copies, never invented research results. Hash integrity is not an external digital signature; use an independently trusted manifest SHA when authenticity is required.

Excluded deliberately: raw text, prepared token IDs, prompts, outputs/predictions, QA heldout data, model/adapter weights, operational transfer logs, cloud resource IDs/auth output, support contacts, historical ZIP and full historical inventory. Historical authenticated remote evidence in the report is not a fresh remote certification by this publication.
