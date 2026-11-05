# Generic final-evaluation metadata fixtures

These fixtures preserve authentic archived contract metadata for offline regression tests. They are **not synthetic research results**, a corpus, checkpoint payloads, or evidence that referenced remote payloads are currently available. Tests may construct explicitly synthetic metadata envelopes around the original contracts; CPU fake-model outputs remain synthetic plumbing evidence.

## Provenance and exactness

`PROVENANCE.json` records each original repository-relative source path, original byte count and SHA-256, fixture byte count and SHA-256, and the exact retained JSON pointers with canonical node SHA-256 hashes. Canonical node hashes use UTF-8 JSON with sorted keys, compact comma/colon separators, `ensure_ascii=False`, `allow_nan=False`, and no trailing newline. Every retained node was checked for structural and canonical-byte equality against its original metadata source. Sources were not modified.

- The three `*-originals-transfer/originals.json` files are **metadata projections, not original archive files**. They retain only the complete, unchanged `checkpoint_contract` nodes for selected CE `ce-2` and KD `kd-1` trials. The round2 projection additionally retains `plan.json/provenance`. Metrics, reports, unselected trials and other archive nodes are omitted.
- `final-teacher-identity-20261026.json` is a projection retaining only unchanged `identity` and `settings` nodes.
- `final-cohort-auth-20261026T1205.json` is a projection retaining only the unchanged `cohort` node.
- `final-teacher-original-manifest-20261026.json` is a **byte-identical metadata copy**. Its original SHA-256 is required by the cohort's Teacher manifest binding; its parameter names and payload size/hash inventory are metadata, not tensors or payload bytes.
- `GENERIC_CPT_EXPOSURE_ADJUDICATION.md` is a **byte-identical historical adjudication copy**. Its explicit uncertainty and historical scope remain unchanged. The source-document SHA-256 assertion in the tests is unchanged.

Only the seven explicit metadata source files were used. No referenced corpus, train/dev/test text, heldout data, QA examples, model weights, optimizer state payloads, cloud endpoints or credentials were opened. Historical metadata source paths and hashes identify provenance; they are not download instructions or renewed execution authorization. Live VM/acquisition status and other unneeded fields were excluded from the identity/cohort projections.

The fixture hashes authenticate the published fixture contents, not independent historical acquisition or current remote custody. Model revisions, trial/seed identities, contract nodes, data fingerprints, dependency hashes and all existing fail-closed regression assertions remain unchanged.
