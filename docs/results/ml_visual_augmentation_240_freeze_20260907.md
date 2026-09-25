# 240-frame visual augmentation intake freeze

Date: 2026-09-07  
Scope: development data collection and admission only; no training, protected-label access, inference, or promotion.

## Outcome

`visual-augmentation-240-v2` is frozen with 240/240 reviewed and admitted candidate frames:

- 144 material/lighting positives: 24 new poses × 3 cross-class materials × 2 lighting conditions.
- 48 regular-condition positives: 48 new poses.
- 48 isolated hard negatives: 24 new poses × 2 lighting conditions, with empty `full_2d` truth verified before review.

The dataset contains 96 independent pose groups. Designed variants retain shared lineage and are not counted as independent poses. All review decisions are marked `AI-assisted` and bind image and label hashes.

## Review and isolation checks

- Positive review: 192 accepted, 0 held; 24 six-way appearance groups and 48 regular views.
- Hard-negative review: 48 accepted, 0 held; 24 complete lighting pairs.
- Internal exact-file overlaps: 0.
- Internal decoded original-pixel overlaps: 0.
- Protected/reference file-fingerprint overlaps: 0.
- Protected/reference pixel-fingerprint overlaps: 0.
- The flawed v1 negative subset is superseded by `hard-negative-isolated-v2`; historical files were not modified or deleted.

Frozen ledger identity: `c8b93067eb823497843b185806720d34f425d76caade1e53e1b98b7638a68eed`  
Frozen ledger file SHA-256: `12f31aa26686edb6b5c086f8c466b8dd5f1213b3eb0c6610b234087096b89bba`  
Positive review identity: `72a6af0f60e1db73cc1927127989d1a0b650640f78eb9d7502790db0f5c9330e`

## Verification

- Relevant canonical, pairing, shutdown, negative-isolation and intake tests: 19 passed.
- Frozen v2.11 baseline contract: 40 pinned files verified; package valid; development/evaluation identity overlap 0.
- Every new artifact remains `training_admitted=false` and `promotable=false`.

## Next gate

Do not start A/B/C/D training yet. First freeze the uniformly reviewed common trusted training pool, the identical initialization, optimizer, step budget, and exact per-arm membership/exposure schedule. The fixed 48-frame paired set remains development regression only; the unseen-scene test stays sealed until a development candidate is selected.
