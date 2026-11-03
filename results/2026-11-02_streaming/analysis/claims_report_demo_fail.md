# Claim check: results/2026-11-02_streaming/analysis/claims_demo_fail.tex

facts: 1633 (C:\Users\aniru\vdaquant\results\2026-11-02_streaming\analysis\facts.json); allowlist: C:\Users\aniru\vdaquant\results\2026-11-02_streaming\analysis\claims_allowlist.txt

numbers checked: 5  matched: 0 (tagged 0, specific 0, ambiguous 0)  allowlisted: 0  ignored: 1  UNMATCHED: 4  fact tags: 4 (bad: 4)

Match strength: tagged = covered by a `% fact:` tag; specific = untagged, >= 3 decimals and <= 3 facts share the value; ambiguous = anything else untagged (many facts share such values, so the match proves little: tag it, narrow with --scope, or use --require-tags).

## Unmatched numbers

| line | section | number | context |
|---|---|---|---|
| 4 | Results | `1.712` | `FP32 cov-TAE is $1.712$` |
| 5 | Results | `-0.6%` | `no matching `% fact:` tag: Delta coding raises it by $-0.6\%$.` |
| 6 | Results | `53.07` | `Stale: Sintel FP32 TAE was $53.07$ before the rerun.` |
| 7 | Results | `1.87` | `no matching `% fact:` tag: Typo in the key: $1.87$ bits` |

## Bad fact tags

| line | key | problem |
|---|---|---|
| 4 | `bonn.vits.fp32.covtae` | value 1.71071 not found on the line (numbers there: 1.712) |
| 5 | `bonn.vits.delta_int3_bin_K8.covtae_rel_pct` | value 0.56388 not found on the line (numbers there: 0.6) |
| 6 | `sintel.vits.fp32.tae` | value 53.0267 not found on the line (numbers there: 53.07) |
| 7 | `sintel.vits.delta_int3_bin_K8.bitz` | unknown fact key |

RESULT: FAIL