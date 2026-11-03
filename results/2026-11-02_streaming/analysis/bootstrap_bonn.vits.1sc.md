### bonn.vits.1sc

1 scenes, 438 frames; FP32 reference: delta1 0.9853, AbsRel 0.0415, TAE 2.318, cov-TAE 1.588, cache 355.9 MB. 95% paired percentile bootstrap over scenes (10000 resamples, seed 0). Positive TAE change = less consistent.

> WARNING: only 1 scenes: the bootstrap has at most 1 distinct resamples, so the CIs are coarse and should not be read as precise.

**Scene-mean aggregation** (TAE metrics as stored by stream_eval)

| Config | bits | cov-TAE | cov-TAE chg % [95% CI] | raw TAE chg % [95% CI] | delta1 chg [95% CI] | AbsRel (chg) |
|---|---|---|---|---|---|---|
| `fp32` | 32.00 | 1.588 | ref | ref | ref | 0.0415 |
| `zero` | 0.00 | 2.186 | +37.7 [n/a] | +25.4 [n/a] | -0.0001 [n/a] | 0.0436 (+0.0021) |

**Frame-weighted aggregation** (weights = frames per scene; delta1 matches stream_eval's pooled aggregate)

| Config | bits | cov-TAE chg % [95% CI] | raw TAE chg % [95% CI] | delta1 chg [95% CI] | stored delta1 chg (no CI) |
|---|---|---|---|---|---|
| `zero` | 0.00 | +37.7 [n/a] | +25.4 [n/a] | -0.0001 [n/a] | -0.0001 |

Notes:
- consistency: max |scene-mean cov-TAE - stored aggregate| = 0.00e+00
