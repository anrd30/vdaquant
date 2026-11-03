### sintel.vitl

23 scenes, 1045 frames; FP32 reference: delta1 0.7530, AbsRel 0.2022, TAE 52.704, cov-TAE 6.663, cache 1515.9 MB. 95% paired percentile bootstrap over scenes (10000 resamples, seed 0). Positive TAE change = less consistent.

**Scene-mean aggregation** (TAE metrics as stored by stream_eval)

| Config | bits | cov-TAE | cov-TAE chg % [95% CI] | raw TAE chg % [95% CI] | delta1 chg [95% CI] | AbsRel (chg) |
|---|---|---|---|---|---|---|
| `fp32` | 32.00 | 6.663 | ref | ref | ref | 0.2022 |
| `int2_g16` (`int2-g16`) | 2.50 | 6.881 | +3.3 [-3.5, +15.6] | -0.5 [-1.7, +6.7] | -0.0011 [-0.0050, +0.0026] | 0.2022 (0.0000) |
| `zero` | 0.00 | 8.833 | +32.6 [+3.7, +86.3] | +5.1 [-3.1, +65.1] | +0.0011 [-0.0079, +0.0097] | 0.2091 (+0.0069) |

**Frame-weighted aggregation** (weights = frames per scene; delta1 matches stream_eval's pooled aggregate)

| Config | bits | cov-TAE chg % [95% CI] | raw TAE chg % [95% CI] | delta1 chg [95% CI] | stored delta1 chg (no CI) |
|---|---|---|---|---|---|
| `int2_g16` | 2.50 | +7.0 [+0.1, +15.9] | +0.7 [-1.2, +7.1] | -0.0018 [-0.0055, +0.0015] | -0.0018 |
| `zero` | 0.00 | +47.4 [+18.1, +87.4] | +12.8 [-1.0, +69.3] | +0.0001 [-0.0088, +0.0086] | +0.0001 |

Notes:
- 19 frames skipped in delta1/AbsRel (no valid GT) but still counted in per-scene n, so frame-weighted delta1 uses approximate weights (max |frame-weighted delta1 - stored aggregate| = 0.0043; the paired differences are much less affected because ref and config share the weights). TAE / cov-TAE are computed on all frames, so n is the right weight there.
- consistency: max |scene-mean cov-TAE - stored aggregate| = 0.00e+00
