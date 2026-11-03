### sintel.vitb

23 scenes, 1045 frames; FP32 reference: delta1 0.7563, AbsRel 0.2090, TAE 55.495, cov-TAE 6.527, cache 720.1 MB. 95% paired percentile bootstrap over scenes (10000 resamples, seed 0). Positive TAE change = less consistent.

**Scene-mean aggregation** (TAE metrics as stored by stream_eval)

| Config | bits | cov-TAE | cov-TAE chg % [95% CI] | raw TAE chg % [95% CI] | delta1 chg [95% CI] | AbsRel (chg) |
|---|---|---|---|---|---|---|
| `fp32` | 32.00 | 6.527 | ref | ref | ref | 0.2090 |
| `int2_g16` (`int2-g16`) | 2.50 | 7.000 | +7.2 [+4.6, +12.6] | +1.1 [-0.7, +11.3] | -0.0007 [-0.0045, +0.0025] | 0.2115 (+0.0025) |
| `delta_int3_bin_K8` (`int3-g16::d8:bin-g16`) | 1.87 | 6.739 | +3.3 [+2.0, +5.4] | -0.1 [-0.6, +2.3] | +0.0003 [-0.0005, +0.0012] | 0.2088 (-0.0003) |
| `zero` | 0.00 | 9.390 | +43.9 [+25.3, +80.1] | +1.5 [-3.5, +27.4] | -0.0055 [-0.0160, +0.0039] | 0.2055 (-0.0035) |

**Frame-weighted aggregation** (weights = frames per scene; delta1 matches stream_eval's pooled aggregate)

| Config | bits | cov-TAE chg % [95% CI] | raw TAE chg % [95% CI] | delta1 chg [95% CI] | stored delta1 chg (no CI) |
|---|---|---|---|---|---|
| `int2_g16` | 2.50 | +8.3 [+5.2, +12.7] | +2.6 [-0.4, +11.9] | -0.0009 [-0.0049, +0.0026] | -0.0009 |
| `delta_int3_bin_K8` | 1.87 | +3.5 [+2.0, +5.4] | +0.3 [-0.5, +2.4] | +0.0001 [-0.0006, +0.0009] | +0.0001 |
| `zero` | 0.00 | +54.0 [+34.2, +82.9] | +6.4 [-1.3, +29.1] | -0.0059 [-0.0164, +0.0037] | -0.0060 |

Notes:
- 19 frames skipped in delta1/AbsRel (no valid GT) but still counted in per-scene n, so frame-weighted delta1 uses approximate weights (max |frame-weighted delta1 - stored aggregate| = 0.0043; the paired differences are much less affected because ref and config share the weights). TAE / cov-TAE are computed on all frames, so n is the right weight there.
- consistency: max |scene-mean cov-TAE - stored aggregate| = 0.00e+00
