### hard.sintel.vits

8 scenes (offline T=32, backbone + temporal quantized). FP32: delta1 0.6876, AbsRel 0.2793, TAE 123.49, cov-TAE 8.564. Bits are the stored effective_bits_per_scalar. Raw-TAE CI: paired bootstrap over scenes (10000 resamples, seed 0); cov-TAE has no per-scene data here.

| Config | bits | delta1 | AbsRel | cov-TAE | cov-TAE chg % (point) | raw TAE | raw TAE chg % [95% CI] | note |
|---|---|---|---|---|---|---|---|---|
| `fp32` | 32.00 | 0.6876 | 0.2793 | 8.56 | ref | 123.49 | ref | |
| `int4_g64` | 4.125 | 0.6879 | 0.2801 | 9.68 | +13.0 | 123.89 | +0.3 [-2.5, +3.6] |  |
| `int4_tensor` | 4.125 | 0.6314 | 0.3106 | 14.49 | +69.1 | 105.59 | -14.5 [-21.6, -6.6] |  |
| `e8_g16_b3` | 3.500 | 0.6836 | 0.2870 | 12.06 | +40.8 | 121.71 | -1.4 [-5.5, +3.5] |  |
| `int3_g16` | 3.500 | 0.6869 | 0.2730 | 11.08 | +29.3 | 123.99 | +0.4 [-3.9, +4.0] |  |
| `e8_g32_b3` | 3.250 | 0.6775 | 0.2847 | 12.44 | +45.3 | 123.19 | -0.2 [-8.8, +6.8] |  |
| `lam16_g64_b3` | 3.125 | 0.6225 | 0.3292 | 52.69 | +515.3 | 106.11 | -14.1 [-23.1, -0.5] |  |
| `e8_g64_b3` | 3.125 | 0.6836 | 0.2824 | 14.97 | +74.8 | 121.79 | -1.4 [-6.2, +6.0] |  |
| `int3_g64` | 3.125 | 0.6938 | 0.2617 | 14.03 | +63.8 | 119.33 | -3.4 [-12.8, -0.9] |  |
| `int3_tensor` | 3.125 | 0.4509 | 0.4360 | 63.88 | +645.8 | 67.83 | -45.1 [-64.5, +7.7] | collapsed output: read with delta1 |
| `e8_g16_b2` | 2.500 | 0.4886 | 0.4724 | 28.92 | +237.7 | 35.14 | -71.5 [-74.6, +17.2] | collapsed output: read with delta1 |
| `int2_g16` | 2.500 | 0.4187 | 0.4633 | 6.38 | -25.5 | 6.43 | -94.8 [-98.7, +0.3] | collapsed output: read with delta1 |
| `e8_g64_b2` | 2.125 | 0.4252 | 0.4396 | 7.61 | -11.2 | 7.76 | -93.7 [-97.6, +10.5] | collapsed output: read with delta1 |
| `int2_g64` | 2.125 | 0.4069 | 0.4661 | 5.96 | -30.4 | 5.99 | -95.1 [-98.5, +10.0] | collapsed output: read with delta1 |

Notes:
- hard setting: per-scene data is raw TAE only (8 scenes, no per-scene cov-TAE / delta1); cov-TAE and delta1 shown as aggregates without CI
