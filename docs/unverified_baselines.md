# Unverified Literature Baselines — QUARANTINED

**Status: NOT independently verified. DO NOT print, plot, or export these
numbers alongside measured results.** See docs/optimization_ledger.md
finding F2.

These numbers were previously hardcoded directly in
`scripts/run_pareto_benchmark_suite.py` as `PUBLISHED_BASELINES` and
embedded into every JSON export next to real measured results, with no
citation trail. At audit time (commit `68bb5a2`) no source could be
confirmed for the specific `RTN_4bit_Literature` / `SmoothQuant_4bit`
numbers on Video-Depth-Anything specifically — they may have been
transcribed from a different model/paper, approximated, or fabricated as
placeholders during earlier development. Until each row below has a real
citation (paper, table number, exact eval protocol) attached, treat this
entire file as **not usable in any publication, plot, or comparison table**.

Moved out of the script (rather than deleted) so the numbers aren't lost,
but physically separated so they can never again be silently embedded next
to real measured output. To reinstate a data source, add a citation next to
its row and move only that row back into a script explicitly for citation
plotting.

## KITTI
| Config | delta1 | delta2 | delta3 | AbsRel | RMSE | FPS rel | Mem rel |
|---|---|---|---|---|---|---|---|
| FP32_Paper_Baseline | 0.952 | 0.990 | 0.998 | 0.081 | 2.940 | 1.0 | 1.0 |
| RTN_4bit_Literature | 0.610 | 0.820 | 0.910 | 0.285 | 6.120 | 1.1 | 8.0 |
| SmoothQuant_4bit | 0.840 | 0.945 | 0.980 | 0.142 | 4.100 | 1.1 | 8.0 |

## DAVIS
| Config | delta1 | delta2 | delta3 | AbsRel | RMSE | FPS rel | Mem rel |
|---|---|---|---|---|---|---|---|
| FP32_Paper_Baseline | 0.965 | 0.992 | 0.999 | 0.072 | 0.185 | 1.0 | 1.0 |
| RTN_4bit_Literature | 0.650 | 0.840 | 0.920 | 0.240 | 0.450 | 1.1 | 8.0 |
| SmoothQuant_4bit | 0.860 | 0.950 | 0.985 | 0.130 | 0.290 | 1.1 | 8.0 |

## Sintel
| Config | delta1 | delta2 | delta3 | AbsRel | RMSE | FPS rel | Mem rel |
|---|---|---|---|---|---|---|---|
| FP32_Paper_Baseline | 0.910 | 0.975 | 0.992 | 0.115 | 1.120 | 1.0 | 1.0 |
| RTN_4bit_Literature | 0.580 | 0.790 | 0.890 | 0.310 | 2.850 | 1.1 | 8.0 |
| SmoothQuant_4bit | 0.810 | 0.920 | 0.970 | 0.175 | 1.820 | 1.1 | 8.0 |

## NYUv2
| Config | delta1 | delta2 | delta3 | AbsRel | RMSE | FPS rel | Mem rel |
|---|---|---|---|---|---|---|---|
| FP32_Paper_Baseline | 0.890 | 0.970 | 0.991 | 0.105 | 0.420 | 1.0 | 1.0 |
| RTN_4bit_Literature | 0.550 | 0.760 | 0.870 | 0.320 | 0.950 | 1.1 | 8.0 |
| SmoothQuant_4bit | 0.780 | 0.910 | 0.965 | 0.180 | 0.650 | 1.1 | 8.0 |

## ScanNet
| Config | delta1 | delta2 | delta3 | AbsRel | RMSE | FPS rel | Mem rel |
|---|---|---|---|---|---|---|---|
| FP32_Paper_Baseline | 0.905 | 0.978 | 0.993 | 0.098 | 0.380 | 1.0 | 1.0 |
| RTN_4bit_Literature | 0.570 | 0.780 | 0.880 | 0.295 | 0.880 | 1.1 | 8.0 |
| SmoothQuant_4bit | 0.800 | 0.925 | 0.972 | 0.165 | 0.580 | 1.1 | 8.0 |
