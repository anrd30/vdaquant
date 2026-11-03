# Claim check: C:/Users/aniru/Downloads/cvpr2027/paper/main.tex

facts: 1633 (C:\Users\aniru\vdaquant\results\2026-11-02_streaming\analysis\facts.json); allowlist: C:\Users\aniru\vdaquant\results\2026-11-02_streaming\analysis\claims_allowlist.txt

numbers checked: 457  matched: 243 (tagged 0, specific 67, ambiguous 176)  allowlisted: 1  ignored: 0  UNMATCHED: 213  fact tags: 0 (bad: 0)

Match strength: tagged = covered by a `% fact:` tag; specific = untagged, >= 3 decimals and <= 3 facts share the value; ambiguous = anything else untagged (many facts share such values, so the match proves little: tag it, narrow with --scope, or use --require-tags).

## Unmatched numbers

| line | section | number | context | nearest facts |
|---|---|---|---|---|
| 56 | (front matter) | `0.900` | `At $3.5$ bits it reaches $\delta_1 = 0.900 \pm 0.005$ on NYU and` |  |
| 57 | (front matter) | `0.897` | `$0.897 \pm 0.003$ on KITTI with` |  |
| 57 | (front matter) | `0.910` | `0.897 \pm 0.003$ on KITTI with ViT-S (FP32: $0.910$, $0.928$; five` |  |
| 57 | (front matter) | `0.928` | `0.003$ on KITTI with ViT-S (FP32: $0.910$, $0.928$; five` | sintel.vits.int2_g16_rht.tae_rel_pct_ci_lo=-0.926234 |
| 58 | (front matter) | `98.9%` | `rotation seeds), retaining $98.9\%$ and over $96\%$ of FP` | hard.sintel.vits.int2_g16.tae_rel_pct_ci_lo=-98.7184; hard.sintel.vits.int2_g16.tae_rel_pct_fw_ci_lo=-98.7184 |
| 89 | Introduction | `0.516` | `$0.711$ to $0.516$. Second, VDA's temporal` |  |
| 114 | Introduction | `98.9%` | `$3.5$ bits ViT-S retains $98.9\%$ (NYU) and over $96\%$` | hard.sintel.vits.int2_g16.tae_rel_pct_ci_lo=-98.7184; hard.sintel.vits.int2_g16.tae_rel_pct_fw_ci_lo=-98.7184 |
| 118 | Introduction | `+0.036` | `$+0.036$ (NYU) and $+0.025$ (KIT` | bonn.vits.delta_int3_bin_K8.tae_rel_pct_fw=-0.0346127; bonn.vits.delta_lam16_b3_K8.tae_rel_pct_ci_lo=-0.0342029 |
| 199 | Attention K/V in Video-Depth | `4.51` | `$1.07 \times 10^8$ scalars at ViT-S and $4.51 \times 10^8$ at ViT-L` | bonn.vits.int2_g16.covtae_rel_pct_ci_hi=4.51537; bonn.vits.int2_g16.covtae_rel_pct_fw_ci_hi=4.51537 |
| 252 | Construction-A codebooks and | `5.0` | `scale per $4$-vector gives $5.0$.` | sintel.vits.delta_int3_bin_K8.covtae_rel_pct_fw_ci_hi=4.9216; sintel.vits.delta_int4_bin_K8.covtae_rel_pct_fw_ci_hi=4.89922 |
| 261 | Construction-A codebooks and | `2.81` | `is $37/16 + 0.5 = 2.81$; Eq.~ is therefore a` | bonn.vits.int3_g16_rht.tae=2.82332; bonn.vits.int3_g16.tae=2.82367 |
| 383 | Rate--distortion on NYU and  | `5.0` | `(ViT-S KITTI). On ViT-S it matches $E_8$ at $5.0$ bits` | sintel.vits.delta_int3_bin_K8.covtae_rel_pct_fw_ci_hi=4.9216; sintel.vits.delta_int4_bin_K8.covtae_rel_pct_fw_ci_hi=4.89922 |
| 387 | Rate--distortion on NYU and  | `0.900` | `bits.} With ViT-S, $\Lambda^A_{16}$ reaches $0.900 \pm` |  |
| 388 | Rate--distortion on NYU and  | `0.897` | `0.005$ on NYU ($-0.010$) and $0.897 \pm 0.003$ on KITTI ($-0` |  |
| 388 | Rate--distortion on NYU and  | `-0.031` | `($-0.010$) and $0.897 \pm 0.003$ on KITTI ($-0.031$).` | mem.vitl.delta_int3_bin_K8.depth_rel_l1_max=0.0307009; hard.sintel.vits.int4_tensor.absrel_diff=0.0313 |
| 390 | Rate--distortion on NYU and  | `-0.028` | `(NYU) and $-0.028$ (KITTI); against unifor` | sintel.vits.lam16_b3.tae_rel_pct=0.0283962; bonn.vits.lam16_b4.tae_rel_pct_ci_hi=0.0285312 |
| 390 | Rate--distortion on NYU and  | `5.0` | `and $-0.028$ (KITTI); against uniform VQ at $5.0$ bits, $30\%$` | sintel.vits.delta_int3_bin_K8.covtae_rel_pct_fw_ci_hi=4.9216; sintel.vits.delta_int4_bin_K8.covtae_rel_pct_fw_ci_hi=4.89922 |
| 391 | Rate--distortion on NYU and  | `-0.011` | `fewer bits for $-0.011$ and $-0.026$. The two r` | mem.vits.lam16_b3.depth_rel_l1_max=0.0111177; sintel.vits.zero.delta1_diff_ci_hi=0.0108089 |
| 391 | Rate--distortion on NYU and  | `-0.026` | `fewer bits for $-0.011$ and $-0.026$. The two rates give a c` | bonn.vits.lam16_b4.tae_rel_pct_fw=0.0258995; bonn.vits.int2_g16_rht.tae_rel_pct_fw=-0.0267942 |
| 393 | Rate--distortion on NYU and  | `98.9%` | `$98.9\%$ (NYU) and over $96\%$` | hard.sintel.vits.int2_g16.tae_rel_pct_ci_lo=-98.7184; hard.sintel.vits.int2_g16.tae_rel_pct_fw_ci_lo=-98.7184 |
| 412 | Rate--distortion on NYU and  | `0.910` | `FP32 & ViT-S & 32.0 & 0.910 & 0.096 & 0.438 & 0.928` |  |
| 412 | Rate--distortion on NYU and  | `0.096` | `& ViT-S & 32.0 & 0.910 & 0.096 & 0.438 & 0.928 & 0.091` | sintel.vits.delta_int3_int2_K8.tae_rel_pct=-0.0967088; bonn.vits.lam16_b3.tae_rel_pct_fw=0.0952055 |
| 412 | Rate--distortion on NYU and  | `0.928` | `& ViT-S & 32.0 & 0.910 & 0.096 & 0.438 & 0.928 & 0.091 & 3.861 \\` | sintel.vits.int2_g16_rht.tae_rel_pct_ci_lo=-0.926234 |
| 412 | Rate--distortion on NYU and  | `3.861` | `.0 & 0.910 & 0.096 & 0.438 & 0.928 & 0.091 & 3.861 \\` |  |
| 413 | Rate--distortion on NYU and  | `5.0` | `Uniform VQ & ViT-S & 5.0 & \textbf{0.911} & 0.095` | sintel.vits.delta_int3_bin_K8.covtae_rel_pct_fw_ci_hi=4.9216; sintel.vits.delta_int4_bin_K8.covtae_rel_pct_fw_ci_hi=4.89922 |
| 413 | Rate--distortion on NYU and  | `0.911` | `orm VQ & ViT-S & 5.0 & \textbf{0.911} & 0.095 & 0.437 & \text` |  |
| 413 | Rate--distortion on NYU and  | `0.923` | `0 & \textbf{0.911} & 0.095 & 0.437 & \textbf{0.923} & 0.092 & 3.912 \\` |  |
| 413 | Rate--distortion on NYU and  | `3.912` | `} & 0.095 & 0.437 & \textbf{0.923} & 0.092 & 3.912 \\` |  |
| 414 | Rate--distortion on NYU and  | `0.911` | `orm VQ & ViT-S & 6.0 & \textbf{0.911} & 0.096 & 0.438 & \text` |  |
| 414 | Rate--distortion on NYU and  | `0.096` | `& ViT-S & 6.0 & \textbf{0.911} & 0.096 & 0.438 & \textbf{0.928}` | sintel.vits.delta_int3_int2_K8.tae_rel_pct=-0.0967088; bonn.vits.lam16_b3.tae_rel_pct_fw=0.0952055 |
| 414 | Rate--distortion on NYU and  | `0.928` | `0 & \textbf{0.911} & 0.096 & 0.438 & \textbf{0.928} & 0.091 & 3.879 \\` | sintel.vits.int2_g16_rht.tae_rel_pct_ci_lo=-0.926234 |
| 414 | Rate--distortion on NYU and  | `3.879` | `} & 0.096 & 0.438 & \textbf{0.928} & 0.091 & 3.879 \\` |  |
| 415 | Rate--distortion on NYU and  | `0.907` | `$ ($g{=}8$) & ViT-S & 4.0 & \textbf{0.907} & 0.097 & 0.443 & \text` |  |
| 415 | Rate--distortion on NYU and  | `0.443` | `& ViT-S & 4.0 & \textbf{0.907} & 0.097 & 0.443 & \textbf{0.925} & 0.090` | sintel.vits.lam16_b3.covtae_rel_pct=0.440804 |
| 415 | Rate--distortion on NYU and  | `0.925` | `0 & \textbf{0.907} & 0.097 & 0.443 & \textbf{0.925} & 0.090 & 3.868 \\` | sintel.vits.int2_g16_rht.tae_rel_pct_ci_lo=-0.926234 |
| 415 | Rate--distortion on NYU and  | `3.868` | `} & 0.097 & 0.443 & \textbf{0.925} & 0.090 & 3.868 \\` |  |
| 416 | Rate--distortion on NYU and  | `5.0` | `$E_8$ ($g{=}8$) & ViT-S & 5.0 & \textbf{0.910} & 0.096` | sintel.vits.delta_int3_bin_K8.covtae_rel_pct_fw_ci_hi=4.9216; sintel.vits.delta_int4_bin_K8.covtae_rel_pct_fw_ci_hi=4.89922 |
| 416 | Rate--distortion on NYU and  | `0.910` | `$ ($g{=}8$) & ViT-S & 5.0 & \textbf{0.910} & 0.096 & 0.438 & \text` |  |
| 416 | Rate--distortion on NYU and  | `0.096` | `$) & ViT-S & 5.0 & \textbf{0.910} & 0.096 & 0.438 & \textbf{0.929}` | sintel.vits.delta_int3_int2_K8.tae_rel_pct=-0.0967088; bonn.vits.lam16_b3.tae_rel_pct_fw=0.0952055 |
| 416 | Rate--distortion on NYU and  | `0.929` | `0 & \textbf{0.910} & 0.096 & 0.438 & \textbf{0.929} & 0.090 & 3.854 \\` | sintel.vits.int2_g16_rht.tae_rel_pct_ci_lo=-0.926234 |
| 416 | Rate--distortion on NYU and  | `3.854` | `} & 0.096 & 0.438 & \textbf{0.929} & 0.090 & 3.854 \\` |  |
| 417 | Rate--distortion on NYU and  | `0.909` | `$ ($g{=}8$) & ViT-S & 6.0 & \textbf{0.909} & 0.097 & 0.443 & \text` |  |
| 417 | Rate--distortion on NYU and  | `0.443` | `& ViT-S & 6.0 & \textbf{0.909} & 0.097 & 0.443 & \textbf{0.928} & 0.091` | sintel.vits.lam16_b3.covtae_rel_pct=0.440804 |
| 417 | Rate--distortion on NYU and  | `0.928` | `0 & \textbf{0.909} & 0.097 & 0.443 & \textbf{0.928} & 0.091 & 3.865 \\` | sintel.vits.int2_g16_rht.tae_rel_pct_ci_lo=-0.926234 |
| 417 | Rate--distortion on NYU and  | `3.865` | `} & 0.097 & 0.443 & \textbf{0.928} & 0.091 & 3.865 \\` |  |
| 418 | Rate--distortion on NYU and  | `0.900` | `$\Lambda^A_{16}$ ($g{=}16$) & ViT-S & 3.5 & 0.900 $\pm$ 0.005 & 0.098 & 0.` |  |
| 418 | Rate--distortion on NYU and  | `0.098` | `{=}16$) & ViT-S & 3.5 & 0.900 $\pm$ 0.005 & 0.098 & 0.443 & 0.897 $\pm$ 0.` | sintel.vits.int2_g16.tae_rel_pct_ci_lo=-0.0971805; sintel.vits.delta_int3_int2_K8.tae_rel_pct=-0.0967088 |
| 418 | Rate--distortion on NYU and  | `0.443` | `& ViT-S & 3.5 & 0.900 $\pm$ 0.005 & 0.098 & 0.443 & 0.897 $\pm$ 0.003 & 0.` | sintel.vits.lam16_b3.covtae_rel_pct=0.440804 |
| 418 | Rate--distortion on NYU and  | `0.897` | `& 3.5 & 0.900 $\pm$ 0.005 & 0.098 & 0.443 & 0.897 $\pm$ 0.003 & 0.103 & 4.` |  |
| 418 | Rate--distortion on NYU and  | `0.103` | `0.005 & 0.098 & 0.443 & 0.897 $\pm$ 0.003 & 0.103 & 4.048 \\` | bonn.vits.int4_g16.tae_rel_pct_ci_lo=-0.102165; bonn.vits.int4_g16.tae_rel_pct_fw_ci_lo=-0.102165 |
| 418 | Rate--distortion on NYU and  | `4.048` | `0.098 & 0.443 & 0.897 $\pm$ 0.003 & 0.103 & 4.048 \\` | hard.sintel.vits.int3_g16.tae_rel_pct_ci_hi=4.04911; hard.sintel.vits.int3_g16.tae_rel_pct_fw_ci_hi=4.04911 |
| 419 | Rate--distortion on NYU and  | `0.915` | `^A_{16}$ ($g{=}16$) & ViT-S & 4.5 & \textbf{0.915} & 0.094 & 0.436 & \text` |  |
| 419 | Rate--distortion on NYU and  | `0.094` | `($g{=}16$) & ViT-S & 4.5 & \textbf{0.915} & 0.094 & 0.436 & \textbf{0.924}` | bonn.vits.lam16_b3.tae_rel_pct_fw=0.0952055; bonn.vits.int4_g16_rht.covtae_rel_pct_ci_hi=-0.0927786 |
| 419 | Rate--distortion on NYU and  | `0.924` | `5 & \textbf{0.915} & 0.094 & 0.436 & \textbf{0.924} & 0.091 & 3.879 \\` | sintel.vits.int2_g16_rht.tae_rel_pct_ci_lo=-0.926234 |
| 419 | Rate--distortion on NYU and  | `3.879` | `} & 0.094 & 0.436 & \textbf{0.924} & 0.091 & 3.879 \\` |  |
| 421 | Rate--distortion on NYU and  | `0.877` | `FP32 & ViT-B & 32.0 & 0.877 & 0.116 & 0.501 & 0.932` |  |
| 421 | Rate--distortion on NYU and  | `0.116` | `& ViT-B & 32.0 & 0.877 & 0.116 & 0.501 & 0.932 & 0.089` | bonn.vits.int4_g16_rht.covtae_rel_pct_ci_lo=-0.113545; bonn.vits.int4_g16_rht.covtae_rel_pct_fw_ci_lo=-0.113545 |
| 421 | Rate--distortion on NYU and  | `0.932` | `& ViT-B & 32.0 & 0.877 & 0.116 & 0.501 & 0.932 & 0.089 & 4.329 \\` | mem.vits.delta_fp32_K8.cache_ratio_vs_fp32=0.933333 |
| 421 | Rate--distortion on NYU and  | `0.089` | `T-B & 32.0 & 0.877 & 0.116 & 0.501 & 0.932 & 0.089 & 4.329 \\` | mem.vits.delta_lam16_b2_K8.depth_rel_l1_mean=0.0896238; bonn.vits.int4_g16.covtae_rel_pct_ci_hi=-0.0896548 |
| 421 | Rate--distortion on NYU and  | `4.329` | `.0 & 0.877 & 0.116 & 0.501 & 0.932 & 0.089 & 4.329 \\` |  |
| 422 | Rate--distortion on NYU and  | `0.902` | `^A_{16}$ ($g{=}16$) & ViT-B & 3.5 & \textbf{0.902} & 0.101 & 0.442 & 0.920` |  |
| 422 | Rate--distortion on NYU and  | `0.101` | `($g{=}16$) & ViT-B & 3.5 & \textbf{0.902} & 0.101 & 0.442 & 0.920 & 0.090` | mem.vits.lam16_b2.depth_rel_l1_max=0.101659; bonn.vits.int4_g16_rht.covtae_rel_pct_fw=-0.102042 |
| 422 | Rate--distortion on NYU and  | `0.442` | `$) & ViT-B & 3.5 & \textbf{0.902} & 0.101 & 0.442 & 0.920 & 0.090 & 4.385` | sintel.vits.lam16_b3.covtae_rel_pct=0.440804; hard.sintel.vits.e8_g64_b2.absrel=0.4396 |
| 422 | Rate--distortion on NYU and  | `0.920` | `-B & 3.5 & \textbf{0.902} & 0.101 & 0.442 & 0.920 & 0.090 & 4.385 \\` |  |
| 422 | Rate--distortion on NYU and  | `4.385` | `tbf{0.902} & 0.101 & 0.442 & 0.920 & 0.090 & 4.385 \\` |  |
| 423 | Rate--distortion on NYU and  | `0.890` | `^A_{16}$ ($g{=}16$) & ViT-B & 4.5 & \textbf{0.890} & 0.109 & 0.482 & \text` |  |
| 423 | Rate--distortion on NYU and  | `0.482` | `$) & ViT-B & 4.5 & \textbf{0.890} & 0.109 & 0.482 & \textbf{0.936} & 0.084` | bonn.vits.int3_g16_rht.covtae_rel_pct_fw=-0.483798 |
| 423 | Rate--distortion on NYU and  | `0.936` | `5 & \textbf{0.890} & 0.109 & 0.482 & \textbf{0.936} & 0.084 & 4.056 \\` | mem.vits.delta_fp32_K8.cache_ratio_vs_fp32=0.933333 |
| 423 | Rate--distortion on NYU and  | `4.056` | `} & 0.109 & 0.482 & \textbf{0.936} & 0.084 & 4.056 \\` |  |
| 425 | Rate--distortion on NYU and  | `0.896` | `FP32 & ViT-L & 32.0 & 0.896 & 0.106 & 0.471 & 0.942` |  |
| 425 | Rate--distortion on NYU and  | `0.471` | `& ViT-L & 32.0 & 0.896 & 0.106 & 0.471 & 0.942 & 0.081 & 3.741` | hard.sintel.vits.e8_g16_b2.absrel=0.4724 |
| 425 | Rate--distortion on NYU and  | `0.942` | `& ViT-L & 32.0 & 0.896 & 0.106 & 0.471 & 0.942 & 0.081 & 3.741 \\` |  |
| 425 | Rate--distortion on NYU and  | `3.741` | `.0 & 0.896 & 0.106 & 0.471 & 0.942 & 0.081 & 3.741 \\` |  |
| 426 | Rate--distortion on NYU and  | `0.920` | `^A_{16}$ ($g{=}16$) & ViT-L & 3.5 & \textbf{0.920} & 0.091 & 0.414 & 0.928` |  |
| 426 | Rate--distortion on NYU and  | `0.928` | `-L & 3.5 & \textbf{0.920} & 0.091 & 0.414 & 0.928 & 0.089 & 3.891 \\` | sintel.vits.int2_g16_rht.tae_rel_pct_ci_lo=-0.926234 |
| 426 | Rate--distortion on NYU and  | `0.089` | `5 & \textbf{0.920} & 0.091 & 0.414 & 0.928 & 0.089 & 3.891 \\` | mem.vits.delta_lam16_b2_K8.depth_rel_l1_mean=0.0896238; bonn.vits.int4_g16.covtae_rel_pct_ci_hi=-0.0896548 |
| 426 | Rate--distortion on NYU and  | `3.891` | `tbf{0.920} & 0.091 & 0.414 & 0.928 & 0.089 & 3.891 \\` | sintel.vits.int2_g16_rht.covtae_rel_pct_fw=3.89367 |
| 427 | Rate--distortion on NYU and  | `0.902` | `^A_{16}$ ($g{=}16$) & ViT-L & 4.5 & \textbf{0.902} & 0.103 & 0.462 & \text` |  |
| 427 | Rate--distortion on NYU and  | `0.103` | `($g{=}16$) & ViT-L & 4.5 & \textbf{0.902} & 0.103 & 0.462 & \textbf{0.940}` | bonn.vits.int4_g16.tae_rel_pct_ci_lo=-0.102165; bonn.vits.int4_g16.tae_rel_pct_fw_ci_lo=-0.102165 |
| 427 | Rate--distortion on NYU and  | `0.462` | `$) & ViT-L & 4.5 & \textbf{0.902} & 0.103 & 0.462 & \textbf{0.940} & 0.083` | hard.sintel.vits.int2_g16.absrel=0.4633; bonn.vits.int3_g16.covtae_rel_pct=-0.463453 |
| 427 | Rate--distortion on NYU and  | `0.940` | `5 & \textbf{0.902} & 0.103 & 0.462 & \textbf{0.940} & 0.083 & 3.761 \\` |  |
| 427 | Rate--distortion on NYU and  | `0.083` | `bf{0.902} & 0.103 & 0.462 & \textbf{0.940} & 0.083 & 3.761 \\` | bonn.vits.int4_g64_asym.covtae_rel_pct_fw_ci_hi=-0.0822038; bonn.vits.int4_g64_asym.covtae_rel_pct_ci_hi=-0.0822038 |
| 427 | Rate--distortion on NYU and  | `3.761` | `} & 0.103 & 0.462 & \textbf{0.940} & 0.083 & 3.761 \\` |  |
| 437 | Across backbones | `0.928` | `FP32 backbones follow the expected order ($0.928 < 0.932 < 0.942$),` | sintel.vits.int2_g16_rht.tae_rel_pct_ci_lo=-0.926234 |
| 437 | Across backbones | `0.932` | `ackbones follow the expected order ($0.928 < 0.932 < 0.942$),` | mem.vits.delta_fp32_K8.cache_ratio_vs_fp32=0.933333 |
| 437 | Across backbones | `0.942` | `follow the expected order ($0.928 < 0.932 < 0.942$),` |  |
| 441 | Across backbones | `0.910` | `FP32 scores are $0.910$ (ViT-S), $0.877$ (ViT-B` |  |
| 441 | Across backbones | `0.877` | `FP32 scores are $0.910$ (ViT-S), $0.877$ (ViT-B) and $0.896$ (Vi` |  |
| 441 | Across backbones | `0.896` | `es are $0.910$ (ViT-S), $0.877$ (ViT-B) and $0.896$ (ViT-L),` |  |
| 443 | Across backbones | `+0.024` | `rates ($+0.025$ and $+0.024$ at $3.5$ bits; $+0.013$` | bonn.vits.delta_lam16_b3_K8.tae_rel_pct_fw=-0.0246815; bonn.vits.lam16_b4.tae_rel_pct=0.0251699 |
| 452 | Rotation ablation | `0.036` | `$\delta_1$ by $0.036$ on NYU ($0.907 \to 0.87` | bonn.vits.delta_int3_bin_K8.tae_rel_pct_fw=-0.0346127; bonn.vits.delta_lam16_b3_K8.tae_rel_pct_ci_lo=-0.0342029 |
| 452 | Rotation ablation | `0.907` | `$\delta_1$ by $0.036$ on NYU ($0.907 \to 0.871$) and $0.025$` |  |
| 452 | Rotation ablation | `0.871` | `$\delta_1$ by $0.036$ on NYU ($0.907 \to 0.871$) and $0.025$ on KITTI` |  |
| 453 | Rotation ablation | `0.900` | `($0.900 \to 0.875$). At $4$ bits` |  |
| 453 | Rotation ablation | `0.875` | `($0.900 \to 0.875$). At $4$ bits the chang` |  |
| 469 | Rotation ablation | `0.910` | `FP32 & --- & 0.910 & 0.928 \\` |  |
| 469 | Rotation ablation | `0.928` | `FP32 & --- & 0.910 & 0.928 \\` | sintel.vits.int2_g16_rht.tae_rel_pct_ci_lo=-0.926234 |
| 470 | Rotation ablation | `0.907` | `$\Lambda^A_{16}$ with RHT & 3 & 0.907 & 0.900 \\` |  |
| 470 | Rotation ablation | `0.900` | `$\Lambda^A_{16}$ with RHT & 3 & 0.907 & 0.900 \\` |  |
| 471 | Rotation ablation | `0.871` | `$\Lambda^A_{16}$ no RHT & 3 & 0.871 & 0.875 \\` |  |
| 471 | Rotation ablation | `0.875` | `$\Lambda^A_{16}$ no RHT & 3 & 0.871 & 0.875 \\` |  |
| 472 | Rotation ablation | `0.915` | `$\Lambda^A_{16}$ with RHT & 4 & 0.915 & 0.924 \\` |  |
| 472 | Rotation ablation | `0.924` | `$\Lambda^A_{16}$ with RHT & 4 & 0.915 & 0.924 \\` | sintel.vits.int2_g16_rht.tae_rel_pct_ci_lo=-0.926234 |
| 473 | Rotation ablation | `0.911` | `$\Lambda^A_{16}$ no RHT & 4 & 0.911 & 0.927 \\` |  |
| 473 | Rotation ablation | `0.927` | `$\Lambda^A_{16}$ no RHT & 4 & 0.911 & 0.927 \\` | sintel.vits.int2_g16_rht.tae_rel_pct_ci_lo=-0.926234 |
| 485 | Group-size sweep | `0.051` | `64$, $\delta_1$ drops by $0.070$ on NYU and $0.051$` | hard.sintel.vits.lam16_g64_b3.absrel_diff=0.0499; bonn.vits.delta_int4_bin_K8.tae_rel_pct_ci_lo=-0.0498864 |
| 486 | Group-size sweep | `0.375` | `on KITTI while saving only $0.375$ bits. $g{=}16$, one sca` | bonn.vits.delta_int3_bin_K8.covtae_rel_pct_ci_lo=0.374168; bonn.vits.delta_int3_bin_K8.covtae_rel_pct_fw_ci_lo=0.374168 |
| 499 | Group-size sweep | `0.907` | `16 & 3.500 & \textbf{0.907} & \textbf{0.900} \\` |  |
| 499 | Group-size sweep | `0.900` | `16 & 3.500 & \textbf{0.907} & \textbf{0.900} \\` |  |
| 500 | Group-size sweep | `0.893` | `32 & 3.250 & 0.893 & 0.888 \\` |  |
| 500 | Group-size sweep | `0.888` | `32 & 3.250 & 0.893 & 0.888 \\` |  |
| 501 | Group-size sweep | `3.167` | `48 & 3.167 & 0.874 & 0.870 \\` |  |
| 501 | Group-size sweep | `0.874` | `48 & 3.167 & 0.874 & 0.870 \\` |  |
| 501 | Group-size sweep | `0.870` | `48 & 3.167 & 0.874 & 0.870 \\` |  |
| 502 | Group-size sweep | `0.837` | `64 & 3.125 & 0.837 & 0.849 \\` |  |
| 502 | Group-size sweep | `0.849` | `64 & 3.125 & 0.837 & 0.849 \\` |  |
| 514 | Temporal window on Sintel | `0.038` | `$0.034$, $0.038$) and cov-TAE rises by $` |  |
| 514 | Temporal window on Sintel | `3.8` | `.034$, $0.038$) and cov-TAE rises by $3.2$--$3.8$ points at every` | sintel.vits.delta_int3_bin_K16.tae_rel_pct_fw_ci_hi=3.73627; sintel.vitl.zero.covtae_rel_pct_ci_lo=3.722 |
| 531 | Temporal window on Sintel | `0.715` | `8 & FP32 & 0.715 & 0.231 & 54.33 & 6.90 \` | sintel.vits.lam16_b2.delta1=0.715654; sintel.vits.zero.delta1=0.716271 |
| 531 | Temporal window on Sintel | `54.33` | `8 & FP32 & 0.715 & 0.231 & 54.33 & 6.90 \\` |  |
| 531 | Temporal window on Sintel | `6.90` | `FP32 & 0.715 & 0.231 & 54.33 & 6.90 \\` | sintel.vits.delta_int3_bin_K16.covtae=6.89259; sintel.vits.delta_int4_bin_K8.covtae=6.91465 |
| 532 | Temporal window on Sintel | `0.680` | `& $\Lambda^A_{16}$ & 0.680 & 0.259 & 50.10 & 10.13` | sintel.vits.int3_g16.covtae_rel_pct_fw_ci_lo=-0.681109; sintel.vits.int3_g16.covtae_rel_pct_ci_lo=-0.678232 |
| 532 | Temporal window on Sintel | `0.259` | `& $\Lambda^A_{16}$ & 0.680 & 0.259 & 50.10 & 10.13 \\` | bonn.vits.delta_int2_K8.covtae_rel_pct_ci_lo=0.257764; bonn.vits.delta_int2_K8.covtae_rel_pct_fw_ci_lo=0.257764 |
| 532 | Temporal window on Sintel | `50.10` | `& $\Lambda^A_{16}$ & 0.680 & 0.259 & 50.10 & 10.13 \\` |  |
| 532 | Temporal window on Sintel | `10.13` | `$\Lambda^A_{16}$ & 0.680 & 0.259 & 50.10 & 10.13 \\` |  |
| 535 | Temporal window on Sintel | `55.56` | `& $\Lambda^A_{16}$ & 0.678 & 0.258 & 55.56 & 10.19 \\` |  |
| 535 | Temporal window on Sintel | `10.19` | `$\Lambda^A_{16}$ & 0.678 & 0.258 & 55.56 & 10.19 \\` |  |
| 537 | Temporal window on Sintel | `54.30` | `32 & FP32 & 0.711 & 0.232 & 54.30 & 5.86 \\` | sintel.vits.bin_g16.tae=54.2807 |
| 537 | Temporal window on Sintel | `5.86` | `FP32 & 0.711 & 0.232 & 54.30 & 5.86 \\` | sintel.vits.delta_int3_bin_K16.covtae_rel_pct_ci_hi=5.86938 |
| 538 | Temporal window on Sintel | `0.673` | `& $\Lambda^A_{16}$ & 0.673 & 0.257 & 51.23 & 9.61 \` |  |
| 538 | Temporal window on Sintel | `0.257` | `& $\Lambda^A_{16}$ & 0.673 & 0.257 & 51.23 & 9.61 \\` | bonn.vits.delta_int3_int2_K8.tae_rel_pct_fw=-0.257721; bonn.vits.delta_int2_K8.covtae_rel_pct_ci_lo=0.257764 |
| 538 | Temporal window on Sintel | `51.23` | `& $\Lambda^A_{16}$ & 0.673 & 0.257 & 51.23 & 9.61 \\` |  |
| 538 | Temporal window on Sintel | `9.61` | `$\Lambda^A_{16}$ & 0.673 & 0.257 & 51.23 & 9.61 \\` |  |
| 551 | Bonn RGB-D | `0.983` | `of FP32 $\delta_1$ ($0.983$ vs.\ $0.987$), with Abs` | bonn.vits.1sc.zero.delta1=0.985132; bonn.vits.1sc.fp32.delta1=0.985263 |
| 553 | Bonn RGB-D | `2.80` | `\emph{improves} at $3.5$ bits ($2.80 \to 2.61$) while cov-TAE` | bonn.vits.int3_g16_rht.tae=2.82332; bonn.vits.int3_g16.tae=2.82367 |
| 554 | Bonn RGB-D | `1.67` | `rises ($1.67 \to 1.77$); across $\Lam` |  |
| 554 | Bonn RGB-D | `1.77` | `rises ($1.67 \to 1.77$); across $\Lambda^A_{16` | bonn.vits.int2_g16.covtae=1.75977 |
| 555 | Bonn RGB-D | `1.67` | `increases monotonically as bits fall ($1.67$, $1.68$, $1.77$),` |  |
| 555 | Bonn RGB-D | `1.68` | `creases monotonically as bits fall ($1.67$, $1.68$, $1.77$),` | bonn.vits.int3_g16.covtae=1.70278; bonn.vits.int3_g16_rht.covtae=1.70285 |
| 555 | Bonn RGB-D | `1.77` | `monotonically as bits fall ($1.67$, $1.68$, $1.77$),` | bonn.vits.int2_g16.covtae=1.75977 |
| 570 | Bonn RGB-D | `2.80` | `& 32.0 & 0.987 & 0.043 & 2.80 & 1.67 \\` | bonn.vits.int3_g16_rht.tae=2.82332; bonn.vits.int3_g16.tae=2.82367 |
| 570 | Bonn RGB-D | `1.67` | `& 32.0 & 0.987 & 0.043 & 2.80 & 1.67 \\` |  |
| 571 | Bonn RGB-D | `5.0` | `Uniform VQ & 5.0 & \textbf{0.987} & 0.043` | sintel.vits.delta_int3_bin_K8.covtae_rel_pct_fw_ci_hi=4.9216; sintel.vits.delta_int4_bin_K8.covtae_rel_pct_fw_ci_hi=4.89922 |
| 571 | Bonn RGB-D | `2.77` | `& 5.0 & \textbf{0.987} & 0.043 & 2.77 & 1.69 \\` | hard.sintel.vits.e8_g32_b3.bits_inventory=2.75 |
| 571 | Bonn RGB-D | `1.69` | `& 5.0 & \textbf{0.987} & 0.043 & 2.77 & 1.69 \\` | bonn.vits.int3_g16.covtae=1.70278; bonn.vits.int3_g16_rht.covtae=1.70285 |
| 573 | Bonn RGB-D | `0.983` | `$\Lambda^A_{16}$ ($g{=}16$) & 3.5 & \textbf{0.983} & 0.053 & 2.61 & 1.77 \` | bonn.vits.1sc.zero.delta1=0.985132; bonn.vits.1sc.fp32.delta1=0.985263 |
| 573 | Bonn RGB-D | `1.77` | `6$) & 3.5 & \textbf{0.983} & 0.053 & 2.61 & 1.77 \\` | bonn.vits.int2_g16.covtae=1.75977 |
| 574 | Bonn RGB-D | `2.77` | `($g{=}16$) & 4.5 & \textbf{0.987} & 0.043 & 2.77 & 1.68 \\` | hard.sintel.vits.e8_g32_b3.bits_inventory=2.75 |
| 574 | Bonn RGB-D | `1.68` | `6$) & 4.5 & \textbf{0.987} & 0.043 & 2.77 & 1.68 \\` | bonn.vits.int3_g16.covtae=1.70278; bonn.vits.int3_g16_rht.covtae=1.70285 |
| 586 | Seed stability | `0.673` | `Sintel the five-seed mean is $0.673 \pm 0.011$ against FP32` |  |
| 600 | Seed stability | `0.9074` | `0 & 0.9074 & 0.9003 & 0.6727 \\` |  |
| 600 | Seed stability | `0.9003` | `0 & 0.9074 & 0.9003 & 0.6727 \\` |  |
| 600 | Seed stability | `0.6727` | `0 & 0.9074 & 0.9003 & 0.6727 \\` |  |
| 601 | Seed stability | `0.8943` | `1 & 0.8943 & 0.8997 & 0.6542 \\` |  |
| 601 | Seed stability | `0.8997` | `1 & 0.8943 & 0.8997 & 0.6542 \\` |  |
| 601 | Seed stability | `0.6542` | `1 & 0.8943 & 0.8997 & 0.6542 \\` |  |
| 602 | Seed stability | `0.9020` | `2 & 0.9020 & 0.8942 & 0.6809 \\` |  |
| 602 | Seed stability | `0.8942` | `2 & 0.9020 & 0.8942 & 0.6809 \\` |  |
| 602 | Seed stability | `0.6809` | `2 & 0.9020 & 0.8942 & 0.6809 \\` | sintel.vits.int3_g16.covtae_rel_pct_fw_ci_lo=-0.681109 |
| 603 | Seed stability | `0.8950` | `3 & 0.8950 & 0.8930 & 0.6786 \\` |  |
| 603 | Seed stability | `0.8930` | `3 & 0.8950 & 0.8930 & 0.6786 \\` |  |
| 603 | Seed stability | `0.6786` | `3 & 0.8950 & 0.8930 & 0.6786 \\` |  |
| 604 | Seed stability | `0.9008` | `4 & 0.9008 & 0.8956 & 0.6804 \\` |  |
| 604 | Seed stability | `0.8956` | `4 & 0.9008 & 0.8956 & 0.6804 \\` |  |
| 604 | Seed stability | `0.6804` | `4 & 0.9008 & 0.8956 & 0.6804 \\` |  |
| 606 | Seed stability | `0.900` | `mean $\pm$ std & $0.900 \pm 0.005$ & $0.897 \pm` |  |
| 606 | Seed stability | `0.897` | `mean $\pm$ std & $0.900 \pm 0.005$ & $0.897 \pm 0.003$ & $0.673 \pm` |  |
| 606 | Seed stability | `0.673` | `d & $0.900 \pm 0.005$ & $0.897 \pm 0.003$ & $0.673 \pm 0.011$ \\` |  |
| 640 | Per-frame accuracy on Sintel | `-0.013` | `\texttt{temple\_3} & 50 & $-0.013$ & $0.015$ & $0.120$ \\` | probe.bonn.residual_var_ratio_min=0.013; sintel.vits.zero.delta1_diff_ci_lo=-0.014107 |
| 640 | Per-frame accuracy on Sintel | `0.120` | `xttt{temple\_3} & 50 & $-0.013$ & $0.015$ & $0.120$ \\` | bonn.vits.delta_int2_K8.tae_rel_pct_ci_lo=-0.121843; bonn.vits.delta_int2_K8.tae_rel_pct_fw_ci_lo=-0.121843 |
| 650 | Per-frame accuracy on Sintel | `0.120` | `deviation (maximum $0.120$), with a mean gap of $-` | bonn.vits.delta_int2_K8.tae_rel_pct_ci_lo=-0.121843; bonn.vits.delta_int2_K8.tae_rel_pct_fw_ci_lo=-0.121843 |
| 650 | Per-frame accuracy on Sintel | `-0.013` | `tion (maximum $0.120$), with a mean gap of $-0.013$.` | probe.bonn.residual_var_ratio_min=0.013; sintel.vits.zero.delta1_diff_ci_lo=-0.014107 |
| 691 | Packed cache and fused kerne | `4.57` | `($4.57\times$ as allocated), an` | sintel.vitb.int2_g16.covtae_rel_pct_ci_lo=4.5909 |
| 712 | Packed cache and fused kerne | `19.65` | `er the eight shapes, the fused kernel takes $19.65$~ms` |  |
| 713 | Packed cache and fused kerne | `23.77` | `against $23.77$~ms for the simulator ($` |  |
| 714 | Packed cache and fused kerne | `4.35` | `ratios from $0.73\times$ to $4.35\times$. The largest gain` |  |
| 715 | Packed cache and fused kerne | `4.35` | `($4.35\times$ at ViT-S, $4.01\t` |  |
| 715 | Packed cache and fused kerne | `4.01` | `($4.35\times$ at ViT-S, $4.01\times$ at ViT-L). On the` | bonn.vits.lam16_b4.bits=4.00012; bonn.vits.delta_int3_rht_K8.bits=4 |
| 738 | Packed cache and fused kerne | `0.39` | `& 1369 & 192 & 1.00 & 0.19 & $5.33\times$ & 0.39 & 2.01 & 1.18 & $1.70\` | sintel.vits.delta_int2_K8.tae_rel_pct_fw_ci_hi=0.384107; sintel.vits.delta_int2_K8.tae_rel_pct_ci_hi=0.383828 |
| 739 | Packed cache and fused kerne | `4.35` | `.10 & $5.33\times$ & 0.23 & 1.18 & 0.27 & $4.35\times$ \\` |  |
| 740 | Packed cache and fused kerne | `3.81` | `.06 & $5.33\times$ & 0.16 & 1.09 & 0.29 & $3.81\times$ \\` |  |
| 741 | Packed cache and fused kerne | `1.27` | `& 5476 & 64 & 1.34 & 0.25 & $5.33\times$ & 1.27 & 2.54 & 3.17 & $0.80\` |  |
| 741 | Packed cache and fused kerne | `2.54` | `64 & 1.34 & 0.25 & $5.33\times$ & 1.27 & 2.54 & 3.17 & $0.80\times$ \` | sintel.vits.delta_int2_K8.bits=2.55952; hard.sintel.vits.int4_g64.tae_rel_pct_ci_lo=-2.51843 |
| 741 | Packed cache and fused kerne | `3.17` | `1.34 & 0.25 & $5.33\times$ & 1.27 & 2.54 & 3.17 & $0.80\times$ \\` |  |
| 742 | Packed cache and fused kerne | `5.35` | `ViT-L & $\mathrm{mm}_0$ & 1369 & 1024 & 5.35 & 1.00 & $5.33\times$ &` | sintel.vitb.delta_int3_bin_K8.covtae_rel_pct_fw_ci_hi=5.36506; mem.vits.lam16_kv_b3.cache_ratio_vs_fp32=5.33306 |
| 742 | Packed cache and fused kerne | `0.61` | `& 1369 & 1024 & 5.35 & 1.00 & $5.33\times$ & 0.61 & 5.91 & 3.16 & $1.87\` | sintel.vits.delta_int3_int2_K8.tae_rel_pct_ci_lo=-0.617389; bonn.vits.delta_int4_bin_K8.covtae_rel_pct=0.600278 |
| 742 | Packed cache and fused kerne | `5.91` | `1024 & 5.35 & 1.00 & $5.33\times$ & 0.61 & 5.91 & 3.16 & $1.87\times$ \` |  |
| 742 | Packed cache and fused kerne | `3.16` | `5.35 & 1.00 & $5.33\times$ & 0.61 & 5.91 & 3.16 & $1.87\times$ \\` | sintel.vitl.zero.tae_rel_pct_ci_lo=-3.13531 |
| 743 | Packed cache and fused kerne | `1.41` | `ViT-L & $\mathrm{mm}_1$ & 361 & 1024 & 1.41 & 0.26 & $5.33\times$ &` | mem.vitl.fp16.peak_ratio_vs_fp32=1.43271; sintel.vits.delta_int3_bin_K16.tae_rel_pct=-1.3802 |
| 743 | Packed cache and fused kerne | `4.01` | `.26 & $5.33\times$ & 0.31 & 1.76 & 0.44 & $4.01\times$ \\` | bonn.vits.lam16_b4.bits=4.00012; bonn.vits.delta_int3_rht_K8.bits=4 |
| 744 | Packed cache and fused kerne | `1.78` | `256 & 1.34 & 0.25 & $5.33\times$ & 0.40 & 1.78 & 0.85 & $2.08\times$ \` | bonn.vits.int2_g16.covtae=1.75977 |
| 744 | Packed cache and fused kerne | `0.85` | `1.34 & 0.25 & $5.33\times$ & 0.40 & 1.78 & 0.85 & $2.08\times$ \\` | sintel.vitb.delta_int3_bin_K8.covis_fraction=0.860362; sintel.vitb.fp32.covis_fraction=0.860362 |
| 744 | Packed cache and fused kerne | `2.08` | `.25 & $5.33\times$ & 0.40 & 1.78 & 0.85 & $2.08\times$ \\` | mem.vits.delta_lam16_b3_K8.peak_ratio_vs_fp32=2.06243; sintel.vits.delta_int3_bin_K16.tae_rel_pct_fw_ci_lo=-2.05811 |
| 745 | Packed cache and fused kerne | `5.35` | `ViT-L & $\mathrm{mm}_3$ & 5476 & 256 & 5.35 & 1.00 & $5.33\times$ &` | sintel.vitb.delta_int3_bin_K8.covtae_rel_pct_fw_ci_hi=5.36506; mem.vits.lam16_kv_b3.cache_ratio_vs_fp32=5.33306 |
| 745 | Packed cache and fused kerne | `2.48` | `& 5476 & 256 & 5.35 & 1.00 & $5.33\times$ & 2.48 & 7.50 & 10.29 & $0.73\` | bonn.vits.int2_g16.bits=2.5; bonn.vits.int2_g16_rht.bits=2.5 |
| 745 | Packed cache and fused kerne | `7.50` | `256 & 5.35 & 1.00 & $5.33\times$ & 2.48 & 7.50 & 10.29 & $0.73\times$ \` |  |
| 745 | Packed cache and fused kerne | `10.29` | `5.35 & 1.00 & $5.33\times$ & 2.48 & 7.50 & 10.29 & $0.73\times$ \\` |  |
| 747 | Packed cache and fused kerne | `16.65` | `\multicolumn{4}{l}{Total} & 16.65 & 3.12 & $5.33\times$ &` |  |
| 747 | Packed cache and fused kerne | `3.12` | `\multicolumn{4}{l}{Total} & 16.65 & 3.12 & $5.33\times$ & 5.84 &` | hard.sintel.vits.e8_g64_b3.bits=3.125; hard.sintel.vits.int3_g64.bits=3.125 |
| 747 | Packed cache and fused kerne | `5.84` | `4}{l}{Total} & 16.65 & 3.12 & $5.33\times$ & 5.84 & 23.77 & 19.65 & $1.21\` | sintel.vits.delta_int3_bin_K16.covtae_rel_pct_ci_hi=5.86938 |
| 747 | Packed cache and fused kerne | `23.77` | `otal} & 16.65 & 3.12 & $5.33\times$ & 5.84 & 23.77 & 19.65 & $1.21\times$ \` |  |
| 747 | Packed cache and fused kerne | `19.65` | `16.65 & 3.12 & $5.33\times$ & 5.84 & 23.77 & 19.65 & $1.21\times$ \\` |  |
| 771 | Packed cache and fused kerne | `13.6` | `$13.6$~FPS ($0.94\times$ FP32)` |  |
| 771 | Packed cache and fused kerne | `0.94` | `$13.6$~FPS ($0.94\times$ FP32), with run-t` | mem.vits.delta_fp32_K8.cache_ratio_vs_fp32=0.933333; sintel.vits.int2_g16_rht.tae_rel_pct_ci_lo=-0.926234 |
| 772 | Packed cache and fused kerne | `14.2` | `both FP32 ($14.5$) and the simulator ($14.2$), so quantization` | hard.sintel.vits.lam16_g64_b3.tae_rel_pct=-14.0701; hard.sintel.vits.lam16_g64_b3.tae_rel_pct_fw=-14.0701 |
| 774 | Packed cache and fused kerne | `19.4` | `$19.4$~FPS. Peak allocated mem` |  |
| 774 | Packed cache and fused kerne | `10.2` | `$19.4$~FPS. Peak allocated memory ($10.2$~GB for both quantized` | hard.sintel.vits.int2_g64.tae_rel_pct_ci_hi=9.97275; hard.sintel.vits.int2_g64.tae_rel_pct_fw_ci_hi=9.97275 |
| 793 | Packed cache and fused kerne | `19.4` | `Fp16 autocast & $19.4 \pm 0.8$ & $1.34\times$` |  |
| 794 | Packed cache and fused kerne | `14.2` | `$\Lambda^A_{16}$ simulator (+QJL) & $14.2 \pm 0.8$ & $0.98\times$` | hard.sintel.vits.lam16_g64_b3.tae_rel_pct=-14.0701; hard.sintel.vits.lam16_g64_b3.tae_rel_pct_fw=-14.0701 |
| 794 | Packed cache and fused kerne | `0.98` | `16}$ simulator (+QJL) & $14.2 \pm 0.8$ & $0.98\times$ & 10210 \\` | bonn.vits.1sc.zero.delta1=0.985132; bonn.vits.1sc.fp32.delta1=0.985263 |
| 795 | Packed cache and fused kerne | `13.6` | `$\Lambda^A_{16}$ fused + CUDA graph & $13.6 \pm 2.2$ & $0.94\times$` |  |
| 795 | Packed cache and fused kerne | `0.94` | `16}$ fused + CUDA graph & $13.6 \pm 2.2$ & $0.94\times$ & 10209 \\` | mem.vits.delta_fp32_K8.cache_ratio_vs_fp32=0.933333; sintel.vits.int2_g16_rht.tae_rel_pct_ci_lo=-0.926234 |
| 809 | Conclusion | `98.9%` | `$98.9\%$ (NYU), over $96\%$ (K` | hard.sintel.vits.int2_g16.tae_rel_pct_ci_lo=-98.7184; hard.sintel.vits.int2_g16.tae_rel_pct_fw_ci_lo=-98.7184 |

## Matched

| line | number | strength | facts (first 3) |
|---|---|---|---|
| 53 | `4.5` | ambiguous | bonn.vits.int2_g16.covtae_rel_pct_ci_hi, bonn.vits.int2_g16.covtae_rel_pct_fw_ci_hi, bonn.vits.int4_g16.bits (+2) |
| 54 | `0.004` | ambiguous | hard.sintel.vits.e8_g16_b3.delta1_agg_diff, hard.sintel.vits.e8_g64_b3.delta1_agg_diff, mem.vits.lam16_b4.depth_rel_l1_mean (+8) |
| 56 | `3.5` | ambiguous | bonn.vits.int3_g16.bits, bonn.vits.int3_g16_rht.bits, hard.sintel.vits.e8_g16_b3.bits (+7) |
| 56 | `0.005` | ambiguous | hard.sintel.vits.e8_g32_b3.absrel_diff, mem.vitb.delta_int3_bin_K8.depth_rel_l1_mean, mem.vits.lam16_b4.depth_rel_l1_max (+11) |
| 57 | `0.003` | ambiguous | hard.sintel.vits.e8_g64_b3.absrel_diff, mem.vits.delta_lam16_b3_K8.depth_rel_l1_mean, mem.vits.lam16_kv_b4.depth_rel_l1_max (+8) |
| 59 | `12.5` | ambiguous | bonn.vits.delta_lam16_b2_K8.covtae_rel_pct_fw |
| 59 | `4.0` | ambiguous | bonn.vits.delta_int3_rht_K8.bits, bonn.vits.lam16_b4.bits, hard.sintel.vits.int3_g16.tae_rel_pct_ci_hi (+2) |
| 63 | `3.0` | ambiguous | bonn.vits.delta_int3_int2_K8.bits, bonn.vits.delta_lam16_b2_K8.tae, bonn.vits.int3_gT.bits (+7) |
| 63 | `5.33` | ambiguous | mem.vits.lam16_kv_b3.cache_ratio_vs_fp32 |
| 64 | `1.21` | ambiguous | sintel.vitl.int2_g16.tae_rel_pct_fw_ci_lo |
| 89 | `0.711` | specific | bonn.vits.delta_int3_bin_K16.covtae_rel_pct_fw |
| 112 | `4.5` | ambiguous | bonn.vits.int2_g16.covtae_rel_pct_ci_hi, bonn.vits.int2_g16.covtae_rel_pct_fw_ci_hi, bonn.vits.int4_g16.bits (+2) |
| 113 | `0.004` | ambiguous | hard.sintel.vits.e8_g16_b3.delta1_agg_diff, hard.sintel.vits.e8_g64_b3.delta1_agg_diff, mem.vits.lam16_b4.depth_rel_l1_mean (+8) |
| 114 | `3.5` | ambiguous | bonn.vits.int3_g16.bits, bonn.vits.int3_g16_rht.bits, hard.sintel.vits.e8_g16_b3.bits (+7) |
| 115 | `12.5` | ambiguous | bonn.vits.delta_lam16_b2_K8.covtae_rel_pct_fw |
| 115 | `4.0` | ambiguous | bonn.vits.delta_int3_rht_K8.bits, bonn.vits.lam16_b4.bits, hard.sintel.vits.int3_g16.tae_rel_pct_ci_hi (+2) |
| 118 | `+0.025` | specific | bonn.vits.lam16_b4.tae_rel_pct |
| 125 | `0.013` | specific | probe.bonn.residual_var_ratio_min |
| 126 | `3.0` | ambiguous | bonn.vits.delta_int3_int2_K8.bits, bonn.vits.delta_lam16_b2_K8.tae, bonn.vits.int3_gT.bits (+7) |
| 127 | `1.21` | ambiguous | sintel.vitl.int2_g16.tae_rel_pct_fw_ci_lo |
| 199 | `1.07` | ambiguous | sintel.vitb.int2_g16.tae_rel_pct, sintel.vits.int3_gT.covtae_rel_pct_fw_ci_lo |
| 250 | `3.5` | ambiguous | bonn.vits.int3_g16.bits, bonn.vits.int3_g16_rht.bits, hard.sintel.vits.e8_g16_b3.bits (+7) |
| 251 | `4.0` | ambiguous | bonn.vits.delta_int3_rht_K8.bits, bonn.vits.lam16_b4.bits, hard.sintel.vits.int3_g16.tae_rel_pct_ci_hi (+2) |
| 261 | `0.5` | ambiguous | bonn.vits.delta_int3_bin_K16.covtae_rel_pct_ci_lo, bonn.vits.delta_int3_bin_K16.covtae_rel_pct_fw_ci_lo, bonn.vits.delta_int3_bin_K8.covtae_rel_pct_fw (+25) |
| 265 | `3.0` | ambiguous | bonn.vits.delta_int3_int2_K8.bits, bonn.vits.delta_lam16_b2_K8.tae, bonn.vits.int3_gT.bits (+7) |
| 265 | `5.33` | ambiguous | mem.vits.lam16_kv_b3.cache_ratio_vs_fp32 |
| 273 | `0.5` | ambiguous | bonn.vits.delta_int3_bin_K16.covtae_rel_pct_ci_lo, bonn.vits.delta_int3_bin_K16.covtae_rel_pct_fw_ci_lo, bonn.vits.delta_int3_bin_K8.covtae_rel_pct_fw (+25) |
| 274 | `0.31` | ambiguous | bonn.vits.delta_int3_int2_K8.covtae_rel_pct_ci_lo, bonn.vits.delta_int3_int2_K8.covtae_rel_pct_fw_ci_lo, hard.sintel.vits.int4_tensor.absrel |
| 275 | `3.5` | ambiguous | bonn.vits.int3_g16.bits, bonn.vits.int3_g16_rht.bits, hard.sintel.vits.e8_g16_b3.bits (+7) |
| 276 | `4.0` | ambiguous | bonn.vits.delta_int3_rht_K8.bits, bonn.vits.lam16_b4.bits, hard.sintel.vits.int3_g16.tae_rel_pct_ci_hi (+2) |
| 377 | `0.005` | ambiguous | hard.sintel.vits.e8_g32_b3.absrel_diff, mem.vitb.delta_int3_bin_K8.depth_rel_l1_mean, mem.vits.lam16_b4.depth_rel_l1_max (+11) |
| 380 | `4.5` | ambiguous | bonn.vits.int2_g16.covtae_rel_pct_ci_hi, bonn.vits.int2_g16.covtae_rel_pct_fw_ci_hi, bonn.vits.int4_g16.bits (+2) |
| 380 | `0.005` | ambiguous | hard.sintel.vits.e8_g32_b3.absrel_diff, mem.vitb.delta_int3_bin_K8.depth_rel_l1_mean, mem.vits.lam16_b4.depth_rel_l1_max (+11) |
| 382 | `+0.013` | specific | probe.bonn.residual_var_ratio_min |
| 382 | `+0.006` | specific | hard.sintel.vits.int3_g64.delta1_agg_diff, sintel.vits.bin_g16.delta1_diff_fw_ci_hi |
| 383 | `0.004` | ambiguous | hard.sintel.vits.e8_g16_b3.delta1_agg_diff, hard.sintel.vits.e8_g64_b3.delta1_agg_diff, mem.vits.lam16_b4.depth_rel_l1_mean (+8) |
| 384 | `+0.005` | ambiguous | hard.sintel.vits.e8_g32_b3.absrel_diff, mem.vitb.delta_int3_bin_K8.depth_rel_l1_mean, mem.vits.lam16_b4.depth_rel_l1_max (+4) |
| 384 | `-0.005` | ambiguous | sintel.vitb.int2_g16.delta1_diff_fw_ci_lo, sintel.vitb.zero.delta1_diff, sintel.vitl.int2_g16.delta1_diff_ci_lo (+4) |
| 384 | `6.0` | ambiguous | hard.sintel.vits.e8_g64_b3.tae_rel_pct_ci_hi, hard.sintel.vits.e8_g64_b3.tae_rel_pct_fw_ci_hi, hard.sintel.vits.int2_g64.covtae (+1) |
| 384 | `+0.004` | specific | mem.vits.lam16_b4.depth_rel_l1_mean, sintel.vitb.zero.delta1_diff_ci_hi, sintel.vitb.zero.delta1_diff_fw_ci_hi |
| 385 | `-0.004` | ambiguous | hard.sintel.vits.e8_g16_b3.delta1_agg_diff, hard.sintel.vits.e8_g64_b3.delta1_agg_diff, sintel.vitb.int2_g16.delta1_diff_ci_lo (+5) |
| 387 | `3.5` | ambiguous | bonn.vits.int3_g16.bits, bonn.vits.int3_g16_rht.bits, hard.sintel.vits.e8_g16_b3.bits (+7) |
| 388 | `0.005` | ambiguous | hard.sintel.vits.e8_g32_b3.absrel_diff, mem.vitb.delta_int3_bin_K8.depth_rel_l1_mean, mem.vits.lam16_b4.depth_rel_l1_max (+11) |
| 388 | `-0.010` | specific | bonn.vits.delta_lam16_b3_K8.covtae_rel_pct, hard.sintel.vits.e8_g32_b3.delta1_agg_diff, sintel.vits.lam16_b2.delta1_diff_fw_ci_lo |
| 388 | `0.003` | ambiguous | hard.sintel.vits.e8_g64_b3.absrel_diff, mem.vits.delta_lam16_b3_K8.depth_rel_l1_mean, mem.vits.lam16_kv_b4.depth_rel_l1_max (+8) |
| 389 | `4.0` | ambiguous | bonn.vits.delta_int3_rht_K8.bits, bonn.vits.lam16_b4.bits, hard.sintel.vits.int3_g16.tae_rel_pct_ci_hi (+2) |
| 389 | `12.5` | ambiguous | bonn.vits.delta_lam16_b2_K8.covtae_rel_pct_fw |
| 389 | `-0.007` | specific | sintel.vits.int2_g16.delta1_agg_diff, sintel.vits.int2_g16.delta1_diff_fw, sintel.vits.zero.absrel_diff |
| 391 | `4.5` | ambiguous | bonn.vits.int2_g16.covtae_rel_pct_ci_hi, bonn.vits.int2_g16.covtae_rel_pct_fw_ci_hi, bonn.vits.int4_g16.bits (+2) |
| 392 | `3.5` | ambiguous | bonn.vits.int3_g16.bits, bonn.vits.int3_g16_rht.bits, hard.sintel.vits.e8_g16_b3.bits (+7) |
| 402 | `3.5` | ambiguous | bonn.vits.int3_g16.bits, bonn.vits.int3_g16_rht.bits, hard.sintel.vits.e8_g16_b3.bits (+7) |
| 404 | `0.005` | ambiguous | hard.sintel.vits.e8_g32_b3.absrel_diff, mem.vitb.delta_int3_bin_K8.depth_rel_l1_mean, mem.vits.lam16_b4.depth_rel_l1_max (+11) |
| 412 | `32.0` | ambiguous | bonn.vits.1sc.fp32.bits, bonn.vits.fp32.bits, hard.sintel.vits.fp32.bits (+3) |
| 412 | `0.438` | specific | sintel.vits.int3_g16.covtae_rel_pct |
| 412 | `0.091` | specific | bonn.vits.delta_int2_rht_K8.tae_rel_pct_ci_hi, bonn.vits.delta_int2_rht_K8.tae_rel_pct_fw_ci_hi |
| 413 | `0.095` | specific | bonn.vits.lam16_b3.tae_rel_pct_fw |
| 413 | `0.437` | specific | sintel.vits.delta_int3_int2_K8.covtae_rel_pct_ci_hi |
| 413 | `0.092` | specific | bonn.vits.delta_int3_rht_K8.tae_rel_pct_ci_lo, bonn.vits.delta_int3_rht_K8.tae_rel_pct_fw_ci_lo |
| 414 | `6.0` | ambiguous | hard.sintel.vits.e8_g64_b3.tae_rel_pct_ci_hi, hard.sintel.vits.e8_g64_b3.tae_rel_pct_fw_ci_hi, hard.sintel.vits.int2_g64.covtae (+1) |
| 414 | `0.438` | specific | sintel.vits.int3_g16.covtae_rel_pct |
| 414 | `0.091` | specific | bonn.vits.delta_int2_rht_K8.tae_rel_pct_ci_hi, bonn.vits.delta_int2_rht_K8.tae_rel_pct_fw_ci_hi |
| 415 | `4.0` | ambiguous | bonn.vits.delta_int3_rht_K8.bits, bonn.vits.lam16_b4.bits, hard.sintel.vits.int3_g16.tae_rel_pct_ci_hi (+2) |
| 415 | `0.097` | specific | sintel.vits.delta_int3_int2_K8.tae_rel_pct, sintel.vits.int2_g16.tae_rel_pct_ci_lo |
| 415 | `0.090` | specific | bonn.vits.int4_g16.covtae_rel_pct_ci_hi, bonn.vits.int4_g16.covtae_rel_pct_fw_ci_hi, mem.vits.delta_lam16_b2_K8.depth_rel_l1_mean |
| 416 | `0.438` | specific | sintel.vits.int3_g16.covtae_rel_pct |
| 416 | `0.090` | specific | bonn.vits.int4_g16.covtae_rel_pct_ci_hi, bonn.vits.int4_g16.covtae_rel_pct_fw_ci_hi, mem.vits.delta_lam16_b2_K8.depth_rel_l1_mean |
| 417 | `6.0` | ambiguous | hard.sintel.vits.e8_g64_b3.tae_rel_pct_ci_hi, hard.sintel.vits.e8_g64_b3.tae_rel_pct_fw_ci_hi, hard.sintel.vits.int2_g64.covtae (+1) |
| 417 | `0.097` | specific | sintel.vits.delta_int3_int2_K8.tae_rel_pct, sintel.vits.int2_g16.tae_rel_pct_ci_lo |
| 417 | `0.091` | specific | bonn.vits.delta_int2_rht_K8.tae_rel_pct_ci_hi, bonn.vits.delta_int2_rht_K8.tae_rel_pct_fw_ci_hi |
| 418 | `3.5` | ambiguous | bonn.vits.int3_g16.bits, bonn.vits.int3_g16_rht.bits, hard.sintel.vits.e8_g16_b3.bits (+7) |
| 418 | `0.005` | ambiguous | hard.sintel.vits.e8_g32_b3.absrel_diff, mem.vitb.delta_int3_bin_K8.depth_rel_l1_mean, mem.vits.lam16_b4.depth_rel_l1_max (+11) |
| 418 | `0.003` | ambiguous | hard.sintel.vits.e8_g64_b3.absrel_diff, mem.vits.delta_lam16_b3_K8.depth_rel_l1_mean, mem.vits.lam16_kv_b4.depth_rel_l1_max (+8) |
| 419 | `4.5` | ambiguous | bonn.vits.int2_g16.covtae_rel_pct_ci_hi, bonn.vits.int2_g16.covtae_rel_pct_fw_ci_hi, bonn.vits.int4_g16.bits (+2) |
| 419 | `0.436` | specific | hard.sintel.vits.int3_tensor.absrel |
| 419 | `0.091` | specific | bonn.vits.delta_int2_rht_K8.tae_rel_pct_ci_hi, bonn.vits.delta_int2_rht_K8.tae_rel_pct_fw_ci_hi |
| 421 | `32.0` | ambiguous | bonn.vits.1sc.fp32.bits, bonn.vits.fp32.bits, hard.sintel.vits.fp32.bits (+3) |
| 421 | `0.501` | specific | bonn.vits.int3_g16.covtae_rel_pct_fw |
| 422 | `3.5` | ambiguous | bonn.vits.int3_g16.bits, bonn.vits.int3_g16_rht.bits, hard.sintel.vits.e8_g16_b3.bits (+7) |
| 422 | `0.090` | specific | bonn.vits.int4_g16.covtae_rel_pct_ci_hi, bonn.vits.int4_g16.covtae_rel_pct_fw_ci_hi, mem.vits.delta_lam16_b2_K8.depth_rel_l1_mean |
| 423 | `4.5` | ambiguous | bonn.vits.int2_g16.covtae_rel_pct_ci_hi, bonn.vits.int2_g16.covtae_rel_pct_fw_ci_hi, bonn.vits.int4_g16.bits (+2) |
| 423 | `0.109` | specific | bonn.vits.delta_int3_rht_K8.covtae_rel_pct_fw, sintel.vitb.delta_int3_bin_K8.tae_rel_pct |
| 423 | `0.084` | specific | bonn.vits.delta_int3_rht_K8.covtae_rel_pct_ci_hi, bonn.vits.delta_int3_rht_K8.covtae_rel_pct_fw_ci_hi |
| 425 | `32.0` | ambiguous | bonn.vits.1sc.fp32.bits, bonn.vits.fp32.bits, hard.sintel.vits.fp32.bits (+3) |
| 425 | `0.106` | specific | bonn.vits.int4_g64_asym.tae_rel_pct_ci_lo, bonn.vits.int4_g64_asym.tae_rel_pct_fw_ci_lo |
| 425 | `0.081` | specific | mem.vitl.int2_g16.depth_rel_l1_max |
| 426 | `3.5` | ambiguous | bonn.vits.int3_g16.bits, bonn.vits.int3_g16_rht.bits, hard.sintel.vits.e8_g16_b3.bits (+7) |
| 426 | `0.091` | specific | bonn.vits.delta_int2_rht_K8.tae_rel_pct_ci_hi, bonn.vits.delta_int2_rht_K8.tae_rel_pct_fw_ci_hi |
| 426 | `0.414` | specific | sintel.vitb.int2_g16.tae_rel_pct_fw_ci_lo |
| 427 | `4.5` | ambiguous | bonn.vits.int2_g16.covtae_rel_pct_ci_hi, bonn.vits.int2_g16.covtae_rel_pct_fw_ci_hi, bonn.vits.int4_g16.bits (+2) |
| 438 | `3.5` | ambiguous | bonn.vits.int3_g16.bits, bonn.vits.int3_g16_rht.bits, hard.sintel.vits.e8_g16_b3.bits (+7) |
| 438 | `0.031` | specific | hard.sintel.vits.int4_tensor.absrel_diff, mem.vitl.delta_int3_bin_K8.depth_rel_l1_max |
| 438 | `0.012` | specific | bonn.vits.delta_int3_bin_K8.tae_rel_pct |
| 439 | `0.014` | specific | sintel.vits.zero.delta1_diff_ci_lo |
| 439 | `4.5` | ambiguous | bonn.vits.int2_g16.covtae_rel_pct_ci_hi, bonn.vits.int2_g16.covtae_rel_pct_fw_ci_hi, bonn.vits.int4_g16.bits (+2) |
| 439 | `-0.004` | ambiguous | hard.sintel.vits.e8_g16_b3.delta1_agg_diff, hard.sintel.vits.e8_g64_b3.delta1_agg_diff, sintel.vitb.int2_g16.delta1_diff_ci_lo (+5) |
| 439 | `0.004` | ambiguous | hard.sintel.vits.e8_g16_b3.delta1_agg_diff, hard.sintel.vits.e8_g64_b3.delta1_agg_diff, mem.vits.lam16_b4.depth_rel_l1_mean (+8) |
| 440 | `-0.002` | ambiguous | bonn.vits.zero.delta1_diff_ci_lo, bonn.vits.zero.delta1_diff_fw_ci_lo, sintel.vitl.int2_g16.delta1_agg_diff (+13) |
| 443 | `+0.025` | specific | bonn.vits.lam16_b4.tae_rel_pct |
| 443 | `3.5` | ambiguous | bonn.vits.int3_g16.bits, bonn.vits.int3_g16_rht.bits, hard.sintel.vits.e8_g16_b3.bits (+7) |
| 443 | `+0.013` | specific | probe.bonn.residual_var_ratio_min |
| 443 | `+0.006` | specific | hard.sintel.vits.int3_g64.delta1_agg_diff, sintel.vits.bin_g16.delta1_diff_fw_ci_hi |
| 444 | `4.5` | ambiguous | bonn.vits.int2_g16.covtae_rel_pct_ci_hi, bonn.vits.int2_g16.covtae_rel_pct_fw_ci_hi, bonn.vits.int4_g16.bits (+2) |
| 452 | `0.025` | specific | bonn.vits.delta_lam16_b3_K8.tae_rel_pct_fw, bonn.vits.lam16_b4.tae_rel_pct |
| 453 | `-0.004` | ambiguous | hard.sintel.vits.e8_g16_b3.delta1_agg_diff, hard.sintel.vits.e8_g64_b3.delta1_agg_diff, sintel.vitb.int2_g16.delta1_diff_ci_lo (+5) |
| 454 | `+0.003` | ambiguous | hard.sintel.vits.e8_g64_b3.absrel_diff, mem.vits.delta_lam16_b3_K8.depth_rel_l1_mean, mem.vits.lam16_kv_b4.depth_rel_l1_max (+3) |
| 485 | `0.070` | specific | bonn.vits.int4_g64_asym.tae_rel_pct |
| 499 | `3.500` | ambiguous | bonn.vits.int3_g16.bits, bonn.vits.int3_g16_rht.bits, hard.sintel.vits.e8_g16_b3.bits (+2) |
| 500 | `3.250` | specific | hard.sintel.vits.e8_g32_b3.bits |
| 502 | `3.125` | ambiguous | hard.sintel.vits.e8_g64_b3.bits, hard.sintel.vits.int3_g64.bits, hard.sintel.vits.int3_tensor.bits (+1) |
| 511 | `3.5` | ambiguous | bonn.vits.int3_g16.bits, bonn.vits.int3_g16_rht.bits, hard.sintel.vits.e8_g16_b3.bits (+7) |
| 513 | `0.035` | specific | bonn.vits.delta_int3_bin_K8.tae_rel_pct_fw |
| 514 | `0.034` | specific | bonn.vits.delta_lam16_b3_K8.tae_rel_pct_ci_lo, bonn.vits.delta_lam16_b3_K8.tae_rel_pct_fw_ci_lo |
| 514 | `3.2` | ambiguous | sintel.vits.delta_int3_bin_K8.tae_rel_pct_fw_ci_hi |
| 524 | `3.5` | ambiguous | bonn.vits.int3_g16.bits, bonn.vits.int3_g16_rht.bits, hard.sintel.vits.e8_g16_b3.bits (+7) |
| 531 | `0.231` | specific | sintel.vits.int2_g16.absrel |
| 534 | `0.712` | specific | sintel.vits.int2_g16.delta1 |
| 534 | `0.234` | specific | mem.vitb.int2_g16.depth_rel_l1_max |
| 534 | `55.41` | ambiguous | sintel.vitl.zero.tae |
| 534 | `6.38` | ambiguous | hard.sintel.vits.int2_g16.covtae |
| 535 | `0.678` | specific | hard.sintel.vits.e8_g32_b3.delta1, sintel.vits.int3_g16.covtae_rel_pct_ci_lo |
| 535 | `0.258` | specific | bonn.vits.delta_int2_K8.covtae_rel_pct_ci_lo, bonn.vits.delta_int2_K8.covtae_rel_pct_fw_ci_lo, bonn.vits.delta_int3_int2_K8.tae_rel_pct_fw |
| 537 | `0.711` | specific | bonn.vits.delta_int3_bin_K16.covtae_rel_pct_fw |
| 537 | `0.232` | specific | bonn.vits.int3_gT.tae_rel_pct |
| 548 | `4.0` | ambiguous | bonn.vits.delta_int3_rht_K8.bits, bonn.vits.lam16_b4.bits, hard.sintel.vits.int3_g16.tae_rel_pct_ci_hi (+2) |
| 549 | `0.001` | ambiguous | bonn.vits.bin_g16.delta1_diff_ci_lo, bonn.vits.bin_g16.delta1_diff_fw_ci_lo, bonn.vits.delta_lam16_b2_K8.absrel_diff (+59) |
| 550 | `0.001` | ambiguous | bonn.vits.bin_g16.delta1_diff_ci_lo, bonn.vits.bin_g16.delta1_diff_fw_ci_lo, bonn.vits.delta_lam16_b2_K8.absrel_diff (+59) |
| 550 | `3.5` | ambiguous | bonn.vits.int3_g16.bits, bonn.vits.int3_g16_rht.bits, hard.sintel.vits.e8_g16_b3.bits (+7) |
| 551 | `0.987` | specific | bonn.vits.delta_int3_bin_K16.covtae_rel_pct_ci_hi, bonn.vits.delta_int3_bin_K16.covtae_rel_pct_fw_ci_hi, bonn.vits.zero.delta1 |
| 551 | `0.053` | specific | bonn.vits.int4_g64_asym.tae_rel_pct_ci_hi, bonn.vits.int4_g64_asym.tae_rel_pct_fw_ci_hi |
| 552 | `0.043` | specific | bonn.vits.int4_g16.tae_rel_pct_ci_hi, bonn.vits.int4_g16.tae_rel_pct_fw_ci_hi |
| 553 | `3.5` | ambiguous | bonn.vits.int3_g16.bits, bonn.vits.int3_g16_rht.bits, hard.sintel.vits.e8_g16_b3.bits (+7) |
| 553 | `2.61` | ambiguous | sintel.vitb.int2_g16.tae_rel_pct_fw |
| 562 | `0.005` | ambiguous | hard.sintel.vits.e8_g32_b3.absrel_diff, mem.vitb.delta_int3_bin_K8.depth_rel_l1_mean, mem.vits.lam16_b4.depth_rel_l1_max (+11) |
| 570 | `32.0` | ambiguous | bonn.vits.1sc.fp32.bits, bonn.vits.fp32.bits, hard.sintel.vits.fp32.bits (+3) |
| 570 | `0.987` | specific | bonn.vits.delta_int3_bin_K16.covtae_rel_pct_ci_hi, bonn.vits.delta_int3_bin_K16.covtae_rel_pct_fw_ci_hi, bonn.vits.zero.delta1 |
| 570 | `0.043` | specific | bonn.vits.int4_g16.tae_rel_pct_ci_hi, bonn.vits.int4_g16.tae_rel_pct_fw_ci_hi |
| 571 | `0.987` | specific | bonn.vits.delta_int3_bin_K16.covtae_rel_pct_ci_hi, bonn.vits.delta_int3_bin_K16.covtae_rel_pct_fw_ci_hi, bonn.vits.zero.delta1 |
| 571 | `0.043` | specific | bonn.vits.int4_g16.tae_rel_pct_ci_hi, bonn.vits.int4_g16.tae_rel_pct_fw_ci_hi |
| 572 | `4.0` | ambiguous | bonn.vits.delta_int3_rht_K8.bits, bonn.vits.lam16_b4.bits, hard.sintel.vits.int3_g16.tae_rel_pct_ci_hi (+2) |
| 572 | `0.987` | specific | bonn.vits.delta_int3_bin_K16.covtae_rel_pct_ci_hi, bonn.vits.delta_int3_bin_K16.covtae_rel_pct_fw_ci_hi, bonn.vits.zero.delta1 |
| 572 | `0.044` | ambiguous | bonn.vits.1sc.zero.absrel, bonn.vits.delta_int2_K8.absrel, bonn.vits.delta_int2_rht_K8.absrel (+17) |
| 572 | `2.75` | ambiguous | hard.sintel.vits.e8_g32_b3.bits_inventory |
| 572 | `1.71` | ambiguous | bonn.vits.delta_int2_rht_K8.covtae, bonn.vits.delta_int3_int2_K8.covtae, bonn.vits.delta_int3_rht_K8.covtae (+11) |
| 573 | `3.5` | ambiguous | bonn.vits.int3_g16.bits, bonn.vits.int3_g16_rht.bits, hard.sintel.vits.e8_g16_b3.bits (+7) |
| 573 | `0.053` | specific | bonn.vits.int4_g64_asym.tae_rel_pct_ci_hi, bonn.vits.int4_g64_asym.tae_rel_pct_fw_ci_hi |
| 573 | `2.61` | ambiguous | sintel.vitb.int2_g16.tae_rel_pct_fw |
| 574 | `4.5` | ambiguous | bonn.vits.int2_g16.covtae_rel_pct_ci_hi, bonn.vits.int2_g16.covtae_rel_pct_fw_ci_hi, bonn.vits.int4_g16.bits (+2) |
| 574 | `0.987` | specific | bonn.vits.delta_int3_bin_K16.covtae_rel_pct_ci_hi, bonn.vits.delta_int3_bin_K16.covtae_rel_pct_fw_ci_hi, bonn.vits.zero.delta1 |
| 574 | `0.043` | specific | bonn.vits.int4_g16.tae_rel_pct_ci_hi, bonn.vits.int4_g16.tae_rel_pct_fw_ci_hi |
| 583 | `3.5` | ambiguous | bonn.vits.int3_g16.bits, bonn.vits.int3_g16_rht.bits, hard.sintel.vits.e8_g16_b3.bits (+7) |
| 585 | `0.005` | ambiguous | hard.sintel.vits.e8_g32_b3.absrel_diff, mem.vitb.delta_int3_bin_K8.depth_rel_l1_mean, mem.vits.lam16_b4.depth_rel_l1_max (+11) |
| 585 | `0.003` | ambiguous | hard.sintel.vits.e8_g64_b3.absrel_diff, mem.vits.delta_lam16_b3_K8.depth_rel_l1_mean, mem.vits.lam16_kv_b4.depth_rel_l1_max (+8) |
| 585 | `0.011` | specific | mem.vits.lam16_b3.depth_rel_l1_max, sintel.vits.int2_g16_rht.delta1_diff_fw_ci_hi, sintel.vits.zero.delta1_diff_ci_hi |
| 586 | `0.011` | specific | mem.vits.lam16_b3.depth_rel_l1_max, sintel.vits.int2_g16_rht.delta1_diff_fw_ci_hi, sintel.vits.zero.delta1_diff_ci_hi |
| 586 | `0.711` | specific | bonn.vits.delta_int3_bin_K16.covtae_rel_pct_fw |
| 591 | `3.5` | ambiguous | bonn.vits.int3_g16.bits, bonn.vits.int3_g16_rht.bits, hard.sintel.vits.e8_g16_b3.bits (+7) |
| 606 | `0.005` | ambiguous | hard.sintel.vits.e8_g32_b3.absrel_diff, mem.vitb.delta_int3_bin_K8.depth_rel_l1_mean, mem.vits.lam16_b4.depth_rel_l1_max (+11) |
| 606 | `0.003` | ambiguous | hard.sintel.vits.e8_g64_b3.absrel_diff, mem.vits.delta_lam16_b3_K8.depth_rel_l1_mean, mem.vits.lam16_kv_b4.depth_rel_l1_max (+8) |
| 606 | `0.011` | specific | mem.vits.lam16_b3.depth_rel_l1_max, sintel.vits.int2_g16_rht.delta1_diff_fw_ci_hi, sintel.vits.zero.delta1_diff_ci_hi |
| 636 | `+0.001` | ambiguous | bonn.vits.delta_lam16_b2_K8.absrel_diff, bonn.vits.fp16.tae_rel_pct_ci_hi, bonn.vits.fp16.tae_rel_pct_ci_lo (+24) |
| 636 | `0.002` | ambiguous | bonn.vits.1sc.zero.absrel_diff, bonn.vits.delta_int4_bin_K8.tae_rel_pct_fw, bonn.vits.fp16.covtae_rel_pct_ci_hi (+26) |
| 636 | `0.007` | ambiguous | mem.vitl.delta_int3_bin_K8.depth_rel_l1_mean, sintel.vitl.zero.absrel_diff, sintel.vits.delta_int3_int2_K8.tae_rel_pct_ci_hi (+3) |
| 637 | `+0.001` | ambiguous | bonn.vits.delta_lam16_b2_K8.absrel_diff, bonn.vits.fp16.tae_rel_pct_ci_hi, bonn.vits.fp16.tae_rel_pct_ci_lo (+24) |
| 637 | `0.008` | ambiguous | hard.sintel.vits.e8_g16_b3.absrel_diff, mem.vits.delta_lam16_b3_K8.depth_rel_l1_max, mem.vits.lam16_b3.depth_rel_l1_mean (+2) |
| 637 | `0.029` | specific | bonn.vits.lam16_b4.tae_rel_pct_ci_hi, bonn.vits.lam16_b4.tae_rel_pct_fw_ci_hi |
| 638 | `+0.001` | ambiguous | bonn.vits.delta_lam16_b2_K8.absrel_diff, bonn.vits.fp16.tae_rel_pct_ci_hi, bonn.vits.fp16.tae_rel_pct_ci_lo (+24) |
| 638 | `0.007` | ambiguous | mem.vitl.delta_int3_bin_K8.depth_rel_l1_mean, sintel.vitl.zero.absrel_diff, sintel.vits.delta_int3_int2_K8.tae_rel_pct_ci_hi (+3) |
| 638 | `0.025` | specific | bonn.vits.delta_lam16_b3_K8.tae_rel_pct_fw, bonn.vits.lam16_b4.tae_rel_pct |
| 639 | `+0.005` | ambiguous | hard.sintel.vits.e8_g32_b3.absrel_diff, mem.vitb.delta_int3_bin_K8.depth_rel_l1_mean, mem.vits.lam16_b4.depth_rel_l1_max (+4) |
| 639 | `0.006` | ambiguous | hard.sintel.vits.int3_g16.absrel_diff, hard.sintel.vits.int3_g64.delta1_agg_diff, sintel.vitb.zero.delta1_agg_diff (+4) |
| 639 | `0.012` | specific | bonn.vits.delta_int3_bin_K8.tae_rel_pct |
| 640 | `0.015` | ambiguous | bonn.vits.delta_lam16_b3_K8.covtae_rel_pct_ci_hi, bonn.vits.delta_lam16_b3_K8.covtae_rel_pct_fw_ci_hi, bonn.vits.lam16_b4.covtae_rel_pct_ci_lo (+3) |
| 645 | `0.03` | ambiguous | bonn.vits.delta_int3_bin_K8.tae_rel_pct_fw, bonn.vits.delta_lam16_b3_K8.tae_rel_pct_ci_lo, bonn.vits.delta_lam16_b3_K8.tae_rel_pct_fw_ci_lo (+13) |
| 646 | `0.005` | ambiguous | hard.sintel.vits.e8_g32_b3.absrel_diff, mem.vitb.delta_int3_bin_K8.depth_rel_l1_mean, mem.vits.lam16_b4.depth_rel_l1_max (+11) |
| 648 | `0.007` | ambiguous | mem.vitl.delta_int3_bin_K8.depth_rel_l1_mean, sintel.vitl.zero.absrel_diff, sintel.vits.delta_int3_int2_K8.tae_rel_pct_ci_hi (+3) |
| 657 | `0.007` | ambiguous | mem.vitl.delta_int3_bin_K8.depth_rel_l1_mean, sintel.vitl.zero.absrel_diff, sintel.vits.delta_int3_int2_K8.tae_rel_pct_ci_hi (+3) |
| 658 | `0.002` | ambiguous | bonn.vits.1sc.zero.absrel_diff, bonn.vits.delta_int4_bin_K8.tae_rel_pct_fw, bonn.vits.fp16.covtae_rel_pct_ci_hi (+26) |
| 688 | `3.0` | ambiguous | bonn.vits.delta_int3_int2_K8.bits, bonn.vits.delta_lam16_b2_K8.tae, bonn.vits.int3_gT.bits (+7) |
| 688 | `5.33` | ambiguous | mem.vits.lam16_kv_b3.cache_ratio_vs_fp32 |
| 703 | `0.0000` | ambiguous | bonn.vits.1sc.zero.bits, bonn.vits.1sc.zero.cache_mb, bonn.vits.bin_g16.delta1_diff_ci_hi (+136) |
| 711 | `5.33` | ambiguous | mem.vits.lam16_kv_b3.cache_ratio_vs_fp32 |
| 713 | `1.21` | ambiguous | sintel.vitl.int2_g16.tae_rel_pct_fw_ci_lo |
| 714 | `0.73` | ambiguous | sintel.vitb.int2_g16.tae_rel_pct_ci_lo, sintel.vits.int2_g16.tae_rel_pct_fw |
| 716 | `0.80` | ambiguous | bonn.vits.delta_int3_bin_K8.covtae_rel_pct_ci_hi, bonn.vits.delta_int3_bin_K8.covtae_rel_pct_fw_ci_hi |
| 716 | `0.73` | ambiguous | sintel.vitb.int2_g16.tae_rel_pct_ci_lo, sintel.vits.int2_g16.tae_rel_pct_fw |
| 738 | `1.00` | ambiguous | bonn.vits.1sc.n_scenes, mem.vits.fp16_kv.cache_ratio_vs_fp32, probe.bonn.bits_saving_min |
| 738 | `0.19` | ambiguous | hard.sintel.vits.e8_g16_b2.absrel_diff, hard.sintel.vits.int2_g64.absrel_diff, sintel.vits.delta_int3_int2_K8.covtae_rel_pct_fw (+1) |
| 738 | `5.33` | ambiguous | mem.vits.lam16_kv_b3.cache_ratio_vs_fp32 |
| 738 | `2.01` | ambiguous | sintel.vits.delta_int3_bin_K16.covtae_rel_pct_fw, sintel.vits.int2_g16.covtae_rel_pct_fw_ci_lo |
| 738 | `1.18` | ambiguous | sintel.vits.lam16_b2.tae_rel_pct_ci_lo |
| 738 | `1.70` | ambiguous | bonn.vits.int3_g16.covtae, bonn.vits.int3_g16_rht.covtae |
| 739 | `0.53` | ambiguous | bonn.vits.delta_int3_bin_K16.covtae_rel_pct_ci_lo, bonn.vits.delta_int3_bin_K16.covtae_rel_pct_fw_ci_lo, hard.sintel.vits.lam16_g64_b3.tae_rel_pct_ci_hi (+3) |
| 739 | `0.10` | ambiguous | bonn.vits.int4_g16.tae_rel_pct_ci_lo, bonn.vits.int4_g16.tae_rel_pct_fw_ci_lo, bonn.vits.int4_g16_rht.covtae_rel_pct_fw (+6) |
| 739 | `5.33` | ambiguous | mem.vits.lam16_kv_b3.cache_ratio_vs_fp32 |
| 739 | `0.23` | ambiguous | bonn.vits.int3_gT.tae_rel_pct, mem.vitb.int2_g16.depth_rel_l1_max, sintel.vits.delta_int2_K8.absrel (+9) |
| 739 | `1.18` | ambiguous | sintel.vits.lam16_b2.tae_rel_pct_ci_lo |
| 739 | `0.27` | ambiguous | bonn.vits.int3_g16.tae_rel_pct, hard.sintel.vits.int2_g16.delta1_agg_diff, hard.sintel.vits.int3_g16.absrel |
| 740 | `0.33` | ambiguous | bonn.vits.int3_g16.covtae_rel_pct_ci_hi, bonn.vits.int3_g16.covtae_rel_pct_fw_ci_hi, bonn.vits.int3_gT.covtae_rel_pct_fw (+3) |
| 740 | `0.06` | ambiguous | bonn.vits.delta_int3_bin_K16.tae_rel_pct, bonn.vits.delta_int3_rht_K8.tae_rel_pct_ci_hi, bonn.vits.delta_int3_rht_K8.tae_rel_pct_fw_ci_hi (+6) |
| 740 | `5.33` | ambiguous | mem.vits.lam16_kv_b3.cache_ratio_vs_fp32 |
| 740 | `0.16` | ambiguous | hard.sintel.vits.e8_g64_b2.absrel_diff, hard.sintel.vits.int3_tensor.absrel_diff |
| 740 | `1.09` | ambiguous | mem.vits.delta_fp32_K8.peak_ratio_vs_fp32 |
| 740 | `0.29` | ambiguous | bonn.vits.delta_int2_rht_K8.tae_rel_pct_ci_lo, bonn.vits.delta_int2_rht_K8.tae_rel_pct_fw_ci_lo, bonn.vits.delta_int3_int2_K8.tae_rel_pct_ci_lo (+10) |
| 741 | `1.34` | ambiguous | bonn.vits.delta_int2_K8.covtae_rel_pct_ci_hi, bonn.vits.delta_int2_K8.covtae_rel_pct_fw_ci_hi |
| 741 | `0.25` | ambiguous | bonn.vits.delta_int2_rht_K8.tae_rel_pct_fw, bonn.vits.delta_int3_int2_K8.tae_rel_pct, sintel.vits.delta_int3_int2_K8.covtae_rel_pct_fw_ci_hi |
| 741 | `5.33` | ambiguous | mem.vits.lam16_kv_b3.cache_ratio_vs_fp32 |
| 741 | `0.80` | ambiguous | bonn.vits.delta_int3_bin_K8.covtae_rel_pct_ci_hi, bonn.vits.delta_int3_bin_K8.covtae_rel_pct_fw_ci_hi |
| 742 | `1.00` | ambiguous | bonn.vits.1sc.n_scenes, mem.vits.fp16_kv.cache_ratio_vs_fp32, probe.bonn.bits_saving_min |
| 742 | `5.33` | ambiguous | mem.vits.lam16_kv_b3.cache_ratio_vs_fp32 |
| 742 | `1.87` | ambiguous | sintel.vitb.delta_int3_bin_K8.bits, sintel.vits.delta_int3_bin_K8.bits |
| 743 | `0.26` | ambiguous | bonn.vits.delta_int2_K8.covtae_rel_pct_ci_lo, bonn.vits.delta_int2_K8.covtae_rel_pct_fw_ci_lo, bonn.vits.delta_int3_int2_K8.covtae_rel_pct_ci_hi (+5) |
| 743 | `5.33` | ambiguous | mem.vits.lam16_kv_b3.cache_ratio_vs_fp32 |
| 743 | `0.31` | ambiguous | bonn.vits.delta_int3_int2_K8.covtae_rel_pct_ci_lo, bonn.vits.delta_int3_int2_K8.covtae_rel_pct_fw_ci_lo, hard.sintel.vits.int4_tensor.absrel |
| 743 | `1.76` | ambiguous | bonn.vits.int2_g16.covtae |
| 743 | `0.44` | ambiguous | hard.sintel.vits.e8_g64_b2.absrel, hard.sintel.vits.int3_tensor.absrel, sintel.vits.delta_int3_int2_K8.covtae_rel_pct_ci_hi (+2) |
| 744 | `1.34` | ambiguous | bonn.vits.delta_int2_K8.covtae_rel_pct_ci_hi, bonn.vits.delta_int2_K8.covtae_rel_pct_fw_ci_hi |
| 744 | `0.25` | ambiguous | bonn.vits.delta_int2_rht_K8.tae_rel_pct_fw, bonn.vits.delta_int3_int2_K8.tae_rel_pct, sintel.vits.delta_int3_int2_K8.covtae_rel_pct_fw_ci_hi |
| 744 | `5.33` | ambiguous | mem.vits.lam16_kv_b3.cache_ratio_vs_fp32 |
| 744 | `0.40` | ambiguous | sintel.vits.lam16_b3.covtae_rel_pct_fw_ci_lo |
| 745 | `1.00` | ambiguous | bonn.vits.1sc.n_scenes, mem.vits.fp16_kv.cache_ratio_vs_fp32, probe.bonn.bits_saving_min |
| 745 | `5.33` | ambiguous | mem.vits.lam16_kv_b3.cache_ratio_vs_fp32 |
| 745 | `0.73` | ambiguous | sintel.vitb.int2_g16.tae_rel_pct_ci_lo, sintel.vits.int2_g16.tae_rel_pct_fw |
| 747 | `5.33` | ambiguous | mem.vits.lam16_kv_b3.cache_ratio_vs_fp32 |
| 747 | `1.21` | ambiguous | sintel.vitl.int2_g16.tae_rel_pct_fw_ci_lo |
| 758 | `4.0` | ambiguous | bonn.vits.delta_int3_rht_K8.bits, bonn.vits.lam16_b4.bits, hard.sintel.vits.int3_g16.tae_rel_pct_ci_hi (+2) |
| 759 | `5.33` | ambiguous | mem.vits.lam16_kv_b3.cache_ratio_vs_fp32 |
| 772 | `14.5` | ambiguous | bonn.vits.delta_lam16_b2_K8.covtae_rel_pct_ci_hi, bonn.vits.delta_lam16_b2_K8.covtae_rel_pct_fw_ci_hi, hard.sintel.vits.int4_tensor.covtae (+3) |
| 775 | `7.4` | ambiguous | bonn.vits.lam16_b2.tae_rel_pct |
| 792 | `14.5` | ambiguous | bonn.vits.delta_lam16_b2_K8.covtae_rel_pct_ci_hi, bonn.vits.delta_lam16_b2_K8.covtae_rel_pct_fw_ci_hi, hard.sintel.vits.int4_tensor.covtae (+3) |
| 792 | `1.0` | ambiguous | bonn.vits.1sc.fp32.delta1, bonn.vits.1sc.n_scenes, bonn.vits.1sc.zero.delta1 (+38) |
| 792 | `1.00` | ambiguous | bonn.vits.1sc.n_scenes, mem.vits.fp16_kv.cache_ratio_vs_fp32, probe.bonn.bits_saving_min |
| 793 | `0.8` | ambiguous | bonn.vits.1sc.fp32.covis_fraction, bonn.vits.1sc.zero.covis_fraction, bonn.vits.bin_g16.covis_fraction (+54) |
| 793 | `1.34` | ambiguous | bonn.vits.delta_int2_K8.covtae_rel_pct_ci_hi, bonn.vits.delta_int2_K8.covtae_rel_pct_fw_ci_hi |
| 794 | `0.8` | ambiguous | bonn.vits.1sc.fp32.covis_fraction, bonn.vits.1sc.zero.covis_fraction, bonn.vits.bin_g16.covis_fraction (+54) |
| 795 | `2.2` | ambiguous | bonn.vits.1sc.zero.covtae, bonn.vits.delta_int3_bin_K16.bits, bonn.vits.delta_int4_bin_K8.bits (+2) |
| 805 | `4.5` | ambiguous | bonn.vits.int2_g16.covtae_rel_pct_ci_hi, bonn.vits.int2_g16.covtae_rel_pct_fw_ci_hi, bonn.vits.int4_g16.bits (+2) |
| 806 | `0.004` | ambiguous | hard.sintel.vits.e8_g16_b3.delta1_agg_diff, hard.sintel.vits.e8_g64_b3.delta1_agg_diff, mem.vits.lam16_b4.depth_rel_l1_mean (+8) |
| 807 | `0.001` | ambiguous | bonn.vits.bin_g16.delta1_diff_ci_lo, bonn.vits.bin_g16.delta1_diff_fw_ci_lo, bonn.vits.delta_lam16_b2_K8.absrel_diff (+59) |
| 807 | `3.5` | ambiguous | bonn.vits.int3_g16.bits, bonn.vits.int3_g16_rht.bits, hard.sintel.vits.e8_g16_b3.bits (+7) |
| 808 | `12.5` | ambiguous | bonn.vits.delta_lam16_b2_K8.covtae_rel_pct_fw |
| 808 | `4.0` | ambiguous | bonn.vits.delta_int3_rht_K8.bits, bonn.vits.lam16_b4.bits, hard.sintel.vits.int3_g16.tae_rel_pct_ci_hi (+2) |
| 812 | `3.0` | ambiguous | bonn.vits.delta_int3_int2_K8.bits, bonn.vits.delta_lam16_b2_K8.tae, bonn.vits.int3_gT.bits (+7) |
| 813 | `1.21` | ambiguous | sintel.vitl.int2_g16.tae_rel_pct_fw_ci_lo |

RESULT: FAIL