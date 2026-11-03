# Overnight queue (laptop)

`scripts/run_overnight.sh` runs every remaining laptop experiment one after another.
Each stage is skipped if its result JSON is already complete, so it is safe to
restart at any time (it continues where it stopped; an interrupted stage reruns).

## Check progress (Git Bash)
    tail -20 /c/Users/aniru/vdaquant/outputs/stream/overnight.log

## Is it running? (PowerShell)
    Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'stream_eval|stream_age_probe' } | Select ProcessId

## Restart it (PowerShell) — runs detached, survives closing Claude
    Start-Process 'C:\Program Files\Git\bin\bash.exe' -ArgumentList '-lc','"cd /c/Users/aniru/vdaquant && bash scripts/run_overnight.sh >> outputs/stream/overnight_stdout.log 2>&1"' -WindowStyle Hidden

## Stop it (PowerShell)
    Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_overnight|stream_eval|stream_age_probe' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

## Stages (approximate laptop times)
| # | Stage | Output | ~time |
|---|---|---|---|
| 1 | Sintel fidelity (fp32, zero, int2, delta int3/int4 keys) | sintel_vits_fidelity.json | 15 min |
| 2 | Bonn fidelity (same) | bonn_vits_fidelity.json | 45 min |
| 3 | Sintel fading memory (17 configs) | sintel_vits_fading.json | 50 min |
| 4 | Attention by age, Sintel + Bonn | age_attention_*.json | 10 min |
| 5 | Bonn fading memory | bonn_vits_fading.json | 2.5 h |
| 6 | Sintel delta ablations (K, key bits, residual bits, predictor) | sintel_vits_ablation.json | 30 min |
| 7 | Bonn delta ablations | bonn_vits_ablation.json | 1.3 h |
| 8 | Bonn ViT-B key configs | bonn_vitb_key.json | 1 h |
| 9 | Bonn ViT-L key configs | bonn_vitl_key.json | 2-3 h |
