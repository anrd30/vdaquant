#!/usr/bin/env bash
# =============================================================================
# ONE COMMAND: the F28 delta re-run, then the 13 probes, serially.
#
#   cd vdaquant && git pull && nohup bash scripts/run_everything.sh > all.log 2>&1 &
#   tail -f all.log
#
# STAGE A (scripts/run_lattice_rerun.sh -> outputs/rerun_f28)
#   Re-measures everything the decoder range fix invalidated. Both lattice
#   quantizers previously reserved part of their representable range while the
#   scalar baseline used all of it, handicapping E8 by ~0.87 dB and D4 by
#   ~2.02 dB at the 3-bit operating point (ledger F28). Also runs the F27
#   identity control. This stage CORRECTS the record.
#
# STAGE B (scripts/run_probes.sh -> outputs/probes)
#   13 experiments that try to find something new, using two knobs that did not
#   exist before the fix: --group-size (scale group decoupled from the lattice
#   dimension) and --v-bits (asymmetric K/V budget). This stage EXTENDS it.
#
# Order matters: Stage A establishes the corrected baseline every probe in
# Stage B is read against, and Stage A's identity control decides whether
# Stage B's cross-scale probe (P11) is interpretable at all.
#
# Both stages are independently resumable and write to separate directories.
# outputs/finals is never touched, so the before/after diff stays available.
# If Stage A fails outright, Stage B still runs -- the probes are diagnostic
# either way, and a failed re-run should not cost the whole night.
#
# Total: roughly 10-16 h on one modern GPU. Safe to interrupt and re-run.
# =============================================================================
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
T0=$SECONDS

echo "############################################################"
echo "# STAGE A — F28 delta re-run          $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"
bash scripts/run_lattice_rerun.sh
A_STATUS=$?
echo ""
echo ">>> Stage A finished with status $A_STATUS after $(( (SECONDS-T0)/60 )) min"

echo ""
echo "############################################################"
echo "# STAGE B — probes                    $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"
bash scripts/run_probes.sh
B_STATUS=$?

echo ""
echo "════════════════════════════════════════════════════════════"
echo "ALL DONE in $(( (SECONDS-T0)/60 )) min   stageA=$A_STATUS  stageB=$B_STATUS"
echo ""
echo "Zip both for download:"
echo "  cd $(pwd) && zip -r all_results.zip outputs/rerun_f28 outputs/probes"
echo "════════════════════════════════════════════════════════════"
