#!/bin/bash
#SBATCH --job-name=hr5_capture_hosts
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2/Derived_Sink_Hosts/canonical_v1/logs/hr5_capture_hosts_%j.out
#SBATCH --error=/scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2/Derived_Sink_Hosts/canonical_v1/logs/hr5_capture_hosts_%j.err
#SBATCH --open-mode=append

set -euo pipefail

REPOSITORY=/home/kjhan/BACKUP/FDM_SINK_MERGE/fdm_sink_merge
cd "${REPOSITORY}"

PYTHONPATH=src python scripts/build_hr5_capture_host_dataset.py \
  --extract --jobs 4 --threads 8
PYTHONPATH=src python scripts/analyze_hr5_capture_hosts.py \
  --maximum-events-per-trace 600000
python scripts/plot_hr5_fable_comparison.py
PYTHONPATH=src python scripts/validate_hr5_fable_outputs.py
PYTHONPATH=src pytest -q
