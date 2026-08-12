#!/bin/bash
#SBATCH --job-name=hr5_checkpoint
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2/Derived_Sink_Hosts/canonical_v1/logs/hr5_checkpoint_%j.out
#SBATCH --error=/scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2/Derived_Sink_Hosts/canonical_v1/logs/hr5_checkpoint_%j.err

set -euo pipefail

cd /home/kjhan/BACKUP/FDM_SINK_MERGE/fdm_sink_merge
date '+checkpoint_time=%F %T %Z'
squeue -j 432582,433623,433648 -o '%.18i %.24j %.8T %.10M %R' || true
sacct -j 432582,433623,433648 \
  --format=JobID,JobName%24,State,Elapsed,Timelimit,ExitCode -n -X || true
PYTHONPATH=src python scripts/report_hr5_capture_host_status.py \
  --output results/hr5/hr5_capture_host_checkpoint_status.json
