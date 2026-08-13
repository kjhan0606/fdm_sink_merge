#!/bin/bash
#SBATCH --job-name=hr5_validate
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2/Derived_Sink_Hosts/canonical_v1/logs/hr5_validate_%j.out
#SBATCH --error=/scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2/Derived_Sink_Hosts/canonical_v1/logs/hr5_validate_%j.err

set -euo pipefail

cd /home/kjhan/BACKUP/FDM_SINK_MERGE/fdm_sink_merge
PYTHONPATH=src python scripts/validate_hr5_fable_outputs.py
PYTHONPATH=src pytest -q
