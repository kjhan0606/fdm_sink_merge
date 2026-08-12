#!/bin/bash
#SBATCH --job-name=hr5_watchdog
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2/Derived_Sink_Hosts/canonical_v1/logs/hr5_watchdog_%j.out
#SBATCH --error=/scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2/Derived_Sink_Hosts/canonical_v1/logs/hr5_watchdog_%j.err

set -euo pipefail

REPOSITORY=/home/kjhan/BACKUP/FDM_SINK_MERGE/fdm_sink_merge
cd "${REPOSITORY}"

job_state() {
  sacct -n -X -j "$1" --format=State | awk 'NF {sub(/\+.*/, "", $1); print $1; exit}'
}

source_job=432582
resume_job=433648
validation_job=433623
source_state=$(job_state "${source_job}")
resume_state=$(job_state "${resume_job}")
validation_state=$(job_state "${validation_job}")

date '+watchdog_time=%F %T %Z'
printf 'source_job=%s source_state=%s\n' "${source_job}" "${source_state:-UNKNOWN}"
printf 'resume_job=%s resume_state=%s\n' "${resume_job}" "${resume_state:-UNKNOWN}"
printf 'validation_job=%s validation_state=%s\n' \
  "${validation_job}" "${validation_state:-UNKNOWN}"

case "${source_state}" in
  COMPLETED)
    case "${validation_state}" in
      PENDING|RUNNING|COMPLETING|COMPLETED) ;;
      FAILED|OUT_OF_MEMORY|NODE_FAIL|TIMEOUT)
        echo "Validation ran but failed; preserving its logs for inspection."
        ;;
      *)
        sbatch jobs/hr5_validate.sbatch
        ;;
    esac
    ;;
  TIMEOUT|FAILED|OUT_OF_MEMORY|NODE_FAIL|CANCELLED)
    case "${resume_state}" in
      PENDING|RUNNING|COMPLETING|COMPLETED) ;;
      FAILED|OUT_OF_MEMORY|NODE_FAIL|TIMEOUT)
        echo "Resume job ran but failed; preserving its logs for inspection."
        ;;
      *)
        sbatch jobs/hr5_capture_hosts.sbatch
        ;;
    esac
    ;;
  *)
    echo "Source job has not reached a terminal state; no recovery action taken."
    ;;
esac

PYTHONPATH=src python scripts/report_hr5_capture_host_status.py \
  --output results/hr5/hr5_capture_host_watchdog_status.json
