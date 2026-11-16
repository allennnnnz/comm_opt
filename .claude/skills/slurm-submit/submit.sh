#!/usr/bin/env bash
# Checked submit for Nano5. Usage:
#   submit.sh [--not-measurement] [--dry-run] <job.sbatch> [extra sbatch args...]
# Measurement jobs (the default) must request --exclusive.
set -euo pipefail

ACCOUNT=MST114546
CONF=$HOME/.config/comm_opt/slurm.env
LOG=$HOME/.config/comm_opt/jobs.tsv
MAX_RUNNING=2

measurement=1; dry=0
while [[ $# -gt 0 && $1 == --* ]]; do
  case $1 in
    --not-measurement) measurement=0 ;;
    --dry-run) dry=1 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
script=${1:?usage: submit.sh [--not-measurement] [--dry-run] <job.sbatch> [sbatch args]}; shift
[[ -f $script ]] || { echo "no such file: $script" >&2; exit 2; }

fail() { echo "NOT SUBMITTED: $*" >&2; exit 1; }
directive() { grep -E "^#SBATCH[[:space:]]+--$1(=|[[:space:]]|$)" "$script" | head -1 || true; }

grep -q '<[^>]*>' <(grep -v '^#[^S]' "$script") && fail "unfilled <...> placeholder in $script"
[[ $(directive account) == *"=$ACCOUNT"* ]] || fail "need #SBATCH --account=$ACCOUNT"
[[ -n $(directive partition) ]] || fail "need #SBATCH --partition"
[[ -n $(directive time) ]] || fail "need #SBATCH --time"
out_line=$(directive output); [[ -n $out_line ]] || fail "need #SBATCH --output"
if (( measurement )); then
  [[ -n $(directive exclusive) ]] || fail "measurement job must use #SBATCH --exclusive (or pass --not-measurement)"
fi

# outputs must live under /home (compute nodes cannot see the login node's /tmp)
for d in output error; do
  p=$(directive $d | sed -E 's/^#SBATCH[[:space:]]+--[a-z]+[= ]//')
  [[ -z $p ]] && continue
  [[ $p == /home/* ]] || fail "--$d must be an absolute path under /home: $p"
  (( dry )) || mkdir -p "$(dirname "$p")"
done

[[ -r $CONF ]] || fail "missing $CONF (needs SLURM_MAIL_USER=...)"
# shellcheck disable=SC1090
source "$CONF"
[[ -n ${SLURM_MAIL_USER:-} ]] || fail "SLURM_MAIL_USER not set in $CONF"

bal=$(timeout 60 wallet "$ACCOUNT" 2>/dev/null | sed -nE 's/.*SU_BALANCE:[[:space:]]*([0-9.]+).*/\1/p')
[[ -n $bal ]] || fail "could not read wallet balance for $ACCOUNT"
awk -v b="$bal" 'BEGIN{exit !(b>0)}' || fail "wallet balance is $bal"

running=$(squeue -h -u "$USER" -t RUNNING | wc -l)
(( running < MAX_RUNNING )) || fail "already $running running jobs (limit $MAX_RUNNING)"

cmd=(sbatch --parsable --mail-type=END,FAIL --mail-user="$SLURM_MAIL_USER" "$@" "$script")
echo "wallet=$bal running=$running measurement=$measurement"
if (( dry )); then echo "DRY RUN: ${cmd[*]/--mail-user=*/--mail-user=<from $CONF>}"; exit 0; fi

jobid=$("${cmd[@]}")
commit=$(git -C "$(dirname "$script")" rev-parse --short HEAD 2>/dev/null || echo none)
[[ -f $LOG ]] || printf 'submitted\tjob_id\tscript\tcommit\tmeasurement\n' > "$LOG"
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$jobid" "$(realpath "$script")" "$commit" "$measurement" >> "$LOG"
echo "submitted job $jobid"
