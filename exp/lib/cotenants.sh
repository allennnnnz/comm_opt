# Co-tenancy record for measurement jobs (D054). Source this from a job script AFTER $OUT exists.
# Nano5 strips --exclusive ("Exclusive mode is not allowed"), so a measurement job owns all 8 GPUs and
# the per-GPU CPU maximum, but small CPU-only jobs could in principle share the node. This records, at
# start and at exit, every Slurm job on our nodes, so data from a shared node can be flagged later.
# Output: $OUT/cotenants_<tag>.txt (raw squeue) and one summary line per tag in $OUT/cotenants_summary.txt.
cotenants_record() {  # cotenants_record <tag>
  local tag=$1 raw="$OUT/cotenants_$1.txt" n
  squeue -h -w "${SLURM_JOB_NODELIST:-$(hostname -s)}" -o "%i|%u|%P|%T|%C|%m|%b|%N" > "$raw" 2>&1 || true
  n=$(awk -F'|' -v me="${SLURM_JOB_ID:-none}" 'NF>1 && $1!=me' "$raw" | wc -l)
  echo "tag=$tag time=$(date -Is) other_jobs_on_our_nodes=$n" >> "$OUT/cotenants_summary.txt"
}
cotenants_record start
trap 'cotenants_record end' EXIT
