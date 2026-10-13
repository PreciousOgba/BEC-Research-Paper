#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "Usage: bash run.sh /path/to/provisional_pool.sqlite /path/to/new_results"
  exit 2
fi
script_dir="$(cd "$(dirname "$0")" && pwd)"
python "$script_dir/scripts/run_primary_lr.py" --pool "$1" --output "$2/primary_experiment"
python "$script_dir/scripts/run_distilbert_cpu.py" --experiment "$2/primary_experiment" --output "$2/distilbert_experiment"
