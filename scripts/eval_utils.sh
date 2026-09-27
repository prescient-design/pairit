#!/usr/bin/env bash
# Shared utilities for evaluation scripts.
# Source this file: source "$(dirname "$0")/eval_utils.sh"

# find_best_checkpoint RUN_DIR
#   Finds the checkpoint with the lowest val_loss across all timestamped
#   subdirectories in the given run directory. This handles preemption
#   correctly: when a run is preempted and resumed, checkpoints from earlier
#   segments (in older timestamp dirs) are still valid candidates.
#
#   Checkpoint filenames are expected to match: epoch=N-step=N-val_loss=X.XXXX.ckpt
#   Falls back to last.ckpt if no val_loss checkpoints are found.
#   Prints the path to stdout; returns 1 if nothing is found.
#   Warnings are printed to stderr.
find_best_checkpoint() {
    local run_dir="${1:?ERROR: find_best_checkpoint requires RUN_DIR as first argument}"

    # Count timestamped subdirectories
    local -a ts_dirs
    mapfile -t ts_dirs < <(find "${run_dir}" -maxdepth 1 -type d -name "20*" 2>/dev/null | sort)

    if [ ${#ts_dirs[@]} -gt 1 ]; then
        echo "WARNING: Found ${#ts_dirs[@]} timestamped dirs in ${run_dir} (likely preemption). Searching all for best checkpoint." >&2
        for d in "${ts_dirs[@]}"; do
            echo "  $(basename "$d"): $(find "$d" -maxdepth 1 -name '*.ckpt' -not -name 'last.ckpt' | wc -l) checkpoints" >&2
        done
    fi

    # Search all timestamped dirs (or run_dir itself if none exist)
    local search_dirs=("${ts_dirs[@]}")
    if [ ${#search_dirs[@]} -eq 0 ]; then
        search_dirs=("${run_dir}")
    fi

    # Find checkpoint with lowest val_loss across all dirs
    local best
    best=$(
        for d in "${search_dirs[@]}"; do
            find "$d" -maxdepth 1 -name "*.ckpt" -not -name "last.ckpt" 2>/dev/null
        done \
        | grep -oP '.*val_loss=[\d.]+\.ckpt' \
        | sort -t= -k4 -n \
        | head -1 \
        || true
    )

    if [ -n "${best}" ]; then
        echo "${best}"
        return 0
    fi

    # Fallback: last.ckpt from the most recent timestamp dir
    local last
    last=$(find "${search_dirs[-1]}" -maxdepth 1 -name "last.ckpt" -type f 2>/dev/null | head -1)
    if [ -n "${last}" ]; then
        echo "WARNING: No val_loss checkpoint found in ${run_dir}, falling back to last.ckpt" >&2
        echo "${last}"
        return 0
    fi

    echo "ERROR: No checkpoints found in ${run_dir}" >&2
    return 1
}

# find_all_checkpoints RUN_DIR
#   Finds all checkpoints (excluding last.ckpt) across all timestamped
#   subdirectories, sorted by step number. Handles preemption correctly.
#   Prints checkpoint paths to stdout (one per line).
find_all_checkpoints() {
    local run_dir="${1:?ERROR: find_all_checkpoints requires RUN_DIR as first argument}"

    # Find timestamped subdirectories
    local -a ts_dirs
    mapfile -t ts_dirs < <(find "${run_dir}" -maxdepth 1 -type d -name "20*" 2>/dev/null | sort)

    if [ ${#ts_dirs[@]} -gt 1 ]; then
        echo "WARNING: Found ${#ts_dirs[@]} timestamped dirs in ${run_dir} (likely preemption). Searching all." >&2
        for d in "${ts_dirs[@]}"; do
            echo "  $(basename "$d"): $(find "$d" -maxdepth 1 -name '*.ckpt' -not -name 'last.ckpt' | wc -l) checkpoints" >&2
        done
    fi

    local search_dirs=("${ts_dirs[@]}")
    if [ ${#search_dirs[@]} -eq 0 ]; then
        search_dirs=("${run_dir}")
    fi

    # Collect and sort by step number
    for d in "${search_dirs[@]}"; do
        find "$d" -maxdepth 1 -name "*.ckpt" -not -name "last.ckpt" 2>/dev/null
    done | sort -t= -k3 -n
}
