#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model_dir="${script_dir}/models"
result_dir="${script_dir}/results"

nuxmv_bin="${1:-}"
if [[ -z "${nuxmv_bin}" ]]; then
    if command -v nuXmv >/dev/null 2>&1; then
        nuxmv_bin="$(command -v nuXmv)"
    elif command -v nuxmv >/dev/null 2>&1; then
        nuxmv_bin="$(command -v nuxmv)"
    else
        echo "Usage: $0 /path/to/nuXmv" >&2
        echo "nuXmv was not found on PATH." >&2
        exit 2
    fi
fi

if [[ ! -x "${nuxmv_bin}" ]]; then
    echo "nuXmv executable not found or not executable: ${nuxmv_bin}" >&2
    exit 2
fi

mkdir -p -- "${result_dir}"

models=(
    "relocalization_navigation.smv"
    "pickup_can.smv"
    "docking_pose_control.smv"
)

for model in "${models[@]}"; do
    model_path="${model_dir}/${model}"
    output_path="${result_dir}/${model%.smv}.txt"

    if [[ ! -f "${model_path}" ]]; then
        echo "Missing model: ${model_path}" >&2
        exit 2
    fi

    echo "Checking ${model}"

    spec_output="$(${nuxmv_bin} -coi "${model_path}" 2>&1)"
    totality_output="$(${nuxmv_bin} -ctt -is -ils -ii "${model_path}" 2>&1)"

    {
        echo "=== Property checking (-coi) ==="
        printf '%s\n' "${spec_output}"
        echo
        echo "=== Transition totality checking (-ctt) ==="
        printf '%s\n' "${totality_output}"
    } > "${output_path}"

    expected_false="$(
        grep -Ec '^[[:space:]]*(LTL|CTL|INVAR)SPEC NAME EXPECTED_FALSE_' \
            "${model_path}" || true
    )"
    actual_false="$(printf '%s\n' "${spec_output}" | grep -c ' is false' || true)"

    if [[ "${actual_false}" -ne "${expected_false}" ]]; then
        echo "Unexpected FALSE count for ${model}: expected ${expected_false}, got ${actual_false}." >&2
        echo "See ${output_path}" >&2
        exit 1
    fi

    if ! grep -q 'transition relation is total' <<< "${totality_output}"; then
        echo "Transition totality check failed for ${model}. See ${output_path}" >&2
        exit 1
    fi
done

echo "All focused checks passed. Raw outputs: ${result_dir}"

