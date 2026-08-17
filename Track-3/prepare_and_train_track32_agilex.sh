#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$SCRIPT_ROOT"

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 /absolute/work-root /absolute/base-model /absolute/init-checkpoint [devices]" >&2
  exit 2
fi

readonly WORK_ROOT="$1"
readonly BASE_MODEL="$2"
readonly INIT_CHECKPOINT="$3"
readonly DEVICE_COUNT="${4:-8}"
readonly ENDPOINT="${N0_TRACK32_HF_ENDPOINT:-https://hf-mirror.com}"
readonly DOWNLOAD_WORKERS="${N0_TRACK32_DOWNLOAD_WORKERS:-16}"
readonly HYSMI="/opt/hyhal/bin/hy-smi"

for value in "$WORK_ROOT" "$BASE_MODEL" "$INIT_CHECKPOINT"; do
  [[ "$value" == /* ]] || { echo "all paths must be absolute" >&2; exit 2; }
done
[[ "$DEVICE_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "devices must be positive" >&2; exit 2; }

readonly RAW_ROOT="$WORK_ROOT/data/official/WorldArena2.0_af1ac34"
readonly DATASET_ROOT="$WORK_ROOT/data/lerobot/agilex_mixed_v1"
readonly ARTIFACT_ROOT="$WORK_ROOT/artifacts/agilex_mixed_v1"
readonly INVENTORY="$WORK_ROOT/provenance/official_agilex_inventory.json"
readonly DOWNLOAD_RECEIPT="$WORK_ROOT/provenance/agilex_download_receipt.json"
readonly ENCODER_IDENTITY="$ARTIFACT_ROOT/encoder_source_identity.json"
readonly LINEAGE="${N0_TRACK32_AGILEX_LINEAGE:-v1}"
readonly REQUEST="$WORK_ROOT/requests/agilex_mixed_step1500_${LINEAGE}.json"
readonly RUN_ROOT="$WORK_ROOT/runs/agilex_mixed_step1500_${LINEAGE}"
readonly LOG_ROOT="$WORK_ROOT/logs/agilex_mixed_step1500_${LINEAGE}"
readonly STATUS="$WORK_ROOT/status/agilex_mixed_step1500_${LINEAGE}.state"
readonly RUN_ID="agilex-mixed-final1500-${LINEAGE}"

mkdir -p "$WORK_ROOT/provenance" "$WORK_ROOT/requests" "$WORK_ROOT/logs" \
  "$WORK_ROOT/status" "$WORK_ROOT/runs"

write_status() {
  local value="$1"
  local temporary="${STATUS}.tmp.$$"
  printf '%s time=%s\n' "$value" "$(date -Iseconds)" >"$temporary"
  mv "$temporary" "$STATUS"
}

on_exit() {
  local code=$?
  if [[ "$code" -ne 0 ]]; then
    write_status "FAILED exit_code=$code"
  fi
}
trap on_exit EXIT

wait_for_hcu_idle() {
  local streak=0 snapshot
  write_status "WAITING_HCU_IDLE"
  while true; do
    snapshot=$($HYSMI 2>/dev/null || true)
    if awk '
      /^[[:space:]]*[0-7][[:space:]]/ {
        count += 1; vram = $6; compute = $7
        gsub(/%/, "", vram); gsub(/%/, "", compute)
        if ((vram + 0) != 0 || (compute + 0) != 0) busy += 1
      }
      END { exit !(count == 8 && busy == 0) }
    ' <<<"$snapshot"; then
      streak=$((streak + 1))
      (( streak < 3 )) || return 0
    else
      streak=0
    fi
    sleep 30
  done
}

if [[ -f /opt/hyhal/env.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/hyhal/env.sh
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

write_status "DOWNLOADING"
if [[ ! -s "$DOWNLOAD_RECEIPT" ]]; then
  python -m script.track3_2.download_agilex \
    --inventory "$INVENTORY" \
    --raw-root "$RAW_ROOT" \
    --receipt "$DOWNLOAD_RECEIPT" \
    --endpoint "$ENDPOINT" \
    --workers "$DOWNLOAD_WORKERS" \
    >"$WORK_ROOT/logs/agilex_download.log" 2>&1
fi

write_status "CONVERTING"
if [[ ! -s "$ARTIFACT_ROOT/prepare_receipt.json" ]]; then
  python -m script.track3_2.prepare_agilex \
    --raw-root "$RAW_ROOT" \
    --artifact-root "$ARTIFACT_ROOT" \
    --dataset-root "$DATASET_ROOT" \
    >"$WORK_ROOT/logs/agilex_prepare.log" 2>&1
fi

if [[ ! -s "$ARTIFACT_ROOT/latent_inventory.json" ]]; then
  wait_for_hcu_idle
  write_status "ENCODING_LATENTS"
  if [[ ! -s "$ENCODER_IDENTITY" ]]; then
    python -m script.track3_2.cache_encoder_identity \
      --model-path "$BASE_MODEL" \
      --output "$ENCODER_IDENTITY"
  fi
  mkdir -p "$LOG_ROOT/latents"
  pids=()
  for ((shard = 0; shard < DEVICE_COUNT; shard++)); do
    CUDA_VISIBLE_DEVICES="$shard" HIP_VISIBLE_DEVICES="$shard" \
      python -m script.track3_2.encode_agilex_latents \
        --dataset-root "$DATASET_ROOT" \
        --model-path "$BASE_MODEL" \
        --encoder-source-identity "$ENCODER_IDENTITY" \
        --num-shards "$DEVICE_COUNT" \
        --shard-index "$shard" \
        --device cuda:0 \
        >"$LOG_ROOT/latents/shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  [[ "$failed" -eq 0 ]] || { write_status "FAILED_LATENTS"; exit 1; }
  python -m script.track3_2.finalize_agilex \
    --artifact-root "$ARTIFACT_ROOT" \
    --dataset-root "$DATASET_ROOT" \
    >"$LOG_ROOT/finalize_artifacts.log" 2>&1
fi

if [[ ! -s "$REQUEST" ]]; then
  python -m script.track3_2.build_agilex_train_request \
    --raw-root "$RAW_ROOT" \
    --dataset-root "$DATASET_ROOT" \
    --artifact-root "$ARTIFACT_ROOT" \
    --base-model "$BASE_MODEL" \
    --empty-embedding "$BASE_MODEL/empty_emb.pt" \
    --init-from "$INIT_CHECKPOINT" \
    --output-root "$RUN_ROOT" \
    --output "$REQUEST" \
    --run-id "$RUN_ID" \
    --devices "$DEVICE_COUNT"
fi

wait_for_hcu_idle
write_status "PREFLIGHT"
python -m n0_twam.cli track32 agilex-train --config "$REQUEST" --dry-run \
  >"$LOG_ROOT/preflight.log" 2>&1
write_status "TRAINING"
python -m n0_twam.cli track32 agilex-train --config "$REQUEST" \
  >"$LOG_ROOT/training.log" 2>&1
write_status "COMPLETE"
