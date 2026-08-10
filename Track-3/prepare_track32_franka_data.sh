#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 /absolute/work-root /absolute/n0-twam-base [device-count]" >&2
  exit 2
fi

work_root="$1"
base_model="$2"
device_count="${3:-8}"
endpoint="${N0_TRACK32_HF_ENDPOINT:-https://hf-mirror.com}"
download_workers="${N0_TRACK32_DOWNLOAD_WORKERS:-16}"

if [[ "$work_root" != /* || "$base_model" != /* ]]; then
  echo "work-root and base-model must be absolute paths" >&2
  exit 2
fi
if ! [[ "$device_count" =~ ^[1-9][0-9]*$ ]]; then
  echo "device-count must be a positive integer" >&2
  exit 2
fi

inventory="$work_root/provenance/official_franka_inventory.json"
download_receipt="$work_root/provenance/download_receipt.json"
raw_root="$work_root/data/official/WorldArena2.0_Franka_FR3_aed59b39"
artifact_root="$work_root/artifacts"
lerobot_root="$work_root/data/lerobot"
dataset_root="$lerobot_root/all600"
encoder_identity="$artifact_root/encoder_source_identity.json"
latent_inventory="$artifact_root/franka_video_latent_inventory.json"
log_root="$work_root/logs/latent_encoding"

mkdir -p "$work_root/provenance" "$work_root/logs"

if [[ ! -f "$inventory" ]]; then
  python -m script.track3_2.freeze_franka_inventory \
    --output "$inventory" \
    --endpoint "$endpoint"
fi

if [[ ! -f "$download_receipt" ]]; then
  python -m script.track3_2.download_franka \
    --inventory "$inventory" \
    --output-root "$raw_root" \
    --receipt "$download_receipt" \
    --endpoint "$endpoint" \
    --workers "$download_workers"
fi

if [[ ! -f "$artifact_root/prepare_receipt.json" ]]; then
  python -m script.track3_2.prepare_franka \
    --inventory "$inventory" \
    --data-root "$raw_root" \
    --artifact-root "$artifact_root" \
    --lerobot-root "$lerobot_root"
fi

if [[ ! -f "$encoder_identity" ]]; then
  python -m script.track3_2.cache_encoder_identity \
    --model-path "$base_model" \
    --output "$encoder_identity"
fi

if [[ ! -f "$latent_inventory" ]]; then
  mkdir -p "$log_root"
  pids=()
  for ((shard = 0; shard < device_count; shard++)); do
    CUDA_VISIBLE_DEVICES="$shard" \
      HIP_VISIBLE_DEVICES="$shard" \
      python -m script.track3_2.encode_franka_latents \
        --dataset-root "$dataset_root" \
        --model-path "$base_model" \
        --encoder-source-identity "$encoder_identity" \
        --num-shards "$device_count" \
        --shard-index "$shard" \
        --device cuda:0 \
        >"$log_root/shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "one or more latent shards failed; inspect $log_root" >&2
    exit 1
  fi
  python -m script.track3_2.finalize_franka_latents \
    --artifact-root "$artifact_root" \
    --lerobot-root "$lerobot_root" \
    --base-model "$base_model"
fi

echo "Track 3.2 Franka data preparation complete: $work_root"
