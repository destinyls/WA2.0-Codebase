#!/usr/bin/env python3
"""Render a unit-labelled RGB/latent/GT Franka last-action demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import matplotlib
import numpy as np
import numpy.typing as npt

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


COLORS = {
    "current": "#222222",
    "gt": "#22a06b",
    "rgb": "#e67e22",
    "latent": "#246bfd",
    "context": "#c7cbd1",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--frames-per-sample", type=int, default=8)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _position_error_cm(
    prediction: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
) -> float:
    return float(np.linalg.norm(prediction[:3] - target[:3]) * 100.0)


def _rotation_error_deg(
    prediction: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
) -> float:
    cosine = float(
        np.clip(abs(np.dot(prediction[3:7], target[3:7])), 0.0, 1.0)
    )
    return float(np.degrees(2.0 * np.arccos(cosine)))


def _axis_limits(
    arrays: tuple[npt.NDArray[np.float32], ...],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    xyz_cm = np.concatenate([array[:, :3] for array in arrays], axis=0) * 100.0
    limits: list[tuple[float, float]] = []
    for axis in range(3):
        low = float(np.min(xyz_cm[:, axis]))
        high = float(np.max(xyz_cm[:, axis]))
        pad = max(0.5, (high - low) * 0.15)
        limits.append((low - pad, high + pad))
    return limits[0], limits[1], limits[2]


def _draw_projection(
    axis: Any,
    *,
    horizontal: int,
    vertical: int,
    horizontal_label: str,
    vertical_label: str,
    current: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
    raw: npt.NDArray[np.float32],
    latent: npt.NDArray[np.float32],
    all_targets: npt.NDArray[np.float32],
    limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> None:
    target_cm = target[:3] * 100.0
    raw_cm = raw[:3] * 100.0
    latent_cm = latent[:3] * 100.0
    current_cm = current[:3] * 100.0
    context_cm = all_targets[:, :3] * 100.0
    axis.scatter(
        context_cm[:, horizontal],
        context_cm[:, vertical],
        s=10,
        color=COLORS["context"],
        alpha=0.55,
        label="other GT samples",
    )
    axis.plot(
        [target_cm[horizontal], raw_cm[horizontal]],
        [target_cm[vertical], raw_cm[vertical]],
        color=COLORS["rgb"],
        linewidth=1.5,
    )
    axis.plot(
        [target_cm[horizontal], latent_cm[horizontal]],
        [target_cm[vertical], latent_cm[vertical]],
        color=COLORS["latent"],
        linewidth=1.5,
    )
    axis.scatter(
        current_cm[horizontal],
        current_cm[vertical],
        marker="*",
        s=90,
        color=COLORS["current"],
        label="current",
        zorder=4,
    )
    axis.scatter(
        target_cm[horizontal],
        target_cm[vertical],
        marker="o",
        s=65,
        color=COLORS["gt"],
        label="GT last action",
        zorder=5,
    )
    axis.scatter(
        raw_cm[horizontal],
        raw_cm[vertical],
        marker="X",
        s=75,
        color=COLORS["rgb"],
        label="RGB Policy",
        zorder=5,
    )
    axis.scatter(
        latent_cm[horizontal],
        latent_cm[vertical],
        marker="^",
        s=70,
        color=COLORS["latent"],
        label="cached latent",
        zorder=5,
    )
    axis.set_xlim(*limits[horizontal])
    axis.set_ylim(*limits[vertical])
    axis.set_xlabel(f"{horizontal_label} (cm)")
    axis.set_ylabel(f"{vertical_label} (cm)")
    axis.grid(alpha=0.25)
    axis.set_aspect("equal", adjustable="box")


def _render_frame(
    *,
    index: int,
    cam_high: npt.NDArray[np.uint8],
    cam_wrist: npt.NDArray[np.uint8],
    current: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
    raw: npt.NDArray[np.float32],
    latent: npt.NDArray[np.float32],
    all_targets: npt.NDArray[np.float32],
    episode_id: int,
    history_frames: int,
    limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> npt.NDArray[np.uint8]:
    figure = plt.figure(figsize=(12.8, 7.2), dpi=100, constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=(1.2, 1.0, 1.0))
    high_axis = figure.add_subplot(grid[0, 0])
    wrist_axis = figure.add_subplot(grid[1, 0])
    xy_axis = figure.add_subplot(grid[0, 1])
    xz_axis = figure.add_subplot(grid[0, 2])
    yz_axis = figure.add_subplot(grid[1, 1])
    text_axis = figure.add_subplot(grid[1, 2])

    high_axis.imshow(cam_high)
    high_axis.set_title("Top RGB — causal current frame")
    high_axis.axis("off")
    wrist_axis.imshow(cam_wrist)
    wrist_axis.set_title("Wrist RGB — causal current frame")
    wrist_axis.axis("off")

    common = {
        "current": current,
        "target": target,
        "raw": raw,
        "latent": latent,
        "all_targets": all_targets,
        "limits": limits,
    }
    _draw_projection(
        xy_axis,
        horizontal=0,
        vertical=1,
        horizontal_label="X",
        vertical_label="Y",
        **common,
    )
    xy_axis.set_title("Robot-base XY")
    _draw_projection(
        xz_axis,
        horizontal=0,
        vertical=2,
        horizontal_label="X",
        vertical_label="Z",
        **common,
    )
    xz_axis.set_title("Robot-base XZ")
    _draw_projection(
        yz_axis,
        horizontal=1,
        vertical=2,
        horizontal_label="Y",
        vertical_label="Z",
        **common,
    )
    yz_axis.set_title("Robot-base YZ")

    rgb_position = _position_error_cm(raw, target)
    latent_position = _position_error_cm(latent, target)
    rgb_rotation = _rotation_error_deg(raw, target)
    latent_rotation = _rotation_error_deg(latent, target)
    text_axis.axis("off")
    text_axis.text(
        0.0,
        1.0,
        "Wipe Policy — selected last future action\n"
        "future offset: +5  |  global action index: 11\n\n"
        f"sample: {index + 1}/{len(all_targets)}\n"
        f"episode: {episode_id}\n"
        f"causal RGB history: {history_frames} frames @ 10 Hz\n\n"
        f"RGB → GT position: {rgb_position:.3f} cm\n"
        f"RGB → GT rotation: {rgb_rotation:.3f}°\n\n"
        f"Latent → GT position: {latent_position:.3f} cm\n"
        f"Latent → GT rotation: {latent_rotation:.3f}°",
        va="top",
        fontsize=12,
        linespacing=1.4,
    )
    handles, labels = xy_axis.get_legend_handles_labels()
    xy_axis.legend(
        handles[-4:],
        labels[-4:],
        loc="upper left",
        frameon=False,
        fontsize=7,
    )
    figure.suptitle(
        "Training-corpus diagnostic — absolute XYZW pose, position in centimetres",
        fontsize=15,
        fontweight="bold",
    )
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba(), dtype=np.uint8)
    rgb = np.ascontiguousarray(rgba[:, :, :3])
    plt.close(figure)
    return rgb


def main() -> int:
    args = _parser().parse_args()
    samples_path = args.samples.expanduser().resolve(strict=True)
    metrics_path = args.metrics.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    receipt = args.receipt.expanduser().resolve(strict=False)
    if output.exists() or receipt.exists():
        raise FileExistsError("demo output and receipt must be new paths")
    if args.fps <= 0 or args.frames_per_sample <= 0:
        raise ValueError("fps and frames-per-sample must be positive")

    arrays = np.load(samples_path, allow_pickle=False)
    raw = np.asarray(arrays["raw_actions"], dtype=np.float32)
    latent = np.asarray(arrays["latent_actions"], dtype=np.float32)
    target = np.asarray(arrays["target_actions"], dtype=np.float32)
    current = np.asarray(arrays["current_actions"], dtype=np.float32)
    cam_high = np.asarray(arrays["cam_high"], dtype=np.uint8)
    cam_wrist = np.asarray(arrays["cam_wrist"], dtype=np.uint8)
    episode_ids = np.asarray(arrays["episode_ids"], dtype=np.int64)
    history_counts = np.asarray(arrays["history_frame_counts"], dtype=np.int64)
    if not (
        raw.shape == latent.shape == target.shape == current.shape
        and raw.ndim == 2
        and raw.shape[1] == 8
        and cam_high.shape[0] == raw.shape[0]
        and cam_wrist.shape[0] == raw.shape[0]
    ):
        raise ValueError("inconsistent visualization sample arrays")

    output.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(
        str(output),
        mode="w",
        format="mp4",
        options={"movflags": "+faststart"},
    )
    stream = container.add_stream("libx264", rate=Fraction(str(args.fps)))
    stream.width = 1280
    stream.height = 720
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "medium"}
    limits = _axis_limits((current, target, raw, latent))
    try:
        for index in range(raw.shape[0]):
            frame = _render_frame(
                index=index,
                cam_high=cam_high[index],
                cam_wrist=cam_wrist[index],
                current=current[index],
                target=target[index],
                raw=raw[index],
                latent=latent[index],
                all_targets=target,
                episode_id=int(episode_ids[index]),
                history_frames=int(history_counts[index]),
                limits=limits,
            )
            for _ in range(args.frames_per_sample):
                video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("demo MP4 was not created")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload = {
        "status": "complete",
        "artifact_type": "franka_last_future_action_visualization",
        "evaluation_type": metrics.get("evaluation_type"),
        "action_selection": str(arrays["action_selection"]),
        "selected_future_offset": int(arrays["selected_future_offset"]),
        "sample_count": int(raw.shape[0]),
        "fps": args.fps,
        "frames_per_sample": args.frames_per_sample,
        "duration_seconds": raw.shape[0] * args.frames_per_sample / args.fps,
        "resolution": [1280, 720],
        "video_codec": "H.264/AVC (libx264)",
        "pixel_format": "yuv420p",
        "faststart": True,
        "position_frame": "robot_base_absolute",
        "position_display_unit": "cm",
        "quaternion_order": "XYZW",
        "samples_sha256": _sha256(samples_path),
        "metrics_sha256": _sha256(metrics_path),
        "video_sha256": _sha256(output),
        "video_path": str(output),
    }
    raw_receipt = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw_receipt)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
