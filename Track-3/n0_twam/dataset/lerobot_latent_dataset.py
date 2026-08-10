# Copyright 2025-2026 NeoteAI Team. All rights reserved.
import json
import logging
import os
from collections.abc import Callable
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import get_episode_data_index
from lerobot.datasets.video_utils import decode_video_frames
from torch.utils.data import DataLoader
from tqdm import tqdm

from n0_twam.dataset.episode_indexing import (
    build_episode_data_index_positions,
    resolve_episode_data_index_position,
)
from n0_twam.dataset.sample_shape_signature import (
    LatentTensorSpec,
    SampleShapeSignature,
    build_sample_shape_signature,
    episode_selection_sha256,
)
from n0_twam.evaluation.tactile_prediction_schema import strict_int64_vector
from n0_twam.evaluation.tactile_sampling import (
    select_latent_crop_start as _select_latent_crop_start,
)
from n0_twam.evaluation.tactile_temporal_contract import (
    decoded_evaluation_frame_count,
)
from n0_twam.integrations.univtac.dataset_view import (
    DatasetView,
    select_content_addressed_crop_start,
)
from n0_twam.tactile_profiles import (
    resolve_repo_tactile_keys,
    validate_profile_segment_inventory,
    validate_tactile_profile_config,
)


def recursive_find_file(directory, filename="info.json"):
    result = []
    ignored_dirs = {"data", "videos", "latents", ".cache", "__pycache__"}
    try:
        for root, dirs, files in os.walk(directory, followlinks=True):
            dirs[:] = [d for d in dirs if d not in ignored_dirs]
            if filename in files:
                full_path = os.path.join(root, filename)
                result.append(full_path)
    except PermissionError:
        print(f"Error: can not access {directory}")
    except Exception as e:
        print(f"Error: {e}")
    return result


def construct_lerobot(
    repo_id,
    config,
):
    # Tolerate broken/in-flight repos (conversion interrupted mid-episode,
    # parquet missing, meta inconsistent): skip with a warning instead of
    # killing the whole multi-dataset init. Essential for train-while-convert.
    try:
        return LatentLeRobotDataset(
            repo_id=repo_id,
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        logging.warning("Skipping unusable repo %s: %s", repo_id, exc)
        return None


def construct_lerobot_multi_processor(
    config,
    num_init_worker=8,
):
    datasets_out_lst = []
    construct_func = partial(
        construct_lerobot,
        config=config,
    )
    repo_list = recursive_find_file(config.dataset_path, "info.json")
    repo_list = [v.split("/meta/info.json")[0] for v in repo_list]
    repo_list = sorted(repo_list)
    if not repo_list:
        return []
    num_init_worker = max(1, min(int(num_init_worker), len(repo_list)))
    if num_init_worker <= 1 or len(repo_list) <= 1:
        datasets_out_lst = [construct_func(repo_id) for repo_id in repo_list]
    else:
        with Pool(num_init_worker) as pool:
            datasets_out_lst = pool.map(construct_func, repo_list)
    skipped = sum(1 for d in datasets_out_lst if d is None)
    if skipped:
        if getattr(config, "tactile_profile", None) is not None:
            raise RuntimeError(
                f"profile-bound dataset initialization rejected {skipped} unusable repos"
            )
        logging.warning("construct_lerobot: skipped %d unusable repos", skipped)
    return [d for d in datasets_out_lst if d is not None]


class MultiLatentLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        num_init_worker=128,
    ):
        num_init_worker = int(getattr(config, "num_init_worker", num_init_worker))
        self._datasets = construct_lerobot_multi_processor(
            config,
            num_init_worker,
        )
        validate_tactile_profile_config(
            config,
            repo_names=(dataset.repo_name for dataset in self._datasets),
        )
        if not self._datasets:
            raise ValueError("dataset contains no valid repositories")
        self.item_id_to_dataset_id, self.acc_dset_num = (
            self._get_item_id_to_dataset_id()
        )

    @property
    def sample_signatures(self) -> tuple[SampleShapeSignature, ...]:
        return tuple(
            signature
            for dataset in self._datasets
            for signature in dataset.sample_signatures
        )

    def __len__(
        self,
    ):
        return sum(len(v) for v in self._datasets)

    def _get_item_id_to_dataset_id(self):
        item_id_to_dataset_id = {}
        acc_dset_num = {}
        acc_nums = [0]
        id = 0
        for dset_id, dset in enumerate(self._datasets):
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_id_to_dataset_id[id] = dset_id
                id += 1
        for did in range(len(self._datasets)):
            acc_dset_num[did] = acc_nums[did]
        return item_id_to_dataset_id, acc_dset_num

    def __getitem__(self, idx) -> dict:
        assert idx < len(self)
        cur_dset = self._datasets[self.item_id_to_dataset_id[idx]]
        local_idx = idx - self.acc_dset_num[self.item_id_to_dataset_id[idx]]
        return cur_dset[local_idx]


class LatentLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id,
        config=None,
    ):
        self.repo_id = repo_id
        self.root = HF_LEROBOT_HOME / repo_id
        self.image_transforms = None
        self.delta_timestamps = None
        selected_episode_ids = getattr(config, "selected_lerobot_episode_ids", None)
        if selected_episode_ids is None:
            self.episodes = None
        else:
            selected = tuple(int(value) for value in selected_episode_ids)
            if (
                not selected
                or len(selected) != len(set(selected))
                or any(value < 0 for value in selected)
            ):
                raise ValueError(
                    "selected_lerobot_episode_ids must be unique non-negative IDs"
                )
            self.episodes = list(selected)
        # ``get_episode_data_index`` compacts an explicit episode selection into
        # a dense ``from``/``to`` table. Latent filenames and immutable
        # DatasetView metadata retain the original LeRobot episode IDs, which
        # can be non-contiguous in a frozen evaluation cohort.
        self._episode_data_index_positions = build_episode_data_index_positions(
            self.episodes
        )
        self.tolerance_s = 1e-4
        self.revision = "v2.1"
        self.video_backend = "pyav"
        self.delta_indices = None
        self.batch_encoding_size = 1
        self.episodes_since_last_encoding = 0
        self.image_writer = None
        self.episode_buffer = None
        self.root.mkdir(exist_ok=True, parents=True)
        try:
            self.meta = LeRobotDatasetMetadata(
                self.repo_id, self.root, self.revision, force_cache_sync=False
            )
        except Exception:
            # LeRobot v3.0 (+ lerobot>=0.3) rejects a full-path repo_id ("must be
            # 'repo_name'") and tries to resolve a pinned revision via the hub.
            # Retry with the basename and no revision -> uses the local snapshot.
            # v2.1 repos succeed on the first call above, so they are untouched.
            self.meta = LeRobotDatasetMetadata(
                Path(self.repo_id).name, self.root, None, force_cache_sync=False
            )

        self.repo_name = Path(self.repo_id).name
        self.config = config
        self.crop_epoch = 0
        self.crop_seed = int(getattr(config, "seed", 42))
        self.crop_window_policy_id = str(
            getattr(
                config,
                "crop_window_policy_id",
                "checkpointed_rng_stream_crop_v1",
            )
        )
        active_view = getattr(config, "active_dataset_view", None)
        if active_view is not None and not isinstance(active_view, DatasetView):
            raise ValueError("active_dataset_view must be a DatasetView")
        self.active_dataset_view = active_view
        self._view_entries_by_episode = (
            {}
            if active_view is None
            else {entry.lerobot_episode_id: entry for entry in active_view.entries}
        )
        per_repo_obs_cam_keys = getattr(config, "per_repo_obs_cam_keys", None) or {}
        self.used_video_keys = list(
            per_repo_obs_cam_keys.get(self.repo_name, config.obs_cam_keys)
        )
        # Mixed-robot pretraining: tactile sensor sets differ per repo (4/2/0
        # streams). per_repo_tactile_keys overrides the global list; an empty
        # list means this repo has no tactile at all.
        self.used_tactile_keys = list(resolve_repo_tactile_keys(config, self.repo_name))
        self.has_tactile_condition = bool(self.used_tactile_keys)
        # When True, episodes whose tactile latents are missing fall back to the
        # CFG tactile-drop path (model's zero-anchor keeps grads sane) instead of
        # raising — required when tactile-less repos mix into pretraining.
        self.tactile_optional = bool(getattr(config, "tactile_optional", False))
        self.synthetic_tactile_data = bool(
            getattr(config, "synthetic_tactile_data", False)
        )
        self.tactile_channels = int(getattr(config, "tactile_in_channels", 3))
        self.raw_tactile_evaluation = bool(
            getattr(config, "raw_tactile_evaluation", False)
        )
        self.deterministic_evaluation_crop_zero = bool(
            getattr(config, "deterministic_evaluation_crop_zero", False)
        )
        if self.deterministic_evaluation_crop_zero and not self.raw_tactile_evaluation:
            raise ValueError(
                "deterministic crop0 is restricted to raw tactile evaluation"
            )

        try:
            assert all(
                (self.root / fpath).is_file()
                for fpath in self.get_episodes_file_paths()
            )
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError) as exc:
            raise FileNotFoundError(
                f"Incomplete local LeRobot dataset under {self.root}"
            ) from exc
        self.episode_data_index = get_episode_data_index(
            self.meta.episodes, self.episodes
        )

        self.latent_path = Path(repo_id) / "latents"
        self.empty_emb = torch.load(config.empty_emb_path, weights_only=False).detach()
        self.cfg_prob = config.cfg_prob
        per_repo_used_action_channel_ids = (
            getattr(config, "per_repo_used_action_channel_ids", None) or {}
        )
        self.used_action_channel_ids = list(
            per_repo_used_action_channel_ids.get(
                self.repo_name,
                getattr(config, "used_action_channel_ids", []),
            )
        )
        if self.used_action_channel_ids:
            action_dim = int(getattr(config, "action_dim", 30))
            inverse_ids = [len(self.used_action_channel_ids)] * action_dim
            for i, j in enumerate(self.used_action_channel_ids):
                inverse_ids[j] = i
            self.inverse_used_action_channel_ids = inverse_ids
        else:
            self.inverse_used_action_channel_ids = list(
                config.inverse_used_action_channel_ids
            )
        # per-robot (embodiment) norm: per_repo_norm_stat maps repo basename
        # ("arx5") or its robot base ("ur" for "ur_3cam") to {q01,q99}; falls
        # back to the global norm_stat when absent.
        #
        # robot_base must find the robot token ANYWHERE in repo_name, not just the
        # first token: some repos are named "<task>_<robot>" (e.g.
        # "<task>_<robot>_<n>cam" variants) where split("_")[0] is
        # the TASK ("stack"/"scoop") and would miss the per-robot stat -> silent
        # global fallback (wrong scale, esp. grippers).
        # So we match any known-robot key (from per_repo_norm) appearing as a token.
        _norm_stat = config.norm_stat
        _per_repo_norm = getattr(config, "per_repo_norm_stat", None) or {}
        if _per_repo_norm:
            _tokens = set(self.repo_name.split("_"))
            _known_robots = sorted(k for k in _per_repo_norm if k)  # drop '' key
            _robot_base = next(
                (k for k in _known_robots if k in _tokens), self.repo_name.split("_")[0]
            )
            _norm_stat = _per_repo_norm.get(
                self.repo_name, _per_repo_norm.get(_robot_base, _norm_stat)
            )
        self.q01 = np.array(_norm_stat["q01"], dtype="float")[None]
        self.q99 = np.array(_norm_stat["q99"], dtype="float")[None]
        self._hf_torch_view = self.hf_dataset.with_format(
            type="torch", columns=["action"], output_all_columns=False
        )
        self._hf_tactile_view = None
        available_columns = set(getattr(self.hf_dataset, "column_names", []))
        self._hf_tactile_source_view = None
        if self.raw_tactile_evaluation:
            required_source_columns = {
                "source.relative_path",
                "source.frame_index",
            }
            missing_source_columns = sorted(required_source_columns - available_columns)
            if missing_source_columns:
                raise ValueError(
                    "raw tactile evaluation requires converted source metadata: "
                    + ", ".join(missing_source_columns)
                )
            self._hf_tactile_source_view = self.hf_dataset.with_format(
                type=None,
                columns=sorted(required_source_columns),
                output_all_columns=False,
            )
        tactile_columns = [
            key for key in self.used_tactile_keys if key in available_columns
        ]
        if self.has_tactile_condition and tactile_columns:
            self._hf_tactile_view = self.hf_dataset.with_format(
                type="torch",
                columns=tactile_columns,
                output_all_columns=False,
            )
        if (
            self.has_tactile_condition
            and self._hf_tactile_view is None
            and self.synthetic_tactile_data
        ):
            logging.warning(
                "Using synthetic tactile videos for training. "
                "Fake tactile streams are enabled because tactile columns were not found in dataset %s.",
                self.repo_id,
            )
        self.filter_mismatched_latents = bool(
            getattr(config, "filter_mismatched_latents", True)
        )
        self._latent_frame_count_cache = {}
        self._shape_payload_cache = {}
        self._sample_signatures = None
        self._meta_filter_counts = {}
        self.parse_meta()

    def _episode_chunk_candidates(self, episode_index: int) -> list[int]:
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        candidates = [episode_chunk]
        if episode_chunk != 0:
            candidates.append(0)
        return candidates

    def get_episodes_file_paths(self) -> list[Path]:
        episodes = (
            self.episodes
            if self.episodes is not None
            else list(range(self.meta.total_episodes))
        )
        fpaths = []
        for ep_idx in episodes:
            data_path = self.meta.get_data_file_path(ep_idx)
            full_path = self.root / data_path
            if not full_path.is_file():
                fallback = self.root / "data" / "chunk-000" / Path(data_path).name
                if fallback.is_file():
                    data_path = fallback.relative_to(self.root)
            fpaths.append(str(data_path))
        return fpaths

    def _resolve_latent_file(
        self, episode_index: int, start_frame: int, end_frame: int, key: str
    ) -> Path:
        filename = f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
        for chunk_index in self._episode_chunk_candidates(episode_index):
            latent_file = self.latent_path / f"chunk-{chunk_index:03d}" / key / filename
            if latent_file.exists():
                return latent_file
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        return self.latent_path / f"chunk-{episode_chunk:03d}" / key / filename

    def _resolve_tactile_latent_file(
        self,
        episode_index: int,
        start_frame: int,
        end_frame: int,
        key: str,
        mode: str,
        raise_on_ambiguous: bool = True,
    ) -> Path | None:
        tactile_root_name = getattr(
            self.config, "tactile_latent_root_name", "latents_tactile"
        )
        tactile_root = self.root / tactile_root_name
        if not tactile_root.exists():
            return None
        filename = f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
        glob_name = f"episode_{episode_index:06d}_*.pth"
        chunks_size = int(self.meta.info.get("chunks_size", 1000))
        chunk_candidates = [episode_index // chunks_size]
        for chunk_index in self._episode_chunk_candidates(episode_index):
            if chunk_index not in chunk_candidates:
                chunk_candidates.append(chunk_index)

        for chunk_index in chunk_candidates:
            directory = tactile_root / mode / f"chunk-{chunk_index:03d}" / key
            if not directory.exists():
                continue
            exact = directory / filename
            if exact.exists():
                return exact
            candidates = list(directory.glob(glob_name))
            if not candidates:
                continue
            if len(candidates) > 1:
                if raise_on_ambiguous:
                    raise FileNotFoundError(
                        f"Tactile latent segment {filename} not found in "
                        f"{directory}, and the episode glob matches "
                        f"{len(candidates)} files — cannot pick one safely. "
                        "Re-encode tactile latents per segment "
                        "(script/encode_tactile_latent.py) so names match the "
                        "video latents."
                    )
                return None
            return candidates[0]
        return None

    def _latent_frame_count(self, latent_file: Path) -> int:
        latent_file = Path(latent_file)
        cached = self._latent_frame_count_cache.get(latent_file)
        if cached is not None:
            return cached
        # mmap=True reads the tensor lazily, so reading only the small
        # 'latent_num_frames' metadata field doesn't pull the whole latent off
        # shared storage — ~4x faster for the validation pass. Fall back if unsupported.
        try:
            payload = torch.load(
                latent_file, map_location="cpu", weights_only=False, mmap=True
            )
        except Exception:
            payload = torch.load(latent_file, map_location="cpu", weights_only=False)
        count = int(payload["latent_num_frames"])
        self._latent_frame_count_cache[latent_file] = count
        return count

    def _load_shape_payload(self, latent_file: Path) -> dict[str, object]:
        """Load only mmap-backed tensor metadata used by the batch signature."""

        latent_file = Path(latent_file)
        cached = self._shape_payload_cache.get(latent_file)
        if cached is not None:
            return cached
        try:
            payload = torch.load(
                latent_file, map_location="cpu", weights_only=False, mmap=True
            )
        except Exception:
            payload = torch.load(latent_file, map_location="cpu", weights_only=False)
        latent = payload.get("latent")
        if not torch.is_tensor(latent) or latent.ndim != 2:
            raise ValueError(f"latent payload has invalid tensor: {latent_file}")
        frames = int(payload["latent_num_frames"])
        height = int(payload["latent_height"])
        width = int(payload["latent_width"])
        if latent.shape[0] != frames * height * width:
            raise ValueError(
                f"latent payload metadata does not match tensor: {latent_file}"
            )
        result: dict[str, object] = {
            "spec": LatentTensorSpec(
                channels=int(latent.shape[1]),
                frames=frames,
                height=height,
                width=width,
                dtype=str(latent.dtype),
            )
        }
        text_emb = payload.get("text_emb")
        if text_emb is not None:
            if not torch.is_tensor(text_emb):
                raise ValueError(f"text_emb must be a tensor: {latent_file}")
            result["text_shape"] = tuple(int(value) for value in text_emb.shape)
            result["text_dtype"] = str(text_emb.dtype)
        frame_ids = payload.get("frame_ids")
        if frame_ids is not None:
            if torch.is_tensor(frame_ids):
                frame_ids = frame_ids.detach().cpu().reshape(-1).tolist()
            result["frame_ids"] = tuple(int(value) for value in frame_ids)
        self._shape_payload_cache[latent_file] = result
        return result

    def _sample_shape_signature(self, meta: dict) -> SampleShapeSignature:
        episode_index = int(meta["episode_index"])
        start_frame = int(meta["start_frame"])
        end_frame = int(meta["end_frame"])
        video_payloads = [
            self._load_shape_payload(
                self._resolve_latent_file(
                    episode_index,
                    start_frame,
                    end_frame,
                    key,
                )
            )
            for key in self.used_video_keys
        ]
        first_payload = video_payloads[0]
        if any(
            key not in first_payload
            for key in ("text_shape", "text_dtype", "frame_ids")
        ):
            raise ValueError("first video latent lacks text/frame metadata")
        tactile_global_specs = []
        tactile_local_specs = []
        for key in self.used_tactile_keys:
            global_file = self._resolve_tactile_latent_file(
                episode_index,
                start_frame,
                end_frame,
                key,
                "global",
            )
            local_file = self._resolve_tactile_latent_file(
                episode_index,
                start_frame,
                end_frame,
                key,
                "local",
            )
            if global_file is None or local_file is None:
                raise FileNotFoundError(
                    f"missing tactile shape metadata for repo={self.repo_name!r} "
                    f"episode={episode_index} key={key!r}"
                )
            tactile_global_specs.append(self._load_shape_payload(global_file)["spec"])
            tactile_local_specs.append(self._load_shape_payload(local_file)["spec"])
        return build_sample_shape_signature(
            video_specs=tuple(payload["spec"] for payload in video_payloads),
            text_shape=first_payload["text_shape"],
            text_dtype=str(first_payload["text_dtype"]),
            frame_ids=first_payload["frame_ids"],
            max_latent_frames=int(getattr(self.config, "max_latent_frames", 0)),
            action_dim=len(self.inverse_used_action_channel_ids),
            action_slots_per_frame=self._sample_action_slots_per_frame(),
            tactile_global_specs=tuple(tactile_global_specs),
            tactile_local_specs=tuple(tactile_local_specs),
            raw_tactile_evaluation=self.raw_tactile_evaluation,
        )

    def _sample_action_slots_per_frame(self) -> int | None:
        """Return an adapter-specific fixed action horizon, if it has one."""

        return None

    @property
    def sample_signatures(self) -> tuple[SampleShapeSignature, ...]:
        """Exact output tensor shapes, computed lazily from latent metadata."""

        if self._sample_signatures is None:
            self._sample_signatures = tuple(
                self._sample_shape_signature(meta) for meta in self.new_metas
            )
        return self._sample_signatures

    def _reject_meta(self, reason: str) -> bool:
        self._meta_filter_counts[reason] = self._meta_filter_counts.get(reason, 0) + 1
        return False

    def _valid_seg_cache_path(self):
        selection_sha256 = self._selected_episode_ids_sha256()
        if selection_sha256 is None:
            return Path(self.root) / ".valid_seg_cache.json"
        return Path(self.root) / f".valid_seg_cache.{selection_sha256}.json"

    def _selected_episode_ids_sha256(self) -> str | None:
        return episode_selection_sha256(self.episodes)

    def _repo_validation_signature(self):
        """Cheap (no torch.load) fingerprint of the repo's data + filter config.
        The valid-segment cache is reused only when this matches, so any latent
        add/remove (re-encode / gapfill), parquet change, camera/tactile-key
        change, or filter-flag flip forces a fresh validation -> never a stale
        cache. Counting latent files is os-level (fast), unlike _check_meta's
        per-segment torch.load."""
        root = Path(self.root)
        vp = root / "latents" / "chunk-000"
        nvid = 0
        if vp.is_dir():
            for cam in vp.iterdir():
                if cam.is_dir():
                    nvid += sum(1 for _ in cam.glob("*.pth"))
        tname = getattr(self.config, "tactile_latent_root_name", "latents_tactile")
        tp = root / tname
        ntac = sum(1 for _ in tp.rglob("*.pth")) if tp.is_dir() else 0
        pp = root / "data" / "chunk-000"
        npq = sum(1 for _ in pp.glob("*.parquet")) if pp.is_dir() else 0
        return {
            "v": 3,
            "tactile_profile": getattr(self.config, "tactile_profile", None),
            "selected_episode_ids_sha256": self._selected_episode_ids_sha256(),
            "parquet": npq,
            "video_latents": nvid,
            "tactile_latents": ntac,
            "video_keys": sorted(self.used_video_keys),
            "tactile_keys": sorted(self.used_tactile_keys),
            "filter": bool(self.filter_mismatched_latents),
        }

    def _load_valid_seg_cache(self):
        """Cached set of valid (episode, start, end) keys iff the on-disk cache
        matches the current data signature; else None (-> full validation)."""
        if not bool(getattr(self.config, "use_valid_seg_cache", True)):
            return None
        p = self._valid_seg_cache_path()
        if not p.is_file():
            return None
        try:
            blob = json.loads(p.read_text())
            if blob.get("signature") != self._repo_validation_signature():
                return None
            return {tuple(int(v) for v in k) for k in blob.get("valid", [])}
        except Exception:
            return None

    def _save_valid_seg_cache(self, valid_set):
        if not bool(getattr(self.config, "use_valid_seg_cache", True)):
            return
        try:
            p = self._valid_seg_cache_path()
            tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
            tmp.write_text(
                json.dumps(
                    {
                        "signature": self._repo_validation_signature(),
                        "valid": [[int(k[0]), int(k[1]), int(k[2])] for k in valid_set],
                    }
                )
            )
            os.replace(tmp, p)  # atomic; concurrent ranks write identical content
        except Exception as exc:
            logging.warning(
                "valid-seg cache write failed for %s: %s", self.repo_id, exc
            )

    def parse_meta(self):
        # One-time per data state: validating every segment (existence + frame
        # consistency of each latent via torch.load) is the slow part of init,
        # and all ranks repeat it. Cache the validated valid-segment set keyed by
        # a data signature so subsequent launches/resumes skip the torch.load
        # pass. Build it once up front.
        cached_valid = self._load_valid_seg_cache()
        out = []
        total = 0
        for key, value in self.meta.episodes.items():
            episode_index = value["episode_index"]
            if self.episodes is not None and episode_index not in self.episodes:
                continue
            tasks = value["tasks"]
            action_config = value["action_config"]
            for acfg in action_config:
                total += 1
                cur_meta = {
                    "episode_index": episode_index,
                    "tasks": tasks,
                }
                cur_meta.update(acfg)

                if cached_valid is not None:
                    check_statu = (
                        int(episode_index),
                        int(cur_meta["start_frame"]),
                        int(cur_meta["end_frame"]),
                    ) in cached_valid
                else:
                    check_statu = self._check_meta(
                        cur_meta["start_frame"],
                        cur_meta["end_frame"],
                        cur_meta["episode_index"],
                    )

                if check_statu:
                    out.append(cur_meta)
        self.new_metas = out
        self._enforce_profile_segment_contract(total=total)
        if cached_valid is None:
            self._save_valid_seg_cache(
                {
                    (
                        int(m["episode_index"]),
                        int(m["start_frame"]),
                        int(m["end_frame"]),
                    )
                    for m in out
                }
            )
            if self._meta_filter_counts:
                logging.warning(
                    "Filtered %d/%d latent segments for repo=%s: %s",
                    total - len(out),
                    total,
                    self.repo_id,
                    self._meta_filter_counts,
                )
        else:
            logging.info(
                "valid-seg cache hit: %d/%d segments for repo=%s",
                len(out),
                total,
                self.repo_id,
            )

    def _enforce_profile_segment_contract(self, *, total: int) -> None:
        """Reject a declared repo when filtering changes its modality roster."""

        profile = getattr(self.config, "tactile_profile", None)
        if profile is None:
            return
        validate_profile_segment_inventory(
            profile=profile,
            repo_name=self.repo_name,
            has_tactile=self.has_tactile_condition,
            valid_segments=len(self.new_metas),
            total_segments=total,
            rejection_counts=self._meta_filter_counts,
        )

    def _check_meta(self, start_frame, end_frame, episode_index):
        expected_frames = None
        for key in self.used_video_keys:
            latent_file = self._resolve_latent_file(
                episode_index, start_frame, end_frame, key
            )
            if not os.path.exists(latent_file):
                return self._reject_meta("missing_video_latent")
            if self.filter_mismatched_latents:
                try:
                    frame_count = self._latent_frame_count(latent_file)
                except Exception as exc:
                    logging.warning(
                        "Failed to read video latent metadata %s: %s", latent_file, exc
                    )
                    return self._reject_meta("bad_video_latent")
                if expected_frames is None:
                    expected_frames = frame_count
                elif frame_count != expected_frames:
                    return self._reject_meta("video_frame_mismatch")

        if (
            self.filter_mismatched_latents
            and self.has_tactile_condition
            and self.used_tactile_keys
            and not self.synthetic_tactile_data
        ):
            for key in self.used_tactile_keys:
                global_file = self._resolve_tactile_latent_file(
                    episode_index,
                    start_frame,
                    end_frame,
                    key,
                    "global",
                    raise_on_ambiguous=False,
                )
                local_file = self._resolve_tactile_latent_file(
                    episode_index,
                    start_frame,
                    end_frame,
                    key,
                    "local",
                    raise_on_ambiguous=False,
                )
                if global_file is None or local_file is None:
                    return self._reject_meta("missing_tactile_latent")
                try:
                    global_frames = self._latent_frame_count(global_file)
                    local_frames = self._latent_frame_count(local_file)
                except Exception as exc:
                    logging.warning(
                        "Failed to read tactile latent metadata repo=%s episode=%s key=%s: %s",
                        self.repo_id,
                        episode_index,
                        key,
                        exc,
                    )
                    return self._reject_meta("bad_tactile_latent")
                if global_frames != local_frames:
                    return self._reject_meta("tactile_local_global_mismatch")
                if expected_frames is not None and global_frames != expected_frames:
                    return self._reject_meta("tactile_video_frame_mismatch")
        return True

    def _get_global_idx(self, episode_index: int, local_index: int):
        index_position = self._episode_data_index_position(episode_index)
        ep_start = self.episode_data_index["from"][index_position]
        return local_index + ep_start

    def _episode_data_index_position(self, episode_index: int) -> int:
        """Resolve an original LeRobot ID to a compact data-index position."""
        return resolve_episode_data_index_position(
            self._episode_data_index_positions,
            episode_index,
        )

    def _get_range_hf_data(self, start_frame, end_frame):
        batch = self._hf_torch_view[start_frame:end_frame]
        return batch

    def _get_tactile_evaluation_source_metadata(
        self,
        *,
        episode_index: int,
        source_row_ids,
    ) -> dict[str, object]:
        """Read raw-row IDs and simulator steps from converted LeRobot metadata."""

        if self._hf_tactile_source_view is None:
            raise RuntimeError("raw tactile evaluation metadata view is disabled")
        index_position = self._episode_data_index_position(episode_index)
        episode_start = int(self.episode_data_index["from"][index_position])
        episode_end = int(self.episode_data_index["to"][index_position])
        episode_length = episode_end - episode_start
        expected_length = decoded_evaluation_frame_count(
            int(getattr(self.config, "max_latent_frames", 0))
        )
        row_ids = strict_int64_vector(
            np.asarray(source_row_ids).reshape(-1),
            label="source row IDs",
            expected_length=expected_length,
        )
        if int(row_ids[-1]) >= episode_length:
            raise ValueError("source row ID exceeds the converted LeRobot episode")
        global_indices = (row_ids + episode_start).tolist()
        metadata = self._hf_tactile_source_view[global_indices]
        raw_paths = metadata.get("source.relative_path")
        raw_steps = metadata.get("source.frame_index")
        if not isinstance(raw_paths, (list, tuple, np.ndarray)) or len(
            raw_paths
        ) != len(global_indices):
            raise ValueError("converted LeRobot source paths are incomplete")

        def _scalar(value, *, label: str):
            array = np.asarray(value)
            if array.size != 1:
                raise ValueError(f"converted LeRobot {label} must be scalar")
            return array.reshape(-1)[0]

        paths = [str(_scalar(value, label="source path")) for value in raw_paths]
        if len(set(paths)) != 1 or not paths[0]:
            raise ValueError("converted LeRobot rows do not share one source path")
        if not isinstance(raw_steps, (list, tuple, np.ndarray)) or len(
            raw_steps
        ) != len(global_indices):
            raise ValueError("converted LeRobot simulator steps are incomplete")
        step_values = []
        for value in raw_steps:
            scalar = _scalar(value, label="simulator step")
            step_values.append(scalar)
        step_ids = strict_int64_vector(
            np.asarray(step_values),
            label="converted LeRobot simulator steps",
            expected_length=len(global_indices),
        )
        return {
            "source_relative_path": paths[0],
            "source_row_ids": torch.from_numpy(row_ids.copy()),
            "source_step_ids": torch.from_numpy(step_ids),
            "lerobot_episode_index": torch.tensor(episode_index, dtype=torch.int64),
        }

    def _flatten_latent_dict(self, latent_dict):
        out = {}
        for key, value in latent_dict.items():
            for inner_key, inner_value in value.items():
                new_key = f"{key}.{inner_key}"
                out[new_key] = inner_value
        return out

    def _get_range_latent_data(self, start_frame, end_frame, episode_index):
        out = {}
        for key in self.used_video_keys:
            latent_file = self._resolve_latent_file(
                episode_index, start_frame, end_frame, key
            )
            assert os.path.exists(latent_file)
            latent_data = torch.load(latent_file, weights_only=False)
            out[key] = latent_data

        return self._flatten_latent_dict(out)

    def _cat_video_latents(self, data_dict):
        latent_lst = []
        for key in self.used_video_keys:
            latent = data_dict[f"{key}.latent"]
            latent_num_frames = data_dict[f"{key}.latent_num_frames"]
            latent_height = data_dict[f"{key}.latent_height"]
            latent_width = data_dict[f"{key}.latent_width"]
            latent = rearrange(
                latent,
                "(f h w) c -> f h w c",
                f=latent_num_frames,
                h=latent_height,
                w=latent_width,
            )
            latent_lst.append(latent)
        cat_latent = torch.cat(latent_lst, dim=2)

        text_emb = data_dict[f"{self.used_video_keys[0]}.text_emb"].detach()
        if torch.rand(1).item() < self.cfg_prob:
            text_emb = self.empty_emb

        out_dict = dict(
            latents=cat_latent,
            text_emb=text_emb,
        )
        return out_dict

    def _action_post_process(
        self, local_start_frame, local_end_frame, latent_frame_ids, action
    ):
        act_shift = int(latent_frame_ids[0] - local_start_frame)
        frame_stride = latent_frame_ids[1] - latent_frame_ids[0]
        action = action[act_shift:]
        action = np.pad(action, pad_width=((frame_stride * 4, 0), (0, 0)), mode="edge")

        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        required_action_num = latent_frame_num * frame_stride * 4

        action = action[:required_action_num]
        action_mask = np.ones_like(action, dtype="bool")
        assert action.shape[0] == required_action_num

        action_paded = np.pad(
            action, ((0, 0), (0, 1)), mode="constant", constant_values=0
        )
        action_mask_padded = np.pad(
            action_mask, ((0, 0), (0, 1)), mode="constant", constant_values=0
        )

        action_aligned = action_paded[:, self.inverse_used_action_channel_ids]
        action_mask_aligned = action_mask_padded[
            :, self.inverse_used_action_channel_ids
        ]
        action_aligned = (action_aligned - self.q01) / (
            self.q99 - self.q01 + 1e-6
        ) * 2.0 - 1.0
        action_aligned = rearrange(
            action_aligned, "(f n) c -> c f n 1", f=latent_frame_num
        )
        action_mask_aligned = rearrange(
            action_mask_aligned, "(f n) c -> c f n 1", f=latent_frame_num
        )
        action_aligned *= action_mask_aligned
        return (
            torch.from_numpy(action_aligned).float(),
            torch.from_numpy(action_mask_aligned).bool(),
        )

    def _load_tactile_latents(
        self,
        episode_index: int,
        local_start_frame: int,
        local_end_frame: int,
        latent_frame_ids,
        truncate_start: int | None = None,
        truncate_end: int | None = None,
        expected_full_frames: int | None = None,
    ) -> dict | None:
        """Load pre-computed GlobalTactile + LocalTactile latents for an episode.

        Expects files produced by script/encode_tactile_latent.py at:
            <dataset_root>/latents_tactile/{global,local}/chunk-XXX/<tactile_key>/episode_*_*.pth

        Returns dict {
            'global':      torch.Tensor (S, C=48, F_lat_truncated, H_lat, W_lat),
            'local':       torch.Tensor (S, C=48, F_lat_truncated, H_lat, W_lat),
            'sensor_ids':  torch.LongTensor (S,)  — sensor index per slot,
        } or None if any sensor's latent is missing.

        truncate_start/truncate_end: indices into F_lat to slice (matches the
        truncation applied to video latents to keep temporal alignment).
        expected_full_frames: the VIDEO latent's full (pre-truncation) frame
        count. Tactile is frame-aligned with video in the attention mask, so a
        mismatched tactile F would silently misalign every tactile frame —
        assert instead of training on wrong semantics.
        """
        sensor_id_map = getattr(self.config, "tactile_sensor_id_map", None)
        tactile_root_name = getattr(
            self.config, "tactile_latent_root_name", "latents_tactile"
        )
        tactile_root = self.root / tactile_root_name
        if not tactile_root.exists():
            return None

        globals_list = []
        locals_list = []
        sensor_ids_list = []
        for key in self.used_tactile_keys:
            global_file = self._resolve_tactile_latent_file(
                episode_index, local_start_frame, local_end_frame, key, "global"
            )
            local_file = self._resolve_tactile_latent_file(
                episode_index, local_start_frame, local_end_frame, key, "local"
            )
            if global_file is None or local_file is None:
                # one sensor missing — skip whole tactile for this batch
                return None

            g_payload = torch.load(global_file, map_location="cpu", weights_only=False)
            l_payload = torch.load(local_file, map_location="cpu", weights_only=False)
            g_flat = g_payload["latent"]  # (F*H*W, C)
            l_flat = l_payload["latent"]
            F_lat = int(g_payload["latent_num_frames"])
            H_lat = int(g_payload["latent_height"])
            W_lat = int(g_payload["latent_width"])
            F_lat_local = int(l_payload.get("latent_num_frames", F_lat))
            if F_lat_local != F_lat:
                raise ValueError(
                    f"Tactile local/global frame mismatch for {key} "
                    f"episode {episode_index}: local={F_lat_local}, "
                    f"global={F_lat}."
                )
            # Tactile is frame-aligned with video tokens (same frame_id in the
            # attention mask), so the FULL tactile F must equal the video's
            # full latent F — otherwise the shared truncate window slices a
            # shifted/short range and every tactile frame silently pairs with
            # the wrong video frame.
            if expected_full_frames is not None and F_lat != expected_full_frames:
                raise ValueError(
                    f"Tactile/video latent frame mismatch for {key} episode "
                    f"{episode_index}: tactile F={F_lat}, video F="
                    f"{expected_full_frames}. Re-encode tactile latents with "
                    "the same --target-fps/frame policy as the video latents."
                )
            # Reshape (F*H*W, C) → (F, H, W, C) → (C, F, H, W)
            g_5d = (
                g_flat.reshape(F_lat, H_lat, W_lat, -1).permute(3, 0, 1, 2).contiguous()
            )
            l_5d = (
                l_flat.reshape(F_lat, H_lat, W_lat, -1).permute(3, 0, 1, 2).contiguous()
            )

            # Slice along F to match the truncated video latent window
            if truncate_start is not None and truncate_end is not None:
                g_5d = g_5d[:, truncate_start:truncate_end]
                l_5d = l_5d[:, truncate_start:truncate_end]

            globals_list.append(g_5d)
            locals_list.append(l_5d)

            # Resolve sensor_id from cfg map, default to enumeration order
            if sensor_id_map and key in sensor_id_map:
                sensor_ids_list.append(int(sensor_id_map[key]))
            else:
                sensor_ids_list.append(len(sensor_ids_list))

        return {
            "global": torch.stack(globals_list, dim=0).contiguous(),  # (S, C, F, H, W)
            "local": torch.stack(locals_list, dim=0).contiguous(),
            "sensor_ids": torch.tensor(sensor_ids_list, dtype=torch.long),  # (S,)
        }

    def __getitem__(self, idx) -> dict:
        idx = idx % len(self.new_metas)
        cur_meta = self.new_metas[idx]
        episode_index = cur_meta["episode_index"]
        start_frame = cur_meta["start_frame"]
        end_frame = cur_meta["end_frame"]
        local_start_frame = start_frame
        local_end_frame = end_frame

        ori_data_dict = self._get_range_latent_data(
            start_frame, end_frame, episode_index
        )

        latent_frame_ids = ori_data_dict[f"{self.used_video_keys[0]}.frame_ids"]
        num_latent_frames = ori_data_dict[
            f"{self.used_video_keys[0]}.latent_num_frames"
        ]

        # Truncate long episodes to avoid CUDA OOM
        start_lat = None  # init so tactile loader can reference outside the if
        end_lat = None
        max_latent_frames = int(getattr(self.config, "max_latent_frames", 0))
        if max_latent_frames > 0 and num_latent_frames > max_latent_frames:
            if (
                self.crop_window_policy_id == "epoch_content_addressed_crop_v1"
                and not self.deterministic_evaluation_crop_zero
            ):
                view_entry = self._view_entries_by_episode.get(int(episode_index))
                if view_entry is None:
                    raise ValueError(
                        "content-addressed crop has no active view episode entry"
                    )
                start_lat = select_content_addressed_crop_start(
                    view_entry,
                    seed=self.crop_seed,
                    epoch=self.crop_epoch,
                    num_latent_frames=int(num_latent_frames),
                    max_latent_frames=max_latent_frames,
                )
            else:
                start_lat = _select_latent_crop_start(
                    num_latent_frames=int(num_latent_frames),
                    max_latent_frames=max_latent_frames,
                    deterministic_evaluation_crop_zero=(
                        self.deterministic_evaluation_crop_zero
                    ),
                )
            end_lat = start_lat + max_latent_frames
            # Truncate latent tensors for each video key
            for key in self.used_video_keys:
                F = ori_data_dict[f"{key}.latent_num_frames"]
                H = ori_data_dict[f"{key}.latent_height"]
                W = ori_data_dict[f"{key}.latent_width"]
                latent = ori_data_dict[f"{key}.latent"].reshape(F, H * W, -1)
                ori_data_dict[f"{key}.latent"] = latent[start_lat:end_lat].reshape(
                    -1, latent.shape[-1]
                )
                ori_data_dict[f"{key}.latent_num_frames"] = max_latent_frames
            # Truncate frame_ids: action code uses (len(frame_ids)-1)//4+1 as latent_frame_num
            # so we need exactly (max_latent_frames-1)*4+1 video frame entries
            needed_vid_frames = (max_latent_frames - 1) * 4 + 1
            vid_start = start_lat * 4
            vid_start = min(
                vid_start, max(0, len(latent_frame_ids) - needed_vid_frames)
            )
            vid_end = vid_start + needed_vid_frames
            latent_frame_ids = latent_frame_ids[vid_start:vid_end]

        start_frame = self._get_global_idx(episode_index, start_frame)
        end_frame = self._get_global_idx(episode_index, end_frame)

        hf_data_frames = self._get_range_hf_data(start_frame, end_frame)
        ori_data_dict.update(hf_data_frames)
        out_dict = self._cat_video_latents(ori_data_dict)
        if self.raw_tactile_evaluation:
            out_dict.update(
                self._get_tactile_evaluation_source_metadata(
                    episode_index=int(episode_index),
                    source_row_ids=latent_frame_ids,
                )
            )
        # ─── NEW: load pre-computed GlobalTactile / LocalTactile latents ───
        # (replaces the old RGB-video → CNN tactile pipeline)
        if self.has_tactile_condition and self.used_tactile_keys:
            tactile_payload = self._load_tactile_latents(
                episode_index=episode_index,
                local_start_frame=local_start_frame,
                local_end_frame=local_end_frame,
                latent_frame_ids=latent_frame_ids,
                truncate_start=start_lat,
                truncate_end=end_lat,
                # video's FULL latent frame count (num_latent_frames is read
                # before truncation) — tactile must match it frame-for-frame.
                expected_full_frames=int(num_latent_frames),
            )
            if tactile_payload is None and not self.tactile_optional:
                raise FileNotFoundError(
                    "Missing tactile latents for cond branch: "
                    f"repo={self.repo_id} episode={episode_index} "
                    f"frames={local_start_frame}:{local_end_frame} "
                    f"keys={self.used_tactile_keys}"
                )

            if tactile_payload is None:
                # tactile_optional: missing tactile falls back to the CFG-drop
                # path (model's zero-anchor keeps gradients FSDP-safe).
                out_dict["tactile_cond_drop"] = torch.tensor(True, dtype=torch.bool)
            else:
                # Keep the real tensors in the sample so missing files are still
                # caught above. The model uses this explicit flag to skip tactile
                # tokens entirely for CFG drop, so tactile modules get no gradient.
                tactile_cfg_prob = float(getattr(self.config, "tactile_cfg_prob", 0.1))
                tactile_cond_drop = torch.rand(1).item() < tactile_cfg_prob

                out_dict["tactile_global_latent"] = tactile_payload["global"]
                out_dict["tactile_local_latent"] = tactile_payload["local"]
                out_dict["tactile_sensor_ids"] = tactile_payload["sensor_ids"]
                out_dict["tactile_cond_drop"] = torch.tensor(
                    tactile_cond_drop, dtype=torch.bool
                )
        else:
            # Repo with no tactile sensors at all (per_repo_tactile_keys = []):
            # explicit drop flag so the model takes the zero-anchor path.
            out_dict["tactile_cond_drop"] = torch.tensor(True, dtype=torch.bool)

        out_dict["actions"], out_dict["actions_mask"] = self._action_post_process(
            local_start_frame,
            local_end_frame,
            latent_frame_ids,
            ori_data_dict["action"],
        )

        out_dict["latents"] = out_dict["latents"].permute(3, 0, 1, 2)
        return out_dict

    def __len__(self):
        return len(self.new_metas)

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic content-addressed crop epoch."""
        resolved = int(epoch)
        if resolved < 0:
            raise ValueError("dataset crop epoch must be non-negative")
        self.crop_epoch = resolved


if __name__ == "__main__":
    from tqdm import tqdm

    from n0_twam.configs import TWAM_CONFIGS

    dset = MultiLatentLeRobotDataset(TWAM_CONFIGS["base"])
    for key, value in dset[0].items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape} tensor")
        elif isinstance(value, np.ndarray):
            print(f"{key}: {value.shape} np")
        else:
            print(f"{key}: {value}")
    print(len(dset))
    dloader = DataLoader(
        dset,
        batch_size=1,
        shuffle=True,
        num_workers=32,
    )
    max_l = 0
    action_list = []
    for data in tqdm(dloader):
        _, _, F, H, W = data["latents"].shape
        max_l = max(max_l, F * H * W)
        action_list.append(data["actions"].flatten(2).permute(0, 2, 1).flatten(0, 1))
    action_all = torch.cat(action_list, dim=0)
    print(max_l)
    print(
        action_all.shape,
        action_all.mean(dim=0),
        action_all.min(dim=0)[0],
        action_all.max(dim=0)[0],
    )
