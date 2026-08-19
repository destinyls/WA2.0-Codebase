# Copyright 2025-2026 NeoteAI Team. All rights reserved.
from .bucket_sampler import BucketedDistributedBatchSampler


class MultiLatentLeRobotDataset:
    def __new__(cls, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        dataset_adapter = str(getattr(config, "dataset_adapter", ""))
        if dataset_adapter == "worldarena_franka_ee10":
            from .lerobot_latent_dataset_franka import (
                MultiLatentLeRobotFrankaDataset,
            )

            return MultiLatentLeRobotFrankaDataset(*args, **kwargs)
        if dataset_adapter == "worldarena_agilex_qpos14":
            from .lerobot_latent_dataset_agilex import (
                MultiLatentLeRobotAgileXDataset,
            )

            return MultiLatentLeRobotAgileXDataset(*args, **kwargs)
        if dataset_adapter:
            raise ValueError(f"unsupported dataset_adapter: {dataset_adapter!r}")
        action_schema = str(getattr(config, "action_schema", ""))
        if action_schema == "qpos8_next_step":
            from .lerobot_latent_dataset_qpos8 import MultiLatentLeRobotQpos8Dataset

            return MultiLatentLeRobotQpos8Dataset(*args, **kwargs)
        action_delta_mode = str(getattr(config, "action_delta_mode", "")).lower()
        if action_delta_mode in {"pi05_delta", "openpi_delta", "pi0.5_delta"}:
            from .lerobot_latent_dataset_pi05_delta import (
                MultiLatentLeRobotPi05DeltaDataset,
            )

            return MultiLatentLeRobotPi05DeltaDataset(*args, **kwargs)
        from .lerobot_latent_dataset import (
            MultiLatentLeRobotDataset as DefaultMultiLatentLeRobotDataset,
        )

        return DefaultMultiLatentLeRobotDataset(*args, **kwargs)


def __getattr__(name):
    """Preserve the historical concrete dataset exports without eager imports."""

    if name == "MultiLatentLeRobotPi05DeltaDataset":
        from .lerobot_latent_dataset_pi05_delta import (
            MultiLatentLeRobotPi05DeltaDataset,
        )

        return MultiLatentLeRobotPi05DeltaDataset
    if name == "MultiLatentLeRobotQpos8Dataset":
        from .lerobot_latent_dataset_qpos8 import MultiLatentLeRobotQpos8Dataset

        return MultiLatentLeRobotQpos8Dataset
    if name == "MultiLatentLeRobotFrankaDataset":
        from .lerobot_latent_dataset_franka import (
            MultiLatentLeRobotFrankaDataset,
        )

        return MultiLatentLeRobotFrankaDataset
    if name == "MultiLatentLeRobotAgileXDataset":
        from .lerobot_latent_dataset_agilex import (
            MultiLatentLeRobotAgileXDataset,
        )

        return MultiLatentLeRobotAgileXDataset
    raise AttributeError(name)


__all__ = [
    "MultiLatentLeRobotDataset",
    "MultiLatentLeRobotPi05DeltaDataset",
    "MultiLatentLeRobotQpos8Dataset",
    "MultiLatentLeRobotFrankaDataset",
    "MultiLatentLeRobotAgileXDataset",
    "BucketedDistributedBatchSampler",
]
