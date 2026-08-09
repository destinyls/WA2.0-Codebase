# Copyright 2025-2026 NeoteAI Team. All rights reserved.
from .bucket_sampler import BucketedDistributedBatchSampler


class MultiLatentLeRobotDataset:
    def __new__(cls, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
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
    raise AttributeError(name)

__all__ = [
    'MultiLatentLeRobotDataset',
    'MultiLatentLeRobotPi05DeltaDataset',
    'MultiLatentLeRobotQpos8Dataset',
    'BucketedDistributedBatchSampler',
]
