import json
from types import SimpleNamespace

import pytest

from n0_twam.integrations.worldarena.franka_source import _validate_timestamps
from script.track3_2 import prepare_franka


def _write_timestamps(tmp_path, *, wrist_raw, wrist_map):
    length = 5
    path = tmp_path / "camera_timestamps.json"
    path.write_text(
        json.dumps(
            {
                "reference_camera": "third_person",
                "raw_timestamps": {
                    "third_person": [0.00, 0.066, 0.133, 0.200, 0.266],
                    "wrist": wrist_raw,
                },
                "frame_maps": {
                    "third_person": [0, 1, 2, 3, 4],
                    "wrist": wrist_map,
                },
            }
        ),
        encoding="utf-8",
    )
    return path, length


@pytest.mark.parametrize(
    ("wrist_raw", "wrist_map"),
    (
        ([0.01, 0.076, 0.143, 0.210, 0.276], [0, 1, 2, 3, 4]),
        ([0.01, 0.076, 0.143, 0.210], [0, 1, 2, 3, 3]),
        ([-0.056, 0.01, 0.076, 0.143, 0.210, 0.276], [1, 2, 3, 4, 5]),
    ),
)
def test_timestamp_audit_accepts_official_nearest_frame_patterns(
    tmp_path, wrist_raw, wrist_map
):
    path, length = _write_timestamps(tmp_path, wrist_raw=wrist_raw, wrist_map=wrist_map)

    _validate_timestamps(path, length=length)


@pytest.mark.parametrize(
    ("wrist_raw", "wrist_map"),
    (
        ([0.01, 0.076, 0.143], [0, 1, 2, 2, 2]),
        ([0.01, 0.076, 0.143, 0.210], [0, 1, 3, 2, 3]),
        ([0.01, 0.076, 0.143, 0.210], [0, 1, 2, 3, 4]),
        ([0.01, 0.076, 0.143, 0.210, 0.276], [1, 2, 3, 4, 4]),
        ([0.50, 0.566, 0.633, 0.700, 0.766], [0, 1, 2, 3, 4]),
    ),
)
def test_timestamp_audit_rejects_invalid_wrist_alignment(
    tmp_path, wrist_raw, wrist_map
):
    path, length = _write_timestamps(tmp_path, wrist_raw=wrist_raw, wrist_map=wrist_map)

    with pytest.raises(ValueError):
        _validate_timestamps(path, length=length)


def test_prepare_source_audit_failure_does_not_publish_output_roots(
    tmp_path, monkeypatch
):
    data_root = tmp_path / "raw"
    data_root.mkdir()
    artifact_root = tmp_path / "artifacts"
    lerobot_root = tmp_path / "lerobot"

    def _fail_audit(*_args, **_kwargs):
        raise ValueError("source audit failed")

    monkeypatch.setattr(prepare_franka, "_audit_all", _fail_audit)
    args = SimpleNamespace(
        data_root=data_root,
        artifact_root=artifact_root,
        lerobot_root=lerobot_root,
        inventory=tmp_path / "inventory.json",
    )

    with pytest.raises(ValueError, match="source audit failed"):
        prepare_franka.prepare(args)

    assert not artifact_root.exists()
    assert not lerobot_root.exists()
