# Copyright 2025-2026 NeoteAI Team. All rights reserved.

import cv2
import numpy as np
import pytest

from n0_twam.integrations.univtac.hdf5_reader import decode_image_payload
from n0_twam.integrations.univtac.schema import (
    LEGACY_UNIVTAC_JPEG_CONTRACT,
    STANDARD_JPEG_RGB_CONTRACT,
)


def _legacy_jpeg_payload(rgb: np.ndarray) -> np.ndarray:
    success, encoded = cv2.imencode(".jpg", rgb)
    assert success
    return encoded


def test_legacy_univtac_jpeg_restores_source_rgb_numeric_order() -> None:
    source_rgb = np.empty((32, 32, 3), dtype=np.uint8)
    source_rgb[..., 0] = 240
    source_rgb[..., 1] = 30
    source_rgb[..., 2] = 10
    payload = _legacy_jpeg_payload(source_rgb)

    legacy_rgb = decode_image_payload(
        payload,
        encoding_contract=LEGACY_UNIVTAC_JPEG_CONTRACT,
    )
    standard_rgb = decode_image_payload(
        payload,
        encoding_contract=STANDARD_JPEG_RGB_CONTRACT,
    )

    assert int(np.argmax(legacy_rgb.mean(axis=(0, 1)))) == 0
    assert int(np.argmax(standard_rgb.mean(axis=(0, 1)))) == 2
    np.testing.assert_allclose(
        legacy_rgb.mean(axis=(0, 1)),
        source_rgb.mean(axis=(0, 1)),
        atol=3.0,
    )


def test_unknown_image_contract_fails_closed() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Unsupported image encoding contract"):
        decode_image_payload(image, encoding_contract="unknown")
