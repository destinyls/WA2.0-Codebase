# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Action codec protocol and shared quantile implementation."""

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from .spec import ActionSpec

FloatArray = npt.NDArray[np.float32]


@runtime_checkable
class ActionCodec(Protocol):
    """Transforms one action representation across data and model boundaries."""

    spec: ActionSpec

    def validate(
        self,
        values: npt.ArrayLike,
        *,
        check_bounds: bool = False,
    ) -> FloatArray: ...

    def normalize(self, values: npt.ArrayLike) -> FloatArray: ...

    def denormalize(self, values: npt.ArrayLike) -> FloatArray: ...

    def wire_to_model(
        self,
        values: npt.ArrayLike,
        *,
        frame_count: int | None = None,
        slots_per_frame: int | None = None,
    ) -> FloatArray: ...

    def model_to_wire(self, values: npt.ArrayLike) -> FloatArray: ...


class QuantileActionCodec:
    """Per-channel q01/q99 normalization with strict shape validation."""

    def __init__(
        self,
        spec: ActionSpec,
        *,
        q01: tuple[float, ...],
        q99: tuple[float, ...],
    ) -> None:
        self.spec = spec
        self._q01 = self._as_channel_vector("q01", q01)
        self._q99 = self._as_channel_vector("q99", q99)
        if np.any(self._q99 < self._q01):
            raise ValueError("q99 must be greater than q01 for every channel")
        self._span = self._q99 - self._q01
        self._denominator = self._span + np.float32(1e-6)
        for value in (self._q01, self._q99, self._span, self._denominator):
            value.setflags(write=False)

    def _as_channel_vector(
        self,
        field_name: str,
        values: tuple[float, ...],
    ) -> FloatArray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (self.spec.dim,):
            raise ValueError(
                f"{field_name} must have shape ({self.spec.dim},), "
                f"got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{field_name} contains non-finite values")
        return array.copy()

    def validate(
        self,
        values: npt.ArrayLike,
        *,
        check_bounds: bool = False,
    ) -> FloatArray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim == 0 or array.shape[-1] != self.spec.dim:
            last_dim = None if array.ndim == 0 else array.shape[-1]
            raise ValueError(
                f"action last dimension must be {self.spec.dim}, got {last_dim}"
            )
        if not np.isfinite(array).all():
            raise ValueError("action contains non-finite values")
        if check_bounds and self.spec.lower_bounds is not None:
            lower = np.asarray(self.spec.lower_bounds, dtype=np.float32)
            upper = np.asarray(self.spec.upper_bounds, dtype=np.float32)
            if np.any(array < lower) or np.any(array > upper):
                raise ValueError("action violates physical bounds")
        return array

    def normalize(self, values: npt.ArrayLike) -> FloatArray:
        array = self.validate(values)
        return ((array - self._q01) / self._denominator * 2.0 - 1.0).astype(
            np.float32,
            copy=False,
        )

    def denormalize(self, values: npt.ArrayLike) -> FloatArray:
        array = self.validate(values)
        return ((array + 1.0) * 0.5 * self._denominator + self._q01).astype(
            np.float32,
            copy=False,
        )

    def wire_to_model(
        self,
        values: npt.ArrayLike,
        *,
        frame_count: int | None = None,
        slots_per_frame: int | None = None,
    ) -> FloatArray:
        """Convert schema-specific wire actions to ``[1, C, F, H, 1]``."""

        array = np.asarray(values, dtype=np.float32)
        if self.spec.wire_layout == "HC":
            array = self.validate(array)
            if array.ndim != 2:
                raise ValueError("HC wire action must have shape [horizon, channels]")
            if frame_count is None or slots_per_frame is None:
                raise ValueError(
                    "HC wire conversion requires frame_count and slots_per_frame"
                )
            if frame_count <= 0 or slots_per_frame <= 0:
                raise ValueError("frame_count and slots_per_frame must be positive")
            if frame_count * slots_per_frame != array.shape[0]:
                raise ValueError(
                    "wire horizon must equal frame_count * slots_per_frame"
                )
            grid = array.reshape(frame_count, slots_per_frame, self.spec.dim)
            return np.transpose(grid, (2, 0, 1))[None, ..., None].copy()
        if self.spec.wire_layout == "CFH":
            if array.ndim != 3 or array.shape[0] != self.spec.dim:
                raise ValueError(
                    f"CFH wire action must have shape [{self.spec.dim}, F, H]"
                )
            if not np.isfinite(array).all():
                raise ValueError("wire action contains non-finite values")
            if frame_count is not None or slots_per_frame is not None:
                raise ValueError("CFH wire conversion derives frame and slot counts")
            return array[None, ..., None].copy()
        raise ValueError(f"Unsupported wire layout: {self.spec.wire_layout!r}")

    def model_to_wire(self, values: npt.ArrayLike) -> FloatArray:
        """Convert model actions to the schema's ``HC`` or ``CFH`` wire layout."""

        array = np.asarray(values, dtype=np.float32)
        if (
            array.ndim != 5
            or array.shape[0] != 1
            or array.shape[1] != self.spec.dim
            or array.shape[-1] != 1
        ):
            raise ValueError(
                f"model action must have shape [1, {self.spec.dim}, F, H, 1], "
                f"got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError("model action contains non-finite values")
        channel_first = array[0, ..., 0]
        if self.spec.wire_layout == "HC":
            return (
                np.transpose(channel_first, (1, 2, 0))
                .reshape(
                    -1,
                    self.spec.dim,
                )
                .copy()
            )
        if self.spec.wire_layout == "CFH":
            return channel_first.copy()
        raise ValueError(f"Unsupported wire layout: {self.spec.wire_layout!r}")
