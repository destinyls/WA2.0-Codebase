# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Production core for bounded latest-only real-time sensor ingestion."""

from .buffers import BoundedLatestRing
from .contracts import ClockStamp, CoherentSnapshot, SensorPacket
from .planner import (
    GatedPlannerResultSlot,
    PlannerGateDecision,
    PlannerRequest,
    PlannerResult,
    PlannerResultPublication,
    PlannerResultGate,
)
from .observations import (
    AgileXObservationSpec,
    FrankaObservationSpec,
    build_agilex_live_observation,
    build_franka_live_observation,
)
from .runtime import SensorSource, ThreadedSensorRuntime
from .synchronizer import (
    LatestCoherentSynchronizer,
    LatestSnapshotSlot,
    SynchronizerConfig,
)

__all__ = (
    "AgileXObservationSpec",
    "BoundedLatestRing",
    "ClockStamp",
    "CoherentSnapshot",
    "FrankaObservationSpec",
    "GatedPlannerResultSlot",
    "LatestCoherentSynchronizer",
    "LatestSnapshotSlot",
    "PlannerGateDecision",
    "PlannerRequest",
    "PlannerResult",
    "PlannerResultPublication",
    "PlannerResultGate",
    "SensorPacket",
    "SensorSource",
    "SynchronizerConfig",
    "ThreadedSensorRuntime",
    "build_agilex_live_observation",
    "build_franka_live_observation",
)
