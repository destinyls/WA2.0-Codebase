# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Generation- and deadline-bound planner request/result contracts."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass

from .contracts import CoherentSnapshot, freeze_payload, require_int, require_text


@dataclass(frozen=True)
class PlannerRequest:
    request_id: int
    snapshot_id: int
    generation: int
    created_mono_ns: int
    deadline_mono_ns: int
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        require_int("request_id", self.request_id)
        require_int("snapshot_id", self.snapshot_id)
        require_int("generation", self.generation)
        created_ns = require_int("created_mono_ns", self.created_mono_ns)
        deadline_ns = require_int("deadline_mono_ns", self.deadline_mono_ns)
        if deadline_ns < created_ns:
            raise ValueError("deadline_mono_ns must not precede created_mono_ns")
        object.__setattr__(
            self,
            "payload",
            freeze_payload(self.payload, name="payload"),
        )


@dataclass(frozen=True)
class PlannerResult:
    request_id: int
    snapshot_id: int
    generation: int
    started_mono_ns: int
    finished_mono_ns: int
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        require_int("request_id", self.request_id)
        require_int("snapshot_id", self.snapshot_id)
        require_int("generation", self.generation)
        started_ns = require_int("started_mono_ns", self.started_mono_ns)
        finished_ns = require_int("finished_mono_ns", self.finished_mono_ns)
        if finished_ns < started_ns:
            raise ValueError("finished_mono_ns must not precede started_mono_ns")
        object.__setattr__(
            self,
            "payload",
            freeze_payload(self.payload, name="payload"),
        )


@dataclass(frozen=True)
class PlannerGateDecision:
    accepted: bool
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be a bool")
        require_text("reason", self.reason)


class PlannerResultGate:
    """Fail closed unless a result still belongs to the active request."""

    def evaluate(
        self,
        *,
        request: PlannerRequest,
        result: PlannerResult,
        now_mono_ns: int,
        latest_request_id: int,
        active_generation: int,
    ) -> PlannerGateDecision:
        if not isinstance(request, PlannerRequest):
            raise TypeError("request must be a PlannerRequest")
        if not isinstance(result, PlannerResult):
            raise TypeError("result must be a PlannerResult")
        now_ns = require_int("now_mono_ns", now_mono_ns)
        latest_id = require_int("latest_request_id", latest_request_id)
        generation = require_int("active_generation", active_generation)

        if request.request_id != latest_id:
            return PlannerGateDecision(False, "superseded request")
        if request.generation != generation or result.generation != generation:
            return PlannerGateDecision(False, "generation mismatch")
        if result.request_id != request.request_id:
            return PlannerGateDecision(False, "request mismatch")
        if result.snapshot_id != request.snapshot_id:
            return PlannerGateDecision(False, "snapshot mismatch")
        if result.started_mono_ns < request.created_mono_ns:
            return PlannerGateDecision(False, "result timing precedes request")
        if (
            now_ns > request.deadline_mono_ns
            or result.finished_mono_ns > request.deadline_mono_ns
        ):
            return PlannerGateDecision(False, "deadline expired")
        if result.finished_mono_ns > now_ns:
            return PlannerGateDecision(False, "result finished in the future")
        return PlannerGateDecision(True, "accepted")


@dataclass(frozen=True)
class PlannerResultPublication:
    """One accepted result exposed to an action arbiter."""

    cursor: int
    request: PlannerRequest
    result: PlannerResult

    def __post_init__(self) -> None:
        require_int("cursor", self.cursor, minimum=1)
        if not isinstance(self.request, PlannerRequest):
            raise TypeError("request must be a PlannerRequest")
        if not isinstance(self.result, PlannerResult):
            raise TypeError("result must be a PlannerResult")


class GatedPlannerResultSlot:
    """Publish no planner result until snapshot, generation, and deadline pass."""

    def __init__(self, *, gate: PlannerResultGate | None = None) -> None:
        self._gate = gate if gate is not None else PlannerResultGate()
        if not isinstance(self._gate, PlannerResultGate):
            raise TypeError("gate must be a PlannerResultGate")
        self._lock = threading.Lock()
        self._active_generation: int | None = None
        self._next_request_id = 0
        self._latest_request: PlannerRequest | None = None
        self._completed_request_id: int | None = None
        self._cursor = 0
        self._publication: PlannerResultPublication | None = None

    @property
    def active_generation(self) -> int | None:
        with self._lock:
            return self._active_generation

    @property
    def cursor(self) -> int:
        with self._lock:
            return self._cursor

    def activate_generation(self, generation: int) -> None:
        checked_generation = require_int("generation", generation)
        with self._lock:
            if (
                self._active_generation is not None
                and checked_generation < self._active_generation
            ):
                return
            if checked_generation == self._active_generation:
                return
            self._active_generation = checked_generation
            self._latest_request = None
            self._completed_request_id = None
            self._publication = None

    def submit(
        self,
        *,
        snapshot: CoherentSnapshot,
        created_mono_ns: int,
        deadline_mono_ns: int,
        payload: Mapping[str, object],
    ) -> PlannerRequest:
        if not isinstance(snapshot, CoherentSnapshot):
            raise TypeError("snapshot must be a CoherentSnapshot")
        if snapshot.snapshot_id < 1:
            raise ValueError("planner requests require a published snapshot ID")
        if created_mono_ns < snapshot.reference_mono_ns:
            raise ValueError("planner request cannot precede its snapshot")
        with self._lock:
            if self._active_generation is None:
                self._active_generation = snapshot.generation
            if snapshot.generation != self._active_generation:
                raise ValueError("snapshot generation is not active")
            self._next_request_id += 1
            request = PlannerRequest(
                request_id=self._next_request_id,
                snapshot_id=snapshot.snapshot_id,
                generation=snapshot.generation,
                created_mono_ns=created_mono_ns,
                deadline_mono_ns=deadline_mono_ns,
                payload=payload,
            )
            self._latest_request = request
            self._completed_request_id = None
            self._publication = None
            return request

    def publish(
        self,
        result: PlannerResult,
        *,
        now_mono_ns: int,
    ) -> PlannerGateDecision:
        if not isinstance(result, PlannerResult):
            raise TypeError("result must be a PlannerResult")
        with self._lock:
            request = self._latest_request
            generation = self._active_generation
            if request is None or generation is None:
                return PlannerGateDecision(False, "no active planner request")
            if result.request_id < request.request_id:
                return PlannerGateDecision(False, "superseded request")
            if self._completed_request_id == result.request_id:
                return PlannerGateDecision(False, "duplicate result")
            decision = self._gate.evaluate(
                request=request,
                result=result,
                now_mono_ns=now_mono_ns,
                latest_request_id=request.request_id,
                active_generation=generation,
            )
            if not decision.accepted:
                return decision
            self._cursor += 1
            self._publication = PlannerResultPublication(
                cursor=self._cursor,
                request=request,
                result=result,
            )
            self._completed_request_id = result.request_id
            return decision

    def take_newer_than(
        self,
        cursor: int,
        *,
        now_mono_ns: int,
    ) -> PlannerResultPublication | None:
        checked_cursor = require_int("cursor", cursor)
        now_ns = require_int("now_mono_ns", now_mono_ns)
        with self._lock:
            if self._publication is None or self._cursor <= checked_cursor:
                return None
            if now_ns > self._publication.request.deadline_mono_ns:
                self._publication = None
                return None
            return self._publication
