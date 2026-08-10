# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from types import ModuleType

import pytest

import n0_twam


def test_legacy_top_level_modules_remain_lazily_accessible(monkeypatch) -> None:
    calls: list[str] = []

    def fake_import(name: str) -> ModuleType:
        calls.append(name)
        return ModuleType(name)

    monkeypatch.setattr(n0_twam, "import_module", fake_import)
    for name in ("configs", "distributed", "models"):
        n0_twam.__dict__.pop(name, None)
        assert getattr(n0_twam, name).__name__ == f"n0_twam.{name}"

    assert calls == [
        "n0_twam.configs",
        "n0_twam.distributed",
        "n0_twam.models",
    ]
    for name in ("configs", "distributed", "models"):
        n0_twam.__dict__.pop(name, None)


def test_unknown_top_level_attribute_still_fails() -> None:
    with pytest.raises(AttributeError):
        getattr(n0_twam, "act")
