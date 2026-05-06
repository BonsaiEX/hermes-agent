from __future__ import annotations

from typing import Any, Dict, List

import pytest

from plugins.memory import byterover as byterover_module
from plugins.memory.byterover import ByteRoverMemoryProvider, _DEFAULT_CONFIG


class _ImmediateThread:
    """start() で即座に target を実行する軽量 thread 代替。"""

    def __init__(self, target=None, daemon=None, name=None):
        self._target = target
        self.daemon = daemon
        self.name = name
        self._alive = False

    def start(self):
        self._alive = True
        try:
            if self._target:
                self._target()
        finally:
            self._alive = False

    def join(self, timeout=None):
        return None

    def is_alive(self):
        return self._alive


@pytest.fixture
def provider(tmp_path, monkeypatch):
    monkeypatch.setattr(byterover_module, "threading", byterover_module.threading)
    monkeypatch.setattr(byterover_module.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(
        byterover_module,
        "_load_byterover_config",
        lambda: dict(_DEFAULT_CONFIG),
    )
    p = ByteRoverMemoryProvider()
    p.initialize("session-1", hermes_home=str(tmp_path), platform="cli")
    return p


def test_load_byterover_config_prefers_memory_section(monkeypatch):
    monkeypatch.setattr(
        byterover_module,
        "load_config",
        lambda: {
            "memory": {
                "byterover": {
                    "curate_every_n_turns": 7,
                    "session_end_timeout": 11,
                }
            },
            "byterover": {
                "curate_every_n_turns": 3,
                "session_end_timeout": 99,
            },
        },
    )

    cfg = byterover_module._load_byterover_config()

    assert cfg["curate_every_n_turns"] == 7
    assert cfg["session_end_timeout"] == 11


def test_load_byterover_config_uses_defaults_when_missing(monkeypatch):
    monkeypatch.setattr(byterover_module, "load_config", lambda: {})

    cfg = byterover_module._load_byterover_config()

    assert cfg == _DEFAULT_CONFIG


def test_sync_turn_accumulates_then_flushes_every_fifth_turn(provider, monkeypatch):
    calls: List[Dict[str, Any]] = []

    def fake_run_brv(args, timeout=None, cwd=None):
        calls.append({"args": args, "timeout": timeout, "cwd": cwd})
        return {"success": True, "output": ""}

    monkeypatch.setattr(byterover_module, "_run_brv", fake_run_brv)

    for idx in range(4):
        provider.sync_turn("x" * 100 + str(idx), "assistant", session_id="session-1")

    assert calls == []
    assert len(provider._pending_turns) == 4

    provider.sync_turn("y" * 100, "assistant", session_id="session-1")

    assert len(calls) == 1
    assert calls[0]["args"][0] == "curate"
    payload = calls[0]["args"][2]
    assert payload.count("User:") == 5
    assert provider._pending_turns == []
    assert provider._pending_chars == 0


def test_sync_turn_short_user_skips_auto_curate_but_keeps_pending(provider, monkeypatch):
    calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        byterover_module,
        "_run_brv",
        lambda args, timeout=None, cwd=None: calls.append({"args": args}) or {"success": True, "output": ""},
    )

    for idx in range(5):
        provider.sync_turn(f"short-{idx}-msg", "assistant", session_id="session-1")

    assert calls == []
    assert len(provider._pending_turns) == 5
    assert provider._pending_chars > 0


def test_sync_turn_failed_curate_keeps_pending(provider, monkeypatch):
    monkeypatch.setattr(
        byterover_module,
        "_run_brv",
        lambda args, timeout=None, cwd=None: {"success": False, "error": "boom"},
    )

    for idx in range(5):
        provider.sync_turn("z" * 100 + str(idx), "assistant", session_id="session-1")

    assert len(provider._pending_turns) == 5
    assert provider._pending_chars > 0


def test_on_session_end_skips_below_pending_threshold(provider, monkeypatch):
    calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        byterover_module,
        "_run_brv",
        lambda args, timeout=None, cwd=None: calls.append({"args": args}) or {"success": True, "output": ""},
    )

    provider.sync_turn("x" * 100, "a", session_id="session-1")
    provider.on_session_end([])

    assert calls == []
    assert len(provider._pending_turns) == 1


def test_on_session_end_flushes_when_pending_threshold_met(provider, monkeypatch):
    calls: List[Dict[str, Any]] = []

    def fake_run_brv(args, timeout=None, cwd=None):
        calls.append({"args": args, "timeout": timeout})
        return {"success": True, "output": ""}

    monkeypatch.setattr(byterover_module, "_run_brv", fake_run_brv)

    for idx in range(3):
        provider.sync_turn("x" * 100 + str(idx), "assistant" * 10, session_id="session-1")

    provider.on_session_end([])

    assert len(calls) == 1
    assert calls[0]["args"][0] == "curate"
    assert calls[0]["timeout"] == _DEFAULT_CONFIG["session_end_timeout"]
    assert provider._pending_turns == []


def test_on_pre_compress_respects_disable_flag(provider, monkeypatch):
    calls: List[Dict[str, Any]] = []
    provider._config["pre_compression_curate"] = False
    monkeypatch.setattr(
        byterover_module,
        "_run_brv",
        lambda args, timeout=None, cwd=None: calls.append({"args": args}) or {"success": True, "output": ""},
    )

    result = provider.on_pre_compress(
        [{"role": "user", "content": "important context"}]
    )

    assert result == ""
    assert calls == []
