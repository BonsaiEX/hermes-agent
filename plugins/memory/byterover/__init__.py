"""ByteRover memory plugin — MemoryProvider interface.

Persistent memory via the ByteRover CLI (``brv``). Organizes knowledge into
a hierarchical context tree with tiered retrieval (fuzzy text → LLM-driven
search). Local-first with optional cloud sync.

Original PR #3499 by hieuntg81, adapted to MemoryProvider ABC.

Requires: ``brv`` CLI installed (npm install -g byterover-cli or
curl -fsSL https://byterover.dev/install.sh | sh).

Config via environment variables (profile-scoped via each profile's .env):
  BRV_API_KEY   — ByteRover API key (for cloud features, optional for local)

Working directory: $HERMES_HOME/byterover/ (profile-scoped context tree)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from hermes_cli.config import cfg_get, load_config
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Timeouts
_QUERY_TIMEOUT = 10   # brv query — should be fast
_CURATE_TIMEOUT = 120  # brv curate — may involve LLM processing

# Minimum lengths to filter noise
_MIN_QUERY_LEN = 10
_MIN_OUTPUT_LEN = 20
_SYNC_JOIN_TIMEOUT = 5.0
_SHUTDOWN_JOIN_TIMEOUT = 10.0

_DEFAULT_CONFIG = {
    "auto_query": True,
    "auto_curate": True,
    "curate_every_n_turns": 5,
    "min_user_chars_for_curate": 80,
    "pre_compression_curate": True,
    "session_end_curate": True,
    "session_end_min_pending_chars": 300,
    "session_end_timeout": 30,
}


# ---------------------------------------------------------------------------
# brv binary resolution (cached, thread-safe)
# ---------------------------------------------------------------------------

_brv_path_lock = threading.Lock()
_cached_brv_path: Optional[str] = None


def _resolve_brv_path() -> Optional[str]:
    """Find the brv binary on PATH or well-known install locations."""
    global _cached_brv_path
    with _brv_path_lock:
        if _cached_brv_path is not None:
            return _cached_brv_path if _cached_brv_path != "" else None

    found = shutil.which("brv")
    if not found:
        home = Path.home()
        candidates = [
            home / ".brv-cli" / "bin" / "brv",
            Path("/usr/local/bin/brv"),
            home / ".npm-global" / "bin" / "brv",
        ]
        for c in candidates:
            if c.exists():
                found = str(c)
                break

    with _brv_path_lock:
        if _cached_brv_path is not None:
            return _cached_brv_path if _cached_brv_path != "" else None
        _cached_brv_path = found or ""
    return found


def _run_brv(args: List[str], timeout: int = _QUERY_TIMEOUT,
             cwd: str = None) -> dict:
    """Run a brv CLI command. Returns {success, output, error}."""
    brv_path = _resolve_brv_path()
    if not brv_path:
        return {"success": False, "error": "brv CLI not found. Install: npm install -g byterover-cli"}

    cmd = [brv_path] + args
    effective_cwd = cwd or str(_get_brv_cwd())
    Path(effective_cwd).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    brv_bin_dir = str(Path(brv_path).parent)
    env["PATH"] = brv_bin_dir + os.pathsep + env.get("PATH", "")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, cwd=effective_cwd, env=env,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode == 0:
            return {"success": True, "output": stdout}
        return {"success": False, "error": stderr or stdout or f"brv exited {result.returncode}"}

    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"brv timed out after {timeout}s"}
    except FileNotFoundError:
        global _cached_brv_path
        with _brv_path_lock:
            _cached_brv_path = None
        return {"success": False, "error": "brv CLI not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _get_brv_cwd() -> Path:
    """Profile-scoped working directory for the brv context tree."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "byterover"


def _parse_bool_setting(value: Any, default: bool) -> bool:
    """設定値を bool に正規化する。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    if value is None:
        return default
    return bool(value)


def _parse_int_setting(value: Any, default: int, minimum: int) -> int:
    """設定値を最小値付き整数へ正規化する。"""
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return default


def _load_byterover_config() -> Dict[str, Any]:
    """memory.byterover を優先し、互換用に top-level byterover も読む。"""
    config = dict(_DEFAULT_CONFIG)
    try:
        raw = load_config() or {}
    except Exception:
        raw = {}

    candidates = [
        raw.get("byterover", {}) or {},
        cfg_get(raw, "memory", "byterover", default={}) or {},
    ]
    for section in candidates:
        if not isinstance(section, dict):
            continue
        config.update({k: v for k, v in section.items() if v is not None})

    config["auto_query"] = _parse_bool_setting(
        config.get("auto_query"), _DEFAULT_CONFIG["auto_query"]
    )
    config["auto_curate"] = _parse_bool_setting(
        config.get("auto_curate"), _DEFAULT_CONFIG["auto_curate"]
    )
    config["pre_compression_curate"] = _parse_bool_setting(
        config.get("pre_compression_curate"),
        _DEFAULT_CONFIG["pre_compression_curate"],
    )
    config["session_end_curate"] = _parse_bool_setting(
        config.get("session_end_curate"),
        _DEFAULT_CONFIG["session_end_curate"],
    )
    config["curate_every_n_turns"] = _parse_int_setting(
        config.get("curate_every_n_turns"),
        _DEFAULT_CONFIG["curate_every_n_turns"],
        1,
    )
    config["min_user_chars_for_curate"] = _parse_int_setting(
        config.get("min_user_chars_for_curate"),
        _DEFAULT_CONFIG["min_user_chars_for_curate"],
        0,
    )
    config["session_end_min_pending_chars"] = _parse_int_setting(
        config.get("session_end_min_pending_chars"),
        _DEFAULT_CONFIG["session_end_min_pending_chars"],
        0,
    )
    config["session_end_timeout"] = _parse_int_setting(
        config.get("session_end_timeout"),
        _DEFAULT_CONFIG["session_end_timeout"],
        1,
    )
    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

QUERY_SCHEMA = {
    "name": "brv_query",
    "description": (
        "Search ByteRover's persistent knowledge tree for relevant context. "
        "Returns memories, project knowledge, architectural decisions, and "
        "patterns from previous sessions. Use for any question where past "
        "context would help."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
        },
        "required": ["query"],
    },
}

CURATE_SCHEMA = {
    "name": "brv_curate",
    "description": (
        "Store important information in ByteRover's persistent knowledge tree. "
        "Use for architectural decisions, bug fixes, user preferences, project "
        "patterns — anything worth remembering across sessions. ByteRover's LLM "
        "automatically categorizes and organizes the memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The information to remember."},
        },
        "required": ["content"],
    },
}

STATUS_SCHEMA = {
    "name": "brv_status",
    "description": "Check ByteRover status — CLI version, context tree stats, cloud sync state.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class ByteRoverMemoryProvider(MemoryProvider):
    """ByteRover persistent memory via the brv CLI."""

    def __init__(self):
        self._cwd = ""
        self._session_id = ""
        self._turn_count = 0
        self._config: Dict[str, Any] = dict(_DEFAULT_CONFIG)
        self._sync_thread: Optional[threading.Thread] = None
        self._session_end_thread: Optional[threading.Thread] = None
        self._pending_turns: List[str] = []
        self._pending_chars = 0
        self._state_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "byterover"

    def is_available(self) -> bool:
        """Check if brv CLI is installed. No network calls."""
        return _resolve_brv_path() is not None

    def get_config_schema(self):
        return [
            {
                "key": "api_key",
                "description": "ByteRover API key (optional, for cloud sync)",
                "secret": True,
                "env_var": "BRV_API_KEY",
                "url": "https://app.byterover.dev",
            },
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        self._cwd = str(_get_brv_cwd())
        self._session_id = session_id
        self._turn_count = 0
        self._config = _load_byterover_config()
        self._pending_turns = []
        self._pending_chars = 0
        Path(self._cwd).mkdir(parents=True, exist_ok=True)

    def system_prompt_block(self) -> str:
        if not _resolve_brv_path():
            return ""
        return (
            "# ByteRover Memory\n"
            "Active. Persistent knowledge tree with hierarchical context.\n"
            "Use brv_query to search past knowledge, brv_curate to store "
            "important facts, brv_status to check state."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Run brv query synchronously before the agent's first LLM call.

        Blocks until the query completes (up to _QUERY_TIMEOUT seconds), ensuring
        the result is available as context before the model is called.
        """
        if not self._config.get("auto_query", True):
            return ""
        if not query or len(query.strip()) < _MIN_QUERY_LEN:
            return ""
        result = _run_brv(
            ["query", "--", query.strip()[:5000]],
            timeout=_QUERY_TIMEOUT, cwd=self._cwd,
        )
        if result["success"] and result.get("output"):
            output = result["output"].strip()
            if len(output) > _MIN_OUTPUT_LEN:
                return f"## ByteRover Context\n{output}"
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """No-op: prefetch() now runs synchronously at turn start."""
        pass

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """完了ターンを pending に積み、条件を満たす時だけ background curate する。"""
        self._turn_count += 1

        if len(user_content.strip()) < _MIN_QUERY_LEN:
            return

        turn_payload = (
            f"User: {user_content[:2000]}\nAssistant: {assistant_content[:2000]}"
        )
        with self._state_lock:
            self._pending_turns.append(turn_payload)
            self._pending_chars += len(turn_payload)

        if not self._config.get("auto_curate", True):
            return
        if len(user_content.strip()) < self._config["min_user_chars_for_curate"]:
            return
        if self._turn_count % self._config["curate_every_n_turns"] != 0:
            return

        self._spawn_pending_curate(
            reason="turn",
            timeout=_CURATE_TIMEOUT,
            thread_name="brv-sync",
        )

    def _spawn_pending_curate(self, *, reason: str, timeout: int, thread_name: str) -> bool:
        """pending snapshot を安全に flush する worker を起動する。"""
        target_attr = "_sync_thread" if reason == "turn" else "_session_end_thread"
        existing = getattr(self, target_attr, None)
        if existing and existing.is_alive():
            existing.join(timeout=_SYNC_JOIN_TIMEOUT)
        if existing and existing.is_alive():
            logger.debug("ByteRover %s curate skipped: previous worker still running", reason)
            return False

        snapshot = self._snapshot_pending()
        if not snapshot:
            return False

        def _sync() -> None:
            result = _run_brv(
                ["curate", "--", snapshot],
                timeout=timeout,
                cwd=self._cwd,
            )
            if result["success"]:
                self._drop_pending_snapshot(snapshot)
                logger.info(
                    "ByteRover %s curate flushed: chars=%d",
                    reason,
                    len(snapshot),
                )
                return
            logger.debug(
                "ByteRover %s curate failed: %s",
                reason,
                result.get("error", "unknown error"),
            )

        thread = threading.Thread(target=_sync, daemon=True, name=thread_name)
        setattr(self, target_attr, thread)
        thread.start()
        return True

    def _snapshot_pending(self) -> str:
        """現在の pending 全体を一貫した snapshot として取り出す。"""
        with self._state_lock:
            if not self._pending_turns:
                return ""
            return "\n\n".join(self._pending_turns)

    def _drop_pending_snapshot(self, snapshot: str) -> None:
        """成功した snapshot 分だけ pending 先頭から取り除く。"""
        if not snapshot:
            return
        snapshot_parts = snapshot.split("\n\n")
        with self._state_lock:
            if self._pending_turns[:len(snapshot_parts)] != snapshot_parts:
                return
            del self._pending_turns[:len(snapshot_parts)]
            self._pending_chars = sum(len(item) for item in self._pending_turns)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes to ByteRover."""
        if action not in ("add", "replace") or not content:
            return

        def _write():
            try:
                label = "User profile" if target == "user" else "Agent memory"
                _run_brv(
                    ["curate", "--", f"[{label}] {content}"],
                    timeout=_CURATE_TIMEOUT, cwd=self._cwd,
                )
            except Exception as e:
                logger.debug("ByteRover memory mirror failed: %s", e)

        t = threading.Thread(target=_write, daemon=True, name="brv-memwrite")
        t.start()

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract insights before context compression discards turns."""
        if not self._config.get("pre_compression_curate", True):
            return ""
        if not messages:
            return ""

        # Build a summary of messages about to be compressed
        parts = []
        for msg in messages[-10:]:  # last 10 messages
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip() and role in ("user", "assistant"):
                parts.append(f"{role}: {content[:500]}")

        if not parts:
            return ""

        combined = "\n".join(parts)

        def _flush():
            try:
                _run_brv(
                    ["curate", "--", f"[Pre-compression context]\n{combined}"],
                    timeout=_CURATE_TIMEOUT, cwd=self._cwd,
                )
                logger.info("ByteRover pre-compression flush: %d messages", len(parts))
            except Exception as e:
                logger.debug("ByteRover pre-compression flush failed: %s", e)

        t = threading.Thread(target=_flush, daemon=True, name="brv-flush")
        t.start()
        return ""

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """終了直前に十分な pending がある時だけ background curate する。"""
        if not self._config.get("session_end_curate", True):
            return
        with self._state_lock:
            pending_chars = self._pending_chars
        if pending_chars < self._config["session_end_min_pending_chars"]:
            return
        self._spawn_pending_curate(
            reason="session_end",
            timeout=self._config["session_end_timeout"],
            thread_name="brv-session-end",
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [QUERY_SCHEMA, CURATE_SCHEMA, STATUS_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if tool_name == "brv_query":
            return self._tool_query(args)
        elif tool_name == "brv_curate":
            return self._tool_curate(args)
        elif tool_name == "brv_status":
            return self._tool_status()
        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=_SHUTDOWN_JOIN_TIMEOUT)
        if self._session_end_thread and self._session_end_thread.is_alive():
            self._session_end_thread.join(
                timeout=min(
                    _SHUTDOWN_JOIN_TIMEOUT,
                    float(self._config.get("session_end_timeout", _CURATE_TIMEOUT)),
                )
            )

    # -- Tool implementations ------------------------------------------------

    def _tool_query(self, args: dict) -> str:
        query = args.get("query", "")
        if not query:
            return tool_error("query is required")

        result = _run_brv(
            ["query", "--", query.strip()[:5000]],
            timeout=_QUERY_TIMEOUT, cwd=self._cwd,
        )

        if not result["success"]:
            return tool_error(result.get("error", "Query failed"))

        output = result.get("output", "").strip()
        if not output or len(output) < _MIN_OUTPUT_LEN:
            return json.dumps({"result": "No relevant memories found."})

        # Truncate very long results
        if len(output) > 8000:
            output = output[:8000] + "\n\n[... truncated]"

        return json.dumps({"result": output})

    def _tool_curate(self, args: dict) -> str:
        content = args.get("content", "")
        if not content:
            return tool_error("content is required")

        result = _run_brv(
            ["curate", "--", content],
            timeout=_CURATE_TIMEOUT, cwd=self._cwd,
        )

        if not result["success"]:
            return tool_error(result.get("error", "Curate failed"))

        return json.dumps({"result": "Memory curated successfully."})

    def _tool_status(self) -> str:
        result = _run_brv(["status"], timeout=15, cwd=self._cwd)
        if not result["success"]:
            return tool_error(result.get("error", "Status check failed"))
        canonical_cwd = self._cwd or str(_get_brv_cwd())
        # 生の brv status はそのまま残しつつ、Hermes 側の正本 project を明示する。
        return json.dumps({
            "status": result.get("output", ""),
            "canonical_cwd": canonical_cwd,
            "project_path": canonical_cwd,
        })


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register ByteRover as a memory provider plugin."""
    ctx.register_memory_provider(ByteRoverMemoryProvider())
