"""Disk-backed UI metadata (thread id + archived transcripts).

LangGraph checkpoints are stored in SQLite (see ``graph_app.open_sqlite_checkpointer``).
This module stores companion JSON so we can reconnect to the **same**
``thread_id`` after closing the browser: the checkpoints file already holds the
message history for that thread.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


UI_STATE_FILE = "ui_state.json"
CHECKPOINT_DB = "checkpoints.sqlite"


def slug_profile(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", name.strip().lower()).strip("-") or ""
    return (cleaned or "default")[:80]


def resolve_profile() -> str:
    return slug_profile(os.getenv("CHATBOT_PROFILE", "default"))


def resolve_storage_root() -> Path:
    raw = (os.getenv("CHATBOT_DATA_DIR") or ".chat_data").strip() or ".chat_data"
    return Path(raw).expanduser().resolve()


def storage_dir_for_profile(profile: str | None = None) -> Path:
    prof = slug_profile(profile or resolve_profile())
    d = resolve_storage_root() / prof
    d.mkdir(parents=True, exist_ok=True)
    return d


def ui_state_path(profile: str | None = None) -> Path:
    return storage_dir_for_profile(profile) / UI_STATE_FILE


def checkpoint_db_path(profile: str | None = None) -> Path:
    return storage_dir_for_profile(profile) / CHECKPOINT_DB


def disk_persistence_enabled() -> bool:
    return os.getenv("CHATBOT_DISABLE_DISK", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    )


def load_ui_state(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def save_ui_state(
    *,
    path: Path,
    thread_id: str,
    archived_contexts: list[str],
    max_prior_sessions: int | None = None,
    display_turns: list[tuple[str, str]] | None = None,
) -> None:
    payload: dict = {
        "version": 2,
        "thread_id": thread_id,
        "archived_contexts": list(archived_contexts),
        "display_turns": [[r, t] for r, t in (display_turns or [])],
    }
    if max_prior_sessions is not None:
        payload["max_prior_sessions"] = max_prior_sessions
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "CHECKPOINT_DB",
    "checkpoint_db_path",
    "disk_persistence_enabled",
    "load_ui_state",
    "resolve_profile",
    "resolve_storage_root",
    "save_ui_state",
    "slug_profile",
    "storage_dir_for_profile",
    "ui_state_path",
]
