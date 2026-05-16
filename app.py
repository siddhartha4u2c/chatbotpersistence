"""
Streamlit shell around the LangGraph chatbot.

**Persistence vs Streamlit**
Each full browser load shows an **empty** chat panel. Right before we mint a new
LangGraph ``thread_id``, we snapshot the **saved** checkpoint thread into
``archived_contexts`` so ``prior_context`` keeps names and recent turns after
refresh — without repainting old bubbles. Manual “New chat session” still rotates
threads and archives explicitly.

JSON + SQLite live under ``CHATBOT_DATA_DIR`` (``CHATBOT_PROFILE`` selects a subfolder).

Deploy on Render:
  pip install -r requirements.txt
  streamlit run app.py --server.port $PORT --server.address 0.0.0.0
"""

from __future__ import annotations

import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from graph_app import (
    compile_chatbot,
    format_session_for_archive,
    open_sqlite_checkpointer,
)
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from session_store import (
    checkpoint_db_path,
    disk_persistence_enabled,
    load_ui_state,
    resolve_profile,
    save_ui_state,
    ui_state_path,
)

REMEMBER_PRIOR_SESSIONS = 2


def _delete_checkpoint_thread(checkpointer, thread_id: str) -> None:
    """Drop saved graph state for ``thread_id`` (transcript already copied to archives)."""

    tid = str(thread_id).strip()
    if not tid:
        return
    deleter = getattr(checkpointer, "delete_thread", None)
    if not callable(deleter):
        return
    try:
        deleter(tid)
    except Exception:
        return


def _thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": str(thread_id), "checkpoint_ns": ""}}


def _msg_text(m) -> str:
    c = getattr(m, "content", "")
    return (c.strip() if isinstance(c, str) else str(c).strip())



def _extract_last_ai_text(invoke_out: object, app, cfg) -> str | None:
    if isinstance(invoke_out, dict):
        for m in reversed(list(invoke_out.get("messages") or [])):
            if getattr(m, "type", None) == "ai":
                txt = _msg_text(m)
                return txt if txt else None
    snap = app.get_state(cfg)
    synced = list((getattr(snap, "values", None) or {}).get("messages") or [])
    for m in reversed(synced):
        if getattr(m, "type", None) == "ai":
            txt = _msg_text(m)
            return txt if txt else None
    return None


def _prior_block(archived: list[str]) -> str:
    """Join archived session blobs into one string for ``prior_context``.

    Each blob is prose built by ``format_session_for_archive``; we wrap them with
    light headers so the model can distinguish multiple ended sessions.
    """
    if not archived:
        return ""
    parts: list[str] = []
    for i, block in enumerate(archived, start=1):
        parts.append(f"--- Earlier session #{i} ---\n{block}")
    return "\n\n".join(parts)


def _persist_meta() -> None:
    """Write ``thread_id`` + archived session blobs beside the SQLite checkpoint file."""

    if not st.session_state.get("persistence_enabled"):
        return
    path = ui_state_path(st.session_state.persistence_profile)
    save_ui_state(
        path=path,
        thread_id=st.session_state.thread_id,
        archived_contexts=list(st.session_state.archived_contexts),
        max_prior_sessions=REMEMBER_PRIOR_SESSIONS,
        display_turns=[],
    )


def _init_state() -> None:
    """Wire LangGraph once per Streamlit visitor session.

    Load ``archived_contexts`` from JSON, absorb the **prior** ``thread_id``
    checkpoint into a new archive snippet (disk only), then mint a fresh
    ``thread_id``. Sidebar stays empty; the model still sees memory via
    ``prior_context``. Keeps the last ``REMEMBER_PRIOR_SESSIONS`` archived sessions.
    """

    if "chat_ready" not in st.session_state:
        load_dotenv(override=False)
        use_disk = disk_persistence_enabled()
        profile = resolve_profile()

        sqlite_conn = None
        db_path = None
        if use_disk:
            db_path = checkpoint_db_path(profile)
            checkpointer_obj, sqlite_conn = open_sqlite_checkpointer(db_path)
        else:
            checkpointer_obj = MemorySaver()

        app, graph_saver = compile_chatbot(checkpointer=checkpointer_obj)
        meta = load_ui_state(ui_state_path(profile)) if use_disk else None

        archived: list[str] = []
        previous_tid: str | None = None
        if isinstance(meta, dict):
            ctx = meta.get("archived_contexts")
            if isinstance(ctx, list):
                archived = [str(x).strip() for x in ctx if str(x).strip()]
            raw_saved = meta.get("thread_id")
            if isinstance(raw_saved, str) and raw_saved.strip():
                previous_tid = raw_saved.strip()

        prev_blob = ""
        if use_disk and previous_tid:
            snap_prev = app.get_state(_thread_config(previous_tid))
            pv = getattr(snap_prev, "values", None) or {}
            msgs_prev = list(pv.get("messages") or [])
            if msgs_prev:
                prev_blob = (format_session_for_archive(msgs_prev) or "").strip()
            _delete_checkpoint_thread(graph_saver, previous_tid)
        if prev_blob and (not archived or archived[-1].strip() != prev_blob):
            archived.append(prev_blob)

        thread_id = str(uuid.uuid4())

        st.session_state.chat_app = app
        st.session_state.checkpointer = graph_saver
        st.session_state._sqlite_conn = sqlite_conn
        st.session_state.persistence_enabled = bool(use_disk)
        st.session_state.persistence_profile = profile
        st.session_state.persistence_folder = (
            str(db_path.parent.resolve()) if db_path is not None else ""
        )
        st.session_state.thread_id = thread_id
        st.session_state.archived_contexts = archived
        st.session_state.chat_viewport_id = 0
        st.session_state.ui_display_turns = []
        st.session_state.chat_ready = True

    elif st.session_state.get("chat_ready") and "ui_display_turns" not in st.session_state:
        st.session_state.ui_display_turns = []

    st.session_state.max_prior_sessions = REMEMBER_PRIOR_SESSIONS

    # Clamp archived sessions to REMEMBER_PRIOR_SESSIONS.
    st.session_state.archived_contexts = st.session_state.archived_contexts[
        -st.session_state.max_prior_sessions :
    ]

    _persist_meta()


def _archive_and_new_thread() -> None:
    """Snapshot the active ``thread_id`` conversation, enqueue it as prior memory, rotate ID.

    ``get_state`` reads the canonical checkpoint merged state (not just the last
    return value). Serialized text is capped per ``format_session_for_archive``.

    The old thread’s checkpoints are **deleted** from the checkpointer so the UI
    does not reload that transcript; the archive text still feeds ``prior_context``
    for the model. A brand-new ``thread_id`` starts with an empty on-screen thread.
    """
    _init_state()
    app = st.session_state.chat_app
    checkpointer = st.session_state.checkpointer

    old_tid = st.session_state.thread_id
    cfg = _thread_config(old_tid)
    snap = app.get_state(cfg)
    values = getattr(snap, "values", None) or {}
    messages = values.get("messages") or []

    if messages:
        blob = format_session_for_archive(messages)
        if blob:
            st.session_state.archived_contexts.append(blob)
            st.session_state.archived_contexts = st.session_state.archived_contexts[
                -st.session_state.max_prior_sessions :
            ]

    _delete_checkpoint_thread(checkpointer, old_tid)
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.ui_display_turns = []
    st.session_state.chat_viewport_id = int(
        st.session_state.get("chat_viewport_id", 0)
    ) + 1
    _persist_meta()


def _run_turn(user_text: str) -> None:
    """Append ``user_text`` inside the LangGraph runnable for the current ``thread_id``.

    Sends only this turn as incremental ``HumanMessage``; LangGraph merges with
    history through ``add_messages`` + checkpointer.

    Includes ``prior_context`` each time so cross-session narration stays wired
    even if the runnable snapshot stores that field.
    """
    _init_state()
    app = st.session_state.chat_app
    prior = _prior_block(st.session_state.archived_contexts)

    cfg = _thread_config(st.session_state.thread_id)
    invoke_out = app.invoke(
        {
            "messages": [HumanMessage(content=user_text)],
            "prior_context": prior,
        },
        config=cfg,
    )
    turns = list(st.session_state.get("ui_display_turns") or [])
    ut = user_text.strip()
    if ut:
        turns.append(("user", ut))
    reply = _extract_last_ai_text(invoke_out, app, cfg)
    if reply:
        turns.append(("assistant", reply))
    st.session_state.ui_display_turns = turns
    _persist_meta()


def main() -> None:
    """Render controls, hydrate session state, show chat history from checkpoints."""

    st.set_page_config(page_title="Memory chatbot (LangGraph)", page_icon=None)
    st.title("Memory-based chatbot")

    load_dotenv(override=False)
    missing_key = not (os.getenv("GROQ_API_KEY") or "").strip()

    with st.sidebar:
        if missing_key:
            st.error("`GROQ_API_KEY` is not set in the environment.")
        if st.button("New chat session", type="primary"):
            _archive_and_new_thread()
            st.rerun()

    _init_state()

    vp = int(st.session_state.get("chat_viewport_id", 0))
    prompt = st.chat_input(
        disabled=missing_key,
        placeholder="Message the assistant…",
        key=f"chat_input_{vp}",
    )

    # Sidebar shows only the current Streamlit session; prior sessions live in ``prior_context``.
    display_rows = list(st.session_state.get("ui_display_turns") or [])

    with st.container(key=f"chat_viewport_{vp}"):
        for role_key, content in display_rows:
            role = "assistant" if role_key == "assistant" else "user"
            with st.chat_message(role):
                st.markdown(content)

    if prompt and not missing_key:
        _run_turn(prompt.strip())
        st.rerun()


if __name__ == "__main__":
    main()
