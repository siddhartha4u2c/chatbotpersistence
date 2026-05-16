"""
LangGraph chat graph + Groq-backed LLM.

Two layers of “memory” work together:

1. **Checkpoint thread (`thread_id`)** — LangGraph’s ``messages`` field uses ``add_messages``, so each ``invoke`` merges new turns within the lifetime of that checkpoint key. Persisted servers use SQLite on disk (see ``open_sqlite_checkpointer``); the Streamlit overlay stores the active ``thread_id`` in JSON so reopening resumes the matching checkpoint row.

2. **`prior_context` string** — The Streamlit layer stores trimmed transcripts
   from *ended* sessions and passes them on each invoke. The chat node turns
   that into extra system prompts so the model can recall facts from earlier
   sessions without sharing the old thread checkpoint.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Annotated, Tuple

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def _groq_chat(
    *,
    temperature: float = 0.5,
    max_tokens: int = 2000,
    model: str | None = None,
) -> ChatOpenAI:
    """Build LangChain OpenAI-compat client pointing at Groq’s chat endpoint.

    Groq exposes an OpenAI-style HTTP API (`/v1/chat/completions`). We use
    `ChatOpenAI` with `base_url` set to Groq instead of importing a separate
    Groq SDK. Reads ``GROQ_API_KEY`` (required) and optional ``GROQ_MODEL``,
    ``GROQ_OPENAI_COMPAT_URL`` from the environment.
    """
    load_dotenv(override=False)
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your environment (.env locally or Render)."
        )

    resolved_model = (
        model
        or os.getenv("GROQ_MODEL", "").strip()
        or "openai/gpt-oss-20b"
    )

    return ChatOpenAI(
        model=resolved_model,
        base_url=os.getenv(
            "GROQ_OPENAI_COMPAT_URL", "https://api.groq.com/openai/v1"
        ).rstrip("/"),
        api_key=key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


class ChatState(TypedDict):
    """Stateful fields LangGraph merges across steps and checkpoints.

    ``messages``: append-only transcript for the current checkpoint
    ``thread_id`` (see ``add_messages`` reducer). Loaded/saved by the saver
    passed to ``compile``.

    ``prior_context``: plain text injected by the UI on each invoke; not part
    of the conversational transcript reducer, but may be persisted in snapshots
    depending on saver behavior. Always treat as optional context the model may
    use for continuity.
    """

    # Conversation for this checkpoint thread_id (persisted via checkpointer).
    messages: Annotated[list[BaseMessage], add_messages]
    # Frozen text blobs from ended sessions (set by caller each invoke); not merged into messages reducer.
    prior_context: str


def build_chat_graph(*, llm: ChatOpenAI | None = None):
    """Define a minimal single-node graph: START → chat → END.

    The graph does not branch; one node sends the accumulated state ``messages``
    (plus system/instruction prompts) to the LLM and returns one assistant reply
    appended to ``messages``. Supply ``llm`` to reuse one client across tests.
    """
    llm_local = llm or _groq_chat()

    def chat_node(state: ChatState):
        """Read user/assistant turns from state, prepend system prompts, call LLM once.

        ``state["messages"]`` already includes full history *for this thread*
        thanks to checkpoints. ``prior_context`` is converted into an additional
        system message so cross-session snippets do not clutter the reducer list
        with fake user turns but still influence replies.
        """
        prior = (state.get("prior_context") or "").strip()
        msgs = state["messages"]

        llm_input: list[BaseMessage] = [
            SystemMessage(
                content=(
                    "You are a helpful assistant. Stay consistent with facts the user has told you "
                    "in this conversation and in any supplied prior-session context."
                )
            )
        ]

        if prior:
            llm_input.append(
                SystemMessage(
                    content=(
                        "Earlier sessions with this user (may be trimmed). Treat as memory for continuity:\n\n"
                        + prior
                    )
                )
            )

        llm_input.extend(msgs)
        reply = llm_local.invoke(llm_input)
        # Returning only new messages merges via add_messages; prior_context is unchanged unless caller overrides.
        return {"messages": [reply]}

    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph


def open_sqlite_checkpointer(db_path: str | Path) -> Tuple[SqliteSaver, sqlite3.Connection]:
    """Create a file-backed LangGraph saver (persists across Streamlit browser sessions).

    ``check_same_thread=False`` matches LangGraph’s recommendation for servers that
    may touch connections off the main thread; SqliteSaver serializes access with a lock.

    Returns both the saver and the raw connection so the app can pin the connection
    on ``st.session_state`` and avoid accidental garbage-collection surprises.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver, conn


def compile_chatbot(
    checkpointer: BaseCheckpointSaver | None = None, llm: ChatOpenAI | None = None
):
    """Attach a checkpointer so each ``thread_id`` keeps its own running state.

    Returns ``(compiled_runnable, checkpointer_instance)``. The UI holds the same
    compiled app and rotates ``thread_id`` when the user starts a “new chat”.
    """
    cp = checkpointer or InMemorySaver()
    return build_chat_graph(llm=llm).compile(checkpointer=cp), cp


def format_session_for_archive(messages: list[BaseMessage], *, max_chars: int = 6000) -> str:
    """Flatten Human/Assistant turns into one string for archived session memory.

    Used when the Streamlit user ends a chat: we serialize the transcript so
    the *next* session can receive it inside ``prior_context``. Non Human/AI
    message types are skipped. Long archives are truncated from the tail to bound
    token growth.
    """

    lines: list[str] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            prefix = "User"
        elif isinstance(m, AIMessage):
            prefix = "Assistant"
        else:
            continue
        text = (
            str(m.content) if isinstance(m.content, str) else str(m.content)
        ).strip()
        if not text:
            continue
        lines.append(f"{prefix}: {text}")

    assembled = "\n".join(lines)
    if len(assembled) <= max_chars:
        return assembled
    return assembled[-max_chars:]


__all__ = [
    "ChatState",
    "compile_chatbot",
    "format_session_for_archive",
    "open_sqlite_checkpointer",
]
