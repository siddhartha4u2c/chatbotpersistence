# Memory-based chatbot (Streamlit · LangGraph · Groq)

Streamlit UI over a LangGraph chat graph with SQLite checkpoints and capped “prior session” text passed into each turn (`prior_context`). The model can recall facts from ended sessions without replaying old UI bubbles after refresh.

## Local setup

Requirements: Python **3.12+** (see `runtime.txt`).

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Copy environment template and add your Groq API key:

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=...
```

Run:

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes (for chatting) | Groq API key |
| `GROQ_MODEL` | No | Overrides default Groq/OpenAI-compat model ID |
| `GROQ_OPENAI_COMPAT_URL` | No | Default `https://api.groq.com/openai/v1` |
| `CHATBOT_DATA_DIR` | No | Root folder for SQLite + `ui_state.json` (default `.chat_data`) |
| `CHATBOT_PROFILE` | No | Profile slug under `CHATBOT_DATA_DIR` (default `default`) |
| `CHATBOT_DISABLE_DISK` | No | Set to `1` / `true` for in-memory checkpoints only |

Never commit `.env` or API keys.

## Git checklist

- If there is no repository yet: `git init` in the project root.
- Confirm `.env` is **not** listed by `git status` (`.gitignore` should exclude it).
- First commit tip: stage `app.py`, `graph_app.py`, `session_store.py`, `requirements.txt`, `runtime.txt`, `render.yaml`, `.env.example`, `README.md`, `.gitignore`, `.gitattributes`, and `.streamlit/config.toml`.

## Deploy on Render

1. Push this repo to GitHub (without `.env`).
2. In Render: **New** → **Blueprint** (or **Web Service**) and point at the repo.
3. Set **Environment** → add secret **`GROQ_API_KEY`** (Render does not read your local `.env`).
4. Confirm **Build** matches `render.yaml` (upgrade pip + `requirements.txt`) and **Start** is the Streamlit command below.

`render.yaml` defines a Python web service with:

- **Build:** `pip install --upgrade pip && pip install -r requirements.txt`
- **Start:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

### Persistence on Render

By default, data is written under `./.chat_data` in the service container. **Ephemeral disk:** that data is lost when the instance is redeployed or moved unless you attach a **persistent disk** and set `CHATBOT_DATA_DIR` to the disk mount path (see Render docs: [Disks](https://render.com/docs/disks)).

## Repository layout

| File | Role |
|------|------|
| `app.py` | Streamlit shell, archiving, LangGraph invoke |
| `graph_app.py` | Graph definition, Groq client, SQLite checkpointer helper |
| `session_store.py` | Paths + `ui_state.json` load/save |
| `requirements.txt` | Python dependencies |
| `render.yaml` | Render Blueprint (optional) |
| `runtime.txt` | Python version hint for hosts that support it |

## License

Use and modify as you like for your own project; add a license file if you open-source the repo publicly.
