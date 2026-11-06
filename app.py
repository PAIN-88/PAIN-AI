import os
import base64
import mimetypes
import time
import re
import json
import html
import hmac
import shutil
import hashlib
import secrets
import sqlite3
import chainlit as cl
from chainlit.types import ThreadDict
from chainlit.server import app as fastapi_app
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from dotenv import load_dotenv
from typing_extensions import TypedDict
from chainlit.input_widget import Select
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

# -----------------------------------------------------------------------------
# Environment & API setup
# -----------------------------------------------------------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing in .env file")

# -----------------------------------------------------------------------------
# Available models
# -----------------------------------------------------------------------------
AVAILABLE_MODELS = {
    "Nemotron Vision (Free)-Image": "nvidia/nemotron-3.5-super-vl",
    "GPT-4o Mini": "openai/gpt-4o-mini",
    "LiquidAI: LFM2.5-2.6B-Coding": "liquid/lfm-2.5-2.6b:free",
    "NVIDIA: Nemotron 3 Nano Omni-Image": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "NVIDIA: Nemotron 3.5 Lightning": "nvidia/nemotron-3.5-lightning:free",
    "Dots Studio: Dots3-Note Preview": "dots-studio/dots-3-note-preview:free",
    "Cohere: North Mini Code-Coding": "cohere/north-mini-code:free",
    "NVIDIA: Nemotron 3.5 Super VL": "nvidia/nemotron-3.5-super-vl:free"
}
DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl"

# -----------------------------------------------------------------------------
# Document settings
# -----------------------------------------------------------------------------
MAX_CHARS_PER_DOC = 30000     # cut each document to this many characters
MAX_CHARS_TOTAL = 60000       # cut all documents together to this many characters

TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml", ".html",
    ".htm", ".css", ".js", ".ts", ".jsx", ".tsx", ".py", ".java", ".c", ".cpp",
    ".h", ".cs", ".go", ".rs", ".php", ".rb", ".sh", ".sql", ".ini", ".toml",
    ".cfg", ".log", ".rtf", ".tex",
}

# -----------------------------------------------------------------------------
# Chat history (sidebar list of past chats) -> needs auth + a data layer
# -----------------------------------------------------------------------------
DB_FILE = "chat_history.db"

def init_db():
    """Create the SQLite tables Chainlit needs (only if they don't exist yet)."""
    con = sqlite3.connect(DB_FILE)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        "id" TEXT PRIMARY KEY, "identifier" TEXT NOT NULL UNIQUE,
        "metadata" TEXT NOT NULL, "createdAt" TEXT
    );
    CREATE TABLE IF NOT EXISTS threads (
        "id" TEXT PRIMARY KEY, "createdAt" TEXT, "name" TEXT, "userId" TEXT,
        "userIdentifier" TEXT, "tags" TEXT, "metadata" TEXT,
        FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS steps (
        "id" TEXT PRIMARY KEY, "name" TEXT NOT NULL, "type" TEXT NOT NULL,
        "threadId" TEXT NOT NULL, "parentId" TEXT, "streaming" INTEGER NOT NULL,
        "waitForAnswer" INTEGER, "isError" INTEGER, "metadata" TEXT, "tags" TEXT,
        "input" TEXT, "output" TEXT, "createdAt" TEXT, "command" TEXT, "start" TEXT,
        "end" TEXT, "generation" TEXT, "showInput" TEXT, "language" TEXT,
        "indent" INTEGER, "defaultOpen" INTEGER, "autoCollapse" INTEGER
    );
    CREATE TABLE IF NOT EXISTS elements (
        "id" TEXT PRIMARY KEY, "threadId" TEXT, "type" TEXT, "url" TEXT,
        "chainlitKey" TEXT, "name" TEXT NOT NULL, "display" TEXT, "objectKey" TEXT,
        "size" TEXT, "page" INTEGER, "language" TEXT, "forId" TEXT, "mime" TEXT,
        "props" TEXT, "autoPlay" INTEGER, "playerConfig" TEXT
    );
    CREATE TABLE IF NOT EXISTS feedbacks (
        "id" TEXT PRIMARY KEY, "forId" TEXT NOT NULL, "threadId" TEXT NOT NULL,
        "value" INTEGER NOT NULL, "comment" TEXT
    );
    CREATE TABLE IF NOT EXISTS accounts (
        "username" TEXT PRIMARY KEY, "password_hash" TEXT NOT NULL, "created_at" TEXT
    );
    CREATE TABLE IF NOT EXISTS chat_state (
        "thread_id" TEXT PRIMARY KEY, "history" TEXT, "documents" TEXT
    );
    """)
    # Upgrade older databases: add any columns newer Chainlit versions need
    needed = {
        "steps": [("autoCollapse", "INTEGER")],
        "elements": [("autoPlay", "INTEGER"), ("playerConfig", "TEXT")],
    }
    for table, columns in needed.items():
        existing = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
        for name, col_type in columns:
            if name not in existing:
                con.execute(f'ALTER TABLE {table} ADD COLUMN "{name}" {col_type}')
    con.commit()
    con.close()

init_db()

FILES_DIR = "saved_files"

def save_state(thread_id: str, history: list, documents: dict):
    """Save the full chat context (history + extracted document text)."""
    con = sqlite3.connect(DB_FILE)
    con.execute(
        'INSERT INTO chat_state (thread_id, history, documents) VALUES (?, ?, ?) '
        'ON CONFLICT(thread_id) DO UPDATE SET history=excluded.history, documents=excluded.documents',
        (thread_id, json.dumps(history), json.dumps(documents)),
    )
    con.commit()
    con.close()

def load_state(thread_id: str):
    con = sqlite3.connect(DB_FILE)
    row = con.execute(
        "SELECT history, documents FROM chat_state WHERE thread_id = ?", (thread_id,)
    ).fetchone()
    con.close()
    if not row:
        return None
    return json.loads(row[0] or "[]"), json.loads(row[1] or "{}")

def get_thread_id() -> str:
    return cl.context.session.thread_id

class SQLiteDataLayer(SQLAlchemyDataLayer):
    """SQLite can't store Python lists, so tags are saved as a JSON string."""
    async def update_thread(self, thread_id, name=None, user_id=None, metadata=None, tags=None):
        if isinstance(tags, list):
            tags = json.dumps(tags)
        return await super().update_thread(
            thread_id=thread_id, name=name, user_id=user_id, metadata=metadata, tags=tags
        )

@cl.data_layer
def get_data_layer():
    return SQLiteDataLayer(conninfo=f"sqlite+aiosqlite:///{DB_FILE}")

# -----------------------------------------------------------------------------
# Accounts: anyone can sign up at /signup, then log in at /login
# -----------------------------------------------------------------------------
USERNAME_RE = re.compile(r"^[a-z0-9_.-]{3,30}$")
MIN_PASSWORD_LEN = 8

def hash_password(password: str, salt: str = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000)
    return f"{salt}${digest.hex()}"

def verify_password(password: str, stored: str) -> bool:
    try:
        salt, _ = stored.split("$", 1)
    except ValueError:
        return False
    return hmac.compare_digest(hash_password(password, salt), stored)

def create_account(username: str, password: str, confirm: str):
    """Returns an error string, or None when the account was created."""
    if not USERNAME_RE.match(username):
        return "Username must be 3-30 characters: letters, numbers, dot, dash or underscore."
    if len(password) < MIN_PASSWORD_LEN:
        return f"Password must be at least {MIN_PASSWORD_LEN} characters."
    if password != confirm:
        return "Passwords do not match."
    con = sqlite3.connect(DB_FILE)
    try:
        con.execute(
            "INSERT INTO accounts (username, password_hash, created_at) VALUES (?, ?, datetime('now'))",
            (username, hash_password(password)),
        )
        con.commit()
    except sqlite3.IntegrityError:
        return "That username is already taken."
    finally:
        con.close()
    return None

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    username = (username or "").strip().lower()
    con = sqlite3.connect(DB_FILE)
    row = con.execute("SELECT password_hash FROM accounts WHERE username = ?", (username,)).fetchone()
    con.close()
    if row and verify_password(password or "", row[0]):
        return cl.User(identifier=username, metadata={"role": "user"})
    return None

def signup_html(message: str = "", ok: bool = False) -> str:
    color = "#15803d" if ok else "#b91c1c"
    note = f'<p style="color:{color}">{html.escape(message)}</p>' if message else ""
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>PAIN - Sign up</title>
<style>
 body{{font-family:system-ui,sans-serif;background:#f4f4f5;display:flex;min-height:100vh;
      align-items:center;justify-content:center;margin:0}}
 .card{{background:#fff;padding:32px;border-radius:12px;width:340px;box-shadow:0 2px 12px #0001}}
 input{{width:100%;padding:10px;margin:6px 0 14px;border:1px solid #d4d4d8;border-radius:8px;box-sizing:border-box}}
 button{{width:100%;padding:11px;border:0;border-radius:8px;background:#18181b;color:#fff;font-size:15px;cursor:pointer}}
 a{{color:#2563eb}}
</style></head><body><div class="card">
<h2 style="margin-top:0">PAIN &mdash; Create account</h2>{note}
<form method="post" action="/signup">
 <label>Username</label><input name="username" required minlength="3" maxlength="30" autocomplete="username">
 <label>Password</label><input name="password" type="password" required minlength="{MIN_PASSWORD_LEN}" autocomplete="new-password">
 <label>Confirm password</label><input name="confirm" type="password" required autocomplete="new-password">
 <button type="submit">Sign up</button>
</form>
<p style="text-align:center">Already have an account? <a href="/login">Log in</a></p>
</div></body></html>"""

async def signup_page():
    return HTMLResponse(signup_html())

async def signup_submit(request: Request):
    form = await request.form()
    username = str(form.get("username", "")).strip().lower()
    error = create_account(username, str(form.get("password", "")), str(form.get("confirm", "")))
    if error:
        return HTMLResponse(signup_html(error), status_code=400)
    return HTMLResponse(signup_html("Account created! You can now log in.", ok=True))

# Chainlit has a catch-all route, so our routes must be placed BEFORE it.
fastapi_app.router.routes.insert(0, APIRoute("/signup", signup_submit, methods=["POST"], response_class=HTMLResponse))
fastapi_app.router.routes.insert(0, APIRoute("/signup", signup_page, methods=["GET"], response_class=HTMLResponse))

# -----------------------------------------------------------------------------
# LLM Factory
# -----------------------------------------------------------------------------
def get_llm(model_name: str):
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        streaming=True,
        max_retries=2
    )

# -----------------------------------------------------------------------------
# State definition
# -----------------------------------------------------------------------------
class UserState(TypedDict):
    user_message: str
    image_data: list
    documents: list              # ALL active documents in this chat
    new_files: list              # names of files uploaded in THIS turn
    response: str
    model: str
    conversation_history: list   # Shared across ALL models

# -----------------------------------------------------------------------------
# File reading helpers
# -----------------------------------------------------------------------------
def read_pdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(f"[Page {i}]\n{text}")
    return "\n\n".join(pages)

def read_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(parts)

def read_xlsx(path: str) -> str:
    from openpyxl import load_workbook
    wb = load_workbook(path, read_only=True, data_only=True)
    parts = []
    for ws in wb.worksheets:
        parts.append(f"[Sheet: {ws.title}]")
        for row in ws.iter_rows(values_only=True):
            if any(cell is not None for cell in row):
                parts.append(" | ".join("" if c is None else str(c) for c in row))
    return "\n".join(parts)

def read_text(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    if b"\x00" in raw[:4096]:
        raise ValueError("binary file")
    return raw.decode("utf-8", errors="replace")

def extract_file_text(path: str, name: str) -> str:
    """Return the text content of a file, based on its extension."""
    ext = os.path.splitext(name or path)[1].lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".docx":
        return read_docx(path)
    if ext in (".xlsx", ".xlsm"):
        return read_xlsx(path)
    if ext in TEXT_EXTENSIONS:
        return read_text(path)
    # Unknown extension: try as plain text
    return read_text(path)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def extract_text(content):
    """Extract plain text from a LangChain message chunk (str or list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = ""
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")
        return text
    return ""

def format_conversation_history(history: list) -> str:
    """Turn the shared history into a text block the new model can read."""
    if not history:
        return ""
    lines = ["Previous conversation history (across models):"]
    for turn in history:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        model_used = turn.get("model", "unknown")
        lines.append(f"[{model_used}] {role.capitalize()}: {content}")
    lines.append("--- End of history ---")
    return "\n".join(lines)

def format_documents(documents: list) -> str:
    """Turn the uploaded documents into a text block for the model."""
    if not documents:
        return ""
    blocks = ["The user has provided the following document(s):"]
    remaining = MAX_CHARS_TOTAL
    for doc in documents:
        if remaining <= 0:
            blocks.append(f"\n===== FILE: {doc['name']} =====\n[Skipped: total size limit reached]")
            continue
        text = doc["text"][: min(MAX_CHARS_PER_DOC, remaining)]
        truncated = len(doc["text"]) > len(text)
        remaining -= len(text)
        blocks.append(
            f"\n===== FILE: {doc['name']} =====\n{text}"
            + ("\n[... content truncated ...]" if truncated else "")
            + f"\n===== END OF FILE: {doc['name']} ====="
        )
    return "\n".join(blocks)

# -----------------------------------------------------------------------------
# Graph node
# -----------------------------------------------------------------------------
async def assistant_node(state: UserState):
    llm = get_llm(state["model"])

    # 1. Build context from the SHARED history (previous models included)
    history_text = format_conversation_history(state.get("conversation_history", []))

    # 2. Uploaded documents (text extracted from PDF / DOCX / XLSX / TXT / code ...)
    documents = state.get("documents", [])
    documents_text = format_documents(documents)

    # 3. Combine everything into one text prompt
    text_parts = []
    if history_text:
        text_parts.append(history_text)
    if documents_text:
        text_parts.append(documents_text)
    text_parts.append(
        state["user_message"]
        or ("Please describe this image." if state.get("image_data") else "Please summarize the document(s).")
    )
    full_text = "\n\n".join(text_parts)

    images = state.get("image_data", [])
    if images:
        # Multimodal message (text + images)
        human_message = HumanMessage(content=[{"type": "text", "text": full_text}, *images])
    else:
        # Plain string works with every model (including text-only ones)
        human_message = HumanMessage(content=full_text)

    full_response = ""
    async for chunk in llm.astream([human_message]):
        token = extract_text(chunk.content)
        if token:
            full_response += token

    # 4. Update the SHARED history with the current turn
    new_history = list(state.get("conversation_history", []))
    user_entry = state["user_message"]
    if images:
        user_entry += " [Image(s) attached]"
    if state.get("new_files"):
        user_entry += f" [Document(s) attached: {', '.join(state['new_files'])}]"

    new_history.append({
        "role": "user",
        "content": user_entry,
        "model": state["model"]
    })
    new_history.append({
        "role": "assistant",
        "content": full_response,
        "model": state["model"]
    })

    return {
        "response": full_response,
        "conversation_history": new_history
    }

# -----------------------------------------------------------------------------
# LangGraph workflow
# -----------------------------------------------------------------------------
builder = StateGraph(UserState)
builder.add_node("assistant", assistant_node)
builder.add_edge(START, "assistant")
builder.add_edge("assistant", END)
graph = builder.compile()

# -----------------------------------------------------------------------------
# Chainlit handlers
# -----------------------------------------------------------------------------
async def setup_chat_settings():
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="Choose AI Model",
                values=list(AVAILABLE_MODELS.keys()),
                initial_index=0
            )
        ]
    ).send()
    selected_model = AVAILABLE_MODELS.get(settings["Model"], DEFAULT_MODEL)
    cl.user_session.set("selected_model", selected_model)
    return settings

@cl.on_chat_start
async def start():
    await setup_chat_settings()
    cl.user_session.set("conversation_history", [])
    cl.user_session.set("documents", {})

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Called when the user clicks an old chat in the sidebar."""
    settings = await setup_chat_settings()

    saved = load_state(thread["id"])
    if saved:
        history, documents = saved
    else:
        # Fallback for old chats saved before this feature existed
        history, documents = [], {}
        for step in thread.get("steps", []):
            text = step.get("output") or ""
            if not text:
                continue
            if step.get("type") == "user_message":
                history.append({"role": "user", "content": text, "model": "earlier session"})
            elif step.get("type") == "assistant_message":
                history.append({"role": "assistant", "content": text, "model": "earlier session"})

    cl.user_session.set("conversation_history", history)
    cl.user_session.set("documents", documents)

    if documents:
        await cl.Message(
            content="Restored your earlier documents: " + ", ".join(f"**{n}**" for n in documents)
        ).send()

@cl.on_settings_update
async def settings_update(settings):
    selected_model = AVAILABLE_MODELS.get(settings["Model"])
    if not selected_model:
        return

    cl.user_session.set("selected_model", selected_model)
    # NOTE: We intentionally do NOT clear conversation_history here.
    await cl.Message(
        content=f"Model switched to **{settings['Model']}**. Previous chat history is preserved."
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    image_data = []
    new_documents = {}
    skipped_files = []

    # Process uploaded files (images + documents)
    for element in (message.elements or []):
        file_path = getattr(element, "path", None)
        mime_type = getattr(element, "mime", None)
        file_name = getattr(element, "name", None) or (os.path.basename(file_path) if file_path else "file")

        if not file_path:
            continue

        if not mime_type:
            mime_type = mimetypes.guess_type(file_path)[0]

        element_type = getattr(element, "type", None)

        # ---- Images ----
        if element_type == "image" or (mime_type and mime_type.startswith("image/")):
            with open(file_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

            mime_type = mime_type or "image/jpeg"
            image_data.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{encoded_image}"
                }
            })
            continue

        # ---- Documents ----
        try:
            text = await cl.make_async(extract_file_text)(file_path, file_name)
            if text.strip():
                new_documents[file_name] = text
                # keep a copy of the original file on disk
                dest_dir = os.path.join(FILES_DIR, get_thread_id())
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy(file_path, os.path.join(dest_dir, file_name))
            else:
                skipped_files.append(f"{file_name} (no readable text found)")
        except Exception as e:
            skipped_files.append(f"{file_name} ({e})")

    if skipped_files:
        await cl.Message(
            content="Could not read these file(s):\n- " + "\n- ".join(skipped_files)
        ).send()

    # Validate input
    user_message = message.content or ""
    if not user_message.strip() and not image_data and not new_documents:
        await cl.Message(content="Please enter a message or upload an image/document.").send()
        return

    # Get session data
    selected_model = cl.user_session.get("selected_model", DEFAULT_MODEL)
    conversation_history = cl.user_session.get("conversation_history", [])

    # Documents stay active for the whole chat (old ones + newly uploaded)
    session_documents = dict(cl.user_session.get("documents", {}))
    session_documents.update(new_documents)
    cl.user_session.set("documents", session_documents)

    # Initial LangGraph state (pass shared history in)
    initial_state = {
        "user_message": user_message,
        "image_data": image_data,
        "documents": [{"name": n, "text": t} for n, t in session_documents.items()],
        "new_files": list(new_documents.keys()),
        "response": "",
        "model": selected_model,
        "conversation_history": conversation_history
    }

    # Stream response
    stream_msg = cl.Message(content="")
    response_started = False
    final_response = ""
    updated_history = conversation_history

    buffer = ""
    last_flush = time.monotonic()
    CHUNK_SIZE = 25
    FLUSH_INTERVAL = 0.05

    try:
        async for mode, data in graph.astream(initial_state, stream_mode=["messages", "updates"]):
            # Stream LLM tokens
            if mode == "messages":
                chunk, metadata = data
                if metadata.get("langgraph_node") == "assistant":
                    token = extract_text(chunk.content)
                    if token:
                        buffer += token
                        current_time = time.monotonic()

                        if (
                            len(buffer) >= CHUNK_SIZE
                            or current_time - last_flush >= FLUSH_INTERVAL
                            or any(mark in buffer for mark in [".", "?", "!", "\n"])
                        ):
                            if not response_started:
                                await stream_msg.send()
                                response_started = True

                            await stream_msg.stream_token(buffer)
                            buffer = ""
                            last_flush = current_time

            # Capture final node output (including updated history)
            elif mode == "updates":
                if "assistant" in data:
                    node_output = data["assistant"]
                    final_response = node_output.get("response", "")
                    updated_history = node_output.get("conversation_history", updated_history)

        # Flush remaining text
        if buffer:
            if not response_started:
                await stream_msg.send()
                response_started = True
            await stream_msg.stream_token(buffer)

        # Finish streaming
        if response_started:
            await stream_msg.update()
        elif final_response:
            await cl.Message(content=final_response).send()
        else:
            await cl.Message(content="The model did not return a response.").send()

        # Persist the updated shared history so the next model can see it
        cl.user_session.set("conversation_history", updated_history)
        save_state(get_thread_id(), updated_history, session_documents)

    except Exception as e:
        await cl.Message(content=f"Error while generating response:\n\n{str(e)}").send()