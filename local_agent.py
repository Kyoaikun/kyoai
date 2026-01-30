import json
import os
import re
import subprocess
import urllib.request
import sys
import time
import threading
import hashlib
from pathlib import Path
from urllib.parse import urlparse

from openai import OpenAI

BASE_DIR = None  # unrestricted access

client = OpenAI(base_url=os.environ.get("LLAMA_BASE_URL", "http://127.0.0.1:8080/v1"), api_key="local")
MODEL = os.environ.get("LLAMA_MODEL", "qwen2.5-14b-q4km")
USER_ROOT = Path(os.environ.get("LLAMA_USER_ROOT", r"F:\kimi\agent\users"))
MEMORY_PATH = Path(os.environ.get("LLAMA_MEMORY_PATH", r"F:\kimi\agent\memory.txt"))
HISTORY_PATH = Path(os.environ.get("LLAMA_HISTORY_PATH", r"F:\kimi\agent\history.jsonl"))
_DEFAULT_RAG_DIRS = os.environ.get(
    "LLAMA_RAG_DIRS",
    r"F:\kimi\agent;F:\kimi\agent\skills;F:\kimi\projects;F:\kimi\patterns",
).split(";")
RAG_DIRS = list(_DEFAULT_RAG_DIRS)
CHROMA_PATH = Path(os.environ.get("LLAMA_CHROMA_PATH", r"F:\kimi\agent\rag"))
EMBED_MODEL = os.environ.get("LLAMA_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
PATTERN_PATH = Path(os.environ.get("LLAMA_PATTERN_PATH", r"F:\kimi\patterns\index.md"))
SESSIONS_PATH = Path(os.environ.get("LLAMA_SESSIONS_PATH", r"F:\kimi\agent\sessions.json"))
DOWNLOAD_DIR = Path(os.environ.get("LLAMA_DOWNLOAD_DIR", r"F:\kimi\agent\downloads"))
AUTO_SAVE_PATTERN = os.environ.get("LLAMA_AUTO_SAVE_PATTERN", "0") == "1"
PATTERN_MIN_CHARS = int(os.environ.get("LLAMA_PATTERN_MIN_CHARS", "120"))
PATTERN_MIN_SCORE = float(os.environ.get("LLAMA_PATTERN_MIN_SCORE", "0.6"))
PATTERN_REQUIRE_TOOL = os.environ.get("LLAMA_PATTERN_REQUIRE_TOOL", "0") == "1"
PATTERN_KEYWORDS = [
    k.strip().lower()
    for k in os.environ.get(
        "LLAMA_PATTERN_KEYWORDS",
        "fix,build,plan,refactor,debug,write,create",
    ).split(",")
    if k.strip()
]

# simple in-process cache to speed repeat reads in the same session
_CACHE = {}
_CACHE_TTL_SEC = 120

# history lookback for "active learning" (reuse prior steps)
_HISTORY_MAX_BYTES = 1_000_000
_SIMILARITY_MIN = 0.25

# optional vector RAG
_CHROMA = None
_EMBED = None

try:
    import chromadb  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    chromadb = None
    SentenceTransformer = None

try:
    from playwright.sync_api import sync_playwright  # type: ignore
except Exception:
    sync_playwright = None

DENY_PATTERN = re.compile(
    r"\b(rm|del|erase|rmdir|Remove-Item|format|diskpart|shutdown|Restart-Computer|Stop-Process|taskkill|takeown|icacls|cipher)\b",
    re.IGNORECASE,
)


def _resolve_path(p: str) -> Path:
    # expand env vars and ~
    p = os.path.expandvars(p)
    if "C:\\Users\\YourUsername" in p:
        p = p.replace("C:\\Users\\YourUsername", os.environ.get("USERPROFILE", "C:\\Users\\Public"))
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path)
    return path.resolve()


def list_dir(path: str = ".", max_bytes: int | None = None) -> str:
    p = _resolve_path(path)
    cache_key = ("list_dir", str(p))
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    if not p.exists():
        return f"Not found: {p}"
    if not p.is_dir():
        return f"Not a directory: {p}"
    items = []
    for child in p.iterdir():
        items.append({
            "name": child.name,
            "type": "dir" if child.is_dir() else "file",
        })
    result = json.dumps(items, indent=2)
    _cache_set(cache_key, result)
    return result


def read_file(path: str, max_bytes: int = 200_000) -> str:
    p = _resolve_path(path)
    cache_key = ("read_file", str(p), max_bytes)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    if not p.exists() or not p.is_file():
        return f"Not a file: {p}"
    data = p.read_bytes()[:max_bytes]
    result = data.decode("utf-8", errors="replace")
    _cache_set(cache_key, result)
    return result


def read_tail(path: str, lines: int = 20, max_bytes: int = 200_000) -> str:
    p = _resolve_path(path)
    cache_key = ("read_tail", str(p), lines, max_bytes)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    if not p.exists() or not p.is_file():
        return f"Not a file: {p}"
    try:
        size = p.stat().st_size
        read_bytes = min(size, max_bytes)
        with p.open("rb") as f:
            f.seek(-read_bytes, os.SEEK_END)
            data = f.read(read_bytes)
        text = data.decode("utf-8", errors="replace")
        tail_lines = text.splitlines()[-lines:]
        result = "\n".join(tail_lines)
    except Exception:
        result = read_file(str(p), max_bytes=max_bytes)
        result = "\n".join(result.splitlines()[-lines:])
    _cache_set(cache_key, result)
    return result


def write_file(path: str, content: str, append: bool = False) -> str:
    p = _resolve_path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except FileNotFoundError:
        return f"Parent path not found: {p.parent}"
    mode = "ab" if append else "wb"
    with open(p, mode) as f:
        f.write(content.encode("utf-8"))
    # invalidate cached reads for this file
    _CACHE.pop(("read_file", str(p), 200_000), None)
    return f"Wrote {len(content)} bytes to {p}"


def fetch_url(url: str, max_bytes: int = 200_000) -> str:
    cache_key = ("fetch_url", url, max_bytes)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "local-agent/1.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read(max_bytes)
        try:
            result = data.decode("utf-8", errors="replace")
        except Exception:
            result = data.decode("latin-1", errors="replace")
    _cache_set(cache_key, result)
    return result


def download_file(url: str, path: str | None = None, max_bytes: int = 200_000_000) -> str:
    try:
        if not path:
            parsed = urlparse(url)
            name = Path(parsed.path).name or "download.bin"
            path = str(DOWNLOAD_DIR / name)
        p = _resolve_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "local-agent/1.0"})
        total = 0
        with urllib.request.urlopen(req, timeout=60) as resp, open(p, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
                if total >= max_bytes:
                    break
        return f"Downloaded {total} bytes to {p}"
    except Exception as e:
        return f"Download failed: {e}"


def fetch_url_rendered(url: str, max_chars: int = 8000) -> str:
    if sync_playwright is None:
        return "Playwright not available"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            text = page.inner_text("body")
            browser.close()
        if len(text) > max_chars:
            return text[:max_chars] + "\n...[truncated]..."
        return text
    except Exception as e:
        return f"Render failed: {e}"


def run_powershell(command: str, cwd: str = ".") -> str:
    if DENY_PATTERN.search(command):
        return "Blocked by safety policy"
    workdir = _resolve_path(cwd)
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            command,
        ],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = completed.stdout.strip()
    err = completed.stderr.strip()
    if err:
        return f"STDERR:\n{err}\nSTDOUT:\n{out}"
    return out or "(no output)"

def _cache_get(key):
    entry = _CACHE.get(key)
    if not entry:
        return None
    value, ts = entry
    if (time.time() - ts) > _CACHE_TTL_SEC:
        _CACHE.pop(key, None)
        return None
    return value


def _cache_set(key, value):
    _CACHE[key] = (value, time.time())

def set_rag_dirs(dirs: list) -> None:
    global RAG_DIRS
    cleaned = [d.strip() for d in dirs if isinstance(d, str) and d.strip()]
    if not cleaned:
        RAG_DIRS = list(_DEFAULT_RAG_DIRS)
        return
    # preserve order, drop duplicates
    seen = set()
    unique = []
    for d in cleaned:
        if d not in seen:
            unique.append(d)
            seen.add(d)
    RAG_DIRS = unique

def set_user_paths(user_root: str) -> None:
    global MEMORY_PATH, HISTORY_PATH, SESSIONS_PATH, CHROMA_PATH
    root = Path(user_root)
    MEMORY_PATH = root / "memory.txt"
    HISTORY_PATH = root / "history.jsonl"
    SESSIONS_PATH = root / "sessions.json"
    CHROMA_PATH = root / "rag"

def _has_rg() -> bool:
    try:
        subprocess.run(["rg", "--version"], capture_output=True, text=True, timeout=2)
        return True
    except Exception:
        return False


def search_text(query: str, max_results: int = 8) -> str:
    # try vector RAG first
    if chromadb and SentenceTransformer:
        try:
            global _CHROMA, _EMBED
            if _CHROMA is None:
                _CHROMA = chromadb.PersistentClient(path=str(CHROMA_PATH))
            if _EMBED is None:
                _EMBED = SentenceTransformer(EMBED_MODEL)
            col = _CHROMA.get_or_create_collection(name="rag")
            qemb = _EMBED.encode([query]).tolist()
            res = col.query(query_embeddings=qemb, n_results=max_results)
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            if docs:
                lines = []
                for d, m in zip(docs, metas):
                    src = m.get("path", "unknown") if isinstance(m, dict) else "unknown"
                    snippet = (d[:400] + "...") if len(d) > 400 else d
                    lines.append(f"{src}: {snippet}")
                return "\n".join(lines)[:2000]
        except Exception:
            pass

    terms = [t for t in re.split(r"[^a-zA-Z0-9]+", query.lower()) if len(t) >= 4]
    if not terms:
        return ""
    pattern = "|".join(re.escape(t) for t in terms[:4])
    results = []
    scored = []
    if _has_rg():
        for d in RAG_DIRS:
            d = d.strip()
            if not d:
                continue
            try:
                cp = subprocess.run(
                    ["rg", "-i", "-n", "--max-count", str(max_results), pattern, d],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if cp.stdout:
                    results.extend(cp.stdout.splitlines())
            except Exception:
                pass
    else:
        # fallback: powershell Select-String
        for d in RAG_DIRS:
            d = d.strip()
            if not d:
                continue
            try:
                cmd = f"Get-ChildItem -Path \"{d}\" -Recurse -File | Select-String -Pattern '{pattern}' -CaseSensitive:$false -List | Select-Object -First {max_results}"
                cp = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", cmd],
                    capture_output=True,
                    text=True,
                    timeout=8,
                )
                if cp.stdout:
                    results.extend(cp.stdout.splitlines())
            except Exception:
                pass
    if not results:
        return ""
    terms_l = terms[:4]
    for line in results:
        low = line.lower()
        score = sum(1 for t in terms_l if t in low)
        if score <= 0:
            continue
        scored.append((score, line))
    if not scored:
        return ""
    scored.sort(key=lambda x: x[0], reverse=True)
    snippet = "\n".join(line for _, line in scored[:max_results])
    return snippet[:2000]


def rag_index(max_files: int = 2000, max_bytes: int = 500_000) -> str:
    if not chromadb or not SentenceTransformer:
        return "RAG dependencies not available"
    try:
        global _CHROMA, _EMBED
        if _CHROMA is None:
            _CHROMA = chromadb.PersistentClient(path=str(CHROMA_PATH))
        if _EMBED is None:
            _EMBED = SentenceTransformer(EMBED_MODEL)
        col = _CHROMA.get_or_create_collection(name="rag")

        exts = {".txt", ".md", ".json", ".yaml", ".yml", ".py", ".ps1", ".toml", ".ini", ".cfg"}
        docs = []
        ids = []
        metas = []
        count = 0
        for root in RAG_DIRS:
            root = root.strip()
            if not root:
                continue
            for path in Path(root).rglob("*"):
                if count >= max_files:
                    break
                if not path.is_file():
                    continue
                if path.suffix.lower() not in exts:
                    continue
                try:
                    if path.stat().st_size > max_bytes:
                        continue
                except Exception:
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                if not text.strip():
                    continue
                doc_id = f"{path}:{path.stat().st_mtime_ns}"
                docs.append(text[:2000])
                ids.append(doc_id)
                metas.append({"path": str(path)})
                count += 1
            if count >= max_files:
                break
        if not docs:
            return "No documents indexed"
        existing = set()
        try:
            data = col.get(include=["ids"])
            existing = set(data.get("ids", []))
        except Exception:
            existing = set()
        f_docs, f_ids, f_metas = [], [], []
        for d, i, m in zip(docs, ids, metas):
            if i in existing:
                continue
            f_docs.append(d)
            f_ids.append(i)
            f_metas.append(m)
        if not f_docs:
            return "No new documents to index"
        embs = _EMBED.encode(f_docs).tolist()
        col.add(documents=f_docs, embeddings=embs, ids=f_ids, metadatas=f_metas)
        return f"Indexed {len(f_docs)} documents"
    except Exception as e:
        return f"Index failed: {e}"


tools = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and folders on the local machine",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_bytes": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file on the local machine",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_bytes": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_tail",
            "description": "Read the last N lines of a text file on the local machine",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "lines": {"type": "integer"},
                    "max_bytes": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a text file on the local machine",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "append": {"type": "boolean"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_powershell",
            "description": "Run a PowerShell command on the local machine",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch a URL and return the first chunk of text",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_bytes": {"type": "integer"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url_rendered",
            "description": "Fetch a URL using a headless browser and return visible text",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_chars": {"type": "integer"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "download_file",
            "description": "Download a file from a URL to disk",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "path": {"type": "string"},
                    "max_bytes": {"type": "integer"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": "Search local files for text and return matching lines",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_index",
            "description": "Build a local vector index for faster retrieval",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_files": {"type": "integer"},
                    "max_bytes": {"type": "integer"},
                },
            },
        },
    },
]


def call_model(messages, tool_choice="auto"):
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=0.2,
        timeout=120,
    )

def _run_with_timer(label, func, *args, **kwargs):
    stop = threading.Event()
    start = time.time()

    def loop():
        while not stop.is_set():
            elapsed = int(time.time() - start)
            sys.stdout.write(f"\r[{label}] {elapsed}s ...")
            sys.stdout.flush()
            time.sleep(1)
        sys.stdout.write("\r")
        sys.stdout.flush()

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    try:
        return func(*args, **kwargs)
    except KeyboardInterrupt:
        stop.set()
        t.join(timeout=1)
        print(f"\n[{label}] cancelled")
        raise
    finally:
        stop.set()
        t.join(timeout=1)
        elapsed_ms = int((time.time() - start) * 1000)
        print(f"[{label}] {elapsed_ms} ms")

def _load_recent_history() -> list:
    if not HISTORY_PATH.exists():
        return []
    try:
        data = HISTORY_PATH.read_bytes()
        if len(data) > _HISTORY_MAX_BYTES:
            data = data[-_HISTORY_MAX_BYTES:]
        lines = data.decode("utf-8", errors="ignore").splitlines()
        records = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                pass
        return records[-200:]
    except Exception:
        return []


def _score_similarity(a: str, b: str) -> float:
    def tok(s):
        return {t for t in re.split(r"[^a-zA-Z0-9]+", s.lower()) if t}
    ta = tok(a)
    tb = tok(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def _needs_tools(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    if re.search(r"\b(read|write|edit|create|delete|list|show|open|tail|run|execute|powershell|cmd|shell|directory|folder|file)\b", t):
        return True
    if re.search(r"[a-zA-Z]:\\\\", text):
        return True
    if "path" in t:
        return True
    return False


def _looks_non_english(text: str) -> bool:
    if not text:
        return False
    # Allow basic ASCII punctuation; flag if there are non-ASCII letters.
    if re.search(r"[^\x00-\x7F]", text):
        return True
    return False


def _needs_plan(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    if any(k in t for k in ["plan", "steps", "step-by-step", "checklist", "roadmap"]):
        return True
    if re.search(r"\b(1\\.|1\\)|2\\.|2\\)|3\\.|3\\))", t):
        return True
    if any(k in t for k in ["first", "then", "after that", "next", "finally"]):
        return True
    if "\n" in text and text.count("\n") >= 2:
        return True
    return False

def _select_skill_prompt(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    if any(k in t for k in ["bug", "error", "trace", "exception", "stack trace", "crash", "failed"]):
        return "Use the Bug Fix skill: identify root cause, apply minimal fix, verify."
    if any(k in t for k in ["edit", "modify", "change", "update", "refactor", "add code", "implement"]):
        return "Use the Code Edit skill: minimal change, verify by reading back."
    if any(k in t for k in ["scan", "search", "where is", "find", "locate", "repo", "project structure"]):
        return "Use the Repo Scan skill: list, search, read relevant sections, summarize."
    if any(k in t for k in ["test", "build", "run", "compile", "benchmark"]):
        return "Use the Test Run skill: confirm command, run, report key output."
    return ""


def _find_similar_task(user_input: str, records: list) -> str:
    best = ("", 0.0, None)
    for r in records:
        prev = r.get("user") or ""
        score = _score_similarity(user_input, prev)
        if score > best[1]:
            best = (prev, score, r)
    if best[1] < _SIMILARITY_MIN or best[2] is None:
        return ""
    tool_trace = best[2].get("tools") or []
    steps = []
    for t in tool_trace:
        tool = t.get("tool")
        args = t.get("args")
        if tool:
            steps.append(f"- {tool} {args}")
        if len(steps) >= 6:
            break
    if not steps:
        return ""
    hint = "Previous similar task steps:\n" + "\n".join(steps)
    return hint[:600]


def _trim_messages(msgs, max_keep=12):
    if len(msgs) <= max_keep:
        return msgs
    return [msgs[0]] + msgs[-(max_keep - 1):]

def _load_sessions():
    if not SESSIONS_PATH.exists():
        return {"current": "default", "sessions": {"default": []}}
    try:
        data = json.loads(SESSIONS_PATH.read_text(encoding="utf-8"))
        if "current" not in data or "sessions" not in data:
            return {"current": "default", "sessions": {"default": []}}
        if data["current"] not in data["sessions"]:
            data["sessions"][data["current"]] = []
        return data
    except Exception:
        return {"current": "default", "sessions": {"default": []}}


def _save_sessions(state):
    SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SESSIONS_PATH.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def _session_messages(state):
    name = state.get("current", "default")
    return state["sessions"].get(name, [])


def _set_session_messages(state, msgs):
    name = state.get("current", "default")
    # keep last 50 turns to avoid huge files
    state["sessions"][name] = msgs[-50:]
    _save_sessions(state)

def load_memory() -> str:
    if MEMORY_PATH.exists():
        try:
            text = MEMORY_PATH.read_text(encoding="utf-8")
            return text[-2000:]
        except Exception:
            return ""
    return ""

def _filter_memory_text(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    keep = []
    mode = None
    for line in lines:
        raw = line.strip()
        low = raw.lower()
        if not raw:
            continue
        if "user preferences" in low:
            mode = "prefs"
            keep.append(line)
            continue
        if "ongoing tasks" in low:
            mode = "tasks"
            keep.append(line)
            continue
        if "important facts" in low:
            mode = "facts"
            keep.append(line)
            continue
        if mode in {"prefs", "tasks", "facts"}:
            if raw.startswith("-") or raw.startswith("*"):
                keep.append(line)
    return "\n".join(keep).strip()

def save_memory(text: str) -> None:
    filtered = _filter_memory_text(text)
    if not filtered:
        return
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_PATH.write_text(filtered, encoding="utf-8")

def _append_pattern_entry(user_input: str, tool_trace: list, assistant_text: str) -> None:
    _append_pattern_entry_structured(
        user_input=user_input,
        tool_trace=tool_trace,
        assistant_text=assistant_text,
        source="auto",
        require_best=True,
    )


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text


def _pattern_score(user_input: str, tool_trace: list, assistant_text: str) -> tuple:
    score = 0.0
    reasons = []
    if tool_trace:
        score += 0.6
        reasons.append("tool-trace")
    if assistant_text and len(assistant_text.strip()) >= PATTERN_MIN_CHARS:
        score += 0.2
        reasons.append("length")
    if PATTERN_KEYWORDS:
        text = f"{user_input} {assistant_text}".lower()
        hits = [k for k in PATTERN_KEYWORDS if k in text]
        if hits:
            score += 0.2
            reasons.append("keywords:" + ",".join(hits[:5]))
    return score, reasons


def _pattern_hash(user_input: str, body: str) -> str:
    h = hashlib.sha256()
    h.update(_normalize_text(user_input).encode("utf-8"))
    h.update(b"\n")
    h.update(_normalize_text(body).encode("utf-8"))
    return h.hexdigest()


def _pattern_seen(hash_value: str) -> bool:
    if not PATTERN_PATH.exists():
        return False
    try:
        text = PATTERN_PATH.read_text(encoding="utf-8", errors="ignore")
        return f"Hash: {hash_value}" in text
    except Exception:
        return False


def _append_pattern_entry_structured(
    user_input: str,
    tool_trace: list,
    assistant_text: str,
    source: str = "auto",
    require_best: bool = False,
    require_tool: bool | None = None,
) -> None:
    if user_input.strip().startswith("/"):
        return
    if not assistant_text.strip():
        return
    if assistant_text.strip().lower().startswith("error"):
        return
    try:
        body = ""
        if tool_trace:
            lines = []
            for t in tool_trace[:8]:
                tool = t.get("tool")
                args = t.get("args")
                lines.append(f"- {tool} {args}")
            if lines:
                body = "Steps:\n" + "\n".join(lines)
        if not body:
            body = "Response:\n" + assistant_text.strip()[:1200]

        score, reasons = _pattern_score(user_input, tool_trace, assistant_text)
        if require_tool is None:
            require_tool = PATTERN_REQUIRE_TOOL
        if require_tool and not tool_trace:
            return
        if require_best and score < PATTERN_MIN_SCORE:
            return

        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        sig = _pattern_hash(user_input, body)
        if _pattern_seen(sig):
            return

        PATTERN_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PATTERN_PATH.open("a", encoding="utf-8") as f:
            f.write("\n## Auto Pattern\n")
            f.write(f"Time: {stamp}\n")
            f.write(f"Source: {source}\n")
            f.write(f"Score: {score:.2f}\n")
            if reasons:
                f.write("Criteria: " + ", ".join(reasons) + "\n")
            f.write(f"User: {user_input.strip()}\n")
            f.write(body + "\n")
            f.write(f"Hash: {sig}\n")
    except Exception:
        pass


def main():
    system = (
        "You are a local assistant. You can access tools to operate on the local machine. "
        "When the user asks you to create or modify files, do it directly using tools instead of just describing steps. "
        "If a task requires reading or writing files or running commands, you MUST call the appropriate tools; otherwise answer directly without tools. "
        "Be concise. Never run destructive commands. Respond in English only and do not use any other language. "
        "If asked for the last N lines of a file, read only the tail and output only those N lines. "
        "After any file write or edit, verify by listing the directory and reading back the file."
    )
    memory = load_memory()
    if memory:
        system = system + " Persistent memory:\n" + memory
    if PATTERN_PATH.exists():
        try:
            patterns = PATTERN_PATH.read_text(encoding="utf-8", errors="ignore")
            if patterns.strip():
                system = system + "\nPattern library:\n" + patterns[-4000:]
        except Exception:
            pass
    sessions = _load_sessions()
    messages = [{"role": "system", "content": system}]
    messages.extend(_session_messages(sessions))

    print("Local agent ready. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("> ").strip()
        except EOFError:
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        if user_input.startswith("/list"):
            names = sorted(sessions.get("sessions", {}).keys())
            print("Sessions: " + ", ".join(names))
            print("Current: " + sessions.get("current", "default"))
            continue
        if user_input.startswith("/new"):
            name = user_input.split(" ", 1)[1].strip() if " " in user_input else ""
            if not name:
                name = f"session_{int(time.time())}"
            sessions["sessions"][name] = []
            sessions["current"] = name
            _save_sessions(sessions)
            messages = [{"role": "system", "content": system}]
            print(f"New session: {name}")
            continue
        if user_input.startswith("/project"):
            name = user_input.split(" ", 1)[1].strip() if " " in user_input else ""
            if not name:
                print("Usage: /project <name>")
                continue
            pname = f"project:{name}"
            if pname not in sessions.get("sessions", {}):
                sessions["sessions"][pname] = []
            sessions["current"] = pname
            _save_sessions(sessions)
            messages = [{"role": "system", "content": system}]
            messages.extend(_session_messages(sessions))
            print(f"Project session: {pname}")
            continue
        if user_input.startswith("/switch"):
            name = user_input.split(" ", 1)[1].strip() if " " in user_input else ""
            if not name or name not in sessions.get("sessions", {}):
                print("Unknown session. Use /list to see sessions.")
                continue
            sessions["current"] = name
            _save_sessions(sessions)
            messages = [{"role": "system", "content": system}]
            messages.extend(_session_messages(sessions))
            print(f"Switched to: {name}")
            continue

        # inject a brief hint from similar past tasks
        history = _load_recent_history()
        hint = _find_similar_task(user_input, history)
        if hint:
            messages.append({"role": "system", "content": hint})

        # lightweight RAG: search local files and add small context
        rag = search_text(user_input)
        if rag:
            messages.append({"role": "system", "content": "Local search results:\n" + rag})

        messages.append({"role": "user", "content": user_input})
        messages = _trim_messages(messages, max_keep=12)
        plan_msg = None
        skill_msg = None
        if _needs_plan(user_input):
            plan_msg = {
                "role": "system",
                "content": "If the task is complex or multi-step, first write a brief plan (2-6 bullets), then execute.",
            }
            messages.append(plan_msg)
        skill_prompt = _select_skill_prompt(user_input)
        if skill_prompt:
            skill_msg = {"role": "system", "content": skill_prompt}
            messages.append(skill_msg)
        try:
            needs_tools = _needs_tools(user_input)
            tool_choice = "required" if needs_tools else "auto"
            response = _run_with_timer("model", call_model, messages, tool_choice=tool_choice)
            msg = response.choices[0].message

            if needs_tools and not msg.tool_calls:
                messages.append({
                    "role": "system",
                    "content": "Tool enforcement: you MUST call tools to complete this request. Return only tool calls.",
                })
                response = _run_with_timer("model", call_model, messages, tool_choice="required")
                msg = response.choices[0].message
        except KeyboardInterrupt:
            print("Cancelled.")
            continue
        except Exception as e:
            print(f"Model error: {e}")
            # reset context to avoid repeated failures
            messages = [messages[0]]
            continue

        tool_trace = []
        while msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                try:
                    if name == "list_dir":
                        result = list_dir(**args)
                    elif name == "read_file":
                        result = read_file(**args)
                    elif name == "read_tail":
                        result = read_tail(**args)
                    elif name == "write_file":
                        result = write_file(**args)
                    elif name == "run_powershell":
                        result = run_powershell(**args)
                    elif name == "fetch_url":
                        result = fetch_url(**args)
                    elif name == "fetch_url_rendered":
                        result = fetch_url_rendered(**args)
                    elif name == "search_text":
                        result = search_text(**args)
                    elif name == "rag_index":
                        result = rag_index(**args)
                    else:
                        result = "Unknown tool"
                except KeyboardInterrupt:
                    result = "Cancelled by user"
                except Exception as e:
                    result = f"Tool error: {e}"

                if isinstance(result, str) and len(result) > 4000:
                    result = result[:4000] + "\n...[truncated]..."
                # tool timings are printed at the end of model call

                tool_trace.append({
                    "tool": name,
                    "args": args,
                    "result_preview": result[:500] if isinstance(result, str) else str(result)[:500],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": result,
                })
            try:
                response = _run_with_timer("model", call_model, messages)
                msg = response.choices[0].message
            except KeyboardInterrupt:
                print("Cancelled.")
                break
            except Exception as e:
                print(f"Model error: {e}")
                break

        assistant_text = msg.content or ""
        if _needs_tools(user_input) and _looks_non_english(assistant_text):
            messages.append({
                "role": "system",
                "content": "Respond in English only. Do not use any other language.",
            })
            messages.append({
                "role": "user",
                "content": "Provide the final answer in English only. Keep it concise.",
            })
            try:
                response = _run_with_timer("model", call_model, messages, tool_choice="auto")
                msg = response.choices[0].message
                assistant_text = msg.content or assistant_text
            except Exception:
                pass
        if plan_msg is not None and plan_msg in messages:
            try:
                messages.remove(plan_msg)
            except ValueError:
                pass
        if skill_msg is not None and skill_msg in messages:
            try:
                messages.remove(skill_msg)
            except ValueError:
                pass
        print(assistant_text)
        messages.append({"role": "assistant", "content": assistant_text})
        # persist session messages (exclude system)
        _set_session_messages(sessions, messages[1:])

        # append to history for persistence across restarts
        try:
            HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "user": user_input,
                "assistant": assistant_text,
                "tools": tool_trace,
            }
            with HISTORY_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass
        if AUTO_SAVE_PATTERN:
            _append_pattern_entry(user_input, tool_trace, assistant_text)
        # Save a short rolling memory summary (bounded to avoid ctx overflow)
        try:
            mem_input = "\n".join(
                m["content"] for m in messages[-10:] if isinstance(m.get("content"), str)
            )
            if len(mem_input) > 3000:
                mem_input = mem_input[-3000:]
            mem_resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Summarize key user preferences, ongoing tasks, and important facts in <=200 words. Be concise."},
                    {"role": "user", "content": mem_input},
                ],
                temperature=0.1,
            )
            mem_text = (mem_resp.choices[0].message.content or "").strip()
            if mem_text:
                save_memory(mem_text)
        except Exception:
            pass


if __name__ == "__main__":
    main()
