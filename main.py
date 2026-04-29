"""
Claude Code Web Chat – FastAPI backend
Streams claude CLI output over WebSocket, one session at a time.
"""

import asyncio
import json
import logging
import os
import re
import signal
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ANSI escape-code stripper
# ---------------------------------------------------------------------------
_ANSI_RE = re.compile(
    r"""
    \x1b   # ESC
    (?:
        [@-Z\\-_]          # Fe sequences (single char after ESC)
      | \[                 # CSI …
        [0-?]*             #   parameter bytes
        [ -/]*             #   intermediate bytes
        [@-~]              #   final byte
      | \]                 # OSC …
        [^\x07\x1b]*       #   payload (stop at BEL or ESC)
        (?:\x07|\x1b\\)    #   ST: BEL or ESC-backslash
    )
    """,
    re.VERBOSE,
)


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# ---------------------------------------------------------------------------
# Session state (single global session – rate-limit: 1 concurrent)
# ---------------------------------------------------------------------------
class Session:
    def __init__(self):
        self.process: asyncio.subprocess.Process | None = None
        self.lock = asyncio.Lock()
        self.active = False

    async def kill(self):
        """Terminate the running subprocess gracefully, then forcefully."""
        if self.process and self.process.returncode is None:
            log.info("Killing claude subprocess pid=%s", self.process.pid)
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
            except ProcessLookupError:
                pass
        self.process = None
        self.active = False


session = Session()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Claude Code Web Chat")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/status")
async def status():
    return {"active": session.active}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("WebSocket connected from %s", ws.client)

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await _send(ws, "error", "Invalid JSON message")
                continue

            kind = msg.get("type", "")

            if kind == "run":
                await _handle_run(ws, msg)
            elif kind == "stop":
                await _handle_stop(ws)
            elif kind == "ping":
                await _send(ws, "pong", "")
            else:
                await _send(ws, "error", f"Unknown message type: {kind!r}")

    except WebSocketDisconnect:
        log.info("WebSocket disconnected – cleaning up session")
        await session.kill()
    except Exception as exc:
        log.exception("Unexpected WS error: %s", exc)
        try:
            await _send(ws, "error", f"Server error: {exc}")
        except Exception:
            pass
        await session.kill()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _send(ws: WebSocket, kind: str, data: str):
    """Send a typed JSON frame; silently drop if socket is closed."""
    try:
        await ws.send_text(json.dumps({"type": kind, "data": data}))
    except Exception:
        pass


async def _handle_stop(ws: WebSocket):
    await session.kill()
    await _send(ws, "status", "stopped")
    await _send(ws, "system", "Session stopped.")


async def _handle_run(ws: WebSocket, msg: dict):
    prompt = (msg.get("prompt") or "").strip()
    work_dir = (msg.get("workdir") or os.path.expanduser("~")).strip()

    if not prompt:
        await _send(ws, "error", "Empty prompt.")
        return

    # Rate-limit: only one session at a time
    if session.active:
        await _send(ws, "error", "A session is already running. Send /stop first.")
        return

    # Validate working directory
    if not os.path.isdir(work_dir):
        await _send(ws, "error", f"Working directory does not exist: {work_dir}")
        return

    async with session.lock:
        if session.active:          # double-check after acquiring lock
            await _send(ws, "error", "A session is already running.")
            return
        session.active = True

    await _send(ws, "status", "running")
    await _send(ws, "system", f"Starting Claude Code in {work_dir} …")

    cmd = [
        "claude",
        "--dangerously-skip-permissions",
        "-p", prompt,
        "--output-format", "stream-json",
        "--verbose",
    ]

    log.info("Launching: %s  cwd=%s", " ".join(cmd), work_dir)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=work_dir,
            # Give the child process its own process group so we can kill it cleanly
            start_new_session=True,
        )
        session.process = proc

        # Stream stdout and stderr concurrently
        await asyncio.gather(
            _stream_pipe(ws, proc.stdout, "stdout"),
            _stream_pipe(ws, proc.stderr, "stderr"),
        )

        await proc.wait()
        rc = proc.returncode
        log.info("claude exited with code %s", rc)

        if rc == 0:
            await _send(ws, "system", "Claude Code finished.")
        else:
            await _send(ws, "system", f"Claude Code exited with code {rc}.")

    except FileNotFoundError:
        await _send(ws, "error", "'claude' binary not found – is it installed and on PATH?")
    except Exception as exc:
        log.exception("Error running subprocess: %s", exc)
        await _send(ws, "error", f"Subprocess error: {exc}")
    finally:
        session.active = False
        session.process = None
        await _send(ws, "status", "stopped")


async def _stream_pipe(ws: WebSocket, pipe: asyncio.StreamReader, pipe_name: str):
    """
    Read lines from a subprocess pipe.

    Claude's --output-format stream-json emits newline-delimited JSON objects
    on stdout.  We parse those and forward only the text content so the
    frontend stays clean.  Raw stderr lines are forwarded as-is (after ANSI
    stripping) for debugging.
    """
    # Track which message IDs already had streaming deltas — skip their full-message event.
    msg_streamed: set[str] = set()

    async for raw_line in pipe:
        line = raw_line.decode("utf-8", errors="replace")

        if pipe_name == "stderr":
            cleaned = strip_ansi(line).rstrip("\n")
            if cleaned:
                await _send(ws, "stderr", cleaned)
            continue

        # stdout: attempt stream-json parse
        line_stripped = line.strip()
        if not line_stripped:
            continue

        text, msg_id, is_delta = _extract_text_from_stream_json(line_stripped)
        if text is not None:
            if is_delta:
                if msg_id:
                    msg_streamed.add(msg_id)
                await _send(ws, "assistant", text)
            elif not msg_id or msg_id not in msg_streamed:
                # Full-message fallback: only emit when streaming didn't already cover it
                await _send(ws, "assistant", text)
        else:
            # Fallback: send raw text after ANSI strip (handles non-JSON output)
            cleaned = strip_ansi(line_stripped)
            if cleaned:
                await _send(ws, "stdout", cleaned)


def _extract_text_from_stream_json(line: str) -> tuple[str | None, str | None, bool]:
    """
    Parse one line of Claude's stream-json output.

    Returns (text, msg_id, is_delta):
      text     – displayable text, or None
      msg_id   – message id for dedup tracking, or None
      is_delta – True if this is a streaming content_block_delta
    """
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None, None, False

    t = obj.get("type", "")

    if t == "text":
        return strip_ansi(obj.get("text", "")), None, False

    if t == "content_block_delta":
        delta = obj.get("delta", {})
        if delta.get("type") == "text_delta":
            text = strip_ansi(delta.get("text", ""))
            # msg_id lives at the top-level "message" key in some versions
            mid = obj.get("message", {}).get("id") if isinstance(obj.get("message"), dict) else None
            return text, mid, True

    # Full assistant message — used when streaming deltas were not emitted
    if t == "assistant":
        msg = obj.get("message", {})
        mid = msg.get("id")
        parts = [
            block.get("text", "")
            for block in msg.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        if parts:
            return strip_ansi("".join(parts)), mid, False

    # Skip "result" — always duplicates the assistant message text.
    return None, None, False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
        # No reload in production
    )
