"""Shared test harness for py-ai.py.

Everything here is dependency-light on purpose: a scriptable fake
OpenAI-compatible streaming endpoint, a temp-directory workspace, and
factories for Agents / Textual stub agents. No pytest required (though
the `test_*` naming means pytest can collect these modules too).

Run a module directly:      python3 tests/test_tools_policy.py
Run everything:             python3 tests/run_all.py
"""

import contextlib
import json
import os
import sys
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# py-ai.py lives one directory up from tests/
ROOT = Path(__file__).resolve().parent.parent
AGENT_PATH = ROOT / "py-ai.py"
sys.path.insert(0, str(ROOT))

# "py-ai" is not a valid module name (hyphens are not identifiers), so the
# single-file agent is loaded from its path and registered under an
# importable alias. Tests keep using `A`, and A.__file__ still points at
# py-ai.py for the source-scanning tests.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("py_ai", AGENT_PATH)
A = importlib.util.module_from_spec(_spec)
sys.modules["py_ai"] = A
_spec.loader.exec_module(A)


# --------------------------------------------------------------------------- #
# Terminal colour
# --------------------------------------------------------------------------- #
def _colour_enabled() -> bool:
    """Colour when writing to a terminal, or when run_all.py forces it
    (it captures child output through a pipe, so isatty() is False there).
    Honours the NO_COLOR convention."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("PRN_TEST_COLOUR") == "1":
        return True
    return sys.stdout.isatty()


COLOUR = _colour_enabled()


def paint(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLOUR else text


def green(text: str) -> str:
    return paint(text, "1;32")


def red(text: str) -> str:
    return paint(text, "1;31")


def yellow(text: str) -> str:
    return paint(text, "1;33")


def dim(text: str) -> str:
    return paint(text, "2")


# --------------------------------------------------------------------------- #
# SSE chunk builders
# --------------------------------------------------------------------------- #
def delta(payload: dict, finish=None, timings=None) -> dict:
    """One OpenAI streaming chunk. `payload` is the `delta` object, e.g.
    {"content": "hi"}, {"reasoning_content": "..."} or {"tool_calls": [...]}."""
    chunk = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "delta": payload, "finish_reason": finish}],
    }
    if timings:  # llama.cpp extension (prefill/cache stats)
        chunk["timings"] = timings
    return chunk


def tool_call_chunk(name: str, arguments: dict, call_id="t1") -> dict:
    return delta(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": call_id,
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ]
        },
        finish="tool_calls",
    )


def answer(text: str, reasoning=None, finish="stop", timings=None) -> list:
    """A complete scripted response: optional reasoning then visible text."""
    chunks = []
    if reasoning:
        chunks.append(delta({"reasoning_content": reasoning}))
    chunks.append(delta({"content": text}, finish=finish, timings=timings))
    return chunks


def risk_json(level: str, reason: str = "because") -> list:
    return [delta({"content": json.dumps({"level": level, "reason": reason})},
                  finish="stop")]


# --------------------------------------------------------------------------- #
# Fake server
# --------------------------------------------------------------------------- #
class FakeServer:
    """Scriptable OpenAI-compatible /chat/completions streaming endpoint.

    server.push(*chunks) queues ONE response (a list of chunk dicts).
    Responses are consumed in order; running out raises in the handler,
    which surfaces as a client error -- a useful signal that the code
    under test made more requests than expected.
    """

    def __init__(self):
        self.script: list = []
        self.requests: list = []
        self.headers: list = []
        self.models: list = ["model-a", "model-b"]
        self.n_ctx: int = 16384
        self.sampler: dict = {"temperature": 0.6, "top_k": 40}
        self.chat_template: str = ""   # /props chat_template, for discovery
        harness = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                """Serves the endpoints py-ai.py probes at startup."""
                if "models" in self.path:
                    payload = {"data": [{"id": name} for name in harness.models]}
                else:  # llama.cpp /props
                    payload = {
                        "default_generation_settings": {
                            "n_ctx": harness.n_ctx,
                            "params": dict(harness.sampler),
                        },
                        "chat_template": harness.chat_template,
                    }
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                harness.headers.append({k.lower(): v for k, v in self.headers.items()})
                harness.requests.append(json.loads(self.rfile.read(length) or "{}"))
                if not harness.script:
                    self.send_response(500)
                    self.end_headers()
                    return
                scripted = harness.script.pop(0)
                if isinstance(scripted, dict) and "status" in scripted:
                    body = scripted["body"].encode()
                    self.send_response(scripted["status"])
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                chunks = scripted
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in chunks:
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def push(self, *chunks) -> "FakeServer":
        self.script.append(list(chunks))
        return self

    def push_error(self, status: int, body: str) -> "FakeServer":
        """Queue an HTTP error response (recovery-layer tests)."""
        self.script.append({"status": status, "body": body})
        return self

    def push_many(self, chunks: list, times: int) -> "FakeServer":
        for _ in range(times):
            self.script.append(list(chunks))
        return self

    @property
    def last_request(self) -> dict:
        return self.requests[-1] if self.requests else {}

    @property
    def count(self) -> int:
        return len(self.requests)

    def reset(self) -> None:
        self.script.clear()
        self.requests.clear()
        self.headers.clear()

    def client(self, model="test-model"):
        return A.HttpxChatClient(self.base_url, "k", model, timeout=10)

    def stop(self) -> None:
        self._server.shutdown()


class NoClient:
    """A client that must never be called: any request is a test failure.
    Used to prove a code path is model-free (heuristic risk, dedup, ...)."""

    model = "no-model"
    url = "http://127.0.0.1:0/v1"
    name = "none"

    def stream_chat(self, *args, **kwargs):
        raise AssertionError("the model was called but should not have been")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def workspace():
    """Temp working directory (chdir'd into) so path confinement, file
    tools, sessions and .skills/.memories all operate on scratch space."""
    previous = os.getcwd()
    directory = tempfile.mkdtemp(prefix="agenttest-")
    os.chdir(directory)
    try:
        yield Path(directory)
    finally:
        os.chdir(previous)


@contextlib.contextmanager
def recorded_console(width=120):
    """Capture everything py-ai.py prints. Yields a getter; call it ONCE
    (rich's export_text clears the buffer)."""
    from rich.console import Console

    previous = A.console
    console = Console(record=True, width=width, force_terminal=False)
    A.console = console
    try:
        yield lambda: console.export_text()
    finally:
        A.console = previous


@contextlib.contextmanager
def approval(*answers):
    """Scripted approval answers; also records how many times we were asked."""
    asked = []
    replies = iter(answers)

    def hook(prompt):
        asked.append(prompt)
        return next(replies, "y")

    previous = A.APPROVAL_HOOK
    A.APPROVAL_HOOK = hook
    try:
        yield asked
    finally:
        A.APPROVAL_HOOK = previous


@contextlib.contextmanager
def log_sink():
    """Collect (line, style) tuples emitted by the logging pipeline."""
    captured = []
    previous = A.LOG_SINK
    A.LOG_SINK = lambda line, style: captured.append((line, style))
    try:
        yield captured
    finally:
        A.LOG_SINK = previous


def make_agent(server=None, **kwargs) -> "A.Agent":
    """An Agent with test-friendly defaults. Pass server=None to get a
    NoClient (model calls become failures)."""
    client = server.client() if server is not None else NoClient()
    options = dict(
        protocol="native",
        retries=1,
        max_nudges=1,
        policy=A.PolicyEngine(root=os.getcwd()),
        tracker=A.TokenTracker(32768, "test-model"),
    )
    options.update(kwargs)
    messages = options.pop("messages", None)
    if messages is not None:  # drive run() with a scripted user turn list
        pending = iter(list(messages) + [None])
        get_user_message = lambda: next(pending, None)  # noqa: E731
    else:
        get_user_message = lambda: None  # noqa: E731
    return A.Agent(client, get_user_message, A.ALL_TOOLS, **options)


def make_stub_agent(**overrides):
    """A minimal object satisfying build_textual_app()'s expectations, for
    headless pilots that exercise the UI without a live agent thread."""

    class StubClient:
        model = "test-model"
        url = "http://127.0.0.1:8080/v1"
        name = "httpx (raw SSE)"

    class StubAgent:
        client = StubClient()
        mode = "native"
        reasoning_mode = "collapsed"
        temperature = None
        user_dry_params: dict = {}
        server_settings: dict = {"temperature": 0.6, "top_k": 40, "min_p": 0.05}
        registry = None
        store = None
        skills = None
        memories = None
        max_tokens = 4096
        retries = 3
        max_nudges = 2
        risk_classifier = "heuristic"
        approve_level = "low"
        yolo = False
        allow_tools = True
        allow_internet = True
        allow_bash_escape = True
        autocompress_percent = 85
        system_prompt = None
        verify = False
        verify_threshold = A.VERIFY_THRESHOLD
        verify_rounds = A.VERIFY_ROUNDS
        verify_budget = A.VERIFY_BUDGET
        verify_samples = 1
        planning = 'off'
        plan = A.Plan()
        verify_command = None
        verify_mode = "revise"
        unattended = False
        engine = 'llamacpp'
        extra_body: dict = {}
        files = A.FileHistory()
        last_answer = ""
        policy = A.PolicyEngine()
        config_entries = A.Agent.config_entries

        def run(self):  # the app runs this on a daemon thread
            import time

            time.sleep(30)

    stub = StubAgent()
    for key, value in overrides.items():
        setattr(stub, key, value)
    return stub


def clear_sinks() -> None:
    """Reset every UI sink; pilots and console tests must not leak them."""
    for name in (
        "LIVE_SINK", "REASONING_SINK", "STATS_SINK", "RAW_SINK", "TOOLS_SINK",
        "SESSION_SINK", "RESET_SINK", "SKILL_SINK", "MEMORY_SINK",
        "APPROVAL_HOOK", "LOG_SINK",
    ):
        setattr(A, name, None)


# --------------------------------------------------------------------------- #
# Tiny runner (so no pytest dependency is needed)
# --------------------------------------------------------------------------- #
def run_module(namespace: dict, title: str) -> int:
    """Runs every test_* callable in `namespace`. Returns the failure count."""
    tests = sorted(
        (name, obj)
        for name, obj in namespace.items()
        if name.startswith("test_") and callable(obj)
    )
    print(f"\n=== {title} ({len(tests)} tests) ===")
    failures = 0
    for name, test in tests:
        label = name[5:].replace("_", " ")
        try:
            with workspace():
                clear_sinks()
                A.set_read_limits(chars=20_000, lines=400)  # restore defaults
                test()
            print(f"  {green('PASS')}  {label}")
        except Exception:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"  {red('FAIL')}  {red(label)}")
            print(dim("        "
                      + traceback.format_exc().replace("\n", "\n        ")))
        finally:
            clear_sinks()
    passed = len(tests) - failures
    verdict = f"--- {title}: {passed}/{len(tests)} passed ---"
    print(green(verdict) if not failures else red(verdict))
    return failures


def main(namespace: dict, title: str) -> None:
    sys.exit(1 if run_module(namespace, title) else 0)
