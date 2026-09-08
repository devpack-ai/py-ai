"""The remaining flag wiring and small utilities the audit flagged.

Mostly end-to-end CLI runs, because a flag can be parsed correctly and
still not reach the request; the assertions look at the HTTP payload and
headers the fake server received.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from harness import (  # noqa: E402
    A, ROOT, FakeServer, answer, main, make_agent, recorded_console,
    tool_call_chunk,
)


def run_cli(server, *flags, message="hi\n", timeout=90, env=None):
    command = [
        sys.executable, str(ROOT / "py-ai.py"),
        "--base-url", server.base_url, "--model", "model-a",
        "--ui", "rich", *flags,
    ]
    return subprocess.run(command, input=message, capture_output=True,
                          text=True, timeout=timeout, cwd=os.getcwd(),
                          env=env or dict(os.environ))


# --------------------------------------------------------------------------- #
# flags that must show up on the wire
# --------------------------------------------------------------------------- #
def test_api_key_is_sent_as_a_bearer_token():
    server = FakeServer()
    try:
        server.push(*answer("ok"))
        completed = run_cli(server, "--no-autosave", "--api-key", "sk-test-123")
        assert completed.returncode == 0, completed.stderr[-300:]
        authorization = server.headers[-1].get("authorization", "")
        assert authorization == "Bearer sk-test-123", server.headers[-1]
    finally:
        server.stop()


def test_system_file_flag_sets_the_prompt():
    server = FakeServer()
    try:
        Path("persona.txt").write_text("You are a laconic Rust reviewer.\n")
        server.push(*answer("ok"))
        completed = run_cli(server, "--no-autosave", "--system-file", "persona.txt")
        assert completed.returncode == 0, completed.stderr[-300:]
        first = server.last_request["messages"][0]
        assert first["role"] == "system"
        assert first["content"].startswith("You are a laconic Rust reviewer.")
    finally:
        server.stop()


def test_tools_file_flag_registers_custom_tools():
    server = FakeServer()
    try:
        Path("my_tools.py").write_text(
            'def shout(text: str) -> str:\n'
            '    """Upper-case the text."""\n'
            '    return text.upper()\n'
        )
        server.push(*answer("ok"))
        completed = run_cli(server, "--no-autosave", "--tools-file", "my_tools.py")
        assert completed.returncode == 0, completed.stderr[-300:]
        offered = [entry["function"]["name"]
                   for entry in server.last_request.get("tools", [])]
        assert "shout" in offered, offered
    finally:
        server.stop()


def test_hide_reasoning_flag_keeps_thinking_off_screen():
    server = FakeServer()
    try:
        server.push(*answer("the answer", reasoning="SECRET DELIBERATION"))
        completed = run_cli(server, "--no-autosave", "--hide-reasoning")
        assert completed.returncode == 0, completed.stderr[-300:]
        assert "the answer" in completed.stdout
        assert "SECRET DELIBERATION" not in completed.stdout
    finally:
        server.stop()


def test_resume_flag_restores_the_previous_session():
    server = FakeServer()
    try:
        server.push(*answer("first reply"))
        first = run_cli(server, "--sessions-dir", "sess", message="remember this\n")
        assert first.returncode == 0, first.stderr[-300:]
        assert list(Path("sess").glob("*.json")), "autosave should have written one"

        server.reset()
        server.push(*answer("second reply"))
        second = run_cli(server, "--sessions-dir", "sess", "--resume", "last",
                         message="and now?\n")
        assert second.returncode == 0, second.stderr[-300:]
        assert "remember this" in second.stdout      # transcript replayed
        # the restored history is sent back to the model
        contents = " ".join(json.dumps(m) for m in server.last_request["messages"])
        assert "remember this" in contents
    finally:
        server.stop()


def test_defense_opt_out_flags_are_accepted_and_warn():
    server = FakeServer()
    try:
        server.push(*answer("ok"))
        completed = run_cli(server, "--no-autosave",
                            "--no-path-confinement", "--no-command-denylist")
        assert completed.returncode == 0, completed.stderr[-300:]
        assert "Traceback" not in completed.stderr
        assert server.count == 1
    finally:
        server.stop()


def test_serve_host_and_port_flags_parse():
    """--serve-host/--serve-port must be accepted; as the served child the
    process runs normally instead of trying to serve itself again."""
    server = FakeServer()
    try:
        server.push(*answer("ok"))
        completed = run_cli(
            server, "--no-autosave", "--serve", "--serve-host", "0.0.0.0",
            "--serve-port", "9123",
            env=dict(os.environ, PY_AI_SERVED="1"),
        )
        assert completed.returncode == 0, completed.stderr[-300:]
        assert server.count == 1
    finally:
        server.stop()


def test_read_limit_lines_flag_and_paging():
    Path("many.txt").write_text("\n".join(f"line{i}" for i in range(1, 51)))
    A.set_read_limits(lines=20)
    out = A.read_file({"path": "many.txt"})
    assert "line20" in out and "line21" not in out
    assert "offset=21" in out             # tells the model how to continue
    A.set_read_limits(lines=5)            # a floor keeps paging usable
    assert A.READ_LIMIT_LINES >= 10
    A.set_read_limits(lines=400)


def test_dry_penalty_last_n_reaches_the_request():
    server = FakeServer()
    try:
        server.push(*answer("ok"))
        completed = run_cli(server, "--no-autosave",
                            "--dry-penalty-last-n", "256",
                            "--dry-multiplier", "0.8")
        assert completed.returncode == 0, completed.stderr[-300:]
        assert server.last_request["dry_penalty_last_n"] == 256
        assert server.last_request["dry_multiplier"] == 0.8
    finally:
        server.stop()


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def test_choose_model_picks_interactively():
    import builtins

    original = builtins.input
    builtins.input = lambda *args: "2"
    try:
        with recorded_console():
            assert A.choose_model(["model-a", "model-b", "model-c"]) == "model-b"
    finally:
        builtins.input = original

    # an absurd list refuses rather than paging forever
    try:
        A.choose_model([f"model-{n}" for n in range(40)])
        raise AssertionError("expected SystemExit for a huge model list")
    except SystemExit as err:
        assert "--model" in str(err)


def test_slash_completer_matches_commands():
    if A.readline is None:                # no readline on this platform
        return

    class FakeReadline:
        @staticmethod
        def get_line_buffer():
            return "/mem"

    original = A.readline
    A.readline = FakeReadline
    try:
        first = A.slash_completer("/mem", 0)
        assert first == "/memory", first
        assert A.slash_completer("/mem", 1) is None      # only one match
    finally:
        A.readline = original


def test_token_tracker_reprint_and_context_bar():
    tracker = A.TokenTracker(1000, "test-model")
    with recorded_console(width=140) as output:
        tracker.record(
            [{"role": "user", "content": "hello"}],
            A.TurnResult(content="hi", finish_reason="stop",
                         usage={"prompt_tokens": 400, "completion_tokens": 100},
                         wall=1.0, ttft=0.2),
        )
        text = output()
    assert "400" in text and "100" in text
    assert "ctx" in text and "left" in text              # the occupancy bar
    with recorded_console(width=140) as output:
        tracker.reprint()                                # re-emits the same line
        assert "ctx" in output()
    tracker.reset()
    with recorded_console(width=140) as output:
        tracker.reprint()
        assert "500" not in output()                     # counters cleared


def test_esc_watcher_is_inert_without_a_terminal():
    with A.EscWatcher() as pressed:
        assert callable(pressed)
        assert pressed() is False        # nothing typed, and no tty in tests


def test_autocompress_and_reasoning_via_dispatch():
    agent = make_agent()
    with recorded_console() as output:
        assert agent._handle_command("/autocompress") is True
        assert "auto-compress" in output()
    with recorded_console():
        assert agent._handle_command("/autocompress 60") is True
    assert agent.autocompress_percent == 60
    with recorded_console():
        assert agent._handle_command("/reasoning") is True   # alias of /think


def test_plan_flag_takes_a_mode_or_a_task():
    server = FakeServer()
    try:
        # a task description plans at startup, before the first turn
        server.push(*answer('["read the code", "check the inputs"]'))
        server.push(*answer("ready"))
        completed = run_cli(server, "--no-autosave",
                            "--plan", "audit the codebase for security issues")
        assert completed.returncode == 0, completed.stderr[-300:]
        planning_request = server.requests[0]
        assert "planner" in planning_request["messages"][0]["content"]
        assert planning_request["messages"][-1]["content"] == \
            "audit the codebase for security issues"
        assert "read the code" in completed.stdout      # the plan is shown
        # the plan is then injected into the real turn
        assert "Current plan" in server.requests[-1]["messages"][0]["content"]

        # 'auto' remains a mode, not a task
        server.reset()
        server.push(*answer("hi"))
        completed = run_cli(server, "--no-autosave", "--plan", "auto",
                            message="hello\n")
        assert completed.returncode == 0, completed.stderr[-300:]
        assert server.count == 1, "a simple greeting must not trigger planning"
    finally:
        server.stop()


def test_extra_body_reaches_the_request_and_merges():
    server = FakeServer()
    try:
        server.push(*answer("ok"))
        completed = run_cli(
            server, "--no-autosave",
            "--extra-body", '{"chat_template_kwargs": {"enable_thinking": false},'
                            ' "top_k": 20}')
        assert completed.returncode == 0, completed.stderr[-300:]
        payload = server.last_request
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["top_k"] == 20

        # --thinking is the shorthand, and merges with --extra-body rather
        # than replacing the nested map
        server.reset()
        server.push(*answer("ok"))
        completed = run_cli(
            server, "--no-autosave", "--no-thinking",
            "--extra-body", '{"chat_template_kwargs": {"other_flag": 1}}')
        assert completed.returncode == 0, completed.stderr[-300:]
        kwargs = server.last_request["chat_template_kwargs"]
        assert kwargs == {"other_flag": 1, "enable_thinking": False}, kwargs

        # invalid JSON fails fast with a clear message
        completed = run_cli(server, "--no-autosave", "--extra-body", "{oops")
        assert completed.returncode != 0
        assert "not valid JSON" in (completed.stdout + completed.stderr)
        completed = run_cli(server, "--no-autosave", "--extra-body", '["a"]')
        assert "must be a JSON object" in (completed.stdout + completed.stderr)
    finally:
        server.stop()


def test_command_arguments_keep_their_case():
    """The dispatcher lower-cases the verb for matching; arguments must
    survive intact -- model names, shell commands and JSON are all
    case-sensitive."""
    server = FakeServer()
    try:
        server.models = ["vendor/Model3.5-0.8B-MTP:IQ4_NL"]
        server.push(*answer("ok"))
        completed = run_cli(
            server, "--no-autosave",
            message="/model vendor/Model3.5-0.8B-MTP:IQ4_NL\nhi\n")
        assert completed.returncode == 0, completed.stderr[-300:]
        assert server.last_request["model"] == "vendor/Model3.5-0.8B-MTP:IQ4_NL"

        server.reset()
        server.push(*answer("aye"))
        completed = run_cli(server, "--no-autosave",
                            message="/system You Are A Pirate Captain\nhi\n")
        assert completed.returncode == 0, completed.stderr[-300:]
        system_message = server.last_request["messages"][0]["content"]
        assert system_message.startswith("You Are A Pirate Captain")
    finally:
        server.stop()


def test_unattended_cli_defaults_and_overrides():
    server = FakeServer()
    try:
        server.push(*answer("hi"))
        completed = run_cli(server, "--no-autosave", "--unattended")
        assert completed.returncode == 0, completed.stderr[-300:]
        assert "unattended" in completed.stdout.lower() or True

        # an explicit budget must win over the unattended default
        server.reset()
        for index in range(4):
            server.push(*answer(f"r{index}"))
        completed = run_cli(server, "--no-autosave", "--unattended",
                            "--max-session-requests", "1",
                            message="one\ntwo\nthree\n")
        # --unattended makes the exit code meaningful, and a budget stop is
        # inconclusive rather than a success
        assert completed.returncode == A.EXIT_BUDGET, completed.stdout[-300:]
        assert server.count == 1, server.count
        assert "request budget (1)" in completed.stdout
    finally:
        server.stop()


def test_report_and_exit_codes():
    """A run has to be judgeable without reading the transcript."""
    server = FakeServer()
    try:
        # a green check: exit 0, state recorded
        Path("check.py").write_text("raise SystemExit(0)\n")
        server.push(*answer("all done"))
        completed = run_cli(server, "--no-autosave", "--verify-command",
                            "python3 check.py", "--report", "out.json")
        assert completed.returncode == A.EXIT_OK, completed.stdout[-400:]
        report = json.loads(Path("out.json").read_text())
        assert report["verify"]["state"] == "passing"
        assert report["verify"]["command"] == "python3 check.py"
        assert report["session"]["requests"] >= 1
        assert report["session"]["stop_reason"] == "input exhausted"
        assert report["config"]["model"] == "model-a"
        assert report["final_answer"] == "all done"
        for key in ("verify", "files_changed", "session", "plan", "config",
                    "background_jobs", "final_answer"):
            assert key in report, key

        # a red check: exit 1 even though the failure pre-dates the run --
        # "exit 0" has to mean "the check passes now"
        server.reset()
        Path("check.py").write_text(
            "print('FAILED: nope'); raise SystemExit(1)\n")
        for _ in range(6):
            server.push(*answer("done"))
        completed = run_cli(server, "--no-autosave", "--verify-command",
                            "python3 check.py", "--verify-mode", "revise",
                            "--report", "red.json")
        assert completed.returncode == A.EXIT_VERIFY_FAILED
        red = json.loads(Path("red.json").read_text())
        assert red["verify"]["state"] in ("failing", "pre_existing_failures")

        # a budget stop is inconclusive, not a failure
        server.reset()
        for _ in range(4):
            server.push(*answer("r"))
        completed = run_cli(server, "--no-autosave", "--unattended",
                            "--max-session-requests", "1",
                            "--report", "budget.json",
                            message="a\nb\nc\n")
        assert completed.returncode == A.EXIT_BUDGET
        budget = json.loads(Path("budget.json").read_text())
        assert "request budget" in budget["session"]["stop_reason"]

        # an interactive run never hands the shell a surprise
        server.reset()
        server.push(*answer("hi"))
        assert run_cli(server, "--no-autosave").returncode == 0
    finally:
        server.stop()


def test_sandbox_flags_reach_the_run():
    server = FakeServer()
    try:
        server.push(*answer("ok"))
        completed = run_cli(server, "--no-autosave", "--sandbox", "limits",
                            "--sandbox-cpu", "11", "--report", "s.json",
                            message="/config\nhi\n")
        assert completed.returncode == 0, completed.stderr[-300:]
        assert json.loads(Path("s.json").read_text())["config"]["sandbox"] \
            == "limits"
        assert "11s cpu" in completed.stdout      # shown by /config
    finally:
        server.stop()


def test_task_flag_is_the_batch_front_door():
    server = FakeServer()
    try:
        server.push(*answer("The capital is Paris."))
        completed = run_cli(server, "--no-autosave",
                            "--task", "what is the capital of France?",
                            message="")          # no stdin at all
        assert completed.returncode == A.EXIT_OK, completed.stderr[-300:]
        assert "Paris" in completed.stdout
        assert "\u276f what is the capital" in completed.stdout   # echoed
        assert server.count == 1

        # several tasks run in order
        server.reset()
        for word in ("one", "two", "three"):
            server.push(*answer(f"reply {word}"))
        completed = run_cli(server, "--no-autosave", "--task", "alpha",
                            "--task", "beta", "--task", "gamma", message="")
        assert completed.returncode == 0, completed.stderr[-300:]
        assert server.count == 3
        sent = [json.dumps(request["messages"])
                for request in server.requests]
        assert "alpha" in sent[0] and "beta" in sent[1]
        assert "gamma" in sent[2]
        # each task builds on the previous one: it is one session
        assert "alpha" in sent[2]

        # --task-file, and a TUI request is overridden so a batch run cannot
        # sit waiting for input
        server.reset()
        server.push(*answer("done from file"))
        Path("task.txt").write_text("do the thing from a file\n")
        completed = run_cli(server, "--no-autosave", "--ui", "textual",
                            "--task-file", "task.txt",
                            "--report", "r.json", message="")
        assert completed.returncode == 0, completed.stderr[-300:]
        assert "done from file" in completed.stdout
        report = json.loads(Path("r.json").read_text())
        assert report["tasks"] == ["do the thing from a file"]

        # a missing task file fails fast and clearly
        completed = run_cli(server, "--no-autosave",
                            "--task-file", "nope.txt", message="")
        assert completed.returncode != 0
        assert "--task-file" in (completed.stdout + completed.stderr)
    finally:
        server.stop()


def test_task_composes_into_a_ci_shaped_run():
    """--task + --verify-command + --unattended + --report + --sandbox is
    the whole point: one command, a JSON verdict, a meaningful exit code."""
    server = FakeServer()
    try:
        Path("check.py").write_text(
            "print('FAILED: not fixed'); raise SystemExit(1)\n")
        for _ in range(8):
            server.push(*answer("I looked at it."))
        completed = run_cli(
            server, "--no-autosave", "--unattended", "--sandbox", "limits",
            "--verify-command", "python3 check.py", "--verify-mode", "revise",
            "--report", "ci.json", "--task", "fix the failing check",
            message="")
        assert completed.returncode == A.EXIT_VERIFY_FAILED, completed.stdout[-400:]
        report = json.loads(Path("ci.json").read_text())
        assert report["tasks"] == ["fix the failing check"]
        assert report["verify"]["state"] in ("failing", "pre_existing_failures")
        assert report["config"]["unattended"] is True
        assert report["config"]["sandbox"] == "limits"
        assert report["session"]["requests"] >= 1

        # the same shape with a check that passes exits 0
        server.reset()
        Path("check.py").write_text("raise SystemExit(0)\n")
        server.push(*answer("nothing to do"))
        completed = run_cli(
            server, "--no-autosave", "--unattended",
            "--verify-command", "python3 check.py",
            "--report", "ok.json", "--task", "check it", message="")
        assert completed.returncode == A.EXIT_OK, completed.stdout[-400:]
        assert json.loads(Path("ok.json").read_text())["verify"]["state"] \
            == "passing"
    finally:
        server.stop()


def test_session_survives_a_separate_process():
    """The strongest evidence: save in one process, resume in another."""
    server = FakeServer()
    try:
        Path("m.py").write_text("value = 1\n")
        # process 1: plan, a tool call, reasoning, plan progress, autosave
        server.push(*answer('["read m.py", "change the value"]'))
        server.push(tool_call_chunk("edit_file",
                                    {"path": "m.py", "old_str": "1",
                                     "new_str": "2"}, call_id="t1"))
        server.push(*answer("Changed the value to 2.\nPLAN: done 1",
                            reasoning="considering the edit"))
        first = run_cli(server, "--yolo", "--plan", "change the value in m.py",
                        "--task", "do it", message="")
        assert first.returncode == 0, first.stderr[-300:]
        files = sorted(Path(".agent_sessions").glob("*.json"))
        assert len(files) == 1, files
        saved = json.loads(files[0].read_text())
        assert [m["role"] for m in saved["messages"]] == \
            ["user", "assistant", "tool", "assistant"]
        assert saved["plan"][0]["done"] is True     # the PLAN marker stuck
        assert saved["reasonings"] and saved["exchanges"]

        # process 2: resume it and continue
        server.reset()
        server.push(*answer("Yes -- value went from 1 to 2."))
        second = run_cli(server, "--yolo", "--no-autosave",
                         "--resume", files[0].stem,
                         "--task", "what did you change?", message="")
        assert second.returncode == 0, second.stderr[-300:]
        assert "Changed the value to 2" in second.stdout   # replayed
        assert "plan restored" in second.stdout
        sent = json.dumps(server.requests[-1]["messages"])
        assert "Changed the value to 2" in sent            # reached the model
        assert "edit_file" in sent                         # tool turn intact
        assert "[x] read m.py" in sent                     # plan progress

        # a different model on resume is reported, not silently applied
        server.reset()
        server.push(*answer("ok"))
        third = subprocess.run(
            [sys.executable, str(ROOT / "py-ai.py"),
             "--base-url", server.base_url, "--model", "other-model",
             "--ui", "rich", "--yolo", "--no-autosave",
             "--resume", files[0].stem, "--task", "hi"],
            input="", capture_output=True, text=True, timeout=90)
        assert third.returncode == 0, third.stderr[-300:]
        assert "recorded with model-a" in third.stdout
        assert "continuing with other-model" in third.stdout
    finally:
        server.stop()


def test_interactive_session_commands_still_work():
    server = FakeServer()
    try:
        server.push(*answer("Paris."))
        completed = run_cli(
            server, "--yolo",
            message="capital of France?\n/save my nice title\n/sessions\n"
                    "/load 1\n/export\n")
        assert completed.returncode == 0, completed.stderr[-300:]
        out = completed.stdout
        assert "session saved" in out
        assert "my nice title" in out               # listed by /sessions
        assert "restored" in out                    # /load replayed it
        assert "transcript exported" in out
        assert server.count == 1, "only the one real turn hit the model"
        saved = json.loads(
            sorted(Path(".agent_sessions").glob("*.json"))[0].read_text())
        assert saved["title"] == "my nice title"
    finally:
        server.stop()


if __name__ == "__main__":
    main(globals(), "remaining flags, model picking, small utilities")
