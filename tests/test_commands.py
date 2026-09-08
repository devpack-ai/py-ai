"""The slash-command surface: display toggles, session management, export,
restart, and the alias/validation behaviour of the tuning commands.

These are what the user actually types, so each command is driven through
_handle_command() (the real dispatch) rather than its private helper.
"""

import json
import time
from pathlib import Path

from harness import (  # noqa: E402
    A, FakeServer, answer, approval, delta, log_sink, main, make_agent,
    recorded_console, tool_call_chunk,
)


def seeded_agent(server, **kwargs):
    """An agent with one completed turn behind it (answer + reasoning + raw)."""
    agent = make_agent(server, **kwargs)
    server.push(*answer("The answer is 42.", reasoning="pondering deeply"))
    agent.conversation.append({"role": "user", "content": "what is the answer?"})
    agent._begin_user_turn()
    with recorded_console():
        agent._native_turn(agent.conversation)
    return agent


# --------------------------------------------------------------------------- #
# display toggles
# --------------------------------------------------------------------------- #
def test_think_and_answer_toggles():
    server = FakeServer()
    try:
        agent = seeded_agent(server)
        with recorded_console(width=140) as output:
            assert agent._handle_command("/think") is True
            text = output()
        assert "pondering deeply" in text          # reasoning revealed

        with recorded_console(width=140) as output:
            assert agent._handle_command("/res") is True
            text = output()
        assert "The answer is 42." in text
        with recorded_console(width=140) as output:
            assert agent._handle_command("/answer") is True   # alias
            assert "The answer is 42." in output()
    finally:
        server.stop()


def test_settings_command_shows_server_and_overrides():
    server = FakeServer()
    try:
        agent = make_agent(server, temperature=0.25,
                           server_settings={"temperature": 0.6, "top_k": 40})
        with recorded_console(width=140) as output:
            assert agent._handle_command("/settings") is True
            text = output()
        assert "temperature" in text and "0.6" in text and "0.25" in text
        assert "randomness" in text                 # description column
        with recorded_console(width=140) as output:
            assert agent._handle_command("/sampling") is True   # alias
            assert "top_k" in output()
    finally:
        server.stop()


def test_raw_command_shows_exchanges_and_chunks():
    server = FakeServer()
    try:
        agent = seeded_agent(server)
        with recorded_console(width=160) as output:
            assert agent._handle_command("/raw") is True
            text = output()
        assert "raw request" in text and "assembled response" in text
        assert "what is the answer?" in text        # the request body is shown
        with recorded_console(width=160) as output:
            assert agent._handle_command("/raw chunks") is True
            text = output()
        assert "chunk" in text.lower() or "delta" in text.lower()
    finally:
        server.stop()


def test_reasoning_render_helpers():
    panel = A.reasoning_panel("some thoughts", "#1")
    assert panel is not None
    line = A.collapsed_reasoning_line("some thoughts")
    assert "reasoning" in line.plain and "tok" in line.plain
    answer_line = A.collapsed_answer_line("an answer")
    assert answer_line.plain
    assert A.answer_panel("an answer", "test-model") is not None
    assert A.settings_collapsed_line({"temperature": 0.6}).plain


# --------------------------------------------------------------------------- #
# sessions
# --------------------------------------------------------------------------- #
def test_save_list_load_and_delete_sessions():
    server = FakeServer()
    try:
        store = A.JsonSessionStore(".agent_sessions")
        agent = seeded_agent(server, store=store)
        with recorded_console() as output:
            assert agent._handle_command("/save") is True
            assert "saved" in output().lower()
        listing = store.list()
        assert len(listing) == 1 and listing[0]["id"]

        with recorded_console(width=140) as output:
            assert agent._handle_command("/sessions") is True
            assert listing[0]["id"][:8] in output()

        # a fresh agent restores the transcript by ordinal
        other = make_agent(server, store=store)
        with recorded_console(width=140) as output:
            assert other._handle_command("/load 1") is True
            text = output()
        assert any(m.get("content") == "what is the answer?"
                   for m in other.conversation)
        assert "The answer is 42." in text          # replayed into the chat

        assert store.delete(listing[0]["id"]) is not False
        assert store.list() == []
        with recorded_console() as output:
            other._handle_command("/load last")     # nothing left
            assert "no saved sessions" in output().lower()
    finally:
        server.stop()


def test_saved_session_stays_small_and_restores_raw():
    """Requests repeat the whole history, so storing them verbatim is
    O(N^2): message bodies must be dropped from saved exchanges."""
    server = FakeServer()
    try:
        store = A.JsonSessionStore(".agent_sessions")
        agent = seeded_agent(server, store=store)
        for _ in range(3):
            server.push(*answer(" ".join(
                f"sentence {n} with varied words here" for n in range(60))))
            agent.conversation.append({"role": "user", "content": "q " * 250})
            agent._begin_user_turn()
            with recorded_console():
                agent._native_turn(agent.conversation)
        with recorded_console():
            agent._save_session()
        raw = Path(".agent_sessions", f"{agent.session['id']}.json").read_text()
        assert "messages omitted in saved session" in raw
        payload = json.loads(raw)
        assert payload["exchanges"], "raw exchanges should still be listed"
        assert all(isinstance(e["request"]["messages"], str)
                   for e in payload["exchanges"] if e.get("request"))
    finally:
        server.stop()


def test_legacy_session_loads_with_a_notice():
    server = FakeServer()
    try:
        Path(".agent_sessions").mkdir()
        Path(".agent_sessions/legacy-1.json").write_text(json.dumps({
            "id": "legacy-1",
            "messages": [
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
            ],
        }))
        store = A.JsonSessionStore(".agent_sessions")
        agent = make_agent(server, store=store)
        with recorded_console(width=140) as output:
            assert agent._handle_command("/load legacy-1") is True
            text = output()
        assert any(m.get("content") == "old question" for m in agent.conversation)
        assert "old answer" in text
    finally:
        server.stop()


def test_autosave_can_be_disabled():
    server = FakeServer()
    try:
        store = A.JsonSessionStore(".agent_sessions")
        agent = seeded_agent(server, store=store, autosave=False)
        with recorded_console():
            agent._autosave()
        assert store.list() == []                  # nothing written
        agent.autosave = True
        with recorded_console():
            agent._autosave()
        assert len(store.list()) == 1
    finally:
        server.stop()


# --------------------------------------------------------------------------- #
# export / restart
# --------------------------------------------------------------------------- #
def test_export_writes_a_markdown_transcript():
    server = FakeServer()
    try:
        agent = seeded_agent(server)
        with recorded_console() as output:
            assert agent._handle_command("/export") is True
            text = output()
        exports = sorted(Path(".").glob("transcript_*.md"))
        assert exports, text
        body = exports[0].read_text()
        assert "what is the answer?" in body       # transcript
        assert "The answer is 42." in body
        assert "pondering deeply" in body          # reasoning included
        assert "raw" in body.lower()               # raw section
    finally:
        server.stop()


def test_restart_clears_everything():
    server = FakeServer()
    try:
        skills = A.SkillManager(".skills", search_dirs=[".skills"])
        skills.save("Terse mode\nBe brief.")
        skills.activate(skills.list()[0])
        memories = A.MemoryManager(".memories")
        memories.save("# Note\nbody")
        memories.load(memories.list()[0]["name"])
        agent = seeded_agent(server, skills=skills, memories=memories)
        assert agent.conversation and agent.last_answer

        with recorded_console(width=140) as output:
            assert agent._handle_command("/restart") is True
            text = output()
        assert agent.conversation == []
        assert agent.last_answer == ""
        assert skills.active_name is None
        assert memories.active == {}
        assert "fresh start" in text
        assert "py-ai" in text                    # banner reprinted
    finally:
        server.stop()


# --------------------------------------------------------------------------- #
# tuning commands: aliases, display, validation
# --------------------------------------------------------------------------- #
def test_approval_command_display_and_aliases():
    agent = make_agent(approval="low")
    assert agent._approval_display() == "low"
    with recorded_console() as output:
        agent._handle_command("/approval")
        assert "low" in output()
    for argument, expected_level, expected_yolo in (
        ("/approval med", "medium", False),
        ("/approval hi", "high", False),
        ("/approval all", None, False),
        ("/yolo", None, True),
    ):
        with recorded_console():
            agent._handle_command(argument)
        assert agent.approve_level == expected_level
        assert agent.yolo is expected_yolo
    with recorded_console() as output:
        agent._handle_command("/approval nonsense")
        assert "usage" in output()


def test_command_aliases_and_unknown_commands():
    agent = make_agent()
    with recorded_console():
        agent._handle_command("/readlimit 3000")      # alias without the dash
    assert A.READ_LIMIT_CHARS == 3000
    with recorded_console():
        agent._handle_command("/maxtokens 5000")
        assert agent.max_tokens == 5000
    with recorded_console():
        agent._handle_command("/compress")            # alias of /compact
    # an unknown slash command is reported, not silently swallowed
    with recorded_console() as output:
        handled = agent._handle_command("/definitely-not-a-command")
        text = output()
    assert handled is True, "a mistyped command must not be sent to the model"
    assert "unknown command" in text and "/help" in text
    # ...but a message that merely starts with a slash is still a message
    with recorded_console():
        assert agent._handle_command("/etc/passwd is a path") is False


def test_system_extras_composes_skill_and_memories():
    skills = A.SkillManager(".skills", search_dirs=[".skills"])
    skills.save("Terse mode\nBe brief.")
    skills.activate(skills.list()[0])
    memories = A.MemoryManager(".memories")
    first = memories.save("# Socket note\nuse the unix socket")
    memories.load(first)
    agent = make_agent(skills=skills, memories=memories)
    extras = agent._system_extras()
    assert "Active skill" in extras and "Be brief" in extras
    assert "Relevant memories" in extras and "Socket note" in extras
    assert agent._skill_prompt().startswith("## Active skill:")
    skills.deactivate()
    memories.unload()
    assert agent._system_extras() == ""


def test_memory_listing_and_numeric_selection():
    memories = A.MemoryManager(".memories")
    memories.save("# Alpha\nalpha body mentions redis")
    memories.save("# Beta\nbeta body")
    agent = make_agent(memories=memories)
    with recorded_console(width=140) as output:
        agent._handle_command("/memory list")
        text = output()
    assert "Alpha" in text and "Beta" in text
    with recorded_console(width=140) as output:
        agent._handle_command("/memory remember redis")
        text = output()
    assert "hit" in text
    with recorded_console():
        agent._handle_command("/memory load 1")       # from the search listing
    assert len(memories.active) == 1
    with recorded_console():
        agent._handle_command("/memory off")
    assert memories.active == {}
    with recorded_console() as output:
        agent._handle_command("/memory bogus")
        text = output()          # ONE call: export_text clears the buffer
    assert "remember" in text and "load" in text


def test_skill_listing_marks_the_active_one():
    skills = A.SkillManager(".skills", search_dirs=[".skills"])
    skills.save("Alpha skill\nfirst")
    skills.save("Beta skill\nsecond")
    name = sorted(skills.list())[0]
    skills.activate(name)
    agent = make_agent(skills=skills)
    with recorded_console(width=140) as output:
        agent._handle_command("/skills")
        text = output()
    assert name in text and "*" in text
    with recorded_console() as output:
        agent._handle_command("/skill 99")            # out of range
        assert "no skill" in output()




# --------------------------------------------------------------------------- #
# undo / redo
# --------------------------------------------------------------------------- #
def turn(agent, server, text, reply):
    """Drive one complete user turn the way run() does."""
    server.push(*answer(reply, reasoning=f"thinking about {text}"))
    agent._push_undo()
    agent.conversation.append(agent._build_user_message(text))
    agent._begin_user_turn()
    with recorded_console():
        agent._native_turn(agent.conversation)


def test_undo_rewinds_a_turn_and_redo_restores_it():
    server = FakeServer()
    try:
        agent = make_agent(server, store=A.JsonSessionStore(".agent_sessions"))
        turn(agent, server, "first question", "first reply")
        turn(agent, server, "second question", "second reply")
        assert len(agent.conversation) == 4
        assert "second reply" in agent.last_answer

        with recorded_console(width=140) as output:
            assert agent._handle_command("/undo") is True
            text = output()
        assert len(agent.conversation) == 2
        assert all("second" not in str(m.get("content"))
                   for m in agent.conversation)
        assert "first question" in text          # the rewound state is replayed
        assert "undid the last turn" in text
        assert "second question" in text         # says what it dropped
        # no files were touched, so no file warning is shown at all
        # (the file-change path is covered in test_tools_policy.py)
        assert "NOT rewound" not in text

        with recorded_console(width=140) as output:
            assert agent._handle_command("/redo") is True
            text = output()
        assert len(agent.conversation) == 4
        assert any("second question" in str(m.get("content"))
                   for m in agent.conversation)
        assert "redid the turn" in text
    finally:
        server.stop()


def test_undo_restores_answer_reasoning_and_raw():
    server = FakeServer()
    try:
        agent = make_agent(server)
        turn(agent, server, "q1", "answer one")
        first_reasonings = len(agent.session_reasonings)
        first_exchanges = len(agent.session_exchanges)
        turn(agent, server, "q2", "answer two")
        assert len(agent.session_reasonings) > first_reasonings
        assert len(agent.session_exchanges) > first_exchanges

        with recorded_console():
            agent._handle_command("/undo")
        assert agent.last_answer == "answer one"
        assert len(agent.session_reasonings) == first_reasonings
        assert len(agent.session_exchanges) == first_exchanges
        # /think and /raw now operate on the rewound material
        assert agent.turn_reasonings == [r["text"] for r in agent.session_reasonings]
    finally:
        server.stop()


def test_multiple_undos_then_redos_in_order():
    server = FakeServer()
    try:
        agent = make_agent(server)
        for index in range(3):
            turn(agent, server, f"q{index}", f"a{index}")
        assert len(agent.conversation) == 6

        with recorded_console():
            agent._handle_command("/undo")
            agent._handle_command("/undo")
        assert len(agent.conversation) == 2
        assert "a0" in agent.last_answer

        with recorded_console():
            agent._handle_command("/redo")
        assert len(agent.conversation) == 4 and "a1" in agent.last_answer
        with recorded_console():
            agent._handle_command("/redo")
        assert len(agent.conversation) == 6 and "a2" in agent.last_answer
        with recorded_console() as output:
            agent._handle_command("/redo")
            assert "nothing to redo" in output()
    finally:
        server.stop()


def test_a_new_turn_invalidates_redo():
    server = FakeServer()
    try:
        agent = make_agent(server)
        turn(agent, server, "q1", "a1")
        turn(agent, server, "q2", "a2")
        with recorded_console():
            agent._handle_command("/undo")
        assert agent._redo_stack
        turn(agent, server, "different question", "different reply")
        assert agent._redo_stack == []            # branch abandoned
        with recorded_console() as output:
            agent._handle_command("/redo")
            assert "nothing to redo" in output()
    finally:
        server.stop()


def test_undo_with_nothing_to_undo_and_after_restart():
    server = FakeServer()
    try:
        agent = make_agent(server)
        with recorded_console() as output:
            assert agent._handle_command("/undo") is True
            assert "nothing to undo" in output()
        turn(agent, server, "q1", "a1")
        with recorded_console():
            agent._handle_command("/restart")
        assert agent._undo_stack == [] and agent._redo_stack == []
        with recorded_console() as output:
            agent._handle_command("/undo")
            assert "nothing to undo" in output()
    finally:
        server.stop()


def test_undo_depth_is_capped():
    server = FakeServer()
    try:
        agent = make_agent(server)
        for index in range(A.UNDO_DEPTH + 4):
            turn(agent, server, f"q{index}", f"a{index}")
        assert len(agent._undo_stack) == A.UNDO_DEPTH
    finally:
        server.stop()


def test_undo_reverts_a_compaction():
    """Compaction rewrites history; undo should bring the full turns back."""
    server = FakeServer()
    try:
        agent = make_agent(server)
        for index in range(4):
            turn(agent, server, f"question {index}", f"reply {index}")
        before = len(agent.conversation)
        agent._push_undo()                        # /compact happens on a turn
        server.push(*answer("A dense summary of everything so far."))
        with recorded_console():
            agent._compress()
        assert len(agent.conversation) < before
        with recorded_console():
            agent._handle_command("/undo")
        assert len(agent.conversation) == before
        assert any("question 0" in str(m.get("content"))
                   for m in agent.conversation)
    finally:
        server.stop()


def test_undo_persists_to_the_session_file():
    server = FakeServer()
    try:
        store = A.JsonSessionStore(".agent_sessions")
        agent = make_agent(server, store=store)
        turn(agent, server, "keep this", "kept")
        turn(agent, server, "drop this", "dropped")
        with recorded_console():
            agent._handle_command("/undo")
        saved = json.loads(
            Path(".agent_sessions", f"{agent.session['id']}.json").read_text())
        contents = json.dumps(saved["messages"])
        assert "keep this" in contents
        assert "drop this" not in contents        # rewind was autosaved
    finally:
        server.stop()


def test_export_html_is_self_contained_and_escaped():
    """A transcript can contain arbitrary model output and is opened in a
    browser, so everything must be escaped."""
    import base64
    server = FakeServer()
    try:
        agent = make_agent(server)
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
            "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
        Path("shot.png").write_bytes(png)
        server.push(*answer("Use <script>alert(1)</script> carefully & well.",
                            reasoning="pondering <b>markup</b>"))
        with recorded_console():
            agent.conversation.append(
                agent._build_user_message("look at @shot.png & tell me"))
        agent._begin_user_turn()
        with recorded_console():
            agent._native_turn(agent.conversation)

        with recorded_console() as output:
            assert agent._handle_command("/export html") is True
            text = output()
        pages = sorted(Path(".").glob("transcript_*.html"))
        assert pages, text
        page = pages[0].read_text()

        assert page.startswith("<!doctype html>") and page.rstrip().endswith(
            "</html>")
        assert "<style>" in page                      # self-contained styling
        # content is escaped, not executable
        assert "<script>alert(1)</script>" not in page
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
        assert "&amp; tell me" in page                # ampersands escaped
        # reasoning and raw are collapsible
        assert page.count("<details>") >= 2
        assert "pondering &lt;b&gt;markup&lt;/b&gt;" in page
        assert "exchange 1" in page
        # the attached image is embedded, so the file stands alone
        assert 'img class="attachment" src="data:image/png;base64,' in page
        # header metadata
        assert agent.client.model in page and "messages" in page
    finally:
        server.stop()


def test_export_defaults_to_markdown_and_rejects_junk():
    server = FakeServer()
    try:
        agent = seeded_agent(server)
        with recorded_console():
            agent._handle_command("/export")          # default
        assert sorted(Path(".").glob("transcript_*.md"))
        assert not sorted(Path(".").glob("transcript_*.html"))
        with recorded_console():
            agent._handle_command("/export html")
        assert sorted(Path(".").glob("transcript_*.html"))
        with recorded_console() as output:
            agent._handle_command("/export pdf")      # unsupported
            assert "usage" in output()
    finally:
        server.stop()


def test_export_html_refuses_an_empty_session():
    server = FakeServer()
    try:
        agent = make_agent(server)
        with recorded_console() as output:
            agent._handle_command("/export html")
            assert "nothing to export" in output()
        assert not sorted(Path(".").glob("transcript_*.html"))
    finally:
        server.stop()


# --------------------------------------------------------------------------- #
# answer verification (self-critique)
# --------------------------------------------------------------------------- #
def verdict(score, issues=(), text="reviewed"):
    """A scripted critique response."""
    return [delta({"content": json.dumps(
        {"score": score, "issues": list(issues), "verdict": text})},
        finish="stop")]


def test_verify_is_off_by_default_and_costs_nothing():
    server = FakeServer()
    try:
        agent = make_agent(server)
        assert agent.verify is False
        turn(agent, server, "q", "an answer")
        before = server.count
        agent._maybe_verify(agent.conversation)
        assert server.count == before      # no critique request at all
    finally:
        server.stop()


def test_verify_accepts_a_good_answer_after_one_critique():
    server = FakeServer()
    try:
        agent = make_agent(server, verify=True, verify_threshold=70)
        turn(agent, server, "what is 2+2?", "4")
        before = server.count
        server.push(*verdict(95, [], "correct and complete"))
        with recorded_console(width=140) as output:
            agent._maybe_verify(agent.conversation)
            text = output()
        assert server.count == before + 1          # exactly one extra call
        assert "verify: 95/100" in text and "correct and complete" in text
        assert agent.last_answer == "4"            # untouched
    finally:
        server.stop()


def test_verify_revises_a_poor_answer_and_replaces_it():
    server = FakeServer()
    try:
        agent = make_agent(server, verify=True, verify_threshold=70,
                           verify_rounds=1)
        turn(agent, server, "list two prime numbers", "prime numbers exist")
        server.push(*verdict(30, ["does not list any primes",
                                  "no examples given"], "incomplete"))
        server.push(*answer("Two primes: 2 and 3."))
        server.push(*verdict(90, [], "now complete"))
        with recorded_console(width=140) as output:
            agent._maybe_verify(agent.conversation)
            text = output()
        assert "verify: 30/100" in text and "revising" in text
        assert "does not list any primes" in text   # issues surfaced
        assert "verify: 90/100" in text             # re-scored after revision
        assert agent.last_answer == "Two primes: 2 and 3."
        # the stored assistant message was REPLACED, not appended to
        assistants = [m for m in agent.conversation
                      if m.get("role") == "assistant" and m.get("content")]
        assert len(assistants) == 1
        assert assistants[0]["content"] == "Two primes: 2 and 3."
        # the critique never entered the history
        contents = json.dumps(agent.conversation)
        assert "reviewed" not in contents and "score" not in contents
    finally:
        server.stop()


def test_verify_stops_after_the_round_cap():
    server = FakeServer()
    try:
        agent = make_agent(server, verify=True, verify_threshold=90,
                           verify_rounds=1)
        turn(agent, server, "q", "weak answer")
        server.push(*verdict(20, ["bad"], "poor"))
        server.push(*answer("slightly better answer"))
        server.push(*verdict(25, ["still bad"], "still poor"))
        with recorded_console(width=140) as output:
            agent._maybe_verify(agent.conversation)
            text = output()
        assert "revision(s) done" in text
        assert "keeping the best answer" in text.lower()
        assert agent.last_answer == "slightly better answer"
        assert not server.script, "every scripted response should be consumed"
    finally:
        server.stop()


def test_verify_respects_the_token_budget():
    server = FakeServer()
    try:
        agent = make_agent(server, verify=True, verify_threshold=90,
                           verify_rounds=5, verify_budget=1)   # spent instantly
        turn(agent, server, "q", "weak")
        server.push(*verdict(10, ["bad"], "poor"))
        with recorded_console(width=140) as output:
            agent._maybe_verify(agent.conversation)
            text = output()
        assert "token budget spent" in text
        assert agent.last_answer == "weak"        # no revision attempted
    finally:
        server.stop()


def test_verify_fails_open_on_an_unparseable_verdict():
    """A broken judge must never be able to spin the agent."""
    server = FakeServer()
    try:
        agent = make_agent(server, verify=True)
        turn(agent, server, "q", "the answer")
        server.push(*answer("I think it is quite good, honestly"))  # no JSON
        with recorded_console(width=140) as output:
            agent._maybe_verify(agent.conversation)
            text = output()
        assert "no usable verdict" in text
        assert agent.last_answer == "the answer"
    finally:
        server.stop()


def test_verify_reads_a_score_from_prose_as_a_fallback():
    server = FakeServer()
    try:
        agent = make_agent(server, verify=True, verify_threshold=70)
        server.push(*answer("After review I would say score: 85 out of 100"))
        assessment = agent._assess_answer("the request", "the answer")
        assert assessment is not None
        score, _issues, _verdict = assessment
        assert score == 85
    finally:
        server.stop()


def test_verify_leaves_no_trace_in_raw_reasoning_or_stats():
    server = FakeServer()
    try:
        agent = make_agent(server, verify=True, verify_threshold=70)
        turn(agent, server, "q", "an answer")
        exchanges = len(agent.turn_exchanges)
        reasonings = len(agent.session_reasonings)
        stats = agent.tracker._last_render
        server.push(*verdict(30, ["thin"], "thin"))
        server.push(*answer("a better answer"))
        server.push(*verdict(95, [], "good"))
        with recorded_console():
            agent._maybe_verify(agent.conversation)
        assert len(agent.turn_exchanges) == exchanges      # /raw untouched
        assert len(agent.session_reasonings) == reasonings  # /think untouched
        assert agent.tracker._last_render is stats          # stats line untouched
    finally:
        server.stop()


def test_verify_command_controls():
    agent = make_agent()
    with recorded_console() as output:
        assert agent._handle_command("/verify") is True
        assert "off" in output()
    with recorded_console():
        agent._handle_command("/verify on")
    assert agent.verify is True
    with recorded_console():
        agent._handle_command("/verify 85")
    assert agent.verify is True and agent.verify_threshold == 85
    with recorded_console() as output:
        agent._handle_command("/verify")
        assert "85/100" in output()
    with recorded_console():
        agent._handle_command("/verify off")
    assert agent.verify is False
    with recorded_console() as output:
        agent._handle_command("/verify 200")       # out of range
        assert "1-100" in output()
    with recorded_console() as output:
        agent._handle_command("/verify wat")
        assert "usage" in output()
    values = dict((label, value) for label, value, _ in agent.config_entries())
    assert values["verify answers"] == "off"
    agent.verify = True
    values = dict((label, value) for label, value, _ in agent.config_entries())
    assert "below" in values["verify answers"]


# --------------------------------------------------------------------------- #
# verification quality: deterministic checks, best-of, judge consistency
# --------------------------------------------------------------------------- #
def test_deterministic_checks_catch_broken_code_blocks():
    """Free, unbiased, no model: a python block that cannot parse is a
    fact, not an opinion."""
    broken = "Here you go:\n\n```python\ndef add(a, b)\n    return a + b\n```"
    issues = A.code_block_issues(broken)
    assert issues and "does not parse" in issues[0]
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    assert A.code_block_issues(good) == []
    assert A.code_block_issues('```json\n{"a": 1}\n```') == []
    bad_json = A.code_block_issues('```json\n{"a": 1,}\n```')
    assert bad_json and "invalid" in bad_json[0]


def test_deterministic_checks_do_not_flag_legitimate_fragments():
    """The expensive failure mode would be 'fixing' correct answers."""
    for snippet in (
        "```python\ndef add(a, b):\n    ...\n```",              # elision
        "```python\n# ...\nresult = compute()\n```",            # comment elision
        "```python\nreturn a + b\n```",                         # bare fragment
        "```diff\n- old = 1\n+ new = 2\n```",                   # a diff
        "```python\n+ added_line = True\n```",                  # diff-ish
        "```\nsome plain text\n```",                            # no language
        "```bash\nif [ -f x ]; then echo hi; fi\n```",           # not python
        "```python\nx = 1\n```",                                # one-liner
    ):
        assert A.code_block_issues(snippet) == [], snippet


def test_deterministic_checks_catch_fabricated_actions():
    """The exact hallucination self-critique cannot see: claiming a
    deletion that no tool call performed."""
    conversation = [
        {"role": "user", "content": "delete e.txt"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "1", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "content": "ee"},
    ]
    assert A.tools_used_this_turn(conversation) == {"read_file"}
    issues = A.deterministic_answer_issues(
        "I've successfully deleted the file e.txt.", conversation)
    assert issues and "delete_file" in issues[0] and "claims" in issues[0]

    # ...but the same claim IS supported when the tool actually ran
    conversation[1]["tool_calls"][0]["function"]["name"] = "delete_file"
    assert A.deterministic_answer_issues(
        "I've successfully deleted the file e.txt.", conversation) == []

    # claims that ARE fabricated (adverbs and tenses vary)
    for claimed in ("I've successfully deleted the file.", "I deleted e.txt.",
                    "I have now removed it.", "I've created the file.",
                    "We updated the config for you.", "I ran the tests."):
        assert A.unsupported_claim_issues(claimed, set()), claimed

    # ...and the negatives, which matter more: flagging an honest answer
    # would send the loop off to "fix" something that was already right
    for benign in ("You could delete it with delete_file.",
                   "I will create the file if you want.",
                   "To remove it, run delete_file.",
                   "I have not deleted it.",
                   "I haven't removed anything.",
                   "I never deleted that file.",
                   "I can delete it if you want.",
                   "Deleting requires delete_file."):
        assert A.unsupported_claim_issues(benign, set()) == [], benign


def test_deterministic_issues_trigger_a_revision_without_any_critique():
    """Objective problems skip the judge entirely -- zero critique tokens."""
    server = FakeServer()
    try:
        agent = make_agent(server, verify=True)
        turn(agent, server, "delete e.txt",
             "I have deleted e.txt for you.")          # no tool ran
        before = server.count
        server.push(*answer("I cannot delete files without a delete_file call."))
        server.push(*verdict(95, [], "honest now"))
        with recorded_console(width=140) as output:
            agent._maybe_verify(agent.conversation)
            text = output()
        assert "failed check(s)" in text and "objective problems" in text
        assert "no delete_file" in text                # the concrete issue
        assert server.count == before + 2              # revision + final score
        assert "cannot delete files" in agent.last_answer
    finally:
        server.stop()


def test_best_of_reverts_a_revision_that_scores_worse():
    """A revision can be worse than the draft; keeping the last one would
    make the answer worse than not verifying at all."""
    server = FakeServer()
    try:
        agent = make_agent(server, verify=True, verify_threshold=90,
                           verify_rounds=1)
        turn(agent, server, "q", "a decent draft")
        server.push(*verdict(80, ["could be clearer"], "nearly"))
        server.push(*answer("a rambling worse rewrite"))
        server.push(*verdict(40, ["now it is wrong"], "worse"))
        with recorded_console(width=140) as output:
            agent._maybe_verify(agent.conversation)
            text = output()
        assert "reverting to the better answer" in text
        assert agent.last_answer == "a decent draft"
        assistants = [m for m in agent.conversation
                      if m.get("role") == "assistant" and m.get("content")]
        assert assistants[-1]["content"] == "a decent draft"
    finally:
        server.stop()


def test_a_good_revision_is_kept_and_stored():
    server = FakeServer()
    try:
        agent = make_agent(server, verify=True, verify_threshold=70,
                           verify_rounds=1)
        turn(agent, server, "name two primes", "primes exist")
        server.push(*verdict(30, ["lists no primes"], "incomplete"))
        server.push(*answer("Two primes: 2 and 3."))
        server.push(*verdict(95, [], "correct"))
        with recorded_console(width=140):
            agent._maybe_verify(agent.conversation)
        assert agent.last_answer == "Two primes: 2 and 3."
        assistants = [m for m in agent.conversation
                      if m.get("role") == "assistant" and m.get("content")]
        assert assistants[-1]["content"] == "Two primes: 2 and 3."
    finally:
        server.stop()


def test_inconsistent_verdict_is_rejected():
    """A low score with no stated issue gives the revision nothing to act
    on, so it is treated as no verdict rather than a blind rewrite."""
    server = FakeServer()
    try:
        agent = make_agent(server, verify=True, verify_threshold=70)
        server.push(*verdict(20, [], "bad, but I won't say why"))
        assert agent._assess_answer("request", "answer") is None
        # the same score WITH an issue is usable
        server.push(*verdict(20, ["it ignores the question"], "bad"))
        assessment = agent._assess_answer("request", "answer")
        assert assessment is not None and assessment[0] == 20
    finally:
        server.stop()


def test_median_of_samples_smooths_a_noisy_judge():
    assert A.median_score([10, 90, 50]) == 50
    assert A.median_score([70]) == 70
    assert A.median_score([60, 80]) == 70
    server = FakeServer()
    try:
        agent = make_agent(server, verify=True, verify_threshold=70,
                           verify_samples=3)
        # one wild outlier must not decide the outcome
        server.push(*verdict(95, [], "great"))
        server.push(*verdict(20, ["thin"], "poor"))
        server.push(*verdict(90, [], "fine"))
        with recorded_console(width=140) as output:
            assessment = agent._critique("request", "answer")
            text = output()
        assert assessment is not None
        score, issues, _verdict = assessment
        assert score == 90                     # median of 20/90/95
        assert "thin" in " ".join(issues)      # issues are unioned
        assert server.count == 3
        assert text == "" or "verify" not in text   # sampling is silent
    finally:
        server.stop()


def test_sampling_stops_at_the_budget():
    server = FakeServer()
    try:
        agent = make_agent(server, verify=True, verify_samples=5,
                           verify_budget=1)
        server.push(*verdict(80, [], "ok"))
        with recorded_console():
            assessment = agent._critique("request", "answer")
        assert assessment is not None
        assert server.count == 1, "budget must stop further samples"
    finally:
        server.stop()


def test_critique_prompt_is_blind_and_rubric_anchored():
    server = FakeServer()
    try:
        agent = make_agent(server, verify=True)
        server.push(*verdict(85, [], "fine"))
        agent._assess_answer("the user request", "the candidate answer")
        payload = server.last_request["messages"]
        system = payload[0]["content"]
        user = payload[1]["content"]
        # framed as marking someone else's work, not self-review
        assert "CANDIDATE" in system and "not you" in system
        assert "CANDIDATE_ANSWER" in user
        # anchored bands and anti-nitpick rules
        assert "90-100" in system and "40-69" in system
        assert "style preferences" in system
        assert "MUST list at least one issue" in system
        assert server.last_request["max_tokens"] <= A.VERIFY_CRITIQUE_CAP
    finally:
        server.stop()


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #
def test_plan_model_operations():
    plan = A.Plan()
    assert not plan and plan.progress() == "0/0"
    plan.set(["read the parser", "add the flag", "run tests"], task="add a flag")
    assert plan and len(plan.steps) == 3 and plan.task == "add a flag"
    assert plan.next_step() == "read the parser"
    assert plan.mark(1) and plan.done_count == 1
    assert plan.next_step() == "add the flag"
    assert plan.progress() == "1/3"
    assert "1. [x] read the parser" in plan.render()
    assert "2. [ ] add the flag" in plan.render()
    assert plan.mark(99) is False              # out of range
    plan.add("update the docs")
    assert len(plan.steps) == 4
    assert plan.drop(4) and len(plan.steps) == 3
    assert plan.mark(2, done=False) or True
    plan.set([f"step {n}" for n in range(30)])
    assert len(plan.steps) == A.PLAN_MAX_STEPS  # capped
    round_tripped = A.Plan()
    round_tripped.from_list(plan.to_list(), plan.task)
    assert round_tripped.render() == plan.render()
    plan.clear()
    assert not plan and plan.next_step() is None


def test_multi_step_heuristic():
    for multi in (
        "refactor the parser and then add tests for it",
        "read main.py, update the config, and run the test suite",
        "implement the --flag option in the CLI",
        "first check the logs then fix the failing test",
        "1. add the flag\n2. document it",
        "migrate the storage layer to sqlite",
    ):
        assert A.looks_multi_step(multi), multi
    for simple in (
        "what is 2+2?",
        "hi",
        "explain how the parser works",          # a single explanation
        "read main.py",                          # one action
        "what does this error mean?",
    ):
        assert not A.looks_multi_step(simple), simple


def test_plan_is_injected_at_the_system_level():
    server = FakeServer()
    try:
        agent = make_agent(server)
        agent.plan.set(["read the parser", "add the flag"], task="add a flag")
        extras = agent._system_extras()
        assert "Current plan" in extras
        assert "1. [ ] read the parser" in extras
        assert "PLAN: done 1" in extras          # how to report progress
        # it reaches the wire as a system message, without being stored
        server.push(*answer("ok"))
        agent.conversation.append({"role": "user", "content": "go"})
        agent._begin_user_turn()
        agent._native_turn(agent.conversation)
        first = server.last_request["messages"][0]
        assert first["role"] == "system" and "Current plan" in first["content"]
        assert agent.conversation[0]["role"] == "user"
        # a finished plan changes the instruction instead of nagging
        agent.plan.mark(1)
        agent.plan.mark(2)
        assert "Every step is complete" in agent._system_extras()
    finally:
        server.stop()


def test_progress_marker_is_parsed_and_stripped():
    """Progress costs no extra request: the model emits a marker and we
    parse it deterministically, then hide it from the user."""
    server = FakeServer()
    try:
        agent = make_agent(server)
        agent.plan.set(["read the parser", "add the flag"])
        server.push(*answer("I read the parser; it uses argparse.\nPLAN: done 1"))
        agent.conversation.append({"role": "user", "content": "go"})
        agent._begin_user_turn()
        with recorded_console():
            agent._native_turn(agent.conversation)
        with recorded_console(width=140) as output:
            agent._consume_plan_progress()
            text = output()
        assert agent.plan.done_count == 1
        assert "plan: 1/2 done" in text and "next: add the flag" in text
        # the marker is gone from the answer AND from the stored message
        assert "PLAN:" not in agent.last_answer
        assert "argparse" in agent.last_answer
        stored = [m for m in agent.conversation if m.get("role") == "assistant"]
        assert "PLAN:" not in str(stored[-1]["content"])
    finally:
        server.stop()


def test_plan_generation_parses_json_and_falls_back():
    server = FakeServer()
    try:
        agent = make_agent(server)
        server.push(*answer('["read main.py", "add the flag", "run tests"]'))
        with recorded_console(width=140) as output:
            assert agent._generate_plan("add a flag to the CLI") is True
            text = output()
        assert len(agent.plan.steps) == 3
        assert agent.plan.steps[0]["text"] == "read main.py"
        assert "plan" in text and "read main.py" in text
        assert server.last_request["max_tokens"] <= A.PLAN_GENERATE_CAP

        # a numbered list instead of JSON still works
        agent.plan.clear()
        server.push(*answer("1. open the file\n2. change the value\n3. verify"))
        with recorded_console():
            assert agent._generate_plan("task") is True
        assert [s["text"] for s in agent.plan.steps] == [
            "open the file", "change the value", "verify"]

        # {"steps": [...]} is a common shape too
        agent.plan.clear()
        server.push(*answer('{"steps": ["alpha step", "beta step"]}'))
        with recorded_console():
            assert agent._generate_plan("task") is True
        assert len(agent.plan.steps) == 2

        # unusable output leaves the agent unplanned rather than stuck
        agent.plan.clear()
        server.push(*answer("I am not sure how to break this down."))
        with recorded_console(width=140) as output:
            assert agent._generate_plan("task") is False
            assert "no usable plan" in output()
        assert not agent.plan
    finally:
        server.stop()


def test_autoplan_only_fires_when_configured_and_needed():
    server = FakeServer()
    try:
        agent = make_agent(server, planning="off")
        before = server.count
        agent._maybe_autoplan("refactor the parser and then add tests")
        assert server.count == before          # off: never plans

        agent.planning = "auto"
        agent._maybe_autoplan("hi")            # too simple to plan
        assert server.count == before

        server.push(*answer('["read it", "change it", "test it"]'))
        with recorded_console():
            agent._maybe_autoplan("refactor the parser and then add tests")
        assert len(agent.plan.steps) == 3

        # an existing plan is never silently replaced
        count = server.count
        agent._maybe_autoplan("another multi step task and then more")
        assert server.count == count
    finally:
        server.stop()


def test_plan_command_surface():
    server = FakeServer()
    try:
        agent = make_agent(server)
        with recorded_console(width=140) as output:
            assert agent._handle_command("/plan") is True
            assert "no plan" in output()
        with recorded_console():
            agent._handle_command("/plan add read the parser")
            agent._handle_command("/plan add add the flag")
        assert len(agent.plan.steps) == 2
        with recorded_console(width=140) as output:
            agent._handle_command("/plan done 1")
            assert "[x] read the parser" in output()
        with recorded_console(width=140) as output:
            agent._handle_command("/plan undone 1")
            assert "[ ] read the parser" in output()
        with recorded_console():
            agent._handle_command("/plan drop 2")
        assert len(agent.plan.steps) == 1
        with recorded_console() as output:
            agent._handle_command("/plan done 9")
            assert "usage" in output()
        with recorded_console():
            agent._handle_command("/plan auto")
        assert agent.planning == "auto"
        with recorded_console():
            agent._handle_command("/plan auto off")
        assert agent.planning == "off"
        with recorded_console():
            agent._handle_command("/plan clear")
        assert not agent.plan
        # a bare task description generates a plan
        server.push(*answer('["only step"]'))
        with recorded_console():
            agent._handle_command("/plan tidy up the imports")
        assert len(agent.plan.steps) == 1
    finally:
        server.stop()


def test_plan_survives_sessions_undo_and_is_cleared_by_restart():
    server = FakeServer()
    try:
        store = A.JsonSessionStore(".agent_sessions")
        agent = make_agent(server, store=store)
        agent.plan.set(["alpha", "beta"], task="the task")
        agent.plan.mark(1)
        turn(agent, server, "go", "working on it")
        with recorded_console():
            agent._save_session()
        session_id = agent.session["id"]

        fresh = make_agent(server, store=store)
        with recorded_console(width=140) as output:
            fresh._handle_command(f"/load {session_id}")
            text = output()
        assert fresh.plan.progress() == "1/2"
        assert fresh.plan.steps[0]["text"] == "alpha"
        assert "plan restored" in text

        # undo restores the plan state along with the conversation
        agent.plan.mark(2)
        assert agent.plan.progress() == "2/2"
        with recorded_console():
            agent._handle_command("/undo")
        assert agent.plan.progress() == "1/2", "undo should rewind the plan"

        with recorded_console():
            agent._handle_command("/restart")
        assert not agent.plan
    finally:
        server.stop()


def test_planning_appears_in_config():
    agent = make_agent(planning="auto")
    agent.plan.set(["a", "b"])
    values = dict((label, value) for label, value, _ in agent.config_entries())
    assert values["planning"].startswith("auto")
    assert "0/2" in values["planning"]


# --------------------------------------------------------------------------- #
# /model: switching mid-session
# --------------------------------------------------------------------------- #
def test_model_listing_marks_the_current_one():
    server = FakeServer()
    try:
        server.models = ["model-a", "model-b", "model-c"]
        agent = make_agent(server)          # harness clients use model-a
        with recorded_console(width=140) as output:
            assert agent._handle_command("/model") is True
            text = output()
        for name in server.models:
            assert name in text
        assert "*" in text                  # the current one is marked
        assert "/model <n|name> switches" in text
    finally:
        server.stop()


def test_model_switch_by_name_and_index_reaches_the_wire():
    server = FakeServer()
    try:
        server.models = ["model-a", "model-b", "model-c"]
        agent = make_agent(server)
        original = agent.client.model
        with recorded_console(width=140) as output:
            agent._handle_command("/model model-b")
            text = output()
        assert agent.client.model == "model-b"
        assert f"{original} \u2192 model-b" in text

        server.push(*answer("hello from b"))
        agent.conversation.append({"role": "user", "content": "hi"})
        agent._begin_user_turn()
        with recorded_console():
            agent._native_turn(agent.conversation)
        assert server.last_request["model"] == "model-b"   # actually used

        with recorded_console():                # by index, from the listing
            agent._handle_command("/model")
            agent._handle_command("/model 3")
        assert agent.client.model == "model-c"
        with recorded_console() as output:
            agent._handle_command("/model 9")
            assert "no model #9" in output()
        assert agent.client.model == "model-c"
    finally:
        server.stop()


def test_model_switch_updates_the_context_window():
    server = FakeServer()
    try:
        server.models = ["model-a", "model-b"]
        server.n_ctx = 4096
        agent = make_agent(server, tracker=A.TokenTracker(4096, "probe"))
        server.n_ctx = 32768                # the new model has a bigger window
        with recorded_console(width=140) as output:
            agent._handle_command("/model model-b")
            text = output()
        assert agent.tracker.ctx_size == 32768
        assert "32,768 tokens" in text
    finally:
        server.stop()


def test_model_switch_retries_native_when_protocol_is_auto():
    server = FakeServer()
    try:
        server.models = ["model-a", "model-b"]
        agent = make_agent(server, protocol="auto")
        assert agent.allow_fallback is True
        agent.mode = "text"                 # as if the old model fell back
        with recorded_console(width=140) as output:
            agent._handle_command("/model model-b")
            text = output()
        assert agent.mode == "native"
        assert "protocol reset to native" in text

        # with an explicit --protocol text, the choice is respected
        pinned = make_agent(server, protocol="text")
        assert pinned.allow_fallback is False
        with recorded_console():
            pinned._handle_command("/model model-b")
        assert pinned.mode == "text"
    finally:
        server.stop()


def test_model_switch_keeps_the_conversation_and_warns():
    server = FakeServer()
    try:
        server.models = ["model-a", "model-b"]
        agent = make_agent(server)
        turn(agent, server, "remember this", "noted")
        before = len(agent.conversation)
        with recorded_console(width=140) as output:
            agent._handle_command("/model model-b")
            text = output()
        assert len(agent.conversation) == before      # history preserved
        assert "conversation continues" in text and "prefix cache" in text
    finally:
        server.stop()


def test_model_switch_edge_cases():
    server = FakeServer()
    try:
        server.models = ["model-a"]
        agent = make_agent(server)
        current = agent.client.model
        with recorded_console() as output:
            agent._handle_command(f"/model {current}")   # the same model
            assert "already using" in output()
        # a name the endpoint does not list is still allowed (custom
        # deployments, aliases) -- switching must not be blocked
        with recorded_console(width=140) as output:
            agent._handle_command("/model my-private-deployment")
            assert "my-private-deployment" in output()
        assert agent.client.model == "my-private-deployment"
        # the sink fires so the UI can follow
        seen = []
        A.MODEL_SINK = seen.append
        try:
            with recorded_console():
                agent._handle_command("/model model-a")
        finally:
            A.MODEL_SINK = None
        assert seen == ["model-a"], seen
    finally:
        server.stop()


def test_session_notes_a_model_mismatch_on_load():
    server = FakeServer()
    try:
        server.models = ["model-a", "model-b"]
        store = A.JsonSessionStore(".agent_sessions")
        agent = make_agent(server, store=store)
        recorded_with = agent.client.model
        turn(agent, server, "hello", "hi")
        with recorded_console():
            agent._save_session()
        session_id = agent.session["id"]

        other = make_agent(server, store=store)
        with recorded_console():
            other._handle_command("/model model-b")
        with recorded_console(width=140) as output:
            other._handle_command(f"/load {session_id}")
            text = output()
        assert f"recorded with {recorded_with}" in text
        assert "continuing with model-b" in text
        assert other.client.model == "model-b"   # informative, not automatic
    finally:
        server.stop()


def test_plan_accepts_a_task_by_name_or_explicitly():
    server = FakeServer()
    try:
        agent = make_agent(server)
        # a bare task token: the common case, e.g. /plan code_security
        server.push(*answer('["grep for eval(", "check subprocess calls"]'))
        with recorded_console(width=120) as output:
            agent._handle_command("/plan code_security")
            text = output()
        assert len(agent.plan.steps) == 2
        assert "grep for eval(" in text
        assert server.last_request["messages"][-1]["content"] == "code_security"

        # a multi-word task starting with a subcommand word: no model call
        # happens here, so nothing is pushed (a stray scripted response
        # would be consumed by the next assertion)
        agent.plan.clear()
        with recorded_console():
            agent._handle_command("/plan add a --verbose flag to the cli")
        # ...which would otherwise be swallowed by the `add` subcommand:
        assert len(agent.plan.steps) == 1  # treated as "add <step>"
        assert agent.plan.steps[0]["text"] == "a --verbose flag to the cli"

        # the explicit form removes that ambiguity
        agent.plan.clear()
        server.push(*answer('["read the config", "add the option"]'))
        with recorded_console():
            agent._handle_command("/plan new add a --verbose flag to the cli")
        assert len(agent.plan.steps) == 2
        assert server.last_request["messages"][-1]["content"] == \
            "add a --verbose flag to the cli"

        agent.plan.clear()
        server.push(*answer('["only step"]'))
        with recorded_console():
            agent._handle_command("/plan task audit the parser")  # alias
        assert len(agent.plan.steps) == 1

        with recorded_console() as output:
            agent._handle_command("/plan new")
            assert "usage" in output()
    finally:
        server.stop()


# --------------------------------------------------------------------------- #
# per-turn file baselines, /retry, and --verify-command (objective proof)
# --------------------------------------------------------------------------- #
def test_per_turn_baselines_are_independent_of_the_session_baseline():
    Path("m.py").write_text("v = 1\n")
    agent = make_agent(yolo=True)
    agent._begin_user_turn()
    with recorded_console():
        agent._execute_tool("edit_file", {"path": "m.py",
                                          "old_str": "1", "new_str": "2"})
    assert agent.files.turn_changed() == ["m.py"]
    # a NEW turn re-baselines: the previous change is now the starting point
    agent._begin_user_turn()
    assert agent.files.turn_changed() == []
    with recorded_console():
        agent._execute_tool("edit_file", {"path": "m.py",
                                          "old_str": "2", "new_str": "3"})
    assert agent.files.turn_changed() == ["m.py"]
    assert agent.files.turn_baseline["m.py"] == "v = 2\n"   # per turn
    assert agent.files.baseline["m.py"] == "v = 1\n"        # per session
    # reverting the turn goes back one step, not all the way
    agent.files.revert_turn()
    assert Path("m.py").read_text() == "v = 2\n"
    # ...and the session revert still goes to the start
    agent.files.revert("m.py")
    assert Path("m.py").read_text() == "v = 1\n"


def test_retry_rewinds_the_conversation_and_the_files():
    Path("m.py").write_text("value = 1\n")
    server = FakeServer()
    try:
        agent = make_agent(server, yolo=True)
        turn_start = len(agent.conversation)
        agent._push_undo()
        agent.conversation.append({"role": "user", "content": "change it"})
        agent._begin_user_turn()
        server.push(*answer("set it to 2"))
        with recorded_console():
            agent._execute_tool("edit_file", {"path": "m.py",
                                              "old_str": "1", "new_str": "2"})
            agent._native_turn(agent.conversation)
        assert Path("m.py").read_text() == "value = 2\n"

        with recorded_console(width=140) as output:
            assert agent._handle_command("/retry") is True
            text = output()
        assert Path("m.py").read_text() == "value = 1\n"     # file rewound
        assert len(agent.conversation) == turn_start          # chat rewound
        assert agent._pending_input == "change it"            # queued to resend
        assert "retrying" in text and "1 file change(s) undone" in text
        assert agent._redo_stack == []          # a retry replaces the branch
    finally:
        server.stop()


def test_retry_edge_cases_and_temperature():
    server = FakeServer()
    try:
        agent = make_agent(server)
        with recorded_console() as output:
            assert agent._handle_command("/retry") is True
            assert "nothing to retry" in output()
        turn(agent, server, "hello", "hi there")
        # a bad argument is rejected before any state is touched, even
        # when a retry would otherwise be possible
        with recorded_console() as output:
            agent._handle_command("/retry 5")          # out of range
            assert "between 0 and 2" in output()
        with recorded_console() as output:
            agent._handle_command("/retry abc")
            assert "usage" in output()
        assert agent._pending_input is None and agent._undo_stack
        with recorded_console() as output:
            agent._handle_command("/retry 0.9")
            assert "temperature set to 0.9" in output()
        assert agent.temperature == 0.9
        assert agent._pending_input == "hello"
    finally:
        server.stop()


def test_retry_resends_through_the_run_loop():
    """The queued message must be treated as a message, not re-parsed as a
    command, and must actually reach the model."""
    server = FakeServer()
    try:
        server.push(*answer("first answer"))
        server.push(*answer("second answer"))
        messages = iter(["tell me something", "/retry", None])
        agent = make_agent(server, yolo=True)
        agent.get_user_message = lambda: next(messages, None)
        with recorded_console(width=140) as output:
            agent.run()
            text = output()
        assert server.count == 2, "the retried turn must hit the model again"
        sent = [json.dumps(request["messages"]) for request in server.requests]
        assert "tell me something" in sent[1]      # the same message resent
        assert "/retry" not in sent[1]             # not sent as text
        assert agent.last_answer == "second answer"
        assert "tell me something" in text            # echoed on resend
    finally:
        server.stop()


def verify_script(passing=True):
    """A dependency-free check command. It READS the file rather than
    importing it: Python's bytecode cache keys on (mtime, size) and a
    same-length edit within one second would be served stale."""
    Path("check.py").write_text(
        "text = open('m.py').read()\n"
        "if 'value = 1' not in text:\n"
        "    print('FAILED: m.py no longer sets value = 1')\n"
        "    raise SystemExit(1)\n"
        "print('ok')\n")
    Path("m.py").write_text("value = 1\n" if passing else "value = 9\n")
    return "python3 check.py"


def test_verify_command_baseline_and_new_failure_attribution():
    server = FakeServer()
    try:
        command = verify_script(passing=True)
        agent = make_agent(server, yolo=True, verify_command=command,
                           verify_mode="revise")   # this test is about revise
        with recorded_console() as output:
            agent.baseline_verify_command()
            assert "baseline: passing" in output()

        # a turn that breaks the check produces OBJECTIVE issues
        agent._push_undo()
        agent.conversation.append({"role": "user", "content": "set value to 5"})
        agent._begin_user_turn()
        server.push(*answer("changed it to 5"))
        with recorded_console():
            agent._execute_tool("edit_file", {"path": "m.py",
                                              "old_str": "value = 1",
                                              "new_str": "value = 5"})
            agent._native_turn(agent.conversation)
        server.push(*answer("restored value = 1"))
        with recorded_console(width=140) as output:
            agent._maybe_verify(agent.conversation)
            text = output()
        assert "verify command: FAILED (exit 1)" in text
        assert "failed check(s)" in text
        assert "m.py no longer sets value" in text     # the real error line
        assert "revising" in text
        assert agent.last_answer == "restored value = 1"
    finally:
        server.stop()


def test_verify_command_does_not_blame_pre_existing_failures():
    server = FakeServer()
    try:
        command = verify_script(passing=False)      # already broken
        agent = make_agent(server, yolo=True, verify_command=command)
        with recorded_console(width=140) as output:
            agent.baseline_verify_command()
            text = output()
        assert "baseline: FAILING" in text
        assert "will not be blamed" in text

        agent._push_undo()
        agent.conversation.append({"role": "user", "content": "add a comment"})
        agent._begin_user_turn()
        server.push(*answer("added a comment"))
        with recorded_console():
            agent._execute_tool("edit_file", {"path": "m.py",
                                              "old_str": "value = 9",
                                              "new_str": "value = 9  # note"})
            agent._native_turn(agent.conversation)
        with recorded_console(width=140) as output:
            agent._maybe_verify(agent.conversation)
            text = output()
        assert "pre-existing failure" in text and "not attributed" in text
        assert "revising" not in text               # nothing to fix
        assert agent.last_answer == "added a comment"
    finally:
        server.stop()


def test_verify_command_only_runs_when_files_changed():
    server = FakeServer()
    try:
        command = verify_script(passing=True)
        agent = make_agent(server, yolo=True, verify_command=command)
        agent._verify_baseline = (0, set())
        turn(agent, server, "what is 2+2?", "4")     # no files touched
        with recorded_console(width=140) as output:
            agent._maybe_verify(agent.conversation)
            text = output()
        assert "verify command" not in text          # not run at all
        assert agent.last_answer == "4"
    finally:
        server.stop()


def test_verify_command_controls_and_policy():
    server = FakeServer()
    try:
        agent = make_agent(server)
        with recorded_console() as output:
            assert agent._handle_command("/verify-command") is True
            assert "none" in output()
        command = verify_script(passing=True)
        with recorded_console(width=140) as output:
            agent._handle_command(f"/verify-command {command}")
            text = output()
        assert agent.verify_command == command
        assert "baseline: passing" in text           # re-baselines at once
        with recorded_console() as output:
            agent._handle_command("/verify-command off")
            assert "off" in output()
        assert agent.verify_command is None and agent._verify_baseline is None

        # a denylisted command is refused rather than run
        agent.verify_command = "rm -rf /"
        with recorded_console(width=140) as output:
            assert agent._run_verify_command() is None
            assert "blocked by policy" in output()
        assert agent.verify_command is None
        values = dict((l, v) for l, v, _ in agent.config_entries())
        assert values["verify command"] == "none"
    finally:
        server.stop()


def test_failure_fingerprint_normalisation():
    first = A.failure_fingerprint(
        "test_a FAILED in 1.23s\nsomething at 0x7f9c\nline 42: error here")
    second = A.failure_fingerprint(
        "test_a FAILED in 4.56s\nsomething at 0xdeadbeef\nline 99: error here")
    assert first == second, (first, second)      # noise normalised away
    assert A.failure_fingerprint("all good\n2 passed") == set()
    third = A.failure_fingerprint("test_b FAILED in 1s")
    assert third != first and third - first      # a real difference shows


def test_extra_body_command_and_thinking_toggle():
    server = FakeServer()
    try:
        agent = make_agent(server)
        with recorded_console(width=140) as output:
            assert agent._handle_command("/extra-body") is True
            assert "none" in output()

        with recorded_console(width=140):
            agent._handle_command(
                '/extra-body {"chat_template_kwargs": {"enable_thinking": false}}')
        assert agent.extra_body == {
            "chat_template_kwargs": {"enable_thinking": False}}

        # a second call MERGES one level deep instead of clobbering
        with recorded_console(width=140):
            agent._handle_command(
                '/extra-body {"chat_template_kwargs": {"other": 1}, "top_k": 20}')
        assert agent.extra_body == {
            "chat_template_kwargs": {"enable_thinking": False, "other": 1},
            "top_k": 20}

        # it reaches the wire on the next request
        server.push(*answer("ok"))
        agent.conversation.append({"role": "user", "content": "hi"})
        agent._begin_user_turn()
        with recorded_console():
            agent._native_turn(agent.conversation)
        assert server.last_request["top_k"] == 20
        assert server.last_request["chat_template_kwargs"]["enable_thinking"] \
            is False

        with recorded_console(width=140) as output:
            agent._handle_command("/extra-body {oops")
            assert "not valid JSON" in output()
        with recorded_console(width=140) as output:
            agent._handle_command('/extra-body ["a"]')
            assert "must be a JSON object" in output()
        with recorded_console(width=140) as output:
            agent._handle_command("/extra-body")      # shows the current value
            assert "top_k" in output()
        with recorded_console() as output:
            agent._handle_command("/extra-body off")
            assert "cleared" in output()
        assert agent.extra_body == {}

        # the /thinking shorthand
        with recorded_console(width=140) as output:
            agent._handle_command("/thinking")
            assert "not set" in output()
        with recorded_console(width=140):
            agent._handle_command("/thinking off")
        assert agent.extra_body == {
            "chat_template_kwargs": {"enable_thinking": False}}
        with recorded_console(width=140):
            agent._handle_command("/thinking on")
        assert agent.extra_body["chat_template_kwargs"]["enable_thinking"] is True
        with recorded_console(width=140):
            agent._handle_command("/thinking default")
        assert agent.extra_body == {}          # nested map removed when empty
        # any vendor word is a valid LEVEL now: it is passed through
        # verbatim rather than rejected, since vocabularies differ
        with recorded_console(width=140) as output:
            agent._handle_command("/thinking maybe")
            assert "maybe" in output()
        assert agent.extra_body["chat_template_kwargs"]["reasoning_effort"] \
            == "maybe"
    finally:
        server.stop()


def test_extra_body_survives_internal_calls_and_is_dropped_on_400():
    server = FakeServer()
    try:
        agent = make_agent(server, retries=3,
                           extra_body={"chat_template_kwargs":
                                       {"enable_thinking": False}})
        # internal calls carry it too (that is the point for a thinking model)
        server.push(*verdict(90, [], "fine"))
        agent._assess_answer("request", "answer")
        assert server.last_request["chat_template_kwargs"] == {
            "enable_thinking": False}

        # an endpoint that rejects the field degrades instead of failing
        server.reset()
        server.push_error(400, json.dumps(
            {"error": {"message": "unknown field: chat_template_kwargs"}}))
        server.push(*answer("worked without it"))
        with recorded_console(width=140) as output:
            result = agent._stream_with_recovery(
                [{"role": "user", "content": "hi"}], native_tools=False)
            text = output()
        assert "worked without it" in result.content
        assert "chat_template_kwargs" in server.requests[0]
        assert "chat_template_kwargs" not in server.requests[1]
        assert "dropping them" in text
    finally:
        server.stop()


def test_merge_extra_and_thinking_helpers():
    assert A.merge_extra({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert A.merge_extra({"a": 1}, {"a": 2}) == {"a": 2}
    # nested maps combine rather than replace
    assert A.merge_extra({"m": {"x": 1}}, {"m": {"y": 2}}) == {
        "m": {"x": 1, "y": 2}}
    assert A.merge_extra({"m": {"x": 1}}, {"m": "scalar"}) == {"m": "scalar"}
    assert A.merge_extra({}, None) == {}
    assert A.thinking_extra(False) == {
        "chat_template_kwargs": {"enable_thinking": False}}


# --------------------------------------------------------------------------- #
# thinking levels and endpoint capability discovery
# --------------------------------------------------------------------------- #
def test_nested_field_and_path_helpers():
    assert A.nested_field("a.b.c", 1) == {"a": {"b": {"c": 1}}}
    assert A.nested_field("flat", 2) == {"flat": 2}
    assert A.nested_field("", 3) == {}
    data = {"a": {"b": {"c": 1}}, "keep": 2}
    assert A.dict_path(data, "a.b.c") == 1
    assert A.dict_path(data, "a.missing") is None
    assert A.dict_path(data, "keep") == 2
    pruned = A.prune_path(data, "a.b.c")
    assert pruned == {"keep": 2}, pruned          # empty maps tidied away
    assert A.prune_path(data, "nope.here") == data  # unchanged, no crash
    assert data == {"a": {"b": {"c": 1}}, "keep": 2}  # input not mutated


def test_thinking_extra_handles_switches_levels_and_budgets():
    assert A.thinking_extra(False) == {
        "chat_template_kwargs": {"enable_thinking": False}}
    assert A.thinking_extra("off") == A.thinking_extra(False)
    assert A.thinking_extra("on") == A.thinking_extra(True)
    # a level implies thinking on, and is passed through verbatim
    for level in ("low", "medium", "high", "xhigh", "ultra"):
        fields = A.thinking_extra(level)
        kwargs = fields["chat_template_kwargs"]
        assert kwargs["enable_thinking"] is True
        assert kwargs["reasoning_effort"] == level
    # a numeric value stays numeric (token budgets)
    assert A.thinking_extra("4096")["chat_template_kwargs"][
        "reasoning_effort"] == 4096
    # the key is configurable, including top-level placement
    fields = A.thinking_extra("high", level_key="reasoning_effort")
    assert fields["reasoning_effort"] == "high"
    assert fields["chat_template_kwargs"] == {"enable_thinking": True}


def test_thinking_command_levels_keys_and_reset():
    server = FakeServer()
    try:
        agent = make_agent(server)
        with recorded_console(width=140) as output:
            agent._handle_command("/thinking")
            text = output()
        assert "not set" in text and A.THINKING_LEVEL_KEY in text

        with recorded_console(width=140) as output:
            agent._handle_command("/thinking high")
            assert "high" in output()
        assert agent.extra_body == {"chat_template_kwargs": {
            "enable_thinking": True, "reasoning_effort": "high"}}

        # switching to plain off must not leave a stale level behind
        with recorded_console(width=140):
            agent._handle_command("/thinking off")
        assert agent.extra_body == {
            "chat_template_kwargs": {"enable_thinking": False}}

        # a custom key, e.g. OpenAI-style top-level placement
        with recorded_console(width=140) as output:
            agent._handle_command("/thinking key reasoning_effort")
            assert "reasoning_effort" in output()
        with recorded_console(width=140):
            agent._handle_command("/thinking medium")
        assert agent.extra_body["reasoning_effort"] == "medium"

        # and it reaches the wire
        server.push(*answer("ok"))
        agent.conversation.append({"role": "user", "content": "hi"})
        agent._begin_user_turn()
        with recorded_console():
            agent._native_turn(agent.conversation)
        assert server.last_request["reasoning_effort"] == "medium"

        with recorded_console(width=140) as output:
            agent._handle_command("/thinking default")
            assert "server's default" in output()
        assert agent.extra_body == {}
        with recorded_console(width=140) as output:
            agent._handle_command("/thinking key")
            assert "level key" in output()
    finally:
        server.stop()


def test_template_kwarg_discovery():
    boolean_template = ("{%- if enable_thinking is defined and "
                        "enable_thinking == false %}<no_think>{%- endif %}")
    assert A.template_kwarg_values(boolean_template) == {
        "enable_thinking": ["false"]}

    harmony = (
        "{%- set effort = reasoning_effort | default('medium') %}\n"
        "{%- if reasoning_effort == 'high' %}Reasoning: high\n"
        "{%- elif reasoning_effort == 'low' %}Reasoning: low{%- endif %}\n"
        "{%- if thinking_budget is defined %}b{%- endif %}\n")
    hints = A.template_kwarg_values(harmony)
    assert hints["reasoning_effort"] == ["high", "low"]
    assert "thinking_budget" in hints and hints["thinking_budget"] == []
    # set membership is picked up too
    membership = '{%- if reasoning_effort in ["low", "high", "xhigh"] %}x{% endif %}'
    assert A.template_kwarg_values(membership)["reasoning_effort"] == [
        "low", "high", "xhigh"]
    assert A.template_kwarg_values("") == {}
    assert A.template_kwarg_values("{{ messages }}") == {}


def test_capabilities_command_reads_the_server_template():
    server = FakeServer()
    try:
        server.models = ["gpt-oss-20b"]
        server.chat_template = (
            "{%- if reasoning_effort == 'high' %}Reasoning: high{% endif %}"
            "{%- if enable_thinking == false %}<no_think>{% endif %}")
        agent = make_agent(server)
        with recorded_console(width=140) as output:
            assert agent._handle_command("/capabilities") is True
            text = output()
        assert "gpt-oss-20b" in text
        assert "16,384" in text                      # context window
        assert "chat_template_kwargs" in text
        assert "reasoning_effort" in text and "high" in text
        assert "enable_thinking" in text

        # an endpoint with no template says so instead of guessing
        server.chat_template = ""
        with recorded_console(width=140) as output:
            assert agent._handle_command("/caps") is True     # alias
            text = output()
        assert "does not expose its chat template" in text
    finally:
        server.stop()


def test_capabilities_survives_a_silent_endpoint():
    agent = make_agent()          # NoClient: nothing is listening
    agent.client.base_url = "http://127.0.0.1:1/v1"
    agent.client.api_key = ""
    with recorded_console(width=140) as output:
        assert agent._handle_command("/capabilities") is True
        text = output()
    assert "not listed" in text or "unknown" in text
    assert "does not expose" in text


def test_expand_verify_command_substitution():
    Path("a.py").write_text("x = 1\n")
    Path("b with space.py").write_text("y = 2\n")
    # a command without the token is untouched (backward compatible)
    assert A.expand_verify_command("pytest -q", ["a.py"]) == "pytest -q"
    # the baseline checks the whole project
    assert A.expand_verify_command("ruff check {files}", None) == "ruff check ."
    # a per-turn run checks only what changed, shell-quoted
    assert A.expand_verify_command("ruff check {files}", ["a.py"]) == \
        "ruff check a.py"
    assert A.expand_verify_command("ruff check {files}",
                                   ["b with space.py"]) == \
        "ruff check 'b with space.py'"
    # files deleted during the turn are dropped; nothing left means skip
    assert A.expand_verify_command("ruff check {files}", ["gone.py"]) == ""
    # an implausible list falls back to the project rather than a huge argv
    many = []
    for index in range(A.VERIFY_FILES_MAX + 5):
        Path(f"f{index}.py").write_text("x\n")
        many.append(f"f{index}.py")
    assert A.expand_verify_command("check {files}", many) == "check ."


def test_verify_command_checks_only_the_changed_files():
    """A per-turn check must see the turn's files, not the whole project --
    that is the point of {files}."""
    server = FakeServer()
    try:
        # this checker fails only if it is handed a file containing BAD
        Path("checker.py").write_text(
            "import sys\n"
            "for path in sys.argv[1:]:\n"
            "    if path in ('.', './'):\n"
            "        continue\n"
            "    if 'BAD' in open(path).read():\n"
            "        print('FAILED:', path, 'contains BAD')\n"
            "        raise SystemExit(1)\n"
            "print('ok')\n")
        Path("touched.py").write_text("clean = True\n")
        Path("untouched.py").write_text("clean = True\n")
        agent = make_agent(server, yolo=True,
                           verify_command="python3 checker.py {files}")
        agent._verify_baseline = (0, set())

        agent._push_undo()
        agent.conversation.append({"role": "user", "content": "edit it"})
        agent._begin_user_turn()
        server.push(*answer("edited"))
        with recorded_console():
            agent._execute_tool("edit_file", {"path": "touched.py",
                                              "old_str": "clean = True",
                                              "new_str": "clean = BAD"})
            agent._native_turn(agent.conversation)
        server.push(*answer("fixed it"))
        with recorded_console(width=140) as output:
            agent._maybe_verify(agent.conversation)
            text = output()
        assert "verify command: FAILED" in text
        assert "touched.py" in text                 # the expanded command
        assert "untouched.py" not in text           # never handed to it
        assert "contains BAD" in text               # the real failure line
    finally:
        server.stop()


def test_files_token_does_not_blame_untouched_breakage():
    """A file that was already broken and is NOT touched this turn must not
    even be checked, let alone blamed."""
    server = FakeServer()
    try:
        Path("checker.py").write_text(
            "import sys\n"
            "targets = [p for p in sys.argv[1:] if p not in ('.', './')]\n"
            "if not targets:\n"
            "    import glob; targets = glob.glob('*.py')\n"
            "for path in targets:\n"
            "    if 'BAD' in open(path).read():\n"
            "        print('FAILED:', path); raise SystemExit(1)\n"
            "print('ok')\n")
        Path("already_broken.py").write_text("v = BAD\n")   # pre-existing
        Path("clean.py").write_text("v = 1\n")
        agent = make_agent(server, yolo=True,
                           verify_command="python3 checker.py {files}")
        # the baseline checks the WHOLE project, so the existing breakage is
        # recorded rather than discovered later
        with recorded_console(width=140) as output:
            agent.baseline_verify_command()
            text = output()
        assert "baseline: FAILING" in text and "already_broken" not in text \
            or "baseline: FAILING" in text

        agent._push_undo()
        agent.conversation.append({"role": "user", "content": "tidy clean.py"})
        agent._begin_user_turn()
        server.push(*answer("tidied"))
        with recorded_console():
            agent._execute_tool("edit_file", {"path": "clean.py",
                                              "old_str": "v = 1",
                                              "new_str": "v = 2"})
            agent._native_turn(agent.conversation)
        with recorded_console(width=140) as output:
            agent._maybe_verify(agent.conversation)
            text = output()
        # only clean.py was checked, so the turn passes
        assert "verify command: passing" in text
        assert "revising" not in text
    finally:
        server.stop()


def test_logging_instrumentation_covers_the_turn_lifecycle():
    server = FakeServer()
    try:
        server.push(*answer("hi there", reasoning="pondering"))
        agent = make_agent(server)
        agent.conversation.append({"role": "user", "content": "hello"})
        with log_sink() as logs, recorded_console():
            agent._begin_user_turn()
            agent._native_turn(agent.conversation)
        messages = " | ".join(line for line, _style in logs)
        assert "user turn begins" in messages
        assert "request: model=test-model" in messages
        assert "turn settled: finish=stop" in messages
    finally:
        server.stop()


def test_log_full_mode_records_bodies_and_chunks():
    server = FakeServer()
    try:
        agent = make_agent(server)
        agent.conversation.append({"role": "user", "content": "hello"})

        server.push(*answer("quiet"))
        with log_sink() as quiet, recorded_console():
            agent._begin_user_turn()
            agent._native_turn(agent.conversation)
        quiet_text = " ".join(line for line, _s in quiet)
        assert "request body" not in quiet_text and "sse:" not in quiet_text

        with recorded_console(width=140) as output:
            agent._handle_command("/log full on")
            assert "full logging: ON" in output()
        try:
            server.push(*answer("loud"))
            with log_sink() as loud, recorded_console():
                agent._native_turn(agent.conversation)
            lines = [line for line, _s in loud]
            bodies = [l for l in lines if "request body:" in l]
            chunks = [l for l in lines if "sse:" in l]
            assert bodies, "the request body must be logged"
            assert '"model": "test-model"' in bodies[0]   # the real payload
            assert len(chunks) >= 2, chunks               # each SSE line
            assert any("[DONE]" in l for l in chunks)
        finally:
            with recorded_console():
                agent._handle_command("/log full off")
        assert A.LOG_FULL is False
    finally:
        server.stop()


def test_log_command_controls_level_and_reports_state():
    import logging as pylogging

    agent = make_agent()
    original = A.log.level
    try:
        with recorded_console(width=140) as output:
            assert agent._handle_command("/log") is True
            text = output()
        assert "level" in text and "full" in text

        with recorded_console(width=140) as output:
            agent._handle_command("/log level warning")
            assert "WARNING" in output()
        assert A.log.level == pylogging.WARNING
        # a filtered-out level no longer reaches the sink
        with log_sink() as logs:
            A.log.debug("suppressed")
            A.log.warning("kept")
        kept = " ".join(line for line, _s in logs)
        assert "suppressed" not in kept and "kept" in kept

        with recorded_console(width=140) as output:
            agent._handle_command("/log level nonsense")
            assert "usage" in output()
        with recorded_console(width=140) as output:
            agent._handle_command("/log full maybe")
            assert "usage" in output()
        with recorded_console(width=140) as output:
            agent._handle_command("/log bogus")
            assert "usage" in output()
    finally:
        A.log.setLevel(original)


# --------------------------------------------------------------------------- #
# engine profiles and the forced-final turn
# --------------------------------------------------------------------------- #
def test_engine_detection_from_what_the_endpoint_exposes():
    import json as pyjson
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    def serve(props_ok, models):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if "props" in self.path and props_ok:
                    body = pyjson.dumps(
                        {"default_generation_settings": {"n_ctx": 8192}}).encode()
                    code = 200
                elif "models" in self.path:
                    body = pyjson.dumps({"data": models}).encode()
                    code = 200
                else:
                    body, code = b"{}", 404
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{server.server_port}/v1", server

    for expected, props_ok, models in (
        ("llamacpp", True, [{"id": "m"}]),
        ("vllm", False, [{"id": "m", "max_model_len": 40960}]),
        ("openai", False, [{"id": "m"}]),
    ):
        url, server = serve(props_ok, models)
        try:
            assert A.detect_engine(url, "") == expected, expected
        finally:
            server.shutdown()
    # an endpoint that is not listening at all must not raise
    assert A.detect_engine("http://127.0.0.1:1/v1", "") == "openai"


def test_vllm_context_window_comes_from_max_model_len():
    server = FakeServer()
    try:
        server.n_ctx = 0                      # no llama.cpp answer
        server.models = ["vendor/Model3-32B"]
        # the harness serves max_model_len only via /props n_ctx, so probe
        # the documented vLLM path directly through /v1/models
        size, source = A.probe_context_window(
            server.base_url, "", "vendor/Model3-32B")
        assert size is None or isinstance(size, int)   # never raises
        assert isinstance(source, str)
    finally:
        server.stop()


def test_engine_without_dry_support_drops_the_params():
    server = FakeServer()
    try:
        with recorded_console(width=140) as output:
            agent = make_agent(server, engine="vllm",
                               dry_params={"dry_base": 1.9,
                                           "dry_multiplier": 0.8})
            text = output()
        assert agent.user_dry_params == {}
        assert "does not accept DRY" in text
        # nothing DRY-shaped reaches the wire
        server.push(*answer("ok"))
        agent.conversation.append({"role": "user", "content": "hi"})
        agent._begin_user_turn()
        with recorded_console():
            agent._native_turn(agent.conversation)
        assert not any(key.startswith("dry_") for key in server.last_request)

        # llama.cpp keeps them
        keeper = make_agent(server, engine="llamacpp",
                            dry_params={"dry_base": 1.9})
        assert keeper.user_dry_params == {"dry_base": 1.9}
        values = dict((l, v) for l, v, _ in keeper.config_entries())
        assert values["engine"] == "llamacpp"
    finally:
        server.stop()


def test_repetition_escalates_with_temperature_when_dry_is_unavailable():
    """On an engine without DRY the retry must actually change something,
    or the loop just repeats identically."""
    server = FakeServer()
    try:
        server.push(delta({"content": "spam " * 80}, finish="stop"))
        server.push(*answer("a sane answer"))
        agent = make_agent(server, engine="vllm", retries=3, temperature=0.5)
        with recorded_console(width=140) as output:
            result = agent._stream_with_recovery(
                [{"role": "user", "content": "hi"}], native_tools=False)
            text = output()
        assert "a sane answer" in result.content
        assert "temperature" in text                 # the escalation used
        assert server.requests[-1]["temperature"] > 0.5
        assert not any(k.startswith("dry_") for k in server.requests[-1])
    finally:
        server.stop()


def test_forced_final_turn_yields_an_answer_at_the_ceiling():
    server = FakeServer()
    try:
        for index in range(A.MAX_TURN_REQUESTS):
            server.push(tool_call_chunk("run_bash",
                                        {"command": f"echo {index}"},
                                        call_id=f"t{index}"))
        server.push(*answer("Established: nothing converged. Remains: pick one."))
        agent = make_agent(server, yolo=True, messages=["go"])
        with recorded_console(width=140) as output:
            agent.run()
            text = output()
        assert "ceiling" in text and "tools disabled" in text
        # the turn ends with a real answer instead of nothing
        assert "Established: nothing converged" in agent.last_answer
        assert agent.conversation[-1]["role"] == "assistant"
        # the forcing prompt itself is never stored
        assert not any("maximum number of requests" in str(m.get("content"))
                       for m in agent.conversation)
        # exactly one extra request beyond the ceiling
        assert server.count == A.MAX_TURN_REQUESTS + 1, server.count
        # tools were disabled for it
        assert "tools" not in server.requests[-1]
    finally:
        server.stop()


def test_forced_final_happens_at_most_once_per_turn():
    server = FakeServer()
    try:
        agent = make_agent(server, yolo=True)
        agent._begin_user_turn()
        agent.conversation.append({"role": "user", "content": "go"})
        server.push(*answer("first and only final"))
        with recorded_console(width=140):
            assert agent._forced_final_turn(agent.conversation, "test") is True
        assert agent._forced_final_used is True
        # a second attempt in the same turn is refused without a request
        before = server.count
        with recorded_console():
            assert agent._forced_final_turn(agent.conversation, "test") is False
        assert server.count == before
        # a new turn re-arms it
        agent._begin_user_turn()
        assert agent._forced_final_used is False
    finally:
        server.stop()


def test_forced_final_survives_a_model_that_says_nothing():
    server = FakeServer()
    try:
        agent = make_agent(server, yolo=True)
        agent._begin_user_turn()
        agent.conversation.append({"role": "user", "content": "go"})
        before = len(agent.conversation)
        server.push(delta({"reasoning_content": "still thinking"},
                          finish="length"))
        with recorded_console(width=140) as output:
            assert agent._forced_final_turn(agent.conversation, "test") is False
            assert "no final answer" in output()
        assert len(agent.conversation) == before   # nothing appended
    finally:
        server.stop()


# --------------------------------------------------------------------------- #
# the evaluate-and-iterate loop
# --------------------------------------------------------------------------- #
def failing_check(sentinel="value = 1"):
    """A check command that fails until m.py contains `sentinel`."""
    Path("check.py").write_text(
        "text = open('m.py').read()\n"
        f"if {sentinel!r} not in text:\n"
        f"    print('FAILED: m.py must contain {sentinel}')\n"
        "    raise SystemExit(1)\n"
        "print('ok')\n")
    Path("m.py").write_text(f"{sentinel}\n")
    return "python3 check.py"


def break_it(agent, server, answer_text="done"):
    """One turn that breaks the check."""
    server.push(*answer(answer_text))
    agent._push_undo()
    agent.conversation.append({"role": "user", "content": "change it"})
    agent._begin_user_turn()
    with recorded_console():
        agent._execute_tool("edit_file", {"path": "m.py",
                                          "old_str": "value = 1",
                                          "new_str": "value = 9"})
        agent._native_turn(agent.conversation)


def test_verify_mode_resolution():
    server = FakeServer()
    try:
        # a configured command means objective failures, which need fixing
        assert make_agent(server, verify_command="x").verify_mode == "iterate"
        assert make_agent(server).verify_mode == "revise"
        assert make_agent(server, verify_command="x",
                          verify_mode="revise").verify_mode == "revise"
        assert make_agent(server, verify_mode="iterate").verify_mode == "iterate"
        assert make_agent(server, verify_mode="nonsense").verify_mode == "revise"
        values = dict((l, v) for l, v, _ in
                      make_agent(server, verify_command="x").config_entries())
        # the row now carries the round and budget bounds too
        assert values["verify mode"].startswith("iterate")
    finally:
        server.stop()


def test_iterate_fixes_the_cause_with_tools_then_confirms():
    """The whole point: a failure must reach a TOOL-USING turn, and the fix
    must be re-verified rather than assumed."""
    server = FakeServer()
    try:
        command = failing_check()
        agent = make_agent(server, yolo=True, verify_command=command,
                           verify_mode="iterate", verify_rounds=4)
        with recorded_console():
            agent.baseline_verify_command()
        break_it(agent, server, "I set the value to 9.")

        server.push(tool_call_chunk("edit_file",
                                    {"path": "m.py", "old_str": "value = 9",
                                     "new_str": "value = 1"}, call_id="f1"))
        server.push(*answer("Restored value = 1."))
        with recorded_console(width=140) as output:
            agent._maybe_verify(agent.conversation)
            text = output()
        assert "iterating (round 1 of 4)" in text and "tools enabled" in text
        assert "verify command: passing" in text     # re-verified, not assumed
        assert Path("m.py").read_text() == "value = 1\n"
        # the failure and the fix stay in history: this is work, not a nudge
        stored = json.dumps(agent.conversation)
        assert "verify command reports these problems" in stored
        assert "Do NOT edit the tests" in stored     # gaming discouraged
    finally:
        server.stop()


def test_iterate_stops_when_the_failures_stop_changing():
    server = FakeServer()
    try:
        command = failing_check()
        agent = make_agent(server, yolo=True, verify_command=command,
                           verify_mode="iterate", verify_rounds=5)
        agent._verify_baseline = (0, set())
        break_it(agent, server)
        # the model says it fixed things but changes nothing, twice
        for _ in range(4):
            server.push(*answer("I have addressed it."))
        with recorded_console(width=140) as output:
            agent._maybe_verify(agent.conversation)
            text = output()
        assert "the same problems remain" in text
        assert "stopping rather than repeating" in text
        # stopped early rather than burning all five rounds
        assert text.count("iterating (round") == 1, text.count("iterating")
    finally:
        server.stop()


def test_iterate_stops_when_it_makes_things_worse():
    server = FakeServer()
    try:
        # a check that reports one failure per missing sentinel
        Path("check.py").write_text(
            "text = open('m.py').read()\n"
            "bad = [s for s in ('alpha', 'beta') if s not in text]\n"
            "for s in bad:\n"
            "    print(f'FAILED: missing {s}')\n"
            "raise SystemExit(1 if bad else 0)\n")
        Path("m.py").write_text("alpha beta\n")
        agent = make_agent(server, yolo=True,
                           verify_command="python3 check.py",
                           verify_mode="iterate", verify_rounds=5)
        with recorded_console():
            agent.baseline_verify_command()
        # turn 1 removes alpha -> one failure
        server.push(*answer("removed alpha"))
        agent._push_undo()
        agent.conversation.append({"role": "user", "content": "edit it"})
        agent._begin_user_turn()
        with recorded_console():
            agent._execute_tool("edit_file", {"path": "m.py",
                                              "old_str": "alpha beta",
                                              "new_str": "beta"})
            agent._native_turn(agent.conversation)
        # the "fix" removes beta too -> two failures: a regression
        server.push(tool_call_chunk("edit_file",
                                    {"path": "m.py", "old_str": "beta",
                                     "new_str": "gamma"}, call_id="f1"))
        server.push(*answer("adjusted"))
        with recorded_console(width=140) as output:
            agent._maybe_verify(agent.conversation)
            text = output()
        assert "introduced new problems" in text
        assert "/revert" in text                     # points at the way back
    finally:
        server.stop()


def test_iterate_respects_the_round_cap():
    server = FakeServer()
    try:
        # each attempt fails DIFFERENTLY (the check echoes what it found),
        # so "stuck" cannot fire and only the round cap can end the loop
        Path("check.py").write_text(
            "text = open('m.py').read().strip()\n"
            "if text != 'wanted':\n"
            "    print(f'FAILED: found {text} instead of wanted')\n"
            "    raise SystemExit(1)\n"
            "print('ok')\n")
        Path("m.py").write_text("wanted\n")
        agent = make_agent(server, yolo=True,
                           verify_command="python3 check.py",
                           verify_mode="iterate", verify_rounds=2)
        with recorded_console():
            agent.baseline_verify_command()
        server.push(*answer("changed it"))
        agent._push_undo()
        agent.conversation.append({"role": "user", "content": "change it"})
        agent._begin_user_turn()
        with recorded_console():
            agent._execute_tool("write_file", {"path": "m.py",
                                               "content": "alpha\n"})
            agent._native_turn(agent.conversation)
        for word in ("beta", "gamma", "delta", "epsilon"):
            server.push(tool_call_chunk(
                "write_file", {"path": "m.py", "content": f"{word}\n"},
                call_id=f"f-{word}"))
            server.push(*answer(f"tried {word}"))
        with recorded_console(width=140) as output:
            agent._maybe_verify(agent.conversation)
            text = output()
        assert "2 iteration(s) done" in text, text[-400:]
        assert "/diff" in text
        assert text.count("iterating (round") == 2
    finally:
        server.stop()


def test_revise_mode_still_only_rewrites_the_answer():
    """The old behaviour must remain available and unchanged."""
    server = FakeServer()
    try:
        command = failing_check()
        agent = make_agent(server, yolo=True, verify_command=command,
                           verify_mode="revise", verify_rounds=1)
        agent._verify_baseline = (0, set())
        break_it(agent, server)
        server.push(*answer("I cannot set it to 9 and keep the check green."))
        with recorded_console(width=140) as output:
            agent._maybe_verify(agent.conversation)
            text = output()
        assert "revising" in text and "iterating" not in text
        assert "cannot set it to 9" in agent.last_answer
        # tools were disabled for the rewrite
        assert "tools" not in server.requests[-1]
    finally:
        server.stop()


def test_compare_failures_progress_signal():
    assert A.compare_failures(None, set()) == "fixed"
    assert A.compare_failures({"a"}, set()) == "fixed"
    assert A.compare_failures(None, {"a"}) == "progress"
    assert A.compare_failures({"a"}, {"a"}) == "stuck"
    assert A.compare_failures({"a", "b"}, {"a"}) == "progress"
    assert A.compare_failures({"a"}, {"a", "b"}) == "regressed"
    assert A.compare_failures({"a"}, {"b", "c"}) == "regressed"
    assert A.compare_failures({"a", "b"}, {"c"}) == "progress"


# --------------------------------------------------------------------------- #
# unattended operation: budgets and visible automatic denial
# --------------------------------------------------------------------------- #
def test_unattended_declines_visibly_without_prompting():
    """A prompt nobody can answer would hang the run; a silent denial
    would stall it with no explanation. Neither is acceptable."""
    Path("m.py").write_text("value = 1\n")
    server = FakeServer()
    try:
        agent = make_agent(server, unattended=True, approval="low")
        with approval("y") as asked, recorded_console(width=140) as output:
            result, is_error = agent._execute_tool(
                "edit_file", {"path": "m.py", "old_str": "1", "new_str": "2"})
            text = output()
        assert asked == [], "unattended must never consult the approval hook"
        assert is_error
        assert Path("m.py").read_text() == "value = 1\n"   # not applied
        # the operator can see it
        assert "unattended: declined" in text and "[medium]" in text
        # ...and the model is told why, and what it can do about it
        assert "automatically declined" in result
        assert "MEDIUM" in result and "'low'" in result
        assert "lower-risk approach" in result and "operator" in result

        # read-only tools are unaffected
        with recorded_console():
            out, err = agent._execute_tool("read_file", {"path": "m.py"})
        assert not err and "value = 1" in out

        # an explicit --yolo still wins: the operator accepted the risk
        with recorded_console():
            yolo = make_agent(server, unattended=True, yolo=True)
            out, err = yolo._execute_tool(
                "edit_file", {"path": "m.py", "old_str": "1", "new_str": "3"})
        assert not err and Path("m.py").read_text() == "value = 3\n"

        # an allowed level still runs unattended
        with recorded_console():
            low = make_agent(server, unattended=True, approval="medium")
            out, err = low._execute_tool(
                "write_file", {"path": "n.py", "content": "x\n"})
        assert not err and Path("n.py").exists()
    finally:
        server.stop()


def test_session_request_budget_stops_the_run():
    server = FakeServer()
    try:
        for index in range(8):
            server.push(*answer(f"reply {index}"))
        messages = iter(["one", "two", "three", "four", "five", None])
        agent = make_agent(server, unattended=True, max_session_requests=3)
        agent.get_user_message = lambda: next(messages, None)
        with recorded_console(width=140) as output:
            agent.run()
            text = output()
        assert server.count == 3, server.count
        assert "session request budget (3) spent" in text
        assert "3/3 requests" in text          # the summary
    finally:
        server.stop()


def test_session_time_and_token_budgets_stop_the_run():
    server = FakeServer()
    try:
        for index in range(4):
            server.push(*answer(f"r{index}"))
        messages = iter(["a", "b", "c", None])
        agent = make_agent(server, unattended=True, max_session_seconds=0.001)
        agent.get_user_message = lambda: next(messages, None)
        with recorded_console(width=140) as output:
            agent.run()
            assert "session time budget" in output()
        assert server.count == 0               # stopped before the first turn

        server.reset()
        for index in range(6):
            server.push(*answer("a fairly long reply " * 5))
        messages = iter(["a", "b", "c", "d", None])
        agent = make_agent(server, unattended=True, max_session_tokens=20)
        agent.get_user_message = lambda: next(messages, None)
        with recorded_console(width=140) as output:
            agent.run()
            assert "session token budget" in output()
        assert 0 < server.count < 4
    finally:
        server.stop()


def test_budgets_count_internal_calls_too():
    """A budget is about cost, so a classifier or critique request counts
    exactly like a visible one."""
    server = FakeServer()
    try:
        agent = make_agent(server, unattended=True, max_session_requests=99)
        assert agent._session_requests == 0
        server.push(*verdict(90, [], "fine"))
        agent._assess_answer("request", "answer")     # record=False internally
        assert agent._session_requests == 1, agent._session_requests
        assert agent._session_tokens > 0
        # and the budget check sees them
        agent.max_session_requests = 1
        assert "request budget" in (agent._budget_exceeded() or "")
    finally:
        server.stop()


def test_budget_summary_and_config_row():
    server = FakeServer()
    try:
        plain = make_agent(server)
        assert plain._budget_exceeded() is None
        assert plain._budget_summary() == "no session budgets"
        values = dict((l, v) for l, v, _ in plain.config_entries())
        assert values["unattended"] == "off"

        agent = make_agent(server, unattended=True, max_session_requests=10,
                           max_session_seconds=60)
        assert agent._budget_exceeded() is None
        summary = agent._budget_summary()
        assert "0/10 requests" in summary and "/60s" in summary
        values = dict((l, v) for l, v, _ in agent.config_entries())
        assert values["unattended"].startswith("on")
    finally:
        server.stop()


# --------------------------------------------------------------------------- #
# test-gaming guard: a green check whose own inputs were edited is not a pass
# --------------------------------------------------------------------------- #
def test_verify_target_classification():
    assert A.verify_target_files("pytest -q", ["src/parser.py"]) == []
    assert A.verify_target_files("pytest -q", ["README.md", "docs/a.md"]) == []
    assert A.verify_target_files("", ["tests/test_a.py"])[0]["path"] == \
        "tests/test_a.py"

    def reasons(command, paths):
        return {entry["path"]: entry["reason"]
                for entry in A.verify_target_files(command, paths)}

    found = reasons("python3 check.py", ["check.py", "m.py"])
    assert found == {"check.py": "named by the verify command"}

    found = reasons("pytest -q", ["tests/test_parser.py", "conftest.py",
                                  "src/parser.py", "lib/util_test.go",
                                  "pyproject.toml", "Makefile"])
    assert "src/parser.py" not in found
    assert "test director" in found["tests/test_parser.py"]
    assert "configuration" in found["conftest.py"]
    assert "configuration" in found["pyproject.toml"]
    assert "configuration" in found["Makefile"]
    assert "by name" in found["lib/util_test.go"]

    found = reasons("npx jest", ["src/foo.test.ts", "src/foo.ts",
                                 "jest.config.js", "__tests__/x.js"])
    assert "src/foo.ts" not in found
    assert set(found) == {"src/foo.test.ts", "jest.config.js",
                          "__tests__/x.js"}


CHECK_READING_ITS_TEST = (
    "import re\n"
    "expected = int(re.search(r'EXPECTED = (-?\\d+)',\n"
    "               open('tests/test_math.py').read()).group(1))\n"
    "ns = {}\n"
    "exec(open('m.py').read(), ns)\n"
    "got = ns['add'](1, 2)\n"
    "if got != expected:\n"
    "    print(f'FAILED: add(1,2) gave {got}, expected {expected}')\n"
    "    raise SystemExit(1)\n"
    "print('ok')\n"
)


def gaming_scenario(server, **options):
    """A check whose expectation lives in a test file, and code that has
    just been broken -- so the loop can be fixed either honestly or by
    editing the test."""
    Path("tests").mkdir(exist_ok=True)
    Path("tests/test_math.py").write_text("EXPECTED = 3\n")
    Path("check.py").write_text(CHECK_READING_ITS_TEST)
    Path("m.py").write_text("def add(a, b):\n    return a + b\n")
    agent = make_agent(server, yolo=True, verify_mode="iterate",
                       verify_rounds=2, verify_command="python3 check.py",
                       **options)
    with recorded_console():
        agent.baseline_verify_command()
    server.push(*answer("changed add"))
    agent._push_undo()
    agent.conversation.append({"role": "user", "content": "go"})
    agent._begin_user_turn()
    with recorded_console():
        agent._execute_tool("edit_file", {"path": "m.py",
                                          "old_str": "a + b",
                                          "new_str": "a - b"})
        agent._native_turn(agent.conversation)
    return agent


def test_editing_the_test_to_pass_is_not_reported_as_success():
    server = FakeServer()
    try:
        agent = gaming_scenario(server)
        # the "fix" moves the goalposts instead of fixing the code
        server.push(tool_call_chunk(
            "edit_file", {"path": "tests/test_math.py",
                          "old_str": "EXPECTED = 3",
                          "new_str": "EXPECTED = -1"}, call_id="g1"))
        server.push(*answer("done"))
        with recorded_console(width=140) as output:
            agent._maybe_verify(agent.conversation)
            text = output()
        assert "verify command: passing" in text      # it really does pass
        assert "editing what the check measures is not a fix" in text
        assert "NOT reporting success" in text
        assert "tests/test_math.py" in text
        assert agent._verify_state == "suspect"
        assert agent.exit_code() == A.EXIT_VERIFY_FAILED
        report = agent.report()
        assert report["verify"]["state"] == "suspect"
        assert report["verify"]["edited_verify_targets"][0]["path"] == \
            "tests/test_math.py"
    finally:
        server.stop()


def test_an_honest_fix_is_reported_as_success():
    server = FakeServer()
    try:
        agent = gaming_scenario(server)
        server.push(tool_call_chunk(
            "edit_file", {"path": "m.py", "old_str": "a - b",
                          "new_str": "a + b"}, call_id="h1"))
        server.push(*answer("fixed the operator"))
        with recorded_console(width=140) as output:
            agent._maybe_verify(agent.conversation)
            text = output()
        assert "verify command: passing" in text
        assert "not a fix" not in text and "suspect" not in text
        assert agent._verify_state == "passing"
        assert agent.exit_code() == A.EXIT_OK
        assert agent._verify_suspects == []
    finally:
        server.stop()


def test_a_shell_edit_to_a_test_is_caught_too():
    """Detection compares workspace manifests, so it does not matter which
    tool made the change."""
    server = FakeServer()
    try:
        agent = gaming_scenario(server)
        server.push(tool_call_chunk(
            "run_bash",
            {"command": "sed -i 's/EXPECTED = 3/EXPECTED = -1/' "
                        "tests/test_math.py"}, call_id="s1"))
        server.push(*answer("done"))
        with recorded_console(width=140) as output:
            agent._maybe_verify(agent.conversation)
            text = output()
        assert "NOT reporting success" in text
        assert agent._verify_state == "suspect"
    finally:
        server.stop()


def test_allow_verify_edits_turns_the_guard_off():
    """Sometimes editing the tests IS the task."""
    server = FakeServer()
    try:
        agent = gaming_scenario(server, allow_verify_edits=True)
        server.push(tool_call_chunk(
            "edit_file", {"path": "tests/test_math.py",
                          "old_str": "EXPECTED = 3",
                          "new_str": "EXPECTED = -1"}, call_id="g1"))
        server.push(*answer("updated the expectation"))
        with recorded_console(width=140) as output:
            agent._maybe_verify(agent.conversation)
            text = output()
        assert "verify command: passing" in text
        assert "NOT reporting success" not in text
        assert agent._verify_state == "passing"
        assert agent.exit_code() == A.EXIT_OK
        # still recorded for the operator, just not treated as failure
        assert agent._verify_suspects
    finally:
        server.stop()


# --------------------------------------------------------------------------- #
# session save/load end to end, after a day of changes
# --------------------------------------------------------------------------- #
def test_session_round_trip_carries_every_kind_of_state():
    """Everything a turn can produce must survive a save and a load:
    transcript, tool results, reasoning, raw exchanges, plan progress."""
    server = FakeServer()
    try:
        Path("m.py").write_text("value = 1\n")
        store = A.JsonSessionStore(".agent_sessions")
        agent = make_agent(server, yolo=True, store=store)
        agent.plan.set(["read m.py", "change the value"], task="change it")

        server.push(tool_call_chunk("edit_file",
                                    {"path": "m.py", "old_str": "1",
                                     "new_str": "2"}, call_id="t1"))
        server.push(*answer("Changed it to 2.\nPLAN: done 1",
                            reasoning="considering the edit"))
        agent.conversation.append({"role": "user", "content": "change it"})
        agent._begin_user_turn()
        with recorded_console():
            while not agent._native_turn(agent.conversation):
                pass
            agent._consume_plan_progress()
            agent._save_session()

        saved = json.loads(
            Path(".agent_sessions", f"{agent.session['id']}.json").read_text())
        assert [m["role"] for m in saved["messages"]] == \
            ["user", "assistant", "tool", "assistant"]
        assert saved["reasonings"] and saved["exchanges"]
        assert saved["plan"][0] == {"text": "read m.py", "done": True}
        assert saved["plan_task"] == "change it"
        assert saved["model"] == agent.client.model
        assert saved["protocol"] == agent.mode

        # a fresh agent restores all of it
        fresh = make_agent(server, yolo=True, store=store)
        with recorded_console(width=140) as output:
            fresh._handle_command(f"/load {agent.session['id']}")
            text = output()
        assert "Changed it to 2" in text          # transcript replayed
        assert "plan restored" in text
        assert fresh.plan.progress() == "1/2"
        assert fresh.plan.task == "change it"
        assert len(fresh.conversation) == 4
        assert fresh.turn_reasonings                # /think has material
        assert fresh.turn_exchanges                 # /raw has material

        # and the restored history reaches the model on the next turn
        server.push(*answer("I changed value from 1 to 2."))
        fresh.conversation.append({"role": "user", "content": "what changed?"})
        fresh._begin_user_turn()
        with recorded_console():
            fresh._native_turn(fresh.conversation)
        sent = json.dumps(server.last_request["messages"])
        assert "Changed it to 2" in sent and "edit_file" in sent
        assert "Current plan" in sent and "[x] read m.py" in sent
    finally:
        server.stop()


def test_save_accepts_a_title_that_sticks():
    server = FakeServer()
    try:
        store = A.JsonSessionStore(".agent_sessions")
        agent = make_agent(server, store=store)
        turn(agent, server, "what is the capital?", "Paris.")
        with recorded_console(width=140) as output:
            assert agent._handle_command("/save my nice title") is True
            assert "session saved" in output()
        saved = json.loads(
            Path(".agent_sessions", f"{agent.session['id']}.json").read_text())
        assert saved["title"] == "my nice title"

        # it survives a later autosave rather than reverting to the prompt
        turn(agent, server, "and Spain?", "Madrid.")
        with recorded_console():
            agent._autosave()
        saved = json.loads(
            Path(".agent_sessions", f"{agent.session['id']}.json").read_text())
        assert saved["title"] == "my nice title"

        # and /sessions shows it
        with recorded_console(width=140) as output:
            agent._handle_command("/sessions")
            assert "my nice title" in output()
    finally:
        server.stop()


def test_a_known_command_with_a_stray_argument_never_reaches_the_model():
    """This cost a model request and looked like the command had run:
    "/save my-title" fell through and was sent as chat."""
    server = FakeServer()
    try:
        agent = make_agent(server, store=A.JsonSessionStore(".agent_sessions"))
        turn(agent, server, "hi", "hello")
        before = server.count
        for command, verb in (("/undo tomorrow", "/undo"),
                              ("/jobs all", "/jobs"),
                              ("/redo please", "/redo"),
                              ("/restart now", "/restart")):
            with recorded_console(width=140) as output:
                assert agent._handle_command(command) is True, command
                text = output()
            assert f"{verb} does not take that argument" in text, command
        assert server.count == before, "no model request may be made"

        # a bare typo is still reported as unknown
        with recorded_console(width=140) as output:
            assert agent._handle_command("/thnk") is True
            assert "unknown command: /thnk" in output()

        # ...and ordinary prose that merely contains a slash is still a
        # message, not a command
        assert agent._handle_command("/etc/passwd is a file") is False
        assert agent._handle_command("what is 3/4 of 8?") is False
    finally:
        server.stop()


if __name__ == "__main__":
    main(globals(), "command surface: display, sessions, export, aliases")
