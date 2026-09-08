"""Feature systems: skills, memory, context compression, sessions,
system-prompt override, config/help surfaces, stats and token budgets.

These exercise the Agent against a scripted fake endpoint, so the
model-facing prompts and the wire payloads are both asserted.
"""

import json
import os
from pathlib import Path

from harness import (  # noqa: E402
    A, FakeServer, answer, delta, main, make_agent, recorded_console,
    tool_call_chunk,
)


# --------------------------------------------------------------------------- #
# skills: one active, injected at the system level in BOTH protocols
# --------------------------------------------------------------------------- #
def test_skill_name_mangling_is_content_addressed():
    first = "Terse mode\nAnswer in at most two sentences."
    second = "Terse mode\nAnswer in at most three sentences."
    name = A.mangle_skill_name(first)
    assert name.startswith("terse-mode-")
    assert name == A.mangle_skill_name(first)          # deterministic
    assert name != A.mangle_skill_name(second)         # content-hashed


def test_skill_store_and_activation():
    skills = A.SkillManager(".skills", search_dirs=[".skills"])
    assert skills.list() == []
    assert "not found" in skills.activate("nope")
    text = "Pirate mode\nAlways answer like a pirate."
    assert "imported" in skills.save(text)
    skills.save(text)                                   # idempotent
    assert len(skills.list()) == 1
    name = skills.list()[0]
    assert "active" in skills.activate(name)
    assert skills.active_name == name and "pirate" in skills.active_text
    assert "no skill" in skills.deactivate() and skills.active_name is None


def test_skill_injection_in_both_protocols():
    server = FakeServer()
    try:
        skills = A.SkillManager(".skills", search_dirs=[".skills"])
        skills.save("Terse mode\nAnswer in at most two sentences.")
        skills.activate(skills.list()[0])

        agent = make_agent(server, skills=skills)
        # text protocol: appended to the system prompt, tool machinery intact
        prompt = agent.text_system_prompt
        assert "Active skill" in prompt and "two sentences" in prompt
        assert "<tool_call>" in prompt and "## read_file" in prompt

        # native protocol: prepended per request, never stored in history
        server.push(*answer("ok"))
        agent.conversation.append({"role": "user", "content": "hi"})
        agent._begin_user_turn()
        agent._native_turn(agent.conversation)
        first = server.last_request["messages"][0]
        assert first["role"] == "system" and "Active skill" in first["content"]
        assert agent.conversation[0]["role"] == "user"   # not persisted

        # deactivating removes it from the very next request
        agent._set_skill("off")
        server.push(*answer("ok"))
        agent._native_turn(agent.conversation)
        assert server.last_request["messages"][0]["role"] == "user"
    finally:
        server.stop()


def test_skill_numeric_selector_and_restart():
    skills = A.SkillManager(".skills", search_dirs=[".skills"])
    skills.save("Alpha skill\nfirst")
    skills.save("Beta skill\nsecond")
    agent = make_agent(skills=skills)
    with recorded_console():
        agent._handle_command("/skills")
        agent._handle_command("/skill 1")
    assert skills.active_name == sorted(skills.list())[0]
    with recorded_console():
        agent._handle_command("/restart")
    assert skills.active_name is None                   # fresh start


# --------------------------------------------------------------------------- #
# memory: model-distilled notes, searchable, multi-active
# --------------------------------------------------------------------------- #
def test_memory_store_search_and_titles():
    memories = A.MemoryManager(".memories")
    assert memories.list() == [] and memories.search("x") == []
    first = memories.save("# Redis socket fix\nUse `unix:///tmp/redis.sock`.")
    memories.save("# Build cache\nccache halves rebuild time. Redis unrelated.")
    listing = memories.list()
    assert len(listing) == 2 and listing[0]["date"]
    assert memories.title(first) == "Redis socket fix"

    hits = memories.search("redis SOCKET")              # case-insensitive
    assert hits[0]["name"] == first and hits[0]["score"] >= 2
    assert any("unix" in snippet for snippet in hits[0]["snippets"])

    assert "loaded" in memories.load(first)
    assert "not found" in memories.load("ghost")
    assert "unloaded" in memories.unload(first) and memories.active == {}


def test_memory_save_distills_via_the_model():
    server = FakeServer()
    try:
        memories = A.MemoryManager(".memories")
        agent = make_agent(server, memories=memories)
        agent.conversation += [
            {"role": "user", "content": "how big is the ctx?"},
            {"role": "assistant", "content": "32k, via /props"},
        ]
        agent._begin_user_turn()
        server.push(*answer("# Context probing\nUse `/props` for n_ctx.",
                            reasoning="summarising"))
        with recorded_console():
            agent._memory_command("save focus on config")

        sent = server.last_request["messages"][0]["content"]
        assert "memory-keeper" in sent                       # the prompt
        assert "focus on: focus on config" in sent           # focus guidance
        assert "how big is the ctx?" in sent                 # transcript included
        names = [m["name"] for m in memories.list()]
        assert any(n.startswith("context-probing-") for n in names)
        saved = memories.read(
        next(n for n in names if n.startswith("context")))
        assert "<think>" not in saved and "summarising" not in saved
    finally:
        server.stop()


def test_multiple_memories_inject_together():
    server = FakeServer()
    try:
        memories = A.MemoryManager(".memories")
        first = memories.save("# Redis socket fix\nUse the unix socket.")
        second = memories.save("# Build cache\nccache helps.")
        memories.load(first)
        memories.load(second)
        agent = make_agent(server, memories=memories)

        assert "Relevant memories" in agent.text_system_prompt
        assert "Redis socket fix" in agent.text_system_prompt

        server.push(*answer("ok"))
        agent.conversation.append({"role": "user", "content": "hi"})
        agent._begin_user_turn()
        agent._native_turn(agent.conversation)
        injected = server.last_request["messages"][0]
        assert injected["role"] == "system"
        assert f"Memory: {first}" in injected["content"]
        assert f"Memory: {second}" in injected["content"]
    finally:
        server.stop()


# --------------------------------------------------------------------------- #
# context compression
# --------------------------------------------------------------------------- #
def test_compression_tail_is_template_safe():
    """llama.cpp's chat template rejects a tail starting on an orphaned
    tool turn (or an assistant tool-call whose result was cut away)."""
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1",
                         "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "big output"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    _head, tail, folded = A.split_for_compression(list(messages), keep=3)
    assert folded == 3 and tail[0]["content"] == "a1"
    _head, tail, folded = A.split_for_compression(list(messages), keep=4)
    assert folded == 3 and tail[0]["content"] == "a1"   # orphan walked forward
    assert A.split_for_compression(list(messages), keep=99) is None
    assert "read_file({})" in A.render_for_summary(messages)
    assert A.abbr_tokens(78825) == "78K" and A.abbr_tokens(1500) == "1.5K"


def test_manual_compact_replaces_history_and_is_bounded():
    server = FakeServer()
    try:
        agent = make_agent(server, max_tokens=8192,
                           store=A.JsonSessionStore(".agent_sessions"))
        agent.conversation[:] = [
            {"role": "user" if i % 2 == 0 else "assistant",
             "content": f"message {i} " * 40}
            for i in range(8)
        ]
        server.push(*answer("Dense summary of the earlier work.",
                            reasoning="thinking"))
        before = len(agent.conversation)
        with recorded_console() as output:
            agent._handle_command("/compact")   # the user-facing path
            text = output()
        assert len(agent.conversation) == 3     # summary + COMPRESS_KEEP=2
        kept, folded = 2, before - 2
        head = agent.conversation[0]
        assert head["role"] == "user"
        assert f"Compressed context -- {folded} earlier turns" in head["content"]
        assert "Dense summary" in head["content"] and "<think>" not in head["content"]
        # a summary request must not inherit the big turn budget
        assert server.last_request["max_tokens"] == 2048
        assert "Summarize the following conversation" in \
            server.last_request["messages"][0]["content"]
        assert "compressed" in text
    finally:
        server.stop()


def test_compression_refuses_to_grow_the_context():
    """A thinking model's reasoning fallback can be longer than what it
    replaces; splicing that in would make things worse."""
    server = FakeServer()
    try:
        agent = make_agent(server)
        agent.conversation[:] = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
            for i in range(8)
        ]
        before = list(agent.conversation)
        server.push(delta({"reasoning_content": "y " * 4000}, finish="length"))
        with recorded_console():
            assert agent._compress() is None
        assert agent.conversation == before              # untouched
    finally:
        server.stop()


def test_autocompress_threshold_and_controls():
    server = FakeServer()
    try:
        agent = make_agent(server, autocompress=85,
                           tracker=A.TokenTracker(1000, "test-model"))
        agent.conversation[:] = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
            for i in range(9)
        ]
        agent.last_result = A.TurnResult(
            usage={"prompt_tokens": 700, "completion_tokens": 100})
        assert agent._maybe_autocompress() is False      # 80% < 85%

        agent.last_result = A.TurnResult(
            usage={"prompt_tokens": 800, "completion_tokens": 100})
        server.push(*answer("auto summary"))
        with recorded_console():
            assert agent._maybe_autocompress() is True   # 90% >= 85%
        assert len(agent.conversation) == 1 + 3          # keep = 9 // 3
        with recorded_console():
            agent._autocompress_command("off")
            assert agent._maybe_autocompress() is False
            agent._autocompress_command("on")
            assert agent.autocompress_percent == 85
            agent._autocompress_command("70")
            assert agent.autocompress_percent == 70
            agent._autocompress_command("400")           # out of range
            assert agent.autocompress_percent == 70
    finally:
        server.stop()


# --------------------------------------------------------------------------- #
# sessions
# --------------------------------------------------------------------------- #
def test_session_save_load_restores_skill_and_history():
    server = FakeServer()
    try:
        skills = A.SkillManager(".skills", search_dirs=[".skills"])
        skills.save("Terse mode\nBe brief.")
        name = skills.list()[0]
        skills.activate(name)
        store = A.JsonSessionStore(".agent_sessions")
        agent = make_agent(server, skills=skills, store=store)
        agent.conversation += [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        with recorded_console():
            agent._save_session()
        assert store.list()

        skills.deactivate()
        agent.conversation.clear()
        with recorded_console() as output:
            agent._load_session("last")
            text = output()
        assert any(m["content"] == "hello" for m in agent.conversation)
        assert skills.active_name == name                # skill restored
        assert "hi there" in text                        # transcript replayed
    finally:
        server.stop()


# --------------------------------------------------------------------------- #
# system prompt override
# --------------------------------------------------------------------------- #
def test_system_prompt_override_keeps_tool_machinery():
    base = A.build_text_system_prompt(A.ALL_TOOLS)
    custom = A.build_text_system_prompt(A.ALL_TOOLS, "You are Jeeves, a butler.")
    assert custom.startswith("You are Jeeves, a butler.")
    assert "coding agent working" not in custom          # persona replaced
    assert "## read_file" in custom and "Available tools:" in custom
    assert "<tool_call>" in custom and "To call a tool" in custom


def test_system_prompt_command_lifecycle():
    server = FakeServer()
    try:
        agent = make_agent(server)
        assert agent.system_prompt is None
        with recorded_console():
            agent._handle_command("/system You are a terse Rust expert.")
            assert agent.system_prompt == "You are a terse Rust expert."
            Path("persona.txt").write_text("You are a pirate captain.\n")
            agent._handle_command("/system file persona.txt")
            assert agent.system_prompt == "You are a pirate captain."
            agent._handle_command("/system default")
            assert agent.system_prompt is None

        agent.system_prompt = "You are HAL 9000."
        server.push(*answer("ok"))
        agent.conversation.append({"role": "user", "content": "hi"})
        agent._begin_user_turn()
        agent._native_turn(agent.conversation)
        first = server.last_request["messages"][0]
        assert first["role"] == "system" and "HAL 9000" in first["content"]
        assert agent.conversation[0]["role"] == "user"   # not stored
    finally:
        server.stop()


# --------------------------------------------------------------------------- #
# config / help surfaces
# --------------------------------------------------------------------------- #
def test_config_entries_reflect_live_state():
    agent = make_agent(approval="low", allow_internet=False, autocompress=85)
    agent.allow_bash_escape = False
    values = dict((label, value) for label, value, _desc in agent.config_entries())
    assert values["internet"].startswith("OFF")
    assert values["bash escape (!)"] == "off"
    assert values["approval"] == "low"
    assert values["autocompress"] == "85%"
    assert values["path confinement"] == "on"
    assert values["command denylist"] == "on"
    assert "chars" in values["read limit"]

    with recorded_console():
        agent._approval_command("yolo")
        agent._system_command("You are a pirate")
    values = dict((label, value) for label, value, _desc in agent.config_entries())
    assert values["approval"] == "off (yolo)"
    assert values["system prompt"] == "custom"


def test_help_and_config_render():
    agent = make_agent()
    with recorded_console(width=140) as output:
        agent._handle_command("/help")
        text = output()
    for expected in ("/compact", "/memory save", "/yolo", "/read-limit",
                     "@path", "!<command>", "/max-tokens"):
        assert expected in text, f"missing from /help: {expected}"

    with recorded_console(width=140) as output:
        agent._handle_command("/config")
        text = output()
    assert "endpoint" in text and "autocompress" in text


def test_sampler_descriptions():
    assert A.sampler_description("temperature").startswith("randomness")
    assert "DRY" in A.sampler_description("dry_multiplier")
    assert A.sampler_description("totally_unknown_param") == ""   # graceful


# --------------------------------------------------------------------------- #
# token budgets and stats
# --------------------------------------------------------------------------- #
def test_max_tokens_command_and_truncation_hint():
    server = FakeServer()
    try:
        agent = make_agent(server, max_tokens=4096)
        with recorded_console() as output:
            agent._handle_command("/max-tokens")
            assert "4096" in output()
        with recorded_console():
            agent._handle_command("/max-tokens 8192")
            assert agent.max_tokens == 8192
            agent._handle_command("/maxtokens 16000")     # alias
            assert agent.max_tokens == 16000
            agent._handle_command("/max-tokens 0")        # rejected
            assert agent.max_tokens == 16000

        server.push(*answer("ok"))
        agent.conversation.append({"role": "user", "content": "hi"})
        agent._begin_user_turn()
        agent._native_turn(agent.conversation)
        assert server.last_request["max_tokens"] == 16000  # live value on the wire

        # finish_reason == length on a REAL turn warns the user
        server.push(*answer("partial", finish="length"))
        with recorded_console(width=140) as output:
            agent._native_turn(agent.conversation)
            text = output()
        assert "max_tokens ceiling" in text and "/max-tokens" in text

        # internal capped calls legitimately hit `length`: no false alarm
        server.push(*answer("x", finish="length"))
        with recorded_console(width=140) as output:
            agent._stream_with_recovery(
                [{"role": "user", "content": "x"}], native_tools=False,
                max_tokens_cap=96, record=False)
            assert "max_tokens ceiling" not in output()
    finally:
        server.stop()


def test_prefill_stats_from_llamacpp_timings():
    assert A.prefill_stats(None) is None
    assert A.prefill_stats({}) is None
    assert A.prefill_stats({"prompt_n": 0}) is None
    tokens, seconds, rate = A.prefill_stats(
        {"prompt_n": 1234, "prompt_ms": 2100, "prompt_per_second": 587.6})
    assert (tokens, round(seconds, 2), round(rate)) == (1234, 2.1, 588)
    tokens, seconds, rate = A.prefill_stats({"prompt_n": 500, "prompt_ms": 1000})
    assert (tokens, seconds, round(rate)) == (500, 1.0, 500)      # derived
    tokens, seconds, rate = A.prefill_stats({"prompt_n": 42})
    assert (tokens, seconds, rate) == (42, None, None)            # count only
    assert "prefill 1,234 @ 588 tok/s" in A.format_prefill((1234, 2.1, 587.6)).plain


def test_stats_line_shows_prefill_and_sink_carries_it():
    server = FakeServer()
    try:
        captured = {}
        A.STATS_SINK = lambda payload: captured.update(payload)
        agent = make_agent(server)
        server.push(delta({"content": "hello"}),
                    delta({}, finish="stop",
                          timings={"prompt_n": 1200, "prompt_ms": 2000.0,
                                   "prompt_per_second": 600.0, "cache_n": 800}))
        agent.conversation.append({"role": "user", "content": "hi"})
        agent._begin_user_turn()
        with recorded_console(width=200) as output:
            agent._native_turn(agent.conversation)
            text = " ".join(output().split())
        assert "prefill 1,200 @ 600 tok/s" in text
        assert captured["prefill"] == 1200 and round(captured["prefill_rate"]) == 600
    finally:
        A.STATS_SINK = None
        server.stop()


# --------------------------------------------------------------------------- #
# non-convergence backstop
# --------------------------------------------------------------------------- #
def test_turn_request_ceiling_stops_endless_tool_calls():
    """Distinct tool calls forever defeat dedup, the loop guard AND the
    nudge cap; only the per-user-turn request ceiling ends it."""
    server = FakeServer()
    try:
        for index in range(A.MAX_TURN_REQUESTS + 5):
            server.push(tool_call_chunk("run_bash", {"command": f"echo {index}"},
                                        call_id=f"t{index}"))
        agent = make_agent(server, yolo=True, messages=["go"])
        with recorded_console(width=140) as output:
            agent.run()
            text = output()
        assert "request ceiling" in text
        assert server.count <= A.MAX_TURN_REQUESTS + 1, server.count
    finally:
        server.stop()


# --------------------------------------------------------------------------- #
# Agent Skills spec conformance (agentskills.io/specification)
# --------------------------------------------------------------------------- #
def test_skill_name_validation_matches_the_spec():
    for valid in ("pdf-processing", "data-analysis", "code-review", "a", "x1",
                  "a" * 64):
        assert A.validate_skill_name(valid) is None, valid
    for invalid, reason in (
        ("PDF-Processing", "uppercase"),
        ("-pdf", "leading hyphen"),
        ("pdf-", "trailing hyphen"),
        ("pdf--processing", "consecutive hyphens"),
        ("pdf_processing", "underscore"),
        ("pdf processing", "space"),
        ("a" * 65, "too long"),
        ("", "empty"),
    ):
        assert A.validate_skill_name(invalid) is not None, reason


def test_generated_names_are_spec_valid():
    for text in ("Terse mode\nBe brief.", "PDF Processing!! (v2)",
                 "###", "  ", "a" * 300):
        name = A.mangle_skill_name(text)
        assert A.validate_skill_name(name) is None, (text[:20], name)


def test_frontmatter_parsing():
    fields, body = A.parse_frontmatter(
        "---\nname: pdf-processing\n"
        "description: Extract text. Use when: handling PDFs.\n"
        "license: Apache-2.0\n"
        "metadata:\n  author: example-org\n  version: \"1.0\"\n"
        "allowed-tools: Bash(git:*) Read\n---\n\n# Body\n\ntext here\n")
    assert fields["name"] == "pdf-processing"
    # the colon inside the value survives (the guide's recommended fallback)
    assert fields["description"] == "Extract text. Use when: handling PDFs."
    assert fields["license"] == "Apache-2.0"
    assert fields["metadata"] == {"author": "example-org", "version": "1.0"}
    assert fields["allowed-tools"] == "Bash(git:*) Read"
    assert body.startswith("# Body") and "---" not in body

    # no frontmatter, and unterminated frontmatter, both degrade to body
    assert A.parse_frontmatter("just text") == ({}, "just text")
    fields, body = A.parse_frontmatter("---\nname: x\nno closing delimiter")
    assert fields == {} and "no closing delimiter" in body

    round_trip = A.build_frontmatter(
        {"name": "n", "description": "Use when: x", "metadata": {"a": "b"}})
    reparsed, _ = A.parse_frontmatter(round_trip + "\n\nbody")
    assert reparsed["name"] == "n"
    assert reparsed["description"] == "Use when: x"   # quoted on write
    assert reparsed["metadata"] == {"a": "b"}


def test_import_writes_a_spec_compliant_skill_directory():
    skills = A.SkillManager(".skills", search_dirs=[".skills"])
    message = skills.save("Terse mode\nAnswer in at most two sentences.")
    assert "imported" in message
    name = skills.list()[0]
    skill_md = Path(".skills") / name / "SKILL.md"
    assert skill_md.is_file(), "a skill must be a directory containing SKILL.md"
    fields, body = A.parse_frontmatter(skill_md.read_text())
    assert fields["name"] == name == skill_md.parent.name   # name == directory
    assert fields["description"]                            # required, non-empty
    assert A.validate_skill_name(fields["name"]) is None
    assert "two sentences" in body and not body.startswith("---")

    # frontmatter supplied by the author is honoured
    skills.save("---\nname: code-review\n"
                "description: Review diffs for bugs. Use when reviewing code.\n"
                "---\n\nLook for off-by-one errors.\n")
    record = skills.record("code-review")
    assert record["description"].startswith("Review diffs")
    assert record["body"] == "Look for off-by-one errors."


def test_discovery_scans_the_cross_client_locations():
    def write_skill(directory, name, description, body="do the thing"):
        target = Path(directory) / name
        target.mkdir(parents=True, exist_ok=True)
        (target / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n")

    write_skill(".skills", "native-skill", "Our own location.")
    write_skill(".agents/skills", "shared-skill", "Cross-client convention.")
    write_skill("vendor/skills", "extra-skill", "An extra searched path.")
    skills = A.SkillManager(
        ".skills", search_dirs=[".skills", ".agents/skills", "vendor/skills"])
    assert skills.list() == ["extra-skill", "native-skill", "shared-skill"]
    assert skills.describe("shared-skill") == "Cross-client convention."


def test_project_scope_shadows_user_scope():
    def write_skill(directory, name, description):
        target = Path(directory) / name
        target.mkdir(parents=True, exist_ok=True)
        (target / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {description}\n---\n\nbody\n")

    write_skill("project/.agents/skills", "code-review", "PROJECT version.")
    write_skill("home/.agents/skills", "code-review", "USER version.")
    skills = A.SkillManager(
        "project/.agents/skills",
        search_dirs=["project/.agents/skills", "home/.agents/skills"])
    assert skills.describe("code-review") == "PROJECT version."
    assert any("shadowed by" in d for d in skills.diagnostics)


def test_validation_is_lenient_but_a_missing_description_skips():
    base = Path(".skills")
    # name disagrees with the directory: a warning, still loaded
    (base / "mismatch-dir").mkdir(parents=True)
    (base / "mismatch-dir" / "SKILL.md").write_text(
        "---\nname: other-name\ndescription: Still usable.\n---\n\nbody\n")
    # an invalid name: a warning, still loaded
    (base / "Bad_Name").mkdir(parents=True)
    (base / "Bad_Name" / "SKILL.md").write_text(
        "---\nname: Bad_Name\ndescription: Also usable.\n---\n\nbody\n")
    # no description: skipped, because disclosure needs one
    (base / "no-description").mkdir(parents=True)
    (base / "no-description" / "SKILL.md").write_text(
        "---\nname: no-description\n---\n\nbody\n")

    skills = A.SkillManager(".skills", search_dirs=[".skills"])
    names = skills.list()
    assert "other-name" in names and "Bad_Name" in names
    assert "no-description" not in names
    joined = " ".join(skills.diagnostics)
    assert "does not match its directory" in joined
    assert "lowercase" in joined
    assert "no description" in joined


def test_legacy_flat_files_still_load():
    Path(".skills").mkdir()
    Path(".skills/old-style.md").write_text("Pirate mode\nTalk like a pirate.\n")
    skills = A.SkillManager(".skills", search_dirs=[".skills"])
    assert skills.list() == ["old-style"]
    record = skills.record("old-style")
    assert record["legacy"] is True
    assert record["description"] == "Pirate mode"      # inferred
    assert "pirate" in skills.read("old-style")
    assert "active" in skills.activate("old-style")


def test_activation_strips_frontmatter_and_lists_resources():
    target = Path(".skills/pdf-processing")
    (target / "scripts").mkdir(parents=True)
    (target / "references").mkdir()
    (target / "SKILL.md").write_text(
        "---\nname: pdf-processing\n"
        "description: Extract PDF text. Use when handling PDFs.\n---\n\n"
        "# PDF Processing\n\nUse pdftotext first.\n")
    (target / "scripts" / "extract.py").write_text("print(1)\n")
    (target / "references" / "REFERENCE.md").write_text("details\n")

    skills = A.SkillManager(".skills", search_dirs=[".skills"])
    # listed by directory type, in the order the spec documents them
    assert skills.resources("pdf-processing") == [
        "scripts/extract.py", "references/REFERENCE.md"]
    message = skills.activate("pdf-processing")
    assert "2 bundled file(s)" in message
    assert skills.active_text.startswith("# PDF Processing")   # body only
    assert "description:" not in skills.active_text

    agent = make_agent(skills=skills)
    prompt = agent._skill_prompt()
    assert "## Active skill: pdf-processing" in prompt
    assert "Extract PDF text" in prompt                # the description
    assert "Use pdftotext first" in prompt              # the instructions
    # resources are LISTED, never inlined (tier 3 stays lazy)
    assert "scripts/extract.py" in prompt
    assert "print(1)" not in prompt and "details" not in prompt


def test_config_entries_works_on_a_minimal_agent():
    """config_entries() feeds the Settings tab through a stub agent in the
    pilots. Every field it reads must be optional-safe, or the table
    silently renders empty -- a failure that is confusing to diagnose from
    the UI. This test fails fast instead."""
    from harness import make_stub_agent

    rows = A.Agent.config_entries(make_stub_agent())
    assert len(rows) >= 20, len(rows)
    labels = [label for label, _value, _description in rows]
    for expected in ("endpoint", "model", "approval", "read limit",
                     "planning", "verify command", "extra request body"):
        assert expected in labels, f"missing config row: {expected}"
    for label, value, description in rows:
        assert isinstance(value, str) and value, f"empty value for {label}"
        assert description, f"missing description for {label}"

    # ...and on an agent missing every optional attribute
    class Bare:
        client = type("C", (), {"model": "m", "url": "u", "name": "n"})()
        mode = "native"
        max_tokens = 1
        temperature = None
        retries = 1
        max_nudges = 1
        reasoning_mode = "collapsed"
        allow_tools = True
        allow_internet = True
        yolo = False
        approve_level = "low"
        risk_classifier = "heuristic"
        autocompress_percent = 0
        system_prompt = None
        planning = "off"
        plan = A.Plan()
        skills = None
        memories = None
        policy = None
    assert len(A.Agent.config_entries(Bare())) >= 20


def test_no_skills_are_shipped_with_the_agent():
    """Skills come from the end user only: a fresh workspace has none, and
    discovery never reaches into the agent's own directory."""
    skills = A.SkillManager(".skills")          # default search dirs
    assert skills.list() == []
    assert skills.diagnostics == []
    assert not hasattr(A, "SKILL_BUNDLED_DIR")
    agent_dir = Path(A.__file__).resolve().parent
    assert not any(str(agent_dir) in str(d) for d in skills.search_dirs)
    # ...and the search paths are exactly the documented user locations
    joined = " ".join(str(d) for d in skills.search_dirs)
    for expected in (".skills", ".agents/skills"):
        assert expected in joined, expected


def test_background_ops_guidance_appears_only_while_jobs_run():
    """The discipline lives in py-ai.py and is injected just-in-time, so a
    normal turn pays nothing for it."""
    server = FakeServer()
    try:
        A.BACKGROUND_JOBS.clear()
        agent = make_agent(server, yolo=True)
        assert A.background_ops_block() == ""
        assert "Background jobs" not in agent._system_extras()

        with recorded_console():
            agent._execute_tool("run_bash", {"command": "sleep 5",
                                             "background": True})
        block = agent._system_extras()
        assert "Background jobs are outstanding" in block
        # the live job list, so the model knows what is outstanding
        job = A.BACKGROUND_JOBS[-1]
        assert f"pid {job['pid']}" in block and job["log"] in block
        assert "[running]" in block
        # the substance, not just a heading
        for expected in ("NO OUTPUT IS NOT DEATH", "/proc/", "du -sh",
                         "pkill", "130 = SIGINT", "137", "wait_background",
                         "sample TWICE"):
            assert expected in block, expected

        # and it reaches the model as a system message
        server.push(*answer("checking"))
        agent.conversation.append({"role": "user", "content": "how is it?"})
        agent._begin_user_turn()
        with recorded_console():
            agent._native_turn(agent.conversation)
        first = server.last_request["messages"][0]
        assert first["role"] == "system"
        assert "Background jobs are outstanding" in first["content"]
        # never stored in the conversation
        assert "NO OUTPUT" not in json.dumps(agent.conversation)

        A.BACKGROUND_JOBS.clear()
        assert A.background_ops_block() == ""
    finally:
        A.BACKGROUND_JOBS.clear()
        server.stop()


def test_sharpened_nudges_are_actionable_and_transient():
    """A nudge is read by a model that just failed to act, so it must be a
    decision procedure -- and it must not be stored, or a stall would cost
    context on every later turn."""
    for text in (A.NUDGE_REASONING_ONLY, A.NUDGE_EMPTY):
        assert len(text) > 120                       # more than "try again"
    # the write -> run -> fix loop leads, because that is the loop a
    # stalled model needs to enter; the later branches cover the rest
    order = [A.NUDGE_REASONING_ONLY.index(tool) for tool in
             ("write_file", "run_bash", "edit_file", "search_files")]
    assert order == sorted(order), order
    assert "5." in A.NUDGE_REASONING_ONLY
    for tool in ("search_files", "read_file", "write_file", "edit_file",
                 "run_bash", "wait_background"):
        assert tool in A.NUDGE_REASONING_ONLY, tool
    assert "plain prose" in A.NUDGE_REASONING_ONLY   # not only build tasks

    server = FakeServer()
    try:
        for protocol in ("native", "text"):
            server.reset()
            server.push(delta({"reasoning_content": "thinking"},
                              finish="stop"))
            server.push(*answer("The answer is 42."))
            agent = make_agent(server, protocol=protocol, max_nudges=2)
            agent.conversation.append({"role": "user", "content": "hi"})
            agent._begin_user_turn()
            with recorded_console():
                turn = (agent._native_turn if protocol == "native"
                        else agent._text_turn)
                turn(agent.conversation)
            sent = json.dumps(server.requests[-1]["messages"])
            stored = json.dumps(agent.conversation)
            assert "Emit that tool call or answer" in sent, protocol      # reached the model
            assert "Emit that tool call or answer" not in stored, protocol  # not persisted
            # ...and no blank assistant message was stored either
            assert not [m for m in agent.conversation
                        if m.get("role") == "assistant"
                        and not str(m.get("content") or "").strip()]
            assert agent.last_answer == "The answer is 42."
    finally:
        server.stop()


def test_config_covers_every_meaningful_flag():
    """/config is the answer to "what is this run actually doing", so a new
    flag without a row is a bug. This test fails when they drift apart."""
    import re

    from harness import make_stub_agent

    source = (Path(A.__file__)).read_text()
    flags = set(re.findall(r'parser\.add_argument\(\s*\n?\s*"(--[a-z0-9-]+)"',
                           source))
    rows = {label for label, _v, _d in
            A.Agent.config_entries(make_stub_agent())}
    descriptions = " ".join(
        d for _l, _v, d in A.Agent.config_entries(make_stub_agent()))
    haystack = " ".join(rows) + " " + descriptions

    # flags that are deliberately absent, with the reason
    exempt = {
        "--api-key",            # secret; the endpoint row is what matters
        "--base-url",           # shown as "endpoint"
        "--model",              # its own row
        "--resume",             # a one-time action, not a setting
        "--serve", "--serve-host", "--serve-port",  # shown via "interface"
        "--yolo",               # shown as approval "off (yolo)"
        "--hide-reasoning", "--raw",       # shown via "reasoning"
        "--no-path-confinement",           # shown as "path confinement"
        "--no-command-denylist",           # shown as "command denylist"
        "--no-autosave",                   # shown as "autosave"
        "--no-thinking",                   # shown as "thinking"
        "--no-tools", "--no-internet",     # shown as "tools" / "internet"
        "--dry-multiplier", "--dry-base",  # sampler params, in /settings
        "--dry-allowed-length", "--dry-penalty-last-n",
        "--skill",              # shown as "active skill"
        "--system-file",        # shown as "system prompt"
        "--tools-file",         # shown in the Tools tab
        "--read-limit-lines",   # folded into "read limit"
        "--verify-threshold", "--verify-samples",  # in "verify answers"
        "--sandbox-cpu", "--sandbox-memory",       # in "sandbox"
        "--sandbox-file", "--sandbox-procs",
        "--max-session-seconds", "--max-session-tokens",  # in the budget row
        "--task-file",          # counted by "batch tasks"
        "--thinking-key",       # named in the thinking description
        "--plan",               # shown as "planning"
        "--transport",          # shown as "transport"
        "--protocol", "--ui", "--engine", "--approval", "--temperature",
        "--timeout", "--retries", "--max-tokens", "--max-nudges",
        "--autocompress", "--system", "--sessions-dir", "--skills-dir",
        "--memories-dir", "--ctx-size", "--log-level", "--log-full",
        "--read-limit-chars", "--extra-body", "--thinking", "--sandbox",
        "--unattended", "--report", "--task", "--verify-command",
        "--verify-mode", "--verify-rounds", "--verify-budget",
        "--allow-verify-edits", "--risk-classifier", "--bash-escape",
        "--max-session-requests", "--verify", "--no-verify", "--tools",
        "--internet", "--no-bash-escape",
    }
    def mentioned(flag):
        # a row label, a de-hyphenated label, or the flag quoted verbatim
        # in a description all count as documenting it
        return (flag in haystack
                or flag.lstrip("-").replace("-", " ") in haystack)

    missing = sorted(flag for flag in flags
                     if flag not in exempt and not mentioned(flag))
    assert not missing, f"flags with no /config row or mention: {missing}"

    # and the rows that were added for this must actually be present
    for expected in ("interface", "context window", "request timeout",
                     "thinking", "session budgets", "report", "batch tasks",
                     "logging", "autosave", "skills dir", "memories dir",
                     "verify target edits"):
        assert expected in rows, expected
    assert len(rows) >= 40, len(rows)


def test_help_covers_every_dispatched_command():
    """A command you can type but cannot find in /help is invisible."""
    import re

    # scan the dispatcher's own source: every "/name" literal in it is a
    # command a user can type. Simpler and less brittle than matching the
    # shapes of if/elif conditions.
    import ast
    import inspect

    tree = ast.parse(Path(A.__file__).read_text())
    body = ""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_handle_command":
            body = ast.get_source_segment(Path(A.__file__).read_text(), node)
            break
    assert body, "could not find _handle_command"
    dispatched = set(re.findall(r'"(/[a-z?-]+)"', body))
    # "/thnk" is the example typo in the unknown-command message, not a
    # command anyone can run
    dispatched.discard("/thnk")
    helped = set()
    for label, _description in A.Agent.HELP_ENTRIES:
        helped |= set(re.findall(r'(/[a-z?-]+)', label))

    assert dispatched, "the dispatch scan found nothing -- fix the test"
    missing = sorted(dispatched - helped)
    assert not missing, f"dispatched but absent from /help: {missing}"
    stale = sorted(helped - dispatched)
    assert not stale, f"in /help but not dispatched: {stale}"

    # and everything typeable is tab-completable
    uncompleted = sorted(dispatched - set(A.SLASH_COMMANDS))
    assert not uncompleted, f"missing from completion: {uncompleted}"
    assert not sorted(set(A.SLASH_COMMANDS) - dispatched)

    # every help entry says something (a few are not slash commands:
    # the ! shell escape and @path attachments)
    for label, description in A.Agent.HELP_ENTRIES:
        assert label[0] in "/!@" and description, label

    # no dispatch literal may contain a space: "/res /answer" as a single
    # string means the command stopped working entirely (this happened --
    # a bulk edit to the help labels hit a dispatch tuple)
    merged = re.findall(r'"(/[a-z-]+ /[^"]+)"', body)
    assert not merged, f"merged command literals in the dispatcher: {merged}"


def test_settings_shows_every_request_override():
    server = FakeServer()
    try:
        agent = make_agent(server, temperature=0.7, max_tokens=2048,
                           dry_params={"dry_base": 1.9},
                           extra_body={"top_k": 20, "chat_template_kwargs":
                                       {"enable_thinking": False}})
        overrides = agent.request_overrides()
        # temperature was not the only thing we send
        assert overrides["temperature"] == 0.7
        assert overrides["max_tokens"] == 2048
        assert overrides["dry_base"] == 1.9
        assert overrides["top_k"] == 20
        # nested extra-body fields are flattened to a readable path
        assert overrides["chat_template_kwargs.enable_thinking"] is False

        agent.server_settings = {"temperature": 0.6, "top_k": 40,
                                 "top_p": 0.95}
        with recorded_console(width=130) as output:
            agent._handle_command("/settings")
            text = output()
        assert "server sampling defaults" in text
        for expected in ("temperature", "top_k", "top_p", "max_tokens",
                         "dry_base", "enable_thinking"):
            assert expected in text, expected
        # a param the server never reported still shows as an override
        assert "2048" in text and "1.9" in text
        # descriptions resolve even for dotted paths
        assert "chat-template switch" in text
    finally:
        server.stop()


def test_settings_is_useful_without_server_defaults():
    """vLLM and hosted APIs publish no defaults; what we send is then the
    only answer available, and more useful than 'nothing to show'."""
    server = FakeServer()
    try:
        agent = make_agent(server, temperature=0.5, max_tokens=1024)
        agent.server_settings = {}
        with recorded_console(width=130) as output:
            agent._handle_command("/settings")
            text = output()
        assert "publishes no defaults" in text
        assert "temperature" in text and "0.5" in text
        assert "max_tokens" in text and "1024" in text
        assert "left to the server's own defaults" in text
    finally:
        server.stop()


def test_sampler_descriptions_cover_the_common_parameters():
    for key in ("temperature", "top_k", "top_p", "min_p", "typical_p",
                "repeat_penalty", "repeat_last_n", "presence_penalty",
                "frequency_penalty", "mirostat", "mirostat_tau",
                "mirostat_eta", "dry_multiplier", "dry_base",
                "dry_allowed_length", "dry_penalty_last_n",
                "xtc_probability", "xtc_threshold", "top_n_sigma", "seed",
                "samplers", "grammar", "logit_bias", "n_probs", "tfs_z",
                "max_tokens", "enable_thinking", "reasoning_effort"):
        assert A.sampler_description(key), key
    # an unknown key degrades to empty rather than raising
    assert A.sampler_description("no_such_param") == ""


if __name__ == "__main__":
    main(globals(), "skills, memory, compression, sessions, config, stats")
