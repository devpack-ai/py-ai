"""Headless Textual pilots: the UI behaviours that only show up when the
app is actually mounted and laid out.

Each of these corresponds to a bug that shipped and was reported from a
screenshot -- first-click blank tabs, clipped Raw JSON, a truncated
Settings tab, a lost approval prompt. Requires `textual`; the module
skips cleanly when it is not installed.
"""

import asyncio
import sys

from harness import A, clear_sinks, main, make_stub_agent, workspace  # noqa: E402

try:
    import textual  # noqa: F401
    from textual.events import Paste
    from textual.widgets import (
        DataTable, Input, RichLog, Static, TabbedContent, TabPane,
    )
    from textual.containers import VerticalScroll
    HAVE_TEXTUAL = True
except ImportError:  # pragma: no cover
    HAVE_TEXTUAL = False


def pilot(coroutine):
    """Run an async pilot body as a sync test."""
    if not HAVE_TEXTUAL:
        print("        (skipped: textual not installed)")
        return
    asyncio.run(coroutine())


def all_text(log: "RichLog") -> str:
    """Every character currently rendered in a RichLog, whitespace-collapsed.

    Reads RichLog's private `lines[]._segments`, whose behaviour differs
    between Textual versions -- so use it only to look for a SHORT marker
    in a SMALL log (does this sink reach this pane at all), never to prove
    a large payload survived intact. Wrapping is asserted against the
    renderable instead, in test_raw_json_wraps_at_every_width."""
    return " ".join(
        " ".join(
            segment.text
            for line in log.lines
            for segment in line._segments
            if segment.text
        ).split()
    )


# --------------------------------------------------------------------------- #
def test_expected_tabs_are_present():
    async def body():
        app = A.build_textual_app(make_stub_agent())
        async with app.run_test(size=(120, 30)) as harness_pilot:
            await harness_pilot.pause(0.4)
            pane_ids = [pane.id for pane in app.query(TabPane)]
            for expected in ("tab-chat", "tab-reasoning", "tab-settings",
                             "tab-raw", "tab-files", "tab-tools", "tab-skills",
                             "tab-memory", "tab-sessions", "tab-logs", "tab-cot"):
                assert expected in pane_ids, f"missing pane: {expected}"
            assert "tab-dspy" not in pane_ids   # removed long ago
            bar = " ".join(str(tab.render()) for tab in app.query("ContentTab"))
            assert "CoT" in bar and "DSPy" not in bar
            app.query_one(TabbedContent).active = "tab-cot"
            await harness_pilot.pause(0.3)
            assert "Chain-of-Thought" in str(app.query_one("#cot", Static).render())

    pilot(body)


def test_settings_tab_has_both_tables_and_scrolls():
    async def body():
        app = A.build_textual_app(make_stub_agent())
        # deliberately short viewport: content must overflow and scroll
        async with app.run_test(size=(120, 24)) as harness_pilot:
            await harness_pilot.pause(0.4)
            app.query_one(TabbedContent).active = "tab-settings"
            await harness_pilot.pause(0.5)
            config = app.query_one("#agentcfg", DataTable)
            sampler = app.query_one("#settings", DataTable)
            assert config.row_count >= 15, config.row_count
            assert len(sampler.columns) == 4      # incl. the description column
            labels = [str(config.get_row_at(i)[0]) for i in range(config.row_count)]
            for expected in ("endpoint", "approval", "internet", "read limit"):
                assert expected in labels, f"missing config row: {expected}"
            scroll = app.query_one("#settings-scroll", VerticalScroll)
            assert scroll.max_scroll_y > 0, "settings pane is not scrollable"
            scroll.scroll_end(animate=False)
            await harness_pilot.pause(0.3)
            assert scroll.scroll_y > 0

    pilot(body)


def test_raw_json_wraps_at_every_width():
    """The reported bug: a long tool `description` ran off the right edge
    and was unreachable.

    Asserted against the renderable itself through a plain Rich console,
    not by scraping the widget: RichLog's `lines[]._segments` is private
    and its behaviour differs between Textual versions, which made this
    pass on one machine and fail on another while the feature was fine.
    """
    from rich.console import Console

    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "hey"}],
        "tools": [
            {"type": "function",
             "function": {"name": tool.name, "description": tool.description,
                          "parameters": tool.input_schema}}
            for tool in A.ALL_TOOLS
        ],
    }
    renderable = A.json_renderable(payload)
    # the wrapping itself: what the fix turned on
    assert type(renderable).__name__ == "Syntax"
    assert renderable.word_wrap is True

    for width in (200, 100, 60):
        console = Console(width=width, record=True, no_color=True,
                          legacy_windows=False)
        console.print(renderable)
        text = console.export_text()
        # nothing is lost, however narrow the pane
        for tail in ("Returns at most 400", "Prefer this over shell",
                     "refused by policy"):
            assert tail in " ".join(text.split()), \
                f"clipped at width {width}: {tail!r}"
        # ...and nothing needs horizontal scrolling to reach
        longest = max(len(line.rstrip()) for line in text.splitlines())
        assert longest <= width, f"width {width}: a line ran to {longest}"


def test_raw_tab_needs_no_horizontal_scrolling():
    """The user-visible half, via public API only: the Raw pane must never
    require a horizontal slider (unusable in the browser under --serve)."""
    async def body():
        class Stub(type(make_stub_agent())):
            def run(self):
                A.RAW_SINK({
                    "request": {"model": "m", "tools": [
                        {"type": "function",
                         "function": {"name": tool.name,
                                      "description": tool.description,
                                      "parameters": tool.input_schema}}
                        for tool in A.ALL_TOOLS]},
                    "chunks": [], "error": None,
                    "result": A.TurnResult(content="Hi", reasoning="x " * 15,
                                           finish_reason="stop", usage={}),
                })
                import time
                time.sleep(30)

        for width in (200, 100):
            app = A.build_textual_app(Stub())
            async with app.run_test(size=(width, 30)) as harness_pilot:
                await harness_pilot.pause(0.5)
                app.query_one(TabbedContent).active = "tab-raw"
                await harness_pilot.pause(0.6)
                raw = app.query_one("#rawlog", RichLog)
                assert raw.max_scroll_x == 0, (
                    f"width {width}: horizontal scrolling required")
                assert raw.lines, f"width {width}: nothing rendered"

    pilot(body)


def test_buffered_tabs_render_on_first_activation():
    """Content written while a tab is hidden must appear on the FIRST
    click (a freshly shown pane can still be 0-width for one refresh)."""
    async def body():
        class Stub(type(make_stub_agent())):
            def run(self):
                A.RAW_SINK({
                    "request": {"model": "m", "messages": []},
                    "chunks": [], "error": None,
                    "result": A.TurnResult(content="RAWMARKER",
                                           finish_reason="stop", usage={}),
                })
                A.log.info("LOGMARKER for the logs tab")
                import time
                time.sleep(30)

        app = A.build_textual_app(Stub())
        async with app.run_test(size=(120, 30)) as harness_pilot:
            await harness_pilot.pause(0.5)
            assert app._pending_raw and app._pending_logs   # buffered while hidden
            app.query_one(TabbedContent).active = "tab-raw"
            await harness_pilot.pause(0.8)
            assert "RAWMARKER" in all_text(app.query_one("#rawlog", RichLog))
            assert not app._pending_raw
            app.query_one(TabbedContent).active = "tab-logs"
            await harness_pilot.pause(0.8)
            assert "LOGMARKER" in all_text(app.query_one("#loglog", RichLog))
            assert not app._pending_logs

    pilot(body)


def test_log_sink_colours_by_level():
    captured = []
    previous = A.LOG_SINK
    A.LOG_SINK = lambda line, style: captured.append((line, style))
    try:
        A.log.debug("dbg")
        A.log.info("nfo")
        A.log.warning("warn")
        A.log.error("err")
    finally:
        A.LOG_SINK = previous
    styles = {style for _line, style in captured}
    assert {"blue", "green", "orange1", "bold red"} <= styles, styles
    formatted = captured[0][0]
    assert " - DEBUG\t - " in formatted and '"dbg"' in formatted
    assert "test_log_sink_colours_by_level[" in formatted   # funcName[lineno]


def test_approval_prompt_reaches_the_user_and_answer_routes_back():
    """The race that lost prompts: the awaiting flag must be armed on the
    UI thread atomically with posting the question."""
    async def body():
        holder = {}

        class Stub(type(make_stub_agent())):
            def run(self):
                holder["answer"] = A.APPROVAL_HOOK(
                    'allow run_bash({"command": "echo x > e.txt"})?  '
                    "[risk: MEDIUM] [Y/n] "
                )
                import time
                time.sleep(30)

        app = A.build_textual_app(Stub())
        async with app.run_test(size=(120, 30)) as harness_pilot:
            await harness_pilot.pause(0.5)
            chat = app.query_one("#chatlog", RichLog)
            assert "Y/n" in all_text(chat), "approval prompt never rendered"
            assert app._awaiting_approval is True
            await harness_pilot.press("n")
            await harness_pilot.press("enter")
            for _ in range(20):
                await harness_pilot.pause(0.1)
                if "answer" in holder:
                    break
            assert holder.get("answer") == "n", holder
            assert app._awaiting_approval is False

    pilot(body)


def test_prompt_attaches_files_via_token_paste_and_completion():
    async def body():
        from pathlib import Path

        Path("notes.py").write_text("x = 1\n")
        Path("with space.md").write_text("# spaced\n")
        Path("sub").mkdir(exist_ok=True)
        Path("sub/data.txt").write_text("y\n")

        app = A.build_textual_app(make_stub_agent())
        async with app.run_test(size=(120, 30)) as harness_pilot:
            await harness_pilot.pause(0.4)
            prompt = app.query_one(Input)

            prompt.value = ""
            prompt.attach_path("notes.py")
            assert prompt.value == "@notes.py "
            prompt.attach_path("with space.md")
            assert '@"with space.md"' in prompt.value      # quoted
            prompt.value = "explain"
            prompt.attach_path("notes.py")
            assert prompt.value == "explain @notes.py "    # appends, not clobbers

            prompt.value = ""                              # drag/paste a path
            prompt.post_message(Paste("'notes.py'"))
            await harness_pilot.pause(0.3)
            assert prompt.value.strip() == "@notes.py"

            prompt.value = ""                              # ordinary paste
            prompt.post_message(Paste("just some text"))
            await harness_pilot.pause(0.3)
            assert "@" not in prompt.value

            prompt.value = "@note"                         # completion
            await harness_pilot.press("tab")
            await harness_pilot.pause(0.2)
            assert prompt.value == "@notes.py"

            prompt.value = "look at @sub/"
            await harness_pilot.press("tab")
            await harness_pilot.pause(0.2)
            assert prompt.value.endswith("@sub/data.txt"), prompt.value

            prompt.value = "/read-l"                       # slash still works
            await harness_pilot.press("tab")
            await harness_pilot.pause(0.2)
            assert prompt.value.startswith("/read-limit")

    pilot(body)


def test_skills_and_memory_tabs_route_through_the_queue():
    """Tab buttons must hand work to the agent thread rather than mutating
    agent state from the UI thread."""
    async def body():
        skills = A.SkillManager(".skills", search_dirs=[".skills"])
        memories = A.MemoryManager(".memories")
        stub = make_stub_agent(skills=skills, memories=memories)
        app = A.build_textual_app(stub)
        async with app.run_test(size=(140, 40)) as harness_pilot:
            await harness_pilot.pause(0.4)
            app.query_one(TabbedContent).active = "tab-skills"
            await harness_pilot.pause(0.4)
            from textual.widgets import Button, TextArea

            app.query_one("#skilledit", TextArea).text = (
                "Pirate mode\nAlways answer like a pirate.")
            app.query_one("#btn-import-skill", Button).press()
            await harness_pilot.pause(0.4)
            assert skills.list(), "skill was not imported"
            name = skills.list()[0]
            assert name.startswith("pirate-mode-")
            table = app.query_one("#skilltable", DataTable)
            assert table.row_count == 1

            # activation is delegated (the agent thread applies /skill)
            app.query_one("#btn-on-skill", Button).press()
            await harness_pilot.pause(0.3)
            skills.activate(name)
            A.SKILL_SINK(name)
            await harness_pilot.pause(0.3)
            assert f"skill: {name}" in str(app.sub_title)   # header badge
            assert "ACTIVE" in str(app.query_one("#skillstatus", Static).render())

            # memory badge composes with the skill badge
            memories.save("# Note\ncontent")
            memories.load(memories.list()[0]["name"])
            A.MEMORY_SINK(list(memories.active))
            await harness_pilot.pause(0.3)
            assert "mem: 1 loaded" in str(app.sub_title)

    pilot(body)


if __name__ == "__main__":
    if not HAVE_TEXTUAL:
        print("textual is not installed: pip install textual")
        sys.exit(0)
    main(globals(), "Textual UI pilots")
