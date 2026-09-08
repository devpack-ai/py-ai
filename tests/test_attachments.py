"""@path attachments: text inlining, multimodal image parts, policy
confinement, false-positive avoidance, and the tunable read budget.
"""

import base64
import json
import os
from pathlib import Path

from harness import (  # noqa: E402
    A, FakeServer, answer, main, make_agent, recorded_console,
)

# smallest valid 1x1 PNG
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGA"
    "hKmMIQAAAABJRU5ErkJggg=="
)


def fixtures():
    Path("notes.py").write_text("def hello():\n    return 42\n")
    Path("sub").mkdir(exist_ok=True)
    Path("sub/data.txt").write_text("x" * 30_000)
    Path("with space.md").write_text("# spaced\n")
    Path("shot.png").write_bytes(PNG_BYTES)


def test_plain_message_is_untouched():
    agent = make_agent()
    message = agent._build_user_message("just a normal message")
    assert message == {"role": "user", "content": "just a normal message"}


def test_text_file_is_inlined_as_a_fenced_block():
    fixtures()
    agent = make_agent()
    with recorded_console() as output:
        message = agent._build_user_message("explain @notes.py please")
        text = output()
    body = message["content"]
    assert isinstance(body, str)
    assert body.startswith("explain @notes.py please")   # typed text preserved
    assert "--- notes.py ---" in body and "return 42" in body
    assert "```py" in body
    assert "attached notes.py" in text                   # user-visible receipt


def test_quoted_paths_with_spaces():
    fixtures()
    agent = make_agent()
    with recorded_console():
        message = agent._build_user_message('look at @"with space.md"')
    assert "# spaced" in message["content"]


def test_image_becomes_multimodal_content_parts():
    fixtures()
    agent = make_agent()
    with recorded_console():
        message = agent._build_user_message("what is in @shot.png ?")
    parts = message["content"]
    assert isinstance(parts, list)
    assert [part["type"] for part in parts] == ["text", "image_url"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "what is in @shot.png ?" in parts[0]["text"]


def test_text_and_image_together():
    fixtures()
    agent = make_agent()
    with recorded_console():
        message = agent._build_user_message("@notes.py and @shot.png")
    parts = message["content"]
    assert isinstance(parts, list)
    assert "return 42" in parts[0]["text"]               # file inlined in text
    assert parts[1]["type"] == "image_url"


def test_oversized_image_is_skipped():
    Path("huge.png").write_bytes(b"\x89PNG" + b"0" * (A.IMAGE_LIMIT_BYTES + 10))
    agent = make_agent()
    with recorded_console() as output:
        message = agent._build_user_message("@huge.png")
        text = output()
    assert message["content"] == "@huge.png"             # not attached
    assert "over the" in text and "limit" in text


def test_attachment_respects_and_follows_the_read_limit():
    fixtures()
    agent = make_agent()
    with recorded_console():
        message = agent._build_user_message("@sub/data.txt")
    assert "truncated at 20,000 chars" in message["content"]

    with recorded_console():
        agent._read_limit_command("2000")
        message = agent._build_user_message("@sub/data.txt")
    assert A.READ_LIMIT_CHARS == 2000
    assert "truncated at 2,000 chars" in message["content"]

    with recorded_console() as output:                   # bare shows current
        agent._read_limit_command("")
        assert "2,000 chars" in output()
    with recorded_console() as output:                   # invalid rejected
        agent._read_limit_command("abc")
        assert "usage" in output()
    assert A.READ_LIMIT_CHARS == 2000


def test_read_limit_flag_helper_sets_both_budgets():
    A.set_read_limits(chars=5000, lines=50)
    assert (A.READ_LIMIT_CHARS, A.READ_LIMIT_LINES) == (5000, 50)
    A.set_read_limits(chars=1)          # clamped to a sane floor
    assert A.READ_LIMIT_CHARS >= 500


def test_attachments_obey_path_confinement():
    agent = make_agent()
    with recorded_console() as output:
        message = agent._build_user_message("@/etc/passwd")
        text = output()
    assert message["content"] == "@/etc/passwd"          # refused, left as text
    assert "escapes the working root" in text


def test_directory_and_missing_paths_are_reported_not_attached():
    agent = make_agent()
    with recorded_console() as output:
        message = agent._build_user_message("@. and @src/nope.py")
        text = output()
    assert message["content"] == "@. and @src/nope.py"
    assert "is a directory" in text and "no such file" in text


def test_emails_and_user_at_host_are_never_treated_as_attachments():
    agent = make_agent()
    for text in ("email me at bob@example.com ok?", "ssh user@host.local",
                 "ping a@b"):
        with recorded_console() as output:
            message = agent._build_user_message(text)
            printed = output().strip()
        assert message["content"] == text, message
        assert printed == "", f"unexpected notice for {text!r}: {printed}"


def test_binary_non_image_is_refused_cleanly():
    Path("blob.bin").write_bytes(b"\x00\x01\x02\xff")
    agent = make_agent()
    with recorded_console() as output:
        message = agent._build_user_message("@blob.bin")
        text = output()
    assert message["content"] == "@blob.bin"
    assert "not readable as text" in text


def test_duplicate_tokens_attach_once():
    fixtures()
    agent = make_agent()
    with recorded_console():
        message = agent._build_user_message("@notes.py vs @notes.py")
    assert message["content"].count("--- notes.py ---") == 1


def test_content_parts_flatten_for_transcripts_and_summaries():
    assert A.message_text([{"type": "text", "text": "hi"},
                           {"type": "image_url", "image_url": {}}]) == "hi"
    assert A.message_text("plain") == "plain"
    assert A.message_text(None) == ""
    fixtures()
    agent = make_agent()
    with recorded_console():
        agent.conversation = [agent._build_user_message("@notes.py @shot.png")]
    transcript = agent._session_transcript()
    assert "return 42" in transcript          # no crash on content parts
    assert "image_url" not in transcript      # binary payload not dumped
    assert "read_file({})" not in A.render_for_summary(agent.conversation)


def test_attachment_reaches_the_wire_as_content_parts():
    fixtures()
    server = FakeServer()
    try:
        server.push(*answer("ok"))
        agent = make_agent(server, yolo=True,
                           messages=["describe @shot.png and @notes.py"])
        with recorded_console():
            agent.run()
        sent = server.last_request["messages"][-1]
        assert isinstance(sent["content"], list)
        assert [part["type"] for part in sent["content"]] == ["text", "image_url"]
        assert "return 42" in sent["content"][0]["text"]
        assert sent["content"][1]["image_url"]["url"].startswith(
            "data:image/png;base64,")
    finally:
        server.stop()


if __name__ == "__main__":
    main(globals(), "@path attachments and read budget")
