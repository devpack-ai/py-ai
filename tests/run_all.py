#!/usr/bin/env python3
"""Run every test module and print a summary.

    python3 tests/run_all.py            # everything
    python3 tests/run_all.py -k policy  # only modules whose name matches

Each module runs in its own subprocess so a hard crash (or a hung Textual
pilot) cannot take the whole run down. Exit code is non-zero if anything
failed, so this is CI-usable as-is.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import dim, green, red, yellow  # noqa: E402

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

HERE = Path(__file__).resolve().parent
MODULES = [
    ("test_tools_policy.py", "tools, policy, risk, loop guards", 180),
    ("test_features.py", "skills, memory, compression, sessions, stats", 300),
    ("test_attachments.py", "@path attachments and read budget", 180),
    ("test_recovery.py", "streaming parsers, recovery layer, text protocol", 300),
    ("test_commands.py", "command surface: display, sessions, export", 300),
    ("test_registry_cli.py", "custom tools, probing, CLI flag wiring", 500),
    ("test_cli_extra.py", "remaining flags, model picking, utilities", 500),
    ("test_textual.py", "Textual UI pilots (needs textual)", 420),
]

SUMMARY_RE = re.compile(r"---\s.*:\s(\d+)/(\d+)\spassed\s---")


def plain(text: str) -> str:
    """ANSI-stripped copy, for matching lines that may be colourised."""
    return ANSI_RE.sub("", text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", metavar="PATTERN", default="",
                        help="only run modules whose filename contains PATTERN")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="stream each module's full output")
    args = parser.parse_args()

    total_passed = total_tests = 0
    failed_modules = []
    started = time.monotonic()

    child_env = dict(os.environ)
    if sys.stdout.isatty() and not os.environ.get("NO_COLOR"):
        child_env["PRN_TEST_COLOUR"] = "1"   # subprocess stdout is a pipe

    for filename, description, timeout in MODULES:
        if args.k and args.k not in filename:
            continue
        print(f"\n>>> {filename} {dim('-- ' + description)}")
        try:
            completed = subprocess.run(
                [sys.executable, filename],
                cwd=HERE, capture_output=True, text=True, timeout=timeout,
                env=child_env,
            )
        except subprocess.TimeoutExpired:
            print(red(f"    TIMEOUT after {timeout}s"))
            failed_modules.append(filename)
            continue

        output = completed.stdout + completed.stderr
        if args.verbose:
            print(output)
        else:  # just the per-test verdicts and the summary
            for line in output.splitlines():
                if plain(line).startswith(("  PASS", "  FAIL", "---",
                                           "        ")):
                    print(line)

        match = SUMMARY_RE.search(plain(output))
        if match:
            passed, count = int(match.group(1)), int(match.group(2))
            total_passed += passed
            total_tests += count
        if completed.returncode != 0:
            failed_modules.append(filename)

    elapsed = time.monotonic() - started
    ok = not failed_modules
    rule = "=" * 62
    total = f"TOTAL: {total_passed}/{total_tests} tests passed in {elapsed:.1f}s"
    print("\n" + (green(rule) if ok else red(rule)))
    print(green(total) if ok else yellow(total))
    if failed_modules:
        print(red("FAILED modules: " + ", ".join(failed_modules)))
    else:
        print(green("All modules green."))
    print(green(rule) if ok else red(rule))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
