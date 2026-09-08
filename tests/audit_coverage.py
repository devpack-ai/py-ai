#!/usr/bin/env python3
"""Coverage audit: what does py-ai.py expose, and what do the tests touch?

Not a line-coverage tool -- an API-surface checker. It parses py-ai.py for
module functions, classes, methods, slash commands and CLI flags, then
greps the test modules for each identifier. The point is to find whole
features with NO test reference at all.

    python3 tests/audit_coverage.py           # summary + gaps
    python3 tests/audit_coverage.py --all     # also list covered items
"""

import ast
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENT = HERE.parent / "py-ai.py"
TEST_FILES = sorted(HERE.glob("test_*.py")) + [HERE / "harness.py"]

# Private helpers that are implementation detail: exercised through public
# paths, not worth naming in a test directly.
IGNORE = {
    "__init__", "__post_init__", "__enter__", "__exit__", "__repr__",
    "main", "compose", "on_mount", "on_unmount", "watch_value",
}


def source_api(tree: ast.AST) -> dict:
    """{'functions': [...], 'classes': {name: [methods]}}"""
    functions, classes = [], {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes[node.name] = [
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name not in IGNORE
            ]
        elif isinstance(node, ast.If):  # nested defs under `if` blocks
            for item in ast.walk(node):
                if isinstance(item, ast.ClassDef):
                    classes.setdefault(item.name, [
                        m.name for m in item.body
                        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and m.name not in IGNORE
                    ])
    return {"functions": [f for f in functions if f not in IGNORE],
            "classes": classes}


def nested_api(tree: ast.AST) -> dict:
    """Classes defined inside functions (build_textual_app's widgets)."""
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = [
                item.name for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name not in IGNORE
            ]
            found.setdefault(node.name, methods)
    return found


def slash_commands(source: str) -> list:
    match = re.search(r"SLASH_COMMANDS = \((.*?)\)", source, re.DOTALL)
    return sorted(set(re.findall(r'"(/[a-z-]+)"', match.group(1)))) if match else []


def cli_flags(source: str) -> list:
    return sorted(set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', source)))


def main() -> int:
    show_all = "--all" in sys.argv
    source = AGENT.read_text()
    tree = ast.parse(source)
    api = source_api(tree)
    api["classes"].update(nested_api(tree))
    tests_text = "\n".join(path.read_text() for path in TEST_FILES)

    def referenced(name: str) -> bool:
        return re.search(rf"\b{re.escape(name)}\b", tests_text) is not None

    sections = []

    functions = api["functions"]
    sections.append(("module functions", functions,
                     [f for f in functions if not referenced(f)]))

    method_items, method_gaps = [], []
    for class_name, methods in sorted(api["classes"].items()):
        for method in methods:
            label = f"{class_name}.{method}"
            method_items.append(label)
            if not referenced(method):
                method_gaps.append(label)
    sections.append(("class methods", method_items, method_gaps))

    commands = slash_commands(source)
    sections.append(("slash commands", commands,
                     [c for c in commands if c not in tests_text]))

    flags = cli_flags(source)
    sections.append(("CLI flags", flags,
                     [f for f in flags
                      if f not in tests_text
                      and f.lstrip("-").replace("-", "_") not in tests_text]))

    total_items = total_gaps = 0
    for title, items, gaps in sections:
        total_items += len(items)
        total_gaps += len(gaps)
        covered = len(items) - len(gaps)
        print(f"\n=== {title}: {covered}/{len(items)} referenced by tests ===")
        if gaps:
            for name in gaps:
                print(f"  GAP  {name}")
        if show_all:
            for name in items:
                if name not in gaps:
                    print(f"  ok   {name}")

    print(f"\n{'=' * 62}")
    print(f"API surface referenced by tests: "
          f"{total_items - total_gaps}/{total_items} "
          f"({100 * (total_items - total_gaps) / max(total_items, 1):.0f}%)")
    print("A GAP means no test names it at all -- triage, not a verdict.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
