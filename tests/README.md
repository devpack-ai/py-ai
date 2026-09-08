# py-ai test suite

Regression tests for `py-ai.py`. **299 tests, ~127s, no pytest required.**

```bash
python3 tests/run_all.py             # everything, with a summary
python3 tests/run_all.py -k recovery # one module
python3 tests/run_all.py -v          # stream full output
python3 tests/test_features.py       # a module directly
python3 tests/audit_coverage.py      # API-surface coverage report
pytest tests/                        # also works (test_* naming)
```

Output is colour-coded when writing to a terminal: **PASS** green, **FAIL**
red, tracebacks dim. `run_all.py` captures each module through a pipe, so it
forwards colour to its children via `PRN_TEST_COLOUR=1` when its own stdout
is a terminal — piping to a file or a CI log stays plain, and `NO_COLOR=1`
disables it everywhere.

Requirements: `httpx`, `rich`. `textual` is needed only for
`test_textual.py`, which skips cleanly without it. Nothing talks to a real
model or the network: a scriptable fake OpenAI SSE endpoint stands in (it
also serves `/v1/models` and llama.cpp's `/props` for the probe tests), and
`search_web` is exercised with a monkeypatched `httpx.post`.

## Layout

| File | Tests | Covers |
|---|---|---|
| `harness.py` | — | Fake SSE server (scriptable responses, HTTP errors, probe endpoints, recorded headers), temp-workspace fixture, console capture, approval/log sinks, agent + Textual-stub factories, mini runner |
| `test_tools_policy.py` | 60 | Built-in tools; `PolicyEngine` (path confinement, denylist evasion, arg cap, holds under `/yolo`); heuristic + model risk classifier; approval gating; loop guard, denial counting, success dedup; capability flags; `!` shell escape; **search_files** (locations, glob, case, binary skip, validation, confinement, bounded streaming traversal with partial-result reporting, symlink-loop and depth safety); **search_web** (harness retries, rate-limit/blocked/empty classification); **file history** (diff preview before approval, baseline snapshots, create/delete revert, untrackable files, `/diff` + `/revert`, shell side effects via workspace manifests: created files revertable, modified/deleted reported cleanly, no false positives on read-only commands); **write_file** (create/overwrite/append, diff preview, confinement, snapshots); **background run_bash** (detachment, log capture, waitpid state, risk floor, /jobs, exit trailer, log pruning that spares running jobs, slow-command handoff, **wait_background**, poll-window inference, log parsing) |
| `test_features.py` | 39 | Skills (**Agent Skills spec conformance**: name rules, frontmatter parsing incl. unquoted colons and metadata maps, `SKILL.md` directory layout, cross-client discovery paths, project-over-user shadowing, lenient validation, legacy flat files, frontmatter stripping and lazy resource listing; injection in both protocols; skills are user-supplied only (nothing shipped, discovery never reaches the agent's own directory); just-in-time background-ops guidance; **/config coverage** (a flag without a row or a mention fails the test); **/help coverage** (every dispatched command documented and tab-completable, no stale entries, no merged command literals); **session round trip** (every kind of turn state saved and restored, including across two processes; `/save <title>` persistence; a known command with a stray argument never reaching the model); **/settings** (all request overrides incl. flattened extra-body, and a useful table when the endpoint publishes no defaults); sharpened nudges that reach the retry but never history); memory (distill, search, multi-inject); compression (tail safety, budget cap, refuses to grow); sessions; system-prompt override; `/config` + `/help`; `max_tokens` + truncation hint; prefill stats; per-turn request ceiling |
| `test_recovery.py` | 25 | `strip_think`, repetition detection, context-overflow detection + tool-result shrinking, protocol-tag escaping, format helpers, `json_renderable` wrapping, exchange trim/restore; retries, nudges, DRY escalation, 400-parameter blame; the text `<tool_call>` protocol round-trip; **collapse detection** (special-token and identical-chunk cuts, discard-and-reorient, history protection, no false positives on healthy streams, bounded aborts) |
| `test_attachments.py` | 15 | `@path` inlining, multimodal image parts, read-limit tuning, policy confinement, email false-positives, content-parts flattening, wire payload |
| `test_commands.py` | 113 | `/think`, `/res`, `/settings`, `/raw [chunks]`, `/save`, `/sessions`, `/load`, delete, legacy sessions, O(N²) trimming, `/export` (markdown + HTML: escaping, inline images, collapsibles), `/restart`, approval aliases, unknown-command feedback, `_system_extras` composition, memory/skill listings; **answer verification** (accept, revise-and-replace, round cap, token budget, fail-open, prose-score fallback, leaves no trace, deterministic code/claim checks with a full false-positive matrix, best-of revert, judge-consistency rejection, median sampling); **planning** (checklist ops, multi-step heuristic with negatives, system-level injection, marker-parsed progress, JSON/numbered plan parsing, autoplan gating, session+undo persistence); **per-turn file baselines**, **/retry** (conversation + file rewind, resend through the run loop, argument validation before side effects), **--verify-command** (baseline, new-failure attribution, pre-existing failures ignored, only runs when files changed, policy-checked, fingerprint normalisation); **iterate loop** (mode resolution, tool-using fix then re-verification, stuck/regressed/round-cap stops, revise mode unchanged, progress comparator); **unattended** (visible automatic denial without prompting, read-only and yolo unaffected, request/time/token budgets, internal calls counted, CLI defaults and overrides); **report + exit codes** (green/red/budget/interactive paths, final-state establishment); **resource limits** (prefix composition, a real CPU and memory runaway stopped, verify command covered); **gaming guard** (target classification with negatives, gamed vs proper fix, shell edits caught via manifests, override flag); **--task** (single, sequenced and file-sourced tasks, UI override, missing file, and the full CI-shaped composition with verify + unattended + sandbox + report); **/model** (listing, switch by name/index, wire payload, context re-probe, native retry under auto, unknown names, session mismatch notice);  **undo/redo** (rewind, redo ordering, redo invalidation, depth cap, reverses a compaction, autosaved) |
| `test_registry_cli.py` | 14 | `tool_from_function` schema inference; `ToolRegistry` (template, hot reload, shadow protection, broken-file containment); server probing + model resolution; CLI flags reaching the wire (sampler, `--no-tools`, `--no-internet`, `--protocol text`, `--system`, dirs, read limit, `--serve` guard) |
| `test_cli_extra.py` | 24 | `--api-key` header, `--system-file`, `--tools-file`, `--hide-reasoning`, `--resume`, defense opt-outs, `--serve-host/port`, DRY flags; `choose_model`, `slash_completer`, `TokenTracker.reprint`, `EscWatcher` |
| `test_textual.py` | 9 | Headless pilots: tab set (incl. CoT), Settings scrolling + both tables, Raw wrapping with nothing clipped, first-click buffered flush, log colours, approval-prompt race, `@`/paste/completion in the prompt, Skills/Memory queue routing |

## Coverage

`audit_coverage.py` parses `py-ai.py` with `ast` and reports which module
functions, class methods, slash commands and CLI flags are referenced by the
tests — currently **64%** of the API surface, up from 39% when the suite was
first assembled.

It is deliberately a *name-based* check, so read the gaps as triage, not a
verdict. Most remaining entries fall into three groups:

1. **Covered indirectly, never named** — `Agent._memory_save` (driven via
   `/memory save`), `_compress_command` (via `/compact`), `_export_transcript`
   (via `/export`), `_replay` (via `/load`), all of `StreamPrinter` /
   `StreamAssembler` (every streaming test flows through them), the
   `AgentApp.sink_*` methods (the pilots drive them), `PromptInput.on_paste`
   (the pilot pastes), `PolicyEngine._within_root` (confinement tests).
2. **Genuinely untested, low value** — Rich line-editing setup
   (`setup_line_editing`, `build_input_prompt`), `sleep_with_progress`,
   `run_textual_ui`, and the internals of individual Textual widget
   callbacks.
3. **Untestable here** — `_serve()` beyond command assembly (it spawns a web
   server), and anything requiring a real TTY.

## Conventions worth keeping

- **Fresh temp cwd per test** (`workspace()`): path confinement, `.skills/`,
  `.memories/`, `.agent_sessions/` and file tools all work on scratch space.
- **Sinks cleared between tests** (`clear_sinks()`): a leaked
  `APPROVAL_HOOK` or `STATS_SINK` causes baffling downstream failures.
- **Assert on the wire, not just the return value.** Several real bugs were
  only visible in the HTTP payload — `server.last_request["max_tokens"]`
  proves the classifier and compression caps applied;
  `messages[0]["role"] == "system"` proves skill/memory injection happened
  per-request *without* being stored in history.
- **CLI flags are tested end-to-end** via subprocess against the fake server,
  because a flag can parse perfectly and still not be plumbed through.
- **`recorded_console()`'s getter must be called once** — rich's
  `export_text()` clears its buffer. (This bit two of these tests during
  development: a second call silently returns `""`.)
- **Prove the negative.** `NoClient` raises if the model is called at all,
  which is how the heuristic classifier, dedup cache and `!` escape are shown
  to be model-free; `test_emails_...` asserts *no* notice is printed.
- **Pilots need `pause()` after state changes.** Textual lays out
  asynchronously; a freshly activated pane can be 0-width for one refresh —
  that exact quirk was the "first click renders blank" bug.

## Bugs this suite has already caught

- `read_file`'s char cap only applied at line granularity (`and out` guard),
  so a single-line file — minified JS, one-line JSON — bypassed
  `READ_LIMIT_CHARS` entirely and could flood the context window.
- A mistyped slash command (`/thnk`) returned `False` and was **sent to the
  model as a prompt** instead of reporting an unknown command.
- `search_files` recorded the *truncated* flag but not the **reason** when the result cap was hit INSIDE a file, so a partial answer looked conclusive. It showed or hid depending on filesystem ordering, which is why it passed here and failed on a real machine.\n- The Raw-pane wrapping test scraped RichLog's private `lines[]._segments`, which differs between Textual versions -- it passed on 3.12/textual 8.x and failed on 3.10 while the feature was fine. Wrapping is now asserted against the renderable through a plain Rich console, with only `max_scroll_x` checked in the pilot.\n- A **known command with an unexpected argument** fell through the dispatcher and was sent to the model as chat, so `/save my-title` cost a request and silently did not save. Now reported, with the relevant /help line.\n- A bulk edit to the `/help` labels also hit a **dispatch tuple**, turning `("/res", "/answer", "/ans")` into one merged string so `/res` stopped working. Caught by the new /help coverage test, which now also forbids merged command literals.\n- `/help` rendered labels as Rich **markup**, so `[path]`, `[on|off]` and 11 other argument syntaxes were parsed as tags and silently deleted. Rendered as `Text()` now.\n- Slash-command arguments were sliced from the **lower-cased** input, so `/model vendor/Model3.5-0.8B` switched to a non-existent lower-cased model and `/system`, `/verify-command` and `/plan` text were mangled. Arguments now come from the raw input.\n- `ToolRegistry.maybe_reload()` detected changes by **mtime only**, so an
  edit to `custom_tools.py` saved within the same filesystem timestamp tick
  as the previous load was silently ignored — the user's new tool simply
  never appeared. Now fingerprinted by content hash. (Found because the
  test passed on one machine and failed on another: a flaky test that was
  right to be suspicious.)

## Gaps

- **No live-model tests.** Behaviour against a real llama.cpp/vLLM endpoint
  (chat-template quirks, genuine `timings`, real multimodal handling) is not
  covered; the fake server only emits what it is told to.
- **Browser (`--serve`) rendering is unverified** beyond the recursion guard
  and flag parsing — nothing clicks through xterm.js.
- **`search_web` parsing is tested against a captured HTML sample**, so a
  DuckDuckGo markup change would pass here and still fail in production.
- **No concurrency stress.** The approval race is covered; sustained
  streaming under the throttled refresh is not.
- **`--transport sdk` is untested** (it requires the `openai` package).

## If a test fails after a change

The suite is deliberately assertive about *contracts the model depends on*:
error wording that teaches the model what to do (`"empty old_str"`,
`"is a directory"`, `"Do NOT immediately re-issue"`), and refusal wording
that must not invite retries. If you reword one of those on purpose, update
the test in the same commit — the wording is load-bearing, not cosmetic.
