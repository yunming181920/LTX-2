"""Pure-logic unit tests for :class:`PromptBank` (the pre-encoded prompt playlist).

``PromptBank`` is a thread-safe ordered list of pre-encoded ``(prompt, context)``
entries with a current index and a queued-advance counter. This test exercises its
advance / clamp / no-skip semantics with **dummy contexts** (ints — the bank never
inspects the context tensor), so it needs no GPU / checkpoint. Importing the module
still pulls in torch, so this runs under the same `uv run python …` as the other
streaming tests (and skips gracefully where torch is unavailable).

Guarantees checked:
  * ``prefetch`` resets to the first prompt.
  * ``drain_one_advance`` applies **at most one** queued advance per call.
  * Rapid ``request_advance`` calls **queue** instead of skipping a prompt.
  * The last entry is a hard clamp (no loop); extra advances are dropped.
  * ``current_context`` / ``current_prompt`` track the applied index.

Run:

    uv run python packages/ltx-pipelines/tests/test_prompt_bank.py
"""

from __future__ import annotations

import sys
import threading


def _import_bank():  # pragma: no cover — import guard for torch-less environments
    try:
        from ltx_pipelines.interactive_session import PromptBank
    except Exception as exc:  # noqa: BLE001 — missing torch / deps in a static-review env
        print(f"SKIP: PromptBank import unavailable ({exc!r}); needs the ltx env.")
        sys.exit(0)
    return PromptBank


def main() -> None:
    PromptBank = _import_bank()

    bank = PromptBank()
    entries = [("zero", 0), ("one", 1), ("two", 2), ("three", 3)]
    n = bank.prefetch(entries)

    assert n == 4, f"prefetch count: {n}"
    assert bank.total == 4 and bank.index == 0 and bank.queued == 0
    assert bank.current_prompt() == "zero" and bank.current_context() == 0
    print(f"[init] {bank.total} entries, index {bank.index}, prompt {bank.current_prompt()!r}")

    # One advance → one step on the next drain; nothing moves until drained.
    bank.request_advance()
    assert bank.queued == 1 and bank.index == 0, "request queues, does not apply"
    assert bank.drain_one_advance() is True
    assert bank.index == 1 and bank.queued == 0
    assert bank.current_prompt() == "one" and bank.current_context() == 1
    assert bank.drain_one_advance() is False, "no pending advance → no move"
    assert bank.index == 1, "index unchanged when nothing queued"
    print(f"[one-step] advanced to index {bank.index} ({bank.current_prompt()!r})")

    # Rapid clicks queue, never skip: from index 1, click Next far past the end.
    for _ in range(10):
        bank.request_advance()
    # Max reachable from index 1 is the last index (3); pending caps at 2.
    assert bank.queued == 2, f"queue should clamp at last-reachable, got {bank.queued}"
    seen: list[int] = []
    while bank.drain_one_advance():
        seen.append(bank.current_context())
    assert seen == [2, 3], f"no-skip order broken: {seen}"
    assert bank.index == 3 and bank.queued == 0, "landed on last, queue drained"
    print(f"[no-skip] visited {seen}, clamped at last index {bank.index}")

    # Clamp at the last entry: further requests/drains are no-ops.
    bank.request_advance()
    bank.request_advance()
    assert bank.queued == 0, "no advance queueable past the last entry"
    assert bank.drain_one_advance() is False
    assert bank.index == 3 and bank.current_prompt() == "three"
    print("[clamp] stuck on last prompt, no loop")

    # prefetch resets back to the first prompt.
    bank.prefetch([("a", 10), ("b", 20)])
    assert bank.total == 2 and bank.index == 0 and bank.current_context() == 10
    bank.request_advance()
    assert bank.drain_one_advance() and bank.current_context() == 20
    bank.request_advance()
    assert bank.drain_one_advance() is False, "two-entry bank clamps after one advance"
    print("[reset] prefetch resets index; two-entry clamp ok")

    # Thread-safety smoke: many threads queue advances while one drains; the index
    # must stay within [0, total-1] and never skip an entry when drained 1:1.
    bank.prefetch([(f"p{i}", i) for i in range(20)])
    stop = threading.Event()

    def clicker() -> None:
        while not stop.is_set():
            bank.request_advance()

    threads = [threading.Thread(target=clicker) for _ in range(4)]
    for t in threads:
        t.start()
    applied = 0
    for _ in range(40):  # far more drains than entries
        if bank.drain_one_advance():
            applied += 1
        assert 0 <= bank.index <= 19, f"index out of range: {bank.index}"
    stop.set()
    for t in threads:
        t.join()
    # At most 19 advances possible on 20 entries (0 → 19); drain is clamped.
    assert bank.index == 19, f"expected to reach last index, got {bank.index}"
    assert applied <= 19, f"applied too many advances: {applied}"
    print(f"[threads] {applied} advances applied; final index {bank.index} within bounds")

    print("\nPROMPT BANK VALIDATION PASSED")


if __name__ == "__main__":
    main()
