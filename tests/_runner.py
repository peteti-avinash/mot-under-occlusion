"""Minimal test runner shared by the test modules.

Pytest is not installed in the project virtualenv, so the suites run
standalone. This exists so more than one test module can share discovery
instead of keeping its own hand-maintained list -- a hand-maintained list
silently skips newly added tests and makes the suite look green while
proving nothing about the new behaviour.
"""

from __future__ import annotations


def run_tests(namespace, title=None):
    """Discover and run every ``test_*`` callable in ``namespace``.

    Returns a process exit code: 0 when everything passed, 1 otherwise.
    """
    tests = sorted(
        (obj for name, obj in namespace.items()
         if name.startswith("test_") and callable(obj)),
        key=lambda fn: fn.__code__.co_firstlineno,
    )
    if title:
        print(title)
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # a crash is a failure, not an abort
            failed += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        print(f"{failed} test(s) failed")
        return 1
    print("all tests passed")
    return 0