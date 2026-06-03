# conftest.py — root-level pytest configuration
# Needed because a `tests/` package directory shares the name `tests`
# with the top-level tests.py file; importlib mode resolves the conflict.
collect_ignore_glob = []
