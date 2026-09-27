"""Local test configuration (not synced from upstream; tools/sync_upstream.py never overwrites this file).

PENDING lists upstream tests that are expected to fail until a pending data sync lands. Each entry must name the
sync that removes it; delete the entry in the same commit as that sync.
"""
import pytest

PENDING: dict = {}


def pytest_collection_modifyitems(config, items):
    for item in items:
        reason = PENDING.get(item.nodeid)
        if reason:
            item.add_marker(pytest.mark.xfail(reason=reason, strict=True))
