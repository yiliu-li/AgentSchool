"""Import regression tests for swarm startup."""

from __future__ import annotations

import importlib
import sys


def test_create_default_tool_registry_does_not_import_mailbox_eagerly():
    for module_name in list(sys.modules):
        if module_name == "agentschool.tools" or module_name.startswith("agentschool.tools."):
            sys.modules.pop(module_name, None)
        if module_name == "agentschool.swarm" or module_name.startswith("agentschool.swarm."):
            sys.modules.pop(module_name, None)

    tools = importlib.import_module("agentschool.tools")
    registry = tools.create_default_tool_registry()

    assert registry.get("bash") is not None
    assert "agentschool.swarm.mailbox" not in sys.modules
    assert "agentschool.swarm.lockfile" not in sys.modules
