from __future__ import annotations

import unittest
from typing import Any

from pydantic import ValidationError

from src.aics_agent_factory import SubAgentSpec


def make_spec(**overrides: Any) -> SubAgentSpec:
    values: dict[str, Any] = {
        "name": "researcher",
        "description": "Researches technical questions.",
        "model": "test:model",
    }
    values.update(overrides)
    return SubAgentSpec(**values)


class SubAgentSpecTests(unittest.TestCase):
    def test_rejects_checkpointer_in_agent_kwargs(self) -> None:
        with self.assertRaisesRegex(ValidationError, "checkpointer"):
            make_spec(agent_kwargs={"checkpointer": True})


if __name__ == "__main__":
    unittest.main()
