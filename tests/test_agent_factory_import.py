from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest


class AgentFactoryImportTests(unittest.TestCase):
    def test_subagent_spec_accepts_langgraph_checkpointer_values(self) -> None:
        script = textwrap.dedent(
            """
            from src.aics_agent_factory import SubAgentSpec

            common = {
                "name": "researcher",
                "description": "Researches technical questions.",
                "model": "test:model",
            }
            assert SubAgentSpec(**common).checkpointer is None
            assert SubAgentSpec(**common, checkpointer=True).checkpointer is True
            assert SubAgentSpec(**common, checkpointer=False).checkpointer is False
            """
        )

        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
