from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

from langchain.agents.middleware import AgentMiddleware
from pydantic import ValidationError

from src import aics_agent_factory as factory
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


class CoreMiddleware(AgentMiddleware):
    pass


class CallerMiddleware(AgentMiddleware):
    pass


class MiddlewareCompositionTests(unittest.TestCase):
    def test_core_runs_first_and_same_object_is_collapsed(self) -> None:
        core = CoreMiddleware()
        caller = CallerMiddleware()

        with patch.object(factory, "CORE_MIDDLEWARE", (core,)):
            result = factory._effective_middleware([caller, caller])

        self.assertEqual(result, (core, caller))

    def test_different_objects_with_same_name_are_rejected(self) -> None:
        first = CallerMiddleware()
        second = CallerMiddleware()

        with self.assertRaisesRegex(ValueError, "CallerMiddleware"):
            factory._combine_middleware((first,), (second,))


class SharedMiddleware(AgentMiddleware):
    pass


class SpecialistMiddleware(AgentMiddleware):
    pass


class SubagentCreationTests(unittest.TestCase):
    def test_subagent_middleware_runs_after_shared_middleware(self) -> None:
        shared = SharedMiddleware()
        specialist = SpecialistMiddleware()
        spec = make_spec(middleware=[specialist])
        compiled_graph = object()

        with patch.object(
            factory,
            "create_langchain_agent",
            return_value=compiled_graph,
        ) as create_langchain_agent:
            result = factory._create_subagent(spec, middleware=(shared,))

        self.assertIs(result, compiled_graph)
        self.assertEqual(
            create_langchain_agent.call_args.kwargs["middleware"],
            [shared, specialist],
        )


if __name__ == "__main__":
    unittest.main()
