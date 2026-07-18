from __future__ import annotations

import asyncio
import unittest
from typing import Any, cast
from unittest.mock import patch

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph
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


class FakeSubagent:
    def __init__(self) -> None:
        self.sync_input: dict[str, Any] | None = None
        self.async_input: dict[str, Any] | None = None

    def invoke(self, input_value: dict[str, Any]) -> dict[str, list[AIMessage]]:
        self.sync_input = input_value
        return {"messages": [AIMessage(content="sync answer")]}

    async def ainvoke(
        self,
        input_value: dict[str, Any],
    ) -> dict[str, list[AIMessage]]:
        self.async_input = input_value
        return {"messages": [AIMessage(content="async answer")]}


class SubagentToolTests(unittest.TestCase):
    def test_sync_tool_returns_last_message_content(self) -> None:
        subagent = FakeSubagent()
        tool = factory.wrap_subagent_as_tool(
            make_spec(name="Technical Researcher"),
            cast(CompiledStateGraph, subagent),
        )

        result = tool.invoke({"question": "What changed?"})

        self.assertEqual(result, "sync answer")
        self.assertEqual(tool.name, "invoke_technical_researcher")
        self.assertEqual(
            subagent.sync_input,
            {"messages": [{"role": "user", "content": "What changed?"}]},
        )

    def test_async_tool_returns_last_message_content(self) -> None:
        subagent = FakeSubagent()
        tool = factory.wrap_subagent_as_tool(
            make_spec(name="Technical Researcher"),
            cast(CompiledStateGraph, subagent),
        )

        result = asyncio.run(tool.ainvoke({"question": "What changed?"}))

        self.assertEqual(result, "async answer")
        self.assertEqual(
            subagent.async_input,
            {"messages": [{"role": "user", "content": "What changed?"}]},
        )


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
