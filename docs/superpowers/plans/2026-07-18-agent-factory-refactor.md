# Agent Factory Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `src/aics_agent_factory.py` importable and enforce predictable checkpointer, middleware-composition, and subagent-tool behavior.

**Architecture:** Keep the existing single-module factory and introduce one private middleware-composition helper. Core, caller, and per-subagent middleware remain ordered groups; duplicate names are resolved before LangChain construction. Standard-library `unittest` regression tests isolate import-time behavior from runtime factory behavior.

**Tech Stack:** Python 3.13, LangChain 1.3+, LangGraph 1.2+, Pydantic 2, `unittest`, Ruff, mypy

## Global Constraints

- `CORE_MIDDLEWARE` remains mandatory and runs first.
- The same middleware object may be repeated and must be collapsed silently.
- Different middleware objects with the same LangChain `.name` must raise `ValueError`.
- `SubAgentSpec.checkpointer` defaults to `None` but accepts `True` and `False` through LangGraph's `Checkpointer` alias.
- An adjacent source comment must explain that `True` lets a tool-wrapped subagent inherit its parent graph's checkpointer.
- Subagent tools must return `result["messages"][-1].content` in synchronous and asynchronous paths.
- Do not add a new test dependency.
- Preserve unrelated user changes in `pyproject.toml`, `uv.lock`, and `.DS_Store`.

---

## File structure

- Modify `src/aics_agent_factory.py`: public models, middleware composition, subagent creation, wrapper documentation, and formatting.
- Create `tests/test_agent_factory_import.py`: subprocess regression test for import-time Pydantic model construction and checkpointer values.
- Create `tests/test_aics_agent_factory.py`: reserved arguments, middleware composition, subagent propagation, and sync/async wrapper behavior.

### Task 1: Restore importability and define the checkpointer contract

**Files:**
- Create: `tests/test_agent_factory_import.py`
- Create: `tests/test_aics_agent_factory.py`
- Modify: `src/aics_agent_factory.py:47-56`
- Modify: `src/aics_agent_factory.py:85-130`

**Interfaces:**
- Consumes: LangGraph `Checkpointer`; Pydantic `SubAgentSpec` validation.
- Produces: `SubAgentSpec.checkpointer: Checkpointer = None`; reserved `checkpointer` keyword validation.

- [ ] **Step 1: Write the failing import and checkpointer regression test**

Create `tests/test_agent_factory_import.py`:

```python
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
```

- [ ] **Step 2: Run the import regression test and verify RED**

Run:

```bash
.venv/bin/python -m unittest tests.test_agent_factory_import.AgentFactoryImportTests.test_subagent_spec_accepts_langgraph_checkpointer_values -v
```

Expected: `FAIL`; the subprocess stderr contains `Unable to evaluate type annotation 'Checkpointer | True | False | None'`.

- [ ] **Step 3: Implement the valid checkpointer declaration and public middleware import**

In `src/aics_agent_factory.py`, use the public LangChain import:

```python
from langchain.agents.middleware import AgentMiddleware
```

Replace the checkpointer field with:

```python
    # Use True to inherit the parent graph's checkpointer when this subagent is
    # invoked as a tool.
    checkpointer: Checkpointer = None
```

- [ ] **Step 4: Run the import regression test and verify GREEN**

Run:

```bash
.venv/bin/python -m unittest tests.test_agent_factory_import.AgentFactoryImportTests.test_subagent_spec_accepts_langgraph_checkpointer_values -v
```

Expected: `OK` with one passing test.

- [ ] **Step 5: Write the failing reserved-checkpointer test**

Create `tests/test_aics_agent_factory.py` with:

```python
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
```

- [ ] **Step 6: Run the reserved-checkpointer test and verify RED**

Run:

```bash
.venv/bin/python -m unittest tests.test_aics_agent_factory.SubAgentSpecTests.test_rejects_checkpointer_in_agent_kwargs -v
```

Expected: `FAIL` because no `ValidationError` is raised.

- [ ] **Step 7: Reserve the dedicated checkpointer argument**

Add `"checkpointer"` to `_RESERVED_AGENT_KWARGS`:

```python
_RESERVED_AGENT_KWARGS = frozenset(
    {
        "model",
        "tools",
        "system_prompt",
        "middleware",
        "checkpointer",
        "name",
    }
)
```

- [ ] **Step 8: Run Task 1 tests and verify GREEN**

Run:

```bash
.venv/bin/python -m unittest tests.test_agent_factory_import tests.test_aics_agent_factory -v
```

Expected: `OK` with two passing tests.

- [ ] **Step 9: Commit Task 1**

```bash
git add src/aics_agent_factory.py tests/test_agent_factory_import.py tests/test_aics_agent_factory.py
git commit -m "fix: define agent checkpointer contract"
```

### Task 2: Compose mandatory and caller middleware safely

**Files:**
- Modify: `tests/test_aics_agent_factory.py`
- Modify: `src/aics_agent_factory.py:176-210`

**Interfaces:**
- Consumes: ordered `Sequence[AgentMiddleware]` groups and each middleware object's `.name`.
- Produces: `_combine_middleware(*groups: Sequence[AgentMiddleware]) -> tuple[AgentMiddleware, ...]`; `_effective_middleware(middleware) -> tuple[AgentMiddleware, ...]`.

- [ ] **Step 1: Write failing tests for ordering, identity deduplication, and name collisions**

Extend `tests/test_aics_agent_factory.py` imports:

```python
from unittest.mock import patch

from langchain.agents.middleware import AgentMiddleware

from src import aics_agent_factory as factory
```

Add the fixture middleware and tests:

```python
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
```

- [ ] **Step 2: Run the middleware-composition tests and verify RED**

Run:

```bash
.venv/bin/python -m unittest tests.test_aics_agent_factory.MiddlewareCompositionTests -v
```

Expected: one failure because the same caller object is repeated, plus one error because `_combine_middleware` does not exist.

- [ ] **Step 3: Implement ordered middleware composition**

Add before `_effective_middleware`:

```python
def _combine_middleware(
    *groups: Sequence[AgentMiddleware],
) -> tuple[AgentMiddleware, ...]:
    """Combine ordered middleware groups and reject ambiguous name collisions."""
    combined: list[AgentMiddleware] = []
    middleware_by_name: dict[str, AgentMiddleware] = {}

    for group in groups:
        for item in group:
            if item.name not in middleware_by_name:
                middleware_by_name[item.name] = item
                combined.append(item)
            elif middleware_by_name[item.name] is not item:
                raise ValueError(
                    f"Middleware name {item.name!r} is used by multiple instances."
                )

    return tuple(combined)
```

Replace `_effective_middleware` with:

```python
def _effective_middleware(
    middleware: Sequence[AgentMiddleware] | None,
) -> tuple[AgentMiddleware, ...]:
    """Combine mandatory core middleware with call-specific middleware."""
    return _combine_middleware(CORE_MIDDLEWARE, middleware or ())
```

- [ ] **Step 4: Run the middleware-composition tests and verify GREEN**

Run:

```bash
.venv/bin/python -m unittest tests.test_aics_agent_factory.MiddlewareCompositionTests -v
```

Expected: `OK` with two passing tests.

- [ ] **Step 5: Write the failing per-subagent middleware test**

Add:

```python
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
```

- [ ] **Step 6: Run the per-subagent middleware test and verify RED**

Run:

```bash
.venv/bin/python -m unittest tests.test_aics_agent_factory.SubagentCreationTests.test_subagent_middleware_runs_after_shared_middleware -v
```

Expected: `FAIL`; actual middleware is `[shared]` rather than `[shared, specialist]`.

- [ ] **Step 7: Apply subagent-specific middleware after the shared stack**

In `_create_subagent`, calculate and pass the complete subagent stack:

```python
    subagent_middleware = _combine_middleware(middleware, spec.middleware)

    return create_langchain_agent(
        model=spec.model,
        tools=list(spec.tools),
        system_prompt=spec.system_prompt,
        middleware=list(subagent_middleware),
        checkpointer=spec.checkpointer,
        name=safe_name,
        **dict(spec.agent_kwargs),
    )
```

- [ ] **Step 8: Run all Task 2 tests and verify GREEN**

Run:

```bash
.venv/bin/python -m unittest tests.test_aics_agent_factory.MiddlewareCompositionTests tests.test_aics_agent_factory.SubagentCreationTests -v
```

Expected: `OK` with three passing tests.

- [ ] **Step 9: Commit Task 2**

```bash
git add src/aics_agent_factory.py tests/test_aics_agent_factory.py
git commit -m "feat: compose agent middleware safely"
```

### Task 3: Preserve the standard subagent-as-tool contract

**Files:**
- Modify: `tests/test_aics_agent_factory.py`
- Modify: `src/aics_agent_factory.py:1-384`

**Interfaces:**
- Consumes: a compiled subagent exposing `invoke` and `ainvoke` and returning a `messages` sequence.
- Produces: `wrap_subagent_as_tool(spec, subagent) -> BaseTool` returning the last message's `.content` for both execution modes.

- [ ] **Step 1: Add sync and async characterization tests before refactoring**

Extend `tests/test_aics_agent_factory.py` imports:

```python
import asyncio
from typing import Any, cast

from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph
```

Add:

```python
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
```

- [ ] **Step 2: Run the wrapper characterization tests**

Run:

```bash
.venv/bin/python -m unittest tests.test_aics_agent_factory.SubagentToolTests -v
```

Expected: `OK` with two passing tests. These tests intentionally capture the already-correct LangChain wrapper behavior; no wrapper logic change is required.

- [ ] **Step 3: Clean documentation and formatting without changing wrapper logic**

In `src/aics_agent_factory.py`:

- Remove accidental doubled spaces left by the AICS-to-core terminology edit.
- Add missing blank lines between top-level declarations.
- Remove trailing whitespace.
- Keep both wrapper returns exactly as:

```python
        return result["messages"][-1].content
```

- Ensure `SubAgentSpec.middleware` documentation says it runs after the shared middleware stack.
- Ensure the module documentation says core middleware is mandatory and caller middleware is additive.

- [ ] **Step 4: Run the complete test suite**

Run:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Expected: all seven tests pass with `OK`.

- [ ] **Step 5: Run static verification**

Run:

```bash
.venv/bin/ruff check src/aics_agent_factory.py tests
.venv/bin/mypy src/aics_agent_factory.py tests
.venv/bin/python -c "import src.aics_agent_factory"
git diff --check
```

Expected: Ruff reports `All checks passed!`; mypy reports success with no issues; the import command and `git diff --check` exit with no output.

- [ ] **Step 6: Review the final diff for scope and contract coverage**

Run:

```bash
git diff -- src/aics_agent_factory.py tests
git status --short
```

Expected: only the agent-factory refactor, its tests, and this plan are part of the implementation scope. Existing unrelated modifications remain unaltered.

- [ ] **Step 7: Commit Task 3**

```bash
git add src/aics_agent_factory.py tests/test_agent_factory_import.py tests/test_aics_agent_factory.py
git commit -m "test: cover agent factory behavior"
```
