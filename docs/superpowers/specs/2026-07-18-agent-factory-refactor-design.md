# Agent Factory Refactor Design

## Goal

Refactor `src/aics_agent_factory.py` into an importable, tested LangChain agent
factory while preserving its intentionally thin subagent-as-tool behavior.

## Public behavior

- `CORE_MIDDLEWARE` remains the mandatory framework middleware stack.
- Middleware passed to `create_agent` runs after `CORE_MIDDLEWARE` and is shared
  by the supervisor and all generated subagents.
- Middleware declared on `SubAgentSpec` runs only on that subagent, after the
  shared stack.
- Middleware is considered duplicate by its LangChain `.name` value.
  Repeating the exact same object is silently collapsed. Different objects with
  the same name raise `ValueError` so configuration conflicts remain visible.
- `SubAgentSpec.checkpointer` uses LangGraph's `Checkpointer` alias and defaults
  to `None`. An adjacent source comment will state explicitly that `True` is
  supported and is the preferred setting when a tool-wrapped subagent should
  inherit the parent graph's checkpointer.
- `checkpointer` cannot also be supplied through `agent_kwargs`.
- Generated subagent tools retain the standard LangChain wrapper pattern: send
  one user message and return `result["messages"][-1].content`. Both synchronous
  and asynchronous invocation remain supported.

## Internal structure

A middleware-composition helper will accept ordered middleware groups. It will
preserve first occurrence order, collapse repeated references to the same
object, and raise for same-name/different-object collisions. The main agent will
combine core and caller middleware. Each subagent will then combine that shared
stack with its own middleware.

The existing name normalization, tool wrapping, tool-name validation, and
LangChain delegation remain focused helpers. Documentation and formatting will
be cleaned up without introducing unrelated behavior.

## Error handling

- Conflicting middleware instances raise `ValueError` before LangChain agent
  construction.
- Shim-owned values in `SubAgentSpec.agent_kwargs`, including `checkpointer`,
  raise Pydantic validation errors with the existing reserved-key message.
- Duplicate generated or ordinary tool names continue to raise `ValueError`.

## Testing

Tests will be written before production changes and will cover:

1. Importing the module and constructing a default `SubAgentSpec`.
2. Accepting `None`, `True`, and `False` as checkpointer values.
3. Rejecting `checkpointer` inside `agent_kwargs`.
4. Middleware ordering across core, caller, and subagent-specific groups.
5. Collapsing the same middleware object while rejecting distinct objects with
   the same name.
6. Returning the last message content through synchronous and asynchronous
   generated tools.

Verification will run the complete test suite, Ruff, mypy, and a direct module
import smoke test.
