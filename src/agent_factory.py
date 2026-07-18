"""Factories for creating opinionated LangChain agents and LangGraph workflows.

The module provides two related framework entry points:

* :func:`create_agent` builds a supervisor agent from ordinary tools and
  declaratively configured subagents.
* :func:`compile_state_graph` applies required decorators to the nodes of a
  fully constructed ``StateGraph`` before compiling it.

Callers describe subagents with :class:`SubAgentSpec`. :func:`create_agent`
then:

1. Creates each subagent with LangChain's native ``create_agent``.
2. Injects the framework and caller middleware into every subagent.
3. Wraps each subagent as an independent LangChain ``BaseTool``.
4. Merges the subagent tools with the supervisor's ordinary tools.
5. Creates the supervisor agent with the same shared middleware stack.

Typical usage::

    researcher = SubAgentSpec(
        name="researcher",
        description="Research technical questions and return grounded findings.",
        model="anthropic:claude-sonnet-4-6",
        system_prompt="You are a rigorous technical researcher.",
        tools=[web_search_tool],
    )

    supervisor = create_agent(
        model="anthropic:claude-opus-4-6",
        system_prompt="Delegate specialist work to the appropriate subagent.",
        tools=[calculator_tool],
        subagents=[researcher],
        middleware=[observability_middleware, guardrail_middleware],
        name="supervisor",
    )

Compiling a custom StateGraph with framework middleware::

    builder = StateGraph(State)
    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("call_model", call_model)
    builder.add_edge(START, "retrieve_context")
    builder.add_edge("retrieve_context", "call_model")
    builder.add_edge("call_model", END)

    graph = compile_state_graph(
        builder,
        nodes_to_decorate_with_guardrails=["call_model"],
    )

The supplied ``StateGraph`` must already contain all of its nodes and edges and
must not have been compiled. :func:`compile_state_graph` applies the compulsory
framework middleware decorators to every supported node function. It also
applies guardrail middleware to the named nodes, which should generally be the
nodes that invoke an LLM. The decorated builder is then compiled and returned
as a ``CompiledStateGraph``.

Agent middleware in :data:`CORE_MIDDLEWARE` is mandatory. Caller middleware
passed to :func:`create_agent` is additive: it is appended to the required
stack and shared by the supervisor and every generated subagent.

StateGraph middleware is implemented by decorating the registered node
functions immediately before compilation. The operation mutates the supplied
builder in place; it does not add, remove, or reconnect nodes and edges.

Important:
    The same middleware objects are supplied to the main agent and its
    subagents. Middleware instances should therefore be stateless or safe for
    concurrent/re-entrant use.

    State-graph node decoration relies on LangGraph's current internal node
    representation. Supported LangGraph versions should be constrained and
    both synchronous and asynchronous execution paths should be tested.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Callable, TypeAlias
from functools import wraps

import inspect

from langchain.agents import create_agent as create_langchain_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool, StructuredTool

from langgraph.graph.state import StateGraph, CompiledStateGraph
from langgraph.types import Checkpointer

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "CORE_MIDDLEWARE",
    "SubAgentSpec",
    "create_agent",
    "merge_tools_and_subagents",
    "wrap_subagent_as_tool",
]


ModelLike = str | BaseChatModel
SystemPrompt = str | SystemMessage | None
NodeFunction: TypeAlias = Callable[..., Any]
NodeDecorator: TypeAlias = Callable[[NodeFunction], NodeFunction]

# Add middleware that must always run on every agent here.
#
# Example:
#
# from core.middleware import AuditMiddleware, GuardrailMiddleware
#
# CORE_MIDDLEWARE: tuple[AgentMiddleware, ...] = (
#     AuditMiddleware(),
#     GuardrailMiddleware(),
# )
#
CORE_MIDDLEWARE: tuple[AgentMiddleware, ...] = ()


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


class SubAgentSpec(BaseModel):
    """Declarative configuration for a generated LangChain subagent.

    The shim creates the subagent with LangChain's ``create_langchain_agent``,
    injects the shared and subagent-specific middleware, and exposes the
    resulting agent as a tool to the main agent.

    Attributes:
        name:
            Name used for the generated LangChain agent and subagent tool.
        description:
            Tool description shown to the supervising agent. It should explain
            what the subagent does and when it should be used.
        model:
            Model identifier or initialized LangChain chat model.
        system_prompt:
            System prompt supplied to the subagent.
        tools:
            Tools available only to this subagent.
        middleware:
            Middleware applied only to this subagent, after the shared
            middleware stack.
        agent_kwargs:
            Additional keyword arguments forwarded to LangChain's
            ``create_langchain_agent`` when constructing this subagent.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    model: ModelLike
    system_prompt: SystemPrompt = None
    tools: list[BaseTool] = Field(default_factory=list)
    middleware: list[AgentMiddleware] = Field(default_factory=list)
    # Use True to inherit the parent graph's checkpointer when this subagent is
    # invoked as a tool.
    checkpointer: Checkpointer = None
    agent_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "description")
    @classmethod
    def validate_non_blank_text(cls, value: str) -> str:
        """Reject blank names and descriptions and trim outer whitespace."""
        value = value.strip()
        if not value:
            raise ValueError("Value must not be blank.")
        return value

    @model_validator(mode="after")
    def validate_agent_kwargs(self) -> SubAgentSpec:
        """Ensure shim-owned arguments are not supplied through agent_kwargs."""
        reserved_keys = _RESERVED_AGENT_KWARGS.intersection(
            self.agent_kwargs
        )
        if reserved_keys:
            joined = ", ".join(sorted(reserved_keys))
            raise ValueError(
                "agent_kwargs contains shim-managed keys: "
                f"{joined}. Use the corresponding SubAgentSpec fields instead."
            )
        return self


def _normalize_name(name: str) -> str:
    """Convert an arbitrary agent name into a safe snake-case identifier.

    Args:
        name:
            Raw human-readable name.
    Returns:
        A non-empty lowercase identifier containing letters, digits, and
        underscores.
    Raises:
        ValueError:
            If no usable characters remain after normalization.
    """
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    if not normalized:
        raise ValueError(
            f"Agent name {name!r} does not contain any usable characters."
        )
    if normalized[0].isdigit():
        normalized = f"agent_{normalized}"
    return normalized


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


def _effective_middleware(
    middleware: Sequence[AgentMiddleware] | None,
) -> tuple[AgentMiddleware, ...]:
    """Combine mandatory core middleware with call-specific middleware."""
    return _combine_middleware(CORE_MIDDLEWARE, middleware or ())


def _create_subagent(
    spec: SubAgentSpec,
    *,
    middleware: Sequence[AgentMiddleware],
) -> CompiledStateGraph:
    """Create one subagent directly with LangChain's ``create_langchain_agent``.

    Args:
        spec:
            Declarative subagent configuration.
        middleware:
            The complete middleware stack shared by the supervisor and every
            generated subagent.

    Returns:
        A compiled LangChain agent graph.
    """
    safe_name = _normalize_name(spec.name)
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


def wrap_subagent_as_tool(
    spec: SubAgentSpec,
    subagent: CompiledStateGraph,
) -> BaseTool:
    """Wrap an already-created subagent as a LangChain structured tool.

    Both synchronous and asynchronous execution are supported. The supervisor
    sees only one string argument, ``question``, together with the name and
    description generated from the ``SubAgentSpec``.

    Args:
        spec:
            Specification containing the exposed tool name and description.
        subagent:
            Agent produced by LangChain's ``create_langchain_agent``.

    Returns:
        A ``BaseTool`` that invokes the subagent and returns its final response.
    """
    safe_name = _normalize_name(spec.name)
    tool_name = f"invoke_{safe_name}"

    def invoke_subagent(question: str) -> str:
        """Delegate a question to the configured specialist subagent."""
        result = subagent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            }
        )
        return result["messages"][-1].content

    async def ainvoke_subagent(question: str) -> str:
        """Delegate a question to the configured specialist asynchronously."""
        result = await subagent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            }
        )
        return result["messages"][-1].content

    return StructuredTool.from_function(
        func=invoke_subagent,
        coroutine=ainvoke_subagent,
        name=tool_name,
        description=spec.description,
    )


def _validate_unique_tool_names(tools: Sequence[BaseTool]) -> None:
    """Raise when ordinary and generated tools expose duplicate names."""
    seen: set[str] = set()
    duplicates: set[str] = set()

    for tool in tools:
        if tool.name in seen:
            duplicates.add(tool.name)
        seen.add(tool.name)

    if duplicates:
        joined = ", ".join(sorted(duplicates))
        raise ValueError(f"Duplicate tool names are not allowed: {joined}.")


def merge_tools_and_subagents(
    tools: Sequence[BaseTool] | None = None,
    subagents: Sequence[SubAgentSpec] | None = None,
    *,
    middleware: Sequence[AgentMiddleware] = (),
) -> list[BaseTool]:
    """Create subagents, wrap them as tools, and merge them with normal tools.

    Subagents are created with LangChain's native ``create_langchain_agent``
    rather than recursively calling ``create_agent``. This keeps the public
    shim thin while still guaranteeing that the supplied middleware is
    installed on every generated subagent.

    Args:
        tools:
            Ordinary tools available directly to the main agent.
        subagents:
            Declarative specifications for specialist subagents.
        middleware:
            Complete middleware stack to inject into every subagent.

    Returns:
        A new list containing ordinary tools followed by generated subagent
        tools.

    Raises:
        ValueError:
            If two resulting tools have the same name.
    """
    merged_tools = list(tools or ())

    for spec in subagents or ():
        subagent = _create_subagent(spec, middleware=middleware)
        merged_tools.append(wrap_subagent_as_tool(spec, subagent))

    _validate_unique_tool_names(merged_tools)
    return merged_tools


def create_agent(
    model: ModelLike,
    tools: Sequence[BaseTool] | None = None,
    *,
    subagents: Sequence[SubAgentSpec] | None = None,
    system_prompt: SystemPrompt = None,
    middleware: Sequence[AgentMiddleware] | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> CompiledStateGraph:
    """Create an agent using LangChain's native ``create_langchain_agent``.

    This is intentionally a thin shim. Its only additional behavior is to
    construct declared subagents, install the same middleware stack on every
    agent, wrap the subagents as tools, and merge those tools with the ordinary
    tool list.

    Args:
        model:
            LangChain model identifier or initialized ``BaseChatModel`` for the
            main agent.
        tools:
            Ordinary tools exposed directly to the main agent.
        subagents:
            Declarative subagents to create and expose as separate tools.
        system_prompt:
            System prompt for the main agent.
        middleware:
            Additional middleware applied to both the main agent and every
            generated subagent. Mandatory middleware from
            :data:`CORE_MIDDLEWARE` is prepended automatically.
        name:
            Optional LangChain name for the main agent.
        **kwargs:
            All remaining arguments are forwarded unchanged to LangChain's
            ``create_langchain_agent``.

    Returns:
        The compiled LangChain agent graph.

    Notes:
        ``SubAgentSpec.agent_kwargs`` configures individual subagents. Keyword
        arguments passed directly to this function configure only the main
        agent, except for ``middleware``, which is intentionally shared with
        all generated subagents.
    """
    effective_middleware = _effective_middleware(middleware)

    merged_tools = merge_tools_and_subagents(
        tools=tools,
        subagents=subagents,
        middleware=effective_middleware,
    )

    return create_langchain_agent(
        model=model,
        tools=merged_tools,
        system_prompt=system_prompt,
        middleware=list(effective_middleware),
        name=_normalize_name(name) if name else None,
        **kwargs,
    )

## Stubs ###
def decorate_with_guardrail(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a node callable with the framework's guardrail hooks.

    The returned wrapper preserves ``func``'s metadata and execution style: an
    asynchronous callable receives an asynchronous wrapper, while a synchronous
    callable receives a synchronous wrapper.

    Args:
        func:
            Node callable to guard before and after invocation.

    Returns:
        A wrapped callable with the same synchronous or asynchronous execution
        style as ``func``.
    """
    if inspect.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Guardrail before invocation
            result = await func(*args, **kwargs)
            # Guardrail after invocation
            return result

        return async_wrapper

    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        # Guardrail before invocation
        result = func(*args, **kwargs)
        # Guardrail after invocation
        return result

    return sync_wrapper

COMPULSORY_DECORATORS: tuple[NodeDecorator, ...] = (
    # decorate_with_otel,
    # decorate_with_logging,
)

def apply_decorators(
    func: NodeFunction,
    decorators: Sequence[NodeDecorator],
) -> NodeFunction:
    # [otel, logging, guardrail] becomes:
    # otel(logging(guardrail(func)))
    for decorator in reversed(decorators):
        func = decorator(func)

    return func


def decorate(
    func: NodeFunction,
    *,
    with_guardrails: bool = False,
)-> NodeFunction:
    """
    Apply required node decorators and optionally apply guardrails.

    Compulsory decorators always wrap the node. When guardrails are requested,
    the guardrail decorator is applied closest to the original node callable,
    inside the compulsory decorators.

    Args:
        func:
            Node callable to decorate.
        guardrail_decorate:
            Whether to additionally wrap ``func`` with the framework guardrail
            decorator.

    Returns:
        The fully decorated node callable.
    """
    decorators = list(COMPULSORY_DECORATORS)

    if with_guardrails:
        decorators.append(decorate_with_guardrail)

    return apply_decorators(func, decorators)

def compile_state_graph(
    builder: StateGraph,
    nodes_to_decorate_with_guardrails: list[str],
) -> CompiledStateGraph:
    """Decorate registered node functions and compile a state graph.

    The function mutates ``builder`` in place before compilation. Every
    registered node exposing a synchronous ``func`` receives the compulsory
    decorators. Nodes named in ``nodes_to_decorate_with_guardrails`` also
    receive the guardrail decorator.

    Args:
        builder:
            Fully constructed, uncompiled state graph containing all nodes and
            edges.
        nodes_to_decorate_with_guardrails:
            Names of registered nodes that require guardrails in addition to
            the compulsory decorators.

    Returns:
        The compiled LangGraph state graph.

    Note:
        Node decoration relies on LangGraph's current internal node
        representation and therefore mutates each node runnable directly.
    """
    guarded_nodes = set(nodes_to_decorate_with_guardrails)
    unknown_nodes = guarded_nodes.difference(builder.nodes)

    if unknown_nodes:
        raise ValueError(
            f"Unknown nodes requested for guardrails: {sorted(unknown_nodes)}"
        )

    for node_name, node_spec in builder.nodes.items():
        node_runnable = node_spec.runnable
        with_guardrails = node_name in guarded_nodes

        sync_func = getattr(node_runnable, "func", None)
        async_func = getattr(node_runnable, "afunc", None)

        if sync_func is None and async_func is None:
            raise TypeError(
                f"Node {node_name!r} is not a simple function node and "
                "cannot be decorated at the function level."
            )

        if sync_func is not None:
            node_runnable.func = decorate(
                sync_func,
                with_guardrails=with_guardrails,
            )

        if async_func is not None:
            node_runnable.afunc = decorate(
                async_func,
                with_guardrails=with_guardrails,
            )

    return builder.compile()
