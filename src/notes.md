To apply decorators or wrap node functions on a `StateGraph` that has already been constructed, you can access the graph's internal nodes via the builder's `nodes` attribute, wrap the functions, and re-register them.

Since `StateGraph` nodes are stored as `Runnable` objects, you can inspect and modify them before you call `.compile()`.

### How to modify nodes after they've been added
You can iterate through the builder's nodes, wrap the underlying functions with your decorators, and use `add_node` again to overwrite them.

```python
# 1. Define your decorator
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Decorator triggered!")
        return func(*args, **kwargs)
    return wrapper

# 2. Access the existing nodes from the builder
# The builder.nodes dictionary stores the mapping of node_name: Runnable
for node_name, node_runnable in builder.nodes.items():
    # Only wrap nodes you want to decorate
    if node_name in ["node_to_decorate_1", "node_to_decorate_2"]:
        
        # 3. Access the original function
        # For simple nodes, the underlying function is often in .func
        original_func = node_runnable.func
        
        # 4. Wrap the function
        decorated_func = my_decorator(original_func)
        
        # 5. Overwrite the node in the builder
        builder.add_node(node_name, decorated_func)

# 6. Now compile the builder
graph = builder.compile()
```

### Important Considerations

* **Node Types**: If your nodes are more complex (e.g., they aren't simple function-based `Runnable`s, or they use `Command` objects), accessing `.func` might not be enough. You may need to inspect the `Runnable` structure depending on how they were originally added.
* **Idempotency**: Because `StateGraph` nodes are often re-run (e.g., during retries or after interrupts), ensure your decorators are **idempotent**. Decorators that perform side effects (like printing or logging) will execute every time the node is re-run, which might cause unexpected behavior.
* **State Updates**: If your decorator modifies the output dictionary returned by the node, ensure it still adheres to the `State` schema defined in your `StateGraph`.

**Relevant docs:**
- [StateGraph (Graph API)](https://docs.langchain.com/oss/python/langgraph/pregel#stategraph-graph-api)
- [Nodes (Conceptual)](https://docs.langchain.com/oss/python/langgraph/graph-api#nodes)

The primary issue with decorating a node that returns a `Command` object is that the `Command` is intended to be **interpreted by the LangGraph runtime**, not just passed along as data.

If your decorator processes the returned value, it must return a valid `Command` object (or the original dict) to the graph. If your decorator doesn't explicitly handle the `Command` type, you risk breaking the graph's control flow.

### Why this is tricky for guardrails

A `Command` object is the way nodes signal to the graph runtime to perform specific actions—like routing to a new node (`goto`), updating state (`update`), or responding to an interrupt (`resume`).

1. **Decorator Return Value**: If your decorator wraps the node function, the wrapped function **must still return exactly what the graph runtime expects**. If the node returns a `Command`, your decorator *must* return that `Command` (or a modified version of it).
2. **Intercepting Before the Node Acts**: If you want to intercept "incoming communications" (the inputs to the node), your decorator should wrap the function and inspect the `state` argument *before* calling the original function.
3. **Intercepting Output**: If you want to intercept the *result* (including `Command` objects), your decorator must be "Command-aware":

```python
from langgraph.types import Command

def guardrail_decorator(func):
    def wrapper(state, *args, **kwargs):
        # 1. Inspect/Intercept INCOMING state
        # validate_state(state) 
        
        # 2. Call the node logic
        result = func(state, *args, **kwargs)
        
        # 3. Intercept/Inspect OUTGOING result
        if isinstance(result, Command):
            # You can inspect result.update, result.goto, etc.
            # You MUST return the Command or a valid new Command
            return result
            
        return result
    return wrapper
```

### The "Before the Node Acts" Problem
If by "intercepting incoming comms" you mean preventing the node function from *even starting* (i.e., short-circuiting), a decorator *can* do this. However, if the condition for short-circuiting is based on the *current state*, it is often better to use **Conditional Edges** or **Command.goto** logic instead of function decorators.

If you use a decorator to short-circuit, you effectively hide that logic from the graph's trace and visualizer. Using Conditional Edges makes that routing logic explicit in your graph structure, which is generally preferred in LangGraph for observability.

**Recommendation:**
If you need to intercept flow, use `Conditional Edges` (for routing) or `Command` objects (for dynamic routing from within nodes). This keeps your guardrail logic visible in your graph definition rather than hidden inside a Python decorator, making your workflow easier to debug in LangSmith.

**Relevant docs:**
- [Conditional Edges Guide](https://docs.langchain.com/oss/python/langgraph/pregel#conditional-edges)
- [Command Type Documentation](https://reference.langchain.com/python/langgraph/types/langgraph.types.Command)