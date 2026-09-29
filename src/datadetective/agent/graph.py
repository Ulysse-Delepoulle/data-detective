"""Wire the nodes into a LangGraph state graph and run it.

The graph:

    write_code --> execute --> decide
                                  |
        success/give_up --> finalize --> END
        retry -------------------------> write_code

The conditional edge after "execute" is what makes the agent
self-correcting: on failure with attempts left it loops back to
write_code, otherwise it finalizes.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from ..llm.base import LLMBackend
from ..sandbox.config import DEFAULT_CONFIG, SandboxConfig
from .nodes import decide_next, finalize, make_execute_node, make_write_code_node
from .state import AgentState


def build_agent(backend: LLMBackend, config: SandboxConfig = DEFAULT_CONFIG):
    """Assemble and compile the agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("write_code", make_write_code_node(backend))
    graph.add_node("execute", make_execute_node(config))
    graph.add_node("finalize", finalize)

    graph.set_entry_point("write_code")
    graph.add_edge("write_code", "execute")
    # After executing, decide_next returns a label; map each label to a node.
    graph.add_conditional_edges(
        "execute",
        decide_next,
        {
            "success": "finalize",
            "retry": "write_code",
            "give_up": "finalize",
        },
    )
    graph.add_edge("finalize", END)

    return graph.compile()


def run_agent(
    question: str,
    dataset_path: str,
    backend: LLMBackend,
    max_attempts: int = 3,
    config: SandboxConfig = DEFAULT_CONFIG,
) -> AgentState:
    """Run the agent end to end and return the final state."""
    app = build_agent(backend, config)
    initial_state: AgentState = {
        "question": question,
        "dataset_path": dataset_path,
        "max_attempts": max_attempts,
        "code": "",
        "execution": None,
        "attempts": 0,
        "answer": "",
    }
    return app.invoke(initial_state)
