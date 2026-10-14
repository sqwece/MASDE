"""
LangGraph Workflow Orchestrator
================================
Builds the multi-agent StateGraph: defines nodes, edges, conditional routing,
and the review-fix loop. This is the "circuit board" connecting all agents.

Workflow Diagram:
    START
      │
      ▼
  ┌─────────────┐
  │ Orchestrator │◄──────────────────────────────┐
  └──────┬──────┘                                │
         │                                       │
    ┌────┴────┬─────────┬──────────┬─────────┐   │
    ▼         ▼         ▼          ▼         ▼   │
┌────────┐┌────────┐┌────────┐┌────────┐┌──────┐│
│Analyst ││Architect││Developer││Reviewer││Synth ││
└───┬────┘└───┬────┘└───┬────┘└───┬────┘└──┬───┘│
    │         │         │          │        │     │
    └─────────┴─────────┴──────────┴────────┘     │
                      │                           │
                      └───────────────────────────┘
                          (review loop)
"""

from typing import Literal
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from .state import TeamState, create_initial_state
from .agents import (
    orchestrator_node,
    analyst_node,
    architect_node,
    developer_node,
    reviewer_node,
    synthesizer_node,
)


# ── Router Function ─────────────────────────────────────────────────────

def router(state: TeamState) -> Literal[
    "analyst", "architect", "developer", "reviewer", "synthesizer", "__end__"
]:
    """
    Conditional edge router: reads state['next_agent'] and routes the graph.

    This is the central switch that the Orchestrator controls.
    All agent nodes return to the Orchestrator, which sets next_agent,
    and this router dispatches accordingly.
    """
    next_agent = state.get("next_agent", "")
    phase = state.get("phase", "done")

    # Terminal condition
    if phase == "done" or next_agent == "":
        return END

    # Map agent name to graph node
    valid_agents = {
        "analyst": "analyst",
        "architect": "architect",
        "developer": "developer",
        "reviewer": "reviewer",
        "synthesizer": "synthesizer",
    }

    target = valid_agents.get(next_agent)
    if target:
        return target

    # Fallback: end if unknown agent
    return END


# ── Graph Builder ───────────────────────────────────────────────────────

def build_workflow() -> StateGraph:
    """
    Construct the multi-agent LangGraph workflow.

    Graph topology:
      - START → orchestrator
      - orchestrator → router → {analyst, architect, developer, reviewer, synthesizer, END}
      - Every sub-agent → back to orchestrator
      - Orchestrator decides next step based on phase/quality

    The review-fix loop is implicit: reviewer → orchestrator → developer → ...
    """
    builder = StateGraph(TeamState)

    # ── Add all agent nodes ──
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("architect", architect_node)
    builder.add_node("developer", developer_node)
    builder.add_node("reviewer", reviewer_node)
    builder.add_node("synthesizer", synthesizer_node)

    # ── Entry point ──
    builder.add_edge(START, "orchestrator")

    # ── Orchestrator conditional routing ──
    builder.add_conditional_edges(
        "orchestrator",
        router,
        {
            "analyst": "analyst",
            "architect": "architect",
            "developer": "developer",
            "reviewer": "reviewer",
            "synthesizer": "synthesizer",
            END: END,
        }
    )

    # ── All sub-agents return to orchestrator ──
    builder.add_edge("analyst", "orchestrator")
    builder.add_edge("architect", "orchestrator")
    builder.add_edge("developer", "orchestrator")
    builder.add_edge("reviewer", "orchestrator")
    builder.add_edge("synthesizer", "orchestrator")

    # ── Compile with memory checkpointing ──
    memory = MemorySaver()
    app = builder.compile(checkpointer=memory)

    return app


# ── Convenience Runner ──────────────────────────────────────────────────

def run_workflow(
    requirement: str,
    max_iterations: int = 3,
    quality_threshold: int = 70,
    thread_id: str = "default",
    verbose: bool = True,
) -> TeamState:
    """
    Run the complete multi-agent workflow for a given requirement.

    Args:
        requirement: The software requirement to build
        max_iterations: Max review-fix iterations per task
        quality_threshold: Minimum review score to pass (0-100)
        thread_id: Unique ID for conversation memory
        verbose: Print progress messages

    Returns:
        Final TeamState with all outputs
    """
    app = build_workflow()
    initial_state = create_initial_state(requirement)
    initial_state["max_iterations"] = max_iterations
    initial_state["quality_threshold"] = quality_threshold

    config = {"configurable": {"thread_id": thread_id}}

    if verbose:
        print(f"\n{'='*60}")
        print(f"🚀 多智能体软件工程团队启动")
        print(f"{'='*60}")
        print(f"📋 需求: {requirement[:200]}{'...' if len(requirement) > 200 else ''}")
        print(f"⚙️  最大迭代: {max_iterations} | 质量阈值: {quality_threshold}/100")
        print(f"{'='*60}\n")

    step = 0
    max_steps = 30  # Safety limit to prevent infinite loops

    for event in app.stream(
        initial_state,
        config=config,
        stream_mode="updates",
    ):
        step += 1
        if step > max_steps:
            if verbose:
                print(f"\n⚠️ 达到最大步骤限制 ({max_steps})，强制终止")
            break

        # event is a dict: {node_name: state_update}
        for node_name, update in event.items():
            if verbose and "messages" in update:
                msgs = update["messages"]
                for msg in msgs:
                    content = getattr(msg, "content", str(msg))
                    # Truncate long messages for display
                    if len(content) > 500:
                        content = content[:500] + "\n... (truncated)"
                    print(f"\n{'─'*50}")
                    print(f"[{node_name.upper()}]")
                    print(f"{'─'*50}")
                    print(content)

            if verbose and "execution_log" in update:
                for log in update.get("execution_log", []):
                    print(f"  📜 {log}")

    final_state = app.get_state(config).values

    if verbose:
        print(f"\n{'='*60}")
        print(f"✅ 工作流完成")
        print(f"{'='*60}")
        final_output = final_state.get("final_output", "")
        if final_output:
            print(final_output[:1000])

    return final_state
