"""
Agent node implementations for each specialist in the multi-agent team.

Each agent function:
  1. Receives the shared TeamState
  2. Constructs its context from the state
  3. Invokes DeepSeek via LangChain with its specialized prompt
  4. Parses structured output
  5. Returns state updates for the next node

Architecture:
  - Reasoning agents (Analyst, Architect, Reviewer): pure LLM call
  - Tool-augmented agents (Developer, Synthesizer): LLM + tool-calling loop
  - Orchestrator: routing + quality-gate decisions
"""

import json
import re
import sys
import os
from typing import Dict, Any, List, Optional

# 将父目录加入路径以导入 config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# noinspection PyUnresolvedReferences
from config import OPENAI_API_KEY

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage, SystemMessage, AIMessage, ToolMessage
)

from .state import TeamState
from .prompts import (
    ORCHESTRATOR_SYSTEM, ANALYST_SYSTEM, ARCHITECT_SYSTEM,
    DEVELOPER_SYSTEM, REVIEWER_SYSTEM, SYNTHESIZER_SYSTEM,
)
from .tools import DEVELOPER_TOOLS, SYNTHESIZER_TOOLS, REASONING_TOOLS

# ── LLM Factory ─────────────────────────────────────────────────────────

def _create_llm(temperature: float = 0.1, bind_tools: list = None) -> ChatOpenAI:
    """Create a DeepSeek LLM instance via LangChain's ChatOpenAI.

    Uses the unified DeepSeek API endpoint (OpenAI-compatible).
    """
    llm = ChatOpenAI(
        model="deepseek-v4-pro",
        api_key=OPENAI_API_KEY,
        base_url="https://api.deepseek.com",
        temperature=temperature,
        timeout=120,
        max_retries=2,
    )
    if bind_tools:
        llm = llm.bind_tools(bind_tools)
    return llm


# ── JSON Extraction Utilities ───────────────────────────────────────────

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Robust JSON extraction from LLM output.

    Handles:
      - Plain JSON
      - JSON in ```json code blocks
      - JSON in ``` code blocks (no language)
      - Leading/trailing whitespace and artifacts
    """
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code blocks
    patterns = [
        r'```json\s*\n(.*?)\n```',
        r'```\s*\n(\{.*?\})\s*\n```',
        r'```json\s*(.*?)\s*```',
        r'```\s*(\{.*?\})\s*```',
        r'\{[\s\S]*\}',  # Last resort: find outermost braces
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except (json.JSONDecodeError, IndexError):
                continue

    return None


def _extract_json_braces(text: str) -> Optional[Dict[str, Any]]:
    """Fallback extractor: find first { and matching }."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# ── Tool Execution ──────────────────────────────────────────────────────

def _execute_tool_call(tool_call) -> str:
    """Execute a single tool call by name, return result string."""
    from . import tools as tools_module

    tool_name = tool_call.get("name", "")
    tool_args = tool_call.get("args", {})

    tool_map = {
        "write_file": tools_module.write_file,
        "read_file": tools_module.read_file,
        "list_output_files": tools_module.list_output_files,
        "estimate_complexity": tools_module.estimate_complexity,
    }

    tool_fn = tool_map.get(tool_name)
    if tool_fn:
        try:
            return tool_fn.invoke(tool_args)
        except Exception as e:
            return f"❌ 工具执行错误 ({tool_name}): {str(e)}"
    else:
        return f"❌ 未知工具: {tool_name}"


def _run_tool_loop(
    llm_with_tools,
    system_prompt: str,
    user_message: str,
    max_rounds: int = 8,
) -> str:
    """Run an agentic tool-calling loop: LLM ↔ tool execution.

    The LLM can call tools multiple times until it produces a final
    text response (without tool calls).

    Args:
        llm_with_tools: LLM instance with bound tools
        system_prompt: System prompt for the agent
        user_message: Task description
        max_rounds: Maximum tool-calling rounds (prevents infinite loops)

    Returns:
        Final text response from the agent
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    for round_num in range(max_rounds):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # Check for tool calls
        tool_calls = getattr(response, "tool_calls", None) or []

        if not tool_calls:
            # No tool calls → agent is done, return the response text
            return response.content

        # Execute all tool calls
        for tc in tool_calls:
            result = _execute_tool_call(tc)
            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tc.get("id", f"call_{round_num}"),
            ))

    # Max rounds reached — return whatever the last response was
    last_ai = [m for m in messages if isinstance(m, AIMessage)]
    return last_ai[-1].content if last_ai else ""


# ═════════════════════════════════════════════════════════════════════════
# AGENT NODE IMPLEMENTATIONS
# ═════════════════════════════════════════════════════════════════════════

# ── Orchestrator Agent (Main Controller) ─────────────────────────────────

def orchestrator_node(state: TeamState) -> Dict[str, Any]:
    """
    Chief Architect — the main controlling agent.

    Responsibilities:
      1. Read current phase and available data
      2. Decide next phase transition
      3. Route to the appropriate sub-agent
      4. Judge quality and trigger rework if needed

    This is the "brain" that controls the entire multi-agent workflow.
    """
    phase = state.get("phase", "init")
    requirement = state.get("requirement", "")
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)
    quality_threshold = state.get("quality_threshold", 70)
    task_plan = state.get("task_plan") or []
    current_task_index = state.get("current_task_index", 0)
    review_results = state.get("review_results") or []
    analysis = state.get("analysis")
    architecture = state.get("architecture")

    # Build context summary
    context_parts = [
        f"## 当前阶段: {phase}",
        f"## 用户需求: {requirement}",
    ]

    if analysis:
        context_parts.append(f"## 需求分析: {json.dumps(analysis, ensure_ascii=False, indent=2)}")

    if architecture:
        arch_summary = {
            "overview": architecture.get("overview", ""),
            "components": architecture.get("components", []),
        }
        context_parts.append(f"## 架构设计: {json.dumps(arch_summary, ensure_ascii=False, indent=2)}")

    if task_plan:
        tasks_summary = [
            {"id": t.get("id"), "title": t.get("title"), "status": t.get("status")}
            for t in task_plan
        ]
        context_parts.append(f"## 任务列表 ({current_task_index + 1}/{len(task_plan)}): {json.dumps(tasks_summary, ensure_ascii=False, indent=2)}")

    if review_results:
        last_review = review_results[-1]
        context_parts.append(
            f"## 最新审查结果: 分数={last_review.get('score', 0)}/100, "
            f"通过={last_review.get('passed', False)}, "
            f"迭代={iteration}/{max_iterations}"
        )

    context = "\n\n".join(context_parts)

    # Decision prompt
    decision_prompt = f"""{context}

请根据当前阶段做出决策。
- 当前迭代次数: {iteration}
- 最大迭代次数: {max_iterations}
- 质量阈值: {quality_threshold}

返回 JSON 格式的决策。"""

    llm = _create_llm(temperature=0.1)
    messages = [
        SystemMessage(content=ORCHESTRATOR_SYSTEM),
        HumanMessage(content=decision_prompt),
    ]
    response = llm.invoke(messages)

    # Parse decision
    result = _extract_json(response.content) or _extract_json_braces(response.content) or {}

    decision = result.get("decision", "continue")
    next_agent = result.get("next_agent", "synthesizer")
    reasoning = result.get("reasoning", "")
    feedback = result.get("feedback_for_agent", "")

    # ── Deterministic phase transitions ──
    new_phase = phase
    new_next_agent = next_agent
    new_iteration = iteration
    new_current_task_index = current_task_index

    if phase == "init":
        new_phase = "analysis"
        new_next_agent = "analyst"

    elif phase == "analysis":
        new_phase = "architecture"
        new_next_agent = "architect"

    elif phase == "architecture":
        new_phase = "development"
        new_next_agent = "developer"
        new_current_task_index = 0

    elif phase == "development":
        new_phase = "review"
        new_next_agent = "reviewer"

    elif phase == "review":
        if review_results:
            last_review = review_results[-1]
            score = last_review.get("score", 0)
            passed = last_review.get("passed", False)

            if passed or iteration >= max_iterations:
                # Move to next task or synthesize
                if current_task_index < len(task_plan) - 1:
                    new_phase = "development"
                    new_next_agent = "developer"
                    new_current_task_index = current_task_index + 1
                    new_iteration = 0
                else:
                    new_phase = "synthesis"
                    new_next_agent = "synthesizer"
            else:
                # Rework needed
                new_phase = "development"
                new_next_agent = "developer"
                new_iteration = iteration + 1
        else:
            new_phase = "synthesis"
            new_next_agent = "synthesizer"

    elif phase == "synthesis":
        new_phase = "done"
        new_next_agent = ""

    else:
        new_next_agent = ""

    # Update task status in task_plan
    updated_task_plan = list(task_plan) if task_plan else []
    if updated_task_plan and 0 <= current_task_index < len(updated_task_plan):
        t = dict(updated_task_plan[current_task_index])
        if phase == "development":
            t["status"] = "in_progress"
        elif phase == "review" and new_phase != "development":
            t["status"] = "completed"
        updated_task_plan[current_task_index] = t

    log_entry = f"[Orchestrator] {phase} → {new_phase} | → {new_next_agent} | {reasoning}"

    return {
        "phase": new_phase,
        "next_agent": new_next_agent,
        "iteration": new_iteration,
        "current_task_index": new_current_task_index,
        "task_plan": updated_task_plan,
        "execution_log": [log_entry],
        "messages": [
            AIMessage(content=f"🎯 **决策**: {reasoning}\n📋 **下一阶段**: {new_phase}\n👤 **下一执行者**: {new_next_agent}\n💬 **指令**: {feedback}")
        ],
    }


# ── Analyst Agent ───────────────────────────────────────────────────────

def analyst_node(state: TeamState) -> Dict[str, Any]:
    """Requirements Analyst — deep-dives into user requirements."""
    requirement = state.get("requirement", "")

    user_msg = f"""请分析以下用户需求：

```
{requirement}
```

请从功能需求、非功能需求、范围边界、技术考量和风险等维度进行全面分析。"""

    llm = _create_llm(temperature=0.2)
    response = llm.invoke([
        SystemMessage(content=ANALYST_SYSTEM),
        HumanMessage(content=user_msg),
    ])

    analysis = _extract_json(response.content) or _extract_json_braces(response.content) or {
        "summary": "无法解析分析结果",
        "functional_requirements": [],
        "non_functional_requirements": [],
        "scope_boundaries": [],
        "complexity_assessment": "medium",
        "tech_considerations": [],
        "risks": [],
        "raw_output": response.content,
    }

    log_entry = f"[Analyst] 分析完成: {analysis.get('summary', '')[:100]}..."

    return {
        "analysis": analysis,
        "execution_log": [log_entry],
        "messages": [
            AIMessage(content=f"📊 **需求分析完成**\n\n"
                       f"**概述**: {analysis.get('summary', 'N/A')}\n\n"
                       f"**功能需求** ({len(analysis.get('functional_requirements', []))}项):\n"
                       + "\n".join(f"  • {r}" for r in analysis.get('functional_requirements', [])[:10])
                       + f"\n\n**复杂度**: {analysis.get('complexity_assessment', 'N/A')}"),
        ],
    }


# ── Architect Agent ─────────────────────────────────────────────────────

def architect_node(state: TeamState) -> Dict[str, Any]:
    """System Architect — designs the technical architecture."""
    requirement = state.get("requirement", "")
    analysis = state.get("analysis") or {}

    user_msg = f"""## 用户需求
{requirement}

## 需求分析结果
{json.dumps(analysis, ensure_ascii=False, indent=2)}

请根据以上信息，设计完整的系统架构方案。"""

    llm = _create_llm(temperature=0.2)
    response = llm.invoke([
        SystemMessage(content=ARCHITECT_SYSTEM),
        HumanMessage(content=user_msg),
    ])

    architecture = _extract_json(response.content) or _extract_json_braces(response.content) or {
        "overview": "无法解析架构设计",
        "components": [],
        "data_flow": "",
        "tech_stack": {},
        "directory_structure": "",
        "design_rationale": "",
        "raw_output": response.content,
    }

    # Generate task plan from architecture
    components = architecture.get("components", [])
    task_plan = []
    for i, comp in enumerate(components):
        task_plan.append({
            "id": f"T{i + 1:03d}",
            "title": f"实现 {comp.get('name', f'组件{i+1}')}",
            "description": comp.get("responsibility", ""),
            "status": "pending",
            "assignee": "developer",
            "priority": "high" if i == 0 else "medium",
            "output_file": f"src/{comp.get('name', f'component_{i+1}').lower().replace(' ', '_')}.py",
        })

    # Add setup task
    task_plan.insert(0, {
        "id": "T000",
        "title": "项目初始化和配置文件",
        "description": "创建项目目录结构、配置文件和基础依赖",
        "status": "pending",
        "assignee": "developer",
        "priority": "high",
        "output_file": "requirements.txt",
    })

    # Add integration task
    task_plan.append({
        "id": f"T{len(task_plan) + 1:03d}",
        "title": "集成测试和主入口",
        "description": "创建主入口文件，集成所有组件，确保可运行",
        "status": "pending",
        "assignee": "developer",
        "priority": "high",
        "output_file": "main.py",
    })

    log_entry = f"[Architect] 架构设计完成: {len(components)} 个组件, {len(task_plan)} 个任务"

    return {
        "architecture": architecture,
        "task_plan": task_plan,
        "execution_log": [log_entry],
        "messages": [
            AIMessage(content=f"🏗️ **架构设计完成**\n\n"
                       f"**概述**: {architecture.get('overview', 'N/A')}\n\n"
                       f"**架构模式**: {architecture.get('pattern', 'N/A')}\n\n"
                       f"**组件** ({len(components)}个):\n"
                       + "\n".join(f"  • {c.get('name', '?')}: {c.get('responsibility', '?')[:80]}"
                                  for c in components)
                       + f"\n\n**任务拆解**: {len(task_plan)} 个开发任务"),
        ],
    }


# ── Developer Agent ─────────────────────────────────────────────────────

def developer_node(state: TeamState) -> Dict[str, Any]:
    """Developer — writes code with file-operation tools.

    This agent is tool-augmented: it can write/read/list files.
    It implements a ReAct-style tool-calling loop.
    """
    requirement = state.get("requirement", "")
    architecture = state.get("architecture") or {}
    task_plan = state.get("task_plan") or []
    current_task_index = state.get("current_task_index", 0)
    review_results = state.get("review_results") or []
    iteration = state.get("iteration", 0)

    # Current task
    if current_task_index < len(task_plan):
        current_task = task_plan[current_task_index]
    else:
        current_task = {"id": "T???", "title": "未知任务", "description": "无任务"}

    # Review feedback (if this is a rework iteration)
    review_feedback = ""
    if review_results and iteration > 0:
        last_review = review_results[-1]
        review_feedback = f"""
## ⚠️ 代码审查反馈（第 {iteration} 轮修改）
**分数**: {last_review.get('score', 'N/A')}/100
**问题列表**:
{json.dumps(last_review.get('issues', []), ensure_ascii=False, indent=2)}

请仔细根据以上反馈修改代码！
"""

    # List existing files
    from .tools import list_output_files
    existing_files = list_output_files.invoke({})

    user_msg = f"""## 项目需求
{requirement}

## 架构设计
{json.dumps(architecture, ensure_ascii=False, indent=2)[:2000]}

## 完整任务列表
{json.dumps([{'id': t['id'], 'title': t['title'], 'status': t['status']} for t in task_plan], ensure_ascii=False, indent=2)}

## 🎯 当前任务
**任务ID**: {current_task.get('id')}
**标题**: {current_task.get('title')}
**描述**: {current_task.get('description')}
**目标文件**: {current_task.get('output_file', '未指定')}

{review_feedback}

## 现有文件
{existing_files}

---
请完成当前任务。如果是项目初始化任务，创建目录结构、配置文件和基础代码。
使用 write_file 工具写入每个文件，最后用 list_output_files 确认所有文件已创建。"""

    llm = _create_llm(temperature=0.1, bind_tools=DEVELOPER_TOOLS)
    response_text = _run_tool_loop(llm, DEVELOPER_SYSTEM, user_msg)

    # Update code artifacts
    code_artifacts = list(state.get("code_artifacts") or [])

    log_entry = f"[Developer] 任务 {current_task.get('id')}: {current_task.get('title')} (迭代 {iteration})"

    return {
        "code_artifacts": code_artifacts,
        "execution_log": [log_entry],
        "messages": [
            AIMessage(content=f"💻 **开发完成**: {current_task.get('title')}\n\n"
                       f"**任务**: {current_task.get('id')} - {current_task.get('title')}\n"
                       f"**迭代**: {iteration}\n\n{response_text[:1500]}"),
        ],
    }


# ── Reviewer Agent ──────────────────────────────────────────────────────

def reviewer_node(state: TeamState) -> Dict[str, Any]:
    """Code Reviewer — inspects code quality and provides actionable feedback."""
    requirement = state.get("requirement", "")
    architecture = state.get("architecture") or {}
    task_plan = state.get("task_plan") or []
    current_task_index = state.get("current_task_index", 0)

    current_task = task_plan[current_task_index] if current_task_index < len(task_plan) else {}

    # Read generated files
    from .tools import list_output_files, read_file
    existing_files = list_output_files.invoke({})

    # Read key files for review
    files_content = ""
    output_dir_result = list_output_files.invoke({})
    file_list = [l.strip().split()[0] for l in output_dir_result.split("\n")[1:] if l.strip()]

    for fname in file_list[:10]:  # Review up to 10 files
        content = read_file.invoke({"filename": fname})
        files_content += f"\n### {fname}\n{content}\n---\n"

    user_msg = f"""## 需求
{requirement}

## 架构
{json.dumps(architecture, ensure_ascii=False, indent=2)[:1500]}

## 当前审查任务
{current_task.get('title', 'N/A')}: {current_task.get('description', '')}

## 已生成文件
{files_content[:6000]}

请对以上代码进行严格审查。"""

    llm = _create_llm(temperature=0.1)
    response = llm.invoke([
        SystemMessage(content=REVIEWER_SYSTEM),
        HumanMessage(content=user_msg),
    ])

    review = _extract_json(response.content) or _extract_json_braces(response.content) or {
        "score": 60,
        "passed": False,
        "issues": [{"severity": "major", "file": "unknown", "description": "无法解析审查结果", "fix_suggestion": "请重新审查"}],
        "strengths": [],
        "review_summary": "审查结果解析失败，请人工检查",
        "recommendation": "revise",
    }

    # Quality gate
    quality_threshold = state.get("quality_threshold", 70)
    score = review.get("score", 0)
    passed = score >= quality_threshold

    review["passed"] = passed

    review_results = list(state.get("review_results") or [])
    review_results.append(review)

    issues_list = review.get("issues", [])
    issues_text = "\n".join(
        f"  • [{i.get('severity', '?')}] {i.get('description', '?')[:100]}"
        for i in issues_list[:8]
    )

    log_entry = f"[Reviewer] 分数: {score}/100 → {'✅ 通过' if passed else '❌ 需修改'} ({len(issues_list)} 个问题)"

    return {
        "review_results": review_results,
        "execution_log": [log_entry],
        "messages": [
            AIMessage(content=f"🔍 **代码审查完成**\n\n"
                       f"**评分**: {score}/100 (阈值: {quality_threshold})\n"
                       f"**结果**: {'✅ 通过' if passed else '❌ 需要修改'}\n"
                       f"**问题** ({len(issues_list)}个):\n{issues_text}\n\n"
                       f"**总结**: {review.get('review_summary', 'N/A')}"),
        ],
    }


# ── Synthesizer Agent ───────────────────────────────────────────────────

def synthesizer_node(state: TeamState) -> Dict[str, Any]:
    """Synthesizer — assembles all outputs into a final deliverable document.

    This agent generates the project README with full documentation.
    """
    requirement = state.get("requirement", "")
    analysis = state.get("analysis") or {}
    architecture = state.get("architecture") or {}
    task_plan = state.get("task_plan") or []
    review_results = state.get("review_results") or []

    # List all generated files
    from .tools import list_output_files
    existing_files = list_output_files.invoke({})

    tasks_done = sum(1 for t in task_plan if t.get("status") == "completed")
    avg_score = sum(r.get("score", 0) for r in review_results) / len(review_results) if review_results else 0

    user_msg = f"""## 项目需求
{requirement}

## 需求分析
{json.dumps(analysis, ensure_ascii=False, indent=2)[:1500]}

## 架构设计
{json.dumps(architecture, ensure_ascii=False, indent=2)[:1500]}

## 开发统计
- 总任务: {len(task_plan)}
- 已完成: {tasks_done}
- 代码审查平均分: {avg_score:.1f}/100

## 已生成文件
{existing_files}

---
请创建完整的项目 README.md 文档，将所有产出汇总整理。
使用 write_file 工具将文档写入 README.md。"""

    llm = _create_llm(temperature=0.3, bind_tools=SYNTHESIZER_TOOLS)
    response_text = _run_tool_loop(llm, SYNTHESIZER_SYSTEM, user_msg)

    log_entry = f"[Synthesizer] 最终文档生成完成, 审查均分: {avg_score:.1f}"

    # Build final output summary
    final_output = f"""
# 🚀 项目开发完成

## 📊 统计
- 需求分析: ✅ 完成
- 架构设计: ✅ 完成 ({len(architecture.get('components', []))} 个组件)
- 开发任务: {tasks_done}/{len(task_plan)} 已完成
- 代码审查: 平均 {avg_score:.1f}/100 分
- 生成文件: 见 output/ 目录

## 📁 产物位置
所有生成的文件位于 `multi_agent_sde/output/` 目录下。

## 🔄 多智能体协作流程回顾
1. **Orchestrator** 接收需求，规划流程
2. **Analyst** 深度分析需求
3. **Architect** 设计系统架构，拆解任务
4. **Developer** 逐任务编写代码（含审查反馈迭代）
5. **Reviewer** 严格审查代码质量
6. **Synthesizer** 汇总生成交付文档
"""

    return {
        "final_output": final_output,
        "execution_log": [log_entry],
        "messages": [
            AIMessage(content=f"📝 **交付文档生成完成**\n\n{final_output}"),
        ],
    }
