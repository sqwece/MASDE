"""
Multi-Agent Software Development Engine (MASDE)
================================================
A multi-agent system where a Chief Architect agent orchestrates
specialized sub-agents through a complete software engineering pipeline:
Analysis → Architecture Design → Code Generation → Review → Iterative Fix.

Technical Highlights:
  - Hierarchical agent orchestration via LangGraph StateGraph
  - Round-trip refinement loop (code → review → fix → review)
  - Tool-augmented agents with real file system operations
  - Structured inter-agent communication via typed state
  - Configurable quality gates and iteration limits
  - Checkpoint-based state persistence (MemorySaver)

Agents:
  - Orchestrator (Chief Architect): plans, routes, judges quality
  - Analyst: deep requirements analysis
  - Architect: system design and component planning
  - Developer: code generation with file I/O tools
  - Reviewer: code quality inspection
  - Synthesizer: final output assembly
"""

__version__ = "1.0.0"
