# MASDE — Multi-Agent Software Development Engine

A 6-agent software engineering system built from scratch with **LangGraph**, where a **Chief Architect** orchestrates specialized agents through a complete development pipeline:

```
Requirements Analysis → Architecture Design → Code Generation →
Quality Review → Iterative Fix → Final Synthesis
```

## Architecture

```
                 ┌─────────────────────────┐
                 │  Orchestrator            │
                 │  (Chief Architect)       │
                 │  plans · routes · judges │
                 └────────────┬─────────────┘
                              │
        ┌─────────────┬───────┴───────┬──────────────┬──────────────┐
        ▼             ▼               ▼              ▼              ▼
  ┌──────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐  ┌─────────────┐
  │ Analyst  │  │ Architect  │  │ Developer  │  │ Reviewer  │  │ Synthesizer │
  │ req.     │  │ system     │  │ code gen   │  │ code      │  │ final       │
  │ analysis │  │ design     │  │ + file I/O │  │ review    │  │ assembly    │
  └──────────┘  └────────────┘  └────────────┘  └───────────┘  └─────────────┘
```

The Developer and Reviewer form a **round-trip refinement loop** — code is generated, reviewed, fixed, and re-reviewed until the quality gates pass.

## Agents

| Agent | Role | Tools |
|-------|------|-------|
| **Orchestrator** (Chief Architect) | Plans the pipeline, routes tasks, judges quality gates | — |
| **Analyst** | Deep requirements analysis | — |
| **Architect** | System design & component planning | — |
| **Developer** | Generates code with real file operations | `write_file` · `read_file` · `list_output_files` · `estimate_complexity` |
| **Reviewer** | Inspects code quality, issues fix feedback | — |
| **Synthesizer** | Assembles the final project output | `write_file` · `read_file` · `list_output_files` |

## Key Design Points

- **Hierarchical orchestration** via `langgraph` `StateGraph` — a supervisor routes work to specialized workers.
- **Round-trip refinement loop** — `code → review → fix → review` until quality gates are satisfied.
- **Typed shared state** (`TypedDict`) as the inter-agent communication protocol.
- **ReAct tool loop** — Developer/Synthesizer invoke file tools via `LangChain` tool-calling.
- **Checkpoint persistence** via `MemorySaver` for resumable runs.
- **Path-traversal protection** — file tools restrict writes/reads to the `output/` directory.

## Tech Stack

- **Orchestration**: LangGraph (`StateGraph` + conditional routing)
- **LLM**: DeepSeek (OpenAI-compatible API via `langchain_openai`)
- **Language**: Python 3.10+

## Project Structure

```
multi_agent_sde/
├── main.py          # CLI entry (interactive + batch mode)
├── orchestrator.py  # StateGraph construction + routing + run loop
├── agents.py        # 6 agent node implementations (incl. ReAct loop)
├── state.py         # TypedDict shared state (inter-agent communication protocol)
├── prompts.py       # 6 system prompts
├── tools.py         # 4 file-operation tools
└── verify_api.py    # API smoke test
```

## Getting Started

This package is designed to be run from its **parent directory** (so it can import a `config` module and run as `multi_agent_sde.main`).

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure the LLM key**

   The agents read `OPENAI_API_KEY` from a `config` module in the parent directory. Create `config.py` next to this package:

   ```python
   # config.py (in the parent directory)
   import os
   from dotenv import load_dotenv

   load_dotenv()
   OPENAI_API_KEY = os.getenv("DEEPSEEK_API_KEY")
   ```

   Then set your key:

   ```bash
   export DEEPSEEK_API_KEY=sk-...      # or put it in a .env file
   ```

3. **Run**

   ```bash
   python -m multi_agent_sde.main                          # interactive mode
   python -m multi_agent_sde.main -r "Build a REST API"    # batch mode
   ```

## Disclaimer

Built as a hands-on exercise in LLM agent orchestration — not production software.
