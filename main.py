#!/usr/bin/env python3
"""
Multi-Agent Software Development Engine (MASDE)
================================================
Entry point for the multi-agent system.

A Chief Architect agent orchestrates a team of specialized AI agents
through a complete software engineering pipeline:
  Requirements Analysis → Architecture Design → Code Generation →
  Quality Review → Iterative Fix → Final Synthesis

Usage:
    # Interactive mode
    python -m multi_agent_sde.main

    # Batch mode (from requirement file)
    python -m multi_agent_sde.main --requirement "Build a REST API for a todo app"

    # With custom quality settings
    python -m multi_agent_sde.main --requirement "..." --max-iterations 5 --quality 80

Technical Stack:
  - LangGraph: Agent workflow orchestration (StateGraph + conditional routing)
  - LangChain: LLM abstraction, tool definitions, message handling
  - DeepSeek: LLM backend via OpenAI-compatible API (from config.py)

Author: Multi-Agent SDE Team
Version: 1.0.0
"""

import argparse
import sys
import os
from pathlib import Path

# Ensure the parent directory is importable for config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# noinspection PyUnresolvedReferences
from multi_agent_sde.orchestrator import run_workflow


# ── Terminal Colors ─────────────────────────────────────────────────────

class Colors:
    """ANSI terminal color codes for pretty output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


# ── Banner ──────────────────────────────────────────────────────────────

BANNER = f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════╗
║     🏗️  Multi-Agent Software Development Engine (MASDE)     ║
║         多智能体软件工程团队 - AI-Powered Dev Team           ║
╚══════════════════════════════════════════════════════════════╝
{Colors.RESET}

{Colors.DIM}Agents: Orchestrator → Analyst → Architect → Developer → Reviewer → Synthesizer{Colors.RESET}
{Colors.DIM}Tech: LangGraph + LangChain + DeepSeek | Pattern: Supervisor-Worker + Review Loop{Colors.RESET}
"""


# ── Interactive Mode ────────────────────────────────────────────────────

def interactive_mode():
    """Run the multi-agent system in interactive CLI mode."""
    print(BANNER)
    print(f"{Colors.GREEN}💡 输入你的软件开发需求，多智能体团队将自动完成全流程。{Colors.RESET}")
    print(f"{Colors.DIM}   (输入 'quit' 退出, 'help' 查看帮助, 'config' 修改配置){Colors.RESET}\n")

    # Default configuration
    config = {
        "max_iterations": 3,
        "quality_threshold": 70,
        "show_detail": True,
    }

    session_count = 0

    while True:
        try:
            user_input = input(f"\n{Colors.BOLD}{Colors.BLUE}你 💬 > {Colors.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{Colors.YELLOW}👋 再见！{Colors.RESET}")
            break

        if not user_input:
            continue

        # Commands
        if user_input.lower() == "quit":
            print(f"{Colors.YELLOW}👋 再见！{Colors.RESET}")
            break

        if user_input.lower() == "help":
            print(f"""
{Colors.CYAN}帮助菜单{Colors.RESET}
{'─'*40}
Commands:
  quit     - 退出程序
  help     - 显示此帮助
  config   - 查看和修改配置
  clear    - 清屏

{Colors.CYAN}配置项:{Colors.RESET}
  max_iterations  - 代码审查最大迭代次数 (默认: 3)
  quality_threshold - 代码审查通过分数 (默认: 70)
  show_detail     - 是否显示详细过程 (默认: True)

{Colors.CYAN}示例需求:{Colors.RESET}
  • 创建一个基于 Flask 的 REST API，支持用户注册、登录和 JWT 认证
  • 实现一个 Python 异步任务队列系统，支持任务优先级和重试
  • 构建一个简单的 key-value 数据库，支持持久化和 TTL
  • 开发一个 Markdown 转 HTML 的 CLI 工具，支持自定义主题
""")
            continue

        if user_input.lower() == "config":
            print(f"""
{Colors.CYAN}当前配置:{Colors.RESET}
  max_iterations  = {config['max_iterations']}
  quality_threshold = {config['quality_threshold']}
  show_detail     = {config['show_detail']}
""")
            try:
                new_max = input(f"  max_iterations [{config['max_iterations']}]: ").strip()
                if new_max:
                    config["max_iterations"] = int(new_max)

                new_q = input(f"  quality_threshold [{config['quality_threshold']}]: ").strip()
                if new_q:
                    config["quality_threshold"] = int(new_q)

                new_d = input(f"  show_detail (y/n) [{'y' if config['show_detail'] else 'n'}]: ").strip().lower()
                if new_d == "y":
                    config["show_detail"] = True
                elif new_d == "n":
                    config["show_detail"] = False
            except ValueError:
                print(f"{Colors.RED}❌ 配置值无效，保持原配置{Colors.RESET}")
            continue

        if user_input.lower() == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            print(BANNER)
            continue

        # ── Process requirement ──
        session_count += 1
        thread_id = f"session_{session_count}"

        print(f"\n{Colors.YELLOW}🔧 团队开始工作...{Colors.RESET}\n")

        try:
            final_state = run_workflow(
                requirement=user_input,
                max_iterations=config["max_iterations"],
                quality_threshold=config["quality_threshold"],
                thread_id=thread_id,
                verbose=config["show_detail"],
            )

            # Success summary
            print(f"\n{Colors.GREEN}{Colors.BOLD}✅ 项目开发完成！{Colors.RESET}")
            output_dir = Path(__file__).resolve().parent / "output"
            if output_dir.exists():
                files = list(output_dir.rglob("*"))
                py_files = [f for f in files if f.suffix == ".py"]
                print(f"{Colors.GREEN}📁 输出目录: {output_dir}{Colors.RESET}")
                print(f"{Colors.GREEN}📄 生成文件: {len([f for f in files if f.is_file()])} 个{Colors.RESET}")
                if py_files:
                    print(f"{Colors.GREEN}🐍 Python文件: {len(py_files)} 个{Colors.RESET}")

        except Exception as e:
            print(f"{Colors.RED}❌ 工作流执行出错: {e}{Colors.RESET}")
            import traceback
            traceback.print_exc()


# ── Batch Mode ──────────────────────────────────────────────────────────

def batch_mode(
    requirement: str,
    max_iterations: int = 3,
    quality_threshold: int = 70,
    verbose: bool = True,
):
    """Run the workflow once from CLI arguments."""
    print(BANNER)

    final_state = run_workflow(
        requirement=requirement,
        max_iterations=max_iterations,
        quality_threshold=quality_threshold,
        thread_id="cli_batch",
        verbose=verbose,
    )

    print(f"\n{Colors.GREEN}{Colors.BOLD}✅ 完成！输出文件位于 multi_agent_sde/output/{Colors.RESET}")

    return final_state


# ── CLI Entry Point ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Software Development Engine (MASDE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python -m multi_agent_sde.main

  # Batch mode
  python -m multi_agent_sde.main -r "Build a REST API for a todo app"

  # Strict quality requirements
  python -m multi_agent_sde.main -r "Build a key-value database" --quality 85 --max-iterations 5
        """,
    )

    parser.add_argument(
        "-r", "--requirement",
        type=str,
        help="Software requirement to process (if omitted, enters interactive mode)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum review-fix iterations per task (default: 3)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=70,
        help="Quality threshold for code review, 0-100 (default: 70)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed progress output",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory (default: multi_agent_sde/output)",
    )

    args = parser.parse_args()

    # Override output directory if specified
    if args.output_dir:
        # noinspection PyUnresolvedReferences
        from multi_agent_sde import tools
        tools.OUTPUT_DIR = Path(args.output_dir)
        tools.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.requirement:
        # Batch mode
        batch_mode(
            requirement=args.requirement,
            max_iterations=args.max_iterations,
            quality_threshold=args.quality,
            verbose=not args.quiet,
        )
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
