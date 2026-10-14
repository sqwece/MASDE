"""
Agent tools for the Multi-Agent Software Development Engine.

All agents use these tools via LangChain's tool-calling mechanism.
Tools provide file system operations so agents can create real project output.
"""

import os
from pathlib import Path
from typing import Optional
from datetime import datetime

from langchain_core.tools import tool

# ── Output directory ─────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def _get_output_dir() -> Path:
    """Ensure and return the output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


# ── File System Tools ───────────────────────────────────────────────────

@tool
def write_file(filename: str, content: str) -> str:
    """
    将代码/内容写入输出目录中的文件。

    Args:
        filename: 文件路径（相对于 output 目录），如 "src/main.py" 或 "README.md"
        content: 要写入的文件内容

    Returns:
        操作结果描述
    """
    output_dir = _get_output_dir()
    filepath = output_dir / filename

    # 防止路径遍历攻击
    resolved = filepath.resolve()
    if not str(resolved).startswith(str(output_dir.resolve())):
        return f"❌ 安全错误: 不允许写入 output 目录之外的文件: {filename}"

    # 确保父目录存在
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # 写入文件
    filepath.write_text(content, encoding="utf-8")

    file_size = len(content.encode("utf-8"))
    return f"✅ 成功写入 {filename} ({file_size} bytes, {content.count(chr(10)) + 1} lines)"


@tool
def read_file(filename: str) -> str:
    """
    读取输出目录中的文件内容。

    Args:
        filename: 文件路径（相对于 output 目录）

    Returns:
        文件内容，或错误信息
    """
    output_dir = _get_output_dir()
    filepath = output_dir / filename

    # 安全校验
    resolved = filepath.resolve()
    if not str(resolved).startswith(str(output_dir.resolve())):
        return f"❌ 安全错误: 不允许读取 output 目录之外的文件: {filename}"

    if not filepath.exists():
        return f"❌ 文件不存在: {filename}"

    if not filepath.is_file():
        return f"❌ 路径不是文件: {filename}"

    content = filepath.read_text(encoding="utf-8")
    if len(content) > 8000:
        content = content[:8000] + "\n\n... (truncated, file too large)"
    return f"📄 {filename}:\n```\n{content}\n```"


@tool
def list_output_files() -> str:
    """
    列出 output 目录中所有已生成的文件。

    Returns:
        文件列表（含大小和修改时间）
    """
    output_dir = _get_output_dir()
    if not output_dir.exists():
        return "📁 output 目录为空"

    files = sorted(output_dir.rglob("*"), key=lambda f: f.stat().st_mtime, reverse=True)
    regular_files = [f for f in files if f.is_file()]

    if not regular_files:
        return "📁 output 目录为空（没有文件）"

    lines = [f"📁 output/ ({len(regular_files)} files):"]
    for f in regular_files[:50]:  # 限制最多显示 50 个文件
        rel = f.relative_to(output_dir)
        size = f.stat().st_size
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%H:%M:%S")
        size_str = f"{size / 1024:.1f}KB" if size >= 1024 else f"{size}B"
        lines.append(f"  {rel}  [{size_str}]  ({mtime})")

    if len(regular_files) > 50:
        lines.append(f"  ... and {len(regular_files) - 50} more files")

    return "\n".join(lines)


@tool
def estimate_complexity(code: str) -> str:
    """
    快速估算代码复杂度（统计函数、类、行数等基础指标）。

    Args:
        code: 代码内容（字符串）

    Returns:
        复杂度统计信息
    """
    lines = code.split("\n")
    total_lines = len(lines)
    blank_lines = sum(1 for l in lines if not l.strip())
    comment_lines = sum(1 for l in lines if l.strip().startswith(("#", "//", "/*", "*", "'''", '"""')))
    code_lines = total_lines - blank_lines - comment_lines

    import re
    functions = len(re.findall(r'def\s+\w+', code))
    classes = len(re.findall(r'class\s+\w+', code))
    imports = len(re.findall(r'^(import|from)\s', code, re.MULTILINE))

    return (
        f"📊 代码统计:\n"
        f"  总行数: {total_lines}\n"
        f"  代码行: {code_lines}\n"
        f"  注释行: {comment_lines}\n"
        f"  空行: {blank_lines}\n"
        f"  函数: {functions}\n"
        f"  类: {classes}\n"
        f"  导入: {imports}\n"
        f"  复杂度评分: {'高' if functions + classes > 10 else '中' if functions + classes > 5 else '低'}"
    )


# ── Tool Lists for Different Agents ─────────────────────────────────────

# Developer and Synthesizer get file operation tools
DEVELOPER_TOOLS = [write_file, read_file, list_output_files, estimate_complexity]
SYNTHESIZER_TOOLS = [write_file, read_file, list_output_files]
# Analyst, Architect, Reviewer are pure reasoning agents (no tools)
REASONING_TOOLS = []
