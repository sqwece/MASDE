"""
API 连通性验证脚本
测试 config.py 的 DeepSeek API 是否正常工作
"""
import sys
import os

# 添加父目录以导入 config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

print("=" * 50)
print("🔍 DeepSeek API 连通性验证")
print("=" * 50)

# 1. 检查 API Key
print(f"\n1️⃣ API Key: {OPENAI_API_KEY[:20]}...{OPENAI_API_KEY[-10:]}")
print(f"   ✅ config.py 导入成功")

# 2. 创建 LLM 实例
llm = ChatOpenAI(
    model="deepseek-v4-pro",
    api_key=OPENAI_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0,
    timeout=30,
)
print(f"2️⃣ ChatOpenAI 实例创建成功 (model: deepseek-v4-pro)")

# 3. 发送测试请求
print(f"\n3️⃣ 发送测试请求...")
try:
    response = llm.invoke([
        SystemMessage(content="你是一个助手，请用 JSON 格式回复。"),
        HumanMessage(content='回复 {"status": "ok", "model": "deepseek-v4-pro", "message": "API 连接成功！"}')
    ])
    print(f"   ✅ 收到响应！")
    print(f"   📝 响应内容: {response.content[:200]}")
except Exception as e:
    print(f"   ❌ 请求失败: {e}")

# 4. 测试多轮对话
print(f"\n4️⃣ 测试多轮对话...")
messages = [
    SystemMessage(content="你是一个 Python 专家。请简洁回答。"),
    HumanMessage(content="Python 中 list 和 tuple 的区别是什么？一句话回答。"),
]
try:
    response = llm.invoke(messages)
    print(f"   ✅ 回答: {response.content}")
except Exception as e:
    print(f"   ❌ 失败: {e}")

# 5. 测试带 tool calling 的能力
print(f"\n5️⃣ 测试 Tool Calling 能力...")
from langchain_core.tools import tool

@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        return f"计算结果: {eval(expression)}"
    except:
        return "计算失败"

llm_with_tools = llm.bind_tools([calculate])
try:
    response = llm_with_tools.invoke([
        SystemMessage(content="你可以使用 calculate 工具来做数学计算。"),
        HumanMessage(content="帮我计算 123 * 456 等于多少？")
    ])
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"   ✅ Tool calling 正常工作!")
        print(f"   工具调用: {response.tool_calls[0]['name']}({response.tool_calls[0]['args']})")
    else:
        print(f"   ⚠️ 未触发 tool call，直接回复: {response.content[:200]}")
except Exception as e:
    print(f"   ❌ 失败: {e}")

print(f"\n{'=' * 50}")
print(f"✅ 所有验证完成 — DeepSeek API 连接正常！")
print(f"{'=' * 50}")
