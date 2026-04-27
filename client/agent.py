from __future__ import annotations
import asyncio
import argparse
import os
import uuid
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
import logging
import colorlog
import json
import tiktoken

# ---------------------------------------------------------------------------
#   日志设置
# ---------------------------------------------------------------------------
LOG_FORMAT = '%(log_color)s%(levelname)-8s%(reset)s %(message)s'
colorlog.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#   常量和模板
# ---------------------------------------------------------------------------
META_SYSTEM_PROMPT = (
    "你是一个分层 AI 系统中的元规划器（META-PLANNER）。用户会提出一个高层次的问题。\n"
    "**首先**：将问题分解成一个*最小的可执行任务序列*。仅以 JSON 格式回复，schema 如下：\n"
    "{ \"plan\": [ {\"id\": INT, \"description\": STRING} ... ] }\n\n"
    "在执行器（EXECUTOR）执行每个任务后，你将收到其结果。\n"
    "请仔细考虑任务中网页和事件的时间描述，并在规划和给出最终答案时将这些因素考虑进去。\n"
    "如果最终答案已完整，请使用以下模板输出：\n"
    "FINAL ANSWER: <answer>\n\n" \
    "你的最终答案应该是一个数字，或尽可能少的文字，或一个逗号分隔的数字和/或字符串列表。如果你被要求提供一个数字，不要使用逗号来书写，也不要使用单位如$或百分号，除非另有说明。如果你被要求提供一个字符串，不要使用冠词和缩写（例如城市），并将数字写成纯文本，除非另有说明。如果你被要求提供一个逗号分隔的列表，请根据列表中的元素是数字还是字符串应用上述规则。\n"
    "请确保最终答案严格遵循问题要求，没有任何额外的分析。\n"
    "如果最终答案不完整，请为剩余的工作发出一个新的 JSON 计划。尽量减少循环。永远不要自己调用工具——那是执行器的工作。\n"
    "⚠️  回复*纯 JSON*。"
)

EXEC_SYSTEM_PROMPT = (
    "你是执行器（EXECUTOR）子代理。你一次从元规划器那里接收一个任务描述。\n"
    "你的工作是完成该任务，如果需要，可以通过函数调用使用可用工具。始终按步骤思考，但回复元规划器时请使用最精简的内容。\n"
    "如果必须调用工具，请生成相应的函数调用，而不是自然语言。\n"
    "完成后，输出一个简洁的结果。不要输出 FINAL ANSWER。"
)

MAX_CTX = 175000
EXE_MODEL = "deepseek-chat" # 默认执行器模型更新为 DeepSeek

# ---------------------------------------------------------------------------
#   API 后端
# ---------------------------------------------------------------------------
class ChatBackend:
    async def chat(self, *_, **__) -> Dict[str, Any]:
        raise NotImplementedError

class DeepSeekBackend(ChatBackend):
    def __init__(self, model: str):
        self.model = model
        self.client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL"),
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        tool_choice: str | None = "auto",
        max_tokens: int = 15000,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": min(max_tokens, 8192),  # DeepSeek API 限制最大值为 8192
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        resp = await self.client.chat.completions.create(**payload)  # type: ignore[arg-type]
        msg = resp.choices[0].message
        raw_calls = getattr(msg, "tool_calls", None)
        tool_calls = None
        if raw_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in raw_calls
            ]
        return {"content": msg.content, "tool_calls": tool_calls}

# ---------------------------------------------------------------------------
#   分层客户端
# ---------------------------------------------------------------------------
MAX_TURNS_MEMORY = 50

def _strip_fences(text: str) -> str:
    import re
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n", "", text)
        text = re.sub(r"\n?```$", "", text)
        return text.strip()
    m = re.search(r"{[\\s\\S]*}", text)
    return m.group(0) if m else text

def _count_tokens(msg: Dict[str, Any], enc) -> int:
    role_tokens = 4
    content = msg.get("content") or ""
    # 确保 content 是字符串
    if not isinstance(content, str):
        content = json.dumps(content)
    return role_tokens + len(enc.encode(content))

def _get_tokenizer(model: str):
    """返回一个分词器；如果模型未知，则回退到 cl100k_base。"""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")
    
def trim_messages(messages: List[Dict[str, Any]], max_tokens: int, model="gpt-3.5-turbo"):

    enc = _get_tokenizer(model)
    total = sum(_count_tokens(m, enc) for m in messages) + 2
    if total <= max_tokens:
        return messages
    system_msg = messages[0]
    kept: List[Dict[str, Any]] = [system_msg]
    total = _count_tokens(system_msg, enc) + 2
    for msg in reversed(messages[1:]):
        t = _count_tokens(msg, enc)
        if total + t > max_tokens:
            break
        kept.insert(1, msg)
        total += t
    return kept

class HierarchicalClient:
    MAX_CYCLES = 3

    def __init__(self, meta_model: str, exec_model: str):
        self.meta_llm = DeepSeekBackend(meta_model)
        self.exec_llm = DeepSeekBackend(exec_model)
        self.sessions: Dict[str, ClientSession] = {}
        self.shared_history: List[Dict[str, Any]] = []

    # ---------- 工具管理 ----------
    async def connect_to_servers(self, scripts: List[str]):
        from contextlib import AsyncExitStack
        self.exit_stack = AsyncExitStack()
        for script in scripts:
            path = Path(script)
            cmd = "python" if path.suffix == ".py" else "node"
            params = StdioServerParameters(command=cmd, args=[str(path)])
            stdio, write = await self.exit_stack.enter_async_context(stdio_client(params))
            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
            await session.initialize()
            for tool in (await session.list_tools()).tools:
                if tool.name in self.sessions:
                    raise RuntimeError(f"工具名称重复 '{tool.name}'.")
                self.sessions[tool.name] = session
        print("已连接工具:", list(self.sessions.keys()))

    async def _tools_schema(self) -> List[Dict[str, Any]]:
        result, cached = [], {}
        for session in self.sessions.values():
            tools_resp = cached.get(id(session)) or await session.list_tools()
            cached[id(session)] = tools_resp
            for tool in tools_resp.tools:
                result.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    }
                )
        return result

    # ---------- 主要处理流程 ----------
    async def process_query(self, query: str, file: str, task_id: str = "interactive") -> str:
        tools_schema = await self._tools_schema()
        self.shared_history = []
        self.shared_history.append({"role": "user", "content": f"{query}\ntask_id: {task_id}\nfile_path: {file}\n"})
        planner_msgs: List[Dict[str, Any]] = [{"role": "system", "content": META_SYSTEM_PROMPT}] + self.shared_history

        for cycle in range(self.MAX_CYCLES):
            meta_reply = await self.meta_llm.chat(planner_msgs)
            meta_content = meta_reply["content"] or ""
            self.shared_history.append({"role": "assistant", "content": meta_content})

            if meta_content.startswith("FINAL ANSWER:"):
                return meta_content[len("FINAL ANSWER:"):].strip()

            try:
                tasks = json.loads(_strip_fences(meta_content))["plan"]
            except Exception as e:
                return f"[规划器错误] {e}: {meta_content}"

            for task in tasks:
                task_desc = f"任务 {task['id']}: {task['description']}"
                exec_msgs: List[Dict[str, Any]] = (
                    [{"role": "system", "content": EXEC_SYSTEM_PROMPT}] +
                    self.shared_history +
                    [{"role": "user", "content": task_desc}]
                )
                while True:
                    exec_msgs = trim_messages(exec_msgs, MAX_CTX, model=EXE_MODEL)
                    exec_reply = await self.exec_llm.chat(exec_msgs, tools_schema)
                    if exec_reply["content"]:
                        result_text = str(exec_reply["content"])
                        self.shared_history.append({"role": "assistant", "content": f"任务 {task['id']} 结果: {result_text}"})
                        break
                    for call in exec_reply.get("tool_calls") or []:
                        t_name = call["function"]["name"]
                        t_args = json.loads(call["function"].get("arguments") or "{}")
                        session = self.sessions[t_name]
                        result_msg = await session.call_tool(t_name, t_args)
                        result_text = str(result_msg.content)
                        exec_msgs.extend([
                            {"role": "assistant", "content": None, "tool_calls": [call]},
                            {"role": "tool", "tool_call_id": call["id"], "name": t_name, "content": result_text},
                        ])
            planner_msgs = [{"role": "system", "content": META_SYSTEM_PROMPT}] + self.shared_history
        return meta_content.strip()

    async def cleanup(self):
        if hasattr(self, "exit_stack"):
            await self.exit_stack.aclose()

# ---------------------------------------------------------------------------
#   命令行及主程序
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="AgentFly – 交互版本")
    parser.add_argument("-q", "--question", type=str, help="你的问题")
    parser.add_argument("-f", "--file", type=str, default="", help="可选的文件路径")
    parser.add_argument("-m", "--meta_model", type=str, default="deepseek-chat", help="元规划器模型")
    parser.add_argument("-e", "--exec_model", type=str, default="deepseek-chat", help="执行器模型")
    base_path = Path(__file__).resolve().parent.parent / "server"
    parser.add_argument("-s", "--servers", type=str, nargs="*", default=[
        str(base_path / "code_agent.py"),
        str(base_path / "craw_page.py"),
        str(base_path / "documents_tool.py"),
        str(base_path / "excel_tool.py"),
        str(base_path / "image_tool.py"),
        str(base_path / "math_tool.py"),
        str(base_path / "search_tool.py"),
        str(base_path / "video_tool.py"),
    ], help="工具服务器脚本的路径")
    return parser.parse_args()

async def run_single_query(client: HierarchicalClient, question: str, file_path: str):
    answer = await client.process_query(question, file_path, str(uuid.uuid4()))
    print("\n最终答案:", answer)

async def main_async(args):
    # --- 修改开始 ---
    # 明确指定 .env 文件的路径
    dotenv_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=dotenv_path)
    # --- 修改结束 ---
    client = HierarchicalClient(args.meta_model, args.exec_model)
    await client.connect_to_servers(args.servers)

    try:
        if args.question:
            await run_single_query(client, args.question, args.file)
        else:
            print("输入 'exit' 退出。")
            while True:
                q = input("\n问题: ").strip()
                if q.lower() in {"exit", "quit", "q"}:
                    break
                f = input("文件路径 (可选): ").strip()
                await run_single_query(client, q, f)
    finally:
        await client.cleanup()

if __name__ == "__main__":
    arg_ns = parse_args()
    asyncio.run(main_async(arg_ns))

