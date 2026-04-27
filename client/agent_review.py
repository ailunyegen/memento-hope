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

# --- 路径修复: 确保无论从哪里运行都能找到 .env 文件 ---
try:
    dotenv_path = Path(__file__).parent / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        print(f"成功从 {dotenv_path} 加载环境变量。")
    else:
        print(f"警告: 在 {dotenv_path} 未找到 .env 文件。")
except Exception as e:
    print(f"加载 .env 文件时出错: {e}")


# ---------------------------------------------------------------------------
#   高级 Prompt 定义 (创新点)
# ---------------------------------------------------------------------------

# --- 动态规划 Prompt ---
# 这个 Prompt 指示 LLM 一次只思考并生成一个步骤，而不是完整的计划。
# 它需要决定是继续执行下一步，还是已经可以得出最终答案。
DYNAMIC_META_PROMPT = (
    "You are the META-PLANNER in a hierarchical AI system. Your role is to achieve a user's high-level goal by breaking it down into a sequence of single, concrete steps.\n"
    "You will receive the user's query and a history of previous steps and their outcomes.\n"
    "Based on the history, you must decide on the **single next logical step** to take.\n\n"
    "**RESPONSE FORMAT:**\n"
    "You **MUST** reply in **ONE** of the following two JSON formats:\n\n"
    "1. If more work is needed, provide the next executable task:\n"
    '{\n'
    '  "next_step": {\n'
    '    "id": INT,          // The sequential number of this step\n'
    '    "description": "STRING" // A clear, concise description of the single action to perform next.\n'
    '  }\n'
    '}\n\n'
    "2. If you are certain you have enough information to answer the user's query, provide the final answer:\n"
    '{\n'
    '  "final_answer": "STRING" // The final, direct answer to the user\'s original question.\n'
    '}\n\n'
    "**RULES:**\n"
    "- **Think step-by-step.** Do not plan multiple steps ahead.\n"
    "- **Analyze the history carefully.** The outcome of the previous step is crucial for deciding the next one.\n"
    "- Never call tools yourself. The EXECUTOR will handle the step you provide.\n"
    "- ⚠️ Reply with pure JSON only, matching one of the two formats above."
)

# --- 失败反思 Prompt ---
# 当一个步骤执行失败时，这个 Prompt 会被触发，引导 LLM 进行“复盘”。
REFLECTION_PROMPT = (
    "You are a REFLECTION agent. A previous step in a plan has failed. Your task is to analyze the situation and learn from the mistake.\n"
    "You will be given the full history of the task, including the step that failed and its error message.\n\n"
    "**YOUR ANALYSIS MUST INCLUDE:**\n"
    "1.  **Root Cause Analysis**: What was the most likely reason for the failure? (e.g., 'File not found', 'Invalid command', 'Incorrect data format').\n"
    "2.  **Lesson Learned**: What is the general principle to learn from this? (e.g., 'Always verify a file exists before trying to read it', 'API parameters must be checked carefully').\n\n"
    "**RESPONSE FORMAT:**\n"
    "You **MUST** reply in the following JSON format:\n"
    '{\n'
    '  "reflection": {\n'
    '    "failure_analysis": "STRING", // Your analysis of why the step failed.\n'
    '    "lesson_learned": "STRING"   // The general takeaway or rule to avoid this mistake in the future.\n'
    '  }\n'
    '}\n\n'
    "⚠️ Reply with pure JSON only."
)

EXEC_SYSTEM_PROMPT = (
    "You are the EXECUTOR sub-agent. You receive one task description at a time\n"
    "from the meta-planner. Your job is to complete the task, using available\n"
    "tools via function calling if needed. Always think step by step but reply\n"
    "with the minimal content needed for the meta-planner. If you must call a\n"
    "tool, produce the appropriate function call instead of natural language.\n"
    "When done, output a concise result. Do NOT output FINAL ANSWER."
)

MAX_CTX = 175000
EXE_MODEL = "deepseek-chat" # 默认执行器模型更新为 DeepSeek

# ---------------------------------------------------------------------------
#   后端 Chat 模型 (已切换至 DeepSeek)
# ---------------------------------------------------------------------------
class ChatBackend:
    async def chat(self, *_, **__) -> Dict[str, Any]:
        raise NotImplementedError

class DeepSeekBackend(ChatBackend):
    def __init__(self, model: str):
        self.model = model
        # 从环境变量加载 DeepSeek 配置
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
        is_json: bool = False, # 新增参数，用于强制JSON输出
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if is_json:
            payload["response_format"] = {"type": "json_object"}

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
        # 安全地处理 content 可能为 None 的情况
        content = msg.content or ""
        return {"content": content, "tool_calls": tool_calls}

# ---------------------------------------------------------------------------
#   分层客户端 (已重构以支持动态规划和反思)
# ---------------------------------------------------------------------------
MAX_TURNS_MEMORY = 50

def _strip_fences(text: str) -> str:
    import re
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n", "", text)
        text = re.sub(r"\n?```$", "", text)
        return text.strip()
    m = re.search(r"{[\s\S]*}", text)
    return m.group(0) if m else text

def _count_tokens(msg: Dict[str, Any], enc) -> int:
    role_tokens = 4
    content = msg.get("content") or ""
    # TODO: 更精确地处理工具调用等复杂内容的 Token 计数
    return role_tokens + len(enc.encode(content))

def _get_tokenizer(model: str):
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
    MAX_CYCLES = 15 # 增加最大循环次数以适应单步规划

    def __init__(self, meta_model: str, exec_model: str):
        self.meta_llm = DeepSeekBackend(meta_model)
        self.exec_llm = DeepSeekBackend(exec_model)
        self.sessions: Dict[str, ClientSession] = {}
        self.shared_history: List[Dict[str, Any]] = []

    async def connect_to_servers(self, scripts: List[str]):
        from contextlib import AsyncExitStack
        self.exit_stack = AsyncExitStack()
        for script in scripts:
            path = Path(script)
            # 确保使用正确的 Python 解释器
            cmd = sys.executable if path.suffix == ".py" else "node"
            params = StdioServerParameters(command=cmd, args=[str(path)])
            stdio, write = await self.exit_stack.enter_async_context(stdio_client(params))
            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
            await session.initialize()
            for tool in (await session.list_tools()).tools:
                if tool.name in self.sessions:
                    raise RuntimeError(f"Duplicate tool name '{tool.name}'.")
                self.sessions[tool.name] = session
        print("已连接的工具:", list(self.sessions.keys()))

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

    def _check_for_failure(self, result_text: str) -> bool:
        """简单的启发式方法来检测执行结果是否表示失败"""
        error_keywords = ["error", "failed", "exception", "not found", "exit code", "❌"]
        return any(keyword in result_text.lower() for keyword in error_keywords)

    async def _execute_step(self, task_desc: str, task_id: int) -> str:
        """执行单个任务步骤并返回结果"""
        tools_schema = await self._tools_schema()
        exec_msgs = (
            [{"role": "system", "content": EXEC_SYSTEM_PROMPT}] +
            self.shared_history +
            [{"role": "user", "content": task_desc}]
        )
        
        while True:
            exec_msgs = trim_messages(exec_msgs, MAX_CTX, model=EXE_MODEL)
            exec_reply = await self.exec_llm.chat(exec_msgs, tools_schema)
            
            # 如果有直接的文本回复，则认为步骤完成
            if exec_reply["content"]:
                return str(exec_reply["content"])

            # 如果有工具调用
            tool_calls = exec_reply.get("tool_calls")
            if not tool_calls:
                 return "Executor did not return content or a tool call. Step finished."

            # 执行工具调用
            exec_msgs.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
            for call in tool_calls:
                t_name = call["function"]["name"]
                try:
                    t_args = json.loads(call["function"].get("arguments") or "{}")
                    session = self.sessions[t_name]
                    result_msg = await session.call_tool(t_name, t_args)
                    result_text = str(result_msg.content)
                except Exception as e:
                    result_text = f"Tool call execution error: {e}"

                exec_msgs.append(
                    {"role": "tool", "tool_call_id": call["id"], "name": t_name, "content": result_text}
                )
            # 在这个实现中，我们假设工具调用后执行器会再次回复，这里简化为直接返回最后一次工具调用的结果
            return result_text


    async def process_query(self, query: str, file: str, task_id_str: str = "interactive") -> str:
        self.shared_history = [{"role": "user", "content": f"Goal: {query}\nAssociated file: {file}\nTask ID: {task_id_str}"}]
        
        step_id = 1
        for cycle in range(self.MAX_CYCLES):
            logger.info(f"\n----- 思考周期 {cycle + 1}/{self.MAX_CYCLES} -----")
            
            # 1. 动态规划：获取下一步
            planner_msgs = [{"role": "system", "content": DYNAMIC_META_PROMPT}] + self.shared_history
            planner_msgs = trim_messages(planner_msgs, MAX_CTX, model=self.meta_llm.model)
            
            logger.info("🤖 规划器正在思考下一步...")
            meta_reply = await self.meta_llm.chat(planner_msgs, is_json=True)
            meta_content = meta_reply["content"]

            try:
                response_json = json.loads(_strip_fences(meta_content))
                
                # 检查是否得到最终答案
                if "final_answer" in response_json:
                    final_answer = response_json["final_answer"]
                    logger.info(f"✅ 规划器得出最终答案: {final_answer}")
                    return final_answer
                
                # 获取下一步
                next_step = response_json["next_step"]
                task_desc = next_step["description"]
                self.shared_history.append({"role": "assistant", "content": meta_content})
                logger.info(f"💡 规划的下一步 (ID: {step_id}): {task_desc}")

                # 2. 执行步骤
                logger.info(f"⚙️ 执行器正在执行步骤 {step_id}...")
                result_text = await self._execute_step(f"Task {step_id}: {task_desc}", step_id)
                logger.info(f"📋 步骤 {step_id} 的结果: {result_text}")
                
                # 3. 检查失败并反思
                if self._check_for_failure(result_text):
                    logger.warning(f"⚠️ 步骤 {step_id} 检测到失败。正在启动反思...")
                    
                    # 将失败结果加入历史
                    self.shared_history.append({"role": "user", "content": f"Outcome of step {step_id}: [FAILED] {result_text}"})
                    
                    reflection_msgs = [{"role": "system", "content": REFLECTION_PROMPT}] + self.shared_history
                    reflection_msgs = trim_messages(reflection_msgs, MAX_CTX, model=self.meta_llm.model)
                    
                    reflection_reply = await self.meta_llm.chat(reflection_msgs, is_json=True)
                    reflection_content = reflection_reply["content"]

                    try:
                        reflection_json = json.loads(_strip_fences(reflection_content))
                        reflection_text = (f"[SELF-REFLECTION on FAILED step {step_id}]\n"
                                           f"Failure Analysis: {reflection_json['reflection']['failure_analysis']}\n"
                                           f"Lesson Learned: {reflection_json['reflection']['lesson_learned']}")
                        logger.info(f"🧠 反思完成: {reflection_text}")
                        self.shared_history.append({"role": "assistant", "content": reflection_text})
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(f"无法解析反思结果: {e}\n原始回复: {reflection_content}")
                        self.shared_history.append({"role": "assistant", "content": "[REFLECTION FAILED] Could not parse reflection."})

                else:
                    # 将成功结果加入历史
                    self.shared_history.append({"role": "user", "content": f"Outcome of step {step_id}: [SUCCESS] {result_text}"})

            except (json.JSONDecodeError, KeyError) as e:
                error_message = f"规划器返回了无效的JSON格式: {e}\n原始回复: {meta_content}"
                logger.error(error_message)
                self.shared_history.append({"role": "user", "content": f"[PLANNER ERROR] {error_message}"})

            step_id += 1
            
        return "任务已达到最大循环次数但未得出最终答案。"


    async def cleanup(self):
        if hasattr(self, "exit_stack"):
            await self.exit_stack.aclose()

# ---------------------------------------------------------------------------
#   命令行与主程序
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Memento - 动态规划与反思版")
    parser.add_argument("-q", "--question", type=str, help="你的问题")
    parser.add_argument("-f", "--file", type=str, default="", help="可选的文件路径")
    parser.add_argument("-m", "--meta_model", type=str, default="deepseek-chat", help="规划器/反思器模型")
    parser.add_argument("-e", "--exec_model", type=str, default="deepseek-chat", help="执行器模型")
    parser.add_argument("-s", "--servers", type=str, nargs="*", default=[
        "../server/code_agent.py",
        "../server/craw_page.py",
        "../server/documents_tool.py",
        "../server/excel_tool.py",
        "../server/image_tool.py",
        "../server/math_tool.py",
        "../server/search_tool.py",
        "../server/video_tool.py",
        "../server/ai_crawl.py",
        "../server/serp_search.py",
    ], help="工具服务器脚本的路径")
    return parser.parse_args()

async def run_single_query(client: HierarchicalClient, question: str, file_path: str):
    answer = await client.process_query(question, file_path, str(uuid.uuid4()))
    print("\n================ FINAL ANSWER ================")
    print(answer)
    print("============================================")


async def main_async(args):
    client = HierarchicalClient(args.meta_model, args.exec_model)
    try:
        await client.connect_to_servers(args.servers)

        if args.question:
            await run_single_query(client, args.question, args.file)
        else:
            print("进入交互模式。输入 'exit' 退出。")
            while True:
                q = input("\n问题: ").strip()
                if q.lower() in {"exit", "quit", "q"}:
                    break
                f = input("文件路径 (可选): ").strip()
                await run_single_query(client, q, f)
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    # 在 Windows 上设置正确的异步事件循环策略
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    arg_ns = parse_args()
    asyncio.run(main_async(arg_ns))

