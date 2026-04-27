from __future__ import annotations

from tqdm import tqdm
import asyncio
import sys
import json
import os
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

import tiktoken

from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
import logging
import colorlog

# --- 日志设置 ---
LOG_FORMAT = '%(log_color)s%(levelname)-8s%(reset)s %(message)s'
colorlog.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# --- 常量定义 ---
MAX_CTX = 175000
EXE_MODEL = "deepseek-chat"
JUDGE_MODEL = "deepseek-chat"

PROMPT_TPL = '''You will be given a question and its ground truth answer list where each item can be a ground truth answer. Provided a pred_answer, you need to judge if the pred_answer correctly answers the question based on the ground truth answer list.
You should first give your rationale for the judgement, and then give your judgement result (i.e., correct or incorrect).

Here is the criteria for the judgement:
1. The pred_answer doesn't need to be exactly the same as any of the ground truth answers, but should be semantically same for the question.
2. Each item in the ground truth answer list can be viewed as a ground truth answer for the question, and the pred_answer should be semantically same to at least one of them.

question: {question}
ground truth answers: {gt_answer}
pred_answer: {pred_answer}

The output should in the following json format:


{{
  "rationale": "...",
  "judgement": "correct" | "incorrect"
}}
'''

# --- 关键修改 1: 使用动态路径并指定 UTF-8 编码 ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

query_list: List[str] = []
ground_truth_map: Dict[str, Any] = {}
data_path = PROJECT_ROOT / "data" / "military_knowledge_base.jsonl"
try:
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            try:
                data = json.loads(line)
                q = data['question']
                query_list.append(q)
                ground_truth_map[q] = data.get("ground_truth", None)
            except Exception as e:
                logger.warning(f"跳过无效行: {line[:50]}... 错误: {e}")
except FileNotFoundError:
    logger.error(f"错误: 数据文件未找到，请确保 '{data_path}' 路径下存在该文件。")
    sys.exit(1)
# --- 修改结束 ---


server_paths: list[str] = [
    str(PROJECT_ROOT / "server" / "code_agent.py"),
    str(PROJECT_ROOT / "server" / "ai_crawl.py"),
    str(PROJECT_ROOT / "server" / "documents_tool.py"),
    str(PROJECT_ROOT / "server" / "image_tool.py"),
    str(PROJECT_ROOT / "server" / "math_tool.py"),
    str(PROJECT_ROOT / "server" / "serp_search.py"),
    str(PROJECT_ROOT / "server" / "video_tool.py"),
]

load_dotenv()

META_SYSTEM_PROMPT = (
    "You are the META-PLANNER in a hierarchical AI system. A user will ask a\n"
    "high-level question. **First**: break the problem into a *minimal sequence*\n"
    "of executable tasks. Reply ONLY in JSON with the schema:\n"
    "{ \"plan\": [ {\"id\": INT, \"description\": STRING} … ] }\n\n"
    "After each task is executed by the EXECUTOR you will receive its result.\n"
    "Please carefully consider the descriptions of the time of web pages and events in the task, and take these factors into account when planning and giving the final answer.\n"
    "If the final answer is complete, output it with the template:\n"
    "FINAL ANSWER: <answer>\n\n"
    " YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.\n"
    "Please ensure that the final answer strictly follows the question requirements, without any additional analysis.\n"
    "If the final ansert is not complete, emit a *new* JSON plan for the remaining work. Keep cycles as\n"
    "few as possible. Never call tools yourself — that's the EXECUTOR's job."
    "⚠️  Reply with *pure JSON only*."
)

EXEC_SYSTEM_PROMPT = (
    "You are the EXECUTOR sub-agent. You receive one task description at a time\n"
    "from the meta-planner. Your job is to complete the task, using available\n"
    "tools via function calling if needed. Always think step by step but reply\n"
    "with the minimal content needed for the meta-planner. If you must call a\n"
    "tool, produce the appropriate function call instead of natural language.\n"
    "When done, output a concise result. Do NOT output FINAL ANSWER."
)

# --- 关键修改 2: 使用 Path 对象构建健壮的路径 ---
MEMORY_JSONL_PATH = PROJECT_ROOT / os.getenv("MEMORY_JSONL_PATH", "memory/memory.jsonl")
TRAINING_DATA_PATH = PROJECT_ROOT / os.getenv("TRAINING_DATA_PATH", "memory/training_data.jsonl")
RETRIEVER_MODEL_PATH = PROJECT_ROOT / os.getenv("RETRIEVER_MODEL_PATH", "memory/ckpts/retriever/best.pt")
# --- 修改结束 ---

MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "8"))
MEMORY_MAX_POS_EXAMPLES = int(os.getenv("MEMORY_MAX_POS_EXAMPLES", str(MEMORY_TOP_K)))
MEMORY_MAX_NEG_EXAMPLES = int(os.getenv("MEMORY_MAX_NEG_EXAMPLES", str(MEMORY_TOP_K)))

# 将项目根目录添加到系统路径中以便导入 memory 模块
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from memory.parametric_memory import CaseRetriever, load_pool
    retriever = CaseRetriever(model_path=str(RETRIEVER_MODEL_PATH))
    logger.info("记忆检索器加载成功")
except Exception as _e:
    retriever = None
    load_pool = None
    logger.warning("记忆检索器不可用: %s", _e)


def build_prompt_from_cases(task_text: str, retrieved_cases: list[dict]) -> str:
    positive_cases: list[dict] = []
    negative_cases: list[dict] = []

    for case in retrieved_cases:
        case_label = case.get('case_label', 'unknown')
        if case_label == 'positive':
            positive_cases.append(case)
        elif case_label == 'negative':
            negative_cases.append(case)

    prompt_parts: list[str] = []

    if positive_cases:
        prompt_parts.append(
            f"Positive Examples - Showing {min(len(positive_cases), MEMORY_MAX_POS_EXAMPLES)} of {len(positive_cases)}:"
        )
        for i, case in enumerate(positive_cases[:MEMORY_MAX_POS_EXAMPLES], 1):
            try:
                plan_str = case.get('plan', '')
                if isinstance(plan_str, str):
                    plan_data = json.loads(plan_str)
                else:
                    plan_data = plan_str
                plan_steps = plan_data.get('plan', [])
                plan_text = "\n".join([f"{step['id']}. {step['description']}" for step in plan_steps])
                prompt_parts.append(f"Example {i}:\nQuestion: {case['case']}\nPlan:\n{plan_text}\n")
            except Exception:
                prompt_parts.append(f"Example {i}:\nQuestion: {case.get('case','')}\nPlan: {case.get('plan','')}\n")

    if negative_cases:
        prompt_parts.append(
            f"Negative Examples - Showing {min(len(negative_cases), MEMORY_MAX_NEG_EXAMPLES)} of {len(negative_cases)}:"
        )
        for i, case in enumerate(negative_cases[:MEMORY_MAX_NEG_EXAMPLES], 1):
            prompt_parts.append(f"Example {i}:\nQuestion: {case.get('case','')}\nPlan: {case.get('plan','')}\n")

    prompt_parts.append(
        "Based on the above examples, please provide a plan for the current task. "
        "Focus on the positive examples and avoid the patterns shown in negative examples.\n\nYour plan:"
    )
    return "\n".join(prompt_parts)


def _strip_fences(text: str) -> str:
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
    return role_tokens + len(enc.encode(content))


def _get_tokenizer(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def trim_messages(messages: List[Dict[str, Any]], max_tokens: int = MAX_CTX):
    enc = _get_tokenizer(EXE_MODEL)
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


class ChatBackend:
    async def chat(self, *_, **__) -> Dict[str, Any]:
        raise NotImplementedError


class OpenAIBackend(ChatBackend):
    def __init__(self, model: str, is_azure: bool):
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
        max_tokens: int = 8000,
    ) -> Dict[str, Any]:
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise ValueError("DEEPSEEK_API_KEY 环境变量未设置")
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        resp = await self.client.chat.completions.create(**payload)
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


@dataclass
class MetaCycle:
    cycle: int
    input_messages: List[str]
    output: str


@dataclass
class ExecStep:
    task_id: int
    input: str
    output: str


@dataclass
class ToolCallRecord:
    tool: str
    arguments: Dict[str, Any]
    result: str


@dataclass
class QueryRecord:
    task_id: str
    query: str
    model_output: str
    plan_json: str
    meta_trace: List[MetaCycle]
    executor_trace: List[ExecStep]
    tool_history: List[ToolCallRecord]
    retrieved_cases: List[Dict[str, Any]] | None = None


class HierarchicalClient:
    MAX_CYCLES = 3

    def __init__(self, meta_model: str, exec_model: str, is_azure: bool = False):
        self.meta_llm = OpenAIBackend(meta_model, is_azure)
        self.exec_llm = OpenAIBackend(exec_model, is_azure)
        self.sessions: Dict[str, ClientSession] = {}
        self.shared_history: List[Dict[str, Any]] = []
        self._memory_pool = None
        self._memory_metadata = None

    async def connect_to_servers(self, scripts: List[str]):
        self.exit_stack = AsyncExitStack()
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"

        for script in scripts:
            try:
                path = Path(script)
                if not path.exists():
                    logger.warning(f"跳过不存在的服务器脚本: {path}")
                    continue

                cmd = "python" if path.suffix == ".py" else "node"
                params = StdioServerParameters(
                    command=cmd,
                    args=[str(path)],
                    env=env,
                    timeout=30 # type: ignore
                )

                try:
                    stdio, write = await self.exit_stack.enter_async_context(stdio_client(params))
                    session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
                    
                    # 设置连接初始化超时
                    try:
                        await asyncio.wait_for(session.initialize(), timeout=10.0)
                    except asyncio.TimeoutError:
                        logger.error(f"服务器连接初始化超时: {path}")
                        continue

                    tools = await session.list_tools()
                    for tool in tools.tools:
                        if tool.name in self.sessions:
                            logger.warning(f"跳过重复的工具名称: {tool.name}")
                            continue
                        self.sessions[tool.name] = session

                except Exception as e:
                    logger.error(f"连接服务器失败 {path}: {e}")
                    continue

            except Exception as e:
                logger.error(f"处理服务器脚本失败 {script}: {e}")
                continue

        if not self.sessions:
            raise RuntimeError("没有成功连接任何服务器")
            
        logger.info("已连接的工具: %s", list(self.sessions.keys()))

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

    def _load_memory(self):
        if retriever and load_pool and MEMORY_JSONL_PATH.exists():
            try:
                self._memory_pool, self._memory_metadata = load_pool(str(MEMORY_JSONL_PATH))
                logger.info("已加载 %d 条记忆条目", len(self._memory_pool))
            except Exception as e:
                logger.warning("加载记忆失败: %s", e)
                self._memory_pool = None
                self._memory_metadata = None

    def _retrieve_cases(self, query: str) -> List[Dict[str, Any]]:
        if not retriever or not self._memory_pool:
            return []
        try:
            # 减小批处理大小以降低内存使用
            batch_size = min(100, len(self._memory_pool))
            metadata = self._memory_metadata or []
            results = []
            
            # 分批处理记忆池
            for i in range(0, len(self._memory_pool), batch_size):
                batch = self._memory_pool[i:i + batch_size]
                batch_metadata = metadata[i:i + batch_size] if metadata else []
                try:
                    batch_results = retriever.retrieve(query, batch, batch_metadata)
                    results.extend(batch_results)
                except Exception as e:
                    logger.warning(f"批次处理失败 ({i}-{i+batch_size}): {e}")
                    continue
            
            # 根据得分排序并返回前K个结果
            if results:
                results.sort(key=lambda x: x.get("score", 0), reverse=True)
                return results[:MEMORY_TOP_K]
            return []
            
        except Exception as e:
            logger.warning("检索案例失败: %s", e)
            return []

    def _add_to_history(self, role: str, content: str):
        self.shared_history.append({"role": role, "content": content})

    def _memory_prompt_for(self, query: str) -> str:
        retrieved = self._retrieve_cases(query)
        if not retrieved:
            return ""
        return build_prompt_from_cases(query, retrieved)

    async def process_query(self, query: str, task_id: str = "interactive") -> QueryRecord:
        tools_schema = await self._tools_schema()
        self.shared_history = []

        self._add_to_history("user", query)

        retrieved_cases = self._retrieve_cases(query)
        mem_prompt = build_prompt_from_cases(query, retrieved_cases) if retrieved_cases else ""

        if mem_prompt:
            self._add_to_history("user", mem_prompt)

        planner_msgs = [{"role": "system", "content": META_SYSTEM_PROMPT}] + self.shared_history

        meta_trace: List[MetaCycle] = []
        executor_trace: List[ExecStep] = []
        tool_history: List[ToolCallRecord] = []
        final_answer: str = ""
        latest_plan_json: str = ""

        for cycle in range(self.MAX_CYCLES):
            meta_reply = await self.meta_llm.chat(planner_msgs)
            meta_content = meta_reply["content"] or ""
            meta_trace.append(MetaCycle(cycle, [m["content"] for m in planner_msgs if m.get("content")], meta_content))
            self._add_to_history("assistant", meta_content)

            if meta_content.startswith("FINAL ANSWER:"):
                final_answer = meta_content[len("FINAL ANSWER:") :].strip()
                break

            try:
                stripped = _strip_fences(meta_content)
                _ = json.loads(stripped)["plan"]
                latest_plan_json = stripped
            except Exception as e:
                final_answer = f"[规划器错误] {e}: {meta_content}"
                break

            tasks = json.loads(latest_plan_json)["plan"]
            for task in tasks:
                task_desc = f"Task {task['id']}: {task['description']}"
                exec_msgs = (
                    [{"role": "system", "content": EXEC_SYSTEM_PROMPT}] + self.shared_history + [{"role": "user", "content": task_desc}]
                )

                while True:
                    exec_msgs = trim_messages(exec_msgs, MAX_CTX)
                    exec_reply = await self.exec_llm.chat(exec_msgs, tools_schema)
                    if exec_reply["content"]:
                        result_text = str(exec_reply["content"])
                        executor_trace.append(ExecStep(task_id=task["id"], input=task_desc, output=result_text))
                        exec_msgs.append({"role": "assistant", "content": result_text})
                        self._add_to_history("assistant", f"Task {task['id']} result: {result_text}")
                        break

                    for call in exec_reply.get("tool_calls") or []:
                        t_name = call["function"]["name"]
                        t_args = json.loads(call["function"].get("arguments") or "{}")
                        session = self.sessions[t_name]
                        result_msg = await session.call_tool(t_name, t_args)
                        result_text = str(result_msg.content)
                        tool_history.append(ToolCallRecord(tool=t_name, arguments=t_args, result=result_text))
                        exec_msgs.extend(
                            [
                                {"role": "assistant", "content": "", "tool_calls": [call]},
                                {"role": "tool", "tool_call_id": call["id"], "name": t_name, "content": result_text},
                            ]
                        )

            planner_msgs = [{"role": "system", "content": META_SYSTEM_PROMPT}] + self.shared_history
        else:
            final_answer = meta_content.strip()

        return QueryRecord(
            task_id=task_id,
            query=query,
            model_output=final_answer,
            plan_json=latest_plan_json,
            meta_trace=meta_trace,
            executor_trace=executor_trace,
            tool_history=tool_history,
            retrieved_cases=retrieved_cases,
        )

    async def cleanup(self):
        try:
            if hasattr(self, 'exit_stack'):
                await self.exit_stack.aclose()
        except Exception as e:
            logger.error(f"清理资源时出错: {e}")


JUDGE_CLIENT = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)


def _ensure_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, (str, int, float, bool)):
        return [str(x)]
    try:
        return [json.dumps(x, ensure_ascii=False)]
    except Exception:
        return [str(x)]


async def llm_judge(question: str, ground_truth: Any, pred_answer: str) -> Dict[str, Any]:
    gt_list = _ensure_list(ground_truth)
    prompt = PROMPT_TPL.format(
        question=question,
        gt_answer=json.dumps(gt_list, ensure_ascii=False),
        pred_answer=pred_answer,
    )
    try:
        resp = await JUDGE_CLIENT.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
        )
        content = resp.choices[0].message.content or ""
        content = _strip_fences(content)
        data = json.loads(content)
        judgement = str(data.get("judgement", "incorrect")).lower().strip()
        if judgement not in ("correct", "incorrect"):
            judgement = "incorrect"
        rationale = str(data.get("rationale", ""))
        return {"judgement": judgement, "rationale": rationale}
    except Exception as e:
        logger.warning("LLM 评判失败: %s", e)
        return {"judgement": "incorrect", "rationale": f"评判失败: {e}"}


def save_training_data(query: str, retrieved_cases: List[Dict[str, Any]], is_correct: bool):
    TRAINING_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_DATA_PATH, "a", encoding="utf-8") as f:
        for case in retrieved_cases:
            training_entry = {
                "query": query,
                "case": case.get("case", ""),
                "case_label": case.get("case_label", "unknown"),
                "plan": case.get("plan", ""),
                "truth_label": is_correct
            }
            f.write(json.dumps(training_entry, ensure_ascii=False) + "\n")


def save_memory_entry(query: str, plan: str, case_label: str):
    MEMORY_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MEMORY_JSONL_PATH, "a", encoding="utf-8") as f:
        memory_entry = {
            "case": query,
            "plan": plan,
            "case_label": case_label
        }
        f.write(json.dumps(memory_entry, ensure_ascii=False) + "\n")


async def main():
    if not query_list:
        logger.warning("查询列表为空 – 请添加要处理的问题。")
        return

    finished_task = []
    result_path = PROJECT_ROOT / "result" / "result_parametric.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    if result_path.exists():
        with open(result_path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                    finished_task.append(record.get('query') or record.get('question'))
                except Exception:
                    continue

    client = None
    try:
        client = HierarchicalClient(
            os.getenv("META_MODEL", "deepseek-chat"),
            os.getenv("EXEC_MODEL", "deepseek-chat"),
            os.getenv("USE_AZURE_OPENAI") == "True",
        )
        
        try:
            await client.connect_to_servers(server_paths)
        except Exception as e:
            logger.error(f"连接服务器失败: {e}")
            return
            
        client._load_memory()

        # 设置每次处理的批次大小
        batch_size = 50
        for task_id, q in enumerate(tqdm(query_list, total=len(query_list), desc="Processing"), start=0):
            if q in finished_task:
                logger.info("任务 %s 已完成，跳过...", q)
                continue
                
            # 每处理一定数量的任务后，重新加载内存以释放资源
            if task_id > 0 and task_id % batch_size == 0:
                logger.info(f"正在重新加载内存以释放资源...")
                client._load_memory()

            try:
                rec = await client.process_query(q, str(task_id))

                pred_answer = rec.model_output
                gt = ground_truth_map.get(q)

                judge_res = await llm_judge(q, gt, pred_answer)
                is_correct = judge_res["judgement"] == "correct"

                rec_dict = asdict(rec)
                rec_dict.update({
                    "question": q,
                    "plan": rec.plan_json,
                    "ground_truth": gt,
                    "pred_answer": pred_answer,
                    "judgement": judge_res["judgement"],
                    "rationale": judge_res["rationale"],
                })

                logger.info("\n最终答案: %s", rec.model_output)
                with open(result_path, "a", encoding="utf-8") as fh:
                    json_line = json.dumps(rec_dict, ensure_ascii=False, default=str)
                    fh.write(json_line + "\n")

                if rec.retrieved_cases:
                    save_training_data(q, rec.retrieved_cases, is_correct)

                case_label = "positive" if is_correct else "negative"
                save_memory_entry(q, rec.plan_json or "", case_label)

                client._load_memory()

            except Exception as e:
                logger.error("处理查询时出错: %s", e, exc_info=True)
                continue

    finally:
        await client.cleanup() # type: ignore


if __name__ == "__main__":
    asyncio.run(main())