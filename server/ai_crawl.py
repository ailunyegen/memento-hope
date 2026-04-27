import os
import logging
from typing import Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv

import colorlog
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from openai import AsyncOpenAI
from crawl4ai import AsyncWebCrawler
from mcp.server.fastmcp import FastMCP

# --- 修改开始 ---
dotenv_path = Path(__file__).resolve().parent.parent / "client" / ".env"
load_dotenv(dotenv_path=dotenv_path)
# --- 修改结束 ---

# --------------------------------------------------------------------------- #
#  日志
# --------------------------------------------------------------------------- #
LOG_FORMAT = '%(log_color)s%(levelname)-8s%(reset)s %(message)s'
colorlog.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("mcp.crawl_extract")

# --------------------------------------------------------------------------- #
#  固定模型
# --------------------------------------------------------------------------- #
DEFAULT_MODEL = "deepseek-chat"

EXTRACTOR_SYSTEM_PROMPT = (
    "你是一个谨慎、简洁的信息提取助手。"
    "给定用户查询和 Markdown 格式的网页内容，不要总结整个页面。"
    "只提取与查询直接相关或高度相关的内容。"
    "如果答案不存在，请明确说明。\n\n"
    "输出格式：\n"
    "1) 直接答案：2-4个简洁的句子（或 '在页面中未找到'）。\n"
    "2) 关键证据：从 Markdown 中提取的简短引用的项目符号列表（逐字引用，最少删减）。\n"
    "3) 实体/数字：与查询相关的重要名称、日期、数字的项目符号列表。\n"
    "4) 不确定性：注意任何模棱两可或缺失的信息。\n"
    "严格基于提供的 Markdown 内容。避免捏造。"
)

# --------------------------------------------------------------------------- #
#  聊天后端
# --------------------------------------------------------------------------- #
class DeepSeekBackend:
    def __init__(self):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("缺少 DEEPSEEK_API_KEY 环境变量。")
        base_url = os.getenv("DEEPSEEK_BASE_URL")
        self.model = DEFAULT_MODEL  
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 30000,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = await self.client.chat.completions.create(**payload)
        msg = resp.choices[0].message
        return {"content": msg.content or ""}

# --------------------------------------------------------------------------- #
#  辅助函数
# --------------------------------------------------------------------------- #
async def _extract_for_query(
    backend: DeepSeekBackend,
    md: str,
    query: str,
    *,
    max_tokens: int = 30000,
    temperature: float = 0.1,
) -> str:
    messages = [
        {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": f"用户查询:\n{query}\n\n页面 Markdown (原文):\n\n{md}"},
    ]
    res = await backend.chat(messages=messages, max_tokens=max_tokens, temperature=temperature)
    return (res["content"] or "").strip()

async def _crawl_markdown(url: str) -> str:
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        md = getattr(result, 'markdown', None)
        return (md or "").strip()


async def _crawl_and_extract(
    url: str,
    query: str,
    *,
    max_tokens: int = 30000,
    temperature: float = 0.1,
) -> str:
    logger.info(f"正在抓取: {url}")
    md = await _crawl_markdown(url)
    logger.info(f"Markdown 长度: {len(md):,} 字符")
    backend = DeepSeekBackend()  
    return await _extract_for_query(
        backend,
        md,
        query=query,
        max_tokens=max_tokens,
        temperature=temperature,
    )

# --------------------------------------------------------------------------- #
#  FastMCP 服务器
# --------------------------------------------------------------------------- #
mcp = FastMCP("crawl-extract")

@mcp.tool()
async def crawl_extract(
    url: str,
    query: str,
    temperature: float = 0.1,
    max_tokens: int = 30000,
) -> str:
    """
    抓取 URL 内容为 Markdown，并仅提取与查询相关的内容。

    参数
    ----------
    url : str
        目标抓取 URL。
    query : str
        信息需求；用于从页面 Markdown 中仅提取最相关的片段。
    temperature : float, optional (default: 0.1)
        提取模型的采样温度。
    max_tokens : int, optional (default: 1400)
        提取模型响应中允许的最大 token 数。

    返回
    -------
    str
        一个紧凑的、四部分的提取结果（直接答案，关键证据，实体/数字，不确定性）。

    """
    return await _crawl_and_extract(
        url=url,
        query=query,
        temperature=temperature,
        max_tokens=max_tokens,
    )

# --------------------------------------------------------------------------- #
#  入口点
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    mcp.run(transport="stdio")

