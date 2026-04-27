"""
mcp_image_analysis.py
FastMCP 服务器 – 视觉工具 (图像 → 描述 / VQA)
"""

# --------------------------------------------------------------------------- #
#  导入
# --------------------------------------------------------------------------- #
import base64
import io
import os
from typing import Optional
from pathlib import Path

import anyio
import openai
import requests
from PIL import Image
from urllib.parse import urlparse
from openai import AsyncOpenAI
from mcp.server.fastmcp import FastMCP
from loguru import logger

from dotenv import load_dotenv
# --- 修改开始 ---
dotenv_path = Path(__file__).resolve().parent.parent / "client" / ".env"
load_dotenv(dotenv_path=dotenv_path)
# --- 修改结束 ---

# --------------------------------------------------------------------------- #
#  辅助类
# --------------------------------------------------------------------------- #
class ImageAnalysisToolkit:
    """
    围绕 DeepSeek Vision 模型的简单封装。
    提供两个公开的协程:
        • image_to_text
        • ask_question_about_image
    """

    def __init__(self, timeout: float | None = None):
        self.timeout = timeout or 15

    # ---------------- public API ------------------------------------------ #
    async def image_to_text(
        self, image_path: str, sys_prompt: Optional[str] = None
    ) -> str:
        """
        返回 *image_path* 的详细描述。
        """
        default_sys = (
            "你是一位专业的图像分析师。请对可见的所有内容，包括任何文本，提供一个丰富而简洁的描述。"
        )
        return await self._chat_with_image(
            image_path,
            user_prompt="请描述这张图片的内容。",
            system_prompt=sys_prompt or default_sys,
        )

    async def ask_question_about_image(
        self,
        image_path: str,
        question: str,
        sys_prompt: Optional[str] = None,
    ) -> str:
        """
        回答关于 *image_path* 的 *question*。
        """
        default_sys = (
            "你通过仔细的视觉检查、阅读任何文本以及根据所见进行推理来回答关于图片的问题。请仔细考虑问题的要求。"
        )
        return await self._chat_with_image(
            image_path,
            user_prompt=question,
            system_prompt=sys_prompt or default_sys,
        )

    # ---------------- 实现 -------------------------------------- #
    async def _chat_with_image(
        self, image_path: str, user_prompt: str, system_prompt: str
    ) -> str:
        """
        核心流程：准备图像，运行 DeepSeek vision chat，返回内容。
        """
        image_url = await self._prepare_image(image_path)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]
        deepseek_client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL"),
        )

        try:
            logger.info("正在发送图片至 DeepSeek ChatCompletion (vision)…")
            response = await deepseek_client.chat.completions.create(
                model="deepseek-vl-chat",
                messages=messages,
            )
            content = response.choices[0].message.content
            return (content or "").strip()
        except Exception as e:
            logger.error(f"DeepSeek 调用失败: {e}")
            raise

    async def _prepare_image(self, path: str) -> str:
        """
        将 *path* (本地路径或 URL) 转换为 DeepSeek Vision 端点可接受的 URL 或 data-URL。
        """
        parsed = urlparse(path)

        # 远程 URL – 直接返回
        if parsed.scheme in ("http", "https"):
            logger.debug(f"使用远程图片 URL: {path}")
            return path

        # 本地文件 – 读取并编码
        logger.debug(f"正在编码本地图片: {path}")
        data = await anyio.to_thread.run_sync(lambda: open(path, "rb").read()) # type: ignore
        mime = Image.open(io.BytesIO(data)).get_format_mimetype()
        b64 = base64.b64encode(data).decode()
        return f"data:{mime};base64,{b64}"


# --------------------------------------------------------------------------- #
#  FastMCP 服务器
# --------------------------------------------------------------------------- #
mcp = FastMCP("image_analysis")
toolkit = ImageAnalysisToolkit()


@mcp.tool()
async def image_to_text(image_path: str, sys_prompt: Optional[str] = None) -> str:
    """
    为位于 *image_path* 的图片生成一个详细的描述性标题。

    参数:
    - image_path (str): 要分析的图片的本地文件路径或 URL。
    - sys_prompt (Optional[str]): 一个可选的系统提示，可以指导或影响图片标题的行为，允许自定义描述风格、详细程度或焦点。

    返回:
    - str: 图片中内容、对象、场景和相关特征的详细自然语言描述。
    """
    return await toolkit.image_to_text(image_path, sys_prompt)


@mcp.tool()
async def ask_question_about_image(
    image_path: str, question: str, sys_prompt: Optional[str] = None
) -> str:
    """
    回答与位于 *image_path* 的图片内容相关的特定问题。

    参数:
    - image_path (str): 要分析的图片的本地文件路径或 URL。
    - question (str): 关于图片内容需要回答的问题。
    - sys_prompt (Optional[str]): 一个可选的系统提示，用于指导推理或回答风格，为图像分析提供上下文或期望的行为。

    返回:
    - str: 基于对图片内容的视觉分析和理解的问题答案。
    """
    return await toolkit.ask_question_about_image(image_path, question, sys_prompt)

# --------------------------------------------------------------------------- #
#  入口点
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    mcp.run(transport="stdio")

