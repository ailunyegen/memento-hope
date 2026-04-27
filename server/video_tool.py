#!/usr/bin/env python
"""
FastMCP 视频辅助服务器 (异步 DeepSeek 版本)。
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
from pathlib import Path
from typing import List

import ffmpeg
import yt_dlp
from mcp.server.fastmcp import FastMCP
from PIL import Image
import cv2
import numpy as np
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from openai import AsyncOpenAI
from dotenv import load_dotenv

# --- 修改开始 ---
# 明确指定 .env 文件的路径，确保总能从 client 目录加载配置
dotenv_path = Path(__file__).resolve().parent.parent / "client" / ".env"
load_dotenv(dotenv_path=dotenv_path)
# --- 修改结束 ---

# --------------------------------------------------------------------------- #
#  DeepSeek 客户端 (异步)
# --------------------------------------------------------------------------- #

deepseek_client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

# --------------------------------------------------------------------------- #
#  FastMCP 实例
# --------------------------------------------------------------------------- #

mcp = FastMCP("video_tools")

# --------------------------------------------------------------------------- #
#  辅助函数
# --------------------------------------------------------------------------- #


def _capture_screenshot(video_file: str, timestamp: float, width: int = 320) -> Image.Image:
    out, _ = (
        ffmpeg.input(video_file, ss=timestamp)
        .filter("scale", width, -1)
        .output("pipe:", vframes=1, format="image2", vcodec="png")
        .run(capture_stdout=True, capture_stderr=True)
    )
    return Image.open(io.BytesIO(out))


def _extract_audio(video_file: str, output_format: str = "mp3") -> str:
    basename = os.path.splitext(video_file)[0]
    out_path = f"{basename}.{output_format}"
    (
        ffmpeg.input(video_file)
        .output(out_path, vn=None, acodec="libmp3lame")
        .run(quiet=True)
    )
    return out_path


async def _transcribe_audio_async(audio_path: str) -> str:
    """使用 Whisper 模型进行转录；如果未配置 API Key 则返回 ''。"""
    # 注意: DeepSeek 可能没有直接的音频转录 API，这里暂时保留 OpenAI 的调用方式
    # 如果 DeepSeek 支持，需要相应修改
    temp_openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not temp_openai_client.api_key: return ""
    
    rsp = await temp_openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=open(audio_path, "rb"),
    )
    return rsp.text.strip()


def _normalize(img: Image.Image, target_width: int = 512) -> Image.Image:
    w, h = img.size
    return img.resize((target_width, int(target_width * h / w)), Image.Resampling.LANCZOS).convert("RGB")


def _extract_keyframes(
    video_path: str,
    frame_interval: float = 4.0,
    max_frames: int = 100,
    target_width: int = 512,
) -> List[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    duration = total / fps if fps else 0
    cap.release()

    desired = min(max(int(duration / frame_interval) or 1, 1), max_frames)

    video = open_video(video_path)
    sm = SceneManager()
    sm.add_detector(ContentDetector())
    sm.detect_scenes(video)
    scenes = sm.get_scene_list()

    frames: List[Image.Image] = []
    if scenes:
        for i in np.linspace(0, len(scenes) - 1, min(len(scenes), desired), dtype=int):
            frames.append(_capture_screenshot(video_path, scenes[i][0].get_seconds()))

    while len(frames) < desired and duration:
        t = len(frames) * frame_interval
        try:
            frames.append(_capture_screenshot(video_path, t))
        except ffmpeg.Error:
            break

    return [_normalize(f, target_width) for f in frames]


def _images_to_base64(imgs: List[Image.Image]) -> List[str]:
    out: List[str] = []
    for im in imgs:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        out.append(base64.b64encode(buf.getvalue()).decode())
    return out


# --------------------------------------------------------------------------- #
#  工具
# --------------------------------------------------------------------------- #

@mcp.tool()
async def download_video(url: str, download_directory: str | None = None) -> str:
    """
    使用 yt_dlp 从给定的 URL 下载视频，并返回本地文件路径。

    参数:
    - url (str): 要下载的视频的 URL。
    - download_directory (Optional[str]): 视频将保存到的目录的可选路径。如果未指定，将使用临时目录。

    返回:
    - str: 下载的视频文件的完整文件路径。
    """
    download_directory = download_directory or tempfile.mkdtemp()
    Path(download_directory).mkdir(parents=True, exist_ok=True)
    template = str(Path(download_directory) / "%(title)s.%(ext)s")
    opts = {"format": "bestvideo+bestaudio/best", "outtmpl": template}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)


@mcp.tool()
async def get_video_bytes(video_path: str) -> bytes:
    """
    读取并返回位于 *video_path* 的视频文件的原始字节。
    """
    with open(video_path, "rb") as fh:
        return fh.read()


@mcp.tool()
async def get_video_screenshots(video_path: str, amount: int = 3) -> List[str]:
    """
    从视频文件中均匀捕获屏幕截图，并以 base64 编码的 JPEG 字符串形式返回。
    """
    probe = ffmpeg.probe(video_path)
    dur = float(probe["format"]["duration"])
    step = dur / (amount + 1)
    imgs = [_capture_screenshot(video_path, (i + 1) * step) for i in range(amount)]
    return _images_to_base64(imgs)


VIDEO_QA_PROMPT = """
使用关键帧和（可选的）转录来回答问题。

转录（可能为空）:
{transcription}

问题:
{question}
""".strip()


@mcp.tool()
async def ask_question_about_video(
    video_path: str,
    question: str,
    use_audio_transcription: bool = False,
) -> str:
    """
    通过分析关键帧和可选的音频转录，使用多模态 DeepSeek 模型回答有关视频文件内容的特定问题。
    """
    if not deepseek_client.api_key:
        return "DEEPSEEK_API_KEY 未设置。"

    frames = _extract_keyframes(video_path)
    images_b64 = _images_to_base64(frames)
    transcription = ""
    if use_audio_transcription:
        # 注意: DeepSeek 可能没有音频转录 API，这里保留了 Whisper 的逻辑作为备用
        # 如果需要此功能，你仍需一个 OPENAI_API_KEY
        transcription = await _transcribe_audio_async(_extract_audio(video_path))

    user_message = [
        {
            "type": "text",
            "text": VIDEO_QA_PROMPT.format(transcription=transcription, question=question),
        },
        *(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
            for b in images_b64
        ),
    ]

    chat = await deepseek_client.chat.completions.create(
        model="deepseek-vl-chat",
        messages=[{"role": "user", "content": user_message}],
        max_tokens=512,
    )
    content = chat.choices[0].message.content
    return (content or "").strip()


# --------------------------------------------------------------------------- #
#  入口点
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    mcp.run(transport="stdio")

