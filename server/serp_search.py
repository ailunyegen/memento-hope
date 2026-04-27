from serpapi import GoogleSearch
from mcp.server.fastmcp import FastMCP
import os
from pathlib import Path
from dotenv import load_dotenv

# --- 修改开始: 确保能从 client/.env 加载配置 ---
try:
    # 构建到 client/.env 文件的路径
    # __file__ 是当前文件的路径, .parent.parent 会上溯两级目录到项目根目录
    dotenv_path = Path(__file__).parent.parent / 'client' / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
    else:
        # 如果找不到，尝试加载当前目录的 .env (作为备用方案)
        load_dotenv()
except Exception as e:
    print(f"Warning: Could not load .env file. Error: {e}")
# --- 修改结束 ---


# --------------------------------------------------------------------------- #
#  FastMCP server instance
# --------------------------------------------------------------------------- #

mcp = FastMCP("serpapi")

# --------------------------------------------------------------------------- #
#  Tools
# --------------------------------------------------------------------------- #

@mcp.tool()
async def google_search(query: str) -> list[dict]:
    """
    Run a Google search via SerpAPI and return the organic results.
    
    Parameters
    ----------
    query : str
        The search query string (e.g., "Coffee")
    
    Returns
    -------
    list[dict]
        The list of organic search results from Google.
    """
    # --- 修改开始: 从环境变量中安全地获取 API Key ---
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return [{"title": "SerpAPI Error", "link": "", "snippet": "SERPAPI_API_KEY not found in .env file."}]
    # --- 修改结束 ---

    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key 
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        return results.get("organic_results", [])
    except Exception as e:
        return [{"title": "SerpAPI Execution Error", "link": "", "snippet": str(e)}]

# --------------------------------------------------------------------------- #
#  Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    mcp.run(transport="stdio")
