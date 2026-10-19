"""提示词加载：从同目录的 ``prompts/*.md`` 读取提示词正文。

约定：一个提示词对应一个 Markdown 文件，修改提示词只需编辑 Markdown 文件，
无需改动代码；模块级只加载一次，改动 Markdown 后需重新导入本模块（或重启
进程）才会生效。文件缺失时在导入期即抛出 :class:`FileNotFoundError`，避免
运行期才暴露问题。
"""

from pathlib import Path

#: 提示词目录名（与本模块同级）
PROMPTS_DIRNAME = "prompts"


def load_prompt(name: str) -> str:
    """读取 ``prompts/<name>.md`` 的提示词正文，并去除首尾空白。

    Args:
        name: 提示词文件名（不含 ``.md`` 扩展名），如 ``intention_understanding``。

    Returns:
        提示词正文文本。

    Raises:
        FileNotFoundError: 提示词文件不存在。
    """
    prompt_path = Path(__file__).parent / PROMPTS_DIRNAME / f"{name}.md"
    return prompt_path.read_text(encoding="utf-8").strip()


INTENTION_UNDERSTANDING_PROMPT = load_prompt("intention_understanding")
QUESTION_REWRITING_PROMPT = load_prompt("question_rewriting")
HYPOTHESIS_GENERATION_PROMPT = load_prompt("hypothesis_generation")
ANSWER_GENERATION_PROMPT = load_prompt("answer_generation")
