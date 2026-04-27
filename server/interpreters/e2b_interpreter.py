import os
from typing import Any, ClassVar, Dict, List, Optional

from .base import BaseInterpreter
from .interpreter_error import InterpreterError
from .logger import get_logger


logger = get_logger(__name__)


class E2BInterpreter(BaseInterpreter):
    r"""E2B Code Interpreter implementation.

    Args:
        require_confirm (bool, optional): If True, prompt user before running
            code strings for security. (default: :obj:`True`)
    """

    _CODE_TYPE_MAPPING: ClassVar[Dict[str, Optional[str]]] = {
        "python": None,
        "py3": None,
        "python3": None,
        "py": None,
        "shell": "bash",
        "bash": "bash",
        "sh": "bash",
        "java": "java",
        "javascript": "js",
        "r": "r",
    }

    
    def __init__(
        self,
        require_confirm: bool = True,
    ) -> None:
        # --- 修改开始: 暂时禁用 E2B 初始化以解决版本兼容性问题 ---
        self.require_confirm = require_confirm
        self._sandbox = None
        logger.warning("E2BInterpreter is temporarily disabled due to potential version conflicts.")
        # try:
        #     from e2b_code_interpreter import Sandbox
        #     self._sandbox = Sandbox(api_key=os.environ.get("E2B_API_KEY"))
        # except ImportError:
        #     self._sandbox = None
        #     logger.error("e2b-code-interpreter is not installed. Please run 'pip install e2b-code-interpreter'")
        # except Exception as e:
        #     self._sandbox = None
        #     logger.error(f"Failed to initialize E2B Sandbox: {e}")
        # --- 修改结束 ---

    def __del__(self) -> None:
        r"""Destructor for the E2BInterpreter class.

        This method ensures that the e2b sandbox is killed when the
        interpreter is deleted.
        """
        if (
            hasattr(self, '_sandbox')
            and self._sandbox is not None
            # and self._sandbox.is_running() # is_running() 可能不存在
        ):
            # self._sandbox.kill() # kill() 可能不存在
            pass

    def run(
        self,
        code: str,
        code_type: str,
    ) -> str:
        r"""Executes the given code in the e2b sandbox.

        Args:
            code (str): The code string to execute.
            code_type (str): The type of code to execute (e.g., 'python',
                'bash').

        Returns:
            str: The string representation of the output of the executed code.

        Raises:
            InterpreterError: If the `code_type` is not supported or if any
                runtime error occurs during the execution of the code.
        """
        # --- 修改开始: 如果沙箱未初始化，则返回错误信息 ---
        if self._sandbox is None:
            return "E2B Sandbox is not available. Please check installation and API key."
        # --- 修改结束 ---

        if code_type not in self._CODE_TYPE_MAPPING:
            raise InterpreterError(
                f"Unsupported code type {code_type}. "
                f"`{self.__class__.__name__}` only supports "
                f"{', '.join(list(self._CODE_TYPE_MAPPING.keys()))}."
            )
        # Print code for security checking
        if self.require_confirm:
            logger.info(
                f"The following {code_type} code will run on your "
                f"e2b sandbox: {code}"
            )
            while True:
                choice = input("Running code? [Y/n]:").lower()
                if choice in ["y", "yes", "ye"]:
                    break
                elif choice not in ["no", "n"]:
                    continue
                raise InterpreterError(
                    "Execution halted: User opted not to run the code. "
                    "This choice stops the current operation and any "
                    "further code execution."
                )

        if self._CODE_TYPE_MAPPING[code_type] is None:
            execution = self._sandbox.run_code(code)
        else:
            execution = self._sandbox.run_code(
                code=code, language=self._CODE_TYPE_MAPPING[code_type]
            )

        if execution.text and execution.text.lower() != "none":
            return execution.text

        if execution.logs:
            if execution.logs.stdout:
                return ",".join(execution.logs.stdout)
            elif execution.logs.stderr:
                return ",".join(execution.logs.stderr)

        return str(execution.error)

    def supported_code_types(self) -> List[str]:
        r"""Provides supported code types by the interpreter."""
        return list(self._CODE_TYPE_MAPPING.keys())

    def update_action_space(self, action_space: Dict[str, Any]) -> None:
        r"""Updates action space for *python* interpreter"""
        raise RuntimeError("E2B doesn't support " "`action_space`.")
