"""从仓库源文件生成 recognition/runtime 的自包含部署快照。

runtime 中只有 engine、backends、cli、loader 等入口是手写的；预处理、推理数据集、
模型、解码和检查点工具都由本脚本从原路径复制，并把项目内绝对导入改写为相对导入。
修改源文件后运行：

    .venv/bin/python tools/sync_recognition_runtime.py

只检查是否同步：

    .venv/bin/python tools/sync_recognition_runtime.py --check
"""

import argparse
import ast
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 生成目标目录；部署时整体拷贝该目录即可。
RUNTIME_DIR = PROJECT_ROOT / "recognition" / "runtime"

# 源文件 → runtime 文件名。新增推理依赖时在这里登记，并补全 RELATIVE_IMPORTS。
GENERATED_FILES = {
    "recognition/components/preprocessing.py": "preprocessing.py",
    "recognition/components/inference_data.py": "dataset.py",
    "recognition/components/converter.py": "converter.py",
    "recognition/models/model.py": "model.py",
    "recognition/utils/checkpoint.py": "checkpoint.py",
    "recognition/utils/openvino.py": "openvino.py",
}

# 仓库模块 → runtime 相对模块；生成时按此表改写 import。
RELATIVE_IMPORTS = {
    "recognition.components.preprocessing": ".preprocessing",
    "recognition.components.inference_data": ".dataset",
    "recognition.components.converter": ".converter",
    "recognition.models.model": ".model",
    "recognition.utils.checkpoint": ".checkpoint",
    "recognition.utils.openvino": ".openvino",
}

# runtime 中手写、不由同步脚本覆盖的文件；检查多余生成文件时用作白名单。
HANDWRITTEN_FILES = {
    "__init__.py",
    "__main__.py",
    "backends.py",
    "cli.py",
    "engine.py",
    "loader.py",
}

# 插入在模块 docstring 之后的生成说明，避免覆盖 __doc__。
GENERATED_NOTICE = (
    "# 本文件由 tools/sync_recognition_runtime.py 从 {source} 生成。\n"
    "# 请勿手改；修改源文件后重新运行同步脚本。\n"
)

# 匹配 from recognition.xxx import ...；未登记的模块直接报错，避免生成物偷偷依赖仓库。
_PROJECT_IMPORT = re.compile(
    r"^(?P<indent>[ \t]*)from[ \t]+(?P<module>recognition(?:\.\w+)*)[ \t]+import\b",
    re.MULTILINE,
)
# 匹配 import recognition.xxx；生成物不允许这种形式，需改成 from ... import。
_PROJECT_BARE_IMPORT = re.compile(
    r"^(?P<indent>[ \t]*)import[ \t]+(?P<module>recognition(?:\.\w+)*)\b",
    re.MULTILINE,
)


def _rewrite_imports(text: str, source_name: str) -> str:
    """把项目内绝对导入改写为 runtime 相对导入。"""
    bare_match = _PROJECT_BARE_IMPORT.search(text)
    if bare_match:
        raise ValueError(
            f"{source_name}: 不支持 'import {bare_match.group('module')}'，请改为 from ... import ..."
        )

    def replace(match: re.Match) -> str:
        module = match.group("module")
        target = RELATIVE_IMPORTS.get(module)
        if target is None:
            raise ValueError(f"{source_name}: 未登记的导入 {module}，请补充 RELATIVE_IMPORTS")
        return f"{match.group('indent')}from {target} import"

    return _PROJECT_IMPORT.sub(replace, text)


def _insert_notice(text: str, source_name: str) -> str:
    """在模块 docstring 之后插入生成说明，保留 __doc__。"""
    notice = GENERATED_NOTICE.format(source=source_name)
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        lines.insert(tree.body[0].end_lineno, "\n" + notice)
    else:
        lines.insert(0, notice + "\n")
    return "".join(lines)


def _assert_self_contained(text: str, source_name: str) -> None:
    """生成后再次用 AST 确认没有残留的项目内导入。"""
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0 and (
                module == "recognition" or module.startswith("recognition.")
            ):
                raise ValueError(f"{source_name}: 生成后仍包含项目导入 {module}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "recognition" or alias.name.startswith("recognition."):
                    raise ValueError(f"{source_name}: 生成后仍包含项目导入 {alias.name}")


def render_generated_files() -> dict[str, str]:
    """返回 {runtime 文件名: 生成内容}，不写磁盘。"""
    rendered = {}
    for source_relative, target_name in GENERATED_FILES.items():
        source_path = PROJECT_ROOT / source_relative
        if not source_path.is_file():
            raise FileNotFoundError(f"生成源文件不存在: {source_path}")
        text = source_path.read_text(encoding="utf-8")
        text = _rewrite_imports(text, source_relative)
        text = _insert_notice(text, source_relative)
        _assert_self_contained(text, source_relative)
        rendered[target_name] = text
    return rendered


def sync_runtime(runtime_dir: Path = RUNTIME_DIR) -> list[str]:
    """写入生成文件，返回发生变化的文件名。"""
    runtime_dir = Path(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    changed = []
    for name, content in render_generated_files().items():
        path = runtime_dir / name
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8")
            changed.append(name)
    return changed


def check_runtime(runtime_dir: Path = RUNTIME_DIR) -> list[str]:
    """返回不同步项；空列表表示 runtime 与源文件一致。"""
    runtime_dir = Path(runtime_dir)
    expected = render_generated_files()
    problems = []
    for name, content in expected.items():
        path = runtime_dir / name
        if not path.is_file():
            problems.append(f"缺少生成文件: {path}")
        elif path.read_text(encoding="utf-8") != content:
            problems.append(f"与源文件不同步: {path}")
    for path in sorted(runtime_dir.glob("*.py")):
        if path.name in expected or path.name in HANDWRITTEN_FILES:
            continue
        if "tools/sync_recognition_runtime.py" in path.read_text(encoding="utf-8"):
            problems.append(f"多余的生成文件: {path}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="同步 recognition/runtime 部署快照")
    parser.add_argument("--check", action="store_true", help="只检查是否同步，不写文件")
    args = parser.parse_args(argv)

    if args.check:
        problems = check_runtime()
        if problems:
            print("runtime 与源文件不同步：")
            for problem in problems:
                print(f"  - {problem}")
            print("请运行: .venv/bin/python tools/sync_recognition_runtime.py")
            return 1
        print("runtime 与源文件同步")
        return 0

    changed = sync_runtime()
    if changed:
        print("已更新:", ", ".join(changed))
    else:
        print("runtime 已是最新")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
