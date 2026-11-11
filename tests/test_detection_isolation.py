"""保证三个检测子系统互不导入，统一检测也不依赖识别子系统。

统一检测 `detection` 只使用自身实现；`detection_small` 与 `detection_larger`
保留历史行为，后者仍可复用识别预处理中的透明图片合成。
"""
import ast
from pathlib import Path

import pytest


# 每个子系统禁止导入的顶层模块；键同时也是检测包目录名。
FORBIDDEN_IMPORTS = {
    "detection": {"detection_small", "detection_larger", "recognition"},
    "detection_small": {"detection", "detection_larger", "recognition"},
    "detection_larger": {"detection", "detection_small"},
}

# 统一检测的根目录入口；小图和大图入口按固定文件名枚举。
ROOT_ENTRIES = {
    "detection": ("detection_export_onnx.py", "detection_inference_onnx.py"),
    "detection_small": ("detection_small_export_onnx.py",
                        "detection_small_inference_640.py",
                        "detection_small_inference_onnx.py",
                        "detection_small_quantize_int8.py"),
    "detection_larger": ("detection_larger_export_onnx.py",
                         "detection_larger_inference_onnx.py"),
}


def source_files(subsystem):
    """返回子系统包内 Python 文件及对应的根目录入口。"""
    root = Path(__file__).resolve().parents[1]
    return [*sorted((root / subsystem).rglob("*.py")),
            *(root / name for name in ROOT_ENTRIES[subsystem])]


@pytest.mark.parametrize("subsystem", sorted(FORBIDDEN_IMPORTS))
def test_detection_subsystems_do_not_import_each_other(subsystem):
    forbidden = FORBIDDEN_IMPORTS[subsystem]
    for path in source_files(subsystem):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                top = module.split(".")[0]
                assert top not in forbidden, f"{path}:{node.lineno} imports {module}"


def test_unified_detection_has_no_legacy_path_references():
    """统一检测的运行时 Python 文件不能引用两个旧检测目录或识别目录。"""
    forbidden = ("detection_small", "detection_larger", "recognition")
    for path in source_files("detection"):
        text = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert name not in text, f"{path} references {name}"


def _resolve_detection_module(root, module):
    """把 detection.* 模块解析为仓库内文件，找不到时直接失败。"""
    target = root.joinpath(*module.split("."))
    if target.with_suffix(".py").is_file():
        return target.with_suffix(".py")
    if (target / "__init__.py").is_file():
        return target / "__init__.py"
    raise AssertionError(f"Unresolved local import: {module}")


def test_train_entry_dependency_closure_is_inside_detection():
    """训练入口递归依赖只能落在 detection/ 内，防止重新引入外部子系统。"""
    root = Path(__file__).resolve().parents[1]
    package = root / "detection"
    visited = set()

    def visit(path):
        path = Path(path)
        if path in visited:
            return
        visited.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                if module == "detection" or module.startswith("detection."):
                    visit(_resolve_detection_module(root, module))

    visit(package / "train.py")
    assert all(package in path.parents for path in visited), sorted(map(str, visited))
    assert package / "components" / "augmentation.py" in visited
