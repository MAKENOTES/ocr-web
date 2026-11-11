"""构建可独立部署的小图检测运行目录。

用法（在仓库根目录执行）::

    .venv/bin/python tools/build_detection_small_bundle.py \
        --model best_detection_small.onnx \
        --out dist/detection_small_runtime

产物包含 detection_small/runtime 部署快照、根目录 ONNX 推理入口、.onnx 模型、
依赖清单和 manifest.json；默认在隔离临时目录中实际推理一次，确认不依赖仓库路径。
构建前会校验 runtime 与源文件同步；修改源文件后先运行
tools/sync_detection_small_runtime.py。
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection_small.utils.config import ONNX_CONFIG_KEY  # noqa: E402
from tools.sync_detection_small_runtime import check_runtime  # noqa: E402


BUNDLE_FORMAT_VERSION = "1"  # 产物布局版本；目录结构或 manifest 字段不兼容时递增。
ENTRY_FILE = "detection_small_inference_onnx.py"  # 根目录 ONNX 推理入口，产物内保持同名。
PACKAGE_FILES = (  # 运行代码白名单；runtime 自包含图片读取、预处理、后处理和后端。
    "detection_small/__init__.py",
    "detection_small/runtime/__init__.py",
    "detection_small/runtime/__main__.py",
    "detection_small/runtime/backends.py",
    "detection_small/runtime/cli.py",
    "detection_small/runtime/config.py",
    "detection_small/runtime/engine.py",
    "detection_small/runtime/image.py",
    "detection_small/runtime/postprocess.py",
    "detection_small/runtime/preprocessing.py",
)
BASE_REQUIREMENTS = (  # 所有后端都需要的运行依赖；版本为下限。
    "numpy>=2.0",
    "pillow>=10.0",
    "opencv-python-headless>=4.8",
    "torch>=2.2",
    "torchvision>=0.17",
)
BACKEND_REQUIREMENTS = {  # 各后端额外需要的推理引擎；键名与 --backend 的取值一致。
    "onnx": ("onnxruntime>=1.17",),
    "openvino": ("openvino>=2024.0", "onnx>=1.15"),
}
ALLOWED_TOP_LEVEL = {  # 产物中允许出现的顶层条目，避免把数据目录带进部署包。
    "detection_small", ENTRY_FILE, "models", "requirements.txt", "README.md", "manifest.json",
}
BACKEND_REQUIREMENTS_PREFIX = "requirements-"  # 按后端依赖文件的前缀。

BUNDLE_README = """# 小图检测部署产物

本目录由 `tools/build_detection_small_bundle.py` 生成，请勿手改；升级模型或代码时重新构建。

## 安装依赖

`requirements.txt` 是基础依赖，再按后端选择一份：

```bash
pip install -r requirements-onnx.txt       # ONNX Runtime（CPU 部署推荐）
pip install -r requirements-openvino.txt   # OpenVINO
```

## 调用

```bash
python detection_small_inference_onnx.py \
  --model_path models/model.onnx \
  --image_path /path/to/image_or_dir \
  --output_dir outputs \
  --backend onnx
```

也可以直接使用 runtime 包：

```python
from detection_small.runtime import collect_images, load_backend, exported_shape, predict_image

backend = load_backend("models/model.onnx", "onnx")
dynamic, max_input_size, fixed_input_size = exported_shape(backend)
predictions, infer_ms = predict_image(
    backend, image, 0.7, 0.3, dynamic, max_input_size, fixed_input_size)
```

`detection_small/runtime` 内部全部使用相对导入，不引用仓库训练代码；需要嵌入其他项目时
可以整体拷贝并重命名包。

## 校验

`manifest.json` 记录模型路径、SHA256、ONNX 导出元数据、git commit 和构建时间。
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def _model_metadata(model_path: Path) -> dict:
    """读取 ONNX 导出时写入的元数据；非 ONNX 模型返回空字典。"""
    if model_path.suffix.lower() != ".onnx":
        return {}
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    return {prop.key: prop.value for prop in model.metadata_props}


def _write_requirements(output_dir: Path) -> list[str]:
    """写入基础依赖和按后端划分的依赖清单，返回生成的文件名。"""
    (output_dir / "requirements.txt").write_text(
        "\n".join(BASE_REQUIREMENTS) + "\n", encoding="utf-8"
    )
    written = ["requirements.txt"]
    for backend, requirements in BACKEND_REQUIREMENTS.items():
        filename = f"requirements-{backend}.txt"
        content = "-r requirements.txt\n" + "\n".join(requirements) + "\n"
        (output_dir / filename).write_text(content, encoding="utf-8")
        written.append(filename)
    return written


def _remove_bytecode(output_dir: Path) -> None:
    """删除自检导入产生的 __pycache__，避免产物携带机器相关的 .pyc。"""
    for cache_dir in output_dir.rglob("__pycache__"):
        shutil.rmtree(cache_dir)


def _prepare_output(output_dir: Path, force: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"输出路径不是目录: {output_dir}")
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"输出目录已存在: {output_dir}；确认覆盖后加 --force")
        # 只覆盖上一次构建产物或空目录，避免 --out 指向源码/数据目录时误删。
        if any(output_dir.iterdir()):
            if not (output_dir / "manifest.json").is_file():
                raise FileExistsError(
                    f"拒绝覆盖非构建产物目录: {output_dir}；请删除或换一个 --out"
                )
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _copy_code(output_dir: Path) -> None:
    for relative_path in PACKAGE_FILES:
        source = PROJECT_ROOT / relative_path
        if not source.is_file():
            raise FileNotFoundError(f"运行代码清单中的文件不存在: {source}")
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    shutil.copy2(PROJECT_ROOT / ENTRY_FILE, output_dir / ENTRY_FILE)


def _copy_model(model_path: Path, output_dir: Path) -> str:
    """拷贝模型及同名外部数据文件，返回模型在产物内的相对路径。"""
    models_dir = output_dir / "models"
    models_dir.mkdir()
    destination = models_dir / model_path.name
    shutil.copy2(model_path, destination)

    companions = [Path(f"{model_path}.data")]
    if model_path.suffix.lower() == ".xml":
        companions.append(model_path.with_suffix(".bin"))
    for companion in companions:
        if companion.is_file():
            shutil.copy2(companion, models_dir / companion.name)
    return str(destination.relative_to(output_dir))


def _verify_bundle(bundle_dir: Path, model_path: Path,
                   python_executable: str = sys.executable) -> str:
    """在临时工作目录、仅用 bundle 作为 PYTHONPATH 实际推理一次。"""
    from PIL import Image

    stray = sorted(
        path.name for path in bundle_dir.iterdir()
        if path.name not in ALLOWED_TOP_LEVEL
        and not (path.name.startswith(BACKEND_REQUIREMENTS_PREFIX)
                 and path.name.endswith(".txt"))
    )
    if stray:
        raise RuntimeError(f"产物包含非白名单条目: {stray}")

    with tempfile.TemporaryDirectory(prefix="detection_small_bundle_check_") as work:
        work_dir = Path(work)
        image_path = work_dir / "check.png"
        Image.new("RGB", (240, 96), (250, 248, 242)).save(image_path)
        output_dir = work_dir / "output"

        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(bundle_dir)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment.pop("PYTHONHOME", None)
        completed = subprocess.run(
            [python_executable, str(bundle_dir / ENTRY_FILE),
             "--model_path", str(model_path),
             "--image_path", str(image_path),
             "--output_dir", str(output_dir),
             "--backend", "onnx", "--warmup_runs", "0"],
            cwd=work_dir, env=environment, capture_output=True, text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "bundle 自检失败：\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        if not list((output_dir / "labels").glob("gt_check.txt")):
            raise RuntimeError(f"bundle 自检未生成标签: {output_dir / 'labels'}")
    return completed.stdout.strip()


def build_bundle(model_path, output_dir, *, force: bool = False,
                 verify: bool = True, create_zip: bool = False,
                 python_executable: str = sys.executable) -> dict:
    """构建部署目录并返回 manifest；自检失败会抛异常并保留产物供排查。"""
    runtime_problems = check_runtime()
    if runtime_problems:
        raise RuntimeError(
            "runtime 与源文件不同步，请先运行 tools/sync_detection_small_runtime.py：\n"
            + "\n".join(runtime_problems)
        )

    model_path = Path(model_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if model_path.suffix.lower() != ".onnx":
        raise ValueError(f"小图检测部署包只接受 .onnx 模型: {model_path}")

    _prepare_output(output_dir, force)
    _copy_code(output_dir)
    model_relative = _copy_model(model_path, output_dir)
    (output_dir / "README.md").write_text(BUNDLE_README, encoding="utf-8")
    requirement_files = _write_requirements(output_dir)

    metadata = _model_metadata(model_path)
    manifest = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "model": {
            "file": model_relative,
            "sha256": _sha256(model_path),
            "config": metadata.get(ONNX_CONFIG_KEY),
        },
        "requirements": requirement_files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if verify:
        _verify_bundle(output_dir, output_dir / model_relative, python_executable)
    _remove_bytecode(output_dir)
    if create_zip:
        shutil.make_archive(
            str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name
        )
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="构建可独立部署的小图检测运行目录")
    parser.add_argument("--model", required=True, help="小图检测 .onnx 模型")
    parser.add_argument("--out", default="dist/detection_small_runtime",
                        help="输出目录；默认 dist/detection_small_runtime")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的输出目录")
    parser.add_argument("--zip", action="store_true", help="同时生成同名 zip")
    parser.add_argument("--skip-verify", action="store_true",
                        help="跳过隔离环境自检；仅在当前环境缺少运行依赖时使用")
    args = parser.parse_args(argv)

    try:
        manifest = build_bundle(
            model_path=args.model,
            output_dir=args.out,
            force=args.force,
            verify=not args.skip_verify,
            create_zip=args.zip,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print("\n构建完成")
    print(f"  输出目录: {Path(args.out).resolve()}")
    print(f"  模型:     {manifest['model']['file']} "
          f"(sha256 {manifest['model']['sha256'][:12]}…)")
    if not args.skip_verify:
        print("  自检:     通过")


if __name__ == "__main__":
    main()
