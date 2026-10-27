"""构建可独立部署的识别运行目录。

用法（在仓库根目录执行）::

    .venv/bin/python tools/build_recognition_bundle.py \
        --model best_recognition_layer.onnx \
        --keys recognition/keys.txt \
        --out dist/recognition_runtime

产物包含运行所需的最少代码、模型、字典、依赖清单和 manifest.json；默认会在
隔离临时目录中实际推理一次，确认 bundle 不依赖仓库路径。
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

# 仓库根目录；白名单拷贝、git commit 和默认字典路径都从这里推导。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recognition.components.converter import LabelConverter  # noqa: E402

# 产物布局版本；目录结构或 manifest 字段不兼容时递增，部署方据此判断是否需要更新接入代码。
BUNDLE_FORMAT_VERSION = "1"
# 运行代码白名单，按项目根目录的相对路径拷贝；不包含数据、权重、日志和合成工具。
PACKAGE_FILES = (
    "recognition/__init__.py",
    "recognition/runtime/__init__.py",
    "recognition/runtime/__main__.py",
    "recognition/runtime/backends.py",
    "recognition/runtime/cli.py",
    "recognition/runtime/engine.py",
    "recognition/runtime/preprocess.py",
    "recognition/components/__init__.py",
    "recognition/components/augmentation.py",
    "recognition/components/converter.py",
    "recognition/components/dataset.py",
    "recognition/components/preprocessing.py",
    "recognition/models/__init__.py",
    "recognition/models/model.py",
    "recognition/utils/__init__.py",
    "recognition/utils/annotation.py",
    "recognition/utils/checkpoint.py",
    "recognition/utils/openvino.py",
)
# 所有后端都需要的运行依赖；版本为下限，部署方可按目标系统调整。
# torch/torchvision 暂时保留在基础依赖中：runtime 仍复用训练侧的 PyTorch 预处理与
# DataLoader，ONNX/OpenVINO 部署也必须在导入时提供它们；后续瘦身时再移出。
BASE_REQUIREMENTS = (
    "numpy>=2.0",
    "pillow>=10.0",
    "opencv-python-headless>=4.8",
    "torch>=2.2",
    "torchvision>=0.17",
)
# 各后端额外需要的推理引擎；键名与 --backend 的取值一致。
# pytorch 后端的依赖已在基础清单中，这里保留空元组以生成完整的后端清单文件。
BACKEND_REQUIREMENTS = {
    "onnx": ("onnxruntime>=1.17",),
    "openvino": ("openvino>=2024.0",),
    "pytorch": (),
}
# 产物中允许出现的顶层条目；用于构建后复查，避免把数据目录带进部署包。
# 按后端生成的 requirements-<backend>.txt 由 REQUIRED_BACKEND_PREFIX 单独匹配。
ALLOWED_TOP_LEVEL = {
    "recognition", "models", "keys.txt", "requirements.txt", "example.py",
    "README.md", "manifest.json",
}
# 按后端依赖文件的前缀；与 BACKEND_REQUIREMENTS 的文件名规则保持一致。
BACKEND_REQUIREMENTS_PREFIX = "requirements-"

# 产物内的最小调用示例；模型路径相对于产物目录，不依赖构建机路径。
EXAMPLE_SCRIPT = r'''"""部署目录最小调用示例：python example.py <图片或目录>"""

import sys
from pathlib import Path

from recognition.runtime import RecognitionEngine, collect_image_paths

ROOT = Path(__file__).resolve().parent
MODEL_SUFFIXES = (".onnx", ".xml", ".pth")


def find_model() -> Path:
    """返回 models 目录中的第一个模型；多个模型时请按业务选择。"""
    models = [
        path
        for suffix in MODEL_SUFFIXES
        for path in sorted((ROOT / "models").glob(f"*{suffix}"))
    ]
    if not models:
        raise FileNotFoundError(f"models 目录中没有可用模型: {ROOT / 'models'}")
    return models[0]


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("用法: python example.py <图片路径或目录>")
    engine = RecognitionEngine(
        model_path=find_model(),
        keys_path=ROOT / "keys.txt",
        backend="auto",
    )
    for result in engine.recognize(collect_image_paths(sys.argv[1]), batch_size=1):
        print(f"{Path(result.path).name}\t{result.text}")


if __name__ == "__main__":
    main()
'''

# 产物内 README 的固定内容；说明依赖分档、调用方式和哈希校验。
BUNDLE_README = """# Recognition 部署产物

本目录由 `tools/build_recognition_bundle.py` 生成，请勿手改；升级模型或代码时重新构建。

## 安装依赖

`requirements.txt` 是基础依赖，再按后端选择一份：

```bash
pip install -r requirements-onnx.txt       # ONNX Runtime（CPU 部署推荐）
pip install -r requirements-openvino.txt   # OpenVINO
pip install -r requirements-pytorch.txt    # PyTorch checkpoint
```

当前 runtime 仍复用训练侧的 PyTorch 预处理与 DataLoader，基础清单已包含
torch/torchvision，因此 ONNX/OpenVINO 部署也需要它们。它们不参与推理，只负责
张量承载与数据加载；后续版本会移除。

## 调用

```bash
python example.py /path/to/image_or_dir
python -m recognition.runtime --model_path models/model.onnx --keys_path keys.txt --image_path /path/to/images
```

```python
from recognition.runtime import RecognitionEngine, collect_image_paths

engine = RecognitionEngine("models/model.onnx", "keys.txt", backend="auto", num_threads=1)
for result in engine.recognize(collect_image_paths("/path/to/images"), batch_size=8):
    print(result.path, result.text)
```

## 校验

`manifest.json` 记录模型和字典的 SHA256、ONNX 导出元数据与构建时间。ONNX 后端启动时
还会按模型内的 `keys_sha256` 校验字典，避免类别顺序不一致导致静默错字。
"""

# 隔离自检脚本：在临时工作目录、只把产物加入 PYTHONPATH 时完成一次真实推理。
VERIFY_SCRIPT = r'''"""在隔离环境验证 bundle 可导入并完成一次推理。"""

import sys
from pathlib import Path

import recognition
from recognition.runtime import RecognitionEngine

root = Path(sys.argv[1]).resolve()
package_dir = Path(recognition.__file__).resolve().parent
if package_dir != root / "recognition":
    raise SystemExit(f"导入了错误位置的 recognition: {package_dir}")

models = [
    path
    for suffix in (".onnx", ".xml", ".pth")
    for path in sorted((root / "models").glob(f"*{suffix}"))
]
if not models:
    raise SystemExit("bundle 中没有模型文件")

engine = RecognitionEngine(models[0], root / "keys.txt", backend="auto")
report = engine.recognize([sys.argv[2]], batch_size=1)
if len(report) != 1:
    raise SystemExit(f"推理结果数量异常: {len(report)}")
print(report.results[0].text)
'''


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
    """读取 ONNX 导出时写入的元数据；其他格式返回空字典。"""
    if model_path.suffix.lower() != ".onnx":
        return {}
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    return {prop.key: prop.value for prop in model.metadata_props}


def _write_requirements(output_dir: Path) -> list[str]:
    """写入基础依赖和按后端划分的三份依赖清单，返回生成的文件名。"""
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


def _prepare_output(output_dir: Path, force: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"输出路径不是目录: {output_dir}")
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"输出目录已存在: {output_dir}；确认覆盖后加 --force"
            )
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


def _copy_model(model_path: Path, output_dir: Path) -> str:
    """拷贝模型及同名外部数据文件，返回模型在产物内的相对路径。"""
    models_dir = output_dir / "models"
    models_dir.mkdir()
    destination = models_dir / model_path.name
    shutil.copy2(model_path, destination)

    # ONNX 大模型可能把权重放在 <模型>.data；OpenVINO IR 的权重是替换扩展名的
    # 同名 .bin（model.xml + model.bin）。两者都必须跟随模型进入产物。
    companions = [Path(f"{model_path}.data")]
    if model_path.suffix.lower() == ".xml":
        companions.append(model_path.with_suffix(".bin"))
    for companion in companions:
        if companion.is_file():
            shutil.copy2(companion, models_dir / companion.name)
    return str(destination.relative_to(output_dir))


def build_bundle(model_path, keys_path, output_dir, *, force: bool = False,
                 verify: bool = True, create_zip: bool = False,
                 python_executable: str = sys.executable) -> dict:
    """构建部署目录并返回 manifest；自检失败会抛异常并保留产物供排查。"""
    model_path = Path(model_path).resolve()
    keys_path = Path(keys_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not keys_path.is_file():
        raise FileNotFoundError(f"字典文件不存在: {keys_path}")

    _prepare_output(output_dir, force)
    _copy_code(output_dir)
    model_relative = _copy_model(model_path, output_dir)
    shutil.copy2(keys_path, output_dir / "keys.txt")
    (output_dir / "example.py").write_text(EXAMPLE_SCRIPT, encoding="utf-8")
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
            "num_layers": metadata.get("num_layers"),
            "num_channels": metadata.get("num_channels"),
            "keys_sha256": metadata.get("keys_sha256"),
            "output_type": metadata.get("output_type"),
        },
        "keys": {
            "file": "keys.txt",
            "sha256": _sha256(keys_path),
            "num_classes": LabelConverter(str(keys_path)).num_classes,
        },
        "requirements": requirement_files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if verify:
        verify_bundle(output_dir, python_executable)
    if create_zip:
        shutil.make_archive(
            str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name
        )
    return manifest


def verify_bundle(bundle_dir, python_executable: str = sys.executable) -> str:
    """在临时工作目录、仅用 bundle 作为 PYTHONPATH 实际推理一次。"""
    from PIL import Image, ImageDraw

    bundle_dir = Path(bundle_dir).resolve()
    stray = sorted(
        path.name
        for path in bundle_dir.iterdir()
        if path.name not in ALLOWED_TOP_LEVEL
        and not (
            path.name.startswith(BACKEND_REQUIREMENTS_PREFIX)
            and path.name.endswith(".txt")
        )
    )
    if stray:
        raise RuntimeError(f"产物包含非白名单条目: {stray}")

    with tempfile.TemporaryDirectory(prefix="recognition_bundle_check_") as work:
        work_dir = Path(work)
        image_path = work_dir / "check.png"
        image = Image.new("RGB", (240, 32), (250, 248, 242))
        ImageDraw.Draw(image).text((8, 8), "Check 123", fill=(20, 20, 20))
        image.save(image_path)

        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(bundle_dir)
        environment.pop("PYTHONHOME", None)
        completed = subprocess.run(
            [python_executable, "-c", VERIFY_SCRIPT, str(bundle_dir), str(image_path)],
            cwd=work_dir, env=environment, capture_output=True, text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            "bundle 自检失败：\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.stdout.strip()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="构建可独立部署的识别运行目录")
    parser.add_argument("--model", required=True, help="识别模型：.onnx、.xml 或 .pth")
    parser.add_argument("--keys", required=True, help="与模型配套的 keys.txt")
    parser.add_argument("--out", default="dist/recognition_runtime",
                        help="输出目录；默认 dist/recognition_runtime")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的输出目录")
    parser.add_argument("--zip", action="store_true", help="同时生成同名 zip")
    parser.add_argument("--skip-verify", action="store_true",
                        help="跳过隔离环境自检；仅在当前环境缺少运行依赖时使用")
    args = parser.parse_args(argv)

    try:
        manifest = build_bundle(
            model_path=args.model,
            keys_path=args.keys,
            output_dir=args.out,
            force=args.force,
            verify=not args.skip_verify,
            create_zip=args.zip,
        )
    except (FileNotFoundError, FileExistsError) as error:
        raise SystemExit(str(error)) from error
    print("\n构建完成")
    print(f"  输出目录: {Path(args.out).resolve()}")
    print(f"  模型:     {manifest['model']['file']} "
          f"(sha256 {manifest['model']['sha256'][:12]}…)")
    print(f"  字典:     keys.txt "
          f"({manifest['keys']['num_classes']} 类, sha256 {manifest['keys']['sha256'][:12]}…)")
    if not args.skip_verify:
        print("  自检:     通过")


if __name__ == "__main__":
    main()
