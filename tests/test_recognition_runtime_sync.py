"""验证 runtime 部署快照与仓库源文件一致，防止生成副本漂移。"""

from tools.sync_recognition_runtime import (
    GENERATED_FILES,
    RUNTIME_DIR,
    check_runtime,
    render_generated_files,
)


def test_generated_runtime_snapshot_is_in_sync():
    problems = check_runtime(RUNTIME_DIR)
    assert not problems, (
        "runtime 与源文件不同步，请运行 tools/sync_recognition_runtime.py：\n"
        + "\n".join(problems)
    )


def test_generated_files_are_marked_and_written():
    rendered = render_generated_files()

    assert set(rendered) == set(GENERATED_FILES.values())
    for name, content in rendered.items():
        assert "tools/sync_recognition_runtime.py" in content, name
        assert (RUNTIME_DIR / name).is_file(), name
