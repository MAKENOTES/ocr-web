import csv

import pytest

from recognition.utils.character_stats import DEFAULT_ANNOTATIONS, build_statistics, main


def test_default_annotations_produce_train_val_synthesis_columns(tmp_path, monkeypatch):
    keys = tmp_path / "keys.txt"
    keys.write_text("甲\n零\n", encoding="utf-8")
    annotations = [tmp_path / path.name for path in DEFAULT_ANNOTATIONS]
    assert [path.name for path in annotations] == [
        "train_annotation.txt", "val_annotation.txt", "synthesis_annotation.txt",
    ]
    for index, path in enumerate(annotations, 1):
        path.write_text(f"a.png\t{'甲' * index}\n", encoding="utf-8")
    monkeypatch.setattr("recognition.utils.character_stats.DEFAULT_ANNOTATIONS", annotations)
    output = tmp_path / "stats"
    assert main(["--keys", str(keys), "--output-dir", str(output)]) == 0
    with (output / "character_stats.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "字符", "train_占比", "val_占比", "synthesis_占比", "合计_占比",
            "train_次数", "val_次数", "synthesis_次数", "合计_次数",
        ]
        rows = {row["字符"]: row for row in reader}
    assert rows["甲"]["synthesis_次数"] == "3"
    assert rows["甲"]["合计_次数"] == "6"
    assert rows["零"]["synthesis_占比"] == "0.00000%"


def test_counts_use_all_text_characters_and_keep_only_dictionary_rows(tmp_path):
    keys = tmp_path / "keys.txt"
    keys.write_text("乙\n甲\n \n零\n", encoding="utf-8")
    train = tmp_path / "train_annotation.txt"
    val = tmp_path / "val_annotation.txt"
    train.write_text('a.png\t 甲甲 乙𠮷 \nb.png\t甲\nempty.png\t \n', encoding="utf-8")
    val.write_text("c.png\t乙𠮷\n", encoding="utf-8")

    rows, summary = build_statistics(keys, [train, val])
    assert [row["character"] for row in rows] == ["甲", "乙", " ", "零"]
    by_char = {row["character"]: row for row in rows}
    assert by_char["甲"]["train_count"] == 3
    assert by_char["甲"]["train_ratio"] == 3 / 6
    assert by_char["甲"]["all_count"] == 3
    assert by_char["甲"]["all_ratio"] == 3 / 8
    assert by_char["乙"]["val_ratio"] == 1 / 2
    assert by_char["零"]["all_count"] == 0
    assert summary["datasets"]["all"]["sample_count"] == 3
    assert summary["datasets"]["all"]["character_count"] == 8
    assert summary["datasets"]["train"]["skipped_empty_labels"] == 1
    assert summary["datasets"]["train"]["key_coverage_ratio"] == 3 / 4
    assert summary["datasets"]["all"]["unknown_character_count"] == 2


def test_cli_preserves_punctuation_and_counts_repeated_records(tmp_path):
    keys = tmp_path / "keys.txt"
    keys.write_text('甲\n，\n"\n', encoding="utf-8")
    annotations = tmp_path / "sample.txt"
    annotations.write_text('a.png\t"甲，"\na.png\t"甲，"\n', encoding="utf-8")
    output = tmp_path / "result"
    assert main(["--keys", str(keys), "--annotations", str(annotations), "--output-dir", str(output)]) == 0
    assert sorted(path.name for path in output.iterdir()) == ["character_stats.csv"]
    with (output / "character_stats.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = {row["字符"]: row for row in csv.DictReader(handle)}
    assert rows['"']["合计_次数"] == "4"
    assert rows['"']["合计_占比"] == "50.00000%"


def test_empty_dataset_has_finite_zero_ratios(tmp_path):
    keys = tmp_path / "keys.txt"
    keys.write_text("甲\n", encoding="utf-8")
    annotations = tmp_path / "empty.txt"
    annotations.write_text("a.png\t\n", encoding="utf-8")
    rows, summary = build_statistics(keys, [annotations])
    assert rows[0]["all_ratio"] == 0
    assert rows[0]["empty_ratio"] == 0
    assert summary["datasets"]["all"]["missing_key_count"] == 1


@pytest.mark.parametrize("content", ["甲\n甲\n", "甲乙\n", "\n"])
def test_invalid_dictionary_is_rejected(tmp_path, content):
    keys = tmp_path / "keys.txt"
    keys.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=str(keys)):
        build_statistics(keys, [])


def test_malformed_annotation_fails_before_creating_outputs(tmp_path, capsys):
    keys = tmp_path / "keys.txt"
    keys.write_text("甲\n", encoding="utf-8")
    annotation = tmp_path / "bad.txt"
    annotation.write_text('a.png, "甲"\n', encoding="utf-8")
    output = tmp_path / "result"
    assert main(["--keys", str(keys), "--annotations", str(annotation), "--output-dir", str(output)]) == 1
    assert f"{annotation}:1" in capsys.readouterr().err
    assert not output.exists()


def test_duplicate_dataset_names_are_rejected(tmp_path):
    keys = tmp_path / "keys.txt"
    keys.write_text("甲\n", encoding="utf-8")
    annotation = tmp_path / "train_annotation.txt"
    annotation.write_text("a.png\t甲\n", encoding="utf-8")
    with pytest.raises(ValueError, match="数据集名称重复"):
        build_statistics(keys, [annotation, annotation])


def test_five_dataset_csv_uses_sorted_percentages_before_counts(tmp_path, capsys):
    keys = tmp_path / "keys.txt"
    keys.write_text("王\n李\n零\n", encoding="utf-8")
    # 各数据集大小不同，合计占比必须按总字符数计算，不能平均各文件占比。
    specifications = [
        ("train", 80, 1000),
        ("val", 60, 3000),
        ("bench2", 100, 5000),
        ("append", 160, 10000),
        ("bench1", 200, 61000),
    ]
    annotations = []
    for name, count, total in specifications:
        path = tmp_path / f"{name}_annotation.txt"
        path.write_text(f"a.png\t{'王' * count}{'李' * (total - count)}\n", encoding="utf-8")
        annotations.append(str(path))
    output = tmp_path / "result"
    assert main([
        "--keys", str(keys), "--annotations", *annotations,
        "--output-dir", str(output), "--chars", "王",
    ]) == 0
    with (output / "character_stats.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "字符", "train_占比", "val_占比", "bench2_占比", "append_占比",
            "bench1_占比", "合计_占比", "train_次数", "val_次数",
            "bench2_次数", "append_次数", "bench1_次数", "合计_次数",
        ]
        rows = {row["字符"]: row for row in reader}
    assert len(rows) == 3  # 终端查询不会过滤 CSV 中其他 keys 字符。
    assert list(rows) == ["李", "王", "零"]
    assert rows["王"]["train_次数"] == "80"
    assert rows["王"]["train_占比"] == "8.00000%"
    assert rows["王"]["val_次数"] == "60"
    assert rows["王"]["val_占比"] == "2.00000%"
    assert rows["王"]["合计_次数"] == "600"
    assert rows["王"]["合计_占比"] == "0.75000%"
    assert rows["零"]["合计_占比"] == "0.00000%"
    assert not (output / "summary.json").exists()
    stdout = capsys.readouterr().out
    assert "王\t8.00000%\t2.00000%" in stdout
    assert "李\t" not in stdout  # --chars 仅限制终端，CSV 仍包含全部字。


def test_default_run_prints_every_key_including_zero_count(tmp_path, capsys):
    keys = tmp_path / "keys.txt"
    keys.write_text("王\n李\n零\n", encoding="utf-8")
    annotation = tmp_path / "train_annotation.txt"
    annotation.write_text("a.png\t王王李\n", encoding="utf-8")
    output = tmp_path / "result"
    assert main([
        "--keys", str(keys), "--annotations", str(annotation),
        "--output-dir", str(output),
    ]) == 0
    stdout = capsys.readouterr().out
    assert "王\t66.66667%\t66.66667%\t2\t2" in stdout
    assert "李\t33.33333%\t33.33333%\t1\t1" in stdout
    assert "零\t0.00000%\t0.00000%\t0\t0" in stdout
    assert "CSV 写入 3 个字符" in stdout


def test_equal_total_percentages_keep_dictionary_order(tmp_path):
    keys = tmp_path / "keys.txt"
    keys.write_text("甲\n乙\n丙\n零\n", encoding="utf-8")
    annotation = tmp_path / "train_annotation.txt"
    annotation.write_text("a.png\t丙乙甲乙甲丙丙\n", encoding="utf-8")
    rows, _ = build_statistics(keys, [annotation])
    assert [row["character"] for row in rows] == ["丙", "甲", "乙", "零"]
    assert [row["all_count"] for row in rows] == [3, 2, 2, 0]
    assert keys.read_text(encoding="utf-8") == "甲\n乙\n丙\n零\n"
