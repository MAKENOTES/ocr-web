"""验证识别解码的置信度门槛：低置信度帧按 blank 处理，其余行为与 CTC 一致。"""

import math

import pytest
import torch

from recognition.components.converter import DECODE_MIN_CONFIDENCE, LabelConverter

# 30 个字符 → 32 类。类别数足够多时，某帧的最大概率可以低于 0.1 却仍是 argmax，
# 这样才能覆盖“低置信度但非 blank”的帧。
KEYS_TEXT = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥天地玄黄宇宙洪荒"
TOKEN_A = 2  # 甲
TOKEN_B = 3  # 乙


def _converter(tmp_path) -> LabelConverter:
    keys_path = tmp_path / "keys.txt"
    keys_path.write_text("".join(f"{char}\n" for char in KEYS_TEXT), encoding="utf-8")
    return LabelConverter(str(keys_path))


def test_micro_sign_output_is_normalized_to_greek_mu(tmp_path):
    keys_path = tmp_path / "keys.txt"
    keys_path.write_text("甲\n\u00b5\n\u03bc\n", encoding="utf-8")
    converter = LabelConverter(str(keys_path))
    micro_idx = converter.char2idx["\u00b5"]
    mu_idx = converter.char2idx["\u03bc"]

    for token in (micro_idx, mu_idx):
        preds = _log_probs([(token, 0.99)], converter.num_classes)
        assert converter.decode(preds, layout="TBC") == ["\u03bc"]


def _log_probs(frames: list[tuple[int, float]], num_classes: int) -> torch.Tensor:
    """把 (token, probability) 序列拼成 [T, 1, C] 张量，剩余概率均分给其它类。

    概率归一化后取 log 模拟模型输出；decode 内部再 softmax 可还原同一分布。
    当 token 概率 > 1/类别数 时该 token 就是该帧 argmax。
    """
    rows = torch.zeros(len(frames), num_classes)
    for index, (token, probability) in enumerate(frames):
        rows[index, :] = (1.0 - probability) / (num_classes - 1)
        rows[index, token] = probability
    return rows.log().unsqueeze(1)


def test_low_confidence_leading_frame_is_dropped_by_default(tmp_path):
    converter = _converter(tmp_path)
    # 帧 0：伪字符 甲（0.05，仍是 argmax）；帧 1：blank；帧 2：真字符 乙（0.97）
    preds = _log_probs(
        [(TOKEN_A, 0.05), (0, 0.97), (TOKEN_B, 0.97)], converter.num_classes
    )

    assert DECODE_MIN_CONFIDENCE == 0.1
    assert converter.decode(preds, layout="TBC") == ["乙"]
    assert converter.decode(preds, layout="TBC", min_confidence=0.0) == ["甲乙"]


def test_blank_frames_decode_to_empty_text(tmp_path):
    converter = _converter(tmp_path)
    preds = _log_probs([(0, 0.95), (0, 0.99)], converter.num_classes)

    assert converter.decode(preds, layout="TBC") == [""]


def test_repeated_char_needs_blank_between_frames(tmp_path):
    converter = _converter(tmp_path)
    adjacent = _log_probs(
        [(TOKEN_A, 0.99), (TOKEN_A, 0.99)], converter.num_classes
    )
    separated = _log_probs(
        [(TOKEN_A, 0.99), (0, 0.99), (TOKEN_A, 0.99)], converter.num_classes
    )

    assert converter.decode(adjacent, layout="TBC") == ["甲"]
    assert converter.decode(separated, layout="TBC") == ["甲甲"]


def test_rejected_frame_breaks_repeated_char_like_blank(tmp_path):
    converter = _converter(tmp_path)
    # 中间帧是 甲 但置信度只有 0.05：默认按 blank 处理，两侧各输出一次；
    # 纯 argmax 会把三帧当连续重复，只输出一个字。
    preds = _log_probs(
        [(TOKEN_A, 0.99), (TOKEN_A, 0.05), (TOKEN_A, 0.99)], converter.num_classes
    )

    assert converter.decode(preds, layout="TBC") == ["甲甲"]
    assert converter.decode(preds, layout="TBC", min_confidence=0.0) == ["甲"]


@pytest.mark.parametrize("threshold", [-0.1, 1.1, math.nan])
def test_invalid_threshold_is_rejected(tmp_path, threshold):
    converter = _converter(tmp_path)
    preds = _log_probs([(TOKEN_A, 0.99)], converter.num_classes)

    with pytest.raises(ValueError, match="min_confidence"):
        converter.decode(preds, layout="TBC", min_confidence=threshold)
