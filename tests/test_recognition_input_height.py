"""输入高度是模型、预处理和部署入口共同遵守的契约，必须只有一套取值。"""

from recognition import evaluate_robustness, evaluate_val, inference, training_data
from recognition.components.inference_data import CNN_RIGHT_CONTEXT
from recognition.components.preprocessing import MIN_IMAGE_WIDTH
from recognition.models.model import IMG_HEIGHT
from recognition.runtime.backends import IMG_HEIGHT as RUNTIME_IMG_HEIGHT
from recognition.runtime.model import IMG_HEIGHT as RUNTIME_MODEL_IMG_HEIGHT


def test_all_entry_points_use_the_model_input_height():
    assert IMG_HEIGHT == 48
    assert {
        training_data.IMG_HEIGHT,
        evaluate_val.IMG_HEIGHT,
        inference.IMG_HEIGHT,
        evaluate_robustness.IMG_HEIGHT,
        RUNTIME_IMG_HEIGHT,
        RUNTIME_MODEL_IMG_HEIGHT,
    } == {IMG_HEIGHT}


def test_narrow_canvas_and_guard_scale_with_input_height():
    # 窄图画布按 1.5 倍高度留白；右侧保护区要覆盖最后一帧所需的一半感受野。
    assert MIN_IMAGE_WIDTH == IMG_HEIGHT * 3 // 2
    assert CNN_RIGHT_CONTEXT >= 35
