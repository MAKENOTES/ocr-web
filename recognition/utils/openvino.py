"""为 CRNN 构建显式长度感知的 OpenVINO GRU 图。"""


def make_gru_length_safe(model):
    """将双向 GRU 改为两个正向 GRU，显式按有效长度反转反向分支。

    本机 OpenVINO 2026.0 CPU 的 GRUSequence 在 sequence_lengths 小于张量
    时间维时仍遍历 padding，导致反向状态污染。正向 GRU 的有效前缀不受后续
    padding 影响，因此用 batch Gather 只反转有效前缀，最后继续使用导出图的
    blank mask。避免 ReverseSequence 被优化器重新融合为 reverse GRU。
    仅用于当前不消费最终 hidden state、且有效长度大于零的识别模型。
    """
    import numpy as np
    from openvino import opset13 as ops

    for node in list(model.get_ordered_ops()):
        if node.get_type_name() != "GRUSequence":
            continue
        attributes = node.get_attributes()
        if attributes["direction"] != "bidirectional":
            continue
        if node.output(1).get_target_inputs():
            raise ValueError("长度修复仅支持不消费 GRU 最终 hidden state 的 CRNN")
        inputs = [node.input_value(i) for i in range(6)]
        data, hidden, lengths, weights, recurrent, bias = inputs
        length64 = ops.convert(lengths, "i64")
        time = ops.gather(ops.shape_of(data, "i64"), ops.constant(1), ops.constant(0))
        positions = ops.unsqueeze(ops.range(ops.constant(0, np.int64), time,
                                            ops.constant(1, np.int64), "i64"), ops.constant(0))
        valid = ops.unsqueeze(length64, ops.constant(1))
        reverse_indices = ops.select(ops.less(positions, valid),
                                     ops.subtract(ops.subtract(valid, ops.constant(1, np.int64)), positions),
                                     ops.add(positions, ops.multiply(valid, ops.constant(0, np.int64))))
        reverse_data = ops.gather(data, reverse_indices, ops.constant(1), batch_dims=1)

        outputs = []
        for direction, source in enumerate([data, reverse_data]):
            index = ops.constant([direction], np.int64)
            h = ops.gather(hidden, index, ops.constant(1))
            w = ops.gather(weights, index, ops.constant(0))
            r = ops.gather(recurrent, index, ops.constant(0))
            b = ops.gather(bias, index, ops.constant(0))
            gru = ops.gru_sequence(source, h, lengths, w, r, b,
                                   hidden_size=attributes["hidden_size"], direction="forward",
                                   activations=attributes["activations"],
                                   activations_alpha=attributes["activations_alpha"],
                                   activations_beta=attributes["activations_beta"],
                                   clip=attributes["clip"],
                                   linear_before_reset=attributes["linear_before_reset"],
                                   name=f"{node.get_friendly_name()}/length_safe_{direction}")
            output = gru.output(0)
            if direction:
                output = ops.unsqueeze(ops.gather(ops.squeeze(output, ops.constant(1)),
                                                  reverse_indices, ops.constant(1), batch_dims=1),
                                       ops.constant(1)).output(0)
            outputs.append(output)
        replacement = ops.concat(outputs, 1)
        replacement.set_friendly_name(node.get_friendly_name() + "/length_safe_output")
        node.output(0).replace(replacement.output(0))
    model.validate_nodes_and_infer_types()
    return model
