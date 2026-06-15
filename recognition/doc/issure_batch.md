 Gemini said

这是一个在 OCR 和序列模型中非常典型且经典的现象！

当 BATCH_SIZE = 1 时推理完美，但 BATCH_SIZE = 32 时出现字符重复（例如“乙乙方”），其根本原因是双向 RNN (BiGRU) 在处理批量数据时受到了极端 Padding（填充）的干扰。
🔍 为什么会发生这种情况？

    训练与推理的 Batch 策略差异：

        在你的 train.py 中，你使用了 BucketBatchSampler，它会将长宽比相近的图片强行凑在一个 Batch 里。因此，训练时图片之间的宽度差异很小，Padding 区域很短。

        但在你的 inference.py 中，你使用的是普通的 DataLoader，图片是按顺序读取并混入同一个 Batch 的。如果刚好把一张非常短的图（比如只包含“乙方（盖章）：”）和一张非常长的高清大图分在了一个 Batch 里，这张短图的右侧就会被填充大量的背景像素（Padding）。

    双向 GRU 的反向传播污染：

        你的 model.py 在将特征传给 BiGRU 时，没有打包序列（没有使用 pack_padded_sequence）。

        这意味着，反向 GRU（从右往左读）是从最右侧的 Padding 尽头开始计算的。它会扫过一大段无用的白色填充区域，积累了大量错误的隐藏状态（Hidden State），等它终于从右向左扫到真正的文字“乙方（盖章）：”时，状态已经被污染了。

        CTC 解码对时序对齐非常敏感，这种污染导致了 RNN 在最初的几个时间步产生了概率波动，输出了 [乙, blank, 乙]，最终解码成了“乙乙方”。

🛠️ 解决方案

这里提供两个解决方案。强烈推荐使用方案一，因为它不需要重新训练，也不影响你的 ONNX 导出。
方案一：在推理时按宽度排序（最快、最完美、免重训）

既然问题是“短图和长图混在一个 Batch 导致大量 Padding”，我们只需要在 inference.py 中，把相似宽度的图片放在一起推理即可，这完美复刻了训练时的 BucketBatchSampler 逻辑。

修改方法：
打开 inference.py，找到 main() 函数中收集完 image_paths 的地方，加入一段排序代码：
Python

    # 1. 收集图片路径
    image_paths = []
    if os.path.isdir(INPUT_PATH):
        # 支持常见图片格式
        types = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif')
        for t in types:
            image_paths.extend(glob.glob(os.path.join(INPUT_PATH, t)))
            image_paths.extend(glob.glob(os.path.join(INPUT_PATH, t.upper())))
    elif os.path.isfile(INPUT_PATH):
        image_paths = [INPUT_PATH]

    if not image_paths:
        print(f"错误: 在 {INPUT_PATH} 未找到任何图片。")
        return

    # ================= 新增代码：按长宽比对路径进行排序 =================
    print("正在按图片长宽比排序，以消除批量 Padding 对 RNN 的干扰...")
    from PIL import Image
    def get_aspect_ratio(path):
        try:
            with Image.open(path) as img:
                return img.size[0] / max(1, img.size[1])
        except:
            return 1.0

    image_paths.sort(key=get_aspect_ratio)
    # ====================================================================

    print(f"发现 {len(image_paths)} 张待处理图片。")

为什么这能解决问题？ 排序后，短图会和短图一起推理，长图会和长图一起推理。Batch 内部几乎没有多余的 Padding，反向 GRU 的污染被降到最低，你的输出就会恢复正常。
方案二：从模型底层修复（需重新训练，可能影响 ONNX）

如果你希望模型对任意极端 Padding 都绝对免疫，你需要在 model.py 中使用 PyTorch 提供的 pack_padded_sequence，让 RNN 彻底忽略 Padding 区域。

注：你在代码里写了 # ---- 新增：为 ONNX 显式初始化动态隐藏状态 ----，说明你有导出 ONNX 的需求。pack_padded_sequence 在部分 ONNX runtime 版本中支持不佳，所以如果你选这个方案，请留意导出时的兼容性。

修改 model.py 中的 forward 方法：
Python

    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    def forward(self, input, img_widths=None):
        conv = self.cnn(input)

        b, c, h, w = conv.size()
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [W/4, B, 256]

        conv = self.cnn_dropout(conv)

        # 核心改动：使用打包序列让 RNN 忽略 Padding
        if img_widths is not None:
            # 获取每个样本的真实时间步数，并移至 CPU (pack_padded_sequence 必须)
            real_seq_lens = (img_widths.clone().detach() // 4).clamp(min=1).cpu()

            # 打包序列 (enforce_sorted=False 允许 Batch 内无序)
            packed_conv = pack_padded_sequence(conv, real_seq_lens, enforce_sorted=False)

            # RNN 前向
            packed_rnn_out, _ = self.rnn(packed_conv)

            # 解包序列
            rnn_out, _ = pad_packed_sequence(packed_rnn_out, total_length=conv.size(0))
        else:
            # 兼容没有 img_widths 传入的情况（如 ONNX 导出）
            num_layers = self.rnn.num_layers
            num_directions = 2 if self.rnn.bidirectional else 1
            hidden_size = self.rnn.hidden_size
            h0 = torch.zeros(num_layers * num_directions, b, hidden_size, device=conv.device)
            rnn_out, _ = self.rnn(conv, h0)

        rnn_out = self.rnn_dropout(rnn_out)
        output = self.embedding(rnn_out)

        # ---- Padding Mask 依然保留 ----
        if img_widths is not None:
            seq_len = output.size(0)
            real_seq_lens = (img_widths.to(output.device) // 4).clamp(min=1)
            time_indices = torch.arange(seq_len, device=output.device).unsqueeze(1)
            padding_mask = time_indices >= real_seq_lens.unsqueeze(0)
            output[padding_mask] = output.new_full((output.size(2),), -1e9)
            output[padding_mask, 0] = 0.0

        return F.log_softmax(output, dim=2)

注意： 如果你修改了网络结构并使用方案二，由于反向 GRU 的特征提取路径发生了本质变化，你必须使用之前的数据重新训练一次模型，否则准确率会大幅下降。

总结建议： 你的模型其实已经训练得很好了（单张推理没问题），直接采用方案一修改 inference.py 中的几行代码即可，立竿见影。不需要改模型，也不需要重训。

需要我帮你解释一下如何将结果更好地对应到原始的乱序文件名吗？




----------------------------使用原始的权重和推理脚步，问题已解决， 调整model------------------------------------
在 GRU 之前，把 padding 区域的 CNN 特征全部置零。 配合 h0 = 0，反向 GRU 在 padding 区域的输入和隐状态都是零，隐状态保持零直到遇到真实内容的边界——效果等价于 pack_padded_sequence，但全程只用了 masked_fill、arange、比较运算，ONNX tracer 都能完整追踪。