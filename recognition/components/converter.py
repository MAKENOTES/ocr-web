"""
用于CTC Loss的标签-文本转换器。
"""
import torch

# CTC 贪心解码的逐帧最低置信度（概率，0~1）。低于该值的帧按 blank 处理。
# 依据：稀疏小图或空白框的首尾帧会以约 0.03~0.08 的概率吐出伪字符
# （如纯白图输出“蚌幕帖”），按 argmax 解码会保留它们；0.1 在 val 上修复
# 23 条首帧多字、train 修复 73 条，未误伤任何原本正确的样本。
# 设为 0 恢复纯 argmax 解码。
DECODE_MIN_CONFIDENCE = 0.1


class LabelConverter:
    """按字典原顺序编码字符并执行 CTC 贪心解码。
    空字典行被跳过，普通空格保留；调用前需用数据工具检查重复字及多字符行。"""
    def __init__(self, keys_path: str):
        self.keys = []
        with open(keys_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 去掉换行符，但保留行内的空格
                char = line.strip('\n').strip('\r')
                # 过滤掉真正的空行（防止文件末尾多余的空行被读入）
                # 注意：如果你的 keys.txt 里空格是写成 " \n" (有一个空格字符)，len 也是 1，不会被过滤，是安全的。
                if len(char) == 0:
                    continue
                self.keys.append(char)

        # 0: [blank], 1: [UNK] (未知字符), 2...: 正常字符
        self.char2idx = {'[blank]': 0, '[UNK]': 1}
        self.idx2char = {0: '[blank]', 1: '[UNK]'}

        for i, char in enumerate(self.keys):
            idx = i + 2  # 从2开始，留出 blank 和 UNK
            self.char2idx[char] = idx
            self.idx2char[idx] = char

        self.num_classes = len(self.char2idx)
        # 空格占用独立类别，启动时提示以便核对原字典。
        if ' ' in self.char2idx:
            print(f"提示: 检测到 '空格' 字符在字典中，ID为 {self.char2idx[' ']}")
        else:
            print("注意: 字典中未检测到 '空格' 字符。如果你的图片包含空格，请检查 keys.txt")

        print(f"Converter 初始化完成: 字典大小 {len(self.keys)}, 总类别数 {self.num_classes} (含blank/unk)")

    def encode(self, text: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_texts = []
        lengths = []

        unknown_warning_count = 0
        unknown_warning_char = []

        for s in text:
            length = 0
            encoded_s = []
            for char in s:
                if char in self.char2idx:
                    encoded_s.append(self.char2idx[char])
                else:
                    # 遇到未知字符，映射为 UNK，而不是丢弃！
                    encoded_s.append(self.char2idx['[UNK]'])
                    unknown_warning_count += 1
                    unknown_warning_char.append(char)
                length += 1

            encoded_texts.extend(encoded_s)
            lengths.append(length)

        if unknown_warning_count > 0:
            # 每批出现字典外字符时打印告警；编码保留为 UNK。
            print(f"警告: 本 Batch 发现 {unknown_warning_count} 个不在字典中的字符，已转换为 [UNK] {unknown_warning_char}")
            pass

        return torch.tensor(encoded_texts, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)

    def decode(
        self,
        preds: torch.Tensor,
        *,
        layout: str,
        min_confidence: float = DECODE_MIN_CONFIDENCE,
    ) -> list[str]:
        """使用显式 layout 对三维 CTC 输出执行带置信度门槛的 greedy decode。

        preds 是 log_softmax、softmax 或原始 logits 均可：函数内部统一做一次
        softmax 取概率。每帧最大概率低于 min_confidence 时按 blank 处理，再按
        CTC 规则合并连续重复并移除 blank。该规则用于丢弃边界帧的低置信度伪字符，
        并让空白框返回空串；min_confidence=0 等价于纯 argmax 解码。
        """
        if preds.ndim != 3:
            raise ValueError(f"preds 必须是三维张量，当前 shape={tuple(preds.shape)}")
        if preds.shape[2] != self.num_classes:
            raise ValueError(
                f"类别维不匹配: output={preds.shape[2]}, keys={self.num_classes}"
            )
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(
                f"min_confidence 必须在 [0, 1] 内，当前为 {min_confidence}"
            )
        if layout == "TBC":
            preds = preds.transpose(0, 1)
        elif layout != "BTC":
            raise ValueError(f"layout 仅支持 'TBC' 或 'BTC'，当前为 {layout!r}")

        # 同时取最大概率索引和该帧置信度；softmax 对 log-prob 输入也是幂等的。
        probabilities, tokens = preds.softmax(dim=2).max(dim=2)

        decoded_texts = []
        for index in range(tokens.size(0)):
            char_list = []
            previous_token = 0
            # CTC 解码：低于门槛的帧视为 blank，blank 会打断重复字符。
            for token, probability in zip(
                tokens[index].tolist(), probabilities[index].tolist()
            ):
                if token == 0 or probability < min_confidence:
                    previous_token = 0
                    continue
                if token != previous_token:
                    char_list.append(self.idx2char.get(token, ''))
                previous_token = token

            decoded_texts.append("".join(char_list))

        return decoded_texts
