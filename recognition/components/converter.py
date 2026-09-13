"""
用于CTC Loss的标签-文本转换器。
"""
import torch

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

    def decode(self, preds: torch.Tensor, *, layout: str) -> list[str]:
        """使用显式 layout 对三维 CTC 输出执行 greedy decode。"""
        if preds.ndim != 3:
            raise ValueError(f"preds 必须是三维张量，当前 shape={tuple(preds.shape)}")
        if preds.shape[2] != self.num_classes:
            raise ValueError(
                f"类别维不匹配: output={preds.shape[2]}, keys={self.num_classes}"
            )
        if layout == "TBC":
            preds = preds.transpose(0, 1)
        elif layout != "BTC":
            raise ValueError(f"layout 仅支持 'TBC' 或 'BTC'，当前为 {layout!r}")

        # 取最大概率的索引
        tokens = preds.argmax(dim=2) # (Batch, Seq)

        decoded_texts = []
        for i in range(tokens.size(0)):
            token_seq = tokens[i]

            char_list = []
            # CTC 解码逻辑: 移除重复，移除 blank
            for j in range(token_seq.size(0)):
                token = token_seq[j].item()
                # 如果不是 blank (0) 且 (是第一个字符 或 与前一个字符不同)
                if token != 0 and (j == 0 or token != token_seq[j-1].item()):
                    char_list.append(self.idx2char.get(token, ''))

            decoded_texts.append("".join(char_list))

        return decoded_texts
