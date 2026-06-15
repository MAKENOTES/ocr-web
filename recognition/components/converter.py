"""
用于CTC Loss的标签-文本转换器。
"""
import torch

class LabelConverter:
    """
    在文本标签和用于CTC Loss的整数序列之间进行转换。
    """
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
        # 3. 【新增】调试打印，确认空格是否被读进来了
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
            # 仅在调试时取消注释，防止刷屏
            print(f"警告: 本 Batch 发现 {unknown_warning_count} 个不在字典中的字符，已转换为 [UNK] {unknown_warning_char}")
            pass

        return torch.tensor(encoded_texts, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)

    def decode(self, preds: torch.Tensor) -> list[str]:
        # preds: (SeqLen, Batch, NumClasses) -> LogSoftmax后的结果 或者 Raw Logits
        # 注意：这里输入通常是 (Seq, Batch, C) 或者 (Batch, Seq, C)，需要确认 train.py 的输出

        # 假设输入是 (SeqLen, Batch, Class) - 这是 PyTorch CTCLoss 的标准输入，但通常解码时我们已经转成了 (Batch, SeqLen, Class)
        # 为了通用性，先检查维度
        if preds.shape[1] != self.num_classes:
             # 可能是 (Seq, Batch, C)
             preds = preds.transpose(0, 1) # -> (Batch, Seq, C)

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