import re


def extract_chars_from_annotation(input_file):
    """从标注文件中提取所有唯一字符（双引号中的文本）。"""
    all_chars = set()

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 匹配双引号中的内容
            match = re.search(r'"(.*?)"', line)
            if match:
                text = match.group(1)
                for char in text:
                    if char != '\n' and char != '\r':
                        all_chars.add(char)

    return all_chars


def load_keys(keys_file):
    """读取 keys.txt，每行一个字符。"""
    keys = set()

    with open(keys_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 去掉行尾换行，但保留字符本身（包括空格）
            char = line.rstrip('\n').rstrip('\r')
            if char != '':
                keys.add(char)

    return keys


def compare_keys(annotation_files, keys_file="recognition/keys.txt",
                 output_file="keys-compare.txt"):
    try:
        # 1. 从所有标注文件中收集字符
        annotation_chars = set()
        for f in annotation_files:
            chars = extract_chars_from_annotation(f)
            print(f"从 {f} 中提取 {len(chars)} 个唯一字符。")
            annotation_chars |= chars

        # 2. 读取现有字典
        keys = load_keys(keys_file)
        print(f"keys 文件 {keys_file} 中共 {len(keys)} 个字符。")

        # 3. 找出标注中存在、但字典里没有的字符（缺失字符）
        missing_in_keys = annotation_chars - keys

        # 4. 排序并写入
        sorted_chars = sorted(missing_in_keys)

        with open(output_file, 'w', encoding='utf-8') as f_out:
            for char in sorted_chars:
                f_out.write(f"{char}\n")

        print(f"\n标注中共 {len(annotation_chars)} 个唯一字符。")
        print(f"标注里有、但 keys 里没有的字符：{len(sorted_chars)} 个。")
        print(f"差异已保存至: {output_file}")

        # 额外提示：keys 里有、但标注中未出现的字符
        unused_in_keys = keys - annotation_chars
        print(f"（参考）keys 里有、但标注中未出现的字符：{len(unused_in_keys)} 个。")

    except FileNotFoundError as e:
        print(f"错误：找不到文件 '{e.filename}'，请检查文件名。")


if __name__ == "__main__":

    compare_keys(
        annotation_files=[
            "recognition/data/train_annotation.txt",
            "recognition/data/val_annotation.txt",
        ],
        keys_file="recognition/keys.txt",
        output_file="recognition/keys-compare.txt",
    )
