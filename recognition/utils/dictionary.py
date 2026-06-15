import re


def extract_chars_to_dict(input_file, output_file="keys.txt"):
    all_chars = set()

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                # 匹配双引号中的内容
                match = re.search(r'"(.*?)"', line)
                if match:
                    text = match.group(1)
                    # 收集字符
                    for char in text:
                        # 如果不需要保留空格，可以改为 if char.strip():
                        if char != '\n' and char != '\r':
                            all_chars.add(char)

        # 4. 排序并写入
        sorted_chars = sorted(list(all_chars))

        with open(output_file, 'w', encoding='utf-8') as f_out:
            for char in sorted_chars:
                f_out.write(f"{char}\n")

        print(f"提取完成！共计 {len(sorted_chars)} 个唯一字符。")
        print(f"字典已保存至: {output_file}")

    except FileNotFoundError:
        print(f"错误：找不到文件 '{input_file}'，请检查文件名。")


if __name__ == "__main__":

    extract_chars_to_dict(
        "recognition/data/train_annotation.txt",
        "recognition/keys.txt")