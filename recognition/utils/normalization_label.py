import os
import re


def replace_english_comma_in_labels(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"❌ 找不到文件: {input_file}")
        return

    processed_lines = 0
    modified_lines = 0

    with open(input_file, 'r', encoding='utf-8') as fin, \
            open(output_file, 'w', encoding='utf-8') as fout:

        for line in fin:
            original_line = line.strip('\n\r')
            if not original_line:
                fout.write('\n')
                continue

            # 定义一个替换函数，只处理双引号内部的匹配项
            def replacer(match):
                inner_text = match.group(1)
                # 如果内部包含英文逗号，就替换成中文逗号
                if ',' in inner_text:
                    nonlocal modified_lines
                    modified_lines += 1
                    inner_text = inner_text.replace(',', '，')
                return f'"{inner_text}"'

            # 用正则精准匹配双引号内的内容，并调用 replacer 替换
            # 这样前半部分的 `图片路径, ` (带空格) 会被完美保留，一字不差
            new_line = re.sub(r'"(.*?)"', replacer, original_line)

            fout.write(f"{new_line}\n")
            processed_lines += 1

    print("=" * 40)
    print(f"✅ 处理完成！")
    print(f"总计处理行数: {processed_lines}")
    print(f"包含英文逗号并成功替换的行数: {modified_lines}")
    print(f"结果已保存至: {output_file}")


if __name__ == "__main__":
    # 请根据实际情况修改路径
    INPUT_TXT = "recognition/data/val_annotation.txt"
    OUTPUT_TXT = "recognition/data/val_annotation_fixed.txt"

    replace_english_comma_in_labels(INPUT_TXT, OUTPUT_TXT)