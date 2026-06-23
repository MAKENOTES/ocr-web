"""
将目录下的 PNG 图片批量转换为 JPG。

背景：训练标注 (train_annotation.txt) 里部分样本写的是 .jpg，但磁盘上实际是
.png，导致加载失败（evaluate_val.py 静默返回空白图 → 空预测）。本脚本把 PNG
转成同名 .jpg，使文件与标注一致。

用法：
    # 转换默认目录 (recognition/data/train_images)，保留原 PNG
    python recognition/utils/png_to_jpg.py

    # 指定目录 + 递归子目录
    python recognition/utils/png_to_jpg.py --dir recognition/data/test_images -r

    # 转换后删除原 PNG
    python recognition/utils/png_to_jpg.py --delete-src

    # 只看会做什么，不实际写文件
    python recognition/utils/png_to_jpg.py --dry-run
"""
import os
import sys
import argparse
import glob

from PIL import Image, ImageFile

# 防止个别截断图报错
ImageFile.LOAD_TRUNCATED_IMAGES = True


def convert_one(png_path: str, quality: int, overwrite: bool, dry_run: bool) -> str:
    """转换单张 PNG -> JPG，返回状态: 'ok' / 'skip' / 'fail'"""
    jpg_path = os.path.splitext(png_path)[0] + ".jpg"

    if os.path.exists(jpg_path) and not overwrite:
        print(f"[跳过] 已存在: {jpg_path}")
        return "skip"

    if dry_run:
        print(f"[试运行] {png_path} -> {jpg_path}")
        return "ok"

    try:
        with Image.open(png_path) as img:
            # JPG 不支持透明通道：带 alpha 的图先合成到白色背景
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                rgba = img.convert("RGBA")
                background = Image.new("RGB", rgba.size, (255, 255, 255))
                background.paste(rgba, mask=rgba.split()[-1])  # 用 alpha 作为遮罩
                rgb = background
            else:
                rgb = img.convert("RGB")

            rgb.save(jpg_path, "JPEG", quality=quality)
        print(f"[转换] {png_path} -> {jpg_path}")
        return "ok"
    except Exception as e:
        print(f"[失败] {png_path}: {e}")
        return "fail"


def main():
    parser = argparse.ArgumentParser(description="批量将 PNG 转换为 JPG")
    parser.add_argument(
        "--dir", default="recognition/data/train_images",
        help="要处理的图片目录 (默认: recognition/data/train_images)"
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true",
        help="递归处理子目录"
    )
    parser.add_argument(
        "-q", "--quality", type=int, default=95,
        help="JPG 保存质量 1-100 (默认: 95)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="同名 .jpg 已存在时覆盖 (默认跳过)"
    )
    parser.add_argument(
        "--delete-src", action="store_true",
        help="转换成功后删除原 PNG 文件"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印将要执行的操作，不实际写/删文件"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"错误: 目录不存在 -> {args.dir}")
        sys.exit(1)

    # 收集 PNG 文件 (大小写后缀都收)
    if args.recursive:
        png_files = glob.glob(os.path.join(args.dir, "**", "*.png"), recursive=True)
        png_files += glob.glob(os.path.join(args.dir, "**", "*.PNG"), recursive=True)
    else:
        png_files = glob.glob(os.path.join(args.dir, "*.png"))
        png_files += glob.glob(os.path.join(args.dir, "*.PNG"))

    png_files = sorted(set(png_files))

    if not png_files:
        print(f"在 {args.dir} 未找到任何 PNG 图片。")
        return

    print(f"发现 {len(png_files)} 张 PNG，开始转换 (quality={args.quality})...")

    ok = skip = fail = deleted = 0
    for png_path in png_files:
        status = convert_one(png_path, args.quality, args.overwrite, args.dry_run)
        if status == "ok":
            ok += 1
            if args.delete_src and not args.dry_run:
                try:
                    os.remove(png_path)
                    deleted += 1
                except OSError as e:
                    print(f"[警告] 删除原图失败 {png_path}: {e}")
        elif status == "skip":
            skip += 1
        else:
            fail += 1

    print("\n" + "=" * 50)
    print("转换完成！")
    print(f"  成功 : {ok}")
    print(f"  跳过 : {skip}")
    print(f"  失败 : {fail}")
    if args.delete_src:
        print(f"  已删除原 PNG: {deleted}")


if __name__ == "__main__":
    main()
