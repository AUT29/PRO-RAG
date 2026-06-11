#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将目录中的 JSON 文件分成三等份，每份复制到新文件夹
"""

import os
import shutil


def get_json_files_from_directory(directory: str):
    """从指定目录获取所有 JSON 文件列表（按文件名排序）"""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"目录不存在：{directory}")

    all_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    all_files.sort()
    return [os.path.join(directory, filename) for filename in all_files]


def split_into_parts(files: list, parts: int) -> list:
    """将文件列表均分为 parts 份（余数从前面的份依次 +1）"""
    total = len(files)
    base_size = total // parts
    remainder = total % parts

    splits = []
    start = 0

    for i in range(parts):
        size = base_size + (1 if i < remainder else 0)
        end = start + size
        splits.append(files[start:end])
        start = end

    return splits


def main():
    # 配置
    SOURCE_DIR = 'PCORAG/output/2wiki_pnet'
    BASE_OUTPUT_DIR = '.'
    NUM_SPLITS = 3

    print("=" * 80)
    print(f"文件分片工具 - 复制文件到 {NUM_SPLITS} 个新文件夹")
    print("=" * 80)

    # 获取所有文件
    print(f"\n📂 源目录：{SOURCE_DIR}")
    try:
        all_files = get_json_files_from_directory(SOURCE_DIR)
    except FileNotFoundError as e:
        print(f"\n❌ 错误：{e}")
        return

    total = len(all_files)
    print(f"📊 找到 {total} 个 JSON 文件")

    if total == 0:
        print("⚠️ 没有找到任何 JSON 文件！")
        return

    # 分成三等份
    splits = split_into_parts(all_files, NUM_SPLITS)

    # 创建输出目录
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # 复制到三个文件夹
    print(f"\n{'=' * 80}")
    print("正在复制文件...")
    print(f"{'=' * 80}")

    for i, split in enumerate(splits, 1):
        output_dir = os.path.join(BASE_OUTPUT_DIR, f'zzz_2wiki_split_{i}')
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n第 {i} 份 → {output_dir} ({len(split)} 个文件)")

        for file_path in split:
            filename = os.path.basename(file_path)
            dest_path = os.path.join(output_dir, filename)
            shutil.copy2(file_path, dest_path)

        print(f"  ✓ 已复制 {len(split)} 个文件")

    print(f"\n{'=' * 80}")
    print("完成！")
    print(f"{'=' * 80}")
    print(f"\n生成的目录结构：")
    for i in range(1, NUM_SPLITS + 1):
        split_dir = os.path.join(BASE_OUTPUT_DIR, f'zzz_2wiki_split_{i}')
        count = len(splits[i - 1])
        print(f"  {split_dir}/ ({count} 个文件)")


if __name__ == "__main__":
    main()
