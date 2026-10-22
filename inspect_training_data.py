#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抽样查看 MiniMind 的 pretrain、SFT 和 DPO JSONL 数据。"""

import argparse
import gzip
import json
import sys
from pathlib import Path

DEFAULT_FILES = {
    "pretrain": "pretrain_t2t_mini.jsonl",
    "sft": "sft_t2t_mini.jsonl",
    "dpo": "dpo.jsonl",
}


def open_jsonl(path):
    """逐行打开普通或 gzip 压缩的 JSONL，不加载整个数据集。"""
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", errors="replace")
    return path.open("r", encoding="utf-8-sig", errors="replace")


def clip(value, max_chars):
    """转为可读文本，并限制终端输出长度。"""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, indent=2)
    if len(value) > max_chars:
        return value[:max_chars] + f"...（已截断，原长 {len(value)} 字符）"
    return value


def show_messages(title, messages, max_chars):
    """显示 SFT/DPO 的 role-content 消息列表。"""
    print(f"  {title}:")
    if isinstance(messages, list):
        for index, message in enumerate(messages, start=1):
            if isinstance(message, dict):
                role = message.get("role", f"消息{index}")
                content = message.get("content", message)
                print(f"    [{role}] {clip(content, max_chars)}")
            else:
                print(f"    [消息{index}] {clip(message, max_chars)}")
    else:
        print(f"    {clip(messages, max_chars)}")


def show_sample(stage, sample, number, max_chars):
    """按阶段显示重点字段。"""
    print(f"\n--- 样本 {number} ---")
    if not isinstance(sample, dict):
        print(clip(sample, max_chars))
    elif stage == "pretrain":
        print(clip(sample.get("text", sample), max_chars))
    elif stage == "sft":
        messages = sample.get("conversations", sample.get("messages"))
        if messages is None:
            print(clip(sample, max_chars))
        else:
            show_messages("conversations", messages, max_chars)
    elif stage == "dpo":
        show_messages("chosen（偏好回答）", sample.get("chosen", []), max_chars)
        show_messages("rejected（非偏好回答）", sample.get("rejected", []), max_chars)
    else:
        print(clip(sample, max_chars))


def preview(stage, path, count, start, max_chars, raw_json=False):
    """从 start（跳过的样本数）起，显示 count 条数据。"""
    print(f"\n{'=' * 68}\n阶段：{stage} | 文件：{path}")
    if not path.is_file():
        print("文件不存在；请检查 --data-dir 或对应的 --*-file 参数。")
        return False

    print(f"文件大小：{path.stat().st_size / (1024 * 1024):.2f} MiB")
    sample_number = shown = 0
    try:
        with open_jsonl(path) as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                sample_number += 1
                if sample_number <= start:
                    continue
                if shown >= count:
                    break
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as error:
                    print(f"\n样本 {sample_number}（第 {line_number} 行）JSON 解析失败：{error}")
                    print(line[:max_chars])
                else:
                    if raw_json:
                        # 按 JSONL 单行格式原样展示结构；ensure_ascii=False 可直接显示中文。
                        print(json.dumps(sample, ensure_ascii=False, separators=(",", ":")))
                    else:
                        show_sample(stage, sample, sample_number, max_chars)
                shown += 1
    except OSError as error:
        print(f"读取失败：{error}")
        return False
    if shown == 0:
        print(f"没有可显示的样本（已跳过前 {start} 条）。")
    return True


def main():
    """解析命令行参数并预览所选阶段。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(
        description="抽样查看 MiniMind 的 JSONL 训练数据，不会加载整个文件。"
    )
    parser.add_argument("--stage", choices=(*DEFAULT_FILES, "all"), default="all",
                        help="查看 pretrain、sft、dpo 或 all，默认 all。")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"),
                        help="数据目录，默认当前目录下的 dataset。")
    parser.add_argument("--pretrain-file", type=Path, help="自定义预训练文件路径。")
    parser.add_argument("--sft-file", type=Path, help="自定义 SFT 文件路径。")
    parser.add_argument("--dpo-file", type=Path, help="自定义 DPO 文件路径。")
    parser.add_argument("--count", type=int, default=3, help="显示样本数，默认 3。")
    parser.add_argument("--start", type=int, default=0, help="跳过前 N 条，默认 0。")
    parser.add_argument("--max-chars", type=int, default=500,
                        help="每段文本最多显示字符数，默认 500。")
    parser.add_argument("--raw-json", action="store_true",
                        help="按 JSONL 原始单行样式打印完整样本，并直接显示中文。")
    args = parser.parse_args()
    if args.count < 1 or args.start < 0 or args.max_chars < 1:
        parser.error("--count 和 --max-chars 必须大于 0，--start 不能小于 0。")

    custom_paths = {
        "pretrain": args.pretrain_file,
        "sft": args.sft_file,
        "dpo": args.dpo_file,
    }
    stages = DEFAULT_FILES if args.stage == "all" else {args.stage: DEFAULT_FILES[args.stage]}
    results = []
    for stage, filename in stages.items():
        path = custom_paths[stage] or (args.data_dir / filename)
        results.append(preview(stage, path, args.count, args.start, args.max_chars, args.raw_json))
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

## 以截断的方式查看
# python inspect_training_data.py --stage all --count 3

## 以完整的方式查看
# python inspect_training_data.py --stage all --count 3 --raw-json