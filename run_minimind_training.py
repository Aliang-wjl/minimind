"""Run MiniMind pretraining and supervised fine-tuning in sequence.

Run from an activated minimind-env with: python run_minimind_training.py
"""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
TRAINER = PROJECT / "trainer"
DATASET = PROJECT / "dataset"
OUT = PROJECT / "out"


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniMind: 预训练 -> 监督微调")
    parser.add_argument("--stage", choices=("both", "pretrain", "sft"), default="both",
                        help="运行阶段；默认先预训练，再做 SFT")
    parser.add_argument("--epochs", type=int, default=1, help="每个阶段的训练轮数，默认 1")
    parser.add_argument("--pretrain-batch-size", type=int, default=16)
    parser.add_argument("--sft-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Windows 上先用 0 更稳妥；运行正常后可尝试 2")
    parser.add_argument("--force", action="store_true",
                        help="允许覆盖已有的训练权重；不会自动续训")
    args = parser.parse_args()

    if args.epochs < 1 or args.pretrain_batch_size < 1 or args.sft_batch_size < 1 or args.num_workers < 0:
        parser.error("epochs、batch size 必须为正整数；num-workers 不能为负数")

    for path in (TRAINER / "train_pretrain.py", TRAINER / "train_full_sft.py",
                 DATASET / "pretrain_t2t_mini.jsonl", DATASET / "sft_t2t_mini.jsonl",
                 PROJECT / "model" / "tokenizer_config.json"):
        if not path.is_file():
            parser.error(f"缺少文件：{path}")

    try:
        import torch
    except ImportError as exc:
        parser.error(f"当前 Python 环境无法导入 PyTorch：{exc}")
    if not torch.cuda.is_available():
        parser.error(f"当前 Python 环境无法使用 CUDA：{sys.executable}")
    if not torch.cuda.is_bf16_supported():
        parser.error("当前 GPU 不支持脚本默认的 BF16 混合精度")

    pretrain_weight = OUT / "pretrain_768.pth"
    sft_weight = OUT / "full_sft_768.pth"
    if args.stage in ("both", "pretrain") and pretrain_weight.exists() and not args.force:
        parser.error(f"预训练权重已存在：{pretrain_weight}\n"
                     "如要覆盖，添加 --force；如只想接着做 SFT，使用 --stage sft")
    if args.stage in ("both", "sft") and sft_weight.exists() and not args.force:
        parser.error(f"SFT 权重已存在：{sft_weight}\n如要覆盖，添加 --force")
    if args.stage == "sft" and not pretrain_weight.is_file():
        parser.error(f"SFT 需要先有预训练权重：{pretrain_weight}")

    print(f"Python: {sys.executable}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"项目目录: {PROJECT}", flush=True)

    if args.stage in ("both", "pretrain"):
        command = [sys.executable, "-u", "train_pretrain.py",
                   "--epochs", str(args.epochs),
                   "--batch_size", str(args.pretrain_batch_size),
                   "--accumulation_steps", "2",
                   "--num_workers", str(args.num_workers),
                   "--log_interval", "50"]
        print("\n开始预训练：", " ".join(command), flush=True)
        subprocess.run(command, cwd=TRAINER, check=True)
        if not pretrain_weight.is_file():
            raise RuntimeError(f"预训练结束，但未找到权重：{pretrain_weight}")

    if args.stage in ("both", "sft"):
        command = [sys.executable, "-u", "train_full_sft.py",
                   "--epochs", str(args.epochs),
                   "--batch_size", str(args.sft_batch_size),
                   "--accumulation_steps", "2",
                   "--num_workers", str(args.num_workers),
                   "--log_interval", "50"]
        print("\n开始监督微调：", " ".join(command), flush=True)
        subprocess.run(command, cwd=TRAINER, check=True)
        if not sft_weight.is_file():
            raise RuntimeError(f"SFT 结束，但未找到权重：{sft_weight}")

    print("\n训练完成。权重保存在：", OUT, flush=True)


if __name__ == "__main__":
    main()
