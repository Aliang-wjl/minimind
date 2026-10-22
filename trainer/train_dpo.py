"""MiniMind 的 DPO（Direct Preference Optimization）训练脚本。

主流程：
1. 从同一份 SFT 权重初始化可训练策略模型与冻结参考模型。
2. 对每条 chosen/rejected 回答，分别计算策略模型和参考模型的序列对数概率。
3. 优化“策略模型相对参考模型更偏好 chosen”的 DPO 目标。
4. 只更新策略模型；参考模型始终作为固定锚点。

说明：本脚本没有额外的 Reward Model，偏好监督直接来自 dpo.jsonl 中的
chosen/rejected 配对。
"""

# os/sys 用于路径、环境和 Python 模块搜索路径管理。
import os
import sys

# 将本文件声明为 trainer 包的一部分，并把项目根目录加入模块搜索路径。
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 提前导入 datasets，规避 Windows 下 pyarrow 与 torch 的 DLL 加载顺序冲突。
import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
# argparse 解析命令行训练参数；time 用于估算剩余训练时间。
import argparse
import time
# warnings 用于关闭不影响训练的提示信息。
import warnings
# torch 提供张量、自动微分和混合精度能力。
import torch
# F 提供 log_softmax、logsigmoid 等无状态函数。
import torch.nn.functional as F
# dist 提供单机多卡 DDP 通信能力。
import torch.distributed as dist
# CPU 环境使用 nullcontext 代替 CUDA autocast。
from contextlib import nullcontext
# optim 提供 AdamW 优化器。
from torch import optim
# DDP 包装策略模型，在多个 GPU 之间同步梯度。
from torch.nn.parallel import DistributedDataParallel
# DataLoader 负责组 batch，DistributedSampler 负责多卡数据切分。
from torch.utils.data import DataLoader, DistributedSampler
# MiniMindConfig 描述策略模型与参考模型的网络结构。
from model.model_minimind import MiniMindConfig
# DPODataset 将偏好数据转换为 chosen/rejected 的输入、目标和 loss mask。
from dataset.lm_dataset import DPODataset
# 复用项目通用的学习率、日志、断点恢复、DDP 和模型加载函数。
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

# 隐藏第三方库的非关键警告，使训练日志更易阅读。
warnings.filterwarnings('ignore')


def logits_to_log_probs(logits, labels):
    """从完整词表 logits 中取出目标 token 的逐位置对数概率。

    Args:
        logits: ``[B,T,V]``，每个位置对全部词表 token 的未归一化分数。
        labels: ``[B,T]``，每个位置真实目标 token 的词表 ID。

    Returns:
        ``[B,T]``，其中元素是 ``log p(label_t | context)``。
    """
    # 在词表维 V 上做 log-softmax，得到数值稳定的完整对数概率分布。
    log_probs = F.log_softmax(logits, dim=2)
    # labels.unsqueeze(2) 变为 [B,T,1]；gather 为每个位置抽取目标 token 的概率。
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    # 返回逐 token 对数概率，后续再用 mask 过滤并沿序列维求和。
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """计算一个 batch 的 DPO 损失。

    输入 batch 的排列约定是 ``[全部 chosen, 全部 rejected]``。目标为：

    ``-log σ(β[(logπ(chosen)-logπ(rejected))
                 -(logπ_ref(chosen)-logπ_ref(rejected))])``。

    参考模型项作为固定锚点，防止策略模型只靠无约束地改变整体概率尺度。
    """
    # loss mask 只保留 assistant 回答 token；沿 T 求和得到每条回答的序列 log-prob。
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    # 策略模型使用完全相同的 mask，保证两模型比较的是同一批回答位置。
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    # 当前 batch 前半是 chosen，后半是 rejected；总 batch 大小必然为偶数。
    batch_size = ref_log_probs.shape[0]
    # 参考模型对优选回答的序列对数概率。
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    # 参考模型对拒绝回答的序列对数概率。
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    # 策略模型对优选回答的序列对数概率。
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    # 策略模型对拒绝回答的序列对数概率。
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    # 策略模型内部 chosen 相对 rejected 的偏好强度。
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    # 参考模型原本具有的偏好强度，作为不可训练的基准。
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    # DPO 只奖励“策略模型相对参考模型增加了多少 chosen 偏好”。
    logits = pi_logratios - ref_logratios
    # beta 控制偏好优化强度；logsigmoid 将相对偏好差转换为二分类式负对数似然。
    loss = -F.logsigmoid(beta * logits)
    # 对 batch 中所有偏好对取平均，得到标量损失。
    return loss.mean()


def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
    """训练策略模型一个 epoch，参考模型只前向、不更新。

    ``model/optimizer/scaler/autocast_ctx/args`` 在主程序初始化后作为模块级变量使用。
    ``start_step`` 用于断点恢复时让日志、学习率和保存步数保持连续。
    """
    # 记录本轮开始时间，用于估算剩余分钟数。
    start_time = time.time()

    # 从 start_step+1 开始编号，恢复训练时不会把 step 又从 1 开始计算。
    for step, batch in enumerate(loader, start=start_step + 1):
        # chosen/rejected 的 x 是模型输入 token，形状均为 [B,T]。
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        # y 是右移一位后的目标 token ID，与模型每个位置的 logits 对齐。
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        # mask=1 的位置属于 assistant 回答，需要计入偏好概率；其余位置忽略。
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)
        # 在 batch 维拼成 [chosen_0...chosen_n, rejected_0...rejected_n]，只需各模型前向一次。
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # 根据全局训练进度计算余弦衰减学习率。
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 将新学习率写入优化器的每个参数组。
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # GPU 上按参数选择 FP16/BF16 autocast；CPU 上该上下文不做任何转换。
        with autocast_ctx:
            # 参考模型被冻结且只提供基准概率，无需构建反向传播计算图。
            with torch.no_grad():
                # 对 chosen/rejected 合并 batch 做参考模型前向。
                ref_outputs = ref_model(x)
                # 提取 [2B,T,V] 的参考模型词表 logits。
                ref_logits = ref_outputs.logits
            # 根据 y 从词表分布中抽取逐 token 的参考对数概率。
            ref_log_probs = logits_to_log_probs(ref_logits, y)

            # 策略模型执行同样前向；该计算图会参与反向传播。
            outputs = model(x)
            # 提取策略模型的 [2B,T,V] logits。
            logits = outputs.logits
            # 抽取策略模型赋给真实 chosen/rejected token 的逐位置对数概率。
            policy_log_probs = logits_to_log_probs(logits, y)

            # 对两模型的 chosen/rejected 序列概率应用 DPO 目标。
            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            # MoE 模型还要加专家负载均衡损失；Dense 模型的 aux_loss 为 0。
            loss = dpo_loss_val + outputs.aux_loss
            # 梯度累积 N 步时先除以 N，使累计梯度尺度与大 batch 平均损失一致。
            loss = loss / args.accumulation_steps

        # GradScaler 在 FP16 下放大 loss 防止梯度下溢；BF16/CPU 下等价于普通 backward。
        scaler.scale(loss).backward()

        # 达到累积步数或 epoch 最后一步时，才真正更新一次策略模型参数。
        if step % args.accumulation_steps == 0 or step == iters:
            # 梯度裁剪前先还原被 GradScaler 放大的梯度。
            scaler.unscale_(optimizer)
            # 将全局梯度范数限制在 grad_clip，减轻异常 batch 导致的训练震荡。
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 执行策略模型参数更新；参考模型不在此 optimizer 中，因此不会变化。
            scaler.step(optimizer)
            # 根据本轮是否溢出更新缩放因子。
            scaler.update()
            # set_to_none=True 比填 0 更节省显存，并允许 PyTorch 跳过部分无梯度参数。
            optimizer.zero_grad(set_to_none=True)

        # 按日志间隔或最后一步打印当前训练状态。
        if step % args.log_interval == 0 or step == iters:
            # 本 epoch 已消耗的墙钟时间。
            spend_time = time.time() - start_time
            # 日志恢复除以 accumulation_steps 前的真实损失数值。
            current_loss = loss.item() * args.accumulation_steps
            # 单独记录核心 DPO 项，便于与 MoE 辅助项区分。
            current_dpo_loss = dpo_loss_val.item()
            # Dense 模型为 0；MoE 模型显示负载均衡惩罚。
            current_aux_loss = outputs.aux_loss.item()
            # 读取优化器当前实际采用的学习率。
            current_lr = optimizer.param_groups[-1]['lr']
            # 用平均每 step 耗时估算本 epoch 剩余分钟数。
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60

            # Logger 只在主进程打印，避免 DDP 多个 rank 重复输出。
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, dpo_loss: {current_dpo_loss:.4f}, aux_loss: {current_aux_loss:.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')

            # 启用 SwanLab/WandB 兼容接口时，把同一组指标发送到可视化面板。
            if wandb:
                wandb.log({"loss": current_loss, "dpo_loss": current_dpo_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # 到保存间隔或最后一步时，仅由主进程写权重，防止多个 rank 竞争同一文件。
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            # 保存时暂时切换评估模式，关闭 Dropout 等训练行为。
            model.eval()
            # MoE 权重文件追加 _moe，避免覆盖同 hidden_size 的 Dense 权重。
            moe_suffix = '_moe' if lm_config.use_moe else ''
            # 构造供推理使用的轻量策略模型权重路径。
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # DDP 模型取 module；普通单卡模型直接使用自身。
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            # torch.compile 包装模型时，再取回内部原始模型。
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            # 读取当前策略模型全部参数和 buffer。
            state_dict = raw_model.state_dict()
            # 转为 FP16 并搬到 CPU 保存，降低磁盘占用且不持有 GPU 张量引用。
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 另外保存模型、优化器、scaler、epoch/step 等完整断点恢复状态。
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            # 保存结束后恢复训练模式。
            model.train()
            # 及时删除临时 state_dict 引用。
            del state_dict

        # 释放当前 batch 的大张量引用，便于下一轮及时回收显存。
        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, outputs, logits, policy_log_probs, loss


if __name__ == "__main__":
    # ----------------------------- 命令行参数 -----------------------------
    # 仅当直接执行 train_dpo.py 时进入这里；被其他模块 import 时不会自动开始训练。
    parser = argparse.ArgumentParser(description="MiniMind DPO (Direct Preference Optimization)")
    # 以下参数分别控制输出、训练规模、数值精度、模型结构、数据与日志。
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='dpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-8, help="初始学习率（建议<=5e-8避免遗忘）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl", help="DPO训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--beta', default=0.15, type=float, help="DPO中的beta参数")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    # 将命令行字符串解析为 args 对象，后续训练函数会读取它。
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 有 RANK 环境变量时初始化 NCCL 进程组并返回当前进程使用的本地 GPU 编号。
    local_rank = init_distributed_mode()
    # DDP 下每个进程只绑定一张 GPU。
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    # 不同 rank 使用不同随机序列，避免多个 GPU 完全相同地采样。
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    # 确保轻量模型权重输出目录存在。
    os.makedirs(args.save_dir, exist_ok=True)
    # 策略模型和参考模型必须使用相同架构配置，才能公平比较 token 概率。
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # --from_resume=1 时查找 DPO 完整恢复点；找不到则从头开始本阶段。
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None

    # ========== 3. 设置混合精度 ==========
    # autocast 需要设备类型；CPU 模式不进入 CUDA 混合精度上下文。
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # 命令行指定 bfloat16 时用 BF16，否则采用 FP16。
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # GPU 上让部分算子自动使用低精度，CPU 上使用空上下文。
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配wandb ==========
    # 未显式开启时保持 None，训练循环据此跳过远程日志。
    wandb = None
    # 只由 DDP 主进程创建日志任务。
    if args.use_wandb and is_main_process():
        # 项目在此以 SwanLab 提供与 wandb 相近的调用接口。
        import swanlab as wandb
        # 恢复点中若保存了运行 ID，则继续写入同一个实验记录。
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        # 有旧 ID 时要求恢复；没有时创建新运行。
        resume = 'must' if wandb_id else None
        # 把关键超参数编码到可读的运行名称中。
        wandb_run_name = f"MiniMind-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        # 初始化可视化实验。
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型和参考模型 ==========
    # 策略模型 πθ：默认从 full_sft 权重开始，参与梯度更新并最终保存为 dpo 权重。
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # 输出策略模型规模，便于确认 Dense/MoE 和 hidden_size 是否符合预期。
    Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')

    # 参考模型 πref：从与策略模型完全相同的初始权重加载，作为固定概率锚点。
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    # 切换评估模式，关闭 Dropout，保证参考概率固定且可重复。
    ref_model.eval()
    # 禁止参考模型所有参数求梯度；DPO 只训练策略模型。
    ref_model.requires_grad_(False)
    # 输出参考模型规模，正常情况下应与策略模型相同。
    Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')

    # 读取 chosen/rejected 偏好对，并生成两套输入、目标与 assistant loss mask。
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # DDP 下将样本划分给各进程；单卡时稍后直接使用随机索引。
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # FP16 需要动态梯度缩放；BF16 的指数范围较大，通常无需开启。
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # 优化器只接收策略模型参数，因此不会意外更新 ref_model。
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从ckp恢复状态 ==========
    # 默认从第 0 个 epoch、第 0 个 step 开始。
    start_epoch, start_step = 0, 0
    # 找到恢复点时还原可训练状态；参考模型仍从原始 full_sft 权重重新加载并冻结。
    if ckp_data:
        # 恢复策略模型参数。
        model.load_state_dict(ckp_data['model'])
        # 恢复 AdamW 的动量、方差等内部状态。
        optimizer.load_state_dict(ckp_data['optimizer'])
        # 恢复 FP16 GradScaler 状态。
        scaler.load_state_dict(ckp_data['scaler'])
        # 恢复训练位置。
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. 编译和分布式包装 ==========
    # 可选用 torch.compile 编译策略模型；参考模型保持普通 eager 模式。
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    # 多卡时只包装需要同步梯度的策略模型；冻结参考模型不需要 DDP。
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    # 从恢复 epoch 开始逐轮训练。
    for epoch in range(start_epoch, args.epochs):
        # DDP sampler 每轮设置 epoch，以产生各 rank 一致但不同轮次的洗牌。
        train_sampler and train_sampler.set_epoch(epoch)
        # 固定“基础种子+epoch”后创建单卡随机样本顺序，保证实验可复现。
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        # 只在恢复的第一个 epoch 跳过已完成 batch；后续 epoch 从头开始。
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # SkipBatchSampler 将样本索引组成 batch，并跳过恢复点之前的 batch。
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # DataLoader 根据 batch_sampler 取样；pin_memory 可加速 CPU 到 GPU 的拷贝。
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        # 恢复训练时把总 iters 加回已跳过步数，使学习率和日志进度连续。
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, ref_model, lm_config, start_step, wandb, args.beta)
        else:
            # 正常新 epoch 从 step 1 开始。
            train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta)

    # ========== 9. 清理分布进程 ==========
    # 等待所有 rank 完成，再释放 NCCL 进程组资源。
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
