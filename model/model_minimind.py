"""MiniMind 核心模型实现（中文教学注释版）。

本文件从下到上可以分成五个层次：
1. ``MiniMindConfig``：保存模型规模、注意力、RoPE 和 MoE 等超参数。
2. 基础算子：RMSNorm、RoPE、GQA 所需的 KV 复制函数。
3. Transformer 子模块：Attention、普通 FFN、MoE FFN、MiniMindBlock。
4. ``MiniMindModel``：Embedding + 多层 Transformer Block + 最终归一化。
5. ``MiniMindForCausalLM``：增加词表投影、语言模型损失和自回归生成。

代码中的常用形状约定：
    B：batch size；T：序列长度；C：hidden_size；
    H：Query 头数；H_kv：Key/Value 头数；D：每个注意力头的维度；V：词表大小。
"""

# math 提供 pi/log/sqrt/ceil；torch 提供张量和自动微分；F 是无状态神经网络函数集。
import math, torch, torch.nn.functional as F

# nn 提供 Module、Linear、Embedding、Dropout 等带参数或状态的网络层。
from torch import nn

# ACT2FN 将字符串（例如 "silu"）映射为实际的激活函数实现。
from transformers.activations import ACT2FN

# PretrainedConfig 负责配置序列化；PreTrainedModel/GenerationMixin 提供 HF 模型兼容接口。
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig

# 标准化返回结构，可同时携带 loss、logits、KV cache、hidden states 和 MoE 辅助损失。
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    """集中保存 MiniMind 的全部结构超参数。

    ``PretrainedConfig`` 让这些字段能够写入 ``config.json``，并被
    Hugging Face 的 ``from_pretrained`` / ``save_pretrained`` 接口识别。
    """

    # Hugging Face 用该字符串标识模型架构类型。
    model_type = "minimind"

    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        """创建一份模型配置。

        Args:
            hidden_size: 每个 token 的隐藏向量维度 C。
            num_hidden_layers: 堆叠的 Transformer Block 数量。
            use_moe: 为 True 时用 MoE FFN，为 False 时用普通 Dense FFN。
            **kwargs: 其余可选配置，也会交给父类保存和处理。
        """
        # 初始化父类，处理通用字段（如名称、返回格式、特殊 token ID 等）。
        super().__init__(**kwargs)

        # Transformer 主干中每个 token 表示的维度，默认 C=768。
        self.hidden_size = hidden_size
        # Transformer Block 的层数，默认 8 层。
        self.num_hidden_layers = num_hidden_layers
        # 是否启用稀疏混合专家前馈网络。
        self.use_moe = use_moe

        # Dropout 概率；默认 0 表示不随机丢弃神经元。
        self.dropout = kwargs.get("dropout", 0.0)
        # 词表大小 V；Embedding 和 lm_head 都围绕该大小建立。
        self.vocab_size = kwargs.get("vocab_size", 6400)
        # 序列开始 token（BOS）的词表 ID。
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        # 序列结束 token（EOS）的词表 ID。
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        # 是否优先使用 PyTorch 的 scaled_dot_product_attention 快速实现。
        self.flash_attn = kwargs.get("flash_attn", True)

        # Query 注意力头数 H，默认 8。
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        # Key/Value 头数 H_kv，默认 4；小于 Query 头数时构成 GQA。
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        # 每个头的维度 D；默认由 C/H 得到 768/8=96。
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)

        # FFN 使用的激活函数名称；silu 与门控乘法组合成 SwiGLU 风格结构。
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        # FFN 中间维度；按 hidden_size*pi 估计后向上取整到 64 的整数倍。
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)

        # 预计算 RoPE 时支持的最大位置数。
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        # RMSNorm 防止除零并改善数值稳定性的极小常数。
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        # RoPE 的频率基数 theta；越大通常越有利于覆盖较长位置尺度。
        self.rope_theta = kwargs.get("rope_theta", 1e6)

        # 是否让输入 Embedding 与输出 lm_head 共用同一份词表权重。
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        # 是否在推理时启用 YaRN，对 RoPE 频率做长度外推。
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)

        # 开启外推时使用 YaRN 参数；关闭时为 None，采用原始 RoPE 频率。
        self.rope_scaling = {
            # 高频区边界参数：控制哪些维度开始插值。
            "beta_fast": 32,
            # 低频区边界参数：控制哪些维度完全缩放。
            "beta_slow": 1,
            # 目标上下文相对原始上下文的长度放大倍数。
            "factor": 16,
            # 原始训练上下文长度，YaRN 以它作为外推基准。
            "original_max_position_embeddings": 2048,
            # 对 cos/sin 幅度的额外缩放系数，本配置保持 1.0。
            "attention_factor": 1.0,
            # 配置类型标识，便于说明这里采用 YaRN。
            "type": "yarn"
        } if self.inference_rope_scaling else None

        # ---------------- MoE 专用配置；use_moe=False 时不会影响主干计算 ----------------
        # 专家总数；每个专家本质上都是一套独立的 FeedForward。
        self.num_experts = kwargs.get("num_experts", 4)
        # 每个 token 实际激活的专家数量 K；默认 Top-1 路由。
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        # 每个专家内部的中间维度，默认与普通 FFN 相同。
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        # 是否把选中的 Top-K 路由概率重新归一化，使它们之和为 1。
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        # 专家负载均衡辅助损失的系数。
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm(torch.nn.Module):
    """RMSNorm：只按均方根缩放特征，不减去均值。

    对最后一维执行 ``x / sqrt(mean(x^2) + eps)``，再乘一个可学习缩放向量。
    相比 LayerNorm，它省略了减均值步骤，计算更简单，常用于现代大语言模型。
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        """初始化归一化层。

        Args:
            dim: 最后一维的特征数，决定可学习缩放参数的长度。
            eps: 加在均方值上的小常数，避免开根号和除法出现数值问题。
        """
        # 初始化 nn.Module 的参数、子模块与 hook 管理能力。
        super().__init__()
        # 保存数值稳定项，forward 时会用到。
        self.eps = eps
        # 每个特征维度一个可学习缩放值；初始化为 1，不改变初始归一化结果。
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        """计算不含可学习权重的 RMS 归一化结果。"""
        # x.pow(2).mean(-1) 计算最后一维的均方；rsqrt 直接计算平方根的倒数。
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """以 float32 完成归一化，再恢复输入 dtype 并乘可学习缩放量。"""
        # 半精度输入先转 float32 可降低均方与开根号的数值误差；type_as 恢复原精度。
        return (self.weight * self.norm(x.float())).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    """预计算 RoPE 在所有位置上的 cos/sin 表。

    RoPE（Rotary Position Embedding）通过旋转 Q/K 向量注入相对位置信息。
    该函数只生成位置 0 到 ``end-1`` 的旋转系数，并不直接修改 Q/K。

    Args:
        dim: 单个注意力头的维度 D，通常必须为偶数。
        end: 要预计算的位置数量。
        rope_base: 决定各维度旋转频率跨度的基数 theta。
        rope_scaling: 可选 YaRN 参数；不为 None 时对频率做长上下文缩放。

    Returns:
        ``(freqs_cos, freqs_sin)``，两者形状均为 ``[end, dim]``。
    """
    # 偶数维度各对应一个基础角频率；attn_factor 用于 YaRN 的可选幅度修正。
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0

    # 如果传入 YaRN 配置，则在不同频段之间平滑插值，避免所有频率被生硬地统一缩放。
    if rope_scaling is not None:  # YaRN: f'(i)=f(i)((1-γ)+γ/s)，γ 是 [0,1] 的线性坡度。
        # 读取 YaRN 参数；缺失字段时使用与配置类一致的默认值。
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )

        # 只有目标长度超过原始上下文长度时，才真正需要做频率外推。
        if end / orig_max > 1.0:
            # 根据给定旋转圈数 b，反推出其大致对应的频率维度位置。
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            # 将连续边界裁剪到合法的半维索引范围，得到插值区间 [low, high]。
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            # ramp 在低维侧为 0、高维侧为 1，中间线性过渡。
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            # 低维保留更多原值，高维逐渐除以 factor，从而延长有效位置范围。
            freqs = freqs * (1 - ramp + ramp / factor)

    # 创建位置索引 [0, 1, ..., end-1]。
    t = torch.arange(end, device=freqs.device)
    # 外积得到每个位置在每个频率上的旋转角，形状 [end, dim/2]。
    freqs = torch.outer(t, freqs).float()
    # 将半维 cos 复制为完整 dim，匹配 rotate_half 采用的前后半维旋转布局。
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    # sin 与 cos 使用相同布局，供旋转公式的交叉项使用。
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    # 返回可供多层 Attention 共享的查表结果。
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """把预计算的 RoPE 旋转应用到 Query 和 Key。

    ``q``/``k`` 进入本函数时通常是 ``[B, T, H, D]``；cos/sin 是 ``[T, D]``。
    在头维处插入广播维后，每个 batch、每个头都使用相同的位置旋转系数。
    """

    def rotate_half(x):
        """将最后一维的前后两半交换，并给原后半部分取负，构成二维旋转的正交项。"""
        # 若 x=[a,b]（a/b 各占半维），则返回 [-b,a]。
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # q*cos + rotate(q)*sin 完成位置相关旋转，并恢复为 q 的原始 dtype。
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    # Key 必须执行完全一致的旋转，Q·K 点积才能体现相对位置差。
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    # Value 不参与 RoPE，所以这里只返回旋转后的 Q 和 K。
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """为 GQA 重复 Key/Value 头，使其数量与 Query 头一致。

    输入形状是 ``[B,T,H_kv,D]``，输出形状是 ``[B,T,H_kv*n_rep,D]``。
    例如 8 个 Q 头、4 个 KV 头时 ``n_rep=2``，每个 KV 头供两个 Q 头共享。
    """
    # 从输入张量解包 batch、序列长度、KV 头数和头维度。
    bs, slen, num_key_value_heads, head_dim = x.shape

    # KV 头数已经等于 Q 头数时无需复制，直接返回原张量。
    if n_rep == 1:
        return x

    # 插入复制维并用 expand 做逻辑广播，再 reshape 合并为实际的头维。
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

class Attention(nn.Module):
    """带 GQA、QK-Norm、RoPE、因果遮罩和 KV Cache 的自注意力层。"""

    def __init__(self, config: MiniMindConfig):
        """根据配置建立 Q/K/V/O 投影、归一化和 Dropout 层。"""
        # 初始化 nn.Module 基类。
        super().__init__()

        # 如果未指定 KV 头数，就令其等于 Q 头数，退化为标准多头注意力 MHA。
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        # 本实现中的 Query 头数 H。
        self.n_local_heads = config.num_attention_heads
        # 本实现中的 Key/Value 头数 H_kv。
        self.n_local_kv_heads = self.num_key_value_heads
        # 每个 KV 头要复制多少次；H/H_kv>1 时即为分组查询注意力 GQA。
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 单个注意力头的特征维度 D。
        self.head_dim = config.head_dim
        # 自回归语言模型必须使用因果注意力，禁止看到未来 token。
        self.is_causal = True

        # 将输入 [B,T,C] 映射为所有 Query 头，共 H*D 个特征。
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        # 只生成 H_kv 个 Key 头；GQA 下参数量和 KV Cache 都更小。
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # Value 与 Key 使用相同数量的 KV 头。
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # 拼接所有 Query 头的输出后，再投影回隐藏维度 C。
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # 对每个 Query 头的 D 维向量做 RMSNorm，稳定注意力分数。
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # 对每个 Key 头做相同的 QK-Norm。
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # 手写注意力分支中，对 softmax 后的注意力概率做 Dropout。
        self.attn_dropout = nn.Dropout(config.dropout)
        # 输出投影之后、进入残差连接之前的 Dropout。
        self.resid_dropout = nn.Dropout(config.dropout)
        # 保存裸 dropout 概率，传给函数式 SDPA 接口。
        self.dropout = config.dropout
        # 仅当当前 PyTorch 提供 SDPA 且配置允许时，才启用快速注意力路径。
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """执行一次因果自注意力。

        Args:
            x: 当前层输入，形状 ``[B,T,C]``。
            position_embeddings: 当前位置对应的 ``(cos, sin)`` RoPE 表。
            past_key_value: 可选历史 ``(K,V)``，生成阶段用于避免重复计算旧 token。
            use_cache: 是否把拼接后的 K/V 返回给下一次生成。
            attention_mask: ``[B,total_T]``，1 表示有效 token，0 表示 padding。

        Returns:
            ``output``：形状 ``[B,T,C]``；``past_kv``：可选的新 KV Cache。
        """
        # 读取 batch 大小和本次输入的 token 数；第三维就是 hidden_size。
        bsz, seq_len, _ = x.shape

        # 三个线性投影分别产生 Q、K、V，目前仍是把所有头平铺在最后一维。
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 拆出 Query 头维，得到 [B,T,H,D]。
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        # Key 只有 H_kv 个头，形状 [B,T,H_kv,D]。
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        # Value 与 Key 形状相同：[B,T,H_kv,D]。
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 对 Q/K 每个头单独做 RMSNorm；V 不做归一化。
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        # 解包当前 token 位置对应的 cos/sin 表，二者形状通常为 [T,D]。
        cos, sin = position_embeddings
        # 将绝对位置以旋转形式写入 Q/K。
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # 自回归生成时，把历史缓存与本次新 token 的 K/V 沿时间维拼接。
        if past_key_value is not None:
            # past K 的形状是 [B,past_T,H_kv,D]，拼接后时间长度变为 past_T+T。
            xk = torch.cat([past_key_value[0], xk], dim=1)
            # Value 使用相同方式扩展完整上下文。
            xv = torch.cat([past_key_value[1], xv], dim=1)

        # 只有调用者要求缓存时才保留 K/V；训练时通常为 None，以节省引用和返回开销。
        past_kv = (xk, xv) if use_cache else None

        # Attention 内核要求 [B,H,T,D]；同时把 H_kv 个 K/V 头复制到 H 个头。
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))

        # 无 KV Cache、无真实 padding 且序列不止一个 token 时，可走高效 SDPA 分支。
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            # SDPA 内部完成 QK^T、缩放、因果遮罩、softmax、dropout 和与 V 相乘。
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            # 手工计算缩放点积注意力分数，形状 [B,H,T,current_total_T]。
            scores = (xq @ xk.transpose(-2, -1)).float() / math.sqrt(self.head_dim)

            # 上三角位置代表“未来 token”；加 -inf 后，其 softmax 权重变为 0。
            if self.is_causal:
                # 只遮住右下角的本次输入区域；更早的缓存位置对新 token 全部可见。
                scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)

            # padding key 对所有头、所有 query 都不可见；[B,T] 扩成 [B,1,1,T] 后广播。
            if attention_mask is not None:
                # 有效位置加 0，mask=0 的位置加一个极大负数，使 softmax 后近似为 0。
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9

            # 将分数归一化为概率，恢复为 Q 的 dtype，做 Dropout 后再对 V 加权求和。
            output = self.attn_dropout(F.softmax(scores, dim=-1).type_as(xq)) @ xv

        # 从 [B,H,T,D] 转回 [B,T,H,D]，再把 H 和 D 合并成总特征维。
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        # 经过输出投影回到 C 维，并做残差分支上的 Dropout。
        output = self.resid_dropout(self.o_proj(output))
        # 返回注意力输出，以及生成时可复用的未复制 KV Cache。
        return output, past_kv

class FeedForward(nn.Module):
    """SwiGLU 风格的门控前馈网络。

    整体公式为 ``down(SiLU(gate(x)) * up(x))``：gate 分支控制信息通过比例，
    up 分支承载内容，逐元素相乘后再投影回 hidden_size。
    """

    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        """建立门控、上投影、下投影和激活函数。"""
        # 初始化 nn.Module 基类。
        super().__init__()
        # 调用者可覆盖中间维度；未传入时采用全局配置。
        intermediate_size = intermediate_size or config.intermediate_size
        # 门控分支：把最后一维从 C 扩展到 intermediate_size。
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        # 下投影：把中间特征重新压回 C，便于与残差相加。
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        # 内容分支：与 gate_proj 并行地把 C 扩展到 intermediate_size。
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        # 根据配置字符串取得激活函数；默认 ACT2FN['silu'] 即 SiLU。
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """执行门控 FFN，并保持输入输出形状均为 ``[B,T,C]``。"""
        # gate(x) 经激活后逐元素调制 up(x)，down_proj 再投影回隐藏维度。
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MOEFeedForward(nn.Module):
    """稀疏混合专家前馈网络（Mixture of Experts）。

    每个专家都是一套独立的 ``FeedForward``。路由器对每个 token 产生专家概率，
    只执行 Top-K 个专家，再按路由权重合并输出。这样可以扩大总参数容量，同时让
    每个 token 只承担少量专家的计算成本。
    """

    def __init__(self, config: MiniMindConfig):
        """建立路由器和多个独立专家。"""
        # 初始化 nn.Module 基类。
        super().__init__()
        # 保存完整配置，forward 中需要读取 Top-K、归一化与辅助损失参数。
        self.config = config
        # 路由器把每个 token 的 C 维表示映射为 num_experts 个专家分数。
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        # ModuleList 注册所有专家参数，使优化器和 state_dict 能发现它们。
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        # 保留激活函数引用；当前 forward 不直接使用，专家内部会使用同一配置的激活函数。
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """逐 token 路由、执行选中的专家，并计算负载均衡损失。

        Args:
            x: 形状 ``[B,T,C]`` 的 token 隐藏状态。

        Returns:
            与 x 同形状的专家混合结果；辅助损失保存在 ``self.aux_loss``。
        """
        # 记录原始三维形状，函数末尾需要还原。
        batch_size, seq_len, hidden_dim = x.shape
        # 合并 batch 和时间维，让路由器把每个 token 当作独立样本：[B*T,C]。
        x_flat = x.view(-1, hidden_dim)
        # gate 产生专家 logits，softmax 后得到每个 token 对所有专家的概率：[B*T,E]。
        scores = F.softmax(self.gate(x_flat), dim=-1)
        # 对每个 token 选概率最大的 K 个专家，同时返回概率和专家编号：[B*T,K]。
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)

        # 根据配置决定是否重新归一化被选中的路由权重。
        if self.config.norm_topk_prob:
            # Top-K（K>1）时，让被选中专家的权重之和严格为 1。
            if self.config.num_experts_per_tok > 1:
                # 加 1e-20 防止极端数值情况下除以 0。
                topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
            else:
                # Top-1 时重新计算最大概率；detach 输入，避免这条辅助路径把梯度传回 x。
                top1 = torch.topk(F.softmax(self.gate(x_flat.detach()), dim=-1), k=1, dim=-1, sorted=False)[0]

                # 前向数值为 1.0，但反向对 top1 的导数仍为 1，这是直通估计技巧。
                topk_weight = top1 - top1.detach() + 1.0  # k=1：输出不缩放，路由器仍可获梯度。

        # 创建展平后的输出缓冲区；每个专家的结果会累加到相应 token 行。
        y = torch.zeros_like(x_flat)

        # 逐个专家处理分配给它的 token；未被分配的 token 不进入该专家，节省计算。
        for i, expert in enumerate(self.experts):
            # mask[n,k] 表示第 n 个 token 的第 k 个候选是否为当前专家 i。
            mask = (topk_idx == i)

            # 当前 batch 中至少有一个 token 选择了该专家时，才真正执行专家网络。
            if mask.any():
                # 找到选择当前专家的 token 行号；any 合并 Top-K 槽位维。
                token_idx = mask.any(dim=-1).nonzero().flatten()
                # 提取这些 token 对当前专家的混合权重，并整理为 [命中数,1] 以便广播。
                weight = topk_weight[mask].view(-1, 1)
                # 执行专家、乘路由权重，再用 index_add_ 累加回每个 token 对应的位置。
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))

            # 训练时即使专家未被选中，也用乘 0 的方式让它出现在计算图中。
            elif self.training:
                # 数值结果不变、梯度为 0；主要避免 DDP 将该专家判定为 unused parameters。
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())

        # 训练且系数大于 0 时，计算鼓励专家负载均衡的辅助损失。
        if self.training and self.config.router_aux_loss_coef > 0:
            # one_hot 后对 token 维求均值，得到各 Top-K 槽位对每个专家的实际选择频率。
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            # 将硬路由负载与平均软概率结合；乘专家数和系数后得到标量辅助损失。
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            # Dense/推理语境不需要路由辅助损失；创建同设备、同 dtype 的标量 0。
            self.aux_loss = scores.new_zeros(1).squeeze()

        # 把 [B*T,C] 恢复为 Transformer 主干所需的 [B,T,C]。
        return y.view(batch_size, seq_len, hidden_dim)

class MiniMindBlock(nn.Module):
    """一个 Pre-Norm Transformer 解码器块。

    数据依次经过 ``RMSNorm → 因果自注意力 → 残差``，再经过
    ``RMSNorm → Dense/MoE FFN → 残差``。输入输出形状均为 ``[B,T,C]``。
    """

    def __init__(self, layer_id: int, config: MiniMindConfig):
        """建立一层注意力、两层归一化和一层前馈网络。"""
        # 初始化 nn.Module 基类。
        super().__init__()
        # 当前层使用的因果自注意力模块。
        self.self_attn = Attention(config)
        # 注意力之前的 Pre-Norm。
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # FFN 之前的 Pre-Norm。
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 按 use_moe 选择普通 FFN 或稀疏专家 FFN；二者输入输出接口一致。
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """完成一个解码器块的注意力与前馈计算。"""
        # 暂存块输入，稍后与注意力输出构成第一条残差连接。
        residual = hidden_states

        # 先归一化，再执行自注意力；同时可能返回当前层的新 KV Cache。
        hidden_states, present_key_value = self.self_attn(
            # 第一个位置参数是归一化后的 token 隐藏状态。
            self.input_layernorm(hidden_states), position_embeddings,
            # 这三个位置参数依次传入历史缓存、是否缓存、padding mask。
            past_key_value, use_cache, attention_mask
        )

        # 第一条残差连接：Attention(Norm(x)) + x。
        hidden_states += residual
        # 第二条残差连接：当前状态 + FFN(Norm(当前状态))。
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        # 返回本层隐藏状态，以及生成阶段需要的当前层 KV Cache。
        return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    """MiniMind 的裸 Transformer 主干，不包含最终词表投影和语言模型损失。"""

    def __init__(self, config: MiniMindConfig):
        """建立词嵌入、N 个解码器块、最终 RMSNorm 和共享 RoPE 缓冲区。"""
        # 初始化 nn.Module 基类。
        super().__init__()
        # 保存配置，前向阶段重建 RoPE 时仍需读取相关参数。
        self.config = config
        # 缓存词表大小 V 和层数 N，便于构造后续模块。
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        # 把整数 token ID 查表转换为 C 维连续向量：[B,T] → [B,T,C]。
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # 对输入 embedding 做 Dropout。
        self.dropout = nn.Dropout(config.dropout)
        # 创建 N 个相互独立、结构相同的 Transformer Block。
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        # 所有 Block 之后再做一次最终 RMSNorm。
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 一次性预计算整个最大位置范围的 RoPE cos/sin，供所有层重复使用。
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        # 注册为非持久缓冲区：随 model.to(device) 移动，但不写入 state_dict。
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        # sin 表使用同样的缓冲区策略，避免每层、每步重复计算。
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        """让 token IDs 依次通过 Embedding 和全部 Transformer Block。

        Args:
            input_ids: 整数 token ID，形状 ``[B,T]``。
            attention_mask: 有效 token 标记，形状 ``[B,total_T]``。
            past_key_values: 每层一组历史 ``(K,V)``，用于增量生成。
            use_cache: 是否收集并返回每层的新 KV Cache。
            **kwargs: 兼容上层 Hugging Face 调用中可能附带的额外字段。

        Returns:
            ``hidden_states``：最终隐藏状态；``presents``：逐层缓存；
            ``aux_loss``：所有 MoE 层的负载均衡损失之和。
        """
        # 获取 batch 大小和本次输入序列长度。
        batch_size, seq_length = input_ids.shape

        # 某些 transformers 版本传入 DynamicCache；本实现只接受旧式 list/tuple，故忽略它。
        if hasattr(past_key_values, 'layers'): past_key_values = None
        # 没有缓存时，为每个 Block 准备一个 None，方便后面直接 zip 遍历。
        past_key_values = past_key_values or [None] * len(self.layers)
        # 若有缓存，历史长度就是 K 的时间维；否则当前输入从位置 0 开始。
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # token ID 经 Embedding 变为 [B,T,C]，随后应用输入 Dropout。
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # transformers>=5.x 的 meta-device 初始化可能丢失非持久 RoPE buffer；检测到异常时重建。
        if self.freqs_cos[0, 0] == 0:
            # 按原始配置重新计算完整 cos/sin 表。
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            # 把重建结果移动到与隐藏状态相同的设备，避免 CPU/GPU 混用。
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)

        # 只截取本次 token 对应的位置区间；生成时从 past 长度 start_pos 之后开始。
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])

        # 收集每个 Transformer Block 返回的新 KV Cache。
        presents = []
        # 逐层传递隐藏状态，同时给每层配对自己的历史缓存。
        for layer, past_key_value in zip(self.layers, past_key_values):
            # 当前层完成 Attention、FFN 与两次残差连接。
            hidden_states, present = layer(
                # 当前隐藏状态与共享的位置旋转表。
                hidden_states,
                position_embeddings,
                # 使用关键字参数明确传入该层历史缓存。
                past_key_value=past_key_value,
                # 决定 present 是否保存真实 K/V。
                use_cache=use_cache,
                # 把 padding mask 继续传给 Attention。
                attention_mask=attention_mask
            )
            # 即使未启用 cache，仍会追加 None，从而保持列表长度与层数一致。
            presents.append(present)

        # 对最后一层输出做最终 RMSNorm。
        hidden_states = self.norm(hidden_states)
        # 只汇总 MoE 层的 aux_loss；Dense 模型从同设备的标量 0 开始并保持结果为 0。
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        # 返回主干表示、逐层缓存和 MoE 辅助损失，交给 CausalLM 包装层继续处理。
        return hidden_states, presents, aux_loss

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """用于因果语言建模的完整 MiniMind。

    它在 ``MiniMindModel`` 主干之后增加 ``lm_head``，将隐藏状态映射为词表 logits；
    提供 labels 时还会计算 next-token 交叉熵，并提供一个轻量的自回归 ``generate``。
    """

    # 告诉 Hugging Face 该模型对应的配置类。
    config_class = MiniMindConfig
    # 声明 lm_head 与输入 Embedding 的权重绑定关系，供保存/加载等框架逻辑识别。
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MiniMindConfig = None):
        """创建 Transformer 主干和词表输出头，并按配置决定是否共享词嵌入权重。"""
        # 调用者不传配置时使用默认 MiniMindConfig。
        self.config = config or MiniMindConfig()
        # 初始化 PreTrainedModel，让框架记录配置并建立通用模型能力。
        super().__init__(self.config)
        # 裸 MiniMind 主干负责生成每个 token 的 C 维隐藏状态。
        self.model = MiniMindModel(self.config)
        # 语言模型头把 [B,T,C] 投影成 [B,T,V]；不使用 bias。
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # 权重共享时，输入查表和输出词表分类真正引用同一个 Parameter 对象。
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight

        # 执行 Hugging Face 的模型初始化收尾逻辑（初始化权重、处理绑定等）。
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        """完成主干前向、词表投影，并可选地计算 next-token loss。

        Args:
            input_ids: ``[B,T]`` 的 token IDs。
            attention_mask: ``[B,total_T]`` 的有效位置标记。
            past_key_values: 可选逐层 KV Cache。
            use_cache: 是否返回更新后的缓存。
            logits_to_keep: 只计算末尾若干位置的 logits；0 表示保留全部位置。
            labels: ``[B,T]`` 的监督目标；值为 -100 的位置不计入损失。
            **kwargs: 供 Hugging Face 接口兼容的额外参数。

        Returns:
            ``MoeCausalLMOutputWithPast``，包含 loss、aux_loss、logits、缓存和隐藏状态。
        """
        # 主干返回最终隐藏状态、各层新缓存和所有 MoE 层的辅助损失之和。
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)

        # int=0 时 slice(-0,None) 等价于 slice(0,None)，即保留全部 token；正整数只保留末尾 N 个。
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # 选取所需隐藏位置并映射到 V 维词表分数。
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # 推理时没有 labels，因此默认不计算损失。
        loss = None

        # 训练/评估传入 labels 时，计算“当前位置预测下一个 token”的交叉熵。
        if labels is not None:
            # 丢弃 logits 最后一个位置、labels 第一个位置，使 logits[t] 对齐 labels[t+1]。
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            # 展平 batch 和时间维；label=-100 的 padding/非监督位置由 ignore_index 忽略。
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)

        # 使用标准结构返回，调用者可通过 res.loss、res.logits 等属性读取结果。
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)

    # https://github.com/jingyaogong/minimind/discussions/611
    # inference_mode 关闭梯度记录和版本计数，降低生成阶段的显存与运行开销。
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        """用自回归循环生成后续 token。

        每轮只预测一个新 token：读取最后位置 logits，依次应用温度、重复惩罚、
        Top-K、Top-P，再进行采样或贪心选择；遇到 EOS 或达到上限后停止。

        Args:
            inputs: 初始 ``input_ids``，形状 ``[B,T]``。
            attention_mask: 初始有效 token 标记。
            max_new_tokens: 最多新增多少个 token。
            temperature: logits 温度；越小分布越尖锐，越大越随机。
            top_p: nucleus sampling 的累计概率阈值。
            top_k: 每轮最多保留概率最高的 K 个候选。
            eos_token_id: 序列结束 token；为 None 时不提前终止。
            streamer: 可选流式输出器，每生成一个 token 就推送一次。
            use_cache: 是否利用 KV Cache 避免重复计算历史 token。
            num_return_sequences: 每条输入并行复制并生成多少个候选。
            do_sample: True 时按概率采样，False 时选择最大 logit。
            repetition_penalty: 已出现 token 的重复惩罚系数，1.0 表示关闭。
            **kwargs: 可包含 input_ids、past_key_values、return_kv 等兼容参数。

        Returns:
            默认返回包含原 prompt 的完整 token IDs；``return_kv=True`` 时返回字典。
        """
        # kwargs 中的 input_ids 优先于 inputs；repeat 沿 batch 维复制多个候选序列。
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        # attention_mask 必须与输入做相同复制；未提供时保持 None。
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        # 允许调用者从已有 KV Cache 接着生成；读取后从 kwargs 删除，避免重复传参。
        past_key_values = kwargs.pop("past_key_values", None)
        # 为每条候选建立结束标记；初始时所有序列都尚未完成。
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        # 使用 streamer 时先把完整 prompt 推给它；streamer 通常自行决定是否显示 prompt。
        if streamer:
            streamer.put(input_ids.cpu())

        # 每轮生成一个 token，最多循环 max_new_tokens 次。
        for _ in range(max_new_tokens):
            # 已有缓存时，past_len 是主干已经计算过的历史 token 数。
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            # 只把尚未进入缓存的 token 送入模型；首次循环 past_len=0，会处理完整 prompt。
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            # 新 token 将成为下一轮的有效上下文，因此在 mask 末尾追加一列 1。
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            # 只需要最后位置预测下一个 token 的 logits；除以温度调整概率分布尖锐程度。
            logits = outputs.logits[:, -1, :] / temperature

            # repetition_penalty!=1 时，降低当前序列已经出现过的 token 的再次出现倾向。
            if repetition_penalty != 1.0:
                # 每条序列见过的 token 集合不同，因此逐条处理。
                for i in range(input_ids.shape[0]):
                    # unique 取得已出现过的词表 ID。
                    seen = torch.unique(input_ids[i])
                    # 读取这些 token 当前对应的 logits。
                    score = logits[i, seen]
                    # 正分数做除法、负分数做乘法，二者都会让候选相对变差。
                    logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)

            # Top-K：仅保留每条序列分数最高的 K 个 token。
            if top_k > 0:
                # 低于第 K 大阈值的 logits 设为 -inf，使其采样概率变为 0。
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')

            # Top-P：保留从高到低累计概率刚好覆盖 p 的最小候选集合。
            if top_p < 1.0:
                # 对每条序列的 logits 降序排序，并保留原词表索引。
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                # softmax 后做累计和；累计概率超过 top_p 的位置先标记为删除。
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                # 右移 mask，确保第一个使累计概率越过阈值的 token 仍被保留。
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                # scatter 把排序空间的 mask 放回原词表位置，再将被过滤 logits 设为 -inf。
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')

            # 采样模式按 softmax 概率抽取；非采样模式直接选择最大 logit（贪心解码）。
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)

            # 已经结束的序列后续固定追加 EOS，保证并行 batch 的形状一致。
            if eos_token_id is not None:
                next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)

            # 把本轮生成结果拼到完整序列尾部，成为下一轮上下文。
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            # 启用缓存时保存本轮输出的各层 K/V；关闭时下一轮仍从头计算全部序列。
            past_key_values = outputs.past_key_values if use_cache else None

            # 流式输出器每轮接收刚生成的 token；转到 CPU 便于解码和展示。
            if streamer:
                streamer.put(next_token.cpu())

            # 配置了 EOS 时，更新各序列完成状态，并在全部完成后提前结束循环。
            if eos_token_id is not None:
                # 已完成状态与“本轮 token 等于 EOS”做逻辑或，完成后不会重新变回 False。
                finished |= next_token.squeeze(-1).eq(eos_token_id)

                # batch 中所有候选都生成 EOS 后，无需继续到 max_new_tokens。
                if finished.all():
                    break

        # 告知 streamer 生成结束，使其刷新缓冲区并完成输出。
        if streamer:
            streamer.end()

        # 某些调用方还想继续生成时，可以同时取回最终 token IDs 与 KV Cache。
        if kwargs.get("return_kv"):
            return {'generated_ids': input_ids, 'past_kv': past_key_values}

        # 默认只返回包含 prompt 和所有新 token 的完整序列。
        return input_ids
