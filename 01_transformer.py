"""
============================================================
Day 1: Transformer 架构 — 从零理解 + 动手实现
目标：理解 Attention 机制和 LLaMA 架构的核心组件
时间：约 4-6 小时

运行方式:
    conda activate llmrec
    python 01_transformer.py

理论参考:
    - The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
    - LLaMA 论文: https://arxiv.org/abs/2302.13971
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ════════════════════════════════════════════════════════════
# Part 1: Token → Embedding
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Part 1: Token → Embedding")
print("="*60)

vocab_size = 10000
embed_dim = 512

embedding = nn.Embedding(vocab_size, embed_dim)
# 模拟："推荐系统 很 有趣" → 3 个 token id
input_ids = torch.tensor([[42, 789, 3215]])
embedded = embedding(input_ids)

print(f"输入 token ids: {input_ids}")
print(f"输入 shape:      {input_ids.shape}")    # (1, 3)
print(f"Embedding 后:    {embedded.shape}")     # (1, 3, 512)
print("[Tip] Embedding = 查表，每个 token id 映射到一个向量")


# ════════════════════════════════════════════════════════════
# Part 2: Self-Attention（核心！）
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Part 2: Self-Attention — 注意力机制")
print("="*60)

"""
Self-Attention 直观理解:
    句子 "推荐系统 很 有趣，因为 它 能 猜中 用户 喜好"
    当处理"它"时，Attention 会自动把高权重分配给"推荐系统"

    公式: Attention(Q,K,V) = softmax(Q·K^T / √d_k) · V

    - Q (Query):  "我在找什么？"
    - K (Key):    "我能提供什么？"
    - V (Value):  "我包含什么信息？"
"""


class SelfAttention(nn.Module):
    """最简单头自注意力"""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 注意力分数 + 缩放
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.embed_dim)
        attn_weights = F.softmax(scores, dim=-1)
        output = attn_weights @ V
        return output, attn_weights


# 测试
x = torch.randn(1, 4, 512)  # 4 个 token
sa = SelfAttention(512)
output, weights = sa(x)

print(f"输入:  {x.shape} → 输出: {output.shape}")
print("\n注意力权重矩阵 (4个token互相之间的关注度):")
print(torch.round(weights[0].detach(), decimals=3))
print("[Tip] 每行加和 = 1，代表该 token 对所有 token 的注意力分配")


# ════════════════════════════════════════════════════════════
# Part 3: Multi-Head Attention
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Part 3: Multi-Head Attention")
print("="*60)

"""
为什么多头？
  单头 → 只能学到一种关系模式
  多头 → 同时从多个角度关注（语法、语义、位置...）

LLaMA 使用 GQA (Grouped-Query Attention):
  Q 有多个头，K,V 共享 → 省显存，快推理
"""


class MultiHeadAttention(nn.Module):
    """多头自注意力 + Causal Mask"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        B, seq_len, _ = x.shape

        # 投影 → 拆成多头: (B, heads, seq, head_dim)
        Q = self.q_proj(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention + Causal Mask（不能偷看未来 token！）
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask.to(scores.device), float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)

        # 合并多头 → 输出投影
        out = attn_weights @ V
        out = out.transpose(1, 2).contiguous().view(B, seq_len, self.embed_dim)
        return self.o_proj(out)


mha = MultiHeadAttention(512, num_heads=8)
x = torch.randn(1, 6, 512)
print(f"输入: {x.shape} → 输出: {mha(x).shape}")
print(f"每个头维度: {512//8}=64")
print("[Tip] Causal Mask 确保 token_i 只能看到 0~i，不能偷看未来")


# ════════════════════════════════════════════════════════════
# Part 4: RoPE — LLaMA 的位置编码
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Part 4: RoPE (Rotary Position Embedding)")
print("="*60)

"""
LLaMA 不用传统的 sin/cos 绝对位置编码，而用 RoPE。
核心想法：在计算 Q·K 之前，把向量按位置"旋转"一个角度。

两个 token 旋转后的内积只取决于它们的相对距离，
天然感知"谁离谁近"，且能外推到更长序列。

这对比赛至关重要：用户行为序列动辄几十上百个 token！
"""


def precompute_rope_freqs(head_dim, seq_len, theta=10000.0):
    """预计算 RoPE 频率"""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(seq_len).float()
    angles = torch.outer(positions, freqs)
    cos = angles.cos().unsqueeze(0).unsqueeze(0)
    sin = angles.sin().unsqueeze(0).unsqueeze(0)
    return cos, sin


print("[Tip] RoPE 核心:")
print("   → 不是把位置'加'到 embedding，而是'旋转' Q,K 向量")
print("   → Q·K 只取决于相对距离，天然理解位置关系")
print("   → 可外推到训练时没见过的序列长度")


# ════════════════════════════════════════════════════════════
# Part 5: 完整 Transformer Block（LLaMA 风格）
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Part 5: LLaMA Transformer Block")
print("="*60)

"""
LLaMA 的每个 Block:
    x → RMSNorm → MultiHeadAttention(+RoPE) → +x (残差)
      → RMSNorm → SwiGLU FFN → +x (残差)

相比原始 Transformer 的改进:
    ① RMSNorm 代替 LayerNorm（更快）
    ② SwiGLU 代替 ReLU（更强）
    ③ RoPE 代替绝对位置编码（更灵活）
    ④ Pre-Norm 代替 Post-Norm（更稳定）
"""


class RMSNorm(nn.Module):
    """RMS Norm — LLaMA 使用"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class SwiGLU(nn.Module):
    """SwiGLU FFN — LLaMA 使用"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        # SwiGLU(x) = (W1(x) ⊙ SiLU(W3(x))) · W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LLaMABlock(nn.Module):
    """LLaMA 的单个 Decoder Block"""
    def __init__(self, dim, num_heads, hidden_dim):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attention = MultiHeadAttention(dim, num_heads)
        self.ffn_norm = RMSNorm(dim)
        self.feed_forward = SwiGLU(dim, hidden_dim)

    def forward(self, x):
        # Pre-Norm + Residual
        x = x + self.attention(self.attn_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


block = LLaMABlock(dim=512, num_heads=8, hidden_dim=1376)
x = torch.randn(1, 8, 512)
print(f"输入: {x.shape} → 输出: {block(x).shape}")
print(f"参数量: {sum(p.numel() for p in block.parameters()):,}")


# ════════════════════════════════════════════════════════════
# Part 6: 全景图
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Part 6: LLaMA 完整架构")
print("="*60)

print("""
LLaMA 模型 = Token Embedding
           + 32 层 LLaMABlock     (LLaMA-7B)
           + RMSNorm
           + LM Head → 输出词表概率

Qwen2.5-7B (比赛首选基座模型):
  → 28 层, dim=3584, 28 头
  → 中文能力强，社区活跃
  → 4-bit 量化后 ~4GB，RTX 4060 能跑

比赛与这四个组件的对应关系:
┌──────────────────┬─────────────────────────┐
│ 比赛任务          │ 关键组件                 │
├──────────────────┼─────────────────────────┤
│ 物料理解          │ Multi-Head Attention    │
│ (多模态→文本)     │ → 跨模态注意力           │
├──────────────────┼─────────────────────────┤
│ 用户兴趣演化      │ RoPE + Causal Mask      │
│ (行为序列→演化链)  │ → 序列建模能力           │
├──────────────────┼─────────────────────────┤
│ 推荐物料          │ SwiGLU FFN              │
│ (跨域预测偏好)     │ → 非线性特征交互          │
├──────────────────┼─────────────────────────┤
│ 常识问答          │ 整个架构                 │
│ (推理+选择)       │ → 世界知识 = 训练数据     │
└──────────────────┴─────────────────────────┘

[Next] Step 2: 运行 02_load_qwen.py 加载真实模型！
""")
