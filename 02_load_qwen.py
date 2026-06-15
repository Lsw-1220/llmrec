"""
============================================================
Day 2: 加载 OneReason-0.8B 模型 + 推理实战
目标：学会从 HuggingFace 加载模型、理解 Tokenizer、完成推理
时间：约 3-4 小时

运行方式:
    conda activate llmrec
    python 02_load_qwen.py

理论参考:
    - OneReason 技术报告: https://arxiv.org/abs/2606.06260
    - OpenOneRec GitHub: https://github.com/Kuaishou-OneRec/OpenOneRec
    - BitsAndBytes 4-bit: https://huggingface.co/docs/bitsandbytes

模型: OneReason-0.8B-pretrain (快手 OneRec 团队)
    - 基于 Qwen3 架构的推荐系统推理基座模型
    - Fast-Slow Thinking 架构
    - 0.8B 参数, bf16 ≈ 1.6GB, 4bit ≈ 0.4GB
    - RTX 4060 8GB 绰绰有余！

比赛要求: 使用 OneReason-0.8B-pretrain 作为基座 → 进行 SFT 微调
============================================================
"""

import torch
import os

print(f"PyTorch: {torch.__version__}")
print(f"CUDA:    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print(f"显存:    {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"空闲:    {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

# ════════════════════════════════════════════════════════════
# Part 0: 配置 — 选模型 + 设路径
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Part 0: 模型选择与路径配置")
print("="*60)

# 三个可选模型（uncomment 你想用的那个）
# 比赛官方基座模型 (本地已下载)
import os
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OneReason-0.8B-pretrain-competition")
# 如果要用 HF 在线版: MODEL_PATH = "OpenOneRec/OneReason-0.8B-pretrain-competition"

# 模型下载到哪里？（默认 ~/.cache/huggingface/hub/）
# 想存到 D 盘的话，取消下行注释：
# os.environ["HF_HOME"] = "D:/huggingface_models"

print(f"模型:  {MODEL_PATH}")
print(f"HF缓存: {os.environ.get('HF_HOME', '~/.cache/huggingface/hub/')}")
print("[Tip] 首次运行会自动从 HuggingFace 下载模型权重")
print("[Tip] Qwen 需要联网认证，如果下载失败请运行: huggingface-cli login")

# ════════════════════════════════════════════════════════════
# Part 1: Tokenizer — 把文本变成模型能理解的 Token
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Part 1: Tokenizer 原理与使用")
print("="*60)

"""
Tokenizer 做什么？
    文本 → Tokenizer → Token IDs → Embedding → 模型
    "推荐系统很酷" → [42, 789, 3215, 9876] → 4个向量 → Transformer

OneReason 使用 Qwen3 的 BPE tokenizer
    - vocab_size ≈ 151643 (同 Qwen 系列)
    - 中文一个汉字 ≈ 1-2 个 token
    - 英文一个单词 ≈ 1-3 个 token
    - 推荐系统专用 token（如 item id、用户行为等）

Chat Template 注意：
    OneReason-0.8B-pretrain 是预训练基座，不是 Instruct 模型。
    - 可能没有 chat_template（需要我们自己构建）
    - 或者有简单的 chat_template（继承自 Qwen3）

    推荐场景的数据格式通常是:
      [用户历史行为序列] + [候选物料] → [模型预测]
    而不是传统的"system/user/assistant"对话格式。
"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 1.1 基础 tokenization
text = "推荐系统很有趣"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print(f"\n原始文本: '{text}'")
print(f"Tokens ({len(tokens)}): {tokens}")
print(f"Token IDs ({len(token_ids)}): {token_ids}")
print(f"Decode 验证: '{tokenizer.decode(token_ids)}'")

# 1.2 中英文的 token 效率对比
cn_text = "快手探索者LLM推荐系统挑战赛"
en_text = "Kuaishou Explorer LLM Recommendation Challenge"

cn_ids = tokenizer.encode(cn_text)
en_ids = tokenizer.encode(en_text)
print(f"\nToken 效率对比:")
print(f"  中文: '{cn_text}' → {len(cn_ids)} tokens (每字 ~{len(cn_ids)/len(cn_text):.1f} tokens)")
print(f"  英文: '{en_text}' → {len(en_ids)} tokens (每词 ~{len(en_ids)/len(en_text.split()):.1f} tokens)")
print(f"  [Tip] 中文 token 效率较低，中文一条 prompt 可能比英文贵 2-3 倍")

# 1.3 特殊 token
print(f"\n特殊 Token:")
print(f"  PAD token:    {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
print(f"  EOS token:    {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
print(f"  BOS token:    {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
print(f"  Vocab size:   {tokenizer.vocab_size}")

# 1.4 Chat Template 演示
print(f"\nChat Template 演示:")
print(f"  Chat template 是否存在: {tokenizer.chat_template is not None}")

if tokenizer.chat_template is not None:
    # 构建一条对话
    messages = [
        {"role": "system", "content": "你是一个推荐系统助手"},
        {"role": "user", "content": "推荐一款适合学生的笔记本电脑"},
    ]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,    # 只看文本，不 tokenize
        add_generation_prompt=True
    )
    print(f"\n格式化后的 prompt (前200字):")
    print(f"  {formatted[:200]}...")

    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    print(f"\n返回类型: {type(tokenized_chat).__name__}")
    print(f"Tokenized shape: {tokenized_chat.input_ids.shape}  (batch, seq_len)")
    print(f"前10个 token ids: {tokenized_chat.input_ids[0][:10].tolist()}")
    print(f"后10个 token ids: {tokenized_chat.input_ids[0][-10:].tolist()}")
else:
    print("\n[注意] OneReason-pretrain 没有 chat_template，这是正常的！")
    print("  预训练基座模型不包含对话格式，我们需要在 SFT 阶段训练它。")
    print("  当前直接 tokenize 文本即可：")
    sample = tokenizer("推荐一款适合学生的笔记本电脑", return_tensors="pt")
    print(f"  直接 tokenize → shape: {sample.input_ids.shape}")
    print(f"  Token IDs: {sample.input_ids[0][:15].tolist()}...")
    tokenized_chat = sample  # 用简单 tokenize 的结果

# ════════════════════════════════════════════════════════════
# Part 2: 加载模型
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Part 2: 加载 OneReason-0.8B 模型")
print("="*60)

"""
显存分析 (OneReason-0.8B):
    bf16 全精度:  0.8B × 2 bytes = 1.6 GB
    4-bit 量化:   0.8B × 0.5 bytes = 0.4 GB

    RTX 4060 8GB 完全够用 bf16，无需量化！
    但作为学习，我们仍演示量化的使用方式。

    OneReason 相比通用 LLM 的区别:
      - 预训练数据是推荐场景的交互序列（用户→视频/商品）
      - Item Tokenizer: 把视频/商品 ID 也变成 token
      - 理解的不只是自然语言，还有用户行为模式
"""

from transformers import AutoModelForCausalLM

print(f"\n正在加载模型: {MODEL_PATH}")
print("=" * 40)

# 0.8B 模型直接用 bf16，不用量化
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",          # 自动分配层到 GPU
    torch_dtype=torch.bfloat16,  # bf16 省一半显存
    trust_remote_code=True,     # OneReason 可能需要自定义代码
)

print(f"\n[配置] bf16 全精度加载 (0.8B ≈ 1.6GB，显存充足)")

# ════════════════════════════════════════════════════════════
# Part 2.5 (可选): 如果你以后想用 4-bit 量化加载 7B 模型
# ════════════════════════════════════════════════════════════
"""
如果以后比赛发更大的模型（如 OneReason-7B），可以这样加载:

from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 4-bit 量化
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时用 bf16
    bnb_4bit_use_double_quant=True,         # 双重量化
    bnb_4bit_quant_type="nf4",              # NormalFloat4
)
model = AutoModelForCausalLM.from_pretrained(
    "OpenOneRec/OneReason-7B-pretrain",
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
"""

# ════════════════════════════════════════════════════════════
# Part 3: 模型信息 — 看看里面有什么
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Part 3: 模型架构信息")
print("="*60)

# 获取模型配置
config = model.config
print(f"\n模型架构: {config.model_type}")
print(f"层数:     {config.num_hidden_layers}")
print(f"隐藏维度:  {config.hidden_size}")
print(f"注意力头:  {config.num_attention_heads}")
print(f"词表大小:  {config.vocab_size}")
print(f"最大位置:  {config.max_position_embeddings}")
print(f"中间维度:  {config.intermediate_size}")

# 参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数量:   {total_params / 1e9:.2f}B")
print(f"可训练参数: {trainable_params / 1e6:.1f}M")

# 显存使用
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"\nGPU 显存使用:")
    print(f"  已分配: {allocated:.2f} GB")
    print(f"  已保留: {reserved:.2f} GB")
    print(f"  空闲:   {torch.cuda.get_device_properties(0).total_memory / 1024**3 - reserved:.2f} GB")

# ════════════════════════════════════════════════════════════
# Part 4: 推理 — 让模型生成回复
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Part 4: 推理生成 (Inference)")
print("="*60)

"""
生成策略一览:

1. Greedy (贪心):
   每步选概率最高的 token → 确定性输出，适合"标准答案"场景

2. Sampling (采样):
   按概率分布随机选 → 多样性强，适合创意场景
   参数 temperature 控制"随机程度"
     - T=0.1: 接近贪心，稳定
     - T=0.7: 平衡
     - T=2.0: 非常随机，可能胡言乱语

3. Top-p (Nucleus Sampling):
   只从累积概率 ≤ p 的 token 中选 → 排除小概率垃圾

4. Top-k:
   只从概率最高的 k 个 token 中选

推荐场景通常用:
  - T=0.1~0.3, top_p=0.95 → 稳定但有余地
  - 或者直接用贪心 (do_sample=False)
"""

# 准备输入（兼容无 chat_template 的 pretrain 模型）
test_messages = [
    {"role": "system", "content": "你是一个推荐系统助手。"},
    {"role": "user", "content": "我喜欢玩开放世界动作游戏，推荐三款类似的游戏"},
]

if tokenizer.chat_template is not None:
    inputs = tokenizer.apply_chat_template(
        test_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
else:
    # pretrain 模型没有 chat template，手动拼接 prompt
    prompt_text = "你是一个推荐系统助手。\n用户：我喜欢玩开放世界动作游戏，推荐三款类似的游戏\n回答："
    inputs = tokenizer(prompt_text, return_tensors="pt")

# 把 input 移到模型所在设备
device = next(model.parameters()).device
inputs = {k: v.to(device) for k, v in inputs.items()}

print(f"\n输入 prompt (decoded):")
print(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)[:300])

print(f"\n输入长度: {inputs['input_ids'].shape[1]} tokens")
print(f"正在生成...")

# 生成
with torch.inference_mode():  # 比 torch.no_grad() 更安全
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,        # 最多生成 256 个新 token
        do_sample=True,            # 使用采样（非贪心）
        temperature=0.7,           # 中等随机度
        top_p=0.95,               # Nucleus sampling
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,  # 遇到 EOS 就停
    )

# 解码输出
generated_ids = outputs[0][inputs["input_ids"].shape[1]:]  # 只取新生成的部分
response = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(f"\n{'─'*50}")
print(f"模型回复 (生成了 {len(generated_ids)} tokens):")
print(response)
print(f"{'─'*50}")

# ════════════════════════════════════════════════════════════
# Part 5: 不同生成策略对比
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Part 5: 生成策略对比")
print("="*60)

# 准备一个推荐相关的测试 prompt
short_messages = [
    {"role": "system", "content": "你是一个推荐系统助手。"},
    {"role": "user", "content": "推荐一部科幻电影，用一句话回答。"},
]

if tokenizer.chat_template is not None:
    short_inputs = tokenizer.apply_chat_template(
        short_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
else:
    prompt_text = "你是一个推荐系统助手。\n用户：推荐一部科幻电影，用一句话回答。\n回答："
    short_inputs = tokenizer(prompt_text, return_tensors="pt")
short_inputs = {k: v.to(device) for k, v in short_inputs.items()}

strategies = [
    {"name": "Greedy (T=0)",        "do_sample": False},
    {"name": "Sample T=0.3",        "do_sample": True, "temperature": 0.3, "top_p": 0.95},
    {"name": "Sample T=0.7",        "do_sample": True, "temperature": 0.7, "top_p": 0.95},
    {"name": "Sample T=1.5 (wild)", "do_sample": True, "temperature": 1.5, "top_p": 0.95},
]

print("\n同一 prompt，不同策略的回复:\n")
for strat in strategies:
    with torch.inference_mode():
        out = model.generate(
            **short_inputs,
            max_new_tokens=64,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **{k: v for k, v in strat.items() if k != "name"}
        )
    generated = out[0][short_inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    # 只取第一行
    text = text.split("\n")[0]
    print(f"  {strat['name']:20s} → {text[:80]}")

print(f"\n[Tip] 推荐任务建议 T=0.1~0.2，保证输出稳定可复现")
print(f"[Tip] 创意任务 (如生成推荐理由) 建议 T=0.6~0.8")

# ════════════════════════════════════════════════════════════
# Part 6: 模型内部 — 看看 Logits 和 Hidden States
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Part 6: 模型内部 — Logits 与 Hidden States")
print("="*60)

"""
了解模型"思考过程"对比赛很重要：
  - Logits: 词表上每个 token 的"得分"，softmax 后变概率
  - Hidden States: 每层 Transformer 的中间表示
  - Attention Weights: 模型在"关注"什么 token
"""

# 6.1 直接调用 forward (不 generate)
simple_text = "推荐一部电影"
simple_inputs = tokenizer(simple_text, return_tensors="pt").to(device)

with torch.inference_mode():
    # output_hidden_states=True 返回所有层的 hidden states
    # output_attentions=True 返回所有层的 attention weights
    model_output = model(
        **simple_inputs,
        output_hidden_states=True,
        output_attentions=True,
    )

print(f"\n输入: '{simple_text}' → {simple_inputs['input_ids'].shape[1]} tokens")

# Logits: 最后位置的 logits 就是"下一个 token 的预测"
logits = model_output.logits  # (batch, seq_len, vocab_size)
last_token_logits = logits[0, -1, :]  # 最后位置的预测
top5_probs, top5_ids = torch.topk(torch.softmax(last_token_logits, dim=-1), k=5)

print(f"\n最后一个位置预测的下一个 token (Top-5):")
for i, (prob, tok_id) in enumerate(zip(top5_probs, top5_ids)):
    tok = tokenizer.decode([tok_id])
    print(f"  {i+1}. '{tok}' → {prob:.4f}")

# Hidden States: 每层 Transformer 的输出
hidden_states = model_output.hidden_states  # tuple of (num_layers+1) tensors
print(f"\nHidden States:")
print(f"  层数: {len(hidden_states)} (含 Embedding 层)")
print(f"  每层 shape: {hidden_states[0].shape}  (batch, seq_len, hidden_dim)")
print(f"  [Tip] hidden_states[0] = Embedding 输出")
print(f"  [Tip] hidden_states[-1] = 最后一层输出 (直接喂给 LM Head)")

# Attention Weights: 看模型在关注什么
attentions = model_output.attentions  # tuple of num_layers tensors (可能为空)
if attentions and len(attentions) > 0:
    avg_attn = attentions[-1].mean(dim=1)
    print(f"\n最后一层 Attention (avg over heads):")
    print(f"  Shape: {avg_attn.shape}  (batch, seq_len, seq_len)")
    print(f"  (这就是 Day 1 里我们手写的那个 attention weights!)")

    tokens = [tokenizer.decode([tid]) for tid in simple_inputs['input_ids'][0]]
    print(f"\n  Tokens: {tokens}")
    print(f"  Attention 矩阵 (每个 token 对各个位置的关注度):")
    for i, tok in enumerate(tokens):
        attn_row = avg_attn[0, i, :].cpu()
        attention_str = " ".join([f"{w:.3f}" for w in attn_row])
        print(f"    '{tok:8s}' → [{attention_str}]")
else:
    print(f"\n[提示] SDPA attention 不支持返回 attention weights")
    print(f"  这是因为 sdpa 使用 Flash Attention 实现，不保存中间权重矩阵。")
    print(f"  这是推理速度优化的代价——想要看 Attention 需要切换到 eager 模式。")
    print(f"  切换方法: model = AutoModelForCausalLM.from_pretrained(..., attn_implementation='eager')")

# ════════════════════════════════════════════════════════════
# Part 7: 推荐场景实战 — 模拟物料理解任务
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Part 7: 推荐场景实战 — 物料理解")
print("="*60)

"""
比赛任务 1：物料理解
  - 输入：视频的标题、标签、简介等文本描述
  - 输出：视频的类别、标签、适用人群等结构化理解

下面模拟这个场景，让模型理解一个"视频物料"。
"""

# 模拟一条视频物料描述
video_material = """
视频标题: 牛肉拉面教程
视频描述: 兰州师傅教你正宗牛肉拉面做法，从和面到拉面全过程详解
视频标签: #美食 #面食 #拉面 #教程 #兰州
视频长度: 5分30秒
"""

material_prompt = [
    {"role": "system", "content": """你是一个视频内容理解助手。
请分析给定的视频物料，输出：
1. 视频类型
2. 目标观众
3. 3个关键词
4. 适合推荐的场景（如：晚饭不知道吃什么、想学做饭等）
用简洁的 JSON 格式回复。"""},
    {"role": "user", "content": f"请分析这个视频物料：{video_material}"},
]

if tokenizer.chat_template is not None:
    material_inputs = tokenizer.apply_chat_template(
        material_prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)
else:
    sys_prompt = material_prompt[0]["content"]
    user_prompt = material_prompt[1]["content"]
    prompt_text = f"{sys_prompt}\n\n输入：{user_prompt}\n分析结果："
    material_inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

with torch.inference_mode():
    # **material_inputs 解包 BatchEncoding → input_ids + attention_mask
    material_output = model.generate(
        **material_inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.2,  # 低温度，保证稳定
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

material_response = tokenizer.decode(
    material_output[0][material_inputs.input_ids.shape[1]:],
    skip_special_tokens=True
)

print(f"\n{'─'*50}")
print(f"物料分析结果:")
print(material_response)
print(f"{'─'*50}")

# ════════════════════════════════════════════════════════════
# Part 8: 实战技巧 — Batch 推理与显存管理
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Part 8: 实战技巧 — Batch 推理")
print("="*60)

"""
比赛数据通常有几万条，一条条推理太慢。Batch 推理能提速 3-8 倍。

但 Batch 也吃显存：
  - batch=1:  KV Cache ≈ 1.5GB (seq=2048)
  - batch=4:  KV Cache ≈ 6.0GB (seq=2048)
  - batch=8:  KV Cache ≈ 12GB → 炸显存！

对于 8GB 显卡，推荐 batch=2~4。

技巧：Left-padding
  正常对话是 Right-padding (pad 在右边)
  但生成时应该 Left-padding (pad 在左边)，
  这样生成的 token 都从右边开始，模型不受 pad 干扰。
"""

# 演示 batch 推理
batch_messages = [
    [{"role": "user", "content": "推荐一部动作电影"}],
    [{"role": "user", "content": "推荐一本推理小说"}],
]

# 手动 tokenize 后 padding
tokenizer.padding_side = "left"  # 关键：生成时用 left padding
if tokenizer.chat_template is not None:
    batch_inputs = tokenizer.apply_chat_template(
        batch_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        return_dict=True,
    ).to(device)
else:
    # 手动构建 batch prompt
    batch_texts = ["推荐一部动作电影\n回答：", "推荐一本推理小说\n回答："]
    batch_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True).to(device)

print(f"\nBatch 推理: {len(batch_messages)} 条 prompt")
print(f"  Input shape:      {batch_inputs['input_ids'].shape}")
print(f"  Attention mask:   {batch_inputs['attention_mask'].shape}")
print(f"  (pad_token_id={tokenizer.pad_token_id}, pad side=left)")

# 验证：看两条 prompt 的 pad 情况
for i in range(len(batch_messages)):
    ids = batch_inputs["input_ids"][i]
    mask = batch_inputs["attention_mask"][i]
    real_tokens = mask.sum().item()
    pad_tokens = ids.shape[0] - real_tokens
    print(f"  Sample {i}: {pad_tokens} pad tokens + {real_tokens} real tokens = {ids.shape[0]} total")

# Batch 生成
with torch.inference_mode():
    batch_outputs = model.generate(
        batch_inputs["input_ids"],
        attention_mask=batch_inputs["attention_mask"],
        max_new_tokens=64,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

print(f"\nBatch 生成结果:")
for i in range(len(batch_messages)):
    response = tokenizer.decode(
        batch_outputs[i][batch_inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    print(f"  [{i+1}] {response[:100].strip()}...")

# ════════════════════════════════════════════════════════════
# 总结
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Day 2 总结")
print("="*60)

print("""
今天学到的:
  1. Tokenizer — 文本↔Token IDs 的双向转换，Chat Template 格式化对话
  2. 加载模型 — AutoModelForCausalLM + device_map="auto"
  3. 推理生成 — generate() 的各种参数 (temperature, top_p, top_k...)
  4. 模型内部 — Logits、Hidden States、Attention Weights
  5. Batch 推理 — Left-padding + attention_mask
  6. Pretrain vs Instruct — 基座模型与对话模型的核心区别

你的比赛模型:
  模型: OpenOneRec/OneReason-0.8B-pretrain
  路径: ~/.cache/huggingface/hub/models--OpenOneRec--OneReason-0.8B-pretrain/
  架构: Qwen3-based, 0.8B 参数 → bf16 只需 1.6GB 显存

比赛四维能力对应:
  ┌──────────────────┬─────────────────────────┐
  │ 懂物料             │ Part 7 物料理解 prompt   │
  │ 懂用户             │ 长序列 tokenize + 推理   │
  │ 懂推荐             │ Batch 推理 + 生成策略    │
  │ 懂世界             │ Greedy 解码 + 常识问答   │
  └──────────────────┴─────────────────────────┘

[Next] Step 3: 运行 03_lora_sft.py 学习 LoRA 微调！
""")
