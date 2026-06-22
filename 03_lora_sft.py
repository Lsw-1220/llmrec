"""
============================================================
Day 3: LoRA/QLoRA 微调 — 基于真实比赛数据格式
目标：适配 dataexample.txt 中的 13 种任务，完成 LoRA SFT
时间：约 4-6 小时

运行方式:
    conda activate llmrec
    python 03_lora_sft.py                      # 默认参数
    python 03_lora_sft.py --lr 1e-4 --epochs 3  # 自定义超参
    python 03_lora_sft.py --lora_r 16           # 更高 LoRA 秩

数据来源: dataexample.txt（包含四维能力共 13 种任务类型）
============================================================
"""
import torch
import os
import json
import re
import glob
import argparse
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# ════════════════════════════════════════════════════════════
# CLI 参数（比赛调参用）
# ════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="LoRA SFT 微调 — LLMRec 挑战赛")
parser.add_argument("--lr", type=float, default=5e-5, help="学习率 (默认 5e-5)")
parser.add_argument("--epochs", type=int, default=5, help="训练轮数 (默认 5)")
parser.add_argument("--max_len", type=int, default=2048, help="最大序列长度 (默认 2048)")
parser.add_argument("--grad_accum", type=int, default=8, help="梯度累积步数 (默认 8)")
parser.add_argument("--lora_r", type=int, default=8, help="LoRA 秩 (默认 8)")
parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha (默认 16)")
args = parser.parse_args()

# ════════════════════════════════════════════════════════════
# 环境检查
# ════════════════════════════════════════════════════════════
print(f"PyTorch: {torch.__version__}")
print(f"CUDA:    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    free = (torch.cuda.get_device_properties(0).total_memory -
            torch.cuda.memory_allocated()) / 1024**3
    print(f"显存空闲: {free:.1f} GB")

# ════════════════════════════════════════════════════════════
# Part 0: LoRA 原理
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part 0: LoRA 核心原理")
print("=" * 60)
print("""
LoRA = 冻结模型 + 低秩适配矩阵 A×B
0.8B 全量微调 ~8.4GB vs LoRA ~3.6GB → 8GB 显卡刚好够

关键概念:
  - r (秩): 低秩矩阵的维度，r=8 时每个适配器只有 2×r×dim 个参数
  - alpha: 缩放因子，实际缩放 = alpha/r，alpha=16, r=8 → 缩放 2×
  - target_modules: 要替换的线性层（Q/K/V/O + FFN 三门）
  - label 遮蔽: SFT 只对 assistant 回复算 loss，prompt 部分设为 -100
""")

# ════════════════════════════════════════════════════════════
# Part 1: 加载并解析真实比赛数据
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part 1: 解析 dataexample.txt")
print("=" * 60)

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataexample.txt")

# 四个维度的 section 标题（中文冒号或英文冒号）
SECTIONS = [
    ("物料", r"物料数据示例[：:]"),
    ("用户", r"用户数据示例[：:]"),
    ("推荐", r"推荐数据示例[：:]"),
    ("世界", r"世界知识数据示例[：:]"),
]


def parse_dataexample(filepath):
    """
    解析 dataexample.txt 为统一的 SFT 训练格式。

    文件结构：4 个 section，每个以"XX数据示例："开头，后面跟一个 JSON 对象。
    每个 JSON 对象的顶层 key 是任务类型名，value 包含 messages + metadata。

    已知的数据不一致：
      - content 字段可能是纯字符串或 [{"type":"text","text":"..."}] 列表
      - messages 字段可能是列表或 JSON 字符串（推荐数据）
      - metadata 字段可能是字典或 JSON 字符串
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    dataset = []
    task_counter = {name: 0 for name, _ in SECTIONS}

    for sec_name, sec_pattern in SECTIONS:
        # 找到该 section 的起始位置
        match = re.search(sec_pattern, content)
        if not match:
            print(f"  [警告] 未找到 section: {sec_name}")
            continue

        # section 内容 = 从匹配结束到下一个 section 开头（或文件末尾）
        start = match.end()
        # 找下一个 section 的位置
        next_match = None
        for _, next_pattern in SECTIONS:
            nm = re.search(next_pattern, content[start:])
            if nm:
                candidate = start + nm.start()
                if next_match is None or candidate < next_match:
                    next_match = candidate

        sec_text = content[start:next_match].strip() if next_match else content[start:].strip()

        if not sec_text:
            print(f"  [警告] {sec_name} section 为空")
            continue

        # 解析 JSON
        try:
            tasks = json.loads(sec_text)
        except json.JSONDecodeError as e:
            print(f"  [警告] {sec_name} section JSON 解析失败: {e}")
            print(f"         前 100 字符: {sec_text[:100]}")
            continue

        for task_name, task_data in tasks.items():
            # ── 提取 messages ──
            raw_msgs = task_data.get("messages")
            if raw_msgs is None:
                print(f"  [跳过] {sec_name}/{task_name}: 无 messages 字段")
                continue

            if isinstance(raw_msgs, str):
                try:
                    messages = json.loads(raw_msgs)
                except json.JSONDecodeError as e:
                    print(f"  [跳过] {sec_name}/{task_name}: messages JSON 解析失败: {e}")
                    continue
            elif isinstance(raw_msgs, list):
                messages = raw_msgs
            else:
                print(f"  [跳过] {sec_name}/{task_name}: 未知 messages 类型 {type(raw_msgs)}")
                continue

            # ── 提取 answer ──
            metadata = task_data.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {"answer": metadata}
            answer = metadata.get("answer", "")

            if not answer:
                print(f"  [跳过] {sec_name}/{task_name}: 无 answer")
                continue

            # ── 转换为统一的 ChatML 格式 ──
            # content 字段可能是 "text" 或 [{"type":"text","text":"..."}]
            converted = []
            for msg in messages:
                msg_content = msg.get("content", "")

                # 如果是列表格式，提取 text
                if isinstance(msg_content, list):
                    texts = []
                    for block in msg_content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            texts.append(block["text"])
                    msg_content = "\n".join(texts)

                converted.append({
                    "role": msg["role"],
                    "content": msg_content,
                })

            # 添加 assistant 回答
            converted.append({
                "role": "assistant",
                "content": answer,
            })

            dataset.append({
                "task": task_name,
                "category": sec_name,
                "messages": converted,
            })
            task_counter[sec_name] += 1

    return dataset, task_counter


dataset, counter = parse_dataexample(DATA_FILE)

print(f"\n加载了 {len(dataset)} 条训练样本:")
for cat, count in counter.items():
    print(f"  {cat}: {count} 条")

# 列出所有任务类型
task_types = sorted(set(d["task"] for d in dataset))
print(f"\n任务类型 ({len(task_types)} 种): {task_types}")

# ════════════════════════════════════════════════════════════
# Part 2: Tokenizer + 数据 Tokenize
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part 2: 将数据转换为模型可用格式")
print("=" * 60)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "OneReason-0.8B-pretrain-competition")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 注意：不要覆盖 pad_token！
# OneReason 的原始配置：pad_token=<|endoftext|> (id 151643), eos_token=<|im_end|> (id 151645)
# 之前错误地将 pad_token 设为 eos_token，会导致训练和推理不一致
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"[注意] tokenizer 无 pad_token，已设为 eos_token: {tokenizer.pad_token}")
else:
    print(f"Pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"EoS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")

# OneReason 的 thinking 模式说明:
#   chat_template 支持 enable_thinking 参数
#   - enable_thinking=True: 模型会先 <think>...</think> 再输出答案（推理链）
#   - enable_thinking=False: 跳过思考，直接输出答案
#   训练数据中不包含 <think> 块，对应 non-thinking 模式
print("Thinking 模式: 训练数据为 non-thinking 格式（无 <think> 块）")

# 看看数据长什么样
for d in dataset:
    sample_text = tokenizer.apply_chat_template(
        d["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    if d["task"] in ("video", "Product", "single"):
        print(f"\n示例 [{d['category']}/{d['task']}]:")
        print(sample_text[:300])
        break

# 统计长度（帮助判断 max_len 是否够用）
def get_length(example):
    return len(tokenizer.apply_chat_template(
        example["messages"],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )[0])

lengths = [get_length(d) for d in dataset]
avg_len = sum(lengths) // len(lengths)
print(f"\n数据长度统计: min={min(lengths)}, max={max(lengths)}, avg={avg_len} tokens")
if max(lengths) > args.max_len:
    truncated = sum(1 for l in lengths if l > args.max_len)
    print(f"  ⚠ {truncated} 条样本超过 max_len={args.max_len}，将被截断！")

# ════════════════════════════════════════════════════════════
# Part 3: 加载模型 + 挂载 LoRA
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part 3: 加载模型 + 配置 LoRA")
print("=" * 60)

from transformers import AutoModelForCausalLM

print(f"加载基座模型: {MODEL_PATH}")
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
print(f"模型已加载, 显存: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ════════════════════════════════════════════════════════════
# Part 4: SFT 训练
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part 4: SFT 训练")
print("=" * 60)

# ── 使用 SFTTrainer（trl）───────────────────────────────────
# SFTTrainer 自动处理以下关键问题（手写循环容易出错）:
#   1. Label 遮蔽：只对 assistant 回复计算 loss，prompt 部分设为 -100
#   2. 梯度累积：多步累积再更新，等效更大 batch size
#   3. 混合精度：bf16 自动开启，减少显存占用
#   4. Checkpoint：按策略自动保存，训练中断可恢复
#   5. 数据 collation：自动 pad + 截断
# ─────────────────────────────────────────────────────────────

from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# 转换为 HuggingFace Dataset（SFTTrainer 需要）
train_dataset = Dataset.from_list(dataset)
train_dataset = train_dataset.shuffle(seed=42)

sft_config = SFTConfig(
    max_length=args.max_len,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=args.grad_accum,
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    bf16=True,
    warmup_ratio=0.05,
    logging_steps=1,
    save_strategy="epoch",
    save_total_limit=2,
    output_dir="./03_lora_adapter",
    report_to="none",
    # SFTTrainer 会自动用 tokenizer 的 chat_template 处理 messages 格式
    # 自动实现 label 遮蔽：prompt 部分的 labels 设为 -100
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

print(f"训练配置:")
print(f"  数据: {len(train_dataset)} 条")
print(f"  Epochs: {args.epochs}, 有效 batch: {1 * args.grad_accum}")
print(f"  学习率: {args.lr}, max_length: {args.max_len}")
print(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
print(f"  总步数: ~{len(train_dataset) * args.epochs // args.grad_accum}")

trainer.train()

print("训练完成！")

# ── 手写训练循环（教学注释）─────────────────────────────────
# 下面是 SFTTrainer 底层做的事情，保留作为理解参考：
#
# from torch.utils.data import DataLoader, Dataset as TorchDataset
# from torch.optim import AdamW
# from transformers import get_linear_schedule_with_warmup
#
# class SFTDataset(TorchDataset):
#     def __init__(self, data, tokenizer, max_length=2048):
#         self.input_ids = []
#         self.labels = []          # ← 关键：单独存 labels
#         for item in data:
#             text = tokenizer.apply_chat_template(
#                 item["messages"],
#                 tokenize=False,
#                 add_generation_prompt=False,
#             )
#             encoded = tokenizer(
#                 text, truncation=True, max_length=max_length,
#                 padding=False, return_tensors="pt",
#             )
#             input_ids = encoded["input_ids"][0]
#
#             # ★ Label 遮蔽：找到 assistant 回复的起始位置
#             # 只有 assistant 的回答部分参与 loss 计算
#             labels = input_ids.clone()
#             # 把 prompt 部分的 token 设为 -100（忽略）
#             # 方法：找到 <|im_start|>assistant\n 之后的位置
#             assistant_start = self._find_assistant_start(text, tokenizer, input_ids)
#             labels[:assistant_start] = -100
#
#             self.input_ids.append(input_ids)
#             self.labels.append(labels)
#
#     def _find_assistant_start(self, text, tokenizer, input_ids):
#         """找到 assistant 回复的起始 token 位置"""
#         # 在 tokenized 序列中找到最后一个 assistant 标记后的位置
#         assistant_token = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
#         for i in range(len(input_ids) - len(assistant_token) + 1):
#             if input_ids[i:i+len(assistant_token)].tolist() == assistant_token:
#                 return i + len(assistant_token)
#         return len(input_ids)  # fallback：全部遮蔽（不应该发生）
#
#     def __len__(self):
#         return len(self.input_ids)
#
#     def __getitem__(self, idx):
#         return {
#             "input_ids": self.input_ids[idx],
#             "attention_mask": torch.ones_like(self.input_ids[idx]),
#             "labels": self.labels[idx],     # ← 用遮蔽后的 labels
#         }
#
# # 梯度累积：每 accumulation_steps 步才做一次 optimizer.step()
# accumulation_steps = 8
# optimizer = AdamW(model.parameters(), lr=5e-5)
#
# model.train()
# for epoch in range(5):
#     for step, batch in enumerate(train_loader):
#         batch = {k: v.to(model.device) for k, v in batch.items()}
#         loss = model(**batch).loss / accumulation_steps  # loss 归一化
#         loss.backward()
#
#         if (step + 1) % accumulation_steps == 0:
#             optimizer.step()
#             optimizer.zero_grad()
# ─────────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════
# Part 5: 保存 LoRA 适配器
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part 5: 保存 LoRA 适配器")
print("=" * 60)

adapter_path = "./03_lora_adapter"
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)

files = glob.glob(os.path.join(adapter_path, "*"))
total_mb = sum(os.path.getsize(f) for f in files if os.path.isfile(f)) / 1024**2
print(f"LoRA 适配器已保存到: {adapter_path} ({total_mb:.1f} MB)")

# ════════════════════════════════════════════════════════════
# Part 6: SFT 前后对比
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part 6: SFT 前后对比")
print("=" * 60)


def ask_model(m, tok, question, system="你是一个推荐系统助手。"):
    """用模型推理，返回回复文本"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    inputs = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt",
    ).to(m.device)

    with torch.inference_mode():
        outputs = m.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.3,
            top_p=0.95,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    return tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# 四维度测试问题
test_questions = [
    ("懂物料", "请解读以下视频token: <|video_begin|><s_a_5254><s_b_6442><s_c_7598> /no_think",
     "你是一个精准的视频语义解析器，能够将抽象的视频token转化为通俗易懂的视频内容。"),
    ("懂世界", "三角形ABC中，A(1,5), B(1,1), C(3,1)，是什么三角形？A.锐角 B.直角 C.钝角 /no_think",
     "你是一个知识问答助手。"),
    ("懂推荐", "一个喜欢看美食和旅游视频的用户，最近看了<|video_begin|><s_a_3915><s_b_8150><s_c_535>，还可能喜欢什么？ /no_think",
     "你是一位推荐系统专家，根据用户偏好推荐合适的内容。"),
    ("懂用户", "用户连续观看了多个汽车评测视频后，开始搜索二手车价格，用户的兴趣发生了什么变化？ /no_think",
     "你是一位用户行为分析专家，擅长洞察用户兴趣演化。"),
]

# ── Step 1: 先用 SFT 后的模型推理 ──
print("\n--- SFT 后的模型 ---")
sft_results = {}
for category, question, system in test_questions:
    response = ask_model(model, tokenizer, question, system)
    sft_results[category] = response
    print(f"\n{'─'*50}")
    print(f"【{category}】{question[:40]}...")
    print(f"  [SFT后]: {response[:200]}")

# ── Step 2: 释放 SFT 模型，加载 base 模型 ──
print("\n\n--- 加载基座模型进行对比 ---")
del model
torch.cuda.empty_cache()

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,
)

print("\n--- SFT 前（基座模型）---")
for category, question, system in test_questions:
    response = ask_model(base_model, tokenizer, question, system)
    print(f"\n{'─'*50}")
    print(f"【{category}】{question[:40]}...")
    print(f"  [SFT前]: {response[:200]}")
    if category in sft_results:
        print(f"  [SFT后]: {sft_results[category][:200]}")

# 释放 base 模型
del base_model
torch.cuda.empty_cache()

# ════════════════════════════════════════════════════════════
# 总结
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Day 3 总结")
print("=" * 60)
print(f"""
训练了 {len(dataset)} 条真实数据（{len(task_types)} 种任务, 四维能力）
适配器: {total_mb:.0f} MB

关键改进:
  ✅ Label 遮蔽 — 只对 assistant 回复算 loss
  ✅ SFTTrainer — 梯度累积 + 混合精度 + 自动 checkpoint
  ✅ max_length=2048 — 覆盖推荐/用户数据的长序列
  ✅ 四维度对比 — 修复 OOM，依次加载模型

测试: python chat.py -m sft
       python chat.py -m onereason  (对比基座)
""")
