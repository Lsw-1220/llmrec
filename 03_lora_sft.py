"""
============================================================
Day 3: LoRA/QLoRA 微调 — 基于真实比赛数据格式
目标：适配 dataexample.txt 中的 12 种任务，完成 LoRA SFT
时间：约 4-6 小时

运行方式:
    conda activate llmrec
    python 03_lora_sft.py

数据来源: dataexample.txt（包含四维能力共 12 种任务类型）
============================================================
"""
import torch
import os
import json
import re
import glob
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

print(f"PyTorch: {torch.__version__}")
print(f"CUDA:    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    free = (torch.cuda.get_device_properties(0).total_memory -
            torch.cuda.memory_allocated()) / 1024**3
    print(f"显存空闲: {free:.1f} GB")

# ════════════════════════════════════════════════════════════
# Part 0: LoRA 原理（同之前）
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part 0: LoRA 核心原理")
print("=" * 60)
print("""
LoRA = 冻结模型 + 低秩适配矩阵 A×B
0.8B 全量微调 ~8.4GB vs LoRA ~3.6GB → 8GB 显卡刚好够
""")

# ════════════════════════════════════════════════════════════
# Part 1: 加载并解析真实比赛数据
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part 1: 解析 dataexample.txt")
print("=" * 60)

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataexample.txt")

def parse_dataexample(filepath):
    """
    解析 dataexample.txt 为统一的 SFT 训练格式。
    每行是一个 JSON 对象，格式:
      {
        "task_name": {
          "messages": [...] 或 "[...]" (JSON 字符串),
          "metadata": {"answer": "..."} 或 "..." (JSON 字符串)
        }
      }
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # 数据按大段分隔："物料数据示例：", "用户数据示例：", "推荐数据示例：", "世界知识数据示例："
    # 每个大段包含一个 JSON 对象
    # 策略: 找到所有 {...} 形式的顶级 JSON 对象

    # 移除中文标题行
    raw = content
    raw = re.sub(r'^.*数据示例.*$', '', raw, flags=re.MULTILINE)

    # 尝试找到所有顶级 JSON 块（每个任务是一个顶级 key-value）
    dataset = []
    task_counter = {"物料": 0, "用户": 0, "推荐": 0, "世界": 0}

    # 分块解析：找到 "XXX数据示例" 之后的内容
    sections = re.split(r'^(?:物料|用户|推荐|世界知识)数据示例[：:]\s*$', content, flags=re.MULTILINE)
    section_names = ["header", "物料", "用户", "推荐", "世界"]

    for sec_name, sec_text in zip(section_names, sections):
        if sec_name == "header":
            continue

        # 找到这个 section 中的 JSON 对象
        # 尝试直接 JSON.parse
        sec_text = sec_text.strip()
        if not sec_text:
            continue

        try:
            tasks = json.loads(sec_text)
        except json.JSONDecodeError:
            # 可能有格式问题，尝试修复
            print(f"  [警告] {sec_name} section JSON 解析失败，跳过")
            continue

        for task_name, task_data in tasks.items():
            # ── 提取 messages ──
            raw_msgs = task_data.get("messages")

            if raw_msgs is None:
                # 像 {video_itemic_pattern: {messages: ...}} 外层有包裹
                # 已经解包了，继续
                continue

            if isinstance(raw_msgs, str):
                # messages 是 JSON 字符串（推荐数据）
                try:
                    messages = json.loads(raw_msgs)
                except json.JSONDecodeError as e:
                    print(f"  [跳过] {task_name}: messages JSON 解析失败: {e}")
                    continue
            elif isinstance(raw_msgs, list):
                messages = raw_msgs
            else:
                print(f"  [跳过] {task_name}: 未知 messages 类型 {type(raw_msgs)}")
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
                print(f"  [跳过] {task_name}: 无 answer")
                continue

            # ── 转换为统一的 ChatML 格式 ──
            # content 字段可能是 "text" 或 [{"type":"text","text":"..."}]
            converted = []
            for msg in messages:
                content = msg.get("content", "")

                # 如果是列表格式，提取 text
                if isinstance(content, list):
                    texts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            texts.append(block["text"])
                    content = "\n".join(texts)

                converted.append({
                    "role": msg["role"],
                    "content": content,
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
task_types = set(d["task"] for d in dataset)
print(f"\n任务类型 ({len(task_types)} 种): {sorted(task_types)}")

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

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"Pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")

# 看看数据长什么样
for d in dataset:
    sample_text = tokenizer.apply_chat_template(
        d["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    # 找一条像推荐/对话的任务展示
    if d["task"] in ("video", "Product", "single"):
        print(f"\n示例 [{d['category']}/{d['task']}]:")
        print(sample_text[:250])
        break

# 统计长度
def get_length(example):
    return len(tokenizer.apply_chat_template(
        example["messages"],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )[0])

lengths = [get_length(d) for d in dataset]
print(f"\n数据长度统计: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)//len(lengths)} tokens")

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
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ════════════════════════════════════════════════════════════
# Part 4: 训练
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part 4: SFT 训练")
print("=" * 60)

from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


class SFTDataset(TorchDataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.input_ids = []
        self.attention_masks = []
        for item in data:
            text = tokenizer.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            encoded = tokenizer(
                text, truncation=True, max_length=max_length,
                padding=False, return_tensors="pt",
            )
            self.input_ids.append(encoded["input_ids"][0])
            self.attention_masks.append(encoded["attention_mask"][0])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.input_ids[idx].clone(),
        }


def collate_fn(batch):
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids, attention_masks, labels = [], [], []
    for item in batch:
        pad_len = max_len - item["input_ids"].shape[0]
        input_ids.append(torch.cat([
            item["input_ids"],
            torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)
        ]))
        attention_masks.append(torch.cat([
            item["attention_mask"],
            torch.zeros(pad_len, dtype=torch.long)
        ]))
        labels.append(torch.cat([
            item["labels"],
            torch.full((pad_len,), -100, dtype=torch.long)
        ]))
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(labels),
    }


train_dataset = SFTDataset(dataset, tokenizer, max_length=1024)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

optimizer = AdamW(model.parameters(), lr=2e-4)
total_steps = len(train_loader) * 3
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2, num_training_steps=total_steps)

print(f"训练数据: {len(train_dataset)} 条, {len(train_loader)} batch/epoch")
print(f"总步数: {total_steps} (3 epochs)")

model.train()
for epoch in range(3):
    total_loss = 0
    for step, batch in enumerate(train_loader):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()

        if (step + 1) % max(1, len(train_loader) // 3) == 0 or step == 0:
            print(f"  Epoch {epoch+1}, Step {step+1}/{len(train_loader)}, "
                  f"Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

    avg_loss = total_loss / len(train_loader)
    print(f"  → Epoch {epoch+1} 完成, Avg Loss: {avg_loss:.4f}")

print("训练完成！")

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
total_mb = sum(os.path.getsize(f) for f in files) / 1024**2
print(f"LoRA 适配器已保存到: {adapter_path} ({total_mb:.1f} MB)")

# ════════════════════════════════════════════════════════════
# Part 6: SFT 前后对比
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part 6: SFT 前后对比")
print("=" * 60)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,
)

def ask_model(m, tok, question, system="你是一个推荐系统助手。"):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    inputs = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt"
    ).to(m.device)

    with torch.inference_mode():
        outputs = m.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.3,
            top_p=0.95,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    return tok.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


for category, question in [
    ("懂物料", "请解读以下视频token: <|video_begin|><s_a_5254><s_b_6442><s_c_7598>"),
    ("懂世界", "三角形ABC中，A(1,5), B(1,1), C(3,1)，是什么三角形？A.锐角 B.直角 C.钝角"),
]:
    print(f"\n{'─'*50}")
    print(f"【{category}】")
    print(f"\n[SFT 前]: {ask_model(base_model, tokenizer, question)[:150]}")
    print(f"[SFT 后]: {ask_model(model, tokenizer, question)[:150]}")

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
适配器: {total_mb:.0f} MB, 占总参数 0.63%

测试: python chat.py -m sft
       python chat.py -m onereason  (对比基座)
""")
