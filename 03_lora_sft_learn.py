"""
============================================================
Day 3 交互式学习版：逐段理解 LoRA SFT 微调
============================================================
用法:
    python 03_lora_sft_learn.py --part 1   # 只跑 Part 1（数据解析）
    python 03_lora_sft_learn.py --part 2   # 只跑 Part 2（Tokenizer）
    python 03_lora_sft_learn.py --part 3   # 只跑 Part 3（模型+LoRA）
    python 03_lora_sft_learn.py --part 4   # 只跑 Part 4（训练）
    python 03_lora_sft_learn.py --part 5   # 只跑 Part 5（保存）
    python 03_lora_sft_learn.py --part 6   # 只跑 Part 6（对比）
    python 03_lora_sft_learn.py --all      # 跑全部（等同于 03_lora_sft.py）

学习建议：按顺序从 Part 1 开始，每跑完一个 Part 理解后再跑下一个
============================================================
"""
import torch
import os
import json
import re
import glob
import argparse

# ════════════════════════════════════════════════════════════
# 全局配置
# ════════════════════════════════════════════════════════════
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(ROOT, "dataexample.txt")
MODEL_PATH = os.path.join(ROOT, "OneReason-0.8B-pretrain-competition")

# 四个维度的 section 标题
SECTIONS = [
    ("物料", r"物料数据示例[：:]"),
    ("用户", r"用户数据示例[：:]"),
    ("推荐", r"推荐数据示例[：:]"),
    ("世界", r"世界知识数据示例[：:]"),
]

# ════════════════════════════════════════════════════════════
# Part 1: 数据解析 — 比赛数据长什么样？
# ════════════════════════════════════════════════════════════
def run_part1():
    """
    💡 学习目标：理解比赛数据的 4 个维度、13 种任务、数据格式差异

    关键知识：
    - dataexample.txt 有 4 个 section，每个以"XX数据示例："开头
    - 每个 section 是一个 JSON 对象，顶层 key 是任务名
    - messages 可能是列表（物料/用户/世界）或 JSON 字符串（推荐）
    - content 可能是纯字符串或 [{"type":"text","text":"..."}] 列表
    - metadata 可能是字典或 JSON 字符串
    - answer 是 assistant 应该输出的内容（训练标签）
    """
    print("=" * 60)
    print("Part 1: 解析 dataexample.txt — 理解比赛数据格式")
    print("=" * 60)

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    print(f"文件大小: {len(content)} 字符\n")

    # 逐 section 解析
    dataset = []
    for sec_name, sec_pattern in SECTIONS:
        match = re.search(sec_pattern, content)
        if not match:
            print(f"  [警告] 未找到 section: {sec_name}")
            continue

        # 找到该 section 的内容范围
        start = match.end()
        next_match = None
        for _, next_pattern in SECTIONS:
            nm = re.search(next_pattern, content[start:])
            if nm:
                candidate = start + nm.start()
                if next_match is None or candidate < next_match:
                    next_match = candidate

        sec_text = content[start:next_match].strip() if next_match else content[start:].strip()

        # 解析 JSON
        try:
            tasks = json.loads(sec_text)
        except json.JSONDecodeError as e:
            print(f"  [警告] {sec_name} JSON 解析失败: {e}")
            continue

        print(f"\n📁 {sec_name} section:")
        print(f"   JSON 顶层 keys: {list(tasks.keys())}")

        for task_name, task_data in tasks.items():
            # 提取 messages
            raw_msgs = task_data.get("messages")
            if raw_msgs is None:
                continue

            if isinstance(raw_msgs, str):
                messages = json.loads(raw_msgs)
                msg_type = "JSON字符串"
            else:
                messages = raw_msgs
                msg_type = "列表"

            # 提取 answer
            metadata = task_data.get("metadata", {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            answer = metadata.get("answer", "")
            answer_preview = answer[:80].replace("\n", "\\n") + ("..." if len(answer) > 80 else "")

            # 提取 content 类型
            first_content = messages[0].get("content", "") if messages else ""
            content_type = "纯字符串" if isinstance(first_content, str) else "[{type:text}] 列表"

            print(f"\n   📋 任务: {task_name}")
            print(f"      messages 类型: {msg_type}")
            print(f"      content 类型:  {content_type}")
            print(f"      消息轮数:      {len(messages)}")
            print(f"      answer 预览:   {answer_preview}")

            # 转换为统一格式
            converted = []
            for msg in messages:
                msg_content = msg.get("content", "")
                if isinstance(msg_content, list):
                    texts = []
                    for block in msg_content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            texts.append(block["text"])
                    msg_content = "\n".join(texts)
                converted.append({"role": msg["role"], "content": msg_content})

            converted.append({"role": "assistant", "content": answer})

            dataset.append({
                "task": task_name,
                "category": sec_name,
                "messages": converted,
            })

    # 汇总
    print(f"\n{'='*50}")
    print(f"✅ 解析完成: {len(dataset)} 条训练样本")
    counter = {}
    for d in dataset:
        counter[d["category"]] = counter.get(d["category"], 0) + 1
    for cat, count in counter.items():
        print(f"   {cat}: {count} 条")

    # 💡 展示一条完整样本，让你看到 SFT 数据长什么样
    print(f"\n{'='*50}")
    print("💡 示例：一条完整的 SFT 训练样本（懂世界/单选题）")
    sample = next(d for d in dataset if d["task"] == "single")
    for msg in sample["messages"]:
        role_tag = {"system": "🤖 system", "user": "👤 user", "assistant": "✅ assistant"}[msg["role"]]
        content_preview = msg["content"][:200].replace("\n", "\\n")
        print(f"\n  {role_tag}:")
        print(f"    {content_preview}")

    return dataset


# ════════════════════════════════════════════════════════════
# Part 2: Tokenizer — 文本怎么变成数字？
# ════════════════════════════════════════════════════════════
def run_part2(dataset):
    """
    💡 学习目标：理解 Tokenizer 的作用、chat_template、token 长度

    关键知识：
    - Tokenizer 把文本变成整数 ID 列表（模型只认识数字）
    - chat_template 把 system/user/assistant 消息格式化为模型认识的格式
    - OneReason 用 ChatML 格式: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...
    - SFT 训练时 add_generation_prompt=False（不需要"请继续生成"的提示）
    - 训练时只有 assistant 的部分算 loss，prompt 部分被遮蔽（-100）
    - pad_token 用于批量训练时对齐不同长度的序列
    """
    print("\n" + "=" * 60)
    print("Part 2: Tokenizer — 文本 → 数字")
    print("=" * 60)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print(f"词表大小: {len(tokenizer)}")
    print(f"Pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"EoS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")

    # 💡 演示：一条消息如何被 tokenize
    print(f"\n{'─'*50}")
    print("💡 演示：chat_template 格式化过程")
    sample = next(d for d in dataset if d["task"] == "single")

    # 1. 原始 messages
    print("\n  Step 1: 原始 messages（Python 字典列表）")
    print(f"    {len(sample['messages'])} 条消息")

    # 2. 应用 chat_template
    formatted = tokenizer.apply_chat_template(
        sample["messages"],
        tokenize=False,           # 先看文本形式
        add_generation_prompt=False,  # 训练时不需要 generation prompt
    )
    print(f"\n  Step 2: chat_template 格式化后（前 500 字符）")
    print(f"    {formatted[:500]}")

    # 3. Tokenize
    token_ids = tokenizer.apply_chat_template(
        sample["messages"],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )
    # transformers 5.x: apply_chat_template 返回 dict，取 input_ids
    token_ids_tensor = token_ids["input_ids"]
    print(f"\n  Step 3: Tokenize 后 → {token_ids_tensor.shape[1]} 个 token ID")
    print(f"    前 10 个: {token_ids_tensor[0][:10].tolist()}")

    # 💡 关键：label 遮蔽
    print(f"\n{'─'*50}")
    print("💡 关键概念：Label 遮蔽（只对 assistant 回复算 loss）")
    print("""
    假设 tokenize 后的序列:
      [system_tokens] [user_tokens] [assistant_tokens]

    训练 labels:
      [   -100    ] [   -100    ] [assistant_tokens]   ← 只有这部分算 loss

    -100 是 PyTorch CrossEntropyLoss 的 ignore_index
    这就是 SFTTrainer 自动帮你做的事情！
    如果不做遮蔽，模型会浪费大量容量学习复制 prompt
    """)

    # 统计各任务的 token 长度
    print(f"{'─'*50}")
    print("💡 各任务的 token 长度（决定 max_length 设置）")

    lengths_by_task = {}
    for d in dataset:
        ids = tokenizer.apply_chat_template(
            d["messages"], tokenize=True,
            add_generation_prompt=False, return_tensors="pt",
        )
        task = f"{d['category']}/{d['task']}"
        lengths_by_task[task] = ids["input_ids"].shape[1]

    # 按长度排序显示
    for task, length in sorted(lengths_by_task.items(), key=lambda x: -x[1]):
        bar = "█" * min(length // 100, 50)
        flag = " ⚠ 超 2048!" if length > 2048 else ""
        print(f"  {task:25s} {length:5d} tokens  {bar}{flag}")

    return tokenizer


# ════════════════════════════════════════════════════════════
# Part 3: 模型 + LoRA — 冻结大模型，只训练小适配器
# ════════════════════════════════════════════════════════════
def run_part3():
    """
    💡 学习目标：理解 LoRA 的原理和配置

    关键知识：
    - LoRA = 冻结原始权重 W，加一个低秩分解 W' = W + A×B
    - A 是 (d×r) 矩阵，B 是 (r×d) 矩阵，r << d
    - 实际缩放 = alpha/r，所以 alpha=16, r=8 → 缩放 2x
    - target_modules 指定哪些层加 LoRA：
      - Q/K/V/O：注意力机制的 4 个线性层
      - gate/up/down：FFN 的 3 个门控层
    - bias="none"：不训练偏置项（参数太少不值得）
    - lora_dropout：防止过拟合的正则化

    显存估算（0.8B 模型 + LoRA）：
    - 基座 bf16: ~1.6 GB
    - LoRA 参数: ~6M × 4 bytes ≈ 24 MB
    - 优化器状态: ~6M × 8 bytes ≈ 48 MB
    - 激活值（seq_len=2048, batch=1）: ~2-3 GB
    - 总计: ~4-5 GB，RTX 4060 够用
    """
    print("\n" + "=" * 60)
    print("Part 3: 加载模型 + 挂载 LoRA")
    print("=" * 60)

    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    # 加载基座模型
    print(f"加载基座模型: {MODEL_PATH}")
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",          # 自动分配 GPU
        torch_dtype=torch.bfloat16, # 半精度，省显存
        trust_remote_code=True,     # 信任自定义代码
    )
    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"✅ 模型已加载, 显存: {vram:.1f} GB")

    # 💡 看看模型结构
    print(f"\n模型类型: {type(model).__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params/1e9:.2f}B ({total_params:,})")

    # LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,     # 因果语言模型
        r=8,                               # 秩（核心超参）
        lora_alpha=16,                     # 缩放因子
        lora_dropout=0.05,                 # 正则化
        target_modules=[                   # 要加 LoRA 的层
            "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力
            "gate_proj", "up_proj", "down_proj",       # FFN
        ],
        bias="none",                       # 不训练偏置
    )

    print(f"\n💡 LoRA 配置:")
    print(f"  r={lora_config.r}, alpha={lora_config.lora_alpha}")
    print(f"  缩放倍数: alpha/r = {lora_config.lora_alpha}/{lora_config.r} = {lora_config.lora_alpha/lora_config.r}")
    print(f"  target_modules: {lora_config.target_modules}")
    print(f"  dropout: {lora_config.lora_dropout}")

    # 挂载 LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    vram_after = torch.cuda.memory_allocated() / 1024**3
    print(f"\n显存变化: {vram:.1f} GB → {vram_after:.1f} GB (+{vram_after-vram:.2f} GB)")

    return model


# ════════════════════════════════════════════════════════════
# Part 4: SFT 训练 — 用 SFTTrainer 自动化训练
# ════════════════════════════════════════════════════════════
def run_part4(model, dataset, tokenizer):
    """
    💡 学习目标：理解 SFTTrainer 做了什么

    SFTTrainer 自动处理的 5 件事（手写容易出错）：
    1. Label 遮蔽：只对 assistant 回复计算 loss
    2. 梯度累积：每 accumulation_steps 步才更新参数
    3. 混合精度：bf16 自动开启
    4. Checkpoint 保存：按策略自动保存
    5. 数据 collation：自动 pad + 截断

    关键超参解释：
    - learning_rate=5e-5: LoRA SFT 的典型值（全量微调可以用 2e-5）
    - gradient_accumulation_steps=8: 等效 batch_size=1×8=8
    - warmup_ratio=0.05: 前 5% 步数线性升温学习率
    - bf16=True: 用 bfloat16 混合精度训练
    """
    print("\n" + "=" * 60)
    print("Part 4: SFT 训练")
    print("=" * 60)

    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    # 转换为 HuggingFace Dataset
    train_dataset = Dataset.from_list(dataset)
    train_dataset = train_dataset.shuffle(seed=42)

    sft_config = SFTConfig(
        max_length=2048,                      # 最大序列长度
        per_device_train_batch_size=1,         # 每张卡 batch=1
        gradient_accumulation_steps=8,         # 累积 8 步再更新 → 等效 batch=8
        num_train_epochs=5,                    # 训练 5 轮
        learning_rate=5e-5,                    # 学习率
        bf16=True,                             # 混合精度
        warmup_ratio=0.05,                     # 预热比例
        logging_steps=1,                       # 每步打印 loss
        save_strategy="epoch",                 # 每 epoch 存一次
        save_total_limit=2,                    # 最多存 2 个 checkpoint
        output_dir="./03_lora_adapter",        # 输出目录
        report_to="none",                      # 不上报 wandb 等
    )

    print(f"训练配置:")
    print(f"  数据: {len(train_dataset)} 条")
    print(f"  有效 batch: {1 * 8} (1 per_device × 8 accum)")
    print(f"  总步数: ~{len(train_dataset) * 5 // 8}")

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("\n🚀 开始训练...")
    trainer.train()

    print("\n✅ 训练完成！")
    return model


# ════════════════════════════════════════════════════════════
# Part 5: 保存适配器
# ════════════════════════════════════════════════════════════
def run_part5(model, tokenizer):
    """
    💡 学习目标：LoRA 适配器 vs 完整模型

    关键知识：
    - save_pretrained 只保存 LoRA 参数（几十 MB），不保存基座（1.6 GB）
    - 推理时需要：基座模型 + LoRA 适配器 = 完整 SFT 模型
    - merge_and_unload() 可以把 LoRA 合并进基座，推理更快
    """
    print("\n" + "=" * 60)
    print("Part 5: 保存 LoRA 适配器")
    print("=" * 60)

    adapter_path = "./03_lora_adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    files = glob.glob(os.path.join(adapter_path, "*"))
    total_mb = sum(os.path.getsize(f) for f in files if os.path.isfile(f)) / 1024**2

    print(f"LoRA 适配器: {total_mb:.1f} MB")
    print(f"基座模型: ~1600 MB")
    print(f"比例: {total_mb/1600*100:.2f}%")
    print(f"\n💡 这就是 LoRA 的好处：只存 0.5% 的参数，却能改变模型行为")


# ════════════════════════════════════════════════════════════
# Part 6: SFT 前后对比
# ════════════════════════════════════════════════════════════
def run_part6(model, tokenizer):
    """
    💡 学习目标：理解如何评估 SFT 效果

    关键知识：
    - 8GB 显卡不能同时加载两个模型！
    - 策略：先跑 SFT 模型 → 释放 → 加载 base → 对比
    - 推理用 torch.inference_mode()（比 no_grad 更快）
    - temperature 控制随机性：0.3（保守）→ 1.0（创意）
    - 比赛用 Pass@64 = 生成 64 次，命中一次就算对
    """
    print("\n" + "=" * 60)
    print("Part 6: SFT 前后对比")
    print("=" * 60)

    from transformers import AutoModelForCausalLM

    def ask_model(m, tok, question, system="你是一个推荐系统助手。"):
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

    # 四维度测试
    test_questions = [
        ("懂物料", "请解读以下视频token: <|video_begin|><s_a_5254><s_b_6442><s_c_7598> /no_think",
         "你是一个精准的视频语义解析器。"),
        ("懂世界", "三角形ABC中，A(1,5), B(1,1), C(3,1)，是什么三角形？A.锐角 B.直角 C.钝角 /no_think",
         "你是一个非常聪明的助手，请直接遵循指示作答。"),
    ]

    # SFT 后
    print("\n--- SFT 后 ---")
    sft_results = {}
    for category, question, system in test_questions:
        response = ask_model(model, tokenizer, question, system)
        sft_results[category] = response
        print(f"\n【{category}】")
        print(f"  SFT后: {response[:200]}")

    # 释放 SFT 模型，加载基座
    print("\n\n--- 释放 SFT 模型，加载基座 ---")
    del model
    torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,
    )

    print("\n--- SFT 前（基座） ---")
    for category, question, system in test_questions:
        response = ask_model(base_model, tokenizer, question, system)
        print(f"\n【{category}】")
        print(f"  SFT前: {response[:200]}")
        if category in sft_results:
            print(f"  SFT后: {sft_results[category][:200]}")

    del base_model
    torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA SFT 交互式学习")
    parser.add_argument("--part", type=int, choices=[1,2,3,4,5,6], help="只跑某个 Part")
    parser.add_argument("--all", action="store_true", help="跑全部 Part")
    args = parser.parse_args()

    # 环境检查
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:     {torch.cuda.get_device_name(0)}")
        free = (torch.cuda.get_device_properties(0).total_memory -
                torch.cuda.memory_allocated()) / 1024**3
        print(f"显存空闲: {free:.1f} GB")

    # 决定跑哪些 Part
    run_all = args.all or args.part is None

    if run_all or args.part == 1:
        dataset = run_part1()
    else:
        # 需要先有 dataset
        dataset = run_part1()

    if run_all or args.part == 2:
        tokenizer = run_part2(dataset)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    if run_all or args.part == 3:
        model = run_part3()

    if args.part == 4 or run_all:
        if 'model' not in dir():
            model = run_part3()
        model = run_part4(model, dataset, tokenizer)

    if args.part == 5 or run_all:
        if 'model' not in dir():
            print("请先跑 Part 3+4 训练模型")
            exit(1)
        run_part5(model, tokenizer)

    if args.part == 6 or run_all:
        if 'model' not in dir():
            print("请先跑 Part 3+4 训练模型")
            exit(1)
        run_part6(model, tokenizer)
