"""
============================================================
Day 3: LoRA/QLoRA 微调 — 把 OneReason 从"背物料"教成"会推荐"
目标：理解 LoRA 原理、构建 SFT 数据、完成微调、对比推理
时间：约 4-6 小时

运行方式:
    conda activate llmrec
    python 03_lora_sft.py

前置: 已下载 OneReason-0.8B-pretrain-competition 到本地

理论参考:
    - LoRA 论文: https://arxiv.org/abs/2106.09685
    - QLoRA 论文: https://arxiv.org/abs/2305.14314
    - PEFT 文档: https://huggingface.co/docs/peft
============================================================
"""
import torch
import os
import json
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)

print(f"PyTorch: {torch.__version__}")
print(f"CUDA:    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    free = (torch.cuda.get_device_properties(0).total_memory -
            torch.cuda.memory_allocated()) / 1024**3
    print(f"显存空闲: {free:.1f} GB")

# ════════════════════════════════════════════════════════════
# Part 0: LoRA 是什么？为什么 0.8B 还要用 LoRA？
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part 0: LoRA 核心原理")
print("=" * 60)

print("""
LoRA = Low-Rank Adaptation

一句话：冻结原模型所有参数，只在旁边挂两根"小棍子"来学习。

    ┌─────────────────────────┐
    │  原始权重 W (冻结不动)    │  ← 1.6 GB, 不训练
    │       +                  │
    │  LoRA: A @ B (低秩矩阵)   │  ← ~10 MB, 只训这个！
    │       =                  │
    │   W + ΔW (微调后权重)     │
    └─────────────────────────┘

数学:
    ΔW = B × A
    其中 A: (d, r), B: (r, d), r << d
    比如 d=1024, r=8 → A@B 只有 1024×8×2 = 16K 参数

为什么 0.8B 还要用 LoRA？

全量微调的显存:
  模型权重: 1.6 GB (bf16)
  优化器状态 (Adam): 3.2 GB (momentum + variance)
  梯度: 1.6 GB
  激活值: ~2 GB (batch + 长序列)
  ──────────────
  总计: ~8.4 GB  ← 超过 RTX 4060 的 8GB！

LoRA 的显存:
  模型权重: 1.6 GB (冻结)
  LoRA 参数: ~0.01 GB
  优化器 + 梯度: ~0.03 GB
  激活值: ~2 GB
  ──────────────
  总计: ~3.6 GB  ← 还剩 4.4 GB，随便调 batch！
""")

# ════════════════════════════════════════════════════════════
# Part 1: 构建 SFT 训练数据
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part 1: SFT 数据构建")
print("=" * 60)

"""
SFT = Supervised Fine-Tuning = 给模型看"输入 → 正确答案"的例子

比赛四个维度对应的数据格式:
  懂物料: "分析这个视频" → JSON 结构化理解
  懂用户: "用户行为序列" → 兴趣演化结论
  懂推荐:  "推荐三款游戏" → 推荐列表
  懂世界:  "什么是推荐系统？" → 常识回答

数据格式 (ChatML, 同 Qwen):
  system + user → assistant 回答

下面的演示数据只含 20 条，训练几分钟就完。
真实比赛有几十万条，训练逻辑完全一样。
"""

def build_demo_dataset():
    """构建覆盖四维能力的演示数据集"""
    data = []

    # ── 懂物料 (5条) ──
    material_prompt = """你是一个视频内容理解助手，请分析给定的视频物料。
输出 JSON 格式: {"类型": "...", "关键词": [...], "目标观众": "...", "推荐场景": "..."}"""

    material_samples = [
        {
            "input": "视频标题: 5分钟瘦肚子教程\n标签: #健身 #瘦肚子 #教程\n描述: 无需器械，在家就能练的瘦肚子动作",
            "output": '{"类型": "健身教程", "关键词": ["瘦肚子", "健身", "居家", "无器械"], "目标观众": "想减肥的年轻人", "推荐场景": "饭后刷手机看到健身内容时"}'
        },
        {
            "input": "视频标题: 兰州牛肉面做法\n标签: #美食 #面食 #教程\n描述: 从和面到拉面全过程",
            "output": '{"类型": "美食教程", "关键词": ["牛肉面", "拉面", "兰州", "面食"], "目标观众": "喜欢做饭的人", "推荐场景": "晚饭不知道吃什么时"}'
        },
        {
            "input": "视频标题: iPhone 15 Pro 评测\n标签: #数码 #评测 #苹果\n描述: 全面解析 A17 Pro 芯片性能",
            "output": '{"类型": "数码评测", "关键词": ["iPhone", "苹果", "手机", "评测"], "目标观众": "想换手机的人", "推荐场景": "浏览数码内容时"}'
        },
        {
            "input": "视频标题: 猫咪搞笑合集\n标签: #萌宠 #搞笑 #猫咪\n描述: 各种猫咪翻车名场面",
            "output": '{"类型": "萌宠娱乐", "关键词": ["猫咪", "搞笑", "萌宠", "解压"], "目标观众": "宠物爱好者", "推荐场景": "工作累了想放松时"}'
        },
        {
            "input": "视频标题: 考研数学复习攻略\n标签: #考研 #数学 #学习\n描述: 三个月备考规划，从基础到冲刺",
            "output": '{"类型": "学习教程", "关键词": ["考研", "数学", "复习", "备考"], "目标观众": "考研学生", "推荐场景": "深夜刷到学习视频时"}'
        },
    ]
    for s in material_samples:
        data.append({
            "messages": [
                {"role": "system", "content": material_prompt},
                {"role": "user", "content": f"请分析这个视频物料：\n{s['input']}"},
                {"role": "assistant", "content": s["output"]},
            ]
        })

    # ── 懂推荐 (5条) ──
    rec_prompt = "你是一个推荐系统助手，根据用户偏好推荐相关内容。"

    rec_samples = [
        {"input": "我喜欢看科幻电影，推荐三部", "output": "推荐你三部高分科幻电影：\n1. 《星际穿越》- 硬科幻+亲情\n2. 《银翼杀手2049》- 赛博朋克美学\n3. 《降临》- 语言学与外星文明\n三部都是 IMDb 8.0+ 的神作。"},
        {"input": "推荐适合新手做的家常菜", "output": "新手友好的三道家常菜：\n1. 番茄炒蛋 - 10分钟搞定，零失败\n2. 可乐鸡翅 - 只需4种调料\n3. 蒜蓉西兰花 - 健康又快手的蔬菜\n从第一道开始练手吧！"},
        {"input": "最近有什么好听的流行歌？", "output": "根据近期热度，推荐你三首：\n1. 周杰伦《圣诞星》- 经典周氏情歌\n2. Taylor Swift《Fortnight》- 全球热单\n3. 张碧晨《笼》- 影视OST爆款\n曲风各不相同，看你的口味。"},
        {"input": "我想买一个适合学生的笔记本电脑", "output": "学生党笔记本推荐（2026年）：\n1. 联想小新Pro 16 - 性价比之王，学习办公绰绰有余\n2. 华为MateBook 14 - 轻薄便携，多屏协同方便\n3. MacBook Air M4 - 长续航，编程设计都流畅\n预算5000左右选1，有华为手机选2，预算够选3。"},
        {"input": "推荐几个适合周末去玩的地方", "output": "周末出游推荐：\n1. 城市周边古镇 - 感受慢生活，吃特色小吃\n2. 户外露营地 - 约上朋友烧烤看星空\n3. 博物馆/美术馆 - 安静涨知识还免费\n看你想要热闹还是安静。"},
    ]
    for s in rec_samples:
        data.append({
            "messages": [
                {"role": "system", "content": rec_prompt},
                {"role": "user", "content": s["input"]},
                {"role": "assistant", "content": s["output"]},
            ]
        })

    # ── 懂用户 (5条) ──
    user_prompt = "你是一个用户行为分析师，从用户交互历史中洞察兴趣演化。"

    user_samples = [
        {"input": "用户行为: 浏览→点赞→收藏→分享 [健身视频] → 浏览→点赞 [减脂餐视频] → 浏览→搜索 [跑步鞋]", "output": "分析：用户处于健康塑形阶段。从泛化健身开始，逐步聚焦到饮食控卡和跑步装备。兴趣从'被动观看'演化到'主动搜索购买'，推荐跑步装备和减脂食谱。"},
        {"input": "用户行为: 浏览 [考研数学] → 收藏 [考研英语] → 浏览→浏览 [考研政治] → 搜索 [研究生宿舍]", "output": "分析：典型的考研备考路径。从数学发散到全科覆盖，最后搜索宿舍说明已进入择校阶段。兴趣从'学习资料收集'演化到'学校选择'，推荐考研经验和院校分析。"},
        {"input": "用户行为: 浏览 [iPhone评测] → 收藏 [安卓旗舰对比] → 搜索 [手机拍照排名] → 浏览 [二手手机] ", "output": "分析：用户有明确换机需求。从苹果出发，对比安卓，关注拍照和性价比。最后看二手说明预算敏感。兴趣从'品牌偏好'到'性价比决策'，推荐中端性价比机型。"},
        {"input": "用户行为: 搜索 [三亚酒店] → 收藏 [三亚攻略] → 浏览 [防晒霜] → 搜索 [机票特价]", "output": "分析：出行决策全链路。从选住宿→收攻略→买装备→订交通，用户已进入出行准备后期。兴趣从'目的地探索'聚焦到'出行执行'，推荐三亚当地玩法和比价工具。"},
        {"input": "用户行为: 浏览 [Python教程] → 点赞 [AI入门] → 收藏 [机器学习] → 搜索 [GPU云服务器]", "output": "分析：技术学习路径演进。从基础编程快速过渡到AI实战，搜索GPU说明已经开始动手训练模型。兴趣从'入门探索'到'深度实践'，推荐深度学习实战课程。"},
    ]
    for s in user_samples:
        data.append({
            "messages": [
                {"role": "system", "content": user_prompt},
                {"role": "user", "content": s["input"]},
                {"role": "assistant", "content": s["output"]},
            ]
        })

    # ── 懂世界 (5条) ──
    world_prompt = "你是一个知识渊博的助手，用简洁准确的语言回答问题。"

    world_samples = [
        {"input": "推荐系统是什么？", "output": "推荐系统是一种信息过滤技术，通过分析用户的历史行为、偏好和相似用户的行为模式，从海量内容中筛选出用户可能感兴趣的物品。核心方法包括协同过滤、基于内容的推荐和深度学习推荐。"},
        {"input": "什么是深度学习？", "output": "深度学习是机器学习的一个分支，使用多层神经网络从数据中自动学习特征表示。它模仿人脑神经元的工作方式，通过反向传播算法训练模型。广泛应用于图像识别、自然语言处理和推荐系统等领域。"},
        {"input": "解释一下注意力机制", "output": "注意力机制（Attention）让模型在处理序列数据时学会关注重要的部分。它通过计算 Query 和 Key 的相似度得到注意力权重，再用权重对 Value 加权求和。核心思想是：不是所有输入都同等重要，应该给关键信息更高的权重。"},
        {"input": "快餐和正餐的营养区别是什么？", "output": "快餐通常高热量、高脂肪、高钠，但蛋白质和膳食纤维不足；正餐更均衡，包含足量的蛋白质、蔬菜和全谷物。长期吃快餐可能导致肥胖、心血管问题。建议一周快餐不超过2次。"},
        {"input": "LLM 是什么意思？", "output": "LLM（Large Language Model，大语言模型）是使用海量文本数据训练的深度学习模型，参数量通常在数十亿级别。它能理解、生成和翻译自然语言，代表模型有 GPT、Claude、Qwen 等。核心能力来源于 Transformer 架构和规模法则。"},
    ]
    for s in world_samples:
        data.append({
            "messages": [
                {"role": "system", "content": world_prompt},
                {"role": "user", "content": s["input"]},
                {"role": "assistant", "content": s["output"]},
            ]
        })

    return data


dataset = build_demo_dataset()
print(f"\n构建了 {len(dataset)} 条训练数据")
print(f"  - 懂物料: 5 条")
print(f"  - 懂推荐: 5 条")
print(f"  - 懂用户: 5 条")
print(f"  - 懂世界: 5 条")

# ════════════════════════════════════════════════════════════
# Part 2: 加载 Tokenizer + 数据格式化
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part 2: 将数据转换为模型可用的格式")
print("=" * 60)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "OneReason-0.8B-pretrain-competition")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# tokenizer 可能没有 pad_token，给它补一个
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"Pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")

def tokenize_length(example):
    """返回 token 数量"""
    return len(tokenizer.apply_chat_template(
        example["messages"],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )[0])

# 看看数据长什么样
sample_text = tokenizer.apply_chat_template(
    dataset[0]["messages"],
    tokenize=False,
    add_generation_prompt=False,
)
print(f"\n第1条样本:")
print(sample_text[:300])

# 统计 token 长度
lengths = [tokenize_length(d) for d in dataset]
print(f"\n数据长度统计: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)//len(lengths)} tokens")

# ════════════════════════════════════════════════════════════
# Part 3: 加载模型 + 挂载 LoRA
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part 3: 加载模型 + 配置 LoRA")
print("=" * 60)

from transformers import AutoModelForCausalLM

print(f"加载基座模型: {MODEL_PATH}")
torch.cuda.empty_cache()  # 清理碎片
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

print(f"模型已加载")
print(f"  显存已分配: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
print(f"  显存已保留: {torch.cuda.memory_reserved()/1024**3:.1f} GB")
print(f"  显存空闲:   {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved())/1024**3:.1f} GB")

# ── LoRA 配置 ──
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,   # 因果语言模型
    r=8,                             # 低秩维度 (rank)，越大能力越强但参数越多
    lora_alpha=16,                   # 缩放因子，通常设为 r 的 2 倍
    lora_dropout=0.05,              # 轻微 dropout 防过拟合
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # Qwen3 的线性层
    bias="none",                     # 不训练 bias
)

print(f"\nLoRA 配置:")
print(f"  rank (r):     {lora_config.r}")
print(f"  alpha:        {lora_config.lora_alpha}")
print(f"  dropout:      {lora_config.lora_dropout}")
print(f"  target:       {lora_config.target_modules}")

# ── 挂载 LoRA ──
print(f"\n挂载 LoRA 适配器...")
torch.cuda.empty_cache()
model = get_peft_model(model, lora_config)
print(f"挂载完成")
print(f"  显存已分配: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
print(f"  显存空闲:   {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved())/1024**3:.1f} GB")

# 看看可训练参数的比例
model.print_trainable_parameters()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"\n前 LoRA: {total/1e9:.2f}B 参数, 全部可训")
print(f"后 LoRA: {total/1e9:.2f}B 参数, 可训 {trainable/1e6:.2f}M ({trainable/total*100:.2f}%)")

# ════════════════════════════════════════════════════════════
# Part 4: 手写 SFT 训练循环
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part 4: SFT 训练（手写训练循环）")
print("=" * 60)

"""
不用 SFTTrainer（有 DLL 冲突），直接手写 PyTorch 训练循环。
核心步骤: tokenize → forward → compute loss → backward → update
"""

from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# ── 准备 tokenized 数据 ──
class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.input_ids = []
        self.attention_masks = []
        self.labels = []

        for item in data:
            text = tokenizer.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors="pt",
            )
            # labels = input_ids（模型在 forward 里自动 shift）
            self.input_ids.append(encoded["input_ids"][0])
            self.attention_masks.append(encoded["attention_mask"][0])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.input_ids[idx].clone(),  # LM 任务: labels = input_ids
        }

def collate_fn(batch):
    """Padding：把不同长度的序列补到一样长"""
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids = []
    attention_masks = []
    labels = []

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
            torch.full((pad_len,), -100, dtype=torch.long)  # -100 = 忽略
        ]))

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(labels),
    }

train_dataset = SFTDataset(dataset, tokenizer, max_length=512)
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,
)

print(f"训练数据: {len(train_dataset)} 条, {len(train_loader)} 个 batch/epoch")

# ── 优化器 + 学习率调度 ──
optimizer = AdamW(model.parameters(), lr=2e-4)
total_steps = len(train_loader) * 3  # 3 epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2,
    num_training_steps=total_steps,
)

# ── 训练 ──
print(f"\n开始训练: 3 epochs, lr=2e-4, batch=2")
print(f"总步数: {total_steps}")
print("=" * 40)

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

        if (step + 1) % 3 == 0 or step == 0:
            print(f"  Epoch {epoch+1}, Step {step+1}/{len(train_loader)}, "
                  f"Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

    avg_loss = total_loss / len(train_loader)
    print(f"  → Epoch {epoch+1} 完成, Avg Loss: {avg_loss:.4f}")

print("训练完成！")

# ════════════════════════════════════════════════════════════
# Part 5: 保存 LoRA 适配器
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part 5: 保存/加载 LoRA 适配器")
print("=" * 60)

adapter_path = "./03_lora_adapter"
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)

# 看看保存的文件
import glob
files = glob.glob(os.path.join(adapter_path, "*"))
total_mb = sum(os.path.getsize(f) for f in files) / 1024**2
print(f"LoRA 适配器已保存到: {adapter_path}")
print(f"文件数: {len(files)}, 总大小: {total_mb:.1f} MB")
for f in sorted(files):
    size_kb = os.path.getsize(f) / 1024
    print(f"  {os.path.basename(f):30s} {size_kb:8.1f} KB")

print(f"\n[Tip] 比赛提交可能只需要这个 {total_mb:.1f}MB 的适配器文件！")
print(f"[Tip] 对比基座模型 {1528:.0f}MB，缩小了 {1528/total_mb:.0f}x")

# ════════════════════════════════════════════════════════════
# Part 6: 推理对比 — 训练前 vs 训练后
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part 6: SFT 前后对比")
print("=" * 60)

# ── 加载未训练的基座模型做对比 ──
print("加载未训练的基座模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

def ask_model(m, tok, question, system="你是一个推荐系统助手。"):
    """用模型回答一个问题"""
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
            **inputs,  # 正确解包 BatchEncoding
            max_new_tokens=256,
            do_sample=True,
            temperature=0.3,
            top_p=0.95,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

    response_ids = outputs[0][inputs.input_ids.shape[1]:]
    return tok.decode(response_ids, skip_special_tokens=True)


test_questions = [
    ("懂推荐", "推荐三款适合新手的烹饪食谱"),
    ("懂世界", "什么是推荐系统？"),
    ("懂物料", "分析这个视频：标题'红烧肉教程'，标签 #美食 #红烧肉 #家常菜"),
]

for category, question in test_questions:
    print(f"\n{'─'*50}")
    print(f"【{category}】问题: {question}")
    print(f"{'─'*50}")

    print("\n[SFT 前 - 基座模型]:")
    pretrain_answer = ask_model(base_model, tokenizer, question)
    print(pretrain_answer[:200])

    print("\n[SFT 后 - LoRA 模型]:")
    sft_answer = ask_model(model, tokenizer, question)
    print(sft_answer[:200])

# 清理基座模型，释放显存
del base_model
torch.cuda.empty_cache()

# ════════════════════════════════════════════════════════════
# Part 7: 用 chat.py 风格的交互来测试
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part 7: 交互测试 (可选)")
print("=" * 60)

print("""
你可以直接加载训练好的 LoRA 适配器进行对话:

    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, "./03_lora_adapter")

或者用 chat.py 加载也一样（需要把 MODEL_PATH 改到 adapter 路径）。

[Tip] 当前演示只用了 20 条训练数据，效果有限。
      真实比赛有几十万条数据 + 多轮训练，效果会天差地别。
""")

# ════════════════════════════════════════════════════════════
# 总结
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Day 3 总结")
print("=" * 60)

print("""
今天学到的:
  1. LoRA 原理 — 冻结原模型，只训低秩适配矩阵 A×B
  2. 显存优势 — 全量 8GB vs LoRA 3.6GB，省了一半多
  3. SFT 数据 — 用 ChatML 格式构建四维能力的训练集
  4. PEFT 配置 — r/alpha/target_modules 的含义和选择
  5. SFTTrainer — 几行代码完成训练循环
  6. 适配器保存 — ~20MB vs 原模型 1.6GB，缩小 80x

比赛实战提醒:
  ┌──────────────────┬──────────────────────────────────┐
  │ 你现在            │ 比赛时                           │
  ├──────────────────┼──────────────────────────────────┤
  │ 20 条数据         │ 几十万条数据                     │
  │ 3 epochs          │ 1-3 epochs (数据多不用多轮)     │
  │ 单任务混合        │ 多任务配比调优                   │
  │ 本地评测          │ 提交 CKPT → 闭源评测            │
  │ 看 loss 判断       │ LLM-as-Judge 打分               │
  └──────────────────┴──────────────────────────────────┘

[Next] 拿到比赛数据后:
  1. 分析数据分布和四个维度的配比
  2. 调 LoRA 超参 (r, alpha, lr)
  3. 多轮实验，找到最佳 checkpoint
  4. 用 chat.py -m sft 测试最终效果
""")
