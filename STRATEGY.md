# LLMRec 挑战赛 — 策略分析与备战指南

> 基于 SIGIR 2026 快手探索者 LLM-Rec 挑战赛规则、OneReason-0.8B 模型特性、dataexample.txt 数据分析

---

## 一、比赛考察重点

本质上考的是 **"如何在推荐场景下把一个 0.8B pretrain 模型 SFT 好"**，而不是通用 LLM 能力。

### 考察重点（按重要性排序）

| 排名 | 重点 | 为什么 |
|------|------|--------|
| **1** | **Itemic Token 理解与生成** | 这是 OneReason 区别于所有通用 LLM 的核心。itemic token（`<|video_begin|><s_a_xxx><s_b_xxx><s_c_xxx>`）是模型内部的"推荐语言"，必须让模型能精确地读和写这些 token。不懂这个，推荐维度直接 0 分 |
| **2** | **训练数据工程** | 比赛给的是 system/prompt/response 格式。如何从 13 种任务类型扩展到足够覆盖测试集的数据量、如何平衡四维度比例、如何构造 thinking 数据——这些决定了上限 |
| **3** | **推理策略（Pass@64）** | 物料和推荐维度用 Pass@64 评估——生成 64 个候选，命中一个就算对。不需要"每次都对"，而是需要"多样性够高，让正确答案出现在 top64 里" |
| **4** | **Thinking 模式控制** | 推荐维度要求 thinking+non-thinking 各 32 条。OneReason 的 chat_template 支持 `<think>...</think>`，但没有 thinking 训练数据，这是个缺口 |
| **5** | **长序列处理** | 推荐/用户数据动辄 4000-8000 tokens。RTX 4060 8GB + LoRA 能否处理？万擎平台呢？ |

### 不那么重要的

- 模型架构修改（0.8B 参数量不允许大改）
- RLHF/DPO（复杂度高、收益不确定，初赛不用考虑）
- 大规模数据清洗（比赛数据已经是标准格式）

---

## 二、四维度评估指标与数据特征

### 评估指标

| 维度 | 对应能力层 | 评估指标 | 核心挑战 |
|------|-----------|---------|---------|
| 懂物料 | R0 Perception | Pass@64 + LLM-as-Judge | itemic token ↔ 自然语言双向转换 |
| 懂用户 | R2 Evolution | F1 + 演化链推理 | 从长历史中提取逻辑链（JSON 格式） |
| 懂推荐 | R3 Recommendation | Pass@64（thinking+non-thinking 各 32） | 跨域历史 → 生成正确 itemic token |
| 懂世界 | 常识问答 | Accuracy 精确匹配 | 格式化输出选项字母 |

### 数据特征（来自 dataexample.txt 13 条样本）

```
物料 5 种: video_itemic_pattern, video, live, product, Ad
用户 2 种: Topic（token 提取）, Logic（行为逻辑链 JSON）
推荐 4 种: video, live, Product, Ad（messages 是 JSON 字符串，最长 ~8000 tokens）
世界 2 种: single（单选）, multi（多选）
```

- 推荐数据极长，用户历史跨多个域（直播/电商/视频/广告）
- 推荐数据的 messages 和 metadata 都是 JSON 字符串（不是对象），解析时需双重反序列化
- Logic 任务的 answer 是嵌套 JSON，包含逻辑链和推理验证
- 常识问答数据最短最简单，是"最容易拿分"的维度

### Itemic Token 体系

```
<|domain_begin|><s_a_xxxx><s_b_xxxx><s_c_xxxx>

5 个域:
  短视频:  <|video_begin|>   例: <|video_begin|><s_a_3334><s_b_4643><s_c_625>
  商品:    <|prod_begin|>    例: <|prod_begin|><s_a_2147><s_b_7978><s_c_5031>
  广告:    <|ad_begin|>      例: <|ad_begin|><s_a_7939><s_b_6234><s_c_4978>
  直播:    <|living_begin|>  例: <|living_begin|><s_a_4515><s_b_6234><s_c_6278>
  通用:    <|sid_begin|>     例: <|sid_begin|><s_a_340><s_b_6566><s_c_5603>

三层 codebook，每层 8192 个码（0-8191）
子 token 不标记为 special（normalized=true），参与正常 token embedding
```

### Thinking 模式

- chat_template 支持 `enable_thinking` 参数
- `enable_thinking=True`: 模型先 `<think>...</think>` 再输出答案
- `enable_thinking=False`: 插入空 `<think>\n\n</think>` 跳过思考
- 训练数据中无 `<think>` 块 → 对应 non-thinking 模式
- 推荐维度要求 thinking + non-thinking **各 32 条**
- 需要额外构造 thinking 训练数据，或用 `/no_think` 和 `/think` 控制

---

## 三、7/1 前备战清单

当前进度：✅ 环境搭建 → ✅ Transformer 原理 → ✅ 模型加载推理 → ✅ LoRA SFT 脚本

### 立即可做（本地 RTX 4060）

**1. 跑通完整训练流程（最重要）**
```bash
conda activate llmrec
python 03_lora_sft.py  # 用 dataexample.txt 的 13 条数据
python chat.py -m sft   # 验证微调后模型
python chat.py -m onereason  # 对比基座
```

**2. 深度理解 itemic token 体系**
- 用 `chat.py -m onereason` 对基座模型做实验
- 测试：给 itemic token → 模型能否解读？给描述 → 模型能否生成正确的 token？
- 这是基座模型的 pretrain 能力，SFT 只是"教会它按比赛格式回答"

**3. 理解 chat_template 的 thinking 机制**
- 测试 `enable_thinking=True/False` 的输出差异
- 确认 non-thinking 模式下输出格式是否符合比赛要求

**4. 构造本地评估脚本**
- 按比赛指标写简易评测：对 13 条样本生成 64 次 → 算 Pass@64
- 这是后续调参的依据，没有本地评估就是盲调

### 需要等比赛数据的

**5. 万擎平台熟悉**
- 了解提交方式、GPU 配置、推理时间限制
- 可能在万擎上训练（比本地 4060 更强）

**6. 数据扩展策略**
- 比赛数据到手后，如何从少量样本扩展：
  - 格式变换（同义改写 system prompt）
  - 数据增强（用户历史截断/打乱）
  - 按维度平衡采样比例

---

## 四、上分策略（初赛 7.1-7.31）

### 第一周：基线建立

1. **用比赛数据跑一轮 LoRA SFT**，提交看基线分数
2. **四维度分别看**，哪个最弱先攻哪个
3. 判断：**懂世界（常识问答）最容易拿分**——Accuracy 精确匹配，模型 pretrain 已有知识，只需教会它输出格式 `"正确答案是 B"`

### 第二周：Pass@64 优化（大分项）

4. **物料理解 + 推荐维度** 用 Pass@64 评估，这是"最大分池"
5. 关键技巧：
   - **高 temperature 采样**（0.7-1.0）增加多样性
   - **分 thinking/non-thinking 各 32 条**
   - **repetition_penalty** 防止重复生成同一个 itemic token
   - **多轮采样**：不同 temperature 采样混合

### 第三周：数据工程

6. **训练数据配比**：推荐维度数据量 >> 其他维度（因为输入最长、任务最复杂）
7. **Thinking 数据构造**：
   - 最简单方案：在 prompt 加 `/no_think` → non-thinking；加 `/think` + 手写推理链 → thinking
   - 进阶：用更大的模型（Qwen3-8B）生成 thinking trace 作为训练数据

### 第四周：调参 + 融合

8. **LoRA 参数调优**：r=8→16→32，看哪个效果最好
9. **学习率搜索**：5e-5 附近
10. **多 checkpoint 融合**：不同 epoch 的 LoRA 适配器做 ensemble

---

## 五、OneReason 模型性能参考

### 跨域推荐（OneReason-8B，仅供参考趋势）

| 模型 | C-Video Pass@64 | C-Product Pass@64 | C-Ad Pass@64 | C-Live Pass@64 |
|------|:---:|:---:|:---:|:---:|
| LC-Rec-SFT-Only-8B | 0.22 | 0.06 | 2.83 | 0.89 |
| LC-Rec-PT-SFT-8B | 1.49 | 3.95 | 15.85 | 19.32 |
| OneReason SFT non-thinking | 1.33 | 3.94 | 15.73 | 18.05 |
| OneReason RFT thinking | **2.41** | **5.47** | **17.78** | **21.10** |

**关键洞察**：
- SFT-only（无 pretrain）性能极差 → **必须用 OneReason pretrain 基座**
- RFT（RL）> SFT → 但初赛可能只需 SFT 就够
- thinking 模式在推荐上不一定优于 non-thinking（C-Video thinking=0.71 vs non-thinking=1.33）
- 但 thinking + non-thinking 混合 64 条可以取长补短

### R0-R2 推理任务

| 模型 | R0 物料理解 | R0 Grounding | R2 Selection | R2 Topic Gen |
|------|:---:|:---:|:---:|:---:|
| OneReason SFT non-thinking | 36.84 | 3.95 | 35.07 | 33.87 |
| OneReason RFT thinking | 36.78 | 1.35 | 42.42 | 39.57 |

**关键洞察**：
- 物料理解上限约 37%，grounding 极低（~5%）→ 这是难点
- thinking 对 R2 演化任务帮助大（42 vs 35），对 R0 感知帮助不大

---

## 六、关键风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| Thinking 训练数据缺失 | 推荐维度 32 条 thinking 输出可能格式错误 | 用大模型蒸馏 thinking trace |
| 推荐数据极长（8000+ tokens） | 本地 4060 跑不动，需万擎平台 | 确认万擎 GPU 配置 |
| Itemic token 生成不准 | Pass@64 全错 | 强化物料理解维度的训练比例 |
| 过拟合 13 种格式 | 测试集可能有未见过的任务变体 | 数据增强 + 适当 dropout |
| Pad token 混淆 | 训练用 `<|endoftext|>` (151643)，推理用 `<|im_end|>` (151645) 导致异常 | 统一使用原始配置，不覆盖 pad_token |

---

## 七、03_lora_sft.py 改进记录

### 已修复的 Bug

| # | 严重度 | 问题 | 修复 |
|---|--------|------|------|
| 1 | **致命** | Labels 未遮蔽 — loss 在全部 token 上计算 | 改用 SFTTrainer，自动只对 assistant 回复算 loss |
| 2 | **致命** | Part 6 同时加载两个模型，8GB OOM | 先推理 SFT 模型 → 释放 → 加载 base → 对比 |
| 3 | **高** | max_length=1024 太短，推荐数据被截断 | 默认改为 2048 + 截断警告 |
| 4 | **高** | 无梯度累积，batch=1 极不稳定 | SFTTrainer 配置 gradient_accumulation_steps=8 |
| 5 | **中** | pad_token 被错误覆盖 | 保留 OneReason 原始 pad_token `<|endoftext|>` |
| 6 | **中** | lr=2e-4 对 LoRA SFT 偏高 | 默认改为 5e-5，加 argparse 支持 |

### 新增功能

- CLI 参数：`--lr`, `--epochs`, `--max_len`, `--grad_accum`, `--lora_r`, `--lora_alpha`
- 手写训练循环保留为教学注释（含 label 遮蔽的 `_find_assistant_start` 方法）
- 四维度测试问题（懂物料/懂世界/懂推荐/懂用户）
- 数据长度统计 + 截断警告
- chat.py eos_token_id 修复为 `[151645, 151643]`

---

## 八、时间线

| 日期 | 事件 | 行动 |
|------|------|------|
| 6.13-6.29 | 报名注册 | ✅ 已完成 |
| **6.15-6.30** | **赛前准备** | 跑通流程 + 理解 itemic token + 本地评估脚本 |
| 7.1-7.31 | 初赛 | 拿数据 → 基线 → Pass@64 优化 → 数据工程 |
| 8.1-8.31 | 复赛（引入 LLM-as-Judge） | 优化回答质量（不仅是格式正确） |
| 9.1-9.15 | 代码复现审核 | 确保代码可复现 |
| 9 月底 | 决赛答辩 | 准备 PPT + 技术报告 |

---

**核心结论**：这次比赛的胜负手不在模型架构，而在 **itemic token 理解 + Pass@64 采样策略 + thinking 数据构造**。先跑通流程，再逐个击破。
