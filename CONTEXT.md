# LLMRec 挑战赛 — 上下文记忆文件

## 赛事信息
- **名称**：快手探索者 LLM-Rec 挑战赛 @SIGIR 2026
- **平台**：https://ks-llmrec.streamlake.com
- **核心任务**：基于 LLM 的 SFT，覆盖四个维度：
  1. **物料理解** — 解析视频模态和语义（Pass@64 / LLM-as-Judge）
  2. **用户兴趣演化** — 从交互历史洞察需求演化（F1 + 演化链推理）
  3. **推荐物料** — 基于跨域历史预测偏好（Pass@64，thinking+non-thinking 各 32 条）
  4. **常识问答** — 推理与选择题（Accuracy 精确匹配）
- **数据格式**：system / prompt / response 三字段
- **算力平台**：快手万擎（WanQing）

## 时间线
- 6.13-6.29：报名注册（已报名）
- 7.1-7.31：初赛
- 8.1-8.31：复赛（引入 LLM-as-Judge）
- 9.1-9.15：代码复现审核
- 9月底：决赛答辩

## 环境
- **conda 环境名**：`llmrec`
- **Python**：3.10
- **GPU**：NVIDIA RTX 4060 Laptop (8GB VRAM)
- **关键包**：torch 2.5.1, transformers 5.12, peft 0.19, bitsandbytes 0.49, trl 1.6, accelerate 1.14

## 完整学习与备战计划

### 阶段一：基础能力（✅ 已完成）
1. ✅ 环境搭建（conda llmrec, RTX 4060 8GB）
2. ✅ Transformer 架构从零理解 — `01_transformer.py`
3. ✅ OneReason 模型加载与推理 — `02_load_qwen.py`

### 阶段二：SFT 微调技术（🔄 进行中）
4. 🔄 LoRA/QLoRA 微调 — `03_lora_sft_learn.py` 逐 Part Debug

   | Part | 内容 | 状态 |
   |------|------|------|
   | Part 1 | 数据解析（4 维度 13 种任务） | ✅ 已跑通 |
   | Part 2 | Tokenizer + chat_template | ⏳ 下一步 |
   | Part 3 | 模型加载 + LoRA 挂载 | ⏳ |
   | Part 4 | SFTTrainer 训练 | ⏳ 10-20min，需 GPU |
   | Part 5 | 保存 LoRA 适配器 | ⏳ |
   | Part 6 | SFT 前后对比 | ⏳ |

### 阶段三：比赛专项（⏳ 待开始）
5. ⏳ **本地评估脚本** — 按比赛指标（Pass@64/F1/Accuracy）写评测
6. ⏳ **Thinking 模式实验** — enable_thinking=True/False 输出差异
7. ⏳ **推理策略优化** — 高 temperature 多样性采样，Pass@64 最大化
8. ⏳ **数据工程** — 比赛数据到手后的清洗、增强、维度平衡

### 阶段四：调参 & 实战（⏳ 待开始）
9. ⏳ **LoRA 超参搜索** — r=8/16/32, lr, epochs
10. ⏳ **万擎平台部署** — 提交格式、GPU 资源、推理时间限制
11. ⏳ **多 checkpoint 融合 / ensemble**

### 阶段五：复赛 & 答辩（8-9 月）
12. ⏳ LLM-as-Judge 优化（回答质量，不仅是格式正确）
13. ⏳ 代码复现审核准备
14. ⏳ 决赛答辩材料

## 项目目录
```
D:\research\Experiment\llmrec-challenge\
├── CONTEXT.md           ← 这个文件（上下文记忆）
├── STRATEGY.md          ← 比赛策略分析
├── README.md            ← 项目说明
├── 01_transformer.py    ← Day 1: Transformer 原理 ✅
├── 02_load_qwen.py      ← Day 2: 模型加载+推理 ✅
├── 03_lora_sft.py       ← Day 3: LoRA SFT 正式脚本 ✅
├── 03_lora_sft_learn.py ← Day 3: 交互式学习版（可分 Part Debug）
├── chat.py              ← 交互对话（多模型切换）
├── dataexample.txt      ← 比赛训练数据示例（13 种任务）
├── OneReason-0.8B-pretrain-competition/  ← 基座模型
├── 03_lora_adapter/     ← LoRA 适配器（训练生成）
└── .vscode/launch.json  ← Debug 配置
```

## 用户偏好
- 使用 conda 虚拟环境（不用 venv/global pip）
- 在 VS Code 中编辑和运行代码
- 比赛经验有限，需要从基础教起

---

**在新会话中说**："请读 CONTEXT.md 然后继续教我"
