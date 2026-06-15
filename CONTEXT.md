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

## 学习计划（5 步骤）
1. ✅ 环境搭建 — 完成
2. ✅ Transformer 架构从零理解 — `01_transformer.py` 已验证通过
3. 🔄 加载 Qwen2.5 模型推理 — `02_load_qwen.py` 正在创建
4. ⏳ LoRA/QLoRA 微调原理与实操
5. ⏳ LLaMA-Factory SFT 全流程

## 项目目录
```
D:\research\Experiment\llmrec-challenge\
├── CONTEXT.md          ← 这个文件
├── 01_transformer.py   ← Day 1 教程代码 ✅
├── 02_load_qwen.py     ← Day 2 进行中 🔄
├── 03_lora_sft.py      ← 待创建
└── 04_llamafactory/    ← 待创建
```

## 用户偏好
- 使用 conda 虚拟环境（不用 venv/global pip）
- 在 VS Code 中编辑和运行代码
- 比赛经验有限，需要从基础教起

---

**在新会话中说**："请读 CONTEXT.md 然后继续教我"
