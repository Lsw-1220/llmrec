# LLMRec Challenge — 快手 OneReason SFT 微调

SIGIR 2026 快手探索者 LLM-Rec 挑战赛，基于 OneReason-0.8B 的四维推荐能力 SFT。

## 环境

```bash
conda create -n llmrec python=3.10 -y
conda activate llmrec
pip install -r requirements.txt
```

## 模型下载

比赛基座模型：[OpenOneRec/OneReason-0.8B-pretrain-competition](https://huggingface.co/OpenOneRec/OneReason-0.8B-pretrain-competition)

```bash
git clone https://huggingface.co/OpenOneRec/OneReason-0.8B-pretrain-competition
cd OneReason-0.8B-pretrain-competition
git lfs pull
```

或从网盘下载后放到项目根目录。

## 脚本

| 脚本 | 用途 | 命令 |
|------|------|------|
| `01_transformer.py` | 从零理解 Transformer + Attention | `python 01_transformer.py` |
| `02_load_qwen.py` | 加载 OneReason 模型 + 推理 | `python 02_load_qwen.py` |
| `03_lora_sft.py` | LoRA SFT 微调 4 维能力 | `python 03_lora_sft.py` |
| `chat.py` | 交互对话（支持多模型） | 见下方 |

### chat.py

```bash
python chat.py                   # 默认 OneReason 基座
python chat.py -m onereason      # OneReason 基座
python chat.py -m sft            # SFT 微调后的模型
python chat.py -m qwen           # Qwen2.5-0.5B 对比
python chat.py --list            # 列出可用模型
```

对话中：
- 输入 `quit` / `exit` / `q` 退出
- 输入 `/clear` 清空对话历史
- 输入 `/len` 查看当前上下文 token 数

## 项目结构

```
├── 01_transformer.py             # Day 1: Transformer 原理
├── 02_load_qwen.py               # Day 2: 模型加载 + 推理
├── 03_lora_sft.py                # Day 3: LoRA SFT 微调
├── chat.py                       # 交互对话
├── OneReason-0.8B-pretrain-competition/  # 基座模型（本地）
├── 03_lora_adapter/              # LoRA 适配器（训练生成）
└── CONTEXT.md                    # 比赛上下文
```

## 比赛维度

| 维度 | 训练数据示例 |
|------|-------------|
| 懂物料 | 视频物料 → JSON 结构化理解 |
| 懂用户 | 行为序列 → 兴趣演化分析 |
| 懂推荐 | 用户偏好 → 推荐列表 |
| 懂世界 | 常识问答 → 准确回答 |

## 学习路径

1. ✅ `01_transformer.py` — 手写 Attention / RoPE / SwiGLU
2. ✅ `02_load_qwen.py` — Tokenizer / 4bit 量化 / Batch 推理
3. ✅ `03_lora_sft.py` — LoRA 原理 / SFT 数据 / 训练循环
4. ⏳ 比赛数据到手后：替换训练数据，调参实验
