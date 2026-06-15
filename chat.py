"""
============================================================
模型交互对话 — 支持 OneReason / Qwen 自由切换
用法:
    python chat.py                          # 默认 OneReason
    python chat.py -m qwen                  # 切到 Qwen
    python chat.py --model onereason        # 切到 OneReason
    python chat.py --list                   # 列出可用模型
输入 quit / exit / q 退出
============================================================
"""
import argparse
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── 模型注册表（加新模型在这里加一行就行）─────────────
MODELS = {
    "onereason": os.path.join(ROOT, "OneReason-0.8B-pretrain-competition"),
    "qwen":      "Qwen/Qwen2.5-0.5B-Instruct",
    "sft":       os.path.join(ROOT, "03_lora_adapter"),   # LoRA 适配器
}
DEFAULT_MODEL = "onereason"
# ─────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="LLM 交互对话")
parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                    choices=list(MODELS.keys()),
                    help=f"选择模型 (默认: {DEFAULT_MODEL})")
parser.add_argument("--list", action="store_true",
                    help="列出所有可用模型")
args = parser.parse_args()

if args.list:
    print("可用模型:")
    for name, path in MODELS.items():
        tag = " (默认)" if name == DEFAULT_MODEL else ""
        print(f"  {name:12s} → {path}{tag}")
    exit(0)

MODEL_PATH = MODELS[args.model]
print(f"加载模型: {args.model} → {MODEL_PATH}")

if args.model == "sft":
    # SFT 模式：基座模型 + LoRA 适配器
    BASE_PATH = MODELS["onereason"]
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model = model.merge_and_unload()  # 合并 LoRA 到基座，推理更快
    print(f"[SFT 模式] 基座 + LoRA 适配器已合并")
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

print(f"就绪！显存: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
print("多轮对话模式 — 上下文自动累积")
print("输入 /clear 清空历史, /len 看 token 数, quit 退出\n")

# 对话历史（持续累积，带上下文记忆）
messages = [{"role": "system", "content": "你是一个推荐系统助手。"}]

while True:
    user_input = input(">>> ").strip()
    if not user_input:
        continue
    if user_input.lower() in ("quit", "exit", "q"):
        break
    if user_input == "/clear":
        messages = [messages[0]]  # 只保留 system prompt
        print("[历史已清空]\n")
        continue
    if user_input == "/len":
        temp = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            return_tensors="pt"
        )
        print(f"[当前上下文: {temp.shape[1]} tokens]\n")
        continue

    messages.append({"role": "user", "content": user_input})

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    print(f"[输入 {inputs.input_ids.shape[1]} tokens, 生成中...]")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")],
        )

    response_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)

    # 把助手回复也加入历史
    messages.append({"role": "assistant", "content": response})

    print(f"\n{response}\n")
    print("-" * 50)
