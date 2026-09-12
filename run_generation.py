import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    return tokenizer, model


def make_prompt(tokenizer, text: str) -> str:
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_one(tokenizer, model, text: str, max_new_tokens: int):
    prompt = make_prompt(tokenizer, text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000

    input_tokens = int(inputs["attention_mask"].sum().item())
    new_tokens = int(output.shape[1] - inputs["input_ids"].shape[1])
    generated_text = tokenizer.decode(
        output[0, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )
    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / 1024**2
        if torch.cuda.is_available()
        else 0.0
    )

    return {
        "model": model.config.name_or_path,
        "device": str(model.device),
        "input_tokens": input_tokens,
        "output_tokens": new_tokens,
        "latency_ms": elapsed_ms,
        "generated_tokens_per_second": (
            new_tokens / (elapsed_ms / 1000) if elapsed_ms else 0.0
        ),
        "peak_gpu_memory_mb": peak_memory_mb,
        "text": generated_text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="Explain why batching can improve GPU inference throughput in two sentences.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this project")

    tokenizer, model = load_model(args.model)
    result = generate_one(tokenizer, model, args.prompt, args.max_new_tokens)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
