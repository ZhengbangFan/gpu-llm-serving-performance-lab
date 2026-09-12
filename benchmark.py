from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT = Path("results/transformers_baseline.json")
PROMPT_PREFIX = "Explain one systems concept clearly in two short sentences:"


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile_value / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_prompts(count: int) -> list[str]:
    return [f"{PROMPT_PREFIX} request {index}." for index in range(count)]


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


def generate_batch(tokenizer, model, prompts: list[str], max_new_tokens: int) -> dict:
    encoded_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    encoded = tokenizer(
        encoded_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    encoded = {name: value.to(model.device) for name, value in encoded.items()}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    input_tokens = [int(value) for value in encoded["attention_mask"].sum(dim=1).tolist()]
    output_width = int(output.shape[1] - encoded["input_ids"].shape[1])
    output_tokens = [output_width for _ in prompts]
    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / 1024**2
        if torch.cuda.is_available()
        else 0.0
    )
    return {
        "elapsed_ms": elapsed_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "peak_memory_mb": peak_memory_mb,
    }


def run_direct(tokenizer, model, prompts: list[str], max_new_tokens: int) -> dict:
    latencies = []
    input_tokens = []
    output_tokens = []
    peak_memory_mb = 0.0
    errors = 0
    started = time.perf_counter()

    for prompt in prompts:
        try:
            result = generate_batch(tokenizer, model, [prompt], max_new_tokens)
            latencies.append(result["elapsed_ms"])
            input_tokens.extend(result["input_tokens"])
            output_tokens.extend(result["output_tokens"])
            peak_memory_mb = max(peak_memory_mb, result["peak_memory_mb"])
        except Exception:
            errors += 1

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return summarize_result(
        mode="direct",
        requested=len(prompts),
        completed=len(latencies),
        errors=errors,
        elapsed_ms=elapsed_ms,
        request_latencies=latencies,
        batch_latencies=latencies,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        batch_sizes=[1 for _ in latencies],
        peak_memory_mb=peak_memory_mb,
    )


def run_batched(
    tokenizer,
    model,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
) -> dict:
    request_latencies = []
    batch_latencies = []
    input_tokens = []
    output_tokens = []
    batch_sizes = []
    peak_memory_mb = 0.0
    errors = 0
    started = time.perf_counter()

    for offset in range(0, len(prompts), batch_size):
        prompt_batch = prompts[offset : offset + batch_size]
        try:
            result = generate_batch(tokenizer, model, prompt_batch, max_new_tokens)
            batch_latency = result["elapsed_ms"]
            batch_latencies.append(batch_latency)
            request_latencies.extend([batch_latency] * len(prompt_batch))
            input_tokens.extend(result["input_tokens"])
            output_tokens.extend(result["output_tokens"])
            batch_sizes.extend([len(prompt_batch)] * len(prompt_batch))
            peak_memory_mb = max(peak_memory_mb, result["peak_memory_mb"])
        except Exception:
            errors += len(prompt_batch)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return summarize_result(
        mode="batched",
        requested=len(prompts),
        completed=len(request_latencies),
        errors=errors,
        elapsed_ms=elapsed_ms,
        request_latencies=request_latencies,
        batch_latencies=batch_latencies,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        batch_sizes=batch_sizes,
        peak_memory_mb=peak_memory_mb,
    )


def summarize_result(
    mode: str,
    requested: int,
    completed: int,
    errors: int,
    elapsed_ms: float,
    request_latencies: list[float],
    batch_latencies: list[float],
    input_tokens: list[int],
    output_tokens: list[int],
    batch_sizes: list[int],
    peak_memory_mb: float,
) -> dict:
    elapsed_seconds = elapsed_ms / 1000.0
    total_output_tokens = sum(output_tokens)
    return {
        "mode": mode,
        "requested": requested,
        "completed": completed,
        "errors": errors,
        "error_rate": errors / requested if requested else 0.0,
        "wall_time_ms": elapsed_ms,
        "throughput_requests_per_second": (
            completed / elapsed_seconds if elapsed_seconds else 0.0
        ),
        "generated_tokens_per_second": (
            total_output_tokens / elapsed_seconds if elapsed_seconds else 0.0
        ),
        "request_latency_ms": summarize(request_latencies),
        "batch_latency_ms": summarize(batch_latencies),
        "input_tokens_total": sum(input_tokens),
        "output_tokens_total": total_output_tokens,
        "peak_gpu_memory_mb": peak_memory_mb,
        "batch_size_distribution": dict(Counter(batch_sizes)),
    }


def gpu_metadata() -> dict:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    properties = torch.cuda.get_device_properties(0)
    return {
        "cuda_available": True,
        "device": torch.cuda.get_device_name(0),
        "total_memory_mb": properties.total_memory / 1024**2,
        "torch_cuda_build": torch.version.cuda,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark direct vs tensor-batched generation")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=["direct", "batched", "both"], default="both")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.requests < 1 or args.batch_size < 1 or args.max_new_tokens < 1 or args.warmup < 0:
        parser.error("requests, batch-size, and max-new-tokens must be positive; warmup cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU serving benchmark")

    tokenizer, model = load_model(args.model)
    warmup_prompts = make_prompts(args.warmup)
    if warmup_prompts:
        generate_batch(tokenizer, model, warmup_prompts, args.max_new_tokens)

    prompts = make_prompts(args.requests)
    results = []
    if args.mode in {"direct", "both"}:
        results.append(run_direct(tokenizer, model, prompts, args.max_new_tokens))
    if args.mode in {"batched", "both"}:
        results.append(
            run_batched(
                tokenizer,
                model,
                prompts,
                args.batch_size,
                args.max_new_tokens,
            )
        )

    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "gpu": gpu_metadata(),
            "model": args.model,
        },
        "config": {
            "requests": args.requests,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "warmup": args.warmup,
            "prompt_template": PROMPT_PREFIX,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
