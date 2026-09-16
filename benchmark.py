from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:  # Keep metric/scheduler imports usable in CPU-only CI environments.
    import torch
except ImportError:  # pragma: no cover - exercised only without dependencies
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover - exercised only without dependencies
    AutoModelForCausalLM = None
    AutoTokenizer = None

from arrival_scheduler import TOKEN_TIMING_BOUNDARY, run_virtual_arrival_schedule
from metrics import percentile, summarize, summarize_arrival_scheduling


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT = Path("results/transformers_baseline.json")
PROMPT_PREFIX = "Explain one systems concept clearly in two short sentences:"


def synchronize() -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def make_prompts(count: int) -> list[str]:
    return [f"{PROMPT_PREFIX} request {index}." for index in range(count)]


def load_model(model_name: str):
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError(
            "PyTorch and Transformers are required to load a benchmark model"
        )
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

    if torch is not None and torch.cuda.is_available():
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
        if torch is not None and torch.cuda.is_available()
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


def run_arrival(
    tokenizer,
    model,
    prompts: list[str],
    max_batch_size: int,
    batch_wait_timeout_ms: float,
    arrival_interval_ms: float,
    max_new_tokens: int,
) -> dict:
    """Run a deterministic controlled-arrival batching experiment.

    Arrival times are represented on a virtual clock (request ``i`` arrives at
    ``i * arrival_interval_ms``).  The scheduler dispatches batches when they
    reach ``max_batch_size`` or the oldest request reaches the timeout.  Model
    execution remains sequential through ``generate_batch``; no threads or
    sleeps are introduced, making this mode safe to run repeatedly.

    ``model.generate`` is a completion API here, so TTFT/ITL are explicitly
    left unset.  ``token_timing_boundary`` identifies the measured boundary as
    the complete non-streaming batch generation call.
    """

    try:
        arrival_interval = float(arrival_interval_ms)
    except (TypeError, ValueError) as exc:
        raise ValueError("arrival_interval_ms must be finite and non-negative") from exc
    if not math.isfinite(arrival_interval) or arrival_interval < 0.0:
        raise ValueError("arrival_interval_ms must be finite and non-negative")
    arrival_times_ms = [index * arrival_interval for index in range(len(prompts))]
    batch_results: dict[int, dict] = {}

    def execute_batch(request_ids: tuple[int, ...], batch_index: int) -> dict:
        prompt_batch = [prompts[index] for index in request_ids]
        started = time.perf_counter()
        try:
            result = generate_batch(tokenizer, model, prompt_batch, max_new_tokens)
            batch_results[batch_index] = {
                **result,
                "request_ids": request_ids,
            }
            # Use the measured generate_batch duration; the scheduler uses it
            # as virtual execution time while retaining deterministic arrivals.
            return {"batch_execution_ms": result["elapsed_ms"]}
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            message = f"{type(exc).__name__}: {exc}"
            batch_results[batch_index] = {
                "elapsed_ms": elapsed_ms,
                "input_tokens": [],
                "output_tokens": [],
                "peak_memory_mb": 0.0,
                "request_ids": request_ids,
            }
            return {
                "batch_execution_ms": elapsed_ms,
                "error": True,
                "error_message": message,
            }

    records = run_virtual_arrival_schedule(
        arrival_times_ms,
        max_batch_size=max_batch_size,
        batch_wait_timeout_ms=batch_wait_timeout_ms,
        execution_provider=execute_batch,
    )

    # Add request-level token/memory details without changing the scheduler's
    # model-free contract.  Successful batched generation returns one token
    # count per request; errors intentionally have no fabricated token count.
    peak_memory_mb = 0.0
    input_tokens: list[int] = []
    output_tokens: list[int] = []
    batch_latencies: dict[int, float] = {}
    for batch_index, result in batch_results.items():
        peak_memory_mb = max(peak_memory_mb, float(result.get("peak_memory_mb", 0.0)))
        input_tokens.extend(int(value) for value in result.get("input_tokens", []))
        output_tokens.extend(int(value) for value in result.get("output_tokens", []))
        batch_latencies[batch_index] = float(result.get("elapsed_ms", 0.0))
    for record in records:
        result = batch_results.get(int(record["batch_id"]), {})
        request_index = int(record["request_id"])
        request_ids = tuple(int(value) for value in result.get("request_ids", ()))
        request_offset = request_ids.index(request_index) if request_index in request_ids else -1
        request_inputs = result.get("input_tokens", [])
        request_outputs = result.get("output_tokens", [])
        if not record["error"] and 0 <= request_offset < len(request_inputs):
            record["input_tokens"] = int(request_inputs[request_offset])
        if not record["error"] and 0 <= request_offset < len(request_outputs):
            record["output_tokens"] = int(request_outputs[request_offset])

    completed = sum(1 for record in records if not record["error"])
    errors = len(records) - completed
    wall_time_ms = max(
        (float(record["batch_end_time_ms"]) for record in records),
        default=0.0,
    )
    request_latencies = [
        float(record["end_to_end_latency_ms"])
        for record in records
        if not record["error"]
    ]
    batch_latency_values = list(batch_latencies.values())
    batch_sizes = [int(record["batch_size"]) for record in records if not record["error"]]
    arrival_summary = summarize_arrival_scheduling(records)
    elapsed_seconds = wall_time_ms / 1000.0
    return {
        "mode": "arrival",
        "requested": len(prompts),
        "completed": completed,
        "errors": errors,
        "error_rate": errors / len(prompts) if prompts else 0.0,
        "wall_time_ms": wall_time_ms,
        "throughput_requests_per_second": completed / elapsed_seconds if elapsed_seconds else 0.0,
        "generated_tokens_per_second": sum(output_tokens) / elapsed_seconds if elapsed_seconds else 0.0,
        "request_latency_ms": summarize(request_latencies),
        "batch_latency_ms": summarize(batch_latency_values),
        "queue_wait_ms": arrival_summary["queue_wait_ms"],
        "batch_execution_ms": arrival_summary["batch_execution_ms"],
        "input_tokens_total": sum(input_tokens),
        "output_tokens_total": sum(output_tokens),
        "peak_gpu_memory_mb": peak_memory_mb,
        "batch_size_distribution": dict(Counter(batch_sizes)),
        "arrival_scheduling": arrival_summary,
        "arrival_interval_ms": arrival_interval,
        "batch_wait_timeout_ms": float(batch_wait_timeout_ms),
        "max_batch_size": int(max_batch_size),
        "clock": "virtual_ms",
        "requests": records,
        "token_timing_boundary": TOKEN_TIMING_BOUNDARY,
        "ttft_ms": None,
        "itl_ms": None,
    }


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
    if torch is None or not torch.cuda.is_available():
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
    parser.add_argument("--mode", choices=["direct", "batched", "arrival", "both"], default="both")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument(
        "--batch-size",
        "--max-batch-size",
        dest="batch_size",
        type=int,
        default=4,
        help="Maximum requests per generated batch",
    )
    parser.add_argument(
        "--arrival-interval-ms",
        type=float,
        default=100.0,
        help="Virtual inter-arrival interval for --mode arrival (milliseconds)",
    )
    parser.add_argument(
        "--batch-wait-timeout-ms",
        type=float,
        default=25.0,
        help="Maximum virtual wait for the oldest queued request in arrival mode",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if (
        args.requests < 1
        or args.batch_size < 1
        or args.max_new_tokens < 1
        or args.warmup < 0
        or not math.isfinite(args.arrival_interval_ms)
        or args.arrival_interval_ms < 0
        or not math.isfinite(args.batch_wait_timeout_ms)
        or args.batch_wait_timeout_ms < 0
    ):
        parser.error(
            "requests, batch-size, and max-new-tokens must be positive; "
            "arrival interval and batch wait timeout must be finite and "
            "non-negative; warmup cannot be negative"
        )
    return args


def main() -> None:
    args = parse_args()
    if torch is None or not torch.cuda.is_available():
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
    if args.mode == "arrival":
        results.append(
            run_arrival(
                tokenizer,
                model,
                prompts,
                args.batch_size,
                args.batch_wait_timeout_ms,
                args.arrival_interval_ms,
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
            "arrival_interval_ms": args.arrival_interval_ms,
            "batch_wait_timeout_ms": args.batch_wait_timeout_ms,
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
