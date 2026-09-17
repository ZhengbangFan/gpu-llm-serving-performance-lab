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
from metrics import (
    percentile,
    summarize,
    summarize_arrival_scheduling,
    summarize_token_timing,
)


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT = Path("results/transformers_baseline.json")
PROMPT_PREFIX = "Explain one systems concept clearly in two short sentences:"
STREAMING_TIMING_BOUNDARY = "local_manual_decode_cuda_synchronized"


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


def _model_device(model):
    """Return the device used by a loaded model, including simple test doubles."""

    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration) as exc:
        raise RuntimeError("model must expose a device or parameters") from exc


def _past_key_values(model_output):
    """Extract the cache from both model-output objects and tuple-like outputs."""

    past = getattr(model_output, "past_key_values", None)
    if past is None and isinstance(model_output, (tuple, list)) and len(model_output) > 1:
        past = model_output[1]
    if past is None:
        raise RuntimeError("model.forward did not return past_key_values")
    return past


def generate_streaming_batch(
    tokenizer,
    model,
    prompts: list[str],
    max_new_tokens: int,
) -> dict:
    """Run a local, CUDA-synchronized, fixed-length manual decode.

    This is intentionally separate from :meth:`model.generate`: every token
    is selected with greedy ``argmax`` and fed through ``model.forward`` with
    ``past_key_values``/``use_cache``.  It measures local GPU work only; it is
    not HTTP result streaming and does not implement continuous batching.
    """

    if torch is None:
        raise RuntimeError("PyTorch is required for streaming generation")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise ValueError("max_new_tokens must be a positive integer")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")

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
    device = _model_device(model)
    encoded = {name: value.to(device) for name, value in encoded.items()}
    attention_mask = encoded["attention_mask"]
    input_tokens = [int(value) for value in attention_mask.sum(dim=1).tolist()]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ``generate`` derives position ids from the attention mask for left-
    # padded inputs.  Manual forwarding must do the same or pad columns would
    # shift the rotary positions of the real prompt tokens.
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 1)

    # Synchronization before the clock starts makes the first timestamp a
    # completed prefill boundary rather than time spent draining old work.
    synchronize()
    start_time_ms = time.perf_counter() * 1000.0
    with torch.inference_mode():
        prefill_output = model.forward(
            input_ids=encoded["input_ids"],
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        synchronize()
        prefill_done_ms = time.perf_counter() * 1000.0

        logits = prefill_output.logits[:, -1, :]
        next_tokens = torch.argmax(logits, dim=-1)
        # The first timestamp is after both first-token selection and its GPU
        # synchronization, so TTFT represents completed work.
        synchronize()
        token_timestamps_ms = [time.perf_counter() * 1000.0]
        past_key_values = _past_key_values(prefill_output)

        for _ in range(1, max_new_tokens):
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ),
                dim=1,
            )
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
            decode_output = model.forward(
                input_ids=next_tokens.unsqueeze(-1),
                attention_mask=attention_mask,
                position_ids=position_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = decode_output.logits[:, -1, :]
            next_tokens = torch.argmax(logits, dim=-1)
            past_key_values = _past_key_values(decode_output)
            synchronize()
            token_timestamps_ms.append(time.perf_counter() * 1000.0)

    timing = summarize_token_timing(
        token_timestamps_ms,
        start_time_ms=start_time_ms,
    )
    prefill_ms = prefill_done_ms - start_time_ms
    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / 1024**2
        if torch.cuda.is_available()
        else 0.0
    )
    output_tokens = [max_new_tokens for _ in prompts]
    return {
        "prefill_ms": prefill_ms,
        "ttft_ms": timing["ttft_ms"],
        "batch_itl_values_ms": timing["itl_values_ms"],
        "batch_itl_ms": timing["itl_ms"],
        "decode_ms": timing["decode_ms"],
        "total_generation_latency_ms": timing["total_generation_ms"],
        "total_generation_ms": timing["total_generation_ms"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "peak_memory_mb": peak_memory_mb,
        "timing_boundary": STREAMING_TIMING_BOUNDARY,
        "token_timing_boundary": STREAMING_TIMING_BOUNDARY,
    }


def summarize_streaming_result(
    *,
    requested: int,
    completed: int,
    errors: int,
    elapsed_ms: float,
    batch_results: list[dict],
    requests: list[dict],
    batch_sizes: list[int],
    peak_memory_mb: float,
    batch_records: list[dict] | None = None,
) -> dict:
    """Build the JSON-safe batch/request schema for local streaming runs."""

    prefill_values = [float(result["prefill_ms"]) for result in batch_results]
    ttft_values = [float(result["ttft_ms"]) for result in batch_results]
    decode_values = [float(result["decode_ms"]) for result in batch_results]
    total_values = [
        float(result["total_generation_latency_ms"]) for result in batch_results
    ]
    batch_itl_values = [
        float(value)
        for result in batch_results
        for value in result.get("batch_itl_values_ms", [])
    ]
    output_tokens = [
        int(request["output_tokens"])
        for request in requests
        if request.get("error") is None and request.get("output_tokens") is not None
    ]
    request_total_values = [
        float(request["total_generation_latency_ms"])
        for request in requests
        if request.get("error") is None
        and request.get("total_generation_latency_ms") is not None
    ]
    if batch_records is None:
        batch_records = [
            {
                "batch_id": int(result.get("batch_id", batch_index)),
                "batch_size": int(batch_sizes[batch_index])
                if batch_index < len(batch_sizes)
                else None,
                "prefill_ms": float(result["prefill_ms"]),
                "ttft_ms": float(result["ttft_ms"]),
                "batch_itl_values_ms": list(result.get("batch_itl_values_ms", [])),
                "batch_itl_ms": dict(result.get("batch_itl_ms", summarize([]))),
                "decode_ms": float(result["decode_ms"]),
                "total_generation_latency_ms": float(
                    result["total_generation_latency_ms"]
                ),
                "total_generation_ms": float(
                    result.get(
                        "total_generation_ms",
                        result["total_generation_latency_ms"],
                    )
                ),
                "output_tokens": [int(value) for value in result.get("output_tokens", [])],
                "peak_gpu_memory_mb": float(result.get("peak_memory_mb", 0.0)),
                "timing_boundary": STREAMING_TIMING_BOUNDARY,
                "token_timing_boundary": STREAMING_TIMING_BOUNDARY,
                "error": None,
            }
            for batch_index, result in enumerate(batch_results)
        ]
    elapsed_seconds = float(elapsed_ms) / 1000.0
    total_output_tokens = sum(output_tokens)
    return {
        "mode": "streaming",
        "requested": requested,
        "completed": completed,
        "errors": errors,
        "error_rate": errors / requested if requested else 0.0,
        "wall_time_ms": float(elapsed_ms),
        "throughput_requests_per_second": (
            completed / elapsed_seconds if elapsed_seconds else 0.0
        ),
        "generated_tokens_per_second": (
            total_output_tokens / elapsed_seconds if elapsed_seconds else 0.0
        ),
        "request_latency_ms": summarize(request_total_values),
        "batch_latency_ms": summarize(total_values),
        "prefill_ms": summarize(prefill_values),
        "ttft_ms": summarize(ttft_values),
        "batch_itl_ms": summarize(batch_itl_values),
        "batch_itl_values_ms": batch_itl_values,
        "decode_ms": summarize(decode_values),
        "total_generation_latency_ms": summarize(total_values),
        "total_generation_ms": summarize(total_values),
        "input_tokens_total": sum(
            int(request["input_tokens"])
            for request in requests
            if request.get("error") is None and request.get("input_tokens") is not None
        ),
        "output_tokens": output_tokens,
        "output_tokens_total": total_output_tokens,
        "peak_gpu_memory_mb": float(peak_memory_mb),
        "batch_size_distribution": dict(Counter(batch_sizes)),
        "timing_boundary": STREAMING_TIMING_BOUNDARY,
        "token_timing_boundary": STREAMING_TIMING_BOUNDARY,
        "batches": batch_records,
        "requests": requests,
    }


def run_streaming(
    tokenizer,
    model,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
) -> dict:
    """Run fixed-length manual token timing over sequential local batches."""

    started = time.perf_counter()
    batch_results: list[dict] = []
    request_records: list[dict] = []
    batch_sizes: list[int] = []
    batch_records: list[dict] = []
    peak_memory_mb = 0.0
    errors = 0

    for batch_id, offset in enumerate(range(0, len(prompts), batch_size)):
        prompt_batch = prompts[offset : offset + batch_size]
        try:
            result = generate_streaming_batch(
                tokenizer,
                model,
                prompt_batch,
                max_new_tokens,
            )
            batch_results.append(result)
            batch_sizes.append(len(prompt_batch))
            peak_memory_mb = max(peak_memory_mb, float(result["peak_memory_mb"]))
            batch_records.append(
                {
                    "batch_id": batch_id,
                    "batch_size": len(prompt_batch),
                    "prefill_ms": float(result["prefill_ms"]),
                    "ttft_ms": float(result["ttft_ms"]),
                    "batch_itl_values_ms": list(result["batch_itl_values_ms"]),
                    "batch_itl_ms": dict(result["batch_itl_ms"]),
                    "decode_ms": float(result["decode_ms"]),
                    "total_generation_latency_ms": float(
                        result["total_generation_latency_ms"]
                    ),
                    "total_generation_ms": float(result["total_generation_ms"]),
                    "output_tokens": [int(value) for value in result["output_tokens"]],
                    "peak_gpu_memory_mb": float(result["peak_memory_mb"]),
                    "timing_boundary": STREAMING_TIMING_BOUNDARY,
                    "token_timing_boundary": STREAMING_TIMING_BOUNDARY,
                    "error": None,
                }
            )
            for request_offset, (input_tokens, output_tokens) in enumerate(
                zip(result["input_tokens"], result["output_tokens"])
            ):
                request_records.append(
                    {
                        "request_id": offset + request_offset,
                        "batch_id": batch_id,
                        "input_tokens": int(input_tokens),
                        "output_tokens": int(output_tokens),
                        "prefill_ms": float(result["prefill_ms"]),
                        "ttft_ms": float(result["ttft_ms"]),
                        "batch_itl_values_ms": list(result["batch_itl_values_ms"]),
                        "batch_itl_ms": dict(result["batch_itl_ms"]),
                        "decode_ms": float(result["decode_ms"]),
                        "total_generation_latency_ms": float(
                            result["total_generation_latency_ms"]
                        ),
                        "total_generation_ms": float(result["total_generation_ms"]),
                        "peak_gpu_memory_mb": float(result["peak_memory_mb"]),
                        "timing_boundary": STREAMING_TIMING_BOUNDARY,
                        "token_timing_boundary": STREAMING_TIMING_BOUNDARY,
                        "error": None,
                    }
                )
        except Exception as exc:
            errors += len(prompt_batch)
            message = f"{type(exc).__name__}: {exc}"
            for request_id in range(offset, offset + len(prompt_batch)):
                request_records.append(
                    {
                        "request_id": request_id,
                        "batch_id": batch_id,
                        "input_tokens": None,
                        "output_tokens": None,
                        "prefill_ms": None,
                        "ttft_ms": None,
                        "batch_itl_values_ms": [],
                        "batch_itl_ms": summarize([]),
                        "decode_ms": None,
                        "total_generation_latency_ms": None,
                        "total_generation_ms": None,
                        "peak_gpu_memory_mb": 0.0,
                        "timing_boundary": STREAMING_TIMING_BOUNDARY,
                        "token_timing_boundary": STREAMING_TIMING_BOUNDARY,
                        "error": message,
                    }
                )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return summarize_streaming_result(
        requested=len(prompts),
        completed=len(request_records) - errors,
        errors=errors,
        elapsed_ms=elapsed_ms,
        batch_results=batch_results,
        requests=request_records,
        batch_sizes=batch_sizes,
        peak_memory_mb=peak_memory_mb,
        batch_records=batch_records,
    )


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
    parser.add_argument(
        "--mode",
        choices=["direct", "batched", "arrival", "streaming", "both"],
        default="both",
    )
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
        if args.mode == "streaming":
            generate_streaming_batch(
                tokenizer,
                model,
                warmup_prompts,
                args.max_new_tokens,
            )
        else:
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
    if args.mode == "streaming":
        results.append(
            run_streaming(
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
