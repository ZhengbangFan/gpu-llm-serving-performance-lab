# GPU LLM Serving Performance Lab

A reproducible GPU inference benchmark for a real instruction-tuned language model. The current baseline uses Hugging Face Transformers directly so the workload, tensor batching, and measurements remain easy to inspect on Windows.

## What This Measures

- CUDA inference on an NVIDIA RTX 4080
- Direct generation versus padded tensor batching
- End-to-end request latency and latency percentiles
- Generated-token throughput and request throughput
- Peak allocated GPU memory and error rate
- Hardware, software, model, and workload provenance in JSON

This is an experimental serving/performance lab, not a claim of production-scale LLM serving. Phase 1 adds a deterministic controlled-arrival scheduler around the existing sequential Transformers call. The `streaming` mode adds local token-by-token timing for the same Transformers model; it is a manual decode benchmark, not a network streaming endpoint or a continuous-batching scheduler.

## Reproduce

```text
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python.exe run_generation.py --model Qwen/Qwen2.5-1.5B-Instruct --max-new-tokens 32
.venv/Scripts/python.exe benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct --mode both --requests 8 --batch-size 4 --max-new-tokens 32 --warmup 1 --output results/qwen25_1.5b_baseline.json
.venv/Scripts/python.exe benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct --mode arrival --requests 16 --batch-size 4 --arrival-interval-ms 10 --batch-wait-timeout-ms 25 --max-new-tokens 32 --output results/arrival.json
.venv/Scripts/python.exe benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct --mode streaming --requests 4 --batch-size 4 --max-new-tokens 32 --warmup 1 --output results/streaming.json
.venv/Scripts/python.exe -c "from arrival_scheduler import simulate_arrival_batches; print(simulate_arrival_batches([0, 5, 10], 2, 8, execution_time_ms=3))"
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m py_compile benchmark.py arrival_scheduler.py metrics.py run_generation.py
```

The first run downloads the model from Hugging Face. CUDA is required for `direct`, `batched`, `arrival`, and `streaming`; the benchmark exits if no CUDA device is available.
The scheduler and metric helpers are pure Python, so the `-c` simulation, tests, and `py_compile` command do not require CUDA or model downloads. `arrival` mode still loads the configured model and executes one batch at a time.

## Arrival Mode And Metrics

Arrival mode assigns request `i` a virtual arrival time of `i * --arrival-interval-ms`. The scheduler dispatches the oldest queued requests when the queue reaches `--batch-size` or when the oldest request reaches `--batch-wait-timeout-ms`. It records one JSON object per request with arrival time, queue wait, batch execution time, end-to-end latency, batch size, and an error/error message field.

Metric definitions:

- **Padding ratio:** `1 - sum(unpadded input lengths) / (request count * padded width)`. The padded width defaults to the longest sequence in the batch; an explicit width can be supplied to the helper.
- **Percentiles:** p50, p95, and p99 use linear interpolation over sorted values. Empty distributions report `0.0`.
- **Queue wait:** batch dispatch time minus request arrival time.
- **Batch execution:** batch completion time minus dispatch time; this is the measured `model.generate` call duration.
- **End-to-end latency:** batch completion time minus request arrival time (`queue wait + batch execution`).
- **Errors:** requests whose batch failed (or were marked with an error), counted in `errors` and `error_rate`.

### Token Timing And Boundaries

`direct`, `batched`, and `arrival` continue to use `model.generate` as a completion API. They do not expose token timestamps: arrival records retain `ttft_ms`/`itl_ms` as `null` and identify their `token_timing_boundary` as `batch_generate_elapsed_ms_non_streaming`. Direct and batched summaries omit token timing and timing-boundary fields.

`streaming` performs a fixed-length greedy decode with `model.forward(..., past_key_values=..., use_cache=True)` instead of calling `model.generate`. It uses the existing chat template and left-padded inputs. CUDA is synchronized before timing, after prefill/first-token selection, and after every decode-step token selection so each timestamp represents completed GPU work. Generation always selects exactly `--max-new-tokens` tokens; EOS is not used to shorten the run.

Streaming result payloads expose batch-level summaries and per-request records. Each request includes a `batch_id` so it can be mapped back to the batch that produced it. The `batches` array contains one `batch_size` entry per generated batch, and `batch_size_distribution` summarizes those generated batches rather than expanding sizes per request. Failed request records retain their `batch_id` and carry empty batch ITL values. The timing fields are:

- **Prefill:** time from the synchronized prefill start until prefill forward work completes (`prefill_ms`); first-token selection is reported separately within TTFT.
- **TTFT (time to first token):** elapsed time from the same prefill start to the synchronized first-token selection (`ttft_ms`). In this manual path, TTFT includes prefill and first-token selection.
- **Batch ITL (inter-token latency):** elapsed time between consecutive synchronized token selections for the shared batch decode timeline (`batch_itl_ms`). The same batch timeline is attached to each request record under `batch_itl_values_ms`/`batch_itl_ms`; it is not an independent per-request GPU timeline. A one-token output has an empty ITL distribution.
- **Decode:** elapsed time after first-token selection through the final synchronized decode-step selection (`decode_ms`).
- **Total generation latency:** elapsed time from prefill start through the final synchronized token selection (`total_generation_latency_ms`).
- **Output token count:** exactly `--max-new-tokens` for each successful request (`output_tokens`).
- **Peak memory:** peak allocated CUDA memory observed for the measured batch (`peak_gpu_memory_mb`).

The explicit `timing_boundary` value identifies this as local, CUDA-synchronized manual decoding. It must not be interpreted as HTTP/network streaming latency, socket flush latency, or continuous-batching service behavior.

## Baseline

Hardware and software:

- GPU: NVIDIA GeForce RTX 4080, 16,375.5 MiB
- PyTorch: 2.14.0+cu126
- Model: Qwen/Qwen2.5-1.5B-Instruct
- Workload: 8 requests, 32 generated tokens per request, one warmup request
- Batch configuration: direct requests versus padded tensor batches of 4

| Mode | Request throughput | Generated-token throughput | Request P50 | Request P95 | Peak allocated GPU memory | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct | 1.63 req/s | 52.08 tok/s | 613.86 ms | 631.28 ms | 2,956.6 MiB | 0/8 |
| Tensor batch 4 | 6.42 req/s | 205.30 tok/s | 622.33 ms | 623.16 ms | 2,968.9 MiB | 0/8 |

In this controlled baseline, tensor batching improved aggregate request throughput by 3.94x and generated-token throughput by 3.94x. The result is a small, reproducible baseline rather than a general production capacity claim; larger workload sweeps are needed before drawing broader conclusions.

Raw result: `results/qwen25_1.5b_baseline.json`

## Token Timing Validation

See [validation report](validation/README.md) for raw GPU measurements and reproduction commands. The warmed validation uses 8 requests, batch size 4, 32 tokens per request, and one four-request warmup batch on Qwen2.5-1.5B-Instruct / RTX 4080. It completed with zero errors. These single-run checks establish working timing instrumentation. The bounded repeated GPU milestone is now recorded in [the experiment report](experiments/repeated_gpu_run/report.md): 12 exact-token correctness cases and 36 measured runs (288 requests), with raw data, variability, padding ratios, and plots. See [completion status](experiments/completion_status.md) for executed commands and outcomes. The broader arrival sweep below remains future work.

## Project Structure

- `benchmark.py`: direct, tensor-batched, arrival, and local token-timing generation benchmarks
- `metrics.py`: pure-Python padding, percentile, arrival-accounting, and token-timestamp helpers
- `arrival_scheduler.py`: deterministic virtual-time batching scheduler used by arrival mode and CPU tests
- `run_generation.py`: one-request CUDA smoke test
- `test_benchmark.py`, `test_metrics_scheduler.py`, `test_streaming_schema.py`: baseline and CPU-only metric/scheduler/streaming-schema tests
- `results/`: locally generated benchmark output; ignored by Git by default

## Experiment Plan

Run one controlled sweep while keeping the model, prompt template, output length, and warmup fixed: use arrival intervals of 0, 10, 25, and 100 ms; compare max batch sizes 1, 2, 4, and 8; and hold the wait timeout at 25 ms. For each run, save the JSON output and compare request throughput, queue-wait p50/p95, end-to-end latency p50/p95, batch-size distribution, padding ratio (when input lengths are exported), and error rate. For streaming runs, also compare prefill, TTFT, batch ITL, decode, and total-generation summaries at the same fixed output length. Repeat each point three times and report the median; do not overwrite `results/qwen25_1.5b_baseline.json`.
