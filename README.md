# GPU LLM Serving Performance Lab

A reproducible GPU inference benchmark for a real instruction-tuned language model. The current baseline uses Hugging Face Transformers directly so the workload, tensor batching, and measurements remain easy to inspect on Windows.

## What This Measures

- CUDA inference on an NVIDIA RTX 4080
- Direct generation versus padded tensor batching
- End-to-end request latency and latency percentiles
- Generated-token throughput and request throughput
- Peak allocated GPU memory and error rate
- Hardware, software, model, and workload provenance in JSON

This is an experimental serving/performance lab, not a claim of production-scale LLM serving. Streaming TTFT/ITL, concurrent arrival modeling, and vLLM/SGLang adapters are planned follow-up experiments.

## Reproduce

```text
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python.exe run_generation.py --model Qwen/Qwen2.5-1.5B-Instruct --max-new-tokens 32
.venv/Scripts/python.exe benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct --mode both --requests 8 --batch-size 4 --max-new-tokens 32 --warmup 1 --output results/qwen25_1.5b_baseline.json
.venv/Scripts/python.exe -m pytest -q
```

The first run downloads the model from Hugging Face. CUDA is required; the benchmark exits if no CUDA device is available.

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

## Project Structure

- `benchmark.py`: direct versus tensor-batched generation benchmark
- `run_generation.py`: one-request CUDA smoke test
- `test_benchmark.py`: percentile, metric, and saved-result tests
- `results/`: locally generated benchmark output; ignored by Git by default

## Next Experiments

1. Sweep batch size, prompt length, output length, and request count.
2. Add concurrent arrivals and report queueing behavior separately from model execution.
3. Add streaming generation to measure time to first token and inter-token latency.
4. Compare a mature serving engine such as vLLM or SGLang in a Linux/WSL2 environment.
5. Use a profiler to connect scheduler and memory settings to measured throughput.
