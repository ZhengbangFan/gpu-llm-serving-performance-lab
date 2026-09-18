# Completion status

Completed locally; no commit or push was performed. Repository: `D:/GPU LLM Serving Performance Lab`.
Base HEAD: `44fff0f1d8c5596cd3e2c8277ab1dfa674ae461e`.

## Implemented

- `benchmark.py`: keyword-only `capture_token_ids=False`; captures selected tensors only when requested and converts to IDs after timing. Default retains no generated-token tensors and makes no additional device-to-host token copies; only a conditional check is added at each selection. Capture runs are excluded from performance results.
- `test_decode_correctness.py`: download-free CPU Qwen2 model tests for exact greedy IDs, mixed left padding, one/multiple output tokens, unchanged default schema.
- `run_experiments.py`: offline model loaded once, 12 correctness cases; shuffled 12-shape sweep, 3 repeats and 8 requests per repeat; full per-shape warmup; incremental JSON; CSV and matplotlib charts/report. Refuses an existing output directory.
- `test_experiments.py`: repeat median/min/max, weighted padding, duplicate rejection.
- `README.md`: links to measured report and this status.

## Commands actually executed (Bash, repository root)

| Command | Exit | Outcome |
|---|---:|---|
| `git status --short && git log -1 --oneline` | 0 | Initially clean, base 44fff0f |
| `HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m pytest test_decode_correctness.py -q` | 1 | Expected RED: 2 failures, capture keyword not implemented |
| `HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m pytest -q` | 0 | Initial GREEN: 23 passed |
| `uv pip install --python .venv/Scripts/python.exe matplotlib` | 0 | Installed matplotlib 3.11.2 and dependencies only in project venv |
| `HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m pytest test_experiments.py -q` | 1 | Expected RED: runner module not implemented |
| `HF_HUB_OFFLINE=1 .venv/Scripts/python.exe -m pytest -q && HF_HUB_OFFLINE=1 .venv/Scripts/python.exe run_experiments.py --output experiments/repeated_gpu_run` | 0 | 24 passed; 12 exact-token cases passed; all 36 runs completed, 288 requests |
| `git status --short && git diff --check && .venv/Scripts/python.exe -m py_compile benchmark.py run_experiments.py test_decode_correctness.py test_experiments.py` | 0 | Syntax and whitespace checks passed |
| `git diff --exit-code -- results validation && git diff --check` | 0 | Previous tracked results/validation unchanged; whitespace passed |

Additional `.venv/Scripts/python.exe -c` artifact validation exited 0: parsed and asserted 12 exact matching correctness cases, 36 unique shape/repeat records, 288 successful requests, output counts, 12 warmup files, 36 individual measured files, 12 CSV rows; Pillow verified all 3 PNGs. Parsed generation configuration confirms `(eos_token_id, min_new_tokens, repetition_penalty) = (None, None, 1.0)`.

## Real GPU outcomes

RTX 4080, FP16 Qwen/Qwen2.5-1.5B-Instruct, SDPA, PyTorch 2.14.0+cu126, Transformers 5.17.0.
Short input length: 39 tokens. Mixed input lengths: `[39, 78, 182, 338, 39, 78, 182, 338]`.
Correctness checks use batch sizes 1/2/4, output tokens 1/16, short/mixed prompts, fresh matching greedy GenerationConfig. All exact comparisons passed; no mismatch assertion was relaxed.

Batch-4 median generated-token throughput: short 16/64 output tokens = 93.54 / 104.21 tok/s; mixed 16/64 = 138.53 / 104.37 tok/s. Corresponding batch-4 versus batch-1 ratios: 3.03x / 2.42x / 5.48x / 2.41x. Mixed padding: batch 2 = 23.4%, batch 4 = 52.9%. Maximum throughput sample CV across shapes = 26.54%; this substantial variability precludes stable capacity or causal padding claims. No samples were removed.

## Artifacts

All under `experiments/repeated_gpu_run/`:
- `report.md`: English methods, results table, variability, scope and reproduction.
- `metadata.json`: hardware/software/model revision and settings.
- `correctness.json`: full exact token pairs, prompts, masks, generation settings.
- `raw_index.json`: all 36 measured records (incrementally replaced after each run).
- `raw_{short|mixed}_b{1|2|4}_n{16|64}_r{0|1|2}.json`: individual measurements, per-batch/request timings and errors.
- `warmup_{short|mixed}_b{1|2|4}_n{16|64}.json`: all 12 warmups.
- `aggregate.csv`: 12 rows with repeat median/min/max/sample standard deviation.
- `throughput.png`, `ttft.png`, `itl.png`: matplotlib plots, median with observed min/max whiskers.

## Issues and remaining scope

No execution blocker remains. One oversized tool write timed out before execution; code was subsequently written in smaller pieces. A patch initially failed ambiguity validation without modifying any file, then succeeded with specific context.
Existing pynvml deprecation warning remains. Model load emitted a Transformers warning about unused sampling defaults; correctness explicitly uses its own fresh greedy config. uv fell back from hardlinks to copying across drives. Git reports normal LF-to-CRLF notices. These did not prevent completion.

Performance includes CUDA synchronization per token and local Python overhead; it is not a network or continuous-batching benchmark. Three repeats and uncontrolled desktop GPU conditions limit interpretation. Correctness is bounded to this model and tested prompts; output length 64 was measured but not separately compared against `generate`. Publishing and independent review belong to the parent agent; nothing was committed or pushed.
