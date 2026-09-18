"""Small random CPU model: no downloads or GPU required."""
import pytest
import torch
from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM
import benchmark


class LeftPaddingTokenizer:
    pad_token_id = 0
    padding_side = "left"

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def __call__(self, prompts, **kwargs):
        rows = [[int(t) for t in p.split()] for p in prompts]
        width = max(map(len, rows))
        ids = torch.tensor([[0] * (width - len(r)) + r for r in rows])
        return {"input_ids": ids, "attention_mask": (ids != 0).long()}


@pytest.mark.parametrize("count", [1, 4])
def test_optional_capture_matches_cpu_greedy_and_default_schema(monkeypatch, count):
    monkeypatch.setattr(benchmark, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(7)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, pad_token_id=0)).eval()
    tokenizer = LeftPaddingTokenizer()
    prompts = ["3 4", "5 6 7 8"]
    captured = benchmark.generate_streaming_batch(tokenizer, model, prompts, count,
                                                  capture_token_ids=True)
    encoded = tokenizer(prompts)
    with torch.inference_mode():
        expected = model.generate(**encoded, generation_config=GenerationConfig(
            max_new_tokens=count, do_sample=False, eos_token_id=None,
            pad_token_id=0, use_cache=True))[:, encoded["input_ids"].shape[1]:].tolist()
    assert captured["generated_token_ids"] == expected
    assert all(len(row) == count for row in captured["generated_token_ids"])
    plain = benchmark.generate_streaming_batch(tokenizer, model, prompts, count)
    assert "generated_token_ids" not in plain
    assert plain.keys() == captured.keys() - {"generated_token_ids"}
    assert plain["output_tokens"] == captured["output_tokens"] == [count, count]
