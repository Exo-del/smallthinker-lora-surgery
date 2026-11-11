# SmallThinker LoRA — format surgery (run #1 of the tag fix)

Intermediate adapter, **not** a finished model. It exists because it proves the
fix works and it is the starting point for the opener run.

## What it is

Continuation of `Exo-del/smallthinker-lora-v1` for **120 optimizer steps**
(1,920 examples, 0.66 h, single T4) with two changes:

| | |
|---|---|
| loss | `--tag-weight 30` on `<think>` 151667, `</think>` 151668, `<tool_call>` 151657, `</tool_call>` 151658, `<tool_response>` 151665/151666, `<|im_end|>` 151645 |
| LoRA targets | `q_proj,k_proj,v_proj,o_proj` **+ `lm_head`** (r=8, alpha=16) |
| init | `--init-adapter` from v1 (matching keys only; the new `lm_head` LoRA starts at zero) |
| optimiser | lr 1e-4 cosine over 120 steps, warmup 10, wd 0, grad-accum 8, seq 3072 |
| memory | peak 12.17 GiB, `skipped_oom 0` |

## Result: the tags went from rank ~10,000 to rank 1–2

`format_probe.py`, greedy, 140 new tokens, same prompt as the v1 evaluation:

```
v1    : vigilante <tool_call>{...}            p(<think>) = 0        rank 10,454
run #1: </think> The user wants me to read ... p(<think>) = 0.0159  rank 2
```

Per prompt: `p(</think>) = 0.0163` (rank 1) vs `p(<think>) = 0.0159` (rank 2),
`p(<tool_call>) = 0.0025` (rank 3) — the model now knows a think tag belongs at
the start of the turn, but it picks the **closing** tag. Everything after that
first token is right: coherent reasoning, then the correct `<tool_call>` block,
then `<|im_end|>`. Corpus control prompts behave identically, so this is not
prompt-dependent.

That is exactly what the next run attacks: `<think>` only ever occurs as the
first supervised token of a turn, so `--first-target-k 1 --first-target-weight`
trains precisely that decision.

## Files

- `adapter_model.safetensors` — LoRA weights only. PEFT also wrote
  `lm_head.base_layer.weight` (466 MB, the frozen tied matrix) into the
  notebook output; it is stripped here because the base model already provides
  it. Load with `PeftModel.from_pretrained(base, this_dir)` — `target_modules`
  includes `lm_head`.
- `adapter_config.json` — `base_model_name_or_path` rewritten to the HF id.
- `run_summary.json` — the trainer's exit summary.
