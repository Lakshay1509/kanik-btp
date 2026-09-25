# Running our MDAgents fork

Upstream MDAgents commit `3adbd760` (2024-11-10). Run `git diff` to see every change made to upstream files.

## 1. One-time setup (no key needed)

```bash
cd kanik-btp && source .venv/bin/activate && cd MDAgents
python prepare_data.py      # downloads MedQA, MedMCQA, MedMCQA-Indic (Hindi) and freezes the subsets
./smoke_test.sh             # full offline pipeline test with the fake "mock" model; ends with SMOKE TEST PASSED
```

## 2. Pick a model (whichever access you get first)

Models are listed in `models.json`. Adding one takes one line.

| You have | Do this | `--model` |
|---|---|---|
| Sarvam API key | `export SARVAM_API_KEY=...` and ask Sarvam to enable **V2 (OpenAI-compatible) access** for the key | `sarvam-105b` |
| OpenAI key | `export OPENAI_API_KEY=...` | `gpt-4o-mini` (+ `--aux_model gpt-3.5-turbo` to match upstream exactly) |
| OpenRouter key | `export OPENROUTER_API_KEY=...` | `nemotron-nano-openrouter` |
| A GPU (e.g. RTX 6000) | start a local server (below) | `sarvam-30b-local`, `nemotron-nano-local` |

Local server: pick by GPU memory. Check each model card for exact flags.

```bash
# 24 GB: 4-bit GGUF via llama.cpp
llama-server -hf sarvamai/sarvam-30b-gguf:Q4_K_M --port 8000 -ngl 99 -c 65536 -np 4 --jinja
# 48 GB+: FP8 via vLLM   (96 GB: drop -fp8 for full precision)
vllm serve sarvamai/sarvam-30b-fp8 --port 8000 --served-model-name sarvam-30b --trust-remote-code --max-model-len 32768
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 --port 8001 --served-model-name nemotron-3-nano --trust-remote-code
```

## 3. Run

Always start with a 20-question pilot. Every run resumes where it stopped if you re-run the same command.

```bash
M=sarvam-105b                       # or any name from models.json
python main.py --model $M --dataset medmcqa --num_samples 20 --tag _pilot
python evaluate.py output/${M}_medmcqa_adaptive_pilot.jsonl     # real calls, tokens and cost per question
```

Experiments (Phases 1–3 in `../todo.md`). Datasets: `medqa` (300), `medmcqa` (500), `medmcqa_hi` (500, Hindi).

```bash
python main.py      --model $M --dataset medmcqa --difficulty adaptive --num_samples 500   # MDAgents
python main.py      --model $M --dataset medmcqa --difficulty intermediate --num_samples 200   # forced level (also basic/advanced)
python baselines.py --model $M --dataset medmcqa --method cotsc --num_samples 500          # also zeroshot / cot
```

Parallel workers (local GPU, or API rate limit allowing): add `--shard i/N` to N copies of the command.

```bash
for i in 0 1 2 3; do python main.py --model $M --dataset medmcqa --num_samples 500 --shard $i/4 & done; wait
```

## 4. Score

```bash
python evaluate.py output/*.jsonl                                   # accuracy, macro-F1, by complexity, silent rate, calls, tokens, cost
python evaluate.py compare 'output/A*.jsonl' 'output/B*.jsonl'      # paired McNemar test + 95% CI
```

Outputs: `output/<model>_<dataset>_<level>.jsonl` holds one line per question. `output/calls/` holds one line per LLM call.
