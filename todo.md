# TODO — Multi-Agent Medical Decision-Making with Indian LLMs

Rule: one change per phase. Don't start a phase until the one before it has a saved, reproducible result.
Each experiment has to state: baseline → modification → hypothesis → dataset → metrics → expected analysis (see `PHASE0.md` §9).

Legend: `[ ]` todo · `[x]` done · `[~]` in progress · `[?]` needs your decision

---

## Phase 0 — Understand & design (no code)
- [x] Read proposal + MDAgents / Catfish / TeamMedAgents papers
- [x] Inspect MDAgents repo (`main.py`, `utils.py`), map paper → code
- [x] Verify MedMCQA facts from official repo
- [x] Verify current Indian LLM / API availability (web)
- [x] Literature novelty check (web)
- [x] Write `PHASE0.md`
- [x] Decisions answered (`PHASE0.md` §11)
- [ ] **You:** ask your guide or department for GPU access or API funds. Take the pilot's cost-per-question number (Phase 1) with you
- [?] **You confirm** → Phase 1 starts

## Phase 1 — Code (no key needed) ✅ built 2026-09-25 · then Exp 1 when access arrives
Code lives in `MDAgents/`. How to run: `MDAgents/RUN.md`. Upstream diff: `cd MDAgents && git diff`.
- [x] Clone MDAgents (commit `3adbd760`, 2024-11-10); venv `.venv` (Python 3.14), deps pinned in `requirements.txt`
- [x] **LLM adapter** `llm.py` + `models.json`: any OpenAI-compatible endpoint (Sarvam v2, OpenAI, OpenRouter, local vLLM/llama.cpp) + offline `mock` model. All hard-coded models removed (`utils.py` Agent, complexity checker, both recruiters, MDT members); upstream's gpt-3.5 roles → `--aux_model`
- [x] Call log per LLM call (qid, model, role, tokens in/out, latency) → `output/calls/`
- [x] Per-question seed (same option order in every run/shard); results written per question; resume after crash; `--shard i/N` parallel workers; per-question errors recorded, not fatal
- [x] `prepare_data.py`: MedQA US test 5-opt (300), MedMCQA validation (500, stratified by subject), MedMCQA-Indic Hindi (same 500 ids); frozen ids in `data/subsets/`. Split sizes confirmed: train 182,822 / validation 4,183 / test 6,150
- [x] `baselines.py`: zero-shot, CoT, CoT-SC(5) — MDAgents `--difficulty basic` is the few-shot single agent
- [x] `evaluate.py`: accuracy, macro-F1, by complexity, unparsed %, errors, silent-agreement rate, finals-reached-moderator %, calls/tokens/sec/cost per question; `compare` = McNemar + bootstrap CI
- [x] `smoke_test.sh` passes offline (mock model; all paths, Hindi, baselines, shards, resume); real client path tested against a fake local server
- [x] Upstream bugs **logged, not fixed** (baseline stays upstream): `finals_collected` in trace (`utils.py:376-413`); complexity fallback flagged when checker names no level (upstream silently reused the previous answer)
  - [ ] still to decide as a flagged ablation: `utils.py:525` advanced decision ignores other MDTs; 3-shot (paper) vs 5-shot (code)
- [ ] **Waiting on access** — any ONE unlocks the pilot: Sarvam key (+ V2 whitelisting), OpenAI key, OpenRouter key, or a GPU (RTX 6000)
- [ ] Pilot: 20 MedMCQA questions → real calls/tokens/₹ per question (`RUN.md` §3)
- [ ] **Exp 1** (needs OpenAI key): gpt-4o-mini `--aux_model gpt-3.5-turbo`, MedQA 300, adaptive + forced levels; compare to paper Table 5 (83.6%). No OpenAI key ever → report as a limitation

## Phase 2 — Exp 2: Indian LLM + MedQA
- [ ] Indian model = whichever access arrives: `sarvam-105b` (API) or `sarvam-30b-local` (GPU); twin control `nemotron-nano-*`
- [ ] Single-agent baselines on the same questions: zero-shot, few-shot, CoT, CoT-SC (self-consistency)
- [ ] MDAgents adaptive + each fixed complexity level
- [ ] Record the complexity-routing distribution and whether any output fails to parse (format robustness)
- [ ] Paired test (McNemar) + bootstrap CI vs single-agent

## Phase 3 — Exp 3: Indian LLM + MedMCQA
- [ ] Stratified subset of MedMCQA dev by `subject_name` (size set in PHASE0 §11), fixed seed, IDs saved to file
- [ ] Same conditions as Phase 2
- [ ] Per-subject accuracy; accuracy by routed complexity; MedQA-vs-MedMCQA gap
- [ ] Also run the Western backbone (gpt-4o-mini) on MedMCQA so the comparison is backbone × dataset

## Phase 4 — Exp 4: + Catfish Agent (one change)
- [ ] Silent-agreement rate = share of intermediate cases where no agent chose to speak (`num_yes==0` in round 1). Measure it on the Phase 3 logs **first**; if it's already low, Catfish has nothing to fix
- [ ] Add Catfish: complexity-aware trigger + tone-calibrated dissent prompt, placed in team / moderator / both
- [ ] Metrics: accuracy, silent-agreement rate, answer-revision rate (right→wrong, wrong→right), extra calls/tokens

## Phase 5 — Exp 5: Hindi input via IndicTrans2 (translate → reason)
- [ ] Hindi subset, same IDs as the English one. Source per PHASE0 §11.5: `ekacare/MedMCQA-Indic` `hi` (check the licence first) or our own IndicTrans2 En→Hi. Save it; never regenerate silently
- [ ] Check the MedMCQA-Indic licence (empty on HF). If it's unclear, contact Eka Care
- [ ] Translation quality check: back-translate Hi→En with IndicTrans2, chrF/BLEU against the original English, plus a ~50-item check by a Hindi reader (not medical: fluency and meaning; medical-term errors are flagged by comparing against the English)
- [ ] Pipeline: Hindi → IndicTrans2 Hi→En → MDAgents (English)
- [ ] Metrics: accuracy drop vs English; correlation between translation quality and correctness

## Phase 6 — Exp 6: native Hindi reasoning
- [ ] Same Hindi subset, fed directly to the agents (Hindi prompts + Hindi input)
- [ ] Compare against Phase 5 and English. Report parse-failure rate separately

## Phase 7 — Ablations
- [ ] Complexity: adaptive vs forced basic / intermediate / advanced
- [ ] Number of agents / rounds (cost vs accuracy)
- [ ] Original vs bug-fixed advanced path (`utils.py:525`)
- [ ] Catfish placement (team / moderator / both)
- [?] One TeamMedAgents mechanism (their ablation found Shared Mental Model best on MedMCQA). Only if Phase 4 shows collaboration is still the bottleneck

## Phase 8 — Analysis
- [ ] Error analysis: sample wrong answers per condition, tag the type (knowledge / reasoning / translation / format / conformity)
- [ ] Cost table: calls, tokens, latency, ₹/$ per question
- [ ] One results table per RQ

## Phase 9 — Paper + diagram
- [ ] LaTeX paper (Intro → … → Conclusion). Numbers come only from `results/`, citations only from verified sources
- [ ] Architecture diagram as draw.io XML
- [ ] Reproducibility appendix: commit hash, model IDs + dates, seeds, subset ID files, prompts
