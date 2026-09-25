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
- [?] **You confirm** Phase 0 + answer the open decisions in `PHASE0.md` §11

## Phase 1 — Setup + minimal fixes + Exp 1 (original MDAgents reproduction)
- [ ] Clone MDAgents into `MDAgents/`, record the commit hash
- [ ] Pin the env (Python ≥3.12, needed because `utils.py:454` uses a backslash inside an f-string expression)
- [ ] **LLM adapter**: one `chat(model, messages, temperature)` function over the OpenAI-compatible API (`base_url` + key + model id from a small `models.yaml`). Replace every hard-coded model:
  - [ ] `Agent.__init__/chat/temp_responses` (`utils.py:22,48-51,69-72`): `--model gpt-4` silently calls `gpt-4o-mini`
  - [ ] complexity checker hard-coded `gpt-3.5` (`utils.py:246`)
  - [ ] intermediate recruiter hard-coded `gpt-3.5` (`utils.py:286`)
  - [ ] advanced recruiter + all MDT members hard-coded `gpt-4o-mini` (`utils.py:95,465`)
- [ ] Log every call (model, prompt tokens, completion tokens, latency, role, stage) to a JSONL file. `total_api_calls` in `main.py:34` is never incremented
- [ ] Global `--seed` (option shuffle `utils.py:235,266,335` is unseeded)
- [ ] Data loaders → one common schema `{id, question, options{A..}, answer_idx, meta}`:
  - [ ] MedQA US test (5-option = MDAgents paper setting; 4-option as a secondary setting)
  - [ ] MedMCQA dev (labels public). Confirm split sizes on download
- [ ] Fix `main.py:48`: results are only saved when dataset == medqa
- [ ] Answer extractor (regex, then one LLM fallback) + scorer script: accuracy, macro-F1, per-complexity accuracy, calls, tokens, latency. The repo has **no** evaluation code
- [ ] Record the known bugs **without fixing them yet** (fixes go in as a separate, flagged ablation so the baseline stays the original):
  - [ ] `utils.py:525` advanced final decision reads only the initial-assessment report and ignores the other MDTs and FRDT
  - [ ] `utils.py:401-413` if no agent speaks in round 1, `final_answer=None` goes to the moderator
  - [ ] Paper says 3-shot for low complexity; code uses 5 (`utils.py:262`)
- [ ] **Exp 1**: gpt-4o-mini, MedQA, adaptive + solo + fixed-intermediate + fixed-advanced. Compare against the paper's Table 5 (full MedQA with GPT-4o mini)
- [ ] Sanity run: 10 questions end to end, then the full subset

## Phase 2 — Exp 2: Indian LLM + MedQA
- [ ] Add the Indian model(s) chosen in PHASE0 §11 to `models.yaml`
- [ ] Single-agent baselines on the same questions: zero-shot, few-shot, CoT, CoT-SC (self-consistency)
- [ ] MDAgents adaptive + each fixed complexity level
- [ ] Record the complexity-routing distribution and whether any output fails to parse (format robustness)
- [ ] Paired test (McNemar) + bootstrap CI vs single-agent

## Phase 3 — Exp 3: Indian LLM + MedMCQA
- [ ] Stratified subset of MedMCQA dev by `subject_name` (size set in PHASE0 §11), fixed seed, IDs saved to file
- [ ] Same conditions as Phase 2
- [ ] Per-subject accuracy; accuracy by routed complexity; MedQA-vs-MedMCQA gap
- [ ] Also run the Western backbone (gpt-4o-mini) on MedMCQA so the comparison is backbone × dataset
- [ ] (stretch, only if RQ3 needs it) PubMedQA / DDXPlus

## Phase 4 — Exp 4: + Catfish Agent (one change)
- [ ] Silent-agreement rate = share of intermediate cases where no agent chose to speak (`num_yes==0` in round 1). Measure it on the Phase 3 logs **first**; if it's already low, Catfish has nothing to fix
- [ ] Add Catfish: complexity-aware trigger + tone-calibrated dissent prompt, placed in team / moderator / both
- [ ] Metrics: accuracy, silent-agreement rate, answer-revision rate (right→wrong, wrong→right), extra calls/tokens

## Phase 5 — Exp 5: Hindi input via IndicTrans2 (translate → reason)
- [ ] Hindi subset, same IDs as the English one. Source per PHASE0 §11.5: `ekacare/MedMCQA-Indic` `hi` (check the licence first) or our own IndicTrans2 En→Hi. Save it; never regenerate silently
- [ ] Translation quality check: back-translate Hi→En, chrF/BLEU against the original English, plus a manual check of ~50 items (medical terms, negations, numbers)
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
