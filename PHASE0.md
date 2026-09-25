# Phase 0: Multi-Agent Medical Decision-Making with Indian LLMs

*Written 2026-09-25. Nothing below is a result; every number quoted is from a cited source. Tracking lives in `todo.md`.*

**Short version**
- We take **MDAgents** (NeurIPS 2024) as it is and swap its GPT backbone for an **Indian LLM**. The plan is **Sarvam-105B**, because Krutrim has paused model work (see §5).
- We test it on **MedMCQA** (Indian PG entrance-exam questions) and on **MedQA** (US board exam), then add **one change at a time**: the Catfish dissent agent, then Hindi input translated with IndicTrans2, then native Hindi reasoning.
- The MDAgents repo is small (2 files, about 590 lines). It has hard-coded models, no evaluation code and a few bugs (§2.3), so the first code step is an **LLM adapter** plus a scorer.
- Multi-agent work on MedMCQA **already exists**. Multi-agent debate on an **Indian LLM backbone** does not appear to (§6).

---

## 1. What we are building

A system that answers medical exam questions using a **team of AI "doctors"** instead of one AI. Every doctor in the team is an **Indian-made LLM**. We then measure, honestly:

1. Does the team beat a single Indian LLM?
2. Does the team's way of organising itself (the complexity check, then recruiting specialists) still work when the underlying model isn't GPT?
3. Does it hold up on Indian exam content, and on Hindi input?

This is a **benchmark study**. It is not a clinical tool, and high accuracy on exam MCQs does **not** mean the system is safe for patients.

## 2. What MDAgents gives us

### 2.1 The idea (from the paper)

A real hospital doesn't send every case to a full committee. MDAgents copies that idea:

```
Medical question
  → Complexity check      (an LLM labels it basic / intermediate / advanced)
  → Expert recruitment    (an LLM picks which specialists to "hire")
  → Multi-agent reasoning
       basic        → one doctor (PCP), few-shot
       intermediate → 5 specialists debate in rounds (MDT)
       advanced     → 3 teams of 3 (initial assessment → others → final review) (ICT)
  → Synthesis             (a moderator or decision-maker reads all opinions)
  → Final decision
```

Headline result: **83.6% on the full MedQA 5-option test set with GPT-4o-mini**. For comparison, the paper's single-agent CoT-SC scored 77.2% (Table 5, arXiv 2404.15155). The main tables use only **50 samples per dataset**.

### 2.2 Paper → code map (repo commit cloned 2026-09-25)

| Paper stage | Where it lives | What we change |
|---|---|---|
| Entry, loop over questions | `main.py:15-56` | add `--seed`, save results for every dataset, not just `medqa` (`main.py:48`) |
| Data loading | `utils.load_data` `:213` (reads `../data/{ds}/test.jsonl`, which doesn't match the README) | one loader per dataset, all mapping to the same schema |
| Prompt formatting | `utils.create_question` `:229` (MedQA only) | add MedMCQA (`opa..opd`, `cop`) |
| LLM calls | `utils.Agent` `:12-88` (Gemini or OpenAI only; **any gpt name except gpt-3.5 is silently sent to `gpt-4o-mini`**, `:48-51`) | route through **one adapter function** (§3) |
| Complexity check | `utils.determine_difficulty` `:240` (**hard-coded `gpt-3.5`**, `:246`) | use the experiment's backbone |
| Basic path (PCP) | `utils.process_basic_query` `:257` | no change |
| Intermediate path (MDT): recruit + debate | `utils.process_intermediate_query` `:282` (recruiter **hard-coded `gpt-3.5`**, `:286`) | use the backbone; Catfish hooks in here (Phase 4) |
| Advanced path (ICT) | `utils.process_advanced_query` `:459`, `Group` `:90` (**hard-coded `gpt-4o-mini`**, `:95,465`) | use the backbone |
| Final decision | moderator `:448-451`, decision-maker `:521-525` | no change for the baseline |
| Evaluation | **does not exist** | new `evaluate.py`: answer extraction, accuracy, F1, cost |

### 2.3 What the repo does *not* give us (found by reading it)

- **No scoring code.** Outputs are raw text such as `{0.0: "..."}`, so we need an answer extractor.
- **No cost logging.** `total_api_calls` (`main.py:34`) is never incremented.
- **No seed.** Option order is shuffled randomly (`utils.py:235,266,335`).
- **Paper ≠ code:**
  - The paper says 3-shot for basic cases; the code uses 5 (`:262`).
  - The advanced path's final decision reads **only the initial-assessment team** and ignores the others (`:525`).
  - **The debate often never reaches the moderator** (`:376-413`). Each agent's final answer is collected only if **all 5 turns** of a round had at least one speaker. If any turn is silent, the loop exits first. When that happens in round 1, the moderator receives `final_answer=None` and effectively answers alone. Catfish reports that MDAgents is "silent" in most intermediate cases, so this is probably the common path.
- **Needs Python ≥ 3.12**, because `:454` puts a backslash inside an f-string expression.

**Policy:** Exp 1 runs the code **as released** (only the model routing changes), so it's a true reproduction. Bug fixes run later as a separate, labelled ablation.

**Rough calls per question (read from the code; the pilot will measure it):**

| Path | Calls |
|---|---|
| Basic | ~7 (+2 for the complexity check) |
| Advanced | ~36 |
| Intermediate | 26 if silent, ~40 brief, ~88 if a full round completes |

Chat history also grows with every turn, so tokens grow faster than calls.

## 3. What we are changing, and why

In order, one per phase:

| # | Change | Why it is needed |
|---|---|---|
| 1 | **LLM adapter**: one `chat(model, messages, temperature)` over the OpenAI-compatible API, with models listed in a small YAML | Five places hard-code GPT models (§2.2). Without the adapter we can't swap backbones and can't honestly say "the whole team is Indian". Sarvam's `/v2/chat/completions` is OpenAI-compatible, so no new client library is needed. |
| 2 | **Evaluation + cost logging** | The repo has none, and the RQs need accuracy, F1, calls, tokens and latency. |
| 3 | **MedMCQA loader** | MDAgents never evaluated MedMCQA (its README only links it). |
| 4 | **Catfish agent** (Phase 4) | Catfish reports that MDAgents shows *silent agreement* in 68.1% of its MedQA failures (arXiv 2505.21503, Table 1). We check whether that also happens with Indian backbones **before** adding a fix. |
| 5 | **Hindi → IndicTrans2 → English** (Phase 5) | This is the realistic case of a Hindi-speaking user. The question is whether an explicit MT step helps or hurts. |
| 6 | **Native Hindi** (Phase 6) | This is the alternative to #5: the model reasons in Hindi directly. Prior work disagrees on which is better (§6). |
| (7) | *Maybe* one TeamMedAgents mechanism | Only if Phase 4 shows collaboration is still the bottleneck. Their ablation found Shared Mental Model best on MedMCQA (arXiv 2508.08115, Fig. 4), and also found no single mechanism wins everywhere. |

**Not changing:** the MDAgents prompts, the debate structure, or the number of agents and rounds (except as ablations). We don't rewrite the repo.

## 4. Why MedMCQA matters

| | **MedQA (US)** | **MedMCQA** |
|---|---|---|
| Source | USMLE-style board exam (US) | **AIIMS PG and NEET-PG** entrance exams (India) |
| Style | Long clinical vignettes ("A 47-year-old woman…") | Mostly **short** questions (average 12.77 tokens), many pure recall |
| Options | 5 (MDAgents setting); a 4-option version also exists | 4 (`opa`–`opd`), plus an expert explanation `exp` |
| Size we use | US test set, 1,273 questions | Dev/validation split, **labels public** (NEET-PG). **Test labels are hidden** |
| Licence | see the MedQA repo | MIT |
| Role here | Reproduce MDAgents and compare backbones | **Main India-grounded benchmark** |

Why it matters:
- **India-grounded content.** It reflects Indian medical syllabi and disease priorities, and 21 subjects including PSM (community medicine), Forensic Medicine and Dental.
- **It stress-tests MDAgents.** MDAgents was built for vignette-style cases, while many MedMCQA items are one-line recall. Complexity routing may send almost everything to "basic", and debating a recall fact may add cost without adding accuracy. Finding that out is itself a result.
- **A Hindi version with the same questions exists** (`ekacare/MedMCQA-Indic`, `hi` config, 4,183 items = the MedMCQA validation set). That gives a clean paired English-vs-Hindi comparison.

What MedMCQA is **not**:
- It is **not real-patient data**. It is exam MCQs.
- It is **not clinical validation**.
- Split sizes confirmed on download: train 182,822 · validation (dev, NEET-PG, labels public) 4,183 · test (AIIMS, labels hidden) 6,150.
- **Subset caveat:** proportional stratification makes **Dental 32%** (158/500) of our subset, and gold answers are skewed (**A = 35%**, D = 16%), so "always A" scores ~35%, not 25%. Report both.

## 5. Where Indian LLMs fit

They are the **reasoning backbone of every agent**: complexity checker, recruiter, specialists, moderator. They are not a separate add-on.

**Availability, checked 2026-09-25 from vendor docs and HF (not from memory):**

| Model | API today | Open weights | Notes |
|---|---|---|---|
| **Sarvam-105B** | ✅ `api.sarvam.ai`, `/v2/chat/completions` (OpenAI-compatible, beta) | ✅ `sarvamai/sarvam-105b`, Apache-2.0 | MoE, 10.3B active, 128K context. **Thinking is on by default**, and thinking tokens are billed as output. Starter plan: 40 req/min, `max_tokens` capped at 4096. ₹29.28 input / ₹73.20 output per 1M tokens; ₹100 free credit. |
| Sarvam-30B, Sarvam-M | ❌ **removed from the API** | ✅ Apache-2.0 (`sarvamai/sarvam-30b`, `sarvamai/sarvam-m`) | Self-host with vLLM or SGLang. Sarvam-M is built on Mistral-Small, so it is not "fully Indian". |
| Krutrim-2 | ❌ **not in the AI Studio catalogue**; Krutrim paused foundation-model work in late 2025 | ⚠️ `krutrim-ai-labs/Krutrim-2-instruct` (12B, Mistral-NeMo-based, custom licence, not updated since early 2025) | **Drop it as a backbone.** The proposal named Krutrim, so this is a scope change. |
| BharatGen Param2-17B | none found | ✅ **non-commercial** licence | possible later ablation only |
| Models under 5B (BharatGPT, Param-1, Indus…) | – | – | too weak; MILU found such models near random |

**Plan:**
- **Primary backbone:** Sarvam-105B through the API.
- **Western reference:** gpt-4o-mini, the same model as MDAgents' Table 5.
- **Reproducibility fallback:** self-hosted Sarvam-30B. Sarvam has already removed two models from its API, so API-only numbers may not be re-runnable later.
- For every run, log the model ID, the date and `reasoning_effort`.

**Existing results for Indian LLMs:**
- I found **no published MedMCQA or MedQA score** for any Sarvam or Krutrim model.
- The only nearby medical data point is a 95-question ophthalmology MCQ study: GPT-5 83%, Kruti 71%, Sarvam 67% (Kamdar et al., Indian J Ophthalmol 2026).

## 6. What is actually novel

**Already done. We cite these, we don't claim them:**
- Multi-agent systems evaluated on MedMCQA: MedAgents (2311.10537), Catfish (2505.21503), TeamMedAgents (2508.08115), MedAgentsBench (2503.07459), and AMR (2608.19029), which reports MDAgents on MedMCQA with GPT-4o.
- MDAgents with weaker or open backbones, often **not** beating a strong single model: MedAgentBoard (2505.12371), MedAgentsBench, TeamMedAgents.
- MDAgents on a non-English exam: Spanish EUNACOM (Altermatt et al., BMC Med Educ 2025).
- Hindi MedMCQA translations: MedMCQA-Indic (HF), HiMed (2605.24635).
- A multi-agent Indic medical system on a non-Indian backbone: ArogyaSutra (2606.13572), Qwen2.5-VL-7B, no MDAgents comparison.
- Translate-to-English vs native reasoning for Hindi medical QA, single-agent: Multi-OphthaLingua (2412.14304).
- English-pivot cost inside multi-agent pipelines: non-medical, Aya-23 (2609.15079).

**Not found in the literature as of 2026-09-25, so these are our contributions:**
1. **MDAgents-style adaptive collaboration on an Indian-developed LLM backbone.** No paper found uses Sarvam, Krutrim, Param, etc. as agents in multi-agent debate, medical or not.
2. **Does the complexity router transfer?** Nobody has checked whether MDAgents' complexity check stays meaningful with a non-GPT, Indic-focused backbone.
3. **A backbone × dataset grid**: {Indian, Western} × {MedQA, MedMCQA} under the same framework, with cost.
4. **Silent agreement measured on an Indian backbone**, before and after Catfish.
5. **IndicTrans2 translate-then-reason vs native Hindi, inside a multi-agent medical pipeline**, on paired Hindi and English MedMCQA.

**Phrasing rules for the paper:**
- Say "the original MDAgents paper did not report MedMCQA", **never** "first multi-agent MedMCQA study".
- Never say "first Hindi medical benchmark".
- Scope what "Indian LLM" means. Sarvam-M is Mistral-based. Before calling Sarvam-105B "trained from scratch", check Sarvam's technical report.
- Always report strong single-agent baselines (CoT-SC at a matched token budget). Prior work shows multi-agent gains often vanish on weaker backbones, and adversarial dissent has **hurt** accuracy in one rare-disease study (2603.06856).

## 7. Research questions

| RQ | Question | Answered by |
|---|---|---|
| RQ1 | Does MDAgents beat a **single** Indian LLM (zero-shot, few-shot, CoT, CoT-SC) on the same questions? | Exp 2, 3 |
| RQ2 | Does **adaptive complexity routing** work with Indian LLMs? Is it better than forcing basic, intermediate or advanced for everything, and is its routing sensible? | Exp 2, 3, 7 |
| RQ3 | Does performance differ between **MedMCQA** (India, short recall) and **MedQA** (US, vignettes), and is the gap a backbone effect or a dataset effect? PubMedQA and DDXPlus are optional. | Exp 3 (2×2 grid) |
| RQ4 | Does **Catfish-style dissent** reduce silent agreement, and does that improve accuracy or only cost? | Exp 4 |
| RQ5 | For Hindi input, is **IndicTrans2 → English reasoning** better or worse than **native Hindi reasoning**, and does MT quality predict errors? | Exp 5, 6 |
| RQ6 | Which modifications actually contribute? | Exp 7 (ablations) |

## 8. Complete high-level architecture

```
        ┌─────────────── INPUT (Exp 5/6 only) ───────────────┐
Hindi Q │ (A) IndicTrans2 hi→en ──► English Q                 │
        │ (B) native: Hindi Q + Hindi prompts, no MT          │
        └────────────────────────────────────────────────────┘
English Q (MedQA / MedMCQA) ──┐
                              ▼
                 ┌────────────────────────┐
                 │ 1. Complexity check     │  1 LLM call
                 └──┬─────────┬─────────┬─┘
             basic  │  interm.│         │ advanced
                    ▼         ▼         ▼
             ┌────────┐ ┌──────────┐ ┌─────────────────────┐
             │ Solo   │ │ MDT      │ │ ICT                  │
             │ (PCP)  │ │ recruit 5│ │ recruit 3 teams × 3  │
             │ few-   │ │ experts  │ │ IAT → other MDTs →   │
             │ shot   │ │ → debate │ │ FRDT                 │
             └───┬────┘ │ rounds   │ └──────────┬──────────┘
                 │      │  ▲       │            │
                 │      │  └ Catfish agent (Exp 4)
                 │      └────┬─────┘            │
                 ▼           ▼                  ▼
                 ┌────────────────────────────────┐
                 │ 2. Moderator / final decision   │
                 └───────────────┬────────────────┘
                                 ▼
               answer extractor → evaluate.py (acc, F1, per-complexity,
                                  silent-agreement, revisions, calls,
                                  tokens, latency, cost)

 every LLM call ──► LLM adapter ──► Sarvam-105B (API) | gpt-4o-mini (API)
                                    | Sarvam-30B (self-hosted, fallback)
                    └─► call log (JSONL): stage, role, tokens, latency
```

The draw.io XML version of this diagram is Phase 9.

## 9. Experiment plan

Same question IDs across every condition, fixed seed, and a pilot of 20 questions before each full run.

| Exp | Baseline → Modification | Hypothesis (not a result) | Data | Metrics | Expected analysis |
|---|---|---|---|---|---|
| 1 | Paper numbers → released code + gpt-4o-mini | We land near the paper's 83.6%. A gap is plausible because of the paper≠code issues (§2.3) and model drift since 2024. | MedQA US 5-opt | acc, per-complexity acc, calls, tokens | Size of the reproduction gap and its likely causes |
| 2 | Single Sarvam-105B → MDAgents(Sarvam) | MDAgents gains are smaller than with GPT, and may be zero against CoT-SC (in line with MedAgentBoard) | MedQA | acc, macro-F1, Δ vs single, routing distribution, parse failures, cost | Does collaboration help this backbone? Is routing sensible? |
| 3 | Exp 2 → MedMCQA | Router sends mostly "basic"; smaller multi-agent gain on short recall items | MedMCQA dev subset (stratified by subject) | as Exp 2 + per-subject acc | 2×2 backbone × dataset; where gains come from |
| 4 | Exp 3 → + Catfish | If silent agreement is high, Catfish lowers it; the accuracy effect is uncertain because dissent can also flip correct answers | MedMCQA (intermediate cases) | silent-agreement rate, revision rate (right→wrong / wrong→right), acc, extra cost | Is less agreement worth the tokens? |
| 5 | Exp 3 (English) → Hindi input + IndicTrans2 hi→en | Some drop vs English, with errors concentrated on medical terms, negations and numbers | Hindi MedMCQA subset (paired IDs) | acc, chrF/BLEU of the back-translation, 50-item manual check | Does MT quality predict wrong answers? |
| 6 | Exp 5 → native Hindi (no MT) | Direction genuinely unknown; prior work disagrees (2412.14304 vs 2609.15079) | same Hindi subset | acc, parse failures, cost | Translate-test vs native, on this backbone |
| 7 | Full system → remove or replace one piece | – | MedMCQA (+ MedQA where relevant) | acc, cost | adaptive vs fixed; agents and rounds; bug-fix; Catfish placement |

Statistics: paired McNemar tests and bootstrap 95% CIs. No claim of improvement without them.

## 10. Rules we commit to

- No number goes in the paper unless it came from `results/`. No citation goes in unless it was opened and checked.
- MedQA ≠ MedMCQA, kept distinct everywhere. MedMCQA is exam MCQs, **not real patients**.
- Benchmark accuracy ≠ clinical validity. This goes in the Limitations section.
- Log every model ID, date, seed, prompt and subset ID file so any run can be repeated.

## 11. Decisions (answered 2026-09-25)

| # | Decision | Answer | Consequence |
|---|---|---|---|
| 1 | Backbone | **Pluggable** (updated 2026-09-25: no API key or GPU confirmed yet). Sarvam-105B API *or* Sarvam-30B on a local GPU, whichever arrives first; twin control Nemotron-3-Nano-30B. Krutrim dropped | Code is model-agnostic (`MDAgents/models.json`) |
| 2 | Budget | **Free credits only (₹100)** + **ask the institute** for GPU or API funds | ⚠️ Blocks everything past the pilot. Phase 1 builds the code and runs a ~20-question pilot on free credits to measure the real cost per question, which becomes the number to take to the guide or department |
| 3 | Subset size | **500 MedMCQA** (stratified by subject) / **300 MedQA** | Same IDs in every condition |
| 4 | GPU | **Unsure** (RTX 6000 possible) | If it arrives: Sarvam-30B + Nemotron locally at ₹0 (commands in `MDAgents/RUN.md`) |
| 5 | Hindi source | **`ekacare/MedMCQA-Indic` (`hi`)** | Check the licence before use; if it's unclear, contact Eka Care |
| 6 | Hindi reviewer | **Hindi reader, not medical** | Reviewer checks fluency and meaning; medical-term errors are flagged by back-translation plus the English original, and the limitation is stated |
| 7 | Thinking mode | **`reasoning_effort=low`**, pinned (default, not asked) | Logged for every run |
| 8 | Scope | **PubMedQA/DDXPlus cut.** TeamMedAgents stays conditional (Phase 7). **gpt-4o-mini stays** as the Western reference | RQ3 = MedQA vs MedMCQA only; the OpenAI cost is part of decision 2 |

---

## 12. Cost estimate (modelled 2026-09-25; the pilot replaces it)

A token model that follows `utils.py` call by call, with today's prices (₹96/$; gpt-4o-mini $0.15/$0.60, gpt-3.5 $0.50/$1.50, Sarvam-105B ₹29.28/₹73.20 per 1M tokens).

It rests on these assumptions:

| Assumption | Value |
|---|---|
| Output length per call type | fixed per type (see the model) |
| Intermediate debate outcome | 60% silent / 30% brief / 10% full round |
| Complexity routing (basic / intermediate / advanced) | MedQA 30/50/20, MedMCQA 50/40/10 |
| Sarvam hidden reasoning ("thinking") | +250 output tokens per call |
| Hindi input tokens | ×1.3 vs English |
| Buffer for reruns | +25% |

| Plan | gpt-4o-mini (Exp 1 + Western ref) | Sarvam-105B API (everything Indian) |
|---|---|---|
| Full (every condition at 300/500) | ₹2.2k ($23) | **₹18.9k** (₹14.6k with thinking off) |
| Lean (forced-complexity + ablations on 200) | ₹1.9k ($19) | **₹11.4k** (₹8.8k with thinking off) |

Alternatives: Sarvam-30B on free Kaggle costs ₹0 but needs ~220–900 GPU-hours, which is 7–30 weeks of a ~30 h/week quota. Sarvam-M on Featherless is a flat $25/month.

## References

All of these were found on 2026-09-25. ✓ means I re-checked the arXiv ID against the arXiv API. Re-verify all of them before they go into the `.bib`.

- MDAgents, Kim et al., NeurIPS 2024. arXiv 2404.15155 (local PDF). https://github.com/mitmedialab/MDAgents
- Catfish Agent, Wang et al., 2025. arXiv 2505.21503 (local PDF)
- TeamMedAgents, Mishra et al., arXiv 2508.08115v3 (local PDF)
- MedMCQA, Pal et al., CHIL 2022, PMLR 174:248-260. https://github.com/medmcqa/medmcqa
- MedQA, Jin et al., 2021. arXiv 2009.13081. https://github.com/jind11/MedQA
- MedAgents, Tang et al., ACL Findings 2024. arXiv 2311.10537
- MedAgentsBench. arXiv 2503.07459 ✓
- MedAgentBoard, Zhu et al. arXiv 2505.12371 ✓
- AMR (Adaptive Memory and Reflection MAS for Medical QA). arXiv 2608.19029 ✓
- ArogyaSutra, Halder et al. arXiv 2606.13572 ✓
- HiMed, Jiang et al. arXiv 2605.24635 ✓
- Translating the Translator, Agrawal et al. arXiv 2609.15079 ✓
- Multi-agent architectures for rare disease diagnosis. arXiv 2603.06856 ✓
- Multi-OphthaLingua, Restrepo et al., AAAI 2025. arXiv 2412.14304
- MILU, Verma et al. arXiv 2411.02538
- HEALTH-PARIKSHA, Gumma et al. arXiv 2410.13671
- Altermatt et al., BMC Med Educ 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12057199/
- Kamdar et al., Indian J Ophthalmol 2026. https://pmc.ncbi.nlm.nih.gov/articles/PMC13002036/
- MedMCQA-Indic. https://huggingface.co/datasets/ekacare/MedMCQA-Indic
- Sarvam API docs: https://docs.sarvam.ai/api/getting-started/models · pricing: https://docs.sarvam.ai/api/getting-started/pricing · rate limits: https://docs.sarvam.ai/api/getting-started/ratelimits
- Krutrim status: https://www.medianama.com/2026/05/223-krutrim-ai-cloud-chip-ai-model-work/ · https://www.olakrutrim.com/ai-studio
- IndicTrans2: https://huggingface.co/ai4bharat/indictrans2-indic-en-1B
