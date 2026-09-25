I am building a research project titled:

"Multi-Agent Medical Decision-Making with Indian LLMs"

I have attached:
1. Indian LLMs project proposal
2. MDAgents paper
3. TeamMedAgents paper
4. Catfish Agent paper

Repositories:
BASE CODE:
https://github.com/mitmedialab/MDAgents

MEDQA:
https://github.com/jind11/MedQA

Use the attached papers as the primary source for the research design. Use web research when current information is required, especially for Indian LLM availability and existing literature.

PROJECT IDEA:
Use MDAgents as our BASE framework and integrate Indian LLMs such as Sarvam/Krutrim as the reasoning backbone.

Our main India-grounded dataset is MedMCQA.
MedQA is used mainly to reproduce/compare with the original MDAgents setup.

The original MDAgents flow is:

Medical Question
→ Complexity Check
→ Expert Recruitment
→ Multi-Agent Reasoning
→ Synthesis
→ Final Decision

We want to investigate whether this framework works effectively with Indian LLMs and whether its collaboration mechanism needs adaptation rather than simply replacing GPT/Claude.

Potential extensions:
- Hindi input + IndicTrans2 translation
- Direct/native Hindi reasoning
- Catfish Agent for reducing silent agreement
- Selected TeamMedAgents teamwork mechanisms

IMPORTANT:
Do not combine all extensions immediately.
First establish the original MDAgents baseline, then add one modification at a time.

RESEARCH QUESTIONS:
1. Does MDAgents improve Indian LLM medical reasoning compared with a single Indian LLM?
2. Does adaptive complexity-based recruitment work effectively with Indian LLMs?
3. Does performance differ on India-grounded MedMCQA versus MedQA/PubMedQA/DDXPlus?
4. Does Catfish-style structured disagreement reduce silent agreement?
5. Does Hindi→English translation help or hurt reasoning compared with direct Hindi reasoning?
6. Which modifications actually contribute to performance?

EXPERIMENTAL PROGRESSION:

1. Original MDAgents + original model + MedQA
2. MDAgents + Indian LLM + MedQA
3. MDAgents + Indian LLM + MedMCQA
4. Indian LLM + MDAgents + Catfish
5. Indian LLM + MDAgents + Hindi/IndicTrans2
6. Indian LLM + MDAgents + native Hindi reasoning
7. Ablation studies

Evaluate:
- Accuracy
- Macro-F1 where appropriate
- Accuracy by complexity
- Single-agent vs multi-agent improvement
- Silent-agreement rate
- Answer revision/disagreement
- Token usage
- Number of LLM calls
- Latency/cost
- Translation quality and downstream accuracy

For every experiment, clearly define:
baseline → modification → hypothesis → dataset → metrics → expected analysis.

IMPLEMENTATION:
Start from the actual MDAgents GitHub repository.
Inspect the repository before proposing code changes.

Map:
paper architecture → repository files/functions → required modifications.

Prefer modular changes and an LLM adapter so that different models can be swapped easily.

Do not rewrite the entire repository unnecessarily.

RESEARCH PAPER:
Eventually build:
Introduction → Research Gap → Related Work → Methodology → Architecture → Dataset → Experiments → Ablations → Results → Error Analysis → Discussion → Limitations → Future Work → Conclusion.
(Give the latex code for the research paper and xml code for the model architecture diagram)

RULES:
- Do not fabricate results.
- Do not fabricate citations.
- Verify novelty claims through literature search.
- Do not assume current Indian LLM/API availability; verify it.
- Clearly distinguish MedQA from MedMCQA.
- Do not call MedMCQA a real-patient dataset.
- Do not treat benchmark accuracy as clinical validation.
- Explain why every proposed modification is needed.
- Keep the project reproducible.

START WITH PHASE 0 ONLY:

Explain in simple terms:
1. What we are building
2. What MDAgents gives us
3. What we are changing
4. Why MedMCQA is important
5. Where Indian LLMs fit
6. What is actually novel
7. Our research questions
8. The complete high-level architecture

After Phase 0, start writing code after my confirmation.