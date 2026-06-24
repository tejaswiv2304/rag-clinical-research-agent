# Clinical Research Ideation Agent

A simple multi-agent LLM system that helps a clinical researcher turn a broad
question (e.g. *"What might drive poor recovery after knee surgery?"*) into a
ranked set of **causal hypotheses worth studying**, each with its likely
confounders, written up as a plain-language research brief.

No dataset needed — you just type a clinical research question.

## Why I built this

I'm interested in multi-agent LLM systems and in how AI can support clinical
research. This project explores how a small team of specialized agents can
reason about *cause and effect* (not just correlation) in a medical context —
which is the core challenge of causal inference in clinical research.

## How it works

There are two versions in this repo:

- **`clinical_ideation_agent.py`** — the simple version: three LLM agents, no
  external data. Good for understanding the core idea.
- **`clinical_ideation_rag.py`** — the **RAG-powered** version (recommended):
  the Knowledge Agent retrieves **real abstracts from PubMed**, embeds them into
  a **vector store**, and summarizes the most relevant ones **with citations**.

The system is **three LLM agents working in sequence**, with a retrieval step
feeding the first one:

1. **RAG retrieval** — search PubMed → embed abstracts → vector similarity search
   → top-k most relevant sources.
2. **Knowledge Agent** — summarizes *only the retrieved sources*, citing each
   with a [number] tag (no hallucinated facts).
3. **Causal Agent** — proposes the top causal hypotheses ("X may cause Y"),
   names the confounders to control for, and ranks them.
4. **Explainer Agent** — turns it into a clean, plain-language research brief,
   with the PubMed sources listed at the end.

```
question
   -> [RAG: PubMed search -> embed -> vector search] -> top-k abstracts
   -> [Knowledge Agent] -> cited evidence summary
   -> [Causal Agent]    -> ranked hypotheses + confounders
   -> [Explainer Agent] -> final research brief (+ sources)
```

The RAG step grounds the system in real, citable literature instead of the
LLM's memory — which is what makes the output trustworthy.

### Tech used

Multi-agent orchestration · Retrieval-Augmented Generation (RAG) ·
vector embeddings + similarity search · PubMed E-utilities API · OpenAI LLM.
The `VectorStore` class is a minimal in-memory store; it can be swapped for
**FAISS, Chroma, or Pinecone** with no change to the rest of the pipeline.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"     # Windows: set OPENAI_API_KEY=your-key

python clinical_ideation_rag.py           # RAG version (recommended)
# or
python clinical_ideation_agent.py         # simple version, no retrieval
```

Then type any clinical research question when prompted. PubMed retrieval needs
no API key (it uses the free NCBI E-utilities API).

> Tip: you can use any LLM provider — just change the `client` and model
> lines at the top of the file.

## Example run

> Note: sample run for illustration. The PMIDs shown are representative of the
> kind of sources the tool retrieves; your actual run will pull live results
> from PubMed and the wording will vary.

**Input:**
```
What factors might cause poor blood-sugar control in type 2 diabetes?
```

**Console output:**
```
[RAG] Retrieving real abstracts from PubMed...

  Retrieved 5 relevant sources:
   [1] PMID 34521013: Medication adherence and glycaemic control in type 2 diabetes...
   [2] PMID 31889423: Association of body mass index with HbA1c in adults with...
   [3] PMID 33245601: Physical activity, diet quality and glycaemic outcomes...
   [4] PMID 35102877: Sleep duration and glycaemic control: a systematic review...
   [5] PMID 32760988: Psychological stress and blood glucose regulation in...

[1/3] Knowledge Agent: summarizing the sources (with citations)...

Several factors are linked to glycaemic control (HbA1c). Poor medication
adherence is consistently associated with higher HbA1c [1]. Higher BMI is
associated with worse control [2], but diet quality and physical activity
influence both BMI and HbA1c, so the relationship may be confounded [3].
Shorter sleep duration [4] and higher psychological stress [5] show
associations, though stress may also reduce medication adherence [1][5],
meaning these factors are interrelated rather than independent.

----------------------------------------------------------------------

[2/3] Causal Agent: proposing causal hypotheses...

1. Medication adherence -> HbA1c   (plausibility: HIGH)
   Confounders to control: psychological stress, socioeconomic status.
2. BMI -> HbA1c                     (plausibility: MEDIUM)
   Confounders to control: diet quality, physical activity.
3. Sleep duration -> HbA1c          (plausibility: EXPLORATORY)
   Confounders to control: stress, age.

Recommended first study: Hypothesis 1 — the effect is likely direct,
clinically actionable, and well supported by the retrieved evidence.

----------------------------------------------------------------------

[3/3] Explainer Agent: writing the research brief...

======================================================================
  FINAL RESEARCH IDEATION BRIEF
======================================================================

Research Ideation Brief — Poor Blood-Sugar Control in Type 2 Diabetes

Top question to investigate:
  Does improving medication adherence cause lower HbA1c?
  Why: It is well supported in the literature and is a direct, actionable
  target that clinicians can influence.

What to control for:
  Psychological stress and socioeconomic status, since both can affect
  adherence and blood sugar independently.

Other questions worth testing:
  - Does higher BMI cause higher HbA1c? (control for diet and exercise)
  - Does shorter sleep raise HbA1c? (exploratory; control for stress)

Caution:
  These are hypotheses, not proven causes. Each needs formal causal testing
  (e.g. with proper confounder adjustment) before any clinical conclusion.

Sources retrieved (PubMed):
  [1] PMID 34521013: Medication adherence and glycaemic control in type 2 diabetes
  [2] PMID 31889423: Association of body mass index with HbA1c in adults
  [3] PMID 33245601: Physical activity, diet quality and glycaemic outcomes
  [4] PMID 35102877: Sleep duration and glycaemic control: a systematic review
  [5] PMID 32760988: Psychological stress and blood glucose regulation
```

## Design notes

- The agents *reason about* causality; they don't run statistical estimation.
  A natural next step would be a fourth agent that tests a chosen hypothesis on
  real data using a causal-inference library (e.g. DoWhy).
- Architecture inspired by role-based multi-agent systems (e.g. the
  analyst → decision pattern used in frameworks like TauricResearch/TradingAgents),
  retargeted from finance to clinical research ideation.

## Possible extensions

- Add a **Skeptic Agent** that challenges each hypothesis ("could something else
  explain this?") in a short debate before ranking.
- Add **RAG** over real PubMed abstracts so the Knowledge Agent cites sources.
- Add a **Validation Agent** that runs DoWhy on a dataset to test one hypothesis.
