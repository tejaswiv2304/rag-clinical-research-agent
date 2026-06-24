"""
Clinical Research Ideation Agent
--------------------------------
A simple multi-agent system that helps a clinical researcher turn a broad
question into a ranked set of *causal hypotheses* worth studying.

Three LLM agents work in sequence, each passing its notes to the next:

    1. Knowledge Agent   -> gathers what is known about the topic
    2. Causal Agent      -> proposes cause-and-effect hypotheses + confounders
    3. Explainer Agent   -> writes a clean, plain-language research brief

No dataset required. You just type a clinical research question.

Author: <your name>
"""

import os
from openai import OpenAI

# ---------------------------------------------------------------------------
# Setup: one client, one helper function.
# An "agent" is just this helper called with a different role + task.
# ---------------------------------------------------------------------------

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"  # cheap and good enough; swap for any model you have


def ask_llm(role: str, task: str) -> str:
    """Send a role (system prompt) + task (user prompt) to the LLM, return text."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": role},
            {"role": "user", "content": task},
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Agent 1 — Knowledge Agent
# ---------------------------------------------------------------------------

def knowledge_agent(question: str) -> str:
    role = (
        "You are a clinical research literature analyst. You summarize what is "
        "currently known about a medical topic in a balanced, evidence-aware way."
    )
    task = (
        f"Research question: \"{question}\"\n\n"
        "Summarize the key factors that are believed to influence this outcome. "
        "For each factor, briefly note whether the evidence is strong, moderate, "
        "or weak. Importantly, point out where factors may influence one another "
        "(so we know correlation might not mean causation). Keep it concise."
    )
    return ask_llm(role, task)


# ---------------------------------------------------------------------------
# Agent 2 — Causal Hypothesis Agent
# ---------------------------------------------------------------------------

def causal_agent(question: str, evidence: str) -> str:
    role = (
        "You are a causal inference researcher. You carefully distinguish "
        "correlation from causation, and you always think about confounders "
        "(hidden factors that could explain an apparent cause-effect link)."
    )
    task = (
        f"Research question: \"{question}\"\n\n"
        f"Evidence summary from the literature agent:\n{evidence}\n\n"
        "Propose the TOP 3 causal hypotheses worth testing, in the form "
        "'X may cause Y'. For EACH hypothesis:\n"
        "  - state how plausible it is (high / medium / exploratory)\n"
        "  - name the most important confounder(s) that must be controlled for\n"
        "Then rank the hypotheses and recommend which ONE to study first, with a "
        "one-sentence reason. Be rigorous but concise."
    )
    return ask_llm(role, task)


# ---------------------------------------------------------------------------
# Agent 3 — Explainer Agent
# ---------------------------------------------------------------------------

def explainer_agent(question: str, hypotheses: str) -> str:
    role = (
        "You write clear, plain-language research briefs for clinicians and "
        "researchers who are not statisticians."
    )
    task = (
        f"Research question: \"{question}\"\n\n"
        f"Causal hypotheses from the causal agent:\n{hypotheses}\n\n"
        "Write a short 'Research Ideation Brief' with these sections:\n"
        "  - Top question to investigate (and why)\n"
        "  - What to control for\n"
        "  - Other questions worth testing\n"
        "  - A clear caution that these are hypotheses, not proven causes, and "
        "    need formal causal testing before any clinical conclusion.\n"
        "Use simple language."
    )
    return ask_llm(role, task)


# ---------------------------------------------------------------------------
# Orchestrator — chains the three agents together.
# This is the whole "multi-agent system": three notes passed in sequence.
# ---------------------------------------------------------------------------

def clinical_ideation_agent(question: str, verbose: bool = True) -> str:
    if verbose:
        print("\n[1/3] Knowledge Agent: gathering evidence...\n")
    evidence = knowledge_agent(question)
    if verbose:
        print(evidence + "\n" + "-" * 70)

    if verbose:
        print("\n[2/3] Causal Agent: proposing causal hypotheses...\n")
    hypotheses = causal_agent(question, evidence)
    if verbose:
        print(hypotheses + "\n" + "-" * 70)

    if verbose:
        print("\n[3/3] Explainer Agent: writing the research brief...\n")
    brief = explainer_agent(question, hypotheses)

    return brief


# ---------------------------------------------------------------------------
# Run it
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  CLINICAL RESEARCH IDEATION AGENT")
    print("  Turns a clinical question into ranked causal hypotheses.")
    print("=" * 70)

    user_question = input(
        "\nEnter a clinical research question\n"
        "(e.g. 'What might drive poor recovery after knee surgery?')\n\n> "
    ).strip()

    if not user_question:
        user_question = "What factors might cause poor blood-sugar control in type 2 diabetes?"
        print(f"\n(No input given — using example: \"{user_question}\")")

    final_brief = clinical_ideation_agent(user_question)

    print("\n" + "=" * 70)
    print("  FINAL RESEARCH IDEATION BRIEF")
    print("=" * 70 + "\n")
    print(final_brief)
