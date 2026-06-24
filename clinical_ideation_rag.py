"""
Clinical Research Ideation Agent  (RAG version)
-----------------------------------------------
A multi-agent LLM system that turns a clinical research question into a ranked
set of *causal hypotheses* worth studying -- grounded in REAL medical
literature via Retrieval-Augmented Generation (RAG).

Pipeline:

    question
      |
      v
    [ RAG retrieval ]  --> search PubMed -> embed abstracts -> vector search
      |
      v
    [ Agent 1: Knowledge ]  --> summarizes the retrieved abstracts (with citations)
      |
      v
    [ Agent 2: Causal ]     --> proposes cause-effect hypotheses + confounders
      |
      v
    [ Agent 3: Explainer ]  --> writes a plain-language research brief

The LLM is grounded in retrieved sources, so the evidence is real and cited.

Author: <your name>
"""

import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
from openai import OpenAI

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

CHAT_MODEL = "gpt-4o-mini"               # for the agents
EMBED_MODEL = "text-embedding-3-small"   # for RAG retrieval
TOP_K = 5                                # how many abstracts to retrieve


def ask_llm(role: str, task: str) -> str:
    """An 'agent' = this helper called with a specific role + task."""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": role},
            {"role": "user", "content": task},
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


# ===========================================================================
# PART 1: RETRIEVAL  -- pull real abstracts from PubMed
# ===========================================================================

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def pubmed_search(query: str, max_results: int = 12) -> list[str]:
    """Return a list of PubMed IDs for a query (free NCBI E-utilities API)."""
    url = (
        f"{EUTILS}/esearch.fcgi?db=pubmed"
        f"&term={urllib.parse.quote(query)}"
        f"&retmax={max_results}&retmode=json&sort=relevance"
    )
    with urllib.request.urlopen(url, timeout=20) as r:
        import json
        data = json.loads(r.read().decode())
    return data.get("esearchresult", {}).get("idlist", [])


def pubmed_fetch_abstracts(pmids: list[str]) -> list[dict]:
    """Fetch title + abstract text for a list of PubMed IDs."""
    if not pmids:
        return []
    url = (
        f"{EUTILS}/efetch.fcgi?db=pubmed"
        f"&id={','.join(pmids)}&rettype=abstract&retmode=xml"
    )
    with urllib.request.urlopen(url, timeout=30) as r:
        xml_data = r.read().decode()

    docs = []
    root = ET.fromstring(xml_data)
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID") or "unknown"
        title = article.findtext(".//ArticleTitle") or ""
        # An abstract can have multiple <AbstractText> sections; join them.
        parts = [a.text or "" for a in article.findall(".//AbstractText")]
        abstract = " ".join(parts).strip()
        if abstract:
            docs.append({"pmid": pmid, "title": title, "abstract": abstract})
    return docs


# ===========================================================================
# PART 2: A SIMPLE VECTOR STORE  -- embed docs, search by similarity
# (Swap this class for FAISS / Chroma / Pinecone in production -- same idea.)
# ===========================================================================

def embed_texts(texts: list[str]) -> np.ndarray:
    """Turn texts into embedding vectors using OpenAI embeddings."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


class VectorStore:
    """Minimal in-memory vector store: stores docs + their embeddings,
    retrieves the most similar ones by cosine similarity."""

    def __init__(self):
        self.docs: list[dict] = []
        self.vectors: np.ndarray | None = None

    def add(self, docs: list[dict]):
        self.docs = docs
        texts = [f"{d['title']}. {d['abstract']}" for d in docs]
        self.vectors = embed_texts(texts)

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        q = embed_texts([query])[0]
        # cosine similarity
        sims = self.vectors @ q / (
            np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(q) + 1e-8
        )
        top_idx = np.argsort(sims)[::-1][:k]
        return [self.docs[i] for i in top_idx]


def retrieve_evidence(question: str) -> list[dict]:
    """Full RAG retrieval step: search PubMed, embed, return top-k abstracts."""
    pmids = pubmed_search(question)
    time.sleep(0.4)  # be polite to the NCBI API
    docs = pubmed_fetch_abstracts(pmids)
    if not docs:
        return []
    store = VectorStore()
    store.add(docs)
    return store.search(question, k=TOP_K)


# ===========================================================================
# PART 3: THE THREE AGENTS
# ===========================================================================

def knowledge_agent(question: str, sources: list[dict]) -> str:
    """Agent 1 -- summarizes the RETRIEVED abstracts, with citations."""
    context = "\n\n".join(
        f"[{i+1}] (PMID {d['pmid']}) {d['title']}\n{d['abstract']}"
        for i, d in enumerate(sources)
    )
    role = (
        "You are a clinical research literature analyst. You summarize ONLY the "
        "sources provided, and you cite them using their [number] tags. You never "
        "invent facts that are not in the sources."
    )
    task = (
        f"Research question: \"{question}\"\n\n"
        f"Retrieved sources:\n{context}\n\n"
        "Summarize the key factors these sources say influence the outcome. "
        "Cite each claim with its [number]. Note where factors may influence each "
        "other (so correlation may not mean causation). Be concise."
    )
    return ask_llm(role, task)


def causal_agent(question: str, evidence: str) -> str:
    """Agent 2 -- proposes ranked causal hypotheses + confounders."""
    role = (
        "You are a causal inference researcher. You distinguish correlation from "
        "causation and always think about confounders (hidden factors that could "
        "explain an apparent cause-effect link)."
    )
    task = (
        f"Research question: \"{question}\"\n\n"
        f"Evidence summary (with citations):\n{evidence}\n\n"
        "Propose the TOP 3 causal hypotheses worth testing ('X may cause Y'). "
        "For each: state plausibility (high/medium/exploratory) and the key "
        "confounder(s) to control for. Then rank them and recommend ONE to study "
        "first, with a one-sentence reason."
    )
    return ask_llm(role, task)


def explainer_agent(question: str, hypotheses: str) -> str:
    """Agent 3 -- writes the plain-language research brief."""
    role = (
        "You write clear, plain-language research briefs for clinicians who are "
        "not statisticians."
    )
    task = (
        f"Research question: \"{question}\"\n\n"
        f"Causal hypotheses:\n{hypotheses}\n\n"
        "Write a short 'Research Ideation Brief' with: Top question to investigate "
        "(and why), What to control for, Other questions worth testing, and a clear "
        "caution that these are hypotheses (not proven causes) needing formal "
        "causal testing. Use simple language."
    )
    return ask_llm(role, task)


# ===========================================================================
# ORCHESTRATOR
# ===========================================================================

def clinical_ideation_agent(question: str) -> str:
    print("\n[RAG] Retrieving real abstracts from PubMed...\n")
    sources = retrieve_evidence(question)

    if not sources:
        print("  (No PubMed results found -- the Knowledge Agent will reason "
              "from general knowledge instead.)\n")
        evidence = ask_llm(
            "You are a clinical research literature analyst.",
            f"Summarize known factors affecting: {question}. Note possible "
            "confounding between factors.",
        )
    else:
        print(f"  Retrieved {len(sources)} relevant sources:")
        for i, d in enumerate(sources):
            print(f"   [{i+1}] PMID {d['pmid']}: {d['title'][:80]}...")
        print()
        print("[1/3] Knowledge Agent: summarizing the sources (with citations)...\n")
        evidence = knowledge_agent(question, sources)
    print(evidence + "\n" + "-" * 70)

    print("\n[2/3] Causal Agent: proposing causal hypotheses...\n")
    hypotheses = causal_agent(question, evidence)
    print(hypotheses + "\n" + "-" * 70)

    print("\n[3/3] Explainer Agent: writing the research brief...\n")
    brief = explainer_agent(question, hypotheses)

    # Append the source list so the brief is fully traceable.
    if sources:
        refs = "\n".join(
            f"  [{i+1}] PMID {d['pmid']}: {d['title']}"
            for i, d in enumerate(sources)
        )
        brief += "\n\nSources retrieved (PubMed):\n" + refs
    return brief


# ===========================================================================
# RUN
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  CLINICAL RESEARCH IDEATION AGENT  (RAG-powered)")
    print("  PubMed retrieval + 3 LLM agents -> ranked causal hypotheses")
    print("=" * 70)

    user_question = input(
        "\nEnter a clinical research question\n"
        "(e.g. 'What might drive poor recovery after knee surgery?')\n\n> "
    ).strip()

    if not user_question:
        user_question = "What factors might cause poor blood-sugar control in type 2 diabetes?"
        print(f"\n(No input -- using example: \"{user_question}\")")

    final_brief = clinical_ideation_agent(user_question)

    print("\n" + "=" * 70)
    print("  FINAL RESEARCH IDEATION BRIEF")
    print("=" * 70 + "\n")
    print(final_brief)
