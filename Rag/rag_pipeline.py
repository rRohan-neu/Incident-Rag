"""Incident KB RAG pipeline (local: Ollama + Chroma).

    python rag_pipeline.py --build                         # index the KB articles in Documents/
    python rag_pipeline.py --query "vpn drops after mfa"   # show the best-matching articles
    python rag_pipeline.py --ask "vpn drops after mfa"     # cited answer from the local LLM

app.py imports search_articles(), answer() and get_article() from here.
"""
import argparse
import json
import re
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "Documents"
PERSIST_DIR = BASE_DIR / "Chroma_store"
ARTICLES_FILE = BASE_DIR / "kb_articles.json"   # full text of each article, for the agent screen
COLLECTION = "incident_kb"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:3b"

# Below this similarity (0 to 1) we say "no KB article covers this" instead of guessing.
# Tune it with test questions that have no article (see SETUP.md).
MIN_SIMILARITY = 0.55

SECTIONS = ["Error messages", "Why this happens and why it comes back", "Team responsible",
            "Steps to resolve", "Workarounds if the steps do not work"]
META_FIELDS = ["KB number", "Title", "Author", "Date issued", "Last updated",
               "Owning team", "Escalation team", "Related incidents"]


# ---------------------------------------------------------------- embeddings
def get_embeddings():
    """nomic-embed-text works best with task prefixes on documents and queries."""
    from langchain_ollama import OllamaEmbeddings

    class NomicEmbeddings(OllamaEmbeddings):
        def embed_documents(self, texts):
            return super().embed_documents([f"search_document: {t}" for t in texts])

        def embed_query(self, text):
            return super().embed_query(f"search_query: {text}")

    return NomicEmbeddings(model=EMBED_MODEL)


def get_store():
    from langchain_chroma import Chroma
    if not PERSIST_DIR.is_dir():
        raise SystemExit("No vector store found. Run: python rag_pipeline.py --build")
    return Chroma(collection_name=COLLECTION, embedding_function=get_embeddings(),
                  persist_directory=str(PERSIST_DIR))


# ---------------------------------------------------------------- parsing the KB Word files
def parse_article(text, kb_id):
    """Split docx2txt output into metadata and the five sections."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    meta, sections, current = {"KB number": kb_id}, {s: [] for s in SECTIONS}, None
    i = 0
    while i < len(lines):
        line = lines[i]
        if line in META_FIELDS and i + 1 < len(lines) and current is None:
            meta[line] = lines[i + 1]
            i += 2
            continue
        if line in SECTIONS:
            current = line
        elif current:
            sections[current].append(line)
        i += 1
    # "Team responsible" lines look like "Owning team: X" / "Escalate to: Y"
    for l in sections["Team responsible"]:
        if l.lower().startswith("owning team:"):
            meta.setdefault("Owning team", l.split(":", 1)[1].strip())
        if l.lower().startswith("escalate to:"):
            meta.setdefault("Escalation team", l.split(":", 1)[1].strip())
    title = meta.get("Title") or next((l for l in lines if l.startswith(kb_id)), kb_id)
    meta["Title"] = re.sub(rf"^{kb_id}\s*", "", title).strip()
    return {"id": kb_id, "meta": meta, "sections": sections}


def article_text(a):
    parts = [f"{a['id']} {a['meta']['Title']}"]
    for s in SECTIONS:
        if a["sections"][s]:
            parts.append(s + ":\n" + "\n".join(a["sections"][s]))
    return "\n\n".join(parts)


# ---------------------------------------------------------------- indexing
def build_index():
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import Docx2txtLoader
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    files = sorted(DOCS_DIR.glob("KB*.docx"))
    files = [f for f in files if re.match(r"KB\d{5}", f.stem)]     # skips KB_Index_and_Known_Gaps.docx
    if not files:
        raise SystemExit(f"No KB articles (KB00001_....docx) found in {DOCS_DIR}")

    articles, chunks = {}, []
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    for f in files:
        kb_id = re.match(r"KB\d{5}", f.stem).group(0)
        raw = Docx2txtLoader(str(f)).load()[0].page_content
        a = parse_article(raw, kb_id)
        articles[kb_id] = a
        header = f"{kb_id} {a['meta']['Title']}"
        for n, piece in enumerate(splitter.split_text(article_text(a))):
            # Every chunk starts with the article title so it is never matched out of context.
            content = piece if piece.startswith(kb_id) else f"{header}\n{piece}"
            chunks.append(Document(page_content=content, metadata={
                "kb_article": kb_id, "title": a["meta"]["Title"], "doc_type": "kb",
                "owning_team": a["meta"].get("Owning team", ""), "chunk": n, "source": f.name}))
    print(f"Loaded {len(articles)} KB articles, split into {len(chunks)} chunks")

    # Rebuild from scratch so re-running --build never duplicates chunks.
    if PERSIST_DIR.is_file():
        PERSIST_DIR.unlink()
    elif PERSIST_DIR.is_dir():
        shutil.rmtree(PERSIST_DIR)
    Chroma.from_documents(chunks, embedding=get_embeddings(), collection_name=COLLECTION,
                          persist_directory=str(PERSIST_DIR),
                          collection_metadata={"hnsw:space": "cosine"})
    ARTICLES_FILE.write_text(json.dumps(articles, indent=1), encoding="utf-8")
    print(f"Saved vector store to {PERSIST_DIR} and article text to {ARTICLES_FILE.name}")


# ---------------------------------------------------------------- search and answer
_store = None
_articles = None


def _get_store():
    global _store
    if _store is None:
        _store = get_store()
    return _store


def get_article(kb_id):
    global _articles
    if _articles is None:
        _articles = json.loads(ARTICLES_FILE.read_text(encoding="utf-8")) if ARTICLES_FILE.exists() else {}
    return _articles.get(kb_id)


def search_articles(text, top=3, fetch=12):
    """Best-matching articles (not chunks) for a piece of text, best first."""
    if not text or len(text.strip()) < 3:
        return []
    best = {}
    for doc, dist in _get_store().similarity_search_with_score(text, k=fetch):
        kb = doc.metadata["kb_article"]
        sim = max(0.0, 1.0 - dist)                       # cosine distance -> similarity
        if kb not in best or sim > best[kb]["similarity"]:
            best[kb] = {"kb": kb, "title": doc.metadata["title"], "owning_team": doc.metadata.get("owning_team", ""),
                        "similarity": round(sim, 3), "snippet": doc.page_content}
    ranked = sorted(best.values(), key=lambda r: r["similarity"], reverse=True)[:top]
    for r in ranked:
        r["confident"] = r["similarity"] >= MIN_SIMILARITY
    return ranked


PROMPT = """You are a service desk assistant. Answer using ONLY the knowledge base articles below.
Rules:
- Cite the KB number for every step, like [KB00023].
- Give the Service Desk steps as a short numbered list. Mention the escalation team if the steps say to escalate.
- Never add steps, commands or settings that are not in the articles.
- If the articles do not cover the problem, say exactly: "No KB article covers this yet." and suggest escalating to the right team.

Knowledge base articles:
{context}

Agent's question or ticket description:
{question}

Answer:"""


def get_llm():
    from langchain_ollama import ChatOllama
    return ChatOllama(model=CHAT_MODEL, temperature=0)


def build_prompt(question):
    """Returns (prompt, matches). Uses only confident matches; empty context means 'not covered'."""
    matches = [m for m in search_articles(question, top=3) if m["confident"]]
    if not matches:
        return None, []
    blocks = []
    for m in matches[:2]:                                # the top two whole articles fit a small model well
        a = get_article(m["kb"])
        blocks.append(article_text(a) if a else m["snippet"])
    return PROMPT.format(context="\n\n---\n\n".join(blocks), question=question), matches


NOT_COVERED = ("No KB article covers this yet. Troubleshoot and escalate as usual, and record the fix in the "
               "ticket so it can become a new KB article after review.")


def answer(question):
    prompt, matches = build_prompt(question)
    if prompt is None:
        return NOT_COVERED, []
    return get_llm().invoke(prompt).content, matches


def stream_answer(question):
    """Yields text pieces as the model writes them (used by the Open WebUI connection)."""
    prompt, matches = build_prompt(question)
    if prompt is None:
        yield NOT_COVERED
        return
    for piece in get_llm().stream(prompt):
        yield piece.content
    yield "\n\nSources: " + ", ".join(f"{m['kb']} {m['title']}" for m in matches)


# ---------------------------------------------------------------- CLI
def main():
    parser = argparse.ArgumentParser(description="Incident KB RAG pipeline")
    parser.add_argument("--build", action="store_true", help="Load, chunk, embed and store the KB articles")
    parser.add_argument("--query", metavar="TEXT", help="Show the best-matching KB articles")
    parser.add_argument("--ask", metavar="TEXT", help="Get a cited answer from the local LLM")
    args = parser.parse_args()

    if args.build:
        build_index()
    if args.query:
        for i, m in enumerate(search_articles(args.query), 1):
            flag = "" if m["confident"] else "  (below threshold: not shown to agents)"
            print(f"[{i}] {m['kb']} {m['title']}  similarity {m['similarity']:.2f}{flag}")
    if args.ask:
        text, matches = answer(args.ask)
        print(text)
        if matches:
            print("\nSources:", ", ".join(m["kb"] for m in matches))
    if not (args.build or args.query or args.ask):
        parser.print_help()


if __name__ == "__main__":
    main()
