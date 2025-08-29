# rag.py
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import List

# Load .env first
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Quick API key checks
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set. Please add it to your .env file.")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set. Please add it to your .env file.")

# Import config (your variables like DOCS_PATH, INDEX_PATH, CHUNK_SIZE... live here)
from config import (
    DOCS_PATH,
    INDEX_PATH,
    REBUILD_INDEX,
    EMBED_MODEL,
    CHAT_MODEL,
    TOP_K,
    SEARCH_TYPE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    QUESTION,
)

# LangChain / vector store / loaders / embeddings / LLM
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

import pandas as pd


# -------------------------
# File discovery & loading
# -------------------------
def find_files(path: Path) -> List[Path]:
    """Return list of file paths to process (supports file or directory)."""
    if path.is_file():
        return [path]
    exts = {".txt", ".md", ".pdf", ".csv"}
    return [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def load_documents(paths: List[Path]) -> List[Document]:
    """
    Load supported files into LangChain Document objects.
    CSV -> each row becomes one Document with metadata (ProductName, Category, Price, etc.)
    TXT/MD -> whole file as one Document
    PDF -> handled by PyPDFLoader (each page becomes a Document)
    """
    docs: List[Document] = []
    for p in paths:
        try:
            suf = p.suffix.lower()
            if suf in {".txt", ".md"}:
                # TextLoader produces Document(s)
                docs.extend(TextLoader(str(p), encoding="utf-8").load())

            elif suf == ".pdf":
                docs.extend(PyPDFLoader(str(p)).load())

            elif suf == ".csv":
                # Read CSV and convert each row into a Document
                df = pd.read_csv(str(p))
                # optional: limit rows for testing (remove or adjust as needed)
                # df = df.head(500)

                # if ProductName/Description columns exist, use them; otherwise stringify row
                for _, row in df.iterrows():
                    # build a readable content string for embeddings / retrieval
                    if "ProductName" in df.columns or "Description" in df.columns:
                        name = str(row.get("ProductName", ""))
                        desc = str(row.get("Description", ""))
                        content = f"Product: {name}\nDescription: {desc}"
                    else:
                        # fallback: join all columns
                        content = " | ".join([f"{col}: {row[col]}" for col in df.columns])

                    # gather metadata for source / filtering / citation
                    metadata = {"source": str(p)}
                    for col in ("ProductID", "ProductName", "Category", "Price", "Description"):
                        if col in df.columns:
                            metadata[col.lower()] = str(row.get(col, ""))

                    docs.append(Document(page_content=content, metadata=metadata))
            else:
                print(f"[WARN] Unsupported file type: {p}")
        except Exception as e:
            print(f"[WARN] Could not load {p}: {e}")
    return docs


# -------------------------
# Splitting / chunking
# -------------------------
def split_documents(docs: List[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


# -------------------------
# FAISS index build/load
# -------------------------
def build_or_load_faiss(chunks: list[Document], rebuild: bool) -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=os.getenv("GEMINI_API_KEY")  # ✅ force usage of your key
    )

    if rebuild:
        print("🔁 Building FAISS index from documents...")
        vs = FAISS.from_documents(chunks, embeddings)
        INDEX_PATH.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(INDEX_PATH))
        print(f"✅ Saved index to: {INDEX_PATH.resolve()}")
        return vs

    print(f"📦 Loading FAISS index from: {INDEX_PATH.resolve()}")
    vs = FAISS.load_local(
        str(INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print("✅ Loaded FAISS index.")
    return vs

# -------------------------
# Retriever and RAG chain
# -------------------------
def make_retriever(vectorstore: FAISS):
    return vectorstore.as_retriever(search_type=SEARCH_TYPE, search_kwargs={"k": TOP_K})


def make_rag_chain(retriever):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a concise, careful assistant. Answer ONLY from the provided context. "
             "If the answer is not in the context, say you don't know. Cite sources by filename and metadata."),
            ("human", "Question:\n{input}\n\nContext:\n{context}"),
        ]
    )

    # Groq as the chat LLM
    llm = ChatGroq(model=CHAT_MODEL, temperature=0.2)

    # Create doc combiner (stuffing)
    doc_chain = create_stuff_documents_chain(llm, prompt)

    # Compose retrieval -> doc_chain
    rag_chain = create_retrieval_chain(retriever, doc_chain)
    return rag_chain


# -------------------------
# Helpers
# -------------------------
def format_sources(ctx: List[Document]) -> str:
    lines = []
    for d in ctx:
        src = d.metadata.get("source") or "unknown"
        # try include product name if available in metadata
        prod = d.metadata.get("productname") or d.metadata.get("product_name") or d.metadata.get("productname")
        if prod:
            lines.append(f"- {Path(src).name} (product: {prod})")
        else:
            lines.append(f"- {Path(src).name}")
    return "\n".join(lines)


# -------------------------
# Main orchestration
# -------------------------
def main():
    # 1) find files
    print(f"📁 Scanning docs under: {DOCS_PATH.resolve() if isinstance(DOCS_PATH, Path) else DOCS_PATH}")
    files = find_files(Path(DOCS_PATH))
    if not files:
        raise SystemExit("No supported files (.txt/.md/.pdf/.csv) found under DOCS_PATH.")

    print(f"📥 Loading {len(files)} files...")
    docs = load_documents(files)
    print(f"   -> Loaded {len(docs)} documents (pre-split).")

    # 2) split into chunks (if rebuilding)
    chunks = []
    if REBUILD_INDEX:
        print(f"✂️  Splitting {len(docs)} docs (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
        chunks = split_documents(docs)
        print(f"   -> Produced {len(chunks)} chunks.")

    # 3) build or load FAISS
    vectorstore = build_or_load_faiss(chunks, rebuild=REBUILD_INDEX)

    # 4) make retriever + rag chain
    retriever = make_retriever(vectorstore)
    rag = make_rag_chain(retriever)

    # 5) run a test question (if provided)
    if QUESTION:
        print(f"\n❓ Question: {QUESTION}")
        # Many chain implementations accept {"input": question}
        try:
            result = rag.invoke({"input": QUESTION})
        except Exception:
            # fallback to calling the chain directly
            result = rag({"input": QUESTION})

        # result may be a dict or a string depending on chain type
        if isinstance(result, dict):
            answer = result.get("answer") or result.get("output") or result.get("result") or str(result)
            ctx = result.get("context", []) or result.get("source_documents", []) or []
        else:
            answer = str(result)
            ctx = []

        print("\n🧠 Answer:\n")
        print(answer.strip() if isinstance(answer, str) else answer)

        


if __name__ == "__main__":
    main()
