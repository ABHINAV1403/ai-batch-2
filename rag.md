[RAG : Retrieval Augmented Generation ]

Why RAG: 
- LLM rely only on training data ( knowledge cutoff )
- Hallucinations: Confindent but wrong answers 

Solution: RAG
- Benefit: Up to date, accurate, domain-specific answers.
- Visual: Compare LLM-only vs RAG pipeline 

RAG - "LLM WITH A LIBRARY + INTERNET"


LLM ONLY      VS            RAG
Pre-trained DATA             External + live data
May hallucinate              Higher Accuracy
Limited (Knowledge cutoff)   Always up to date
General                      Domain adaptation


RAG Architecture

Retriever: -> Find relevant documents (Product Document )
Augmenter: -> Inject into Prompt
Generator (LLM): -> Produces Final answer 

Workflow
(Query: ( hey i have issue with my iphone 14 with battery what should i do ? ))
                                    |
(Retrieve: It will retrieve best probable/answerable chunks to answer )
                                    |
(Augmented: ["prompt"+ [Data of chunk1] + [data of chunk 2]])
                                    |
(Generator: LLM will answer based on the relevant chunk)
                                    |
                            (  Final  Answer   )        


Corpus: Dataset 


Retrieval: Find the best chunk out of the (Dataset to answer )
- Sparse Retrieval : TF-IDF , BM25 
TF (Term Frequency): This measures how freq a term appears in a doc, A higher term freq suggests the term is more relevant to the documents contents
IDF ( Inverse Document Frequency ):  This mesasures how unique and rare a term is across the entire corpus. Terms that appear in many docs have low IDF, which rare terms have high IDF


                                                        |
                                        ( It's improvement of TF-IDF )        
BM25 (ENCODER): Ranking function used by search engines to estimate the relevance of documents to give a search query.

Dense Retrieval: Embedding Vector + Similarity search

[ This is an apple ] -> (GoogleEmbedding) -> [ 0.565,56.,453. 569.60, ]

Semantic Search : ( it searches for the meaning of the query )

This is  my fav food = apple    
This is my  fav  car   -> ( my favourite food is ) -> (semantic search) -> your fav food is apple 
THIS IS my  fav cloth 


Hybrid Retrieval: 
Dense + BM25 


Place to store these 
Vector Databases :  [ 0.565,56.45453. 569.60 ]
Store in vector DB's
Local: FAISS ( pip install faiss-cpu )
Other Vecotr DB's: Pinecone, ChromaDB, Weavite, Milvus

RAG Piplines:
- Ingest DOCs ( pdf, website, knowledge base)


Corpus

[ The DigitalOcean Cloud Controller Manager lets you provision DigitalOcean Load Balancers. To configure advanced settings for the load balancer, add the settings under annotations in the metadata stanza in your service configuration file. To prevent misconfiguration, an invalid value for an annotation results in an error when you apply the config file.

Additional configuration examples are available in the DigitalOcean Cloud Controller Manager repository.   ]


- Chunk text into passage

[ The DigitalOcean Cloud Controller Manager lets you provision DigitalOcean Load Balancers.] : chunk 1
[ o configure advanced settings for the load balancer, add the settings under annotations in the metadata stanza in your service configuration file ] : chunk 2
[ To prevent misconfiguration, an invalid value for an annotation results in an error when you apply the config file.]: chunk 3

- Create Embedding and store in vector DB
[0.434,340,45343,4353,353423,4245]
[343423,3432432,34234,223432,4232]
[443243,34343,22,32,232,2,45]


- Retrieve top-k docs for query  (top-k)
top-k : Most compaitable chunk based on the query (top-k=3)

- Augment query with docs

- Generate final answer 



# Installing dependencies
pip install -U/ langchain langchain-community langchain-google-genai langchain-text-splitters faiss-cpu pypdf python-dotenv langchain-groq

# Set your Google APi key
GOOGLE_API_KEY=''

ai.google.dev

<!-- # Build the index ( from a folder or a single file)
python rag_faiss_gemini.py --docs ./my_docs --index-path ./faiss_index --rebuild -->


# Import
# Langchain core + utilites
form langchain_core.document import Document
from langchain_core.prompts import ChatPromptTemplate

# Text Splitter
from langchain_text_splitters import  RecursiveCharacterTextSplitter

# Loader for file
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# Faiss vector store
from langchain_community.vectorstores import FAISS

# Google Generative Ai ( Gemini )
from langchain_google_genai import  GoogleGenerativeAiEmbedddings
from langchain_groq import ChatGroq


# RAG chain Builders
from langchain.chains.combine_documents import  create_stuff_documents_chain
from langhchain.chain import create_retrieval_chain


# ───────────────────────────────────────────────────────────────────────────────
# CONFIG: change these to your paths/models as you like
# ───────────────────────────────────────────────────────────────────────────────
DOCS_PATH     = Path("./my_docs")          # folder or single file (.txt, .md, .pdf)
INDEX_PATH    = Path("./faiss_index")      # where FAISS index is stored
REBUILD_INDEX = True                       # True to (re)build from docs; False to load
EMBED_MODEL   = "models/embedding-001"
CHAT_MODEL    = "llama-3.3-70b-versatile" 
TOP_K         = 4                          # how many chunks to retrieve
SEARCH_TYPE   = "mmr"                      # "mmr" | "similarity" | "similarity_score_threshold"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 120

# Ask a quick question at the end (set to None to skip)
QUESTION = "Give me a 2-line summary of the docs and cite sources."




# ───────────────────────────────────────────────────────────────────────────────
# STEP 1) ENSURE API KEY
# Why: Google models require GOOGLE_API_KEY; fail early if missing.
# ───────────────────────────────────────────────────────────────────────────────
if not os.environ.get("GOOGLE_API_KEY"):
    raise SystemExit(
        "GOOGLE_API_KEY is not set. Get one at https://ai.google.dev/ and set it.\n"
        "macOS/Linux: export GOOGLE_API_KEY='YOUR_KEY'\n"
        "Windows:     setx GOOGLE_API_KEY \"YOUR_KEY\" (then open a new terminal)"
    )


# ───────────────────────────────────────────────────────────────────────────────
# STEP 2) FIND & LOAD DOCUMENTS
# What: Load .txt, .md with TextLoader; .pdf with PyPDFLoader into LangChain docs.
# ───────────────────────────────────────────────────────────────────────────────
def find_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    exts = {".txt", ".md", ".pdf"}
    return [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def load_documents(paths: list[Path]) -> list[Document]:
    docs: list[Document] = []
    for p in paths:
        try:
            if p.suffix.lower() in {".txt", ".md"}:
                docs.extend(TextLoader(str(p), encoding="utf-8").load())
            elif p.suffix.lower() == ".pdf":
                docs.extend(PyPDFLoader(str(p)).load())
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")
    return docs

# ───────────────────────────────────────────────────────────────────────────────
# STEP 3) SPLIT DOCS INTO CHUNKS
# Why: RAG works best when you chunk long content; overlap preserves context.
# ───────────────────────────────────────────────────────────────────────────────
def split_documents(
    docs: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


# ───────────────────────────────────────────────────────────────────────────────
# STEP 4) MAKE / LOAD FAISS VECTOR STORE
# What: Embed chunks with Google Gemini embeddings, store vectors in FAISS.
# Note: FAISS save/load may use pickle; only load indexes you trust.
# ───────────────────────────────────────────────────────────────────────────────
def build_or_load_faiss(chunks: list[Document], rebuild: bool) -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

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
        allow_dangerous_deserialization=True,  # see note above
    )
    print("✅ Loaded FAISS index.")
    return vs


# STEP 5) BUILD THE RETRIEVER
# What: Turn vector store into a retriever (how we fetch relevant chunks).
# ───────────────────────────────────────────────────────────────────────────────
def make_retriever(vectorstore: FAISS):
    return vectorstore.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={"k": TOP_K},
    )

# ───────────────────────────────────────────────────────────────────────────────
# STEP 6) CREATE THE RAG CHAIN (Prompt + LLM + Stuffing)
# What: A small grounded prompt that stuffs retrieved context into the LLM call.
# ───────────────────────────────────────────────────────────────────────────────
def make_rag_chain(retriever):
    # a) Define a grounded prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a concise, careful assistant. Answer ONLY from the provided "
             "context. If the answer is not in the context, say you don't know. "
             "Cite sources by filename and, if present, page."),
            ("human", "Question:\n{input}\n\nContext:\n{context}"),
        ]
    )

    # b) Choose the chat model (Gemini)
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.2)
    llm = Chatgroq(modle="")

    # c) Create a doc-combining chain (stuff retrieved docs into the prompt)
    doc_chain = create_stuff_documents_chain(llm, prompt)

    # d) Compose the full RAG chain: retriever -> doc_chain
    rag_chain = create_retrieval_chain(retriever, doc_chain)  (chain)
    return rag_chain 


# ───────────────────────────────────────────────────────────────────────────────
# STEP 7) PRETTY-PRINT SOURCES
# What: Show filenames (and page numbers) of retrieved chunks for transparency.
# ───────────────────────────────────────────────────────────────────────────────
def format_sources(ctx: list[Document]) -> str:
    lines = []
    for d in ctx:
        src = d.metadata.get("source") or d.metadata.get("file_path") or "unknown"
        page = d.metadata.get("page")
        name = Path(src).name
        lines.append(f"- {name}" + (f" (page {page})" if page is not None else ""))
    return "\n".join(lines)



# ───────────────────────────────────────────────────────────────────────────────
# STEP 8) GLUE IT ALL TOGETHER
# Run once to build/load the index, then ask a question.
# ───────────────────────────────────────────────────────────────────────────────
def main():
    # 8a) If rebuilding, find+load+split documents
    chunks: list[Document] = []
    if REBUILD_INDEX:
        print(f"📁 Scanning docs under: {DOCS_PATH.resolve()}")
        files = find_files(DOCS_PATH)
        if not files:
            raise SystemExit("No .txt/.md/.pdf files found.")
        print(f"📥 Loading {len(files)} files...")
        docs = load_documents(files)

        print(f"✂️  Splitting {len(docs)} docs (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
        chunks = split_documents(docs)

    # 8b) Build or load FAISS
    vectorstore = build_or_load_faiss(chunks, rebuild=REBUILD_INDEX)

    # 8c) Make retriever + RAG chain
    retriever = make_retriever(vectorstore)
    rag = make_rag_chain(retriever)

    # 8d) Ask a single question (or replace this with your own loop/UI)
    if QUESTION:
        print(f"\n❓ Question: {QUESTION}")
        result = rag.invoke({"input": QUESTION})
        answer = result.get("answer") or result.get("output") or str(result)
        print("\n🧠 Answer:\n" + answer.strip())

        ctx = result.get("context", [])
        if ctx:
            print("\n📚 Sources:")
            print(format_sources(ctx))

if __name__ == "__main__":
    main()





































[TICKET -[
      ticket_id,
      ticket_content,
      ticket_cat, ->(LLM to categorize )   -> ( save it inot Google Sheets ) -> (Ticket Resolution Model will run ) -> (solution ) -> (Update the status of ticker on                                                 sheets )


      ticket_timestamp,
      ticket_by,
      ticket_status
]]

[
    "ticket_id":"34vervewe3t",
    "ticket_content":"i have issue with my phone"
    "ticket_category:"product_maintenance/buy",
    "ticket_timestamp":"205-08-22:00:00:00:IST",
    "ticket_by":"email"
]