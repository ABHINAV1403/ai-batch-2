from pathlib import Path


DOCS_PATH     = Path(r"C:\Users\lenovo\OneDrive\Desktop\infosys\ai-batch-2\products.csv") 
INDEX_PATH    = Path("./faiss_index")
REBUILD_INDEX = True    

EMBED_MODEL   = "models/embedding-001"   #google gemini embedding


CHAT_MODEL    = "llama-3.3-70b-versatile"     #groq llama model

TOP_K         = 20                         # how many chunks to retrieve
SEARCH_TYPE   = "mmr"                      # "mmr" | "similarity" | "similarity_score_threshold"
CHUNK_SIZE    = 300
CHUNK_OVERLAP = 20

# Ask a quick question at the end (set to None to skip)
QUESTION = "List all products from the Sports category with their names, descriptions,and prices."
