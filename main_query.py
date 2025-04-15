from pathlib import Path
from rag.rag_query import retrieve_context

index_path = Path("data/index/ticket_to_ride")
query = "Describe the endgame conditions."

retrieved_chunks = retrieve_context(index_path, query, top_k=3)

print("Retrieved Context Chunks:")
for i, chunk in enumerate(retrieved_chunks, 1):
    print(f"Chunk {i}:\n{chunk}\n{'-'*80}")
