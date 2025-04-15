from pathlib import Path
from rag.rag_ingest import ingest_documents
import json
import logging

logging.getLogger().setLevel(logging.INFO)

# Load the games configuration
with open("../games.json", "r") as f:
    games = json.load(f)

# Iterate over each game configuration
for _, game in games.items():
    game_name = game["name"]
    pdf_url = game["rulebook"]
    index_path = Path(game["index"])
    
    logging.info(f"\n\nIngesting data for {game_name}...")
    ingest_documents(pdf_url, index_path)

