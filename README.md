---
title: LLM Boardgame Assistant
emoji: 🎲
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: "5.23.3"
app_file: app.py
pinned: false
---

# 🧠 LLM Boardgame Assistant

An assistant that summarizes board game rules using an LLM and retrieval-augmented generation (RAG).


---

## 🚀 How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Ingest rulebooks (if not already done):
```bash
python main_ingest.py
```

3. Start the UI:
```bash
python app.py
```
