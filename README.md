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
or using `just`:
```bash
just start-app
```
---

## 🛠️ Future Improvements

- **Better PDF Parsing**: Improve handling of complex layouts (e.g., tables, columns). For example, the rules for Catan were manually curated (unlike the other games, which use official rulebooks), and the resulting text quality is noticeably better.

- **Improved Retrieval**: Refine how relevant information is selected from the documents, so the model receives better context before generating a summary or answer.

- **Fine-Tuning**: Train on Q&A data from community sources like BoardGameGeek to better handle unofficial rules and special cases.

- **Model Improvements**: Experiment with other instruction-tuned models to improve the quality and relevance of responses.

