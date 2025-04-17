start-app:
    lsof -ti:7860 | xargs kill -9 || true
    python app.py

push-all:
    git push origin main
    git push hf main  