from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from rag.rag_query import retrieve_context
from pathlib import Path
import json

with open("games.json", "r") as f:
    games = json.load(f)
games = {k: v["name"] for k, v in games.items()}


loaded_models = {}

def _get_model(model_id):
    if model_id not in loaded_models:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map={"": "cpu"},
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_safetensors=True,
        )
        loaded_models[model_id] = (tokenizer, model)
    return loaded_models[model_id]

def _build_summary_prompt(game_name, context, path="prompts/summary_template.txt"):
    with open(path, "r") as f:
        prompt_template = f.read()    
    return prompt_template.format(game_name=game_name, context=context)

def _build_rag_query_prompt():
    return f"main rules of the game"

def build_prompt_with_rag(game_id, top_k=3):
    """
    Builds a prompt using RAG (Retrieve and Generate) for the given game name.
    """
    print("Using RAG...")
    index_path = Path(f"data/index/{game_id}")
    rag_query = _build_rag_query_prompt()
    retrieved_chunks = retrieve_context(index_path, rag_query, top_k=top_k)
    context = "\n\n".join(retrieved_chunks)

    print("Building prompt...")
    game_name = games[game_id]
    prompt = _build_summary_prompt(game_name, context)
    print(prompt)
    return prompt


def generate_response(prompt, model_id, max_new_tokens=256):
    """
    Generates a response from the model given a prompt.
    """
    print(f"Using model {model_id}...")
    tokenizer, model = _get_model(model_id)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    print("Decoding output...")
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.split("Answer:")[-1].strip()
    print(answer)
    return answer


if __name__ == "__main__":
    game_id, game_name = list(games.items())[1]
    print(f"Summarizing rules for {game_id}...")
    prompt = build_prompt_with_rag(game_id, game_name)   
    response = generate_response(prompt, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print(response)