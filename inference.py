from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# ------------------------
# 1. Load base model + LoRA
# ------------------------
model_name = "distilgpt2"

# Tokenize input prompt-> numerical IDs
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # all sequences in batch have same length
base_model = AutoModelForCausalLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(base_model, "./output")  # load your fine-tuned LoRA weights
model.eval()
device = torch.device("cpu")
model.to(device)

# ------------------------
# 2. Questions to test
# ------------------------
questions = [
    "Who won the fight between Mike Tyson and Evander Holyfield in 1996?",
    "List three notable fights of Muhammad Ali.",
    "Who is considered the GOAT of boxing?",
    "Summarize Canelo Alvarez's professional boxing achievements."
]

# ------------------------
# 3. Inference loop
# ------------------------
for q in questions:
    prompt = f"Q: {q} A:"  # trailing "A:" cues model to answer
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)    
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=50, #max tokens to generate 
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7, #randomness 
        top_p=0.9, #nucleus sampling, focus on top probability mass 
        repetition_penalty=1.2   # avoid repeated tokens
    )


    # decode output -> human readable text 
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(answer)
    print("--------------------------------------------------")
