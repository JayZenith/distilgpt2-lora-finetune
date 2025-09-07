# distilgpt2-lora-finetune

Fine-tune **DistilGPT2** on a custom Q/A-style dataset using **LoRA (Low-Rank Adaptation)** for parameter-efficient training. Includes example code for inference using the trained LoRA weights.

---

## Features

- Uses **PEFT / LoRA** to train only small delta matrices instead of full model weights.
- Efficient training on CPU or GPU with minimal memory usage.
- Prepares dataset in `Q: ... A: ...` instruction format.
- Easy inference with saved LoRA weights.

---

## Requirements

```bash
pip install torch transformers datasets peft
```

## Dataset
- should be a JSON file (boxing_dataset.json) with structure:
```bash
{
  "train": [
    {"instruction": "Who won the fight between Mike Tyson and Evander Holyfield in 1996?", "output": "Evander Holyfield won."},
    {"instruction": "List three notable fights of Muhammad Ali.", "output": "Fight 1, Fight 2, Fight 3"}
  ]
}

```


## Fine-Tuning
```bash
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch

# 1. Load base model
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name).to(torch.float32)

# 2. Load dataset
dataset = load_dataset("json", data_files="boxing_dataset.json", field="train")["train"]

# 3. Tokenize
def tokenize(batch):
    texts = [f"Q: {instr} A: {out}" for instr, out in zip(batch["instruction"], batch["output"])]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=128)
dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# 4. LoRA setup
lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=16, lora_dropout=0.05, bias="none")
model = get_peft_model(model, lora_config)

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=40,
    logging_steps=1,
    save_strategy="no"
)

# 6. Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 7. Trainer
trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=data_collator)

# 8. Train
trainer.train()

# 9. Save LoRA weights
model.save_pretrained("./output")
print("LoRA fine-tuning complete. Weights saved in ./output")

```

## Inference
```bash
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(base_model, "./output")
model.eval()
device = torch.device("cpu")
model.to(device)

questions = [
    "Who won the fight between Mike Tyson and Evander Holyfield in 1996?",
    "List three notable fights of Muhammad Ali.",
    "Who is considered the GOAT of boxing?",
    "Summarize Canelo Alvarez's professional boxing achievements."
]

for q in questions:
    prompt = f"Q: {q} A:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=50,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(answer)
    print("--------------------------------------------------")

```


## Notes
- LoRA only saves small delta weights (./output), so the base model is required for inference.
- This setup is CPU-friendly but will be faster on a GPU.
- Designed for instruction-following / Q&A datasets.