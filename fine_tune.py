from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
import torch
from datasets import load_dataset
import arxiv
import pandas as pd 
from typing import Dict, List 



# # 1. Base model
# model_name = "distilgpt2"
# # text -> numerical IDs
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token 


# model = AutoModelForCausalLM.from_pretrained(model_name) #GPT-Style model precits next token
# model = model.to(torch.float32) #std numerical precision for train/inference (32-bit floats)

# # 2. Load dataset from JSON
# dataset = load_dataset("json", data_files="boxing_dataset.json", field="train")["train"]

# # 3. Tokenize
# # structured input (Q: .. A:) -> aligns with JSON IR idea 
# def tokenize(batch):
#     texts = [f"Q: {instr} A: {out}" for instr, out in zip(batch["instruction"], batch["output"])]
#     return tokenizer(texts, truncation=True, padding="max_length", max_length=128)

# # apply tokenizer to all data 
# dataset = dataset.map(tokenize, batched=True)
# # ready for PyTorch
# dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# # 4. Low-Rank Adaption: add small "delta" matrices instead of retraining full model
# # Fine-tune efficiently with fewer resources 
# lora_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     r=2, # rank of LoRA update
#     lora_alpha=16, # scales delta
#     lora_dropout=0.05, #regularization
#     bias="none"
# )
# model = get_peft_model(model, lora_config)

# # 5. Training args
# # Batch-size: how many ex's processed before updated weights 
# # Epoch: Full passes over dataset
# # FOrward Pass -> compute outputs 
# # Backprop -> calculate gradients from loss 
# # Weights updated using gradients x learning rate 
# training_args = TrainingArguments(
#     output_dir="./output",
#     per_device_train_batch_size=1,    # CPU-friendly tiny batch
#     gradient_accumulation_steps=2,    # simulate batch size 2 (bigger batch if memory is limited)
#     learning_rate=2e-4,
#     num_train_epochs=40,
#     logging_steps=1,
#     save_strategy="no"
# )

# # 6. Data collator
# #  Prepares batches for training
# # Ensures inputs + labels are aligned for loss computation
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm=False   # Causal LM, not masked LM
# )

# # 7. Trainer
# #Handles  training loop: Forward, backprop, logging
# # You must provide model, dataset, and training args 
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     data_collator=data_collator
# )

# # ------------------------
# # 8. Train
# # ------------------------
# trainer.train()

# # 9. Save LoRA
# # saves delta weights, not full base model 
# # During inference: load base model + LoRA weights to produce fine-tuned outputs 
# model.save_pretrained("./output")
# print("Fine-tuning complete. LoRA weights saved in ./output")
