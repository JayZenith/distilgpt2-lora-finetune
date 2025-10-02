from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset_utils import extract_label_from_output, VALID_CLASSES, get_arxiv_paper_info, get_message_prompt_tokenized, build_prompts
from datasets import Dataset
from torch.utils.data import DataLoader
from peft import PeftModel
import torch
import pandas as pd
from transformers import pipeline


def generate_outputs(prompts: list[str], model, tokenizer):
    #tokenize prompts into tensors 
    tokenized: dict[str, torch.Tensor] = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    # produce sequences via generate() 
    output_batch: torch.Tensor = model.generate(
        input_ids=tokenized["input_ids"],
        attention_mask=tokenized["attention_mask"],
        max_new_tokens=20,
        do_sample=True,
        temperature=0.2,
        top_p=1
    )
    
    #Decode into text 
    decoded_batch: list[str] = tokenizer.batch_decode(output_batch, skip_special_tokens=True)
    # PIck one of valid categories 
    predictions: list[str] = extract_label_from_output(decoded_batch, tokenizer)
    #return list of pedictions 
    return predictions

# convinient function, takes a single (title,abstract) -> builds prompt 
# def generate_output_from_input(model, tokenizer, title: str, abstract: str):
#     prompts = get_message_prompt_tokenized(tokenizer, [title], [abstract])
#     predictions = generate_outputs(prompts, model, tokenizer)
#     return predictions

def generate_output_from_input(model, tokenizer, title, abstract):
    system_prompt = (
        "You are an AI system that reads the title and summary of a paper and "
        "classifies it into the correct computer science category.\n"
        "You must return the *Category Description* and explain briefly why.\n\n"
        "Valid categories:\n" +
        "\n".join([f"- {c}" for c in VALID_CLASSES]) +
        "\n\n"
    )
    prompt = system_prompt + f"Title: {title}\nSummary: {abstract}\n\nAnswer:"
    predictions = generate_outputs([prompt], model, tokenizer)  # note: wrapped in a list
    return predictions



def test_model(dataloader: DataLoader, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    comparison_df = {
        "predictions": [],
        "labels": [], #Category Description
        "titles": []
    }
    
    # loop over batches in a test dataloader 
    for batch in dataloader:
        # HuggingFace Dataset + DataLoader returns tensors or dicts of lists
        prompts : list[str] = batch["prompt"]       # list of prompts
        labels: list[str] = batch["labels"]        # list of ground truth labels
        titles: list[str] = batch["title"]         # list of titles

        predictions: list[str] = generate_outputs(prompts=prompts, model=model, tokenizer=tokenizer)

        comparison_df["labels"].extend(labels)
        comparison_df["predictions"].extend(predictions)
        comparison_df["titles"].extend(titles)
    
    # DataFrame
    comparison_df = pd.DataFrame(comparison_df)
    accuracy = (comparison_df["labels"] == comparison_df["predictions"]).mean()
    num_invalid_pred = (~comparison_df["predictions"].isin(VALID_CLASSES)).mean()

    print("\n=== SAMPLE OUTPUTS (first 10) ===")
    print(comparison_df.head(10))

    print("\n=== FULL PREDICTIONS ===")
    for _, row in comparison_df.iterrows():
        print(f"Title: {row['titles']}\nLabel: {row['labels']}\nPred: {row['predictions']}\n---")

    return {"accuracy": accuracy, "invalid prediction": num_invalid_pred}


model_id = "unsloth/Llama-3.2-1B"
device = "cuda" if torch.cuda.is_available() else "cpu"

#pad on left side for valid rectangular tensor 
tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", use_auth_token=True)

# Pad token = EOS (LLaMa has no special pad token)
tokenizer.pad_token= tokenizer.eos_token

#Causal LM precicts next token in sequence as opposed to Masked Models
#predicting probability of word (token) given surrounding words
model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
    model_id,
    use_auth_token=True,
    dtype=torch.float32, #For CPU native arch
    device_map=device
)

df: pd.DataFrame = pd.read_csv("arxiv_dataset.csv")
# Filter rows to only get dataframe of VALID_CLASSES
df: pd.DataFrame = df[df["Category Description"].isin(VALID_CLASSES)]
random_seed=35
# print(df)

# #Shuffle the DataFrame with sett. Splits train/test 
df: pd.DataFrame = df.sample(frac=1, random_state=random_seed).reset_index(drop=True).reset_index()
train_size = 0.8
train_len = int(train_size * len(df))

df_train: pd.DataFrame = df[:train_len]
df_test: pd.DataFrame = df[train_len:]

# use build_prompts to add a "prompt" column before Dataset creation
df_test: pd.DataFrame = build_prompts(df_test)
df_test.to_csv("test.csv", index=False) #save as CSV wtihout row numbers

test_dataset: Dataset = Dataset.from_pandas(df_test)
test_dataloader: DataLoader = DataLoader(
    test_dataset, 
    batch_size=16, 
    shuffle=False,
    collate_fn=lambda x: {k: [d[k] for d in x] for k in x[0]}
)

metrics: dict[str, float] = test_model(test_dataloader, model, tokenizer)
print("\n=== METRICS ===")
print("\n".join([f"{k} = {v}" for k,v in metrics.items()]))


row: pd.Series = df_test.iloc[0]

print("\n=== SINGLE PAPER TEST ===")
print("GROUND TRUTH:", row["Category Description"])
print("\nPROMPT:\n", get_arxiv_paper_info(row))

prediction = generate_output_from_input(model, tokenizer, row["title"], row["summary"])
print("\nMODEL PREDICTION:", prediction)

#####################################################################3

# # ------------------------
# # 1. Load base model + LoRA
# # ------------------------
# model_name = "distilgpt2"

# # Tokenize input prompt-> numerical IDs
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token  # all sequences in batch have same length
# base_model = AutoModelForCausalLM.from_pretrained(model_name)
# model = PeftModel.from_pretrained(base_model, "./output")  # load your fine-tuned LoRA weights
# model.eval()
# device = torch.device("cpu")
# model.to(device)

# # ------------------------
# # 2. Questions to test
# # ------------------------
# questions = [
#     "Who won the fight between Mike Tyson and Evander Holyfield in 1996?",
#     "List three notable fights of Muhammad Ali.",
#     "Who is considered the GOAT of boxing?",
#     "Summarize Canelo Alvarez's professional boxing achievements."
# ]

# # ------------------------
# # 3. Inference loop
# # ------------------------
# for q in questions:
#     prompt = f"Q: {q} A:"  # trailing "A:" cues model to answer
#     inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)    
#     outputs = model.generate(
#         inputs.input_ids,
#         attention_mask=inputs.attention_mask,
#         max_new_tokens=50, #max tokens to generate 
#         eos_token_id=tokenizer.eos_token_id,
#         do_sample=True,
#         temperature=0.7, #randomness 
#         top_p=0.9, #nucleus sampling, focus on top probability mass 
#         repetition_penalty=1.2   # avoid repeated tokens
#     )


#     # decode output -> human readable text 
#     answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print(answer)
#     print("--------------------------------------------------")
