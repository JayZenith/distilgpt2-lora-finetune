import re 
import pandas as pd
from datasets import Dataset 
import torch

VALID_CLASSES = ["Artificial Intelligence", "Computer Vision", "Systems", "Theory"]


def extract_label_from_output(outputs, tokenizer=None):
    """
    Given raw model outputs (list of strings), extract one of the VALID_CLASSES.
    """
    predictions = []
    for out in outputs:
        found = None
        for cls in VALID_CLASSES:
            if cls.lower() in out.lower():
                found = cls
                break
        if found is None:
            found = "INVALID"
        predictions.append(found)
    return predictions


def build_prompts(df: pd.DataFrame):
    system_prompt = (
        "You are an AI system that reads the title and summary of a paper and "
        "classifies it into the correct computer science category.\n"
        "You must return the *Category Description* and explain briefly why.\n\n"
        "Valid categories:\n" +
        "\n".join([f"- {c}" for c in VALID_CLASSES]) +
        "\n\n"
    )

    prompts: list[str] = []
    for _, row in df.iterrows():
        user_prompt = f"Title: {row['title']}\nSummary: {row['summary']}\n\nAnswer:"
        prompts.append(system_prompt + user_prompt)

    df = df.copy()
    df["prompt"] = prompts #Combo of system instructions and paper title + summary
    df["labels"] = df["Category Description"] # used as ground truth when testing
    return df




def get_message_prompt_tokenized(tokenizer, titles: list[str], abstracts: list[str]):
    """
    Builds the system + user message for each sample and tokenizes it.
    Returns a dictionary suitable for model.generate().
    """
    system_prompt = (
        "You are an AI system that reads the title and summary of a paper and "
        "classifies it into the correct computer science category.\n"
        "You must return the *Category Description* and explain briefly why.\n\n"
        "Valid categories:\n" +
        "\n".join([f"- {c}" for c in VALID_CLASSES]) +
        "\n\n"
    )

    prompts = []
    for title, abstract in zip(titles, abstracts):
        user_prompt = f"Title: {title}\nSummary: {abstract}\n\nAnswer:"
        prompts.append(system_prompt + user_prompt)

    tokenized: dict[str, torch.Tensor] = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    return tokenized




def get_arxiv_paper_info(row) -> str:
    """
    Nicely format a row from the dataframe for debugging/demo purposes.
    """
    return f"Title: {row['title']}\nSummary: {row['summary']}\nLabel: {row['Category Description']}"