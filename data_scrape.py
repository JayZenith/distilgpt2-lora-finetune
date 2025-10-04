from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
import torch
from datasets import load_dataset
import arxiv
import pandas as pd 
from typing import Dict, List 

# Conver arxiv category codes (cs.AI) to human readable desrc
category_mapping: Dict[str,str] = {
    "cs.AI": "Artificial Intelligence",
    "cs.CL": "Computation and Language",
    "cs.CV": "Computer Vision",
    "cs.LG": "Machine Learning"
}

# Initialize the client
client = arxiv.Client(
    page_size = 5000, #num of results fetched per req
    delay_seconds = 3, #Wait 3 seconds between rewqs to avoid rate limits
    num_retries = 5 #Retry up to 5 times if req fails (network issue, API throttling (rate-limit), etc.)
)

csv_filename : str = "arxiv_dataset.csv"

# Define the search (what papers to fetch)
search = arxiv.Search(
    query="cat:cs.*", #fetch all cs.* papers
    sort_by=arxiv.SortCriterion.SubmittedDate, #newest papers first
    max_results=1000 # Max papers to retrieve
)

# List storage to hold paper's metadata as dictionary 
papers_data: List[Dict[str, str]] = []

# Fetch Results 
try:
    # client.results(search) returns iterator over papers
    for result in client.results(search): 
        paper_info: Dict[str, str] = {
            "title": result.title,
            "Category": result.primary_category, #(cs.AI, etc.)
            "Category Description": category_mapping.get(result.primary_category, "Unknown Category"),
            "Published": result.published.strftime("%Y-%m-%d"), #Format date YYYY-MM-DD
            "summary": result.summary,
        }
        papers_data.append(paper_info) #Append dict to papers_data 
        if len(papers_data) % 50 == 0: #Every 50 papers, print list to see progress 
            print(papers_data)
except Exception as e: #Catch exception (network issues, API errors) and prints without crashing
    print(e)
finally:
    df = pd.DataFrame(papers_data) #Convert list of dicts into tabular format 
    df.to_csv(csv_filename, index=False) #save as CSV wtihout row numbers
    print(f"Saved {len(df)} papers in {csv_filename}") #Print how many papers saved 
            

