# utils.py
import re
import os
from datasets import load_dataset, load_from_disk

def clean_text(text):
    text = re.sub(r"[^\w\s.,!?'-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_sentiment_dataset(dataset_name, split):
    local_dataset_path = f"./{dataset_name}_test".replace("/", "_")
    if os.path.exists(local_dataset_path):
        dataset = load_from_disk(local_dataset_path)
    else:
        if(dataset_name=="tweets_hate_speech_detection"):
            dataset = load_dataset("tweets_hate_speech_detection", split="train")
        elif(dataset_name=="sst2"):
            dataset = load_dataset("glue", "sst2", split="validation")
        else:
            dataset = load_dataset(dataset_name, split=split)
        if(dataset_name=="amazon_polarity"):
            dataset = dataset.rename_column("content", "text")
        if(dataset_name=="tweets_hate_speech_detection"):
            dataset = dataset.rename_column("tweet", "text")
        if(dataset_name=="sst2"):
            dataset = dataset.rename_column("sentence", "text")
        dataset.save_to_disk(local_dataset_path)
    return dataset