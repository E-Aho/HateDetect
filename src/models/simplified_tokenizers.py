import pandas as pd

from transformers import (
    RobertaTokenizer,
)


def tokenize_hatexplain(dataset: pd.DataFrame, tokenizer: str = "roberta-base"):
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer)
    all_strs = dataset["text"].tolist()
    tokenized_strs = tokenizer(all_strs, truncation=False, padding=True)
    dataset["input_ids"] = tokenized_strs["input_ids"]
    dataset["attention_mask"] = tokenized_strs["attention_mask"]
    return dataset


if __name__ == "__main__":
    from datasets.hatexplain import hatexplain_dataset_path
    dataset = pd.read_parquet(hatexplain_dataset_path)



