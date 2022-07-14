from pathlib import Path

hatexplain_dir = Path("datasets/hatexplain")
dataset_labels = [
    "African", "Islam", "Jewish", "Homosexual", "Women", "Refugee", "Arab", "Caucasian", "Asian", "Hispanic"
]

hatexplain_json_path = hatexplain_dir/"dataset.json"                    # source data
raw_hatexplain_path = hatexplain_dir/"raw_dataset.csv"                  # converted from json into normal csv
processed_hatexplain_path = hatexplain_dir/"preprocessed_dataset.csv"   # after preprocessing
hatexplain_dataset_path = hatexplain_dir/"dataset.parquet"                  # ready for use in models, w/tokenization
