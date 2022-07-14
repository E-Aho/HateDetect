import json
from pathlib import Path
import requests

from datasets.hatexplain import hatexplain_json_path

hatexplain_dataset_url = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json"


def init_hatexplain_dataset(path: Path = hatexplain_json_path, url: str = hatexplain_dataset_url):
    path.parent.mkdir(parents=True, exist_ok=True)
    req = requests.get(url)

    with open(path, "w+") as file:
        json_str = json.dumps(req.json())
        file.write(json_str)


def init_datasets(hatexplain_path: Path, overwrite_existing: bool = True):

    if overwrite_existing or not hatexplain_path.is_file():
        init_hatexplain_dataset(hatexplain_path)


if __name__ == "__main__":
    init_datasets(hatexplain_path=hatexplain_json_path)
