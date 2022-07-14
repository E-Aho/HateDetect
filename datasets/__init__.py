import pandas as pd

from datasets.hatexplain import (hatexplain_json_path, hatexplain_dir,
                                 dataset_labels, processed_hatexplain_path,
                                 raw_hatexplain_path)


def get_raw_df() -> pd.DataFrame:
    return pd.read_csv(raw_hatexplain_path)


def get_hateplain_df() -> pd.DataFrame:
    return pd.read_csv(processed_hatexplain_path)

