from collections import defaultdict

import numpy as np
import pandas as pd

from datasets.hatexplain import hatexplain_json_path, dataset_labels, raw_hatexplain_path, processed_hatexplain_path, \
    hatexplain_dataset_path

import multiprocessing as mp

from src.dataset_utils.string_preprocessing import check_is_english, preprocess_text
from src.models.simplified_tokenizers import tokenize_hatexplain


def init_raw_hatexplain_df() -> pd.DataFrame:
    raw_df = pd.read_json(hatexplain_json_path, orient="index")
    output_rows = []
    for row in raw_df.iterrows():
        annotated_targets = defaultdict(int)

        d_normal = 0
        d_offensive = 0
        d_hate = 0

        series = row[1]

        for annotator in series["annotators"]:
            if (label := annotator["label"]) == "normal":
                d_normal += 1
            elif label == "offensive":
                d_offensive += 1
            else:
                d_hate += 1

            for target in annotator["target"]:
                annotated_targets[target] += 1

        if d_normal == d_offensive == d_hate == 1:
            label = "split"
        else:
            label_dict = {
                "normal": d_normal,
                "offensive": d_offensive,
                "hate": d_hate
            }
            label = max(label_dict, key=label_dict.get)

        all_targets = list(annotated_targets.keys())
        majority_target = [target for target in annotated_targets.keys() if
                           annotated_targets[target] >= 2]  # 2 of 3 selected
        unanimous_target = [target for target in annotated_targets.keys() if
                            annotated_targets[target] == 3]  # all three

        binary_targets = {k: k in majority_target for k in dataset_labels}
        text = (" ".join(series["post_tokens"])).replace(" ' ", "'")

        row_result = {
            "post_id": series["post_id"],
            "label": label,
            "annotated_normal": d_normal,
            "annotated_offensive": d_offensive,
            "annotated_hate": d_hate,
            "targets": majority_target,
            "all_targets": all_targets,
            "unanimous_targets": unanimous_target,
            "tokens": series["post_tokens"],
            "text": text,
        }

        for target, present in binary_targets.items():
            row_result[f"at_{target}".lower()] = present

        output_rows.append(row_result)

    full_dataframe = pd.DataFrame(output_rows)

    return full_dataframe


def preprocess_hatexplain_df(df: pd.DataFrame):

    df["clean_tokens"] = df.apply(lambda x: preprocess_text(x["text"]), axis=1)
    df["clean_text"] = df.apply(lambda x: " ".join(x["clean_tokens"]).replace(" ' ", "'"), axis=1)

    return df

    #     for target, present in binary_targets.items():
    #         row_result[f"at_{target}".lower()] = present
    #
    #     output_rows.append(row_result)
    #
    # full_dataframe = pd.DataFrame(output_rows)
    #
    #
    # print("checking languages...")
    # with mp.Pool(mp.cpu_count()) as pool:
    #     full_dataframe["_eng"] = pool.map(check_is_english, full_dataframe["clean_text"])
    #
    # full_dataframe["is_english"], full_dataframe["p_lang"] = full_dataframe["_eng"].str
    # return full_dataframe


def remove_nonenglish_entries(df: pd.DataFrame):
    """Specific entries are removed, which have been identified by hand to not be English"""
    non_english_entry_ids = [
        "24348147_gab",
        "10167807_gab",
        "25494915_gab",
        "22644270_gab",
        "23631407_gab",
        "17071040_gab",
        "17711046_gab",
        "16944494_gab",
        "9247014_gab",
        "18582627_gab",
        "17426587_gab",
        "21729136_gab",
        "17683055_gab",
        "1078329_gab",
        "1246127173171384320_twitter",
        "27945436_gab",
        "1001155_gab",
        "25520976_gab",
        "1180860204193333249_twitter",
        "24414450_gab",
        "8046583_gab",
        "17664393_gab",
        "20025438_gab",
        "24549535_gab",
        "18030509_gab",
        "17743210_gab",
        "9078994_gab",
        "1265596729677799424_twitter",
        "15581917_gab",
        "1123914189116968964_twitter",
        "1168001573407997956_twitter",
        "24569622_gab",
    ]

    return df[~df.post_id.isin(non_english_entry_ids)]


def prepare_hatexplain_dataset(df: pd.DataFrame):
    """Selects only pertinent features and cleaned text, and encodes y labels one hot"""

    #TODO: drop the "split" column? or remove them entirely? it is just ones which are ambiguous...

    temp_df = df[[
        "post_id",
        "clean_text",
        "label",
    ]]

    one_hot_labels = pd.get_dummies(temp_df["label"])
    temp_df = temp_df.join(one_hot_labels)

    temp_df = temp_df.rename(columns={"clean_text": "text"})
    out_df = temp_df.drop("label", axis=1)
    return out_df


if __name__ == "__main__":
    raw_df = init_raw_hatexplain_df()
    raw_df.to_csv(raw_hatexplain_path)

    preprocessed_df = preprocess_hatexplain_df(raw_df)
    clean_df = remove_nonenglish_entries(preprocessed_df)
    clean_df.to_csv(processed_hatexplain_path)

    final_df = prepare_hatexplain_dataset(clean_df)
    dataset = tokenize_hatexplain(final_df)
    dataset.to_parquet(hatexplain_dataset_path)
