import pandas as pd
import numpy as np
import json

from sentence_transformers import SentenceTransformer
from constants import TRAIN_PATH, TITLE_TO_VEC_FILE


def preprocess(entry):
    """
        Convert a sentence into lowercas and remove unnecessary
        trailing and leading spaces.
    """
    return str(entry).lower().strip()


def main():
    """
        Main function to generate Title to dict files.
    """
    train_file = TRAIN_PATH
    train_df = pd.read_csv(train_file)
    train_df = train_df.drop("listing_id", axis=1)
    train_df.title = train_df.title.apply(preprocess)
    group_by_title = train_df.groupby("title")
    title_to_data_dict = {}

    for key in group_by_title.groups.keys():
        for index in group_by_title.groups.get(key):
            tmp_list = []
            for colm_value in list(train_df.iloc[index].items()):
                tmp_list.append("{} is {} .".format(colm_value[0], colm_value[1]))

            if key in title_to_data_dict:
                title_to_data_dict[key].append(" ".join(tmp_list))
            else:
                title_to_data_dict[key] = [" ".join(tmp_list)]

    with open(TITLE_TO_VEC_FILE, "w") as out_f:
        out_f.write(json.dumps(title_to_data_dict))

    encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    title_to_mean_dict = {}

    for key in title_to_data_dict.keys():
        mean_embd = np.mean(encoder.encode(title_to_data_dict[key]), axis=0)
        title_to_mean_dict[key] = mean_embd


    index_to_title_dict = {}
    title_embd_list = []

    index = 0
    for key in title_to_mean_dict.keys():
        title_embd_list.append(title_to_mean_dict[key])
        index_to_title_dict[index] = key
        index += 1

    np.save(TITLE_TO_VEC_FILE, {"title_embd_array": np.array(title_embd_list), "index_to_title_dict": index_to_title_dict})
    
    
if __name__ == "__main__":
    main()
