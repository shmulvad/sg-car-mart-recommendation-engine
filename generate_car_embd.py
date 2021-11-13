import pandas as pd
import numpy as np
import json

from sentence_transformers import SentenceTransformer
from constants import TRAIN_PATH, CAR_EMBEDDING_MATRIX_PATH


def main():

    train_file = TRAIN_PATH
    train_df = pd.read_csv(train_file)

    encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    series_list = []
    inst_list = []

    for index in range(len(train_df)):
        series_list.append(train_df.iloc[index])

        tmp_list = []
        for key in series_list[-1].keys():
            tmp_list.append("{} is {}.".format(key, series_list[-1][key]))

        inst_list.append(" ".join(tmp_list))

    emb_list = encoder.encode(inst_list)

    np.save(CAR_EMBEDDING_MATRIX_PATH, {"series_list": series_list, "vec_list": emb_list})
    
    
if __main__ == "__name__":
    main()
