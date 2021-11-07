import ast
import json
import re
import string

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from transformers import (BertForSequenceClassification, BertTokenizer,
                          Trainer, TrainingArguments)


def fun_feature(str_1, pattern):
    str_1 = str_1.lower()
    indx = str_1.find("view specs")

    if indx >= 0:
        str_1 = str_1[:indx]

    if len(str_1) > 1:
        str_1 = pattern.sub(",", str_1)
        str_1 = str_1.strip()

        return [token.strip(string.punctuation+" \n\t") for token in str_1.split(",") if len(token.strip()) > 2]
    else:
        return []


def fun_accessory(str_1, pattern):
    str_1 = str_1.lower()

    if len(str_1) > 1:
        str_1 = pattern.sub(",", str_1)
        str_1 = str_1.strip()

        return [token.strip(string.punctuation+" \n\t") for token in str_1.split(",") if len(token.strip()) > 2]
    else:
        return []


def make_clusters():
    """Clustering of features"""
    input_file = "task1/data/train_1.csv"
    train_df = pd.read_csv(input_file)

    features_series = train_df.features
    features_series.dropna(inplace=True)

    accessories_series = train_df.accessories
    accessories_series.dropna(inplace=True)

    pattern = re.compile(r"\. | \.|! ")

    set_2 = set()

    for accessory in accessories_series.values:
        tmp_list = fun(str(accessory), pattern)
        for item in tmp_list:
            set_2.add(item)

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    pool_list = []
    match_dict = {}
    encoding_dict = {}

    for text in list(set_1):
        text_encoding = model.encode([text])
        similar = False

        max_score = None
        max_loc = None

        for loc in range(len(pool_list)):
            sim_score = util.pytorch_cos_sim(
                encoding_dict[loc], text_encoding).numpy()[0][0]

            if sim_score > 0.70:
                similar = True

                if max_score == None or sim_score > max_score:
                    max_score = sim_score
                    max_loc = loc

        if not similar:
            pool_list.append(text)
            encoding_dict[len(pool_list)-1] = text_encoding
            match_dict[len(pool_list)-1] = []
        else:
            match_dict[max_loc].append((text, max_score))

    set_1 = set()

    for feature in features_series.values:
        tmp_list = fun(str(feature), pattern)
        for item in tmp_list:
            set_1.add(item)

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    pool_list = []
    match_dict = {}
    encoding_dict = {}

    for text in list(set_1):
        text_encoding = model.encode([text])
        similar = False

        max_score = None
        max_loc = None

        for loc in range(len(pool_list)):
            sim_score = util.pytorch_cos_sim(
                encoding_dict[loc], text_encoding).numpy()[0][0]

            if sim_score > 0.70:
                similar = True

                if max_score == None or sim_score > max_score:
                    max_score = sim_score
                    max_loc = loc

        if not similar:
            pool_list.append(text)
            encoding_dict[len(pool_list)-1] = text_encoding
            match_dict[len(pool_list)-1] = []
        else:
            match_dict[max_loc].append((text, max_score))

    with open("task1/data/feature_simi_final.jsonl", "w") as out_f:
        for indx in range(len(pool_list)):
            list_ = match_dict[indx]
            sorted_list = sorted(list_, key=lambda x: x[1], reverse=True)

            if len(sorted_list) > 0:
                tmp_dict = {"{}".format(pool_list[indx]): []}
                for i in range(int(len(sorted_list)/2)):
                    tmp_dict[pool_list[indx]].append(sorted_list[i])

                out_f.write(str(tmp_dict)+"\n")



def extract_features():
    """Extract features from clusters"""
    input_file = "task1/data/feature_simi_final.jsonl"
    feature_file = "task1/data/feature.jsonl"
    outlier_file = "task1/data/outlier.jsonl"

    counter = 0
    feature_instances_list = []
    global_dict = {}

    with open(feature_file, "w") as feat_f:
        with open(outlier_file, "w") as outlier_f:
            with open(input_file, "r") as in_f:
                for line in in_f:
                    dict_obj = ast.literal_eval(line)

                    for key in dict_obj.keys():
                        if len(dict_obj[key]) > 0:
                            tmp_dict = {"feature_{}".format(counter): (
                                key, dict_obj[key][-1][1], len(dict_obj[key]))}
                            global_dict["feature_{}".format(counter)] = (
                                key, dict_obj[key][-1][1], len(dict_obj[key]))
                            feature_instances_list.append(len(dict_obj[key]))
                            feat_f.write(json.dumps(tmp_dict)+"\n")
                            counter += 1
                        else:
                            outlier_f.write(line)

    sorted_dict = sorted(global_dict.items(), key=lambda x: x[1][2], reverse=True)
    with open("task1/data/sorted_feature.jsonl", "w") as out_f:
        for key, val in sorted_dict:
            out_f.write(json.dumps({key: val})+"\n")


def assign_feature_to_train_dataset():
    test_df = pd.read_csv("task1/data/test.csv")

    features_series = test_df.features  # train_df.features
    accessories_series = test_df.accessories  # train_df.accessories

    input_file = "task1/data/sorted_feature.jsonl"

    feature_dict = {}
    with open(input_file, "r") as in_f:
        for line in in_f:
            dict_obj = json.loads(line)

            for key in dict_obj.keys():
                feature_dict[key] = dict_obj[key]

    data_dict = {}
    for key in feature_dict.keys():
        data_dict[key] = [0]*len_

    pattern = re.compile("\. | \.|! ")

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    feature_phrase_encoding_dict = {}

    for key in feature_dict.keys():
        feature_phrase_encoding_dict[key] = model.encode(
            [feature_dict[key][0]])

    for index in range(len_):
        acc_list = []
        if not pd.isnull(accessories_series[index]):
            acc_list = fun_accessory(str(accessories_series[index]), pattern)

        feat_list = []
        if not pd.isnull(features_series[index]):
            feat_list = fun_feature(str(features_series[index]), pattern)

        feat_list.extend(acc_list)

        extracted_feature_list = get_feature_list(feat_list)

        for feat in extracted_feature_list:
            data_dict[feat][index] = 1

    data_df = pd.DataFrame(data_dict)

    output_df = pd.concat([test_df, data_df], axis=1)
    output_df.to_csv("task1/data/test_2.tsv", sep="\t", index=False)


def get_feature_list(token_list):
    feature_set = set()

    for token in token_list:
        token_encoding = model.encode([token])

        max_key = None
        max_score = None

        for key in feature_dict.keys():
            sim_score = util.pytorch_cos_sim(
                feature_phrase_encoding_dict[key], token_encoding).numpy()[0][0]

            if (sim_score >= feature_dict[key][1]) and (max_score == None or sim_score > max_score):
                max_score = sim_score
                max_key = key

        if max_key != None:
            feature_set.add(max_key)

    return list(feature_set)


def scale_price_column():
    """Scaling price column"""
    input_file = "task1/data/train_2.tsv"
    train_df = pd.read_csv(input_file, sep="\t")

    min_max_scaler = MinMaxScaler()
    price_series = train_df.price
    print(len(price_series))

    price_series.dropna(inplace=True)
    print(len(price_series))
    price_series = price_series.values.reshape(-1, 1)
    scaled_price = min_max_scaler.fit_transform(price_series)
    squeezed_price = np.squeeze(scaled_price)

    output_df = pd.concat([train_df, pd.Series(
        squeezed_price, name="scaled_description").to_frame()], axis=1)
    output_df.to_csv("task1/data/train_3.tsv", sep="\t", index=False)


class DescriptionData(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index])
                for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item

    def __len__(self):
        return len(self.labels)


def regression_bert():
    input_file = "task1/data/train_3.tsv"
    train_df = pd.read_csv(input_file, sep="\t")

    description_series = train_df.description
    scaled_description_series = train_df.scaled_description

    description_list = description_series.astype(str).to_list()
    scaled_description_list = scaled_description_series.astype(float).to_list()

    train_data, val_data, train_label, val_label = train_test_split(
        description_list, scaled_description_list, test_size=0.1)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_encodings = tokenizer(train_data, truncation=True, padding=True)
    val_encodings = tokenizer(val_data, truncation=True, padding=True)

    train_dataset = DescriptionData(train_encodings, train_label)
    val_dataset = DescriptionData(val_encodings, val_label)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=1)
    output_dir = "task1/output/"
    logging_dir = "task1/logs/"

    training_args = TrainingArguments(output_dir=output_dir, num_train_epochs=2, warmup_steps=50, weight_decay=0.01,
                                      logging_dir=logging_dir, do_train=True, do_eval=True, evaluation_strategy="epoch",
                                      learning_rate=0.001, logging_strategy="epoch", save_strategy="epoch", save_total_limit=1)
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=train_dataset, eval_dataset=val_dataset)

    trainer.train()

    trainer.save_model(output_dir)

    test_file = "task1/data/test_2.tsv"
    test_df = pd.read_csv(test_file, sep="\t")

    test_description_list = test_df.description.astype(str).to_list()
    test_description_label_list = [0.0]*len(test_description_list)

    test_encodings = tokenizer(
        test_description_list, padding=True, truncation=True)

    test_dataset = DescriptionData(test_encodings, test_description_label_list)

    model = BertForSequenceClassification.from_pretrained(
        "task1/output_finetuned_on_description_column", num_labels=1)

    output_dir = "task1/output_finetuned_on_description_column"
    testing_args = TrainingArguments(output_dir=output_dir,  do_train=False, do_eval=False,
                                     do_predict=True)
    trainer = Trainer(model=model, args=testing_args)

    test_output = trainer.predict(test_dataset)

    min_max_scaler = MinMaxScaler()
    price_series = train_df.price
    price_series = price_series.values.reshape(-1, 1)

    price_output = min_max_scaler.inverse_transform(test_output.predictions)

    pd.DataFrame({"Id": id_list, "Predicted": price_output.flatten().tolist()}).to_csv(
        "task1/submission.csv", index=False)

    output_df = pd.concat([test_df, pd.Series(np.squeeze(
        test_output.predictions), name="scaled_description").to_frame()], axis=1)
    output_df.to_csv("task1/data/test_3.tsv", sep="\t", index=False)
