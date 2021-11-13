from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import cleaner
import constants as const

# Types
Sentence = str
SentenceDict = Dict[str, List[Sentence]]
TitleEmbeddings = np.ndarray


def preprocess_title(title: str) -> str:
    """Preprocessor function for titles that lowercases and strips them"""
    return str(title).lower().strip()


def get_description_sentence(row: pd.Series) -> str:
    """Return the description sentence given by a row"""
    mappings = [str(row[col]) for col in const.DESCRIPTION_COLS
                if row[col] and row[col] != const.UNKNOWN_STR]
    return ', '.join(mappings)


def get_full_sentence(row: pd.Series) -> str:
    """Return the full sentence given by a row"""
    mappings = [f'{col} is {val}' for col, val in row.items()]
    return ', '.join(mappings)


def get_sentence_dict(df: pd.DataFrame) -> SentenceDict:
    """
    Given a dataframe and a sentence function, return a dictionary mapping
    titles to a list of sentences.
    """
    return {
        title: df.iloc[indices].apply(get_description_sentence, axis=1).tolist()
        for title, indices in df.groupby('title').groups.items()
    }


def get_mean_embeddings(sentence_dict: SentenceDict,
                        encoder: SentenceTransformer) \
                        -> Tuple[TitleEmbeddings, Dict[int, str]]:
    """
    Given a dictionary of sentences and an encoder, return a tuple consisting
    of a numpy array of mean embeddings and a dictionary mapping indices to
    titles.
    """
    # Create title to mean embedding of all instances with same title
    title_to_mean_dict = {}
    for title, sentences in tqdm(sentence_dict.items()):
        embeddings = np.mean(encoder.encode(sentences), axis=0)
        title_to_mean_dict[title] = embeddings

    # Save embeddings in fixed order and with an index mapping
    title_embeddings, index_to_title_dict = [], {}
    for idx, (title, embedding) in enumerate(title_to_mean_dict.items()):
        title_embeddings.append(embedding)
        index_to_title_dict[idx] = title

    return np.array(title_embeddings), index_to_title_dict


def main():
    """
    Load the data, clean it, lowercase titles and generate the embeddings for
    both types of sentences.
    """
    print('Loading encoder into memory...')
    encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    print('Reading data...')
    df_train_raw = pd.read_csv(const.TRAIN_PATH)
    df_train = cleaner.clean_preliminary(df_train_raw)
    df_train.drop(columns=['listing_id', 'index'], axis=1, inplace=True)
    df_train.title = df_train.title.apply(preprocess_title)

    # Description sentences (used for task 3)
    print('Generating title sentences...')
    sentence_dict = get_sentence_dict(df_train)
    title_embeddings, index_to_title_dict \
        = get_mean_embeddings(sentence_dict, encoder)
    np.save(const.TITLE_EMBEDDING_DICT_PATH, {
        'title_embeddings': title_embeddings,
        'index_to_title_dict': index_to_title_dict
    })

    # Full sentences (used for task 2)
    print('Generating full sentences...')
    full_sentences = df_train.apply(get_full_sentence, axis=1).tolist()
    full_embeddings = encoder.encode(full_sentences)
    np.save(const.CAR_EMBEDDING_MATRIX_PATH, full_embeddings)

    print('Done!')


if __name__ == '__main__':
    main()
