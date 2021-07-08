import os
import re
import collections
import numpy as np
import pandas as pd
import spacy
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from joblib import Parallel, delayed

# LOAD FILES


def load_data(path):
    """
    Load data sentence-wise
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data.split("\n")


# SPACY TEXT PRE-CLEANING


def cleaner(df, limit=0):
    "Extract relevant text from DataFrame using a regex"
    # Regex pattern for only alphanumeric, hyphenated text with 3 or more chars
    pattern = re.compile(r"[A-Za-z0-9\-]{3,50}")
    df["en_clean"] = df["en"].str.findall(pattern).str.join(" ")
    df["fr_clean"] = df["fr"].str.findall(pattern).str.join(" ")
    if limit > 0:
        return df.iloc[:limit, :].copy()
    else:
        return df


# JOBLIB PROCESSING


def chunker(iterable, total_length, chunksize):
    return (iterable[pos : pos + chunksize] for pos in range(0, total_length, chunksize))


def flatten(list_of_lists):
    "Flatten a list of lists to a combined list"
    return [item for sublist in list_of_lists for item in sublist]


def process_chunk(processor, col):
    preproc_pipe = []
    for doc in processor.pipe(col, batch_size=20):
        preproc_pipe.append(lemmatize_pipe(doc))
    return preproc_pipe


def preprocess_parallel(df, col, processor, chunksize=100):
    executor = Parallel(n_jobs=7, backend="multiprocessing", prefer="processes")
    do = delayed(process_chunk(processor, df[col]))
    tasks = (do(chunk) for chunk in chunker(df[col], len(df), chunksize=chunksize))
    result = executor(tasks)
    return flatten(result)


def lemmatize(processor, stopwords, text):
    """Perform lemmatization and stopword removal in the clean text
    Returns a list of lemmas
    """
    doc = processor(text)
    lemma_list = [str(tok.lemma_).lower() for tok in doc if tok.is_alpha and tok.text.lower() not in stopwords]
    return lemma_list
