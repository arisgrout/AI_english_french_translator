#!/usr/bin/python3

# Adds this file to top-level interpretter path, so runs "as if" from top.
# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ENABLE & TEST GPU
## Turn GPU on in Colab
## Test connection to TensorFlow

import tensorflow as tf
import re
import collections
import numpy as np
import pandas as pd
from modules.functions import *
import modules.project_tests as tests
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

device_name = tf.test.gpu_device_name()
if device_name != "/device:GPU:0":
    raise SystemError("GPU device not found")

print("Found GPU at: {}".format(device_name))

"""
# SPEED TEST GPU "WARM-UP"
%tensorflow_version 2.x
import tensorflow as tf
import timeit

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print(
      '\n\nThis error most likely means that this notebook is not '
      'configured to use a GPU.  Change this in Notebook Settings via the '
      'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
  raise SystemError('GPU device not found')

def cpu():
  with tf.device('/cpu:0'):
    random_image_cpu = tf.random.normal((100, 100, 100, 3))
    net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)
    return tf.math.reduce_sum(net_cpu)

def gpu():
  with tf.device('/device:GPU:0'):
    random_image_gpu = tf.random.normal((100, 100, 100, 3))
    net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
    return tf.math.reduce_sum(net_gpu)
  
# We run each op once to warm up; see: https://stackoverflow.com/a/45067900
cpu()
gpu()

# Run the op several times.
print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
      '(batch x height x width x channel). Sum of ten runs.')
print('CPU (s):')
cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
print(cpu_time)
print('GPU (s):')
gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
print(gpu_time)
print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))

# recommended reimport tactic for module (sometimes doesn't work)
#import importlib
#importlib.reload(modules.functions)
# special iPython setting that reloads modules EVERYTIME a function is run from them.
# https://switowski.com/blog/ipython-autoreload
#%load_ext autoreload
#%autoreload 2
"""

en = pd.Series(load_data("./data/europarl-v7.fr-en.en"))
fr = pd.Series(load_data("./data/europarl-v7.fr-en.fr"))
df = pd.DataFrame({"en": en, "fr": fr})

len(en) == len(fr)
len(en)


def tokenize(x):
    x_tk = Tokenizer(char_level=False)
    x_tk.fit_on_texts(x)
    return x_tk.texts_to_sequences(x), x_tk


def pad(x, length=None):
    if length is None:
        length = max([len(sentence) for sentence in x])
    return pad_sequences(x, maxlen=length, padding="post")


def preprocess(x, y):
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
    return preprocess_x, preprocess_y, x_tk, y_tk


en_pre, fr_pre, en_tk, fr_tk = preprocess(df.en, df.fr)

max_english_sequence_length = en_pre.shape[1]
max_french_sequence_length = fr_pre.shape[1]
english_vocab_size = len(en_tk.word_index)
french_vocab_size = len(fr_tk.word_index)


# SIMPLE
def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = ""
    return " ".join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


def simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    learning_rate = 1e-3
    input_seq = Input(input_shape[1:])
    rnn = GRU(64, return_sequences=True)(input_seq)
    logits = TimeDistributed(Dense(french_vocab_size))(rnn)
    model = Model(input_seq, Activation("softmax")(logits))
    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate), metrics=["accuracy"])
    return model


tests.test_simple_model(simple_model)
tmp_x = pad(en_pre, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, fr_pre.shape[-2], 1))
# Train the neural network
simple_rnn_model = simple_model(tmp_x.shape, max_french_sequence_length, english_vocab_size, french_vocab_size)

with tf.device("/device:GPU:0"):
    simple_rnn_model.fit(tmp_x, fr_pre, batch_size=1024, epochs=10, validation_split=0.2)
    # Print prediction(s)
    print(logits_to_text(simple_rnn_model.predict(tmp_x[:1])[0], fr_tk))


# EMBEDDING
from keras.models import Sequential


def embed_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    learning_rate = 1e-3
    rnn = GRU(64, return_sequences=True, activation="tanh")

    embedding = Embedding(french_vocab_size, 64, input_length=input_shape[1])
    logits = TimeDistributed(Dense(french_vocab_size, activation="softmax"))

    model = Sequential()
    # em can only be used in first layer --> Keras Documentation
    model.add(embedding)
    model.add(rnn)
    model.add(logits)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate), metrics=["accuracy"])

    return model


tests.test_embed_model(embed_model)
tmp_x = pad(en_pre, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, fr_pre.shape[-2]))
embeded_model = embed_model(tmp_x.shape, max_french_sequence_length, english_vocab_size, french_vocab_size)

with tf.device("/device:GPU:0"):
    embeded_model.fit(tmp_x, fr_pre, batch_size=1024, epochs=10, validation_split=0.2)
    print(logits_to_text(embeded_model.predict(tmp_x[:1])[0], fr_tk))


# BIDIRECTIONAL
def bd_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):

    learning_rate = 1e-3
    model = Sequential()
    model.add(Bidirectional(GRU(128, return_sequences=True, dropout=0.1), input_shape=input_shape[1:]))
    model.add(TimeDistributed(Dense(french_vocab_size, activation="softmax")))
    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate), metrics=["accuracy"])
    return model


tests.test_bd_model(bd_model)
tmp_x = pad(en_pre, fr_pre.shape[1])
tmp_x = tmp_x.reshape((-1, fr_pre.shape[-2], 1))
bidi_model = bd_model(tmp_x.shape, fr_pre.shape[1], len(en_tk.word_index) + 1, len(fr_tk.word_index) + 1)

with tf.device("/device:GPU:0"):
    bidi_model.fit(tmp_x, fr_pre, batch_size=1024, epochs=20, validation_split=0.2)
    # Print prediction(s)
    print(logits_to_text(bidi_model.predict(tmp_x[:1])[0], fr_tk))


# ENCODER-DECODER
def encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):

    learning_rate = 1e-3
    model = Sequential()
    model.add(GRU(128, input_shape=input_shape[1:], return_sequences=False))
    model.add(RepeatVector(output_sequence_length))
    model.add(GRU(128, return_sequences=True))
    model.add(TimeDistributed(Dense(french_vocab_size, activation="softmax")))

    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate), metrics=["accuracy"])
    return model


tests.test_encdec_model(encdec_model)
tmp_x = pad(en_pre)
tmp_x = tmp_x.reshape((-1, en_pre.shape[1], 1))
encodeco_model = encdec_model(tmp_x.shape, fr_pre.shape[1], len(en_tk.word_index) + 1, len(fr_tk.word_index) + 1)

with tf.device("/device:GPU:0"):
    encodeco_model.fit(tmp_x, fr_pre, batch_size=1024, epochs=20, validation_split=0.2)
    print(logits_to_text(encodeco_model.predict(tmp_x[:1])[0], fr_tk))


## CUSTOM
def model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):

    model = Sequential()
    model.add(Embedding(input_dim=english_vocab_size, output_dim=128, input_length=input_shape[1]))
    model.add(Bidirectional(GRU(256, return_sequences=False)))
    model.add(RepeatVector(output_sequence_length))
    model.add(Bidirectional(GRU(256, return_sequences=True)))
    model.add(TimeDistributed(Dense(french_vocab_size, activation="softmax")))
    learning_rate = 0.005

    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate), metrics=["accuracy"])

    return model


tests.test_model_final(model_final)
print("Final Model Loaded")

def final_predictions(x, y, x_tk, y_tk):
    tmp_X = pad(en_pre)
    model = model_final(tmp_X.shape, fr_pre.shape[1], len(en_tk.word_index) + 1, len(fr_tk.word_index) + 1)

    with tf.device("/device:GPU:0"):
        model.fit(tmp_X, fr_pre, batch_size=1024, epochs=17, validation_split=0.2)

    y_id_to_word = {value: key for key, value in y_tk.word_index.items()}
    y_id_to_word[0] = ""

    sentence = "he saw a old yellow truck"
    sentence = [x_tk.word_index[word] for word in sentence.split()]
    sentence = pad_sequences([sentence], maxlen=x.shape[-1], padding="post")
    sentences = np.array([sentence[0], x[0]])
    predictions = model.predict(sentences, len(sentences))

    print("Sample 1:")
    print(" ".join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))
    print("Il a vu un vieux camion jaune")
    print("Sample 2:")
    print(" ".join([y_id_to_word[np.argmax(x)] for x in predictions[1]]))
    print(" ".join([y_id_to_word[np.max(x)] for x in y[0]]))

    return model


final_model = final_predictions(en_pre, fr_pre, en_tk, fr_tk)

# SAVE MODELS
all_models = [final_model, encodeco_model, bidi_model, embeded_model, simple_rnn_model]
model_names = ["final_model", "encodeco_model", "bidi_model", "embeded_model", "simple_rnn_model"]

for idx in range(len(all_models)):
    all_models[idx].save(f"./models/{model_names[idx]}_euro.h5")
