# English to French Translator
## Objective
Build a text-based NLP model, for translating english to french.

## Approach
1. Find corpus of text in both English and French.
2. Clean and tokenize textual data.
3. Test different Deep Learning approaches (RNN, RNN with embedding, bidirectional RNN, GRU-LSTM Seq2Seq).
4. Design a DL model that can accurately translate English to French.

## TODO
5. Test against Transformer models (BERT?)
6. Summarize conclusions in this readme / create presentation of results
* Current best performing model combines embedding and bidirectional RNNs.

## Data
Textual data from publically available corpus: https://www.statmt.org/europarl/
* [French to English copy](https://www.statmt.org/europarl/v7/fr-en.tgz)

_Based on tutorial: https://towardsdatascience.com/neural-machine-translation-with-python-c2f0a34f7dd_
