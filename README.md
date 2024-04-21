# n-gram-lang-models

This project ngram.py builds an n-gram language model (with variations). 

The user can select model type to be unigram or bigram, select whether to preprocess corpus, and select smoothing type to be none, Laplace smoothing, or add-k smoothing (can select value of k for this option). 

Words occurring less than 3 times are replaced by UNK in the training set. 

The n-gram probability of each unigram or bigram occurrence in the training set is saved to a json file.

For unigram or smoothed bigram models, the training and validation perplexity is calculated. If a new word is encountered at test time, it is treated as UNK.

See NLP_A1_Report.pdf for more details on implementation and discussion on results.
