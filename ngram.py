# Nishad Raisinghani and Ishwari Joshi
# CS 6320 Assignment 1
# Overall Goal: To build an n-gram-based language model (with variations)

# import libraries
from json import dumps
from math import log, exp
import string
from nltk.stem import WordNetLemmatizer
from nltk import download 

# Ngram class can make a unigram or bigram language model
class Ngram:
    # initialize corpus, type of smoothing, k value (if add-k smoothing is used),
    # and type of model (unigram or bigram)
    def __init__(self, corpus, smoothing_type, k_value, n):
        self.corpus = corpus
        self.vocab_size = self.get_vocab_size()
        self.model = None
        self.n = n
        self.smoothing_type = smoothing_type
        self.k_value = k_value

    # vocabulary size is number of unique tokens in corpus
    def get_vocab_size(self) -> int:
        corpus_list = self.corpus.split()
        # unique_tokens is a dictionary with tokens as keys and None as values
        unique_tokens = {}
        for i in corpus_list:
            if i not in unique_tokens:
                # None is a simple placeholder to account for each unique token
                unique_tokens[i] = None
        vocab_size = len(unique_tokens.keys())
        print(f"Total number of tokens in corpus: {len(corpus_list)}")
        print(
            f"Number of unique tokens in corpus (vocabulary size): {vocab_size}")
        return vocab_size

    # save model probabilities to .json file.
    # will overwrite file on each run
    def save_model(self):
        f = open(f"{self.n}-gram_model.json", 'w')
        f.write(dumps(self.model))
        f.close()

    # calculate unigram or bigram probabilities from training corpus
    def calculate_ngram_probabilities(self, word1: str, word2: str = None) -> float:
        # unigram
        if self.n == 1:
            # unigram count
            count_word = self.corpus.count(word1)
            # total number of words in corpus
            N = len(self.corpus.split())
            # Laplace smoothing
            if self.smoothing_type == 2:
                return (count_word + 1)/(N + self.vocab_size)
            # add-k smoothing
            if self.smoothing_type == 3:
                return (count_word + self.k_value)/(N + self.k_value * self.vocab_size)
            # no smoothing
            return count_word/N
        # bigram
        else:
            # bigram count
            count_word1_word2 = self.corpus.count(f"{word2} {word1}")
            # previous word count.
            # in bigram model, every word is conditioned on the previous word
            count_word2 = self.corpus.count(word2)
            # Laplace smoothing
            if self.smoothing_type == 2:
                return (count_word1_word2 + 1)/(count_word2 + self.vocab_size)
            # add-k smoothing
            if self.smoothing_type == 3:
                return (count_word1_word2 + self.k_value)/(count_word2 + self.k_value * self.vocab_size)
            # no smoothing
            return count_word1_word2/count_word2

    # save model probabilities to .json file
    def get_ngram_probability_map(self) -> dict:
        # unigram
        if self.n == 1:
            # initialize dictionary
            d = {"UNK": self.calculate_ngram_probabilities("UNK")}
            # for each word in corpus, save word as key and unigram
            # probability as value in dictionary
            for i in self.corpus.split():
                if i not in d:
                    d[i] = self.calculate_ngram_probabilities(i)
            self.model = d
        # bigram
        else:
            # initialize dictionary.
            # bigram model uses nested dictionary with outer dictionary having
            # previous word as key and inner dictionary as value.
            # inner dictionary has word as key and bigram probability of
            # (previous word, word) as value.
            d = {"UNK": {}}
            corpus_list = self.corpus.split()
            for i in range(1, len(corpus_list)):
                # get previous word and current word
                word1 = corpus_list[i-1].strip()
                word2 = corpus_list[i].strip()

                # initialize empty dictionary for previous word
                if word1 not in d:
                    d[word1] = {}

                # get probability of (previous word, word)
                probability = self.calculate_ngram_probabilities(word2, word1)
                # add that probability to the dictionary.
                # dictionary is indexed by [previous word][word]
                if word2 not in d[word1]:
                    d[word1][word2] = probability

                # probability that word1 (previous word) is UNK
                probability_word1_unk = self.calculate_ngram_probabilities(
                    word2, "UNK")
                # probability that word2 (current word) is UNK
                probability_word2_unk = self.calculate_ngram_probabilities(
                    "UNK", word1)

                d["UNK"][word2] = probability_word1_unk
                d[word1]["UNK"] = probability_word2_unk

            # probability that word1 and word2 (previous word and word) are UNK
            d["UNK"]["UNK"] = self.calculate_ngram_probabilities("UNK", "UNK")

            self.model = d

    # calculate perplexity
    def calculate_ngram_perplexity(self, dataset):
        corpus_list = dataset.split()
        # perplexity variable holds the sum of logs of n-gram probability
        # over each token in the corpus
        perplexity = 0
        # unigram
        if self.n == 1:
            for i in corpus_list:
                word = i.strip()
                # if word is not in training corpus meaning that
                # if a new word is encountered at test time, treat is as UNK
                if word not in self.model:
                    perplexity -= log(self.model["UNK"])
                # word is in training corpus
                else:
                    perplexity -= log(self.model[word])

            # finish calculating perplexity.
            # len(corpus_list) is total number of tokens in corpus
            return exp((1/len(corpus_list))*perplexity)
        # bigram
        else:
            # go through each bigram in corpus
            for i in range(1, len(corpus_list)):
                word1 = corpus_list[i-1].strip()
                word2 = corpus_list[i].strip()
                if word1 not in self.model:
                    if word2 in self.model["UNK"]:
                        # reaches here if previous word not in training model
                        # and word is (treat previous word as UNK)
                        perplexity -= log(self.model["UNK"][word2])
                    else:
                        # reaches here if previous word and word both not in
                        # training model (treat both as UNK)
                        perplexity -= log(self.model["UNK"]["UNK"])
                else:
                    if word2 in self.model[word1]:
                        # reaches here if previous word and word both in
                        # training model
                        perplexity -= log(self.model[word1][word2])
                    else:
                        # reaches here if previous word in training model
                        # and word is not (treat word as UNK)
                        perplexity -= log(self.model[word1]["UNK"])

            # finish calculating perplexity
            return exp((1/len(corpus_list))*perplexity)

# read and preprocess dataset
def read_process_corpus(corpus_path, preprocess_type):
    f = open(corpus_path, 'r')
    data = f.read()
    f.close()
    if preprocess_type == 2:
        return data
    # lowercase data
    data = data.lower()
    # string.punctuation holds !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~
    # maketrans(str1, str2, str3): str1 is list of characters to replace,
    # str2 is list of characters to replace characters in str1,
    # str3 is list of characters to delete
    data = data.translate(str.maketrans("", "", string.punctuation))
    # lemmatize data
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(
        word) for word in data.split()]
    final_data = " ".join(lemmatized_words)
    return final_data


# replace all tokens that occur fewer than n times in the training set by UNK
# where n is set to be 3
def replace_with_unk(corpus):
    corpus_list = corpus.split()
    # initialize dictionary to hold token as key and count of token as value
    count_tokens = {}
    for i in corpus_list:
        count_tokens[i] = 1 + count_tokens.get(i, 0)
    for index, i in enumerate(corpus_list):
        # replace "rare" tokens
        if count_tokens[i] <= 2:
            corpus_list[index] = "UNK"
    # make corpus as string to be used in later functions
    corpus = " ".join(corpus_list)
    return corpus

# function to run the code


def run():
    download('wordnet')
    
    # dictionary with smoothing types
    smoothing_types = {
        1: "no smoothing",
        2: "Laplace smoothing",
        3: "add-k smoothing"
    }

    # get the type of model from user
    while True:
        model_type = int(
            input("Select model type:\n\t1) Unigram \n\t2) Bigram\n"))
        if model_type not in (1, 2):
            print("Invalid input")
            continue
        else:
            break

    # get preprocessing type
    while True:
        preprocess_type = int(input(
            "Preprocess corpus?\n\t1) Yes \n\t2) No\n"))
        if model_type not in (1, 2):
            print("Invalid input")
            continue
        else:
            break
        
    # get the type of smoothing from user
    while True:
        smoothing_type = int(input(
            "Select smoothing type:\n\t1) No smoothing \n\t2) Laplace smoothing\n\t3) Add-k smoothing\n"))
        if smoothing_type not in (1, 2, 3):
            print("Invalid input")
            continue
        else:
            break

    # if user chooses add-k smoothing, get k value from user
    k_value = 1.0
    if smoothing_type == 3:
        while True:
            k_value = float(input("Enter a k value where k>0 and k<1\n"))
            if k_value <= 0 or k_value >= 1:
                print("k value out of range")
                continue
            else:
                break

    print("Reading training set...")
    corpus = read_process_corpus("A1_DATASET/train.txt", preprocess_type)
    print("Reading validation set...")
    validation_corpus = read_process_corpus("A1_DATASET/val.txt", preprocess_type)
    corpus = replace_with_unk(corpus)

    print("Initializing corpus...")
    ngram = Ngram(corpus, smoothing_type, k_value, model_type)
    print(f"Calculating {model_type}-gram probabilities...")
    ngram.get_ngram_probability_map()
    print("Saving model to a file...")
    ngram.save_model()
    
    # calculate perplexity for unigram or smoothed bigram
    if model_type == 1 or (model_type == 2 and smoothing_type != 1):
        print(f"Calculating perplexity using {smoothing_types[smoothing_type]}...")
        training_perplexity = ngram.calculate_ngram_perplexity(corpus)
        validation_perplexity = ngram.calculate_ngram_perplexity(validation_corpus)
        print(f"Training set perplexity = {training_perplexity:0.3f}")
        print(f"Validation set perplexity = {validation_perplexity:0.3f}")


if __name__ == "__main__":
    run()
