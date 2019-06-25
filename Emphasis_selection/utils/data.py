
import os
import pickle
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import config

def read_text_embeddings(filename):
    embeddings = []
    word2index = {}
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split()
            word2index[line[0]] = i
            embeddings.append(list(map(float, line[1:])))
    assert len(word2index) == len(embeddings)
    return word2index, np.array(embeddings)

def flatten(elems):
    return [e for elem in elems for e in elem]

class Encoder(object):
    def __init__(self, corpus, emb_path):

        self.word2index, self.word_emb = self.get_pretrain_embeddings(emb_path, corpus.get_word_vocab())
        self.index2word = {i: w for w, i in self.word2index.items()}

    def encode_words(self, corpus):
        corpus.train.words = [self.encode(self.word2index, sample) for sample in corpus.train.words]
        corpus.dev.words = [self.encode(self.word2index, sample) for sample in corpus.dev.words]
        corpus.test.words = [self.encode(self.word2index, sample) for sample in corpus.test.words]

    def decode_words(self, corpus):
        corpus.train.words = [self.encode(self.index2word, sample) for sample in corpus.train.words]
        corpus.dev.words = [self.encode(self.index2word, sample) for sample in corpus.dev.words]
        corpus.test.words = [self.encode(self.index2word, sample) for sample in corpus.test.words]

    def encode(self, elem2index, elems):
        return [elem2index[elem] for elem in elems]

    @staticmethod
    def get_encoder(corpus, emb_path, encoder_pkl_path):
        if os.path.exists(encoder_pkl_path):
            encoder = Encoder.load(encoder_pkl_path)
        else:
            encoder = Encoder(corpus, emb_path)
            encoder.save(encoder_pkl_path)

        Encoder.print_stats(encoder)

        return encoder

    def print_stats(self):
        print('[LOG]')
        print("[LOG] Word vocab size: {}".format(len(self.word2index)))


    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'))

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as fp:
            return pickle.load(fp)



    def get_pretrain_embeddings(self, filename, vocab):
        assert len(vocab) == len(set(vocab)), "The vocabulary contains repeated words"

        w2i, emb = read_text_embeddings(filename)
        word2index = {'+pad+': 0, '+unk+': 1}
        embeddings = np.zeros((len(vocab) + 2, emb.shape[1]))

        scale = np.sqrt(3.0 / emb.shape[1])
        embeddings[word2index['+unk+']] = np.random.uniform(-scale, scale, (1, emb.shape[1]))

        perfect_match = 0
        case_match = 0
        no_match = 0

        for i in range(len(vocab)):
            word = vocab[i]
            index = len(word2index)  # do not use i because word2index has predefined tokens

            word2index[word] = index
            if word in w2i:
                embeddings[index] = emb[w2i[word]]
                perfect_match += 1
            elif word.lower() in w2i:
                embeddings[index] = emb[w2i[word.lower()]]
                case_match += 1
            else:
                embeddings[index] = np.random.uniform(-scale, scale, (1, emb.shape[1]))
                no_match += 1
        print("[LOG] Word embedding stats -> Perfect match: {}; Case match: {}; No match: {}".format(perfect_match,
                                                                                                     case_match,
                                                                                                     no_match))
        return word2index, embeddings





class Corpus(object):
    def __init__(self, corpus_path):
        self.train = Dataset(os.path.join(corpus_path, 'train/'))
        self.dev = Dataset(os.path.join(corpus_path, 'dev/'))
        self.test = Dataset(os.path.join(corpus_path, 'test/'))

    @staticmethod
    def get_corpus(corpus_dir, corpus_pkl_path):
        if os.path.exists(corpus_pkl_path):
            with open(corpus_pkl_path, 'rb') as fp:
                corpus= pickle.load(fp)

        else:
            corpus = Corpus(corpus_dir)
            with open(corpus_pkl_path, 'wb') as fp:
                pickle.dump(corpus, fp, -1)
        corpus.print_stats()
        return corpus

    @staticmethod
    def _get_unique(elems):
        corpus = flatten(elems)
        elems, freqs = zip(*Counter(corpus).most_common())
        return list(elems)


    def print_stats(self):

        print("Train dataset: {}".format(len(self.train.words)))
        print("Dev dataset: {}".format(len(self.dev.words)))
        print("Test dataset: {}".format(len(self.test.words)))

    def get_word_vocab(self):
        return self._get_unique(self.train.words + self.dev.words + self.test.words)
    def get_label_vocab(self):
        return self._get_unique(["O", "I"])

class Dataset(object):
    def __init__(self, path):
        self.words  = self.read_conll_format(os.path.join(path, 'bio_probs.txt'))
        self.labels = self.read_conll_format_labels(os.path.join(path, 'bio_probs.txt'))

        assert len(self.words) == len(self.labels)


    def read_conll_format_labels(self, filename):
        lines = self.read_lines(filename) + ['']
        posts, post = [], []
        for line in lines:
            if line:
                probs = line.split("\t")[2]
                # reading probabilities from the last column and also normalaize it by div on 9
                probs = [(int(l)/9) for l in probs.split("|")]
                probs = [probs[2],probs[0]+probs[1] ]

                post.append(probs)
                print("post: ", post)
            elif post:
                posts.append(post)
                post = []
        # a list of lists of words/ labels
        return posts

    def read_conll_format(self, filename):
        lines = self.read_lines(filename) + ['']
        posts, post = [], []
        for line in lines:
            if line:
                words = line.split("\t")[0]
                # print("words: ", words)
                post.append(words)
            elif post:
                posts.append(post)
                post = []
        # a list of lists of words/ labels
        return posts

    def read_lines(self, filename):
        with open(filename, 'r') as fp:
            lines = [line.strip() for line in fp]
        return lines
