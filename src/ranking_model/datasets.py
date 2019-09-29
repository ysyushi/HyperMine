import torch
from torch.utils.data import Dataset
import numpy as np
import itertools
import random
from collections import defaultdict


class Triplets(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, init_embedding, raw_sets):
        """

        :param init_embedding: a gensim.models.keyedvectors.Word2VecKeyedVectors object
        :param raw_sets: a list of raw element sets, each element has an embedding vector in init_embedding
        """
        self.embedding = init_embedding
        self.vocab_size, self.embed_dim = self.embedding.vectors.shape
        self.index2word = self.embedding.index2word
        self.word2index = {word: index for index, word in enumerate(self.index2word)}
        self.raw_sets = raw_sets

        self.raw_sets_vocab = list(itertools.chain(*self.raw_sets))
        self.word2raw_index = {word: idx for idx, word in enumerate(self.raw_sets_vocab)}
        self.train_labels = np.zeros(len(self.raw_sets_vocab), dtype=int)
        self.train_data = np.zeros([len(self.raw_sets_vocab), self.embed_dim])
        for label_id, raw_set in enumerate(self.raw_sets):
            for word in raw_set:
                self.train_labels[self.word2raw_index[word]] = label_id
                self.train_data[self.word2raw_index[word], :] = self.embedding[word]

        self.labels_set = sorted(list(self.train_labels))
        self.label_to_indices = {label: np.where(self.train_labels == label)[0] for label in self.labels_set}
        for label in self.labels_set:  # sanity checking
            if len(self.label_to_indices[label]) < 2:
                print("Error: Do not allow single element set in training data")

        self.train_labels = torch.tensor(self.train_labels)
        self.train_data = torch.tensor(self.train_data)

        self.indice_to_positive_indices = {}  # cache all positive indices list for fast sampling
        for label in self.labels_set:
            for indice in self.label_to_indices[label]:
                self.indice_to_positive_indices[indice] = np.array(
                    [ele for ele in self.label_to_indices[label] if ele != indice])

        self.pointer = 0
        self.neg_pool = list(range(len(self.raw_sets_vocab)))

    def _sample_negative(self, pos_label):
        """ sample an element index that is of label not equal to pos_label

        :param pos_label:
        :return:
        """
        if self.pointer == 0:
            random.shuffle(self.neg_pool)

        while self.train_labels[self.neg_pool[self.pointer]].item() == pos_label:
            self.pointer = (self.pointer + 1) % len(self.raw_sets_vocab)

        negative_index = self.neg_pool[self.pointer]
        self.pointer = (self.pointer + 1) % len(self.raw_sets_vocab)
        return negative_index

    def __getitem__(self, index):
        ele1, label1 = self.train_data[index], self.train_labels[index].item()
        positive_index = np.random.choice(self.indice_to_positive_indices[index])
        negative_index = self._sample_negative(label1)

        ele2 = self.train_data[positive_index]
        ele3 = self.train_data[negative_index]

        return (ele1, ele2, ele3), []

    def __len__(self):
        return len(self.train_data)


class DirectionalTriplets(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    """

    def __init__(self, init_embedding, raw_keyed_sets):
        """

        :param init_embedding: a gensim.models.keyedvectors.Word2VecKeyedVectors object
        :param raw_keyed_sets: a dict, key is the anchor term, each key is unique
        """
        self.embedding = init_embedding
        _, self.embed_dim = self.embedding.vectors.shape

        self.raw_keyed_sets = raw_keyed_sets
        self.anchor_vocab = []  # only contain anchor elements, which are the keys in raw_keyed_sets
        self.word2raw_index = {}  # contain all elements in raw_keyed_sets
        self.anchor2label = {}  # anchor elements to labels, label starts from 0
        self.label2indices = {}  # label id to general word raw index (excluding the anchor word)
        self.raw_index2label_set = defaultdict(
            list)  # word raw index to label set, NOTE: we allow each element to appear in multiple sets
        for label, anchor in enumerate(raw_keyed_sets):
            self.anchor_vocab.append(anchor)
            if anchor not in self.word2raw_index:
                self.word2raw_index[anchor] = len(self.word2raw_index)
            self.anchor2label[anchor] = label
            indices = []
            for ele in raw_keyed_sets[anchor]:
                if ele not in self.word2raw_index:
                    self.word2raw_index[ele] = len(self.word2raw_index)
                raw_index = self.word2raw_index[ele]
                indices.append(raw_index)
                self.raw_index2label_set[raw_index].append(label)
            self.label2indices[label] = indices

        self.raw_index2label_set = {raw_index: set(self.raw_index2label_set[raw_index]) for raw_index in
                                    self.raw_index2label_set}
        self.full_data = np.zeros([len(self.word2raw_index), self.embed_dim])  # (full_vocab_size, embed_dim)

        for word in self.word2raw_index:
            raw_index = self.word2raw_index[word]
            self.full_data[raw_index, :] = self.embedding[word]

        self.full_data = torch.tensor(self.full_data)

        # following code is used to speed up negative sampling
        self.pointer = 0
        self.neg_pool = [self.word2raw_index[word] for word in self.word2raw_index if word not in self.anchor_vocab]
        self.neg_pool_size = len(self.neg_pool)

    def __len__(self):
        return len(self.anchor_vocab)

    def _sample_negative(self, pos_label):
        if self.pointer == 0:
            random.shuffle(self.neg_pool)

        while pos_label in self.raw_index2label_set[self.neg_pool[self.pointer]]:
            self.pointer = (self.pointer + 1) % self.neg_pool_size

        negative_index = self.neg_pool[self.pointer]
        self.pointer = (self.pointer + 1) % self.neg_pool_size
        return negative_index

    def __getitem__(self, index):
        anchor = self.anchor_vocab[index]
        anchor_index = self.word2raw_index[anchor]
        anchor_label = self.anchor2label[anchor]
        ele1 = self.full_data[anchor_index]

        positive_index = np.random.choice(self.label2indices[anchor_label])
        negative_index = self._sample_negative(anchor_label)

        ele2 = self.full_data[positive_index]
        ele3 = self.full_data[negative_index]

        # print(anchor_index, positive_index, negative_index)
        return (ele1, ele2, ele3), []


class DirectionalTripletsWithPairFeature(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    """

    def __init__(self, init_embedding, raw_keyed_sets, pair_features):
        """

        :param init_embedding: a gensim.models.keyedvectors.Word2VecKeyedVectors object
        :param raw_keyed_sets: a dict, key is the anchor term, each key is unique
        :param pair_features: a dict of dicts, hypernym -> {hyponym: a list of float}
        """
        self.embedding = init_embedding
        _, self.embed_dim = self.embedding.vectors.shape

        self.raw_keyed_sets = raw_keyed_sets
        self.anchor_vocab = []  # only contain anchor elements, which are the keys in raw_keyed_sets
        self.word2raw_index = {}  # contain all elements in raw_keyed_sets
        self.anchor2label = {}  # anchor elements to labels, label starts from 0
        self.label2indices = {}  # label id to general word raw index (excluding the anchor word)
        self.raw_index2label_set = defaultdict(list)  # word raw index to label set, NOTE: we allow each element to appear in multiple sets
        self.pair_feature_dim = None

        for label, anchor in enumerate(raw_keyed_sets):
            self.anchor_vocab.append(anchor)
            if anchor not in self.word2raw_index:
                self.word2raw_index[anchor] = len(self.word2raw_index)
            self.anchor2label[anchor] = label
            indices = []
            for ele in raw_keyed_sets[anchor]:
                if ele not in self.word2raw_index:
                    self.word2raw_index[ele] = len(self.word2raw_index)
                raw_index = self.word2raw_index[ele]
                indices.append(raw_index)
                self.raw_index2label_set[raw_index].append(label)
            self.label2indices[label] = indices

        self.raw_index2label_set = {raw_index: set(self.raw_index2label_set[raw_index]) for raw_index in
                                    self.raw_index2label_set}
        self.full_data = np.zeros([len(self.word2raw_index), self.embed_dim])  # (full_vocab_size, embed_dim)

        for word in self.word2raw_index:
            raw_index = self.word2raw_index[word]
            self.full_data[raw_index, :] = self.embedding[word]

        self.full_data = torch.tensor(self.full_data).float()
        print("tensor type of self.full_data", self.full_data.dtype)

        # following code is used to speed up negative sampling
        self.pointer = 0
        self.neg_pool = [self.word2raw_index[word] for word in self.word2raw_index if word not in self.anchor_vocab]
        self.neg_pool_size = len(self.neg_pool)

        # following code is used to store pair features
        self.pair_features = {}  # same structure as pair_features but keyed with word index
        for hypernym in pair_features:
            if hypernym in self.word2raw_index:
                tmp = {}
                for hypo in pair_features[hypernym]:
                    if hypo in self.word2raw_index:
                        tmp[self.word2raw_index[hypo]] = pair_features[hypernym][hypo]
                        if not self.pair_feature_dim:
                            self.pair_feature_dim = len(pair_features[hypernym][hypo])
                self.pair_features[self.word2raw_index[hypernym]] = tmp

    def __len__(self):
        return len(self.anchor_vocab)

    def _sample_negative(self, pos_label):
        if self.pointer == 0:
            random.shuffle(self.neg_pool)

        while pos_label in self.raw_index2label_set[self.neg_pool[self.pointer]]:
            self.pointer = (self.pointer + 1) % self.neg_pool_size

        negative_index = self.neg_pool[self.pointer]
        self.pointer = (self.pointer + 1) % self.neg_pool_size
        return negative_index

    def _get_pair_features(self, hypernym_index, hyponym_index):
        if hypernym_index not in self.pair_features:
            res = np.array(
                [0.28431755, 0.072794236, 0.09576533, 2.8206701e-05, 0.18775113, 0.15142219, 0.23707405, 0.21777077,
                 0.26868778, -0.0077130636, 0.0031342732, -3.1776857e-05, -0.086081468, 0.009919174, 0.019383442,
                 0.00010948985, 0.014362899, -0.090511329, 0.045277614, 0.12446611, 9.6907552e-06, 0.12446611,
                 0.27718776, 0.189549, 0.3725535, -9.2563317e-07, 0.3725535, 0.18822739], dtype=np.float32)
        else:
            if hyponym_index not in self.pair_features[hypernym_index]:
                res = np.array(
                    [0.28431755, 0.072794236, 0.09576533, 2.8206701e-05, 0.18775113, 0.15142219, 0.23707405, 0.21777077,
                     0.26868778, -0.0077130636, 0.0031342732, -3.1776857e-05, -0.086081468, 0.009919174, 0.019383442,
                     0.00010948985, 0.014362899, -0.090511329, 0.045277614, 0.12446611, 9.6907552e-06, 0.12446611,
                     0.27718776, 0.189549, 0.3725535, -9.2563317e-07, 0.3725535, 0.18822739], dtype=np.float32
                )
            else:
                res = self.pair_features[hypernym_index][hyponym_index]

        res = torch.from_numpy(res).float()
        return res

    def __getitem__(self, index):
        anchor = self.anchor_vocab[index]
        anchor_index = self.word2raw_index[anchor]
        anchor_label = self.anchor2label[anchor]
        ele1 = self.full_data[anchor_index]

        positive_index = np.random.choice(self.label2indices[anchor_label])
        negative_index = self._sample_negative(anchor_label)

        ele2 = self.full_data[positive_index]
        ele3 = self.full_data[negative_index]

        positive_pair_feature = self._get_pair_features(anchor_index, positive_index)
        negative_pair_feature = self._get_pair_features(anchor_index, negative_index)
        return (ele1, ele2, ele3), (positive_pair_feature, negative_pair_feature), []

