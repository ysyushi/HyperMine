from collections import defaultdict
import itertools
from tqdm import tqdm
import mmap
import os
import logging
import torch
from gensim.models import KeyedVectors  # used to load word2vec
import hashlib
import itertools
import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
from collections import Counter
import numpy as np
import networkx as nx


class Metrics:

    def __init__(self):
        self.metrics = {}

    def __len__(self):
        return len(self.metrics)

    def add(self, metric_name, metric_value):
        self.metrics[metric_name] = metric_value


class Results:

    def __init__(self, filename):

        self._filename = filename

        open(self._filename, 'a+')

    def _hash(self, x):

        return hashlib.md5(json.dumps(x, sort_keys=True).encode('utf-8')).hexdigest()

    def save(self, hyperparams, dev_f1='NULL', test_f1='NULL', test_precision='NULL', test_recall='NULL',
             test_avg_jaccard='NULL', test_node_precision='NULL', test_node_recall='NULL',
             test_ARI="NULL", test_FMI="NULL", test_NMI="NULL"):

        result = {'dev_f1': dev_f1,
                  'test_f1': test_f1,
                  'test_precision': test_precision,
                  'test_recall': test_recall,
                  'test_avg_jaccard': test_avg_jaccard,
                  'test_node_precision': test_node_precision,
                  'test_node_recall': test_node_recall,
                  'test_ARI': test_ARI,
                  'test_FMI': test_FMI,
                  'test_NMI': test_NMI,
                  'hash': self._hash(hyperparams)}
        result.update(hyperparams)

        with open(self._filename, 'a+') as out:
            out.write(json.dumps(result) + '\n')

    def save_metrics(self, hyperparams, metrics):

        result = metrics.metrics  # a dict
        result["hash"] = self._hash(hyperparams)
        result.update(hyperparams)
        with open(self._filename, 'a+') as out:
            out.write(json.dumps(result) + '\n')

    def best(self):

        results = sorted([x for x in self],
                         key=lambda x: -x['dev_f1'])

        if results:
            return results[0]
        else:
            return None

    def __getitem__(self, hyperparams):

        params_hash = self._hash(hyperparams)

        with open(self._filename, 'r+') as fle:
            for line in fle:
                datum = json.loads(line)

                if datum['hash'] == params_hash:
                    del datum['hash']
                    return datum

        raise KeyError

    def __contains__(self, x):

        try:
            self[x]
            return True
        except KeyError:
            return False

    def __iter__(self):

        with open(self._filename, 'r+') as fle:
            for line in fle:
                datum = json.loads(line)

                del datum['hash']

                yield datum


def save_model(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)


def load_model(model, load_dir, load_prefix, steps):
    model_prefix = os.path.join(load_dir, load_prefix)
    model_path = "{}_steps_{}.pt".format(model_prefix, steps)
    model.load_state_dict(torch.load(model_path))


def save_model_with_architecture(model, save_dir, save_prefix, steps):
    # TODO: this function is not working yet, to be updated later
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    state = {
        "trained_steps": steps,
        "params": model.params,
        "state_dict": model.state_dict()
    }
    torch.save(state, save_path)


def load_model_with_architecture(file_path):
    # TODO: this function is not working yet, to be updated later
    state = torch.load(file_path)
    return state["params"], state["state_dict"]


def save_checkpoint(model, optimizer, save_dir, save_prefix, step):
    """ Save model checkpoint
    """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = "{}_steps_{}.pt".format(save_prefix, step)
    checkpoint = {
        "epoch": step + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(model, optimizer, load_dir, load_prefix, step):
    """ Load model checkpoint

    Note: the output model and optimizer are on CPU and need to be explicitly moved to GPU
    c.f. https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/3

    :param model:
    :param optimizer:
    :param load_dir:
    :param load_prefix:
    :param step:
    :return:
    """
    checkpoint_prefix = os.path.join(load_dir, load_prefix)
    checkpoint_path = "{}_steps_{}.pt".format(checkpoint_prefix, step)
    checkpoint = torch.load(checkpoint_path)
    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, start_epoch


def toGPU(optimizer, device):
    """ Move optimizer from CPU to GPU

    :param optimizer:
    :param device:
    :return:
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def check_model_consistency(args):
    """

    :param args:
    :return:
    """
    if args.use_pair_feature == 0 and not args.modelName.startswith("np_"):
        return False, "model without string pair features must has name starting with \"np_\""
    elif args.use_pair_feature == 1 and args.modelName.startswith("np_"):
        return False, "model with string pair features cannot has name starting with \"np_\""
    elif args.loss_fn == "margin_rank" and not args.modelName.endswith("s"):
        return False, "model trained with MarginRankingLoss must have the combiner that output a single " \
                      "scalar for set-instance pair (i.e., ends with Sigmoid Function)"
    elif args.loss_fn != "margin_rank" and args.modelName.endswith("s"):
        return False, "model not trained with MarginRankingLoss cannot have the combiner that output a single " \
                      "scalar for set-instance pair (i.e., ends with Sigmoid Function)"
    elif args.loss_fn == "self_margin_rank" and "_sd_" not in args.modelName:
        return False, "model trained with self MarginRankingLoss must have the combiner that " \
                      "based on score difference (sd)"
    elif args.loss_fn != "self_margin_rank" and "_sd_" in args.modelName:
        return False, "model not trained with self-based MarginRankingLoss cannot have the combiner that " \
                      "based on score difference (sd)"
    else:
        return True, ""


def myLogger(name='', logpath='./'):
    logger = logging.getLogger(name)
    if len(logger.handlers) != 0:
        print('reuse the same logger: {}'.format(name))
        return logger
    else:
        print('create new logger: {}'.format(name))
    fn = os.path.join(logpath, 'run-{}.log'.format(name))
    if os.path.exists(fn):
        print('[warning] log file {} already existed'.format(fn))
    else:
        print('saving log to {}'.format(fn))

    # following two lines are used to solve no file output problem
    # c.f. https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d][%(funcName)s] %(levelname)s-> %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S', filename=fn, filemode='w')
    # define a new Handler to log to console as well
    console = logging.StreamHandler()
    # set a format which is the same for console use
    formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d][%(funcName)s] %(levelname)s-> %(message)s',
                                  datefmt='%a %d %b %Y %H:%M:%S')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logger.addHandler(console)

    return logger


def load_embedding(fi, embed_name="word2vec"):
    if embed_name == "word2vec":
        embedding = KeyedVectors.load_word2vec_format(fi)
    else:
        # TODO: allow training embedding from scratch later
        print("[ERROR] Please specify the pre-trained embedding")
        exit(-1)

    vocab_size, embed_dim = embedding.vectors.shape
    index2word = ['PADDING_IDX'] + embedding.index2word
    word2index = {word: index for index, word in enumerate(index2word)}
    return embedding, index2word, word2index, vocab_size, embed_dim


def load_embedding_linkedin(input_file, embed_name="word2vec"):
    with open(input_file, "r") as f_in:
        vocab_size, embed_dim = map(int, f_in.readline().strip().split())
        word2index = {}  # {node: row_index}, where type(node) == str, type(row_index) == int
        index2word = []
        # embedding_matrix = np.zeros((num_nodes, dim), dtype=np.float_)
        embedding = {}  # word -> np.array
        idx = 0
        for line in f_in:
            line_split = line.strip().split()
            cur_name = line_split[0]
            word2index[cur_name] = idx
            index2word.append(cur_name)
            # node_index_dict[cur_name] = idx
            embedding[cur_name] = np.asarray(line_split[1:], dtype=np.float_)
            # embedding_matrix[idx]
            idx += 1

    assert idx == vocab_size, "Node number does not match."
    return embedding, index2word, word2index, vocab_size, embed_dim


def load_raw_data(fi):
    raw_data_strings = []
    with open(fi, "r") as fin:
        for line in fin:
            raw_data_strings.append(line.strip())
    return raw_data_strings


def load_dataset(fi):
    positives = set()
    negatives = set()
    token2string = defaultdict(set)  # used for generating negative examples
    with open(fi, "r") as fin:
        for line in fin:
            line = line.strip()
            eid, synset = line.split(" ", 1)
            synset = eval(synset)
            
            for syn in synset:
                for tok in syn.split("_"):
                    token2string[tok].add((syn, eid))
            
            for pair in itertools.combinations(synset, 2):
                pair = frozenset([ele+"||"+eid for ele in pair])
                positives.add(frozenset(pair))
        
        # generate negative
        for token in tqdm(token2string, desc="Generating negative pairs ..."):
            strings = token2string[token]
            if len(strings) < 2:
                continue
            else:
                for pair in itertools.combinations(strings, 2):
                    pair = frozenset([ele[0]+"||"+ele[1] for ele in pair])
                    if pair not in positives:
                        negatives.add(pair)
    return list(positives), list(negatives)


def build_dataset(positives, negatives):
    dataset = []
    dataset.extend([(ele, 1.0) for ele in positives])
    dataset.extend([(ele, 0.0) for ele in negatives])
    return dataset


def print_args(args, interested_args="all"):
    print("\nParameters:")
    if interested_args == "all":
        for attr, value in sorted(args.__dict__.items()):
            print("\t{}={}".format(attr.upper(), value))
    else:
        for attr, value in sorted(args.__dict__.items()):
            if attr in interested_args:
                print("\t{}={}".format(attr.upper(), value))
    print('-' * 89)


def get_num_lines(file_path):
    """ Usage:
    with open(inputFile,"r") as fin:
      for line in tqdm(fin, total=get_num_lines(inputFile)):
            ...
    """
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def KM_Matching(weight_nm):
    """ Maximum weighted matching

    :param weight_nm:
    :return:
    """
    x = len(weight_nm)
    y = len(weight_nm[0])
    n = max(x, y)
    NONE = -1e6
    INF = 1e9
    weight = [[NONE for j in range(n + 1)] for i in range(n + 1)]
    for i in range(x):
        for j in range(y):
            weight[i + 1][j + 1] = weight_nm[i][j]
    lx = [0. for i in range(n + 1)]
    ly = [0. for i in range(n + 1)]
    match = [-1 for i in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            lx[i] = max(lx[i], weight[i][j])
    for root in range(1, n + 1):
        vy = [False for i in range(n + 1)]
        slack = [INF for i in range(n + 1)]
        pre = [0 for i in range(n + 1)]
        py = 0
        match[0] = root
        while True:
            vy[py] = True
            x = match[py]
            delta = INF
            yy = 0
            for y in range(1, n + 1):
                if not vy[y]:
                    if lx[x] + ly[y] - weight[x][y] < slack[y]:
                        slack[y] = lx[x] + ly[y] - weight[x][y]
                        pre[y] = py
                    if slack[y] < delta:
                        delta = slack[y]
                        yy = y
            for y in range(n + 1):
                if vy[y]:
                    lx[match[y]] -= delta
                    ly[y] += delta
                else:
                    slack[y] -= delta
            py = yy
            if match[py] == -1: break
        while True:
            prev = pre[py]
            match[py] = match[prev]
            py = prev
            if py == 0: break
    score = 0.
    for i in range(1, n + 1):
        v = weight[match[i]][i]
        if v > NONE:
            score += v
    return score


def end2end_evaluation_matching(groundtruth, result):  # Jaccard Similarity
    n = len(groundtruth)
    m = len(result)
    G = nx.DiGraph()
    S = n + m
    T = n + m + 1
    C = 1e8
    for i in range(n):
        for j in range(m):
            s1 = groundtruth[i]
            s2 = result[j]
            s12 = set(s1) & set(s2)
            weight = len(s12) / (len(s1) + len(s2) - len(s12))
            weight = int(weight * C)
            if weight > 0:
                G.add_edge(i, n + j, capacity=1, weight=-weight)
    for i in range(n):
        G.add_edge(S, i, capacity=1, weight=0)
    for i in range(m):
        G.add_edge(i + n, T, capacity=1, weight=0)
    mincostFlow = nx.algorithms.max_flow_min_cost(G, S, T)
    mincost = nx.cost_of_flow(G, mincostFlow) / C
    return -mincost / m


def evaluate_clustering(cls_pred, cls_true):
    """ Evaluate clustering results

    :param cls_pred: a list of lists consisting of elements in a model predicted cluster
    :param cls_true: a list of lists consisting of elements in a ground truth cluster
    :return: a dictionary keyed with evaluation metric
    """

    vocab_pred = set(itertools.chain(*cls_pred))
    vocab_true = set(itertools.chain(*cls_true))
    assert (vocab_pred == vocab_true), "Unmatched vocabulary during clustering evaluation"

    # Cluster number
    num_of_predict_clusters = len(cls_pred)

    # Cluster size histogram
    cluster_size2num_of_predicted_clusters = Counter([len(cluster) for cluster in cls_pred])

    # Exact cluster prediction
    pred_cluster_set = set([frozenset(cluster) for cluster in cls_pred])
    gt_cluster_set = set([frozenset(cluster) for cluster in cls_true])
    # print(pred_cluster_set.intersection(gt_cluster_set))
    num_of_exact_set_prediction = len(pred_cluster_set.intersection(gt_cluster_set))

    # Clustering metrics
    word2rank = {}
    wordrank2gt_cluster = {}
    rank = 0
    for cid, cluster in enumerate(cls_true):
        for word in cluster:
            if word not in word2rank:
                word2rank[word] = rank
                rank += 1
            wordrank2gt_cluster[word2rank[word]] = cid
    gt_cluster_vector = [ele[1] for ele in sorted(wordrank2gt_cluster.items())]

    wordrank2pred_cluster = {}
    for cid, cluster in enumerate(cls_pred):
        for word in cluster:
            wordrank2pred_cluster[word2rank[word]] = cid
    pred_cluster_vector = [ele[1] for ele in sorted(wordrank2pred_cluster.items())]

    ARI = adjusted_rand_score(gt_cluster_vector, pred_cluster_vector)
    FMI = fowlkes_mallows_score(gt_cluster_vector, pred_cluster_vector)
    NMI = normalized_mutual_info_score(gt_cluster_vector, pred_cluster_vector)

    # Pair-based clustering metrics
    def pair_set(labels):
        S = set()
        cluster_ids = np.unique(labels)
        for cluster_id in cluster_ids:
            cluster = np.where(labels == cluster_id)[0]
            n = len(cluster)  # number of elements in this cluster
            if n >= 2:
                for i in range(n):
                    for j in range(i + 1, n):
                        S.add((cluster[i], cluster[j]))
        return S

    F_S = pair_set(gt_cluster_vector)
    F_K = pair_set(pred_cluster_vector)
    if len(F_K) == 0:
        pair_recall = 0
        pair_precision = 0
        pair_f1 = 0
    else:
        common_pairs = len(F_K & F_S)
        pair_recall = common_pairs / len(F_S)
        pair_precision = common_pairs / len(F_K)
        eps = 1e-6
        pair_f1 = 2 * pair_precision * pair_recall / (pair_precision + pair_recall + eps)

    # KM matching
    mwm_jaccard = end2end_evaluation_matching(cls_true, cls_pred)
    # n = len(cls_true)
    # m = len(cls_pred)
    # weight = [[0.0 for j in range(m)] for i in range(n)]
    # for i in range(n):
    #     for j in range(m):
    #         s1 = cls_true[i]
    #         s2 = cls_pred[j]
    #         s12 = sum([1 for a in s1 for b in s2 if a == b])
    #         weight[i][j] = s12 / (len(s1) + len(s2) - s12)
    # tot_score = KM_Matching(weight)
    # mwm_jaccard = tot_score / m

    metrics = {"ARI": ARI, "FMI": FMI, "NMI": NMI, "pair_recall": pair_recall, "pair_precision": pair_precision,
               "pair_f1": pair_f1, "predicted_clusters": cls_pred, "num_of_predicted_clusters": num_of_predict_clusters,
               "cluster_size2num_of_predicted_clusters": cluster_size2num_of_predicted_clusters,
               "num_of_exact_set_prediction": num_of_exact_set_prediction,
               "maximum_weighted_match_jaccard": mwm_jaccard}

    return metrics


def load_directional_supervision(fi):
    with open(fi, "r") as fin:
        hyper2hypo = defaultdict(set)
        hypo2hyper = defaultdict(set)
        vocab = []
        for line in fin:
            line = line.strip()
            if line:
                hyper, hypo = line.split("\t")
                hyper2hypo[hyper].add(hypo)
                hypo2hyper[hypo].add(hyper)
                vocab.extend([hyper, hypo])

    hyper2hypo = {hyper: list(hyper2hypo[hyper]) for hyper in hyper2hypo}
    hypo2hyper = {hypo: list(hypo2hyper[hypo]) for hypo in hypo2hyper}
    return hyper2hypo, hypo2hyper


def load_element_pairs(fi, with_label=False):
    pairs = []
    with open(fi, "r") as fin:
        for line in fin:
            line = line.strip()
            if line:
                #  assume first two columns are term pairs and the third column is label
                if with_label:
                    pairs.append(line.split("\t"))
                else:
                    pairs.append(line.split("\t")[:2])
    return pairs


def load_pair_features(fi_key, fi_values, feature_index_list=None):
    feature_matrix = np.load(fi_values)
    pair_features = {}  # hypernym -> {hyponym: ndarray}
    with open(fi_key, "r") as fin:
        for idx, line in tqdm(enumerate(fin), desc="loading pair features ..."):
            line = line.strip()
            if line:
                hyper, hypo = line.split("\t")
                if hyper not in pair_features:
                    pair_features[hyper] = {}

                if feature_index_list:
                    pair_features[hyper][hypo] = feature_matrix[idx, feature_index_list]
                else:
                    pair_features[hyper][hypo] = feature_matrix[idx, :]
    return pair_features
