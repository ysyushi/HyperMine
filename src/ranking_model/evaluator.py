import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
import itertools
import math
import argparse
import os
import numpy as np
import random
from collections import defaultdict


POSITIVE_SAMPLES_FILE = ["/shared/data/li215/linkedin_maple/codes/data/ccs/pairs/positive_ancestor2d_pairs2.txt",
                         "/shared/data/li215/linkedin_maple/codes/data/ccs/pairs/positive_parent2c_pairs2.txt"]
NEGATIVE_SAMPLES_FILE = ["/shared/data/li215/linkedin_maple/codes/data/ccs/pairs/negative_ancestor2d_pairs2.txt",
                         "/shared/data/li215/linkedin_maple/codes/data/ccs/pairs/negative_parent2c_pairs2.txt"]


class Evaluation:

    def __init__(self, criterion="ancestor"):
        self.positives, self.negatives = set(), set()
        c = 1 if criterion == "parent" else 0
        self.read_positive_samples(c)
        self.read_negative_samples(c)

    def read_positive_samples(self, c):
        with open(POSITIVE_SAMPLES_FILE[c]) as f:
            for line in f:
                tokens = line.strip().split("\t")
                self.positives.add((tokens[0], tokens[1]))

    def read_negative_samples(self, c):
        with open(NEGATIVE_SAMPLES_FILE[c]) as f:
            for line in f:
                tokens = line.strip().split("\t")
                self.negatives.add((tokens[0], tokens[1]))

    def sort_input_tuples(self, general_specific_score_tuples, randrank=False):
        tuples_on = self.positives.union(self.negatives)
        assert len(tuples_on) == len(self.positives) + len(self.negatives), "Error on input label pairs"

        # scores = defaultdict(lambda: float('-inf'))
        scores = {}
        for general, specific, score in general_specific_score_tuples:
            if (general, specific) in tuples_on:
                scores[(general, specific)] = score
        min_score = min(scores.values()) if (len(scores) > 0) else 0
        if min_score == 0:
            default_score = -1
        else:
            default_score = min_score - abs(min_score)
        for general, specific in tuples_on:
            if (general, specific) not in scores:
                scores[(general, specific)] = default_score

                # goal: remove ties
        if randrank:
            smallest_difference = (min_score - default_score) / 2
            delta = smallest_difference / len(scores)
            pairs = list(scores.keys())
            pairs.sort()
            random.Random(3).shuffle(
                pairs)  # fixed seed: Yu found that the reciprocal rank metrics are sensitive to this randomization, we fix seed to be 3 so that the mean first reciprocal rank is similar to its expected value (slightly greater than (1/1+...+1/11)/11 = 2.74)
            for i, pair in enumerate(pairs):
                scores[pair] += i * delta

        results = []
        for hyper, hypo in list(self.positives):
            results.append([hyper, hypo, scores[(hyper, hypo)], True])
        for hyper, hypo in list(self.negatives):
            results.append([hyper, hypo, scores[(hyper, hypo)], False])

        return sorted(results, key=lambda t: (-t[2], t[3], t[0], t[1]), )

    # Metrics 1
    @staticmethod
    def precision_at_n(sorted_tuples, num):
        results = [test_result for _, _, _, test_result in sorted_tuples[:num]]
        return sum(results) / len(results)

    # Metrics 2
    @staticmethod
    def precision_at_recall(sorted_tuples, recall):
        results = [test_result for _, _, _, test_result in sorted_tuples]
        positive_threshold = sum(results) * recall

        n_positive = 0
        n_tested = 0
        for test_result in results:
            n_positive += 1 if test_result else 0
            n_tested += 1
            if n_positive >= positive_threshold:
                break
        return n_positive / n_tested

    def get_vocabulary(self):
        vocab = set()
        for hyper, hypo in self.positives:
            vocab.add(hyper)
            vocab.add(hypo)
        return vocab

    # Metrics 3
    def precision_at_all_nodes_covered(self, sorted_tuples):
        vocab = self.get_vocabulary()
        covered = set()

        n_positive = 0
        n_tested = 0
        for general, specific, _, test_result in sorted_tuples:
            if test_result:
                n_positive += 1
                covered.add(general)
                covered.add(specific)
            n_tested += 1
            if len(covered) >= len(vocab):
                break
        return n_positive / n_tested

    @staticmethod
    def calculate_ranks_from_distances(all_distances, positive_relations):
        """
        all_distances: a np array
        positive_relations: a list of array indices
        contributed by Jiaming Shen
        """
        if len(all_distances) == len(positive_relations):
            return [1] * len(positive_relations)
        positive_relation_distances = all_distances[positive_relations]
        negative_relation_distances = np.ma.array(all_distances, mask=False)
        negative_relation_distances.mask[positive_relations] = True
        ranks = list((negative_relation_distances >= positive_relation_distances[:, np.newaxis]).sum(axis=1) + 1)
        return ranks

    @staticmethod
    def calculate_ranks_from_distances_inverse(all_distances, positive_relations):
        """
        all_distances: a np array
        positive_relations: a list of array indices
        contributed by Jiaming Shen
        """
        if len(all_distances) == len(positive_relations):
            return [1] * len(positive_relations)
        positive_relation_distances = all_distances[positive_relations]
        negative_relation_distances = np.ma.array(all_distances, mask=False)
        negative_relation_distances.mask[positive_relations] = True
        ranks = list(1 / ((negative_relation_distances >= positive_relation_distances[:, np.newaxis]).sum(axis=1) + 1))
        return ranks

    # Metrics 4/5
    def calculate_precision_jiaming(self, general_specific_score_tuples, grouping_by="hypernym", strategy="average",
            avg_weights="uniform", inverse=False, ):
        assert grouping_by in ["hypernym", "hyponym"], "Metrics 4/5 must be gropuped by either hypernym or hyponym"
        assert strategy in ["average", "minimum"], "Strategy for metrics 4/5 must be either average(4) or minimum(5)"
        assert avg_weights in ["uniform", "weighted"], "Weights for average must be either uniform or weighted"

        two_layer_dict = defaultdict(lambda: {})
        if grouping_by == "hypernym":
            for a, d in self.positives:
                two_layer_dict[a][d] = [float('-inf'), True]
            for a, d in self.negatives:
                if a in two_layer_dict:
                    two_layer_dict[a][d] = [float('-inf'), False]
        elif grouping_by == "hyponym":
            for a, d in self.positives:
                two_layer_dict[d][a] = [float('-inf'), True]
            for a, d in self.negatives:
                if d in two_layer_dict:
                    two_layer_dict[d][a] = [float('-inf'), False]

        for general, specific, score, _ in general_specific_score_tuples:
            if grouping_by == "hypernym" and general in two_layer_dict and specific in two_layer_dict[general]:
                two_layer_dict[general][specific][0] = score
            elif grouping_by == "hyponym" and specific in two_layer_dict and general in two_layer_dict[specific]:
                two_layer_dict[specific][general][0] = score

        total, count = 0, 0
        for first in two_layer_dict:
            scores = []
            positive_indices = []
            for second in two_layer_dict[first]:
                scores.append(two_layer_dict[first][second][0])
                if two_layer_dict[first][second][1]:
                    positive_indices.append(len(scores) - 1)
            if not inverse:
                result = Evaluation.calculate_ranks_from_distances(np.array(scores), positive_indices)
                if strategy == "average" and avg_weights == "uniform":
                    count += 1
                    total += np.mean(result)
                elif strategy == "average" and avg_weights == "weighted":
                    count += len(positive_indices)
                    total += np.sum(result)
                elif strategy == "minimum" and avg_weights == "uniform":
                    count += 1
                    total += min(result)
                elif strategy == "minimum" and avg_weights == "weighted":
                    count += len(positive_indices)
                    total += min(result) * len(positive_indices)
            if inverse:
                result = Evaluation.calculate_ranks_from_distances_inverse(np.array(scores), positive_indices)
                if strategy == "average" and avg_weights == "uniform":
                    count += 1
                    total += np.mean(result)
                elif strategy == "average" and avg_weights == "weighted":
                    count += len(positive_indices)
                    total += np.sum(result)
                elif strategy == "minimum" and avg_weights == "uniform":
                    count += 1
                    total += max(result)
                elif strategy == "minimum" and avg_weights == "weighted":
                    count += len(positive_indices)
                    total += max(result) * len(positive_indices)

        return total / count


def evaluation_main(inputs):
    """

    :param inputs: a list of triplets (hyper, hypo, score)
    :type inputs: list
    :return:
    :rtype:
    """
    args = {}
    args["randrank"] = True
    args["criterion"] = "ancestor"
    args["k"] = -1
    args["r"] = 0.8
    args["grouping_by"] = "hypernym"

    evaluator = Evaluation(args["criterion"])
    sorted_pairs = evaluator.sort_input_tuples(inputs, args["randrank"])

    metrics = {}
    p_metrics = []  # contain all precision metrics
    r_metrics = []  # contain all mrr metrics (not all ranking based metrics)
    print("* Evaluating on %d pairs based on %s relation" % (len(sorted_pairs), args["criterion"]))
    if args["k"] == -1:
        for k in [100, 300, 1000, 3000]:
            m = Evaluation.precision_at_n(sorted_pairs, k)
            print("1. Precision@(k = %d) = %f" % (k, m))
            metrics["P@{}".format(k)] = m
            if k in [100, 1000]:
                p_metrics.append(m)
    else:
        m = Evaluation.precision_at_n(sorted_pairs, args["k"])
        # print("1. Precision@(k = %d) = %f" % (args["k"], m))
        metrics["P@{}".format(args["k"])] = m
        # p_metrics.append(m)

    m = Evaluation.precision_at_recall(sorted_pairs, args["r"])
    # print("2. Precision@(recall = %f) = %f" % (args["r"], m))
    metrics["P@R={}".format(args["r"])] = m
    # p_metrics.append(m)

    m = evaluator.precision_at_all_nodes_covered(sorted_pairs)
    print("3. Precision@(All Nodes Covered) = %f" % m)
    metrics["P@ANC"] = m
    # p_metrics.append(m)

    print()
    print("The following two metrics are grouped on %s" % args["grouping_by"])

    m = evaluator.calculate_precision_jiaming(sorted_pairs, grouping_by=args["grouping_by"], strategy="average",
                                              avg_weights="uniform", inverse=False)
    # print("4. Macro (Uniform) Mean Average Rank = %f" % m)
    metrics["MacroU-MAR"] = m

    m = evaluator.calculate_precision_jiaming(sorted_pairs, grouping_by=args["grouping_by"], strategy="average",
                                              avg_weights="weighted", inverse=False)
    # print("5. Micro (Weighted) Mean Average Rank = %f" % m)
    metrics["MicroW-MAR"] = m

    m = evaluator.calculate_precision_jiaming(sorted_pairs, grouping_by=args["grouping_by"], strategy="minimum",
                                              avg_weights="uniform", inverse=False)
    # print("6. Macro (Uniform) Mean First Rank = %f" % m)
    metrics["MacroU-MFR"] = m

    m = evaluator.calculate_precision_jiaming(sorted_pairs, grouping_by=args["grouping_by"], strategy="minimum",
                                              avg_weights="weighted", inverse=False)
    # print("7. Micro (Weighted) Mean First Rank = %f" % m)
    metrics["MicroW-MFR"] = m

    m = evaluator.calculate_precision_jiaming(sorted_pairs, grouping_by=args["grouping_by"], strategy="average",
                                              avg_weights="uniform", inverse=True)
    print("8. Macro (Uniform) Mean Average Rank (inverse) = %f" % m)
    metrics["MacroU-MARR"] = m
    r_metrics.append(m)

    m = evaluator.calculate_precision_jiaming(sorted_pairs, grouping_by=args["grouping_by"], strategy="average",
                                              avg_weights="weighted", inverse=True)
    print("9. Micro (Weighted) Mean Average Rank (inverse) = %f" % m)
    metrics["MicroW-MARR"] = m
    r_metrics.append(m)

    m = evaluator.calculate_precision_jiaming(sorted_pairs, grouping_by=args["grouping_by"], strategy="minimum",
                                              avg_weights="uniform", inverse=True)
    print("10. Macro (Uniform) Mean First Rank (inverse) = %f" % m)
    metrics["MacroU-MFRR"] = m
    r_metrics.append(m)

    m = evaluator.calculate_precision_jiaming(sorted_pairs, grouping_by=args["grouping_by"], strategy="minimum",
                                              avg_weights="weighted", inverse=True)
    print("11. Micro (Weighted) Mean First Rank (inverse) = %f" % m)
    metrics["MicroW-MFRR"] = m
    r_metrics.append(m)

    # Finally we fine an overall metrics for tuning
    # overall metric is the geometric mean of (p_metrics arithmetic mean and r_metrics arithmetic mean)
    overall_metric = np.array(p_metrics).mean() * np.array(r_metrics).mean()
    metrics["all"] = overall_metric

    return metrics