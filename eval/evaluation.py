#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 14:21:46 2018
"""

import argparse
import os
import numpy as np
import random
from collections import defaultdict

from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from config import EvaluationConfig
POSITIVE_SAMPLES_FILE = [EvaluationConfig.pos_ancestor_descendant,
                         EvaluationConfig.pos_parent_child]
NEGATIVE_SAMPLES_FILE = [EvaluationConfig.neg_ancestor_descendant,
                         EvaluationConfig.neg_parent_child]


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

        #scores = defaultdict(lambda: float('-inf'))
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
            random.Random(3).shuffle(pairs)  # fixed seed: Yu found that the reciprocal rank metrics are sensitive to this randomization, we fix seed to be 3 so that the mean first reciprocal rank is similar to its expected value (slightly greater than (1/1+...+1/11)/11 = 2.74)
            for i, pair in enumerate(pairs):
                scores[pair] += i * delta

        results = []
        for hyper, hypo in list(self.positives):
            results.append([hyper, hypo, scores[(hyper, hypo)], True])
        for hyper, hypo in list(self.negatives):
            results.append([hyper, hypo, scores[(hyper, hypo)], False])

        return sorted(
            results,
            key=lambda t: (-t[2], t[3], t[0], t[1]),
        )

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
    def calculate_precision_jiaming(
            self, 
            general_specific_score_tuples, 
            grouping_by="hypernym", 
            strategy="average",
            avg_weights="uniform",
            inverse=False,
        ):
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

def printEvalResults(criterion):
    evaluator = Evaluation(criterion)
    with open(args.input_file) as f:

        inputs = []
        for line in f:
            tokens = line.strip().split('\t')
            inputs.append([tokens[0], tokens[1], float(tokens[2])])
        sorted_pairs = evaluator.sort_input_tuples(inputs, args.randrank)
        print("\n=========================================================")
        print("* Evaluating on %d pairs based on **%s** relation" % (len(sorted_pairs), criterion))
        if args.k == -1:
            for k in [100, 300, 1000, 3000]:
                print("1. Precision@(k = %d) = %f" % (k, Evaluation.precision_at_n(sorted_pairs, k)))
        else:
            print("1. Precision@(k = %d) = %f" % (args.k, Evaluation.precision_at_n(sorted_pairs, args.k)))
        #print("2. Precision@(recall = %f) = %f" % (args.r, Evaluation.precision_at_recall(sorted_pairs, args.r)))
        #print("3. Precision@(All Nodes Covered) = %f" % evaluator.precision_at_all_nodes_covered(sorted_pairs))
        print()
        print("The following two metrics are grouped on %s" % args.grouping_by)
        #print("4. Macro (Uniform) Mean Average Rank = %f" % evaluator.calculate_precision_jiaming(sorted_pairs,
        #                                                                                          grouping_by=args.grouping_by,
        #                                                                                          strategy="average",
        #                                                                                          avg_weights="uniform",
        #                                                                                          inverse=False,
        #                                                                                         ))
        #print("5. Micro (Weighted) Mean Average Rank = %f" % evaluator.calculate_precision_jiaming(sorted_pairs,
        #                                                                                           grouping_by=args.grouping_by,
        #                                                                                           strategy="average",
        #                                                                                           avg_weights="weighted",
        #                                                                                           inverse=False,
        #                                                                                          ))
        #print("6. Macro (Uniform) Mean First Rank = %f" % evaluator.calculate_precision_jiaming(sorted_pairs,
        #                                                                                        grouping_by=args.grouping_by,
        #                                                                                        strategy="minimum",
        #                                                                                        avg_weights="uniform",
        #                                                                                        inverse=False,
        #                                                                                       ))
        #print("7. Micro (Weighted) Mean First Rank = %f" % evaluator.calculate_precision_jiaming(sorted_pairs,
        #                                                                                         grouping_by=args.grouping_by,
        #                                                                                         strategy="minimum",
        #                                                                                         avg_weights="weighted",
        #                                                                                         inverse=False,
        #                                                                                        ))
        print("8. Macro (Uniform) Mean Average Rank (inverse) = %f" % evaluator.calculate_precision_jiaming(sorted_pairs,
                                                                                                  grouping_by=args.grouping_by,
                                                                                                  strategy="average",
                                                                                                  avg_weights="uniform",
                                                                                                  inverse=True,
                                                                                                 ))
        print("9. Micro (Weighted) Mean Average Rank (inverse) = %f" % evaluator.calculate_precision_jiaming(sorted_pairs,
                                                                                                   grouping_by=args.grouping_by,
                                                                                                   strategy="average",
                                                                                                   avg_weights="weighted",
                                                                                                   inverse=True,
                                                                                                  ))
        print("10. Macro (Uniform) Mean First Rank (inverse) = %f" % evaluator.calculate_precision_jiaming(sorted_pairs,
                                                                                                grouping_by=args.grouping_by,
                                                                                                strategy="minimum",
                                                                                                avg_weights="uniform",
                                                                                                inverse=True,
                                                                                               ))
        print("11. Micro (Weighted) Mean First Rank (inverse) = %f" % evaluator.calculate_precision_jiaming(sorted_pairs,
                                                                                                 grouping_by=args.grouping_by,
                                                                                                 strategy="minimum",
                                                                                                 avg_weights="weighted",
                                                                                                 inverse=True,
                                                                                                ))
        print("=========================================================\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Calculate evaluation metrics on the given input file")
    parser.add_argument("input_file", help="file name of the input in the format of hypernym`\\t`hypohym`\\t`score")
    parser.add_argument("-c", "--criterion", default="both",
                        help="whether ancestor or parent is used to calculate the metrics")
    parser.add_argument("-k", default=-1, type=int, help="Number of labeled pairs tested in metrics 1")
    parser.add_argument("-r", default=0.8, type=float, help="Recall specific in metrics 2")
    parser.add_argument("-g", "--grouping_by", default="hypernym", help="Grouping by criteria used for metrics 4 and 5")
    parser.add_argument("--randrank", dest="randrank", help="", action="store_true")
    parser.set_defaults(randrank=True)
    parser.add_argument("--notrandrank", dest="randrank", help="", action="store_false")
    args = parser.parse_args()
    assert os.path.isfile(args.input_file), "Input file not found"
    assert args.criterion in ["parent", "ancestor", "both"], "Criterion must be either ancestor or parent or both (default)"

    print("\n=========================================================")
    print("Warnings (for Jiaming and Xinwei): ")
    print("for some of your baselines, exactly one of {parent, ancestor}-mode evaluation is valid, because your algorithm may use the other, which makes the results unreasonably good.")
    print("=========================================================\n")
    
    if args.criterion == "both":
        printEvalResults("ancestor")
        printEvalResults("parent")
    else:
        printEvalResults(args.criterion)
    
    