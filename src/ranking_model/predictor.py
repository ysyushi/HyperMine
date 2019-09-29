import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np


def pair_prediction(pair_net, term_pairs, cuda, options, batch_size=100):
    pair_net.eval()

    prediction_scores = []
    pairs = []
    for term_pair in term_pairs:
        pairs.append(term_pair)
        if len(pairs) % batch_size == 0:
            x1 = torch.tensor([options["embedding"][p[0]] for p in pairs])
            x2 = torch.tensor([options["embedding"][p[1]] for p in pairs])
            if cuda:
                x1 = x1.cuda()
                x2 = x2.cuda()
            scores = list(pair_net(x1, x2).squeeze().cpu().detach().numpy())
            prediction_scores.extend(scores)
            pairs = []

    if len(pairs) != 0:
        x1 = torch.tensor([options["embedding"][p[0]] for p in pairs])
        x2 = torch.tensor([options["embedding"][p[1]] for p in pairs])
        if cuda:
            x1 = x1.cuda()
            x2 = x2.cuda()
        scores = list(pair_net(x1, x2).squeeze().cpu().detach().numpy())
        prediction_scores.extend(scores)

    assert len(prediction_scores) == len(term_pairs), "mismatch prediction scores and test pairs"
    return prediction_scores


def pair_prediction_with_pair_feature(pair_net, term_pairs, pair_features, cuda, options, batch_size=100):

    def get_pair_features(hypernym_word, hyponym_word, pair_feature_dim=4):
        if hypernym_word not in pair_features:
            # res = np.zeros(pair_feature_dim)
            # res = np.array([9.93365765e-01, 1.00132143e+00, -2.30689820e-05, 7.90985048e-01], dtype=np.float32)
            res = np.array(
                [0.28431755, 0.072794236, 0.09576533, 2.8206701e-05, 0.18775113, 0.15142219, 0.23707405, 0.21777077,
                 0.26868778, -0.0077130636, 0.0031342732, -3.1776857e-05, -0.086081468, 0.009919174, 0.019383442,
                 0.00010948985, 0.014362899, -0.090511329, 0.045277614, 0.12446611, 9.6907552e-06, 0.12446611,
                 0.27718776, 0.189549, 0.3725535, -9.2563317e-07, 0.3725535, 0.18822739], dtype=np.float32)
        else:
            if hyponym_word not in pair_features[hypernym_word]:
                # res = np.array([9.93365765e-01, 1.00132143e+00, -2.30689820e-05, 7.90985048e-01], dtype=np.float32)
                res = np.array(
                    [0.28431755, 0.072794236, 0.09576533, 2.8206701e-05, 0.18775113, 0.15142219, 0.23707405, 0.21777077,
                     0.26868778, -0.0077130636, 0.0031342732, -3.1776857e-05, -0.086081468, 0.009919174, 0.019383442,
                     0.00010948985, 0.014362899, -0.090511329, 0.045277614, 0.12446611, 9.6907552e-06, 0.12446611,
                     0.27718776, 0.189549, 0.3725535, -9.2563317e-07, 0.3725535, 0.18822739], dtype=np.float32)
            else:
                res = pair_features[hypernym_word][hyponym_word]

        res = torch.from_numpy(res).float()
        return res

    pair_net.eval()

    prediction_scores = []
    pairs = []
    for term_pair in term_pairs:
        pairs.append(term_pair)
        if len(pairs) % batch_size == 0:
            x1 = torch.tensor([options["embedding"][p[0]] for p in pairs])
            x2 = torch.tensor([options["embedding"][p[1]] for p in pairs])
            r = torch.stack([get_pair_features(p[0], p[1], options["pair_feature_dim"]) for p in pairs], dim=0)
            if cuda:
                x1 = x1.cuda()
                x2 = x2.cuda()
                r = r.cuda()
            scores = list(pair_net(x1, x2, r).squeeze().cpu().detach().numpy())
            prediction_scores.extend(scores)
            pairs = []

    if len(pairs) != 0:
        x1 = torch.tensor([options["embedding"][p[0]] for p in pairs])
        x2 = torch.tensor([options["embedding"][p[1]] for p in pairs])
        r = torch.stack([get_pair_features(p[0], p[1], options["pair_feature_dim"]) for p in pairs], dim=0)
        if cuda:
            x1 = x1.cuda()
            x2 = x2.cuda()
            r = r.cuda()
        scores = list(pair_net(x1, x2, r).squeeze().cpu().detach().numpy())
        prediction_scores.extend(scores)

    assert len(prediction_scores) == len(term_pairs), "mismatch prediction scores and test pairs"
    return prediction_scores
