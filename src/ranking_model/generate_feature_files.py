from collections import defaultdict, Counter
import itertools
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from gensim.models import KeyedVectors  # used to load word2vec
import random

from sklearn.preprocessing import normalize
from sklearn import preprocessing

"""
To Run: 
Put embedding file path in fi_embed
Put supervision pair file path in fi_supervision
Put evaluation pair file path in fi_eval (notice that this is a list and just put the file path as the first element)
Put pairwise feature file paths in pairwise_feature_file_list

Then, change the output file path in 
fo_term (1 place for term pair)
fo_features (2 places for two matrices) 
fo_features_mean (2 places for two matrices column mean)
"""

fi_embed = "/shared/data/qiz3/text_summ/src/HEER/intermediate_data/heer_dblp_tax_60_op_1_mode_0_rescale_0.1_lr_10_lrr_10_actual.emb"
fi_supervision = "/shared/data/jiaming/linkedin-maple/maple/data/supervision_pairs.txt"
fi_eval = ["/shared/data/jiaming/linkedin-maple/maple/data/test_ancestor2d_pairs2.txt"]

li215_dih_scores_dir = '/shared/data/li215/linkedin_maple/codes/output/metapath_score/'
nzhang31_dih_scores_dir = '/shared/data/nzhang31/linkedin_maple/codes/output/metapath_score/'
t2p_score9 = li215_dih_scores_dir + 't2p_score9_try2_all_normalized.tsv'
t2p_weeds_prec = li215_dih_scores_dir + 't2p_weeds_prec3.tsv'
t2p_cl = li215_dih_scores_dir + 't2p_cl.tsv'  # ClarkeDE
t2p_inv_cl = li215_dih_scores_dir + 't2p_inv_cl.tsv'
tpt_cl = li215_dih_scores_dir + 'tpt_cl.tsv'  # ClarkeDE
tpt_inv_cl = li215_dih_scores_dir + 'tpt_inv_cl.tsv'

pairwise_feature_file_list = [
    nzhang31_dih_scores_dir + "tpv_cl/_.tsv",
    nzhang31_dih_scores_dir + "temp_tpcer_10000_score6/_total.tsv",
    nzhang31_dih_scores_dir + "temp_tpcer_10000_score8/_total.tsv",
    nzhang31_dih_scores_dir + "temp_tpcer_10000_score9/_total.tsv",
    nzhang31_dih_scores_dir + "temp_tpcer_10000_inv_cl/_total.tsv",

    nzhang31_dih_scores_dir + "temp_tpcer_100_score6/_total.tsv",
    nzhang31_dih_scores_dir + "temp_tpcer_100_score8/_total.tsv",
    nzhang31_dih_scores_dir + "temp_tpcer_100_score9/_total.tsv",
    nzhang31_dih_scores_dir + "temp_tpcer_100_inv_cl/_total.tsv",

    li215_dih_scores_dir + "tpv/tsv/score6_all_all.tsv",
    li215_dih_scores_dir + "tpv/tsv/score8_all_all.tsv",
    li215_dih_scores_dir + "tpv/tsv/score9_all_all.tsv",
    nzhang31_dih_scores_dir + "tpv_inv_cl/_.tsv",

    li215_dih_scores_dir + "tpt_score6_normalized.tsv",
    li215_dih_scores_dir + "tpt_score8_normalized.tsv",
    li215_dih_scores_dir + "tpt_score9_normalized.tsv",
    tpt_cl,
    tpt_inv_cl,

    li215_dih_scores_dir + "tpa_score6_normalized.tsv",
    li215_dih_scores_dir + "tpa_score8_normalized.tsv",
    li215_dih_scores_dir + "tpa_score9_normalized.tsv",
    nzhang31_dih_scores_dir + "tpa_cl/_.tsv",
    nzhang31_dih_scores_dir + "tpa_inv_cl/_.tsv",

    li215_dih_scores_dir + "t2p_score6_normalized.tsv",
    li215_dih_scores_dir + "t2p_score8_normalized.tsv",
    li215_dih_scores_dir + "t2p_score9_try2_all_normalized.tsv",
    t2p_cl,
    t2p_inv_cl
]

# print("=== Pilot check ===")
# for file_path in pairwise_feature_file_list:
#     with open(file_path, "r") as fin:
#         print("successfully open: {}".format(file_path))


print("=== Loading embedding ===")
cnt = 0
word2embed_string = {}
embed_vocab = []
with open(fi_embed, "r") as fin:
    for idx, line in tqdm(enumerate(fin), "desc: loading embedding"):
        line = line.strip()
        if idx == 0:
            embed_dim = line.split(" ")[1]
            print("Embed dim: {}".format(embed_dim))
            continue
        word, embed_string = line.split(" ", 1)
        embed_vocab.append(word)
embed_vocab_set = set(embed_vocab)

print("=== Loading supervision file ===")
supervision_vocab_set = set()
with open(fi_supervision, "r") as fin:
    for line in fin:
        line = line.strip()
        if line:
            hyper, hypo = line.split("\t")
            supervision_vocab_set.add(hyper)
            supervision_vocab_set.add(hypo)

print("=== Loading evaluation pairs file ===")
evaluation_vocab = []
for fi in fi_eval:
    with open(fi, "r") as fin:
        for line in fin:
            line = line.strip()
            if line:
                segs = line.split("\t")
                evaluation_vocab.append(segs[0].lower())
                evaluation_vocab.append(segs[1].lower())
evaluation_vocab_set = set(evaluation_vocab)

print("Vocab size of supervision: {}".format(len(supervision_vocab_set)))
print("Vocab size of evaluation set: {}".format(len(evaluation_vocab_set)))
union_vocab = supervision_vocab_set | evaluation_vocab_set
print("Vocab size of supervision union evaluation set: {}".format(len(union_vocab)))
print("Vocab size of embedding: {}".format(len(embed_vocab_set)))
print("-"*80)
print("# of terms in supervision that have embedding: {}".format(len(supervision_vocab_set & embed_vocab_set)))
print("# of terms in evaluation that have embedding: {}".format(len(evaluation_vocab_set & embed_vocab_set)))
print("# of terms in unioned set that have embedding: {}".format(len(union_vocab & embed_vocab_set)))

word2index = {ele[1]: ele[0] for ele in enumerate(union_vocab)}
index2word = {word2index[word]: word for word in word2index}

print("Number of pairwise features:{}".format(len(pairwise_feature_file_list)))

pair_feature_dim = len(pairwise_feature_file_list)
term_pair = {}  # term_pair
term2term2feature = {}  # termA -> termB -> []
for fid, file_path in enumerate(pairwise_feature_file_list):
    print("Processing file {}: {}".format(fid, file_path))
    with open(file_path, "r") as fin:
        for line in tqdm(fin):
            line = line.strip()
            if line:
                segs = line.split("\t")
                if segs[0] in word2index and segs[1] in word2index:  # both terms are interesting
                    idx1 = word2index[segs[0]]
                    idx2 = word2index[segs[1]]
                    if idx1 not in term2term2feature:
                        term2term2feature[idx1] = {}
                    if idx2 not in term2term2feature[idx1]:
                        term2term2feature[idx1][idx2] = [-1.0] * pair_feature_dim
                    term2term2feature[idx1][idx2][fid] = float(segs[2])
                    # if (word2index[segs[0]], word2index[segs[1]]) not in term_pair:
                    #     term_pair[(word2index[segs[0]], word2index[segs[1]])] = [-1.0] * pair_feature_dim
                    # term_pair[(word2index[segs[0]], word2index[segs[1]])][fid] = float(segs[2])
    # print("Number of term pairs: {}".format(len(term_pair)))

feature_matrix = []
term_pair_list = []
# for ele in tqdm(term_pair.keys(), desc="generating feature matrix and term pair list"):
#     term_pair_list.append(ele)
#     feature_matrix.append(term_pair[ele])
for term1 in term2term2feature:
    for term2 in term2term2feature[term1]:
        feature = term2term2feature[term1][term2]
        term_pair_list.append((term1, term2))
        feature_matrix.append(feature)

feature_matrix = np.array(feature_matrix, dtype=np.float32)
print("feature_matrix.shape:", feature_matrix.shape)

# save temp
fo_term = "/shared/data/jiaming/linkedin-maple/maple/data/edge.keys3.tsv"
with open(fo_term, "w") as fout:
    for ele in tqdm(term_pair_list, desc="saving term pair keys"):
        fout.write("{}\t{}\n".format(index2word[ele[0]], index2word[ele[1]]))

# fo_features = "/shared/data/jiaming/linkedin-maple/maple/data/edge.values3.raw.npy"
# np.save(fo_features, feature_matrix.astype(np.float32))

# replace "-1" with np.nan
feature_matrix_w_nan = np.where(feature_matrix == -1, np.nan, feature_matrix)

# calculate column mean
column_mean = np.nanmean(feature_matrix_w_nan, axis=0)
print("column_mean:", column_mean)

# find inds to be replaced
inds = np.where(np.isnan(feature_matrix_w_nan))

# save mean normalized feature matrix
feature_matrix[inds] = np.take(column_mean, inds[1])

# scale matrix
feature_matrix_scaled = preprocessing.scale(feature_matrix)


fo_features = "/shared/data/jiaming/linkedin-maple/maple/data/edge.values3.scaled.npy"
np.save(fo_features, feature_matrix_scaled.astype(np.float32))
fo_features_mean = "/shared/data/jiaming/linkedin-maple/maple/data/edge.values3.scaled.mean.txt"
with open(fo_features_mean, "w") as fout:
    fout.write(str(list(np.nanmean(feature_matrix_scaled, axis=0))))


fo_features = "/shared/data/jiaming/linkedin-maple/maple/data/edge.values3.unscaled.npy"
np.save(fo_features, feature_matrix.astype(np.float32))
fo_features_mean = "/shared/data/jiaming/linkedin-maple/maple/data/edge.values3.unscaled.mean.txt"
with open(fo_features_mean, "w") as fout:
    fout.write(str(list(np.nanmean(feature_matrix, axis=0))))
