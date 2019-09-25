#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 20:14:54 2018

@author: liyuchen
"""

import heapq
import math
import multiprocessing as mp
#from label_generation import normalize
import numpy as np
import pickle
import time
import os

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
#from eval.label_generation import normalize
from config import Config, PreprocessingConfig, DIHConfig, MetapathContextConfig
from combine_tsv import get_idx2words

T2P_CL = DIHConfig.t2p_cl
T2P_INV_CL = DIHConfig.t2p_inv_cl

def calc_metapath_scores(
        context_dict, 
        t1_idx_list,
        out_fn_format,
     ):
    '''
    @store scores dict: {1, ..., 10} -> int
     score1: c(w1) * c(w2) * p(w2|w1)
     score2: c(w2) * p(w2|w1)
     score3: c(w1) * p(w2|w1)
     score4: p(w2|w1)
     score5: c(w1) * c(w2) * p(w1|w2)
     score6: c(w2) * p(w1|w2) (same as score3 !!!)
     score7: c(w1) * p(w1|w2)
     score8: p(w1|w2)
     score9: p(w2|w1) - p(w1|w2)
    '''
    scores = {}
    for i in range(1, 11):
        scores[i] = {}
    keyword_list = list(context_dict.keys())
    cnt = 0
    for t1_idx in t1_idx_list:
        if t1_idx < len(keyword_list):
            t1 = keyword_list[t1_idx]
            for t2 in context_dict.keys():
                if t1 != t2:
                    c_t1 = len(context_dict[t1])
                    c_t2 = len(context_dict[t2])
                    c_t1_t2 = len(context_dict[t1].intersection(context_dict[t2]))
                    if c_t1 != 0 and c_t2 != 0:
                        vals = [
                                None, 
                                c_t2 * c_t1_t2, 
                                c_t2 * c_t1_t2 / c_t1, 
                                c_t1_t2, 
                                c_t1_t2 / c_t1, 
                                c_t1 * c_t1_t2, 
                                c_t1_t2, 
                                c_t1 * c_t1_t2 / c_t2, 
                                c_t1_t2 / c_t2, 
                                c_t1_t2 / c_t1 - c_t1_t2 / c_t2
                                ]
                        for i in range(1, 10):
                            if vals[i] != 0:
                                scores[i][(t1, t2)] = vals[i]
                cnt += 1
                if cnt % 10000000 == 0:
                    print('%s processed %d pairs' % (mp.current_process().name, cnt))
                elif cnt % 1000000 == 0:
                    print('\t%s processed %d pairs'% (mp.current_process().name, cnt))
                elif cnt % 100000 == 0:
                    print('\t\t%s processed %d pairs'% (mp.current_process().name, cnt))
    for i in range(1, 11):
        out_fn = (out_fn_format % (i, mp.current_process()._identity[0]))
        with open(out_fn, "wb") as f:
            pickle.dump(scores[i], f)
            #print(scores[i])
    print("\t\t%s saved results" % (mp.current_process().name))

def distribute_metapath_scores_calculation(context_dict, num_workers, out_fn_format):
    keyword_list = list(context_dict.keys())
    print('number of keywords:', len(keyword_list))
    num_pairs = len(keyword_list) * (len(keyword_list) - 1) // 2
    print('num_pairs:', num_pairs)
    
    pool = mp.Pool(processes=num_workers)    
    batch_size = int(math.floor(len(keyword_list)/(num_workers-1)))
    print("batch_size: %d" % batch_size)
    start_pos = [i * batch_size for i in range(0, num_workers)]
    
    results = [
                pool.apply_async(
                        calc_metapath_scores, 
                        args=(context_dict, list(range(start, start+batch_size)), out_fn_format),
                    ) 
                for i, start in enumerate(start_pos)
            ]
    results = [p.get() for p in results]
    return      

def read_stored_file(path_type, score_out_fn_format, target_score_idx='all'):
    """
    If target_score_idx != 'all', then only collect certain score types
    The large structure remains the same
    """
    scores = {}
    for i in range(1, 11):
        scores[i] = {}
    scores_dir = ("output/metapath_score/%s/" % path_type)
    fns = os.listdir(scores_dir)
    print('total number of files:', len(fns))
    cnt = 0
    for fn in fns:
        #print(fn)
        with open(scores_dir + fn, 'rb') as f:
            scores_for_one_word = pickle.load(f)
        score_idx = int(fn.replace('score','').split('_')[0])
        if (target_score_idx=='all' or target_score_idx==score_idx):
            scores[score_idx].update(scores_for_one_word)
            cnt += 1
            if cnt % 100 == 0:
                print('processed %d files' % (cnt))
            elif cnt % 10 == 0:
                print('\tprocessed %d files'% (cnt))
    with open(score_out_fn_format % target_score_idx, 'wb') as f:
        pickle.dump(scores, f)
    return scores

def distribute_score_files_reading(path_type, score_out_fn_format):
    num_workers = 10
    pool = mp.Pool(processes=num_workers)    
    results = [
                pool.apply_async(
                        read_stored_file, 
                        args=(path_type, score_out_fn_format, i),
                    ) 
                for i in range(1, 11)
            ]
    results = [p.get() for p in results]
    return
    

def main_calc_metapath_scores_for_one_metapath(
            context_in_fn, 
            metapath_type,
            num_workers,
        ): 
    with open(context_in_fn, 'rb') as f:
        context = pickle.load(f)
    print('Loaded data. Processing pairs.')
    
    out_single_fn_format = 'output/metapath_score/' + metapath_type + '/score%d_%d.pickle'
    distribute_metapath_scores_calculation(context, num_workers, out_single_fn_format)
    print('Completed processing pairs.')


def main_collect_results_for_one_metapath( 
            metapath_type,
            scores_all_out_fn,
            target_score_idx='all',
        ): 
    print('Collecting results.')
    score_out_fn_format = 'output/metapath_score/' + metapath_type + '_%s.pickle'
    distribute_score_files_reading(metapath_type, score_out_fn_format)
    if target_score_idx == 'all':
        scores_all = {}
        for i in range(1, 11):
            with open(score_out_fn_format % str(i), 'rb') as f:
                score_i = pickle.load(f)
                scores_all[i] = score_i[i]
    with open(scores_all_out_fn, 'wb') as f:
        pickle.dump(scores_all, f)


def main_function_for_one_metapath(
            context_in_fn, 
            metapath_type,
            num_workers,
            scores_all_out_fn,
            target_score_idx='all',
        ): 
    main_calc_metapath_scores_for_one_metapath(
            context_in_fn, 
            metapath_type,
            num_workers,
        )
    main_collect_results_for_one_metapath( 
            metapath_type,
            scores_all_out_fn,
            target_score_idx=target_score_idx,
        )

# 2019.1.5
def classic_score(score_idx, c_t1, c_t2, c_t1_t2):
    if score_idx == 1:
        return c_t2 * c_t1_t2
    elif score_idx == 2:
        return c_t2 * c_t1_t2 / c_t1
    elif score_idx == 3:
        return c_t1_t2
    elif score_idx == 4:
        return c_t1_t2 / c_t1
    elif score_idx == 5:
        return c_t1 * c_t1_t2
    elif score_idx == 6:
        return c_t1_t2
    elif score_idx == 7:
        return c_t1 * c_t1_t2 / c_t2
    elif score_idx == 8:
        return c_t1_t2 / c_t2
    elif score_idx == 9:
        #return c_t1_t2 / c_t1 - c_t1_t2 / c_t2
        return c_t1_t2 / c_t2 - c_t1_t2 / c_t1
    else:
        raise NotImplementedError('The score_idx is not implemented.')
                                


def weeds_prec(u, v):
    '''
    u and v are np arrays, representing the vectors of the two words
    '''
    v_gt_0_mask = v>0
    return u[v_gt_0_mask] @ v[v_gt_0_mask] / sum(u)


def cl(u, v):
    '''
    u and v are np arrays, representing the vectors of the two words
    '''
    return sum(np.minimum(u, v)) / sum(u)


def inv_cl(u, v):
    '''
    u and v are np arrays, representing the vectors of the two words 
    '''
    cl_u_v = cl(u, v)
    return (cl_u_v * (1-cl_u_v)) ** 0.5

def roller_scores_sublist_avoid_embedding(
        d, 
        occurrence_cnts, 
        context_dict,
        t1_idx_list,  
        score_type,
        max_num_scores=None, 
        out_fn=None,
    ):
    '''
    @parem d: d[term][context] is the weight of the term in the context
    @parem occurrence_cnts: for any term t, occurrence_cnts[t] is the count of t
                            in the input file (potentially weighted)
    @param context_dict: context_dict[t] is the set of context that t is in
    @param t1_idx_list: a list of indices of target LHS terms
    @param score_type: one of {'weeds_prec', 'a_pinc', 'cl', 'inv_cl'}
    @param max_num_scores: if None, keep all calculated scores; otherwise, maintain a 
                        priority queue of at most max_num_scores highest scores
    @param out_fn: (optional) where to output the scores
    
    @return scores: scores[(t1, t2)] is the score for the pair (t1, t2)
    '''
    scores = []
    cnt = 0
    keyword_list = list(d.keys())
    

    
    # 2019.1.28
    idx2words = get_idx2words()
    pairs = [('data_mining', 'literature_mining'),
             ('data_mining', 'graph_mining'),
             ('data_mining', 'frequent_pattern_mining'),
             ('data_mining', 'association_rule_mining'),
             ('data_mining', 'sequential_pattern'),
             ('learning_algorithm', 'supervised_learning'),
             ('learning_algorithm', 'unsupervised_learning'),
             ('learning_algorithm', 'semi_supervised_learning'),
             ('learning_algorithm', 'reinforcement_learning')
             ]
    
    for t1_idx in t1_idx_list:
        if t1_idx < len(keyword_list):
            t1 = keyword_list[t1_idx]
            c_t1 = occurrence_cnts[t1]
            for t2 in keyword_list:
                if t1 != t2:
                    context_intersection = context_dict[t1].intersection(context_dict[t2])
                    val = 0
                    if score_type == 'weeds_prec':
                        numerator = sum([d[t1][c] for c in context_intersection])
                        if c_t1 != 0 and numerator != 0:
                            val = numerator / c_t1
                    elif score_type == 'cl':
                        numerator = sum([min(d[t1][c], d[t2][c]) for c in context_intersection])
                        if c_t1 != 0 and numerator != 0:
                            val = numerator / c_t1
                    elif score_type == 'inv_cl':
                        numerator = sum([min(d[t1][c], d[t2][c]) for c in context_intersection])
                        c_t2 = occurrence_cnts[t2]
                        if c_t1 != 0 and c_t2 != 0 and numerator != 0:
                            cl_u_v = numerator / c_t1
                            cl_v_u = numerator / c_t2
                            val = (cl_u_v * (1-cl_v_u)) ** 0.5
                    else:
                        raise NotImplementedError('score_type not implemented.')
                    
                    t1_w = idx2words[t1[2:]]
                    t2_w = idx2words[t2[2:]]
                    if (t1_w, t2_w) in pairs or (t2_w, t1_w) in pairs:
                        print('case:{} {} {}'.format(t2_w, t1_w, val))
                    
                    if val != 0:
                        if max_num_scores == None:
                            scores.append((val, t2, t1)) # reverse
                        else:
                            if len(scores) == 0 or val > scores[0][0]:
                                heapq.heappush(scores, (val, t2, t1)) # reverse
                            while len(scores) > max_num_scores:
                                heapq.heappop(scores)
                cnt += 1
                if cnt % 100000000 == 0:
                    print('\t%s processed %d pairs'% (mp.current_process().name, cnt))
                elif cnt % 10000000 == 0:
                    print('\t\t%s processed %d pairs'% (mp.current_process().name, cnt))
    
    # save results to file
    if out_fn != None:
        with open(out_fn, 'wt') as out_f:
            for (val, t1, t2) in scores:
                out_f.write(t1+'\t'+t2+'\t'+str(val)+'\n')
        print("%s saved results at %s" % (mp.current_process().name, out_fn))
    
    print('Process %s finished weeds_prec_sublist_avoid_embedding.' % mp.current_process().name)


def read_weight_file(fn):
    print('Reading weight file')
    d = {}
    type2nodes = []
    cnt = 0
    with open(fn) as f:
        for line in f:
            line_sp = line.split('\t')
            type1node = line_sp[0]
            type2node = line_sp[1]
            weight = float(line_sp[2])
            if type1node not in d:
                d[type1node] = {}
            if type2node not in d[type1node]:
                d[type1node][type2node] = 0
            d[type1node][type2node] += weight
            type2nodes.append(type2node)  
            cnt += 1
            if cnt % 1000000 == 0:
                print('\t%s processed %d lines'% (mp.current_process().name, cnt))
            elif cnt % 100000 == 0:
                print('\t\t%s processed %d lines'% (mp.current_process().name, cnt))
    return d, type2nodes

def calc_one_embedding(d, type2nodes, t):
    dim = len(type2nodes)
    v = np.zeros(dim)
    for type2node in d[t]:
        v[type2nodes.index(type2node)] = d[t][type2node]
    return v


def create_embedding(d, type2nodes):
    '''
    d is a 2-level dict
    '''
    print('Start creating embeddings.')
    emb = {}
    dim = len(type2nodes)
    cnt = 0
    for type1node in d:
        emb[type1node] = np.zeros(dim)
        for type2node in d[type1node]:
            emb[type1node][type2nodes.index(type2node)] = d[type1node][type2node]
        cnt += 1
        if cnt % 10000 == 0:
            print('\t%s processed %d words'% (mp.current_process().name, cnt))
        elif cnt % 1000 == 0:
            print('\t\t%s processed %d words'% (mp.current_process().name, cnt))
    return emb

def convert_pickle_score_to_tsv(in_fn, out_fn_prefix, score_indices):
    with open(in_fn, 'rb') as in_f:
        scores_all = pickle.load(in_f)
    for i in score_indices:
        out_fn = out_fn_prefix + '_score_' + str(i) + '.tsv'
        with open(out_fn, 'wt') as out_f:
            for (t1, t2) in scores_all[i]:
                out_f.write(t1+'\t'+t2+'\t'+str(scores_all[i][(t1, t2)])+'\n')
        print('Done writing to %s' % out_fn)

def convert_pickle_one_group_score_to_tsv(in_fns, out_fn):
    # clear the output file 
    with open(out_fn, 'wt') as out_f:
        pass
    for in_fn in in_fns:
        if os.path.isfile(in_fn):
            with open(in_fn, 'rb') as in_f:
                try:
                    score = pickle.load(in_f)
                except:
                    print('\tFailed to load from %s' % in_f)
                with open(out_fn, 'at') as out_f:
                    for (t1, t2) in score:
                        out_f.write(t1+'\t'+t2+'\t'+str(score[(t1, t2)])+'\n')
        else:
             print('\t%s does not exist' % in_fn)
        print('\tDone converting %s' % in_fn)
    print('Done writing to %s' % out_fn)

def convert_pickle_all_group_score_to_tsv(
        in_fn_prefix, 
        score_indices, 
        num_files_each_score, 
        out_fn_prefix,
    ):
   for score_idx in score_indices:
       in_fns = [(in_fn_prefix+'%d_%d.pickle' % (score_idx, i)) for i in range(1, num_files_each_score+1)]
       out_fn = out_fn_prefix + '_%d.tsv' % score_idx
       convert_pickle_one_group_score_to_tsv(in_fns, out_fn)

# Weighted
def calc_t2_total_weight(d):
    d2 = {}
    for t1 in d:
        for t2 in d[t1]:
            if t2 not in d2:
                d2[t2] = 0
            d2[t2] += 1
    return d2

def get_context_dict(d):
    context_dict = {}
    for t in d:
        context_dict[t] = {c for c in d[t]}
    return context_dict

def calc_classic_dih_score_subset(in_fn, subset_terms, score_idx, out_fn=None):
    '''
    @param in_fn: the input file containing weights
    @param subset_terms: a set of terms
    @param score_idx: which score (1-10) to use
    @param out_fn: (optional) where to output the scores
    @return scores: scores[(t1, t2)] is the score for the pair (t1, t2)
    '''
    d, type2nodes = read_weight_file(in_fn)
    context_dict = get_context_dict(d)
    scores = {}
    cnt = 0
    
    # Case 1: terms in the subset are the LHS terms
    for t1 in subset_terms:
        if t1 in d:
            c_t1 = sum([d[t1][c] for c in d[t1]])
            for t2 in d.keys():
                if t1 != t2:
                    c_t2 = sum([d[t2][c] for c in d[t2]])
                    context_intersection = context_dict[t1].intersection(context_dict[t2])
                    #TODO: revise for weighted
                    c_t1_t2 = len(context_intersection)
                    if c_t1 != 0 and c_t2 != 0:
                        val= classic_score(score_idx, c_t1, c_t2, c_t1_t2)
                        if val != 0:
                            scores[(t1, t2)] = val
                cnt += 1
                if cnt % 10000 == 0:
                    print('\t%s processed %d pairs'% (mp.current_process().name, cnt))
                elif cnt % 1000 == 0:
                    print('\t\t%s processed %d pairs'% (mp.current_process().name, cnt))
        else:
            print('Node 1 (%s) is not in the input file.' % t1)
    
    # Case 2: terms in the subset are the RHS terms
    for t2 in subset_terms:
        if t2 in d:
            c_t2 = sum([d[t2][c] for c in d[t2]])
            for t1 in d.keys():
                c_t1 = sum([d[t1][c] for c in d[t1]])
                if t1 != t2:
                    context_intersection = context_dict[t1].intersection(context_dict[t2])
                    #TODO: revise for weighted
                    c_t1_t2 = len(context_intersection)
                    if c_t1 != 0 and c_t2 != 0:
                        val= classic_score(score_idx, c_t1, c_t2, c_t1_t2)
                        if val != 0:
                            scores[(t1, t2)] = val
                cnt += 1
                if cnt % 100000 == 0:
                    print('\t%s processed %d pairs'% (mp.current_process().name, cnt))
                elif cnt % 10000 == 0:
                    print('\t\t%s processed %d pairs'% (mp.current_process().name, cnt))
        else:
            print('Node 2 (%s) is not in the input file.' % t2)
    
    # save results to file
    if out_fn != None:
        with open(out_fn, 'wt') as out_f:
            for (t1, t2) in scores:
                out_f.write(t1+'\t'+t2+'\t'+str(scores[(t1, t2)])+'\n')
    print("%s saved results" % (mp.current_process().name))
    
    return scores

def find_highest_scores(score_fn, target_word, num_highest):
    d, type2nodes = read_weight_file(score_fn)
    print('number of relevant words: %d' % len(d))
    
    t2_words = list(d[target_word].keys())
    t2_words_sorted = sorted(t2_words, key=lambda t2: d[target_word][t2], reverse=True)
    print('Printing the `t2`s that give the highest score["%s"][t2]' % target_word)
    for t2 in t2_words_sorted[:num_highest]:
        print(t2 + '\t' + str(d[target_word][t2]))
    print('\t')
    
    t1_words = list(d.keys())
    t1_words.remove(target_word)
    t1_words_sorted = sorted(t1_words, key=lambda t1: d[t1][target_word], reverse=True)
    print('Printing the `t1`s that give the highest score[t1]["%s"]' % target_word)
    for t1 in t1_words_sorted[:num_highest]:
        print(t1 + '\t' + str(d[t1][target_word]))

def calc_classic_dih_score_t1_sublist(
        occurrence_cnts, 
        context_dict,
        t1_idx_list, 
        score_idx, 
        max_num_scores=None, 
        out_fn=None,
    ):
    '''
    @parem occurrence_cnts: for any term t, occurrence_cnts[t] is the count of t
                            in the input file (potentially weighted)
    @param context_dict: context_dict[t] is the set of context that t is in
    @param t1_idx_list: a list of indices of target LHS terms
    @param score_idx: which score (1-10) to use
    @param max_num_scores: if None, keep all calculated scores; otherwise, maintain a 
                        priority queue of at most max_num_scores highest scores
    @param out_fn: (optional) where to output the scores
    
    @return scores: scores[(t1, t2)] is the score for the pair (t1, t2)
    '''
    scores = []
    cnt = 0
    keyword_list = list(occurrence_cnts.keys())
    
    for t1_idx in t1_idx_list:
        if t1_idx < len(keyword_list):
            t1 = keyword_list[t1_idx]
            c_t1 = occurrence_cnts[t1]
            for t2 in keyword_list:
                if t1 != t2:
                    c_t2 = occurrence_cnts[t2]
                    context_intersection = context_dict[t1].intersection(context_dict[t2])
                    #TODO: revise for weighted
                    c_t1_t2 = len(context_intersection)
                    if c_t1 != 0 and c_t2 != 0:
                        val= classic_score(score_idx, c_t1, c_t2, c_t1_t2)
                        if val != 0:
                            if max_num_scores == None:
                                scores.append((val, t1, t2))
                            else:
                                if len(scores) == 0 or val > scores[0][0]:
                                    heapq.heappush(scores, (val, t1, t2))
                                while len(scores) > max_num_scores:
                                    heapq.heappop(scores)
                cnt += 1
                if cnt % 10000000 == 0:
                    print('\t%s processed %d pairs'% (mp.current_process().name, cnt))
                elif cnt % 1000000 == 0:
                    print('\t\t%s processed %d pairs'% (mp.current_process().name, cnt))
    
    # save results to file
    if out_fn != None:
        with open(out_fn, 'wt') as out_f:
            for (val, t1, t2) in scores:
                out_f.write(t1+'\t'+t2+'\t'+str(val)+'\n')
        print("%s saved results at %s" % (mp.current_process().name, out_fn))
    
    print('Process %s finished calc_classic_dih_score_t1_sublist.' % mp.current_process().name)

def distribute_dih_score_calculation(
        d, 
        score_idx, 
        num_workers, 
        out_fn_prefix, 
        max_num_scores_each_thread=None,
    ):
    print('Number of keywords:', len(d))
    occurrence_cnts = {}
    for t in d:
        occurrence_cnts[t] = sum([d[t][c] for c in d[t]])
    
    context_dict = get_context_dict(d)
    
    num_pairs = len(d) * (len(d) - 1) // 2
    print('num_pairs:', num_pairs)
    
    pool = mp.Pool(processes=num_workers)
    batch_size = int(math.floor(len(d)/(num_workers-1)))
    print("batch_size: %d" % batch_size)
    start_pos = [i * batch_size for i in range(0, num_workers)]
    if max_num_scores_each_thread == '100 times':
        max_num_scores_each_thread = 100 * len(d)
    if max_num_scores_each_thread == '200 times':
        max_num_scores_each_thread = 200 * len(d)
    results = [
                pool.apply_async(
                        calc_classic_dih_score_t1_sublist, 
                        args=(
                                occurrence_cnts, 
                                context_dict,
                                list(range(start, start+batch_size)), 
                                score_idx,
                                max_num_scores_each_thread,  # max_num_scores
                                '%s_%d.tsv' % (out_fn_prefix, i),  # out_fn
                            ),
                    ) 
                for i, start in enumerate(start_pos)
            ]
    results = [p.get() for p in results]

def normalize_score_file(in_fn, out_fn):
    print('Start converting %s to %s' % (in_fn, out_fn))
    cnt = 0
    with open(in_fn) as in_f, \
    open(out_fn, 'wt') as out_f:
        for line in in_f:
            line_sp = line.split('\t')
            type1node = line_sp[0]
            type2node = line_sp[1]
            score = line_sp[2]
            type1node_n = normalize(type1node)
            type2node_n = normalize(type2node)
            out_f.write(type1node_n+'\t'+type2node_n+'\t'+score)
            cnt += 1
            if cnt % 10000000 == 0:
                    print('%s processed %d pairs' % (mp.current_process().name, cnt))
            elif cnt % 1000000 == 0:
                print('\t%s processed %d pairs'% (mp.current_process().name, cnt))
    print('Completed converting %s to %s' % (in_fn, out_fn))

def investigate_example_pairs(pairs, score_fn):
    d, type2nodes = read_weight_file(score_fn)
    for (t1, t2) in pairs:
        # parent - child score
        if t1 in d:
            if t2 in d[t1]:
                print(t1, t2, d[t1][t2])
            else:
                print('%s not in d[%s]' % (t2, t1))
        else:
            print('%s not in d' % (t1))
        # child - parent score
        if t2 in d:
            if t1 in d[t2]:
                print(t2, t1, d[t2][t1])
            else:
                print('%s not in d[%s]' % (t1, t2))
        else:
            print('%s not in d' % (t2))

def calc_embedding_score_t1_sublist(
        emb,
        t1_idx_list, 
        score_func, 
        max_num_scores=None, 
        out_fn=None,
    ):
    '''
    @parem emb: for any term t, emb[t] is an np array, representing the embedding of t
    @param t1_idx_list: a list of indices of target LHS terms
    @param score_func: score_func(u, v)
    @param max_num_scores: if None, keep all calculated scores; otherwise, maintain a 
                        priority queue of at most max_num_scores highest scores
    @param out_fn: (optional) where to output the scores
    
    @return scores: scores[(t1, t2)] is the score for the pair (t1, t2)
    '''
    scores = []
    cnt = 0
    keyword_list = list(emb.keys())
    
    for t1_idx in t1_idx_list:
        if t1_idx < len(keyword_list):
            t1 = keyword_list[t1_idx]
            v1 = emb[t1]
            for t2 in keyword_list:
                if t1 != t2:
                    v2 = emb[t2]
                    val= score_func(v1, v2)
                    if val != 0:
                        if max_num_scores == None:
                            scores.append((val, t1, t2))
                        else:
                            if len(scores) == 0 or val > scores[0][0]:
                                heapq.heappush(scores, (val, t1, t2))
                            while len(scores) > max_num_scores:
                                heapq.heappop(scores)
                cnt += 1
                if cnt % 1000000 == 0:
                    print('\t%s processed %d pairs'% (mp.current_process().name, cnt))
                elif cnt % 100000 == 0:
                    print('\t\t%s processed %d pairs'% (mp.current_process().name, cnt))
    
    # save results to file
    if out_fn != None:
        with open(out_fn, 'wt') as out_f:
            for (val, t1, t2) in scores:
                out_f.write(t1+'\t'+t2+'\t'+str(val)+'\n')
        print("%s saved results at %s" % (mp.current_process().name, out_fn))
    
    print('Process %s finished calc_embedding_score_t1_sublist.' % mp.current_process().name)

def calc_embedding_score_t1_sublist_embedding_on_the_fly(
        d,
        type2nodes,
        t1_idx_list, 
        score_func, 
        max_num_scores=None, 
        out_fn=None,
    ):
    '''
    Support passing in a function that calculates the embeddings on the fly
    
    @parem d: d[term][context] is the weight of the term in the context
    @parem type2nodes: a list of context nodes in the input 
    @param t1_idx_list: a list of indices of target LHS terms
    @param score_func: score_func(u, v)
    @param max_num_scores: if None, keep all calculated scores; otherwise, maintain a 
                        priority queue of at most max_num_scores highest scores
    @param out_fn: (optional) where to output the scores
    
    @return scores: scores[(t1, t2)] is the score for the pair (t1, t2)
    '''
    print('%s started calc_embedding_score_t1_sublist_embedding_on_the_fly' % mp.current_process().name)
    scores = []
    cnt = 0
    keyword_list = list(d.keys())
    
    for t1_idx in t1_idx_list:
        if t1_idx < len(keyword_list):
            t1 = keyword_list[t1_idx]
            v1 = calc_one_embedding(d, type2nodes, t1)
            for t2 in keyword_list:
                if t1 != t2:
                    v2 = calc_one_embedding(d, type2nodes, t2)
                    val= score_func(v1, v2)
                    if val != 0:
                        if max_num_scores == None:
                            scores.append((val, t1, t2))
                        else:
                            if len(scores) == 0 or val > scores[0][0]:
                                heapq.heappush(scores, (val, t1, t2))
                            while len(scores) > max_num_scores:
                                heapq.heappop(scores)
                cnt += 1
                if cnt % 10000 == 0:
                    print('\t%s processed %d pairs'% (mp.current_process().name, cnt))
                elif cnt % 1000 == 0:
                    print('\t\t%s processed %d pairs'% (mp.current_process().name, cnt))
    
    # save results to file
    if out_fn != None:
        with open(out_fn, 'wt') as out_f:
            for (val, t1, t2) in scores:
                out_f.write(t1+'\t'+t2+'\t'+str(val)+'\n')
        print("%s saved results at %s" % (mp.current_process().name, out_fn))
    
    print('Process %s finished calc_embedding_score_t1_sublist.' % mp.current_process().name)


def distribute_embedding_score_calculation(
        emb, 
        score_func, 
        num_workers, 
        out_fn_prefix, 
        max_num_scores_each_thread=None,
    ):
    print('Number of keywords:', len(emb))    
    num_pairs = len(emb) * (len(emb) - 1) // 2
    print('num_pairs:', num_pairs)
    
    pool = mp.Pool(processes=num_workers)
    batch_size = int(math.floor(len(emb)/(num_workers-1)))
    print("batch_size: %d" % batch_size)
    start_pos = [i * batch_size for i in range(0, num_workers)]
    if max_num_scores_each_thread == '100 times':
        max_num_scores_each_thread = 100 * len(emb)
    results = [
                pool.apply_async(
                        calc_embedding_score_t1_sublist, 
                        args=(
                                emb, 
                                list(range(start, start+batch_size)), 
                                score_func,
                                max_num_scores_each_thread,  # max_num_scores
                                '%s_%d.tsv' % (out_fn_prefix, i),  # out_fn
                            ),
                    ) 
                for i, start in enumerate(start_pos)
            ]
    results = [p.get() for p in results]

def distribute_embedding_score_calculation_embedding_on_the_fly(
        d,
        type2nodes,
        score_func, 
        num_workers, 
        out_fn_prefix, 
        max_num_scores_each_thread=None,
    ):
    print('Number of keywords:', len(d))    
    num_pairs = len(d) * (len(d) - 1) // 2
    print('num_pairs:', num_pairs)
    
    pool = mp.Pool(processes=num_workers)
    batch_size = int(math.floor(len(d)/(num_workers-1)))
    print("batch_size: %d" % batch_size)
    start_pos = [i * batch_size for i in range(0, num_workers)]
    if max_num_scores_each_thread == '100 times':
        max_num_scores_each_thread = 100 * len(d)
    results = [
                pool.apply_async(
                        calc_embedding_score_t1_sublist_embedding_on_the_fly, 
                        args=(
                                d, 
                                type2nodes,
                                list(range(start, start+batch_size)), 
                                score_func,
                                max_num_scores_each_thread,  # max_num_scores
                                '%s_%d.tsv' % (out_fn_prefix, i),  # out_fn
                            ),
                    ) 
                for i, start in enumerate(start_pos)
            ]
    results = [p.get() for p in results]

def distribute_roller_scores_calculation(
        d, 
        score_type, 
        num_workers, 
        out_fn_prefix, 
        max_num_scores_each_thread=None,
    ):
    print('Number of keywords:', len(d))
    context_dict = get_context_dict(d)
    num_pairs = len(d) * (len(d) - 1) // 2
    print('num_pairs:', num_pairs)
    
    occurrence_cnts = {}
    for t in d:
        occurrence_cnts[t] = sum([d[t][c] for c in d[t]])
    
    pool = mp.Pool(processes=num_workers)
    batch_size = int(math.floor(len(d)/(num_workers-1)))
    print("batch_size: %d" % batch_size)
    start_pos = [i * batch_size for i in range(0, num_workers)]
    if max_num_scores_each_thread == '1000 times':
        max_num_scores_each_thread = 1000 * len(d)
    if max_num_scores_each_thread == '200 times':
        max_num_scores_each_thread = 200 * len(d)
    results = [
                pool.apply_async(
                        roller_scores_sublist_avoid_embedding, 
                        args=(
                                d, 
                                occurrence_cnts,
                                context_dict,
                                list(range(start, start+batch_size)), 
                                score_type,
                                max_num_scores_each_thread,  # max_num_scores
                                '%s_%d.tsv' % (out_fn_prefix, i),  # out_fn
                            ),
                    ) 
                for i, start in enumerate(start_pos)
            ]
    results = [p.get() for p in results]

def collect_all_scores(fns, out_fn):
    cnt = 0
    with open(out_fn, 'wt') as out_f:
        for fn in fns:
            with open(fn) as f:
                for line in f:
                    out_f.write(line)
                    cnt += 1
                    if cnt % 1000000 == 0:
                        print('\t%s processed %d lines'% (mp.current_process().name, cnt))
                    elif cnt % 100000 == 0:
                        print('\t\t%s processed %d lines'% (mp.current_process().name, cnt))
    print("%s saved results at %s" % (mp.current_process().name, out_fn))

def collect_high_scores(fns, out_fn, max_num_scores):
    score_t1_t2_tuples = []
    cnt = 0
    for fn in fns:
        with open(fn) as f:
            for line in f:
                line_sp = line.split('\t')
                t1 = line_sp[0]
                t2 = line_sp[1]
                score = float(line_sp[2])
                if len(score_t1_t2_tuples)==0 or score > score_t1_t2_tuples[0][0]:
                    heapq.heappush(score_t1_t2_tuples, (score, t1, t2))
                    while len(score_t1_t2_tuples) > max_num_scores:
                        heapq.heappop(score_t1_t2_tuples)
                cnt += 1
                if cnt % 1000000 == 0:
                    print('\t%s processed %d lines'% (mp.current_process().name, cnt))
                elif cnt % 100000 == 0:
                    print('\t\t%s processed %d lines'% (mp.current_process().name, cnt))
    cnt = 0
    with open(out_fn, 'wt') as out_f:
        for (score, t1, t2) in score_t1_t2_tuples:
            out_f.write(t1+'\t'+t2+'\t'+str(score)+'\n')
            if cnt % 100000 == 0:
                print('\t%s wrote %d lines'% (mp.current_process().name, cnt))
            elif cnt % 10000 == 0:
                print('\t\t%s wrote %d lines'% (mp.current_process().name, cnt))
    print("%s saved results at %s" % (mp.current_process().name, out_fn))

def convert_context_dict_to_weight_file(context_dict, out_fn):
    cnt = 0
    with open(out_fn, 'wt') as out_f:
        for t in context_dict:
            for c in context_dict[t]:
                out_f.write(t+'\t'+c+'\t'+'1'+'\n')
            cnt += 1
            if cnt % 10000 == 0:
                print('\t%s processed %d keys'% (mp.current_process().name, cnt))
            elif cnt % 1000 == 0:
                print('\t\t%s processed %d keys'% (mp.current_process().name, cnt))
            


'''
Main
'''    

if __name__ == '__main__':

    num_clusters = 1000
    cluster_type = 'temp_tpcer'
    in_fn = '/shared/data/nzhang31/linkedin_maple/codes/output/metapath_context/{}_{}_context.tsv'.format(cluster_type, num_clusters)  # modify as needed
    d, type2nodes = read_weight_file(in_fn)
    score_type = 'weeds_prec'  # modify as needed {'weeds_prec', 'cl', 'inv_cl'}
    num_workers = 30
    out_fn_prefix = '/shared/data/nzhang31/linkedin_maple/codes/output/metapath_score/{}_{}_{}/'.format(cluster_type, num_clusters, score_type)  # modify as needed
    max_num_scores_each_thread = '200 times'
    distribute_roller_scores_calculation(
        d, 
        score_type, 
        num_workers, 
        out_fn_prefix, 
        max_num_scores_each_thread=max_num_scores_each_thread,
    )
    