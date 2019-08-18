class Config:
	some_path_shared_across_the_project = "path"
	vocab = "/shared/data/zlou4/linkedin/maple/data/vocabulary_postlink.txt"

# Additional config for preprocessing
class PreprocessingConfig(Config):
	# to import this path from a script in a child directory, e.g., preprocessing, do the following
	# import sys
	# from os import path
	# sys.path.append(path.dirname(path.dirname(path.abspath(__file__)))) # use the correct number of path.dirname here depending the layers of folder for the current file
	# from config import PreprocessingConfig
	# and then do whatever like you would to a package, e.g., print(PreprocessingConfig.generated_pairs)
	#generated_pairs = "data/ccs/pairs/"
    
    # Text corpus, one paragraph or sentence per line
    # Unnormalized (with punctuations, upper and low cases)
    wiki_linking = '/shared/data/zlou4/linkedin/output/wikilinked_result.txt'
    wiki_page_summaries = '/shared/data/li215/linkedin_maple/codes/maple/data/wiki/wiki_page_summaries.txt'
    dblp_abstracts = '/shared/data/li215/linkedin_maple/codes/data/dblp/dblp_abstracts_unnormalized.txt'
    keyword_spans_dblp = '/shared/data/li215/linkedin_maple/codes/maple/data/supervision/keyword_spans_dblp_abstracts.pickle'
    keyword_spans_wiki = '/shared/data/li215/linkedin_maple/codes/maple/data/supervision/keyword_spans_wiki_page_summaries.pickle'
    sentences_dblp_json = '/shared/data/li215/linkedin_maple/codes/maple/data/supervision/sentences_dblp.json'
    sentences_wiki_json = '/shared/data/li215/linkedin_maple/codes/maple/data/supervision/sentences_wiki.json'
    supervision_pairs_dblp_unnormalized = '/shared/data/li215/linkedin_maple/codes/maple/data/supervision/supervision_pairs_dblp_unnormalized.tsv'
    supervision_all_info_dblp = '/shared/data/li215/linkedin_maple/codes/maple/data/supervision/supervision_all_info_dblp.txt'
    supervision_pairs_dblp_normalized = '/shared/data/li215/linkedin_maple/codes/maple/data/supervision/supervision_pairs_dblp_normalized.tsv'
    supervision_pairs_dblp_normalized_remove_is_a = '/shared/data/li215/linkedin_maple/codes/maple/data/supervision/supervision_pairs_dblp_normalized_remove_is_a.tsv'
    supervision_pairs_dblp_final = '/shared/data/li215/linkedin_maple/codes/maple/data/supervision/supervision_pairs_dblp_final.tsv'
    supervision_pairs_wiki_unnormalized = '/shared/data/li215/linkedin_maple/codes/maple/data/supervision/supervision_pairs_wiki_unnormalized.tsv'
    supervision_all_info_wiki = '/shared/data/li215/linkedin_maple/codes/maple/data/supervision/supervision_all_info_wiki.txt'
    supervision_pairs_wiki_normalized = '/shared/data/li215/linkedin_maple/codes/maple/data/supervision/supervision_pairs_wiki_normalized.tsv'
    supervision_pairs_wiki_normalized_remove_is_a = '/shared/data/li215/linkedin_maple/codes/maple/data/supervision/supervision_pairs_wiki_normalized_remove_is_a.tsv'
    supervision_pairs_wiki_final = '/shared/data/li215/linkedin_maple/codes/maple/data/supervision/supervision_pairs_wiki_final.tsv'
    supervision_pairs_substring = '/shared/data/li215/linkedin_maple/codes/maple/data/supervision/supervision_pairs_substring.tsv'
    embedding_pretrained = '/shared/data/qiz3/text_summ/src/HEER/intermediate_data/pretrained_dblp_tax_actual.emb'
    embedding_heer = '/shared/data/qiz3/text_summ/src/HEER/intermediate_data/heer_dblp_tax_60_op_1_mode_0_rescale_0.1_lr_10_lrr_10_actual.emb'

# Additional config for evaluation
class EvaluationConfig(Config):
	evaluation_pairs_dir = "/shared/data/li215/linkedin_maple/codes/data/ccs/pairs/"
	pos_ancestor_descendant = evaluation_pairs_dir + "positive_ancestor2d_pairs2.txt"
	pos_parent_child = evaluation_pairs_dir + "positive_parent2c_pairs2.txt"
	neg_ancestor_descendant = evaluation_pairs_dir + "negative_ancestor2d_pairs2.txt"
	neg_parent_child = evaluation_pairs_dir + "negative_parent2c_pairs2.txt"

# Metapath contexts
class MetapathContextConfig(Config):
    li215_metapath_context_dir = '/shared/data/li215/linkedin_maple/codes/output/metapath_context/'
    tpa_context = li215_metapath_context_dir + 'tpa_context_normalized.tsv'
    t2p_context = li215_metapath_context_dir + 't2p_context_normalized.tsv'
    tpv_context = li215_metapath_context_dir + 'tpv_context_normalized.tsv'
    tpt_context = li215_metapath_context_dir + 'tpt_context_normalized.tsv'

# Paths to DIH scores
class DIHConfig(Config):
    li215_dih_scores_dir = '/shared/data/li215/linkedin_maple/codes/output/metapath_score/'
    nzhang31_dih_scores_dir = '/shared/data/nzhang31/linkedin_maple/codes/output/metapath_score/'
    t2p_score9 = li215_dih_scores_dir + 't2p_score9_try2_all_normalized.tsv'
    t2p_weeds_prec = li215_dih_scores_dir + 't2p_weeds_prec3.tsv'
    t2p_cl = li215_dih_scores_dir + 't2p_cl.tsv'  # ClarkeDE
    t2p_inv_cl = li215_dih_scores_dir + 't2p_inv_cl.tsv'
    tpt_cl = li215_dih_scores_dir + 'tpt_cl.tsv'  # ClarkeDE
    tpt_inv_cl = li215_dih_scores_dir + 'tpt_inv_cl.tsv'  

    # actually 4 * 6 = 24
    five_dih_scores_six_context_for_nn_list = [
        nzhang31_dih_scores_dir + "temp_tpcer_10000_score6/_total.tsv",
        #nzhang31_dih_scores_dir + "temp_tpcer_10000_score8/_total.tsv",
        nzhang31_dih_scores_dir + "temp_tpcer_10000_score9/_total.tsv",
        nzhang31_dih_scores_dir + "temp_tpcer_10000_weeds_prec/_total.tsv",
        nzhang31_dih_scores_dir + "temp_tpcer_10000_inv_cl/_total.tsv",
        
        nzhang31_dih_scores_dir + "temp_tpcer_100_score6/_total.tsv",
        #nzhang31_dih_scores_dir + "temp_tpcer_100_score8/_total.tsv",
        nzhang31_dih_scores_dir + "temp_tpcer_100_score9/_total.tsv",
        nzhang31_dih_scores_dir + "temp_tpcer_100_weeds_prec/_total.tsv",
        nzhang31_dih_scores_dir + "temp_tpcer_100_inv_cl/_total.tsv",
        
        li215_dih_scores_dir + "tpv/tsv/score6_all_all.tsv",
        #li215_dih_scores_dir + "tpv/tsv/score8_all_all.tsv",
        li215_dih_scores_dir + "tpv/tsv/score9_all_all.tsv",
        #li215_dih_scores_dir + "tpv_cl.tsv",
        nzhang31_dih_scores_dir + "tpv_cl/_.tsv",
        nzhang31_dih_scores_dir + "tpv_inv_cl/_.tsv",

        li215_dih_scores_dir + "tpt_score6_normalized.tsv",
        #li215_dih_scores_dir + "tpt_score8_normalized.tsv",
        li215_dih_scores_dir + "tpt_score9_normalized.tsv",
        tpt_cl,
        tpt_inv_cl,

        li215_dih_scores_dir + "tpa_score6_normalized.tsv",
        #li215_dih_scores_dir + "tpa_score8_normalized.tsv",
        li215_dih_scores_dir + "tpa_score9_normalized.tsv",
        nzhang31_dih_scores_dir + "tpa_cl/_.tsv",
        nzhang31_dih_scores_dir + "tpa_inv_cl/_.tsv",

        li215_dih_scores_dir + "t2p_score6_normalized.tsv",
        #li215_dih_scores_dir + "t2p_score8_normalized.tsv",
        li215_dih_scores_dir + "t2p_score9_try2_all_normalized.tsv", 
        t2p_cl,
        t2p_inv_cl
        ]

    all_dih_scores_for_nn_list = [
        li215_dih_scores_dir + "t2p_score5_normalized.tsv",
        li215_dih_scores_dir + "t2p_score6_normalized.tsv",
        li215_dih_scores_dir + "t2p_score8_normalized.tsv",
        li215_dih_scores_dir + "t2p_score9_normalized.tsv",
        li215_dih_scores_dir + "tpa_score5_normalized.tsv",
        li215_dih_scores_dir + "tpa_score6_normalized.tsv",
        li215_dih_scores_dir + "tpa_score8_normalized.tsv",
        li215_dih_scores_dir + "tpa_score9_normalized.tsv",
        li215_dih_scores_dir + "tpt_score5_normalized.tsv",
        li215_dih_scores_dir + "tpt_score6_normalized.tsv",
        li215_dih_scores_dir + "tpt_score8_normalized.tsv",
        li215_dih_scores_dir + "t2p_score9_try2_all_normalized.tsv", 
        li215_dih_scores_dir + "tpv_score5_all.tsv",
        li215_dih_scores_dir + "tpv/tsv/score6_all_all.tsv",
        li215_dih_scores_dir + "tpv/tsv/score8_all_all.tsv",
        li215_dih_scores_dir + "tpv/tsv/score9_all_all.tsv",
        t2p_weeds_prec,
        t2p_cl,
        t2p_inv_cl,
        tpt_cl,
        tpt_inv_cl,
        li215_dih_scores_dir + "tpv_cl.tsv",
        nzhang31_dih_scores_dir + 'tpcer_10000_score5/1000 times_max_total.tsv',
        nzhang31_dih_scores_dir + 'tpcer_10000_score6/1000 times_max_total.tsv',
        nzhang31_dih_scores_dir + 'tpcer_10000_score7/1000 times_max_total.tsv',
        nzhang31_dih_scores_dir + 'tpcer_10000_score8/1000 times_max_total.tsv',
        # nzhang31_dih_scores_dir + 'tpcer_10000_score9/1000 times_max_total.tsv',
        nzhang31_dih_scores_dir + 'tpcer_10000_weeds_prec/_total.tsv'
        ]
