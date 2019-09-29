import argparse
import os
import torch
from utils import print_args
from datetime import datetime


def read_options():
    parser = argparse.ArgumentParser(description='synonym (set) discovery')

    # Data parameters
    # parser.add_argument('-dataset', default="DPE-wiki", type=str, help='name of the dataset [default: 20ng]')
    # parser.add_argument('-data-format', default="set", type=str, choices=['set', 'sip'],
    #                     help='format of input training dataset')
    # parser.add_argument('-embed-file', default="/export/home/sjiaming/hdcg/heer_linkedin_0.1_0_op_1_mode_0_rescale_0.1_lr_10_lrr_10.emb",
    #                     type=str, help='name of the pretrained node embedding')
    # parser.add_argument('-embed-file',
    #                     default="/export/home/sjiaming/hdcg/heer_skill_5k.emb",
    #                     type=str, help='name of the pretrained node embedding')
    parser.add_argument('-embed-file', default="/shared/data/jiaming/linkedin-maple/maple/data/node.embed", type=str,
                        help='name of the pretrained node embedding')
    # parser.add_argument('-pair-feature-prefix', default="/export/home/sjiaming/hdcg/", type=str,
    #                     help='name of the pair features')
    parser.add_argument('-pair-feature-prefix', default="/shared/data/jiaming/linkedin-maple/maple/data/", type=str,
                        help='name of the pair features')
    # parser.add_argument('-supervision-file',
    #                     default="/export/home/sjiaming/hdcg/sup_pairs_5k.tsv", type=str,
    #                     help='name of the supervision/train pair file')
    parser.add_argument('-supervision-file',
                        default="/shared/data/jiaming/linkedin-maple/maple/data/supervision_pairs2.txt",
                        type=str, help='name of the supervision/train pair file')
    # parser.add_argument('-test-pairs-file',
    #                     default="/export/home/sjiaming/hdcg/eval_pair_all_5k.tsv", type=str,
    #                     help='name of the test pair file (used for evaluation)')
    parser.add_argument('-test-pairs-file',
                        default="/shared/data/jiaming/linkedin-maple/maple/data/test_ancestor2d_pairs2.txt",
                        type=str, help='name of the test pair file (used for evaluation)')
    # parser.add_argument('-pred-pairs-file-in',
    #                     default="/export/home/sjiaming/hdcg/eval_pair_all_5k.tsv", type=str,
    #                     help='name of the term pair file for prediction')
    parser.add_argument('-pred-pairs-file-in',
                        default="/shared/data/jiaming/linkedin-maple/maple/data/test_ancestor2d_pairs2.txt", type=str,
                        help='name of the term pair file for prediction')
    # parser.add_argument('-pred-pairs-file-out',
    #                     default="/export/home/sjiaming/hdcg/eval_pair_all_5k.res.tsv",
    #                     type=str, help='name of the term pair file for prediction results')
    parser.add_argument('-pred-pairs-file-out',
                        default="/shared/data/jiaming/linkedin-maple/maple/output/test_ancestor2d_pairs.res.txt",
                        type=str, help='name of the term pair file for prediction results')
    parser.add_argument('-feature-index', default="", type=str,
                        help='feature index')

    # Synonym Pair Prediction Model parameters
    parser.add_argument('-h1', default=100, type=int, help='first layer hidden size pointNet transformer')
    parser.add_argument('-h2', default=10, type=int, help='first layer hidden size pointNet transformer')

    # Model parameters
    parser.add_argument('-modelName', default='np_lrlr_sd_lrlrdl', type=str,
                        help='which prediction model is used')
    parser.add_argument('-pretrained-embedding', default='embed', type=str,
                        choices=['none', 'embed', 'tfidf', 'fastText-no-subword.embed', 'fastText-with-subword.embed'],
                        help='whether to use pretrained embedding, none means training embedding from scratch')
    parser.add_argument('-embed-fine-tune', default=0, type=int,
                        help='fine tune word embedding or not, 0 means no fine tune')
    parser.add_argument('-embedSize', default=128, type=int, help='embed size for words')
    parser.add_argument('-node-hiddenSize', default=300, type=int, help='hidden size used in node post_embedder')
    parser.add_argument('-edge-hiddenSize', default=50, type=int, help='hidden size used in edge post_embedder')
    parser.add_argument('-combine-hiddenSize', default=600, type=int, help='hidden size used in combiner')
    parser.add_argument('-string-pair-feature-dimension', default=20, type=int,
                        help='additional string feature dimension')
    parser.add_argument('-use-pair-feature', default=0, type=int, help='use string pair feature or not')
    parser.add_argument('-pair-feature-dim', default=28, type=int, help='use string pair feature or not')
    parser.add_argument('-pt-name', default="", type=str, help='name of pair feature transformer')
    parser.add_argument('-mode', default='train', type=str,
                        choices=['train', 'eval', 'set_predict', 'cluster_predict', 'tune_CV', 'tune', 'debug'],
                        help='specify model running mode')

    # Learning options
    parser.add_argument('-batch-size', default=5, type=int, help='batch size for training')
    parser.add_argument('-max-set-size', default=100, type=int, help='maximum size for training batch')
    parser.add_argument('-cold-start-epochs', default=0, type=int,
                        help='number of training epoch used for cold-start training [default 10 epochs]')

    parser.add_argument('-lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('-weight-decay', default=0.0, type=float, help='l2 regularization')
    parser.add_argument('-loss-fn', default="self_margin_rank_bce", type=str,
                        choices=['cross_entropy', 'max_margin', 'margin_rank', 'self_margin_rank',
                                 'self_margin_rank_bce'],
                        help='loss function used in training model [Default: cross entropy loss]')
    parser.add_argument('-margin', default=1.0, type=float, help='margin used in max_margin loss and margin_rank loss')
    parser.add_argument('-epochs', default=500, type=int, help='number of epochs for training')
    parser.add_argument('-neg-sample-size', default=20, type=int,
                        help='number of negative samples generated for each set')
    parser.add_argument('-neg-sample-method', default="complete_random", type=str,
                        choices=["complete_random", "share_token", "mixture"], help='negative sampling method')
    parser.add_argument('-use-fullset-batch', default=0.0, type=float,
                        help='generate fullset batch based on leave-nothing-out principle or not')
    parser.add_argument('-use-noisyset-batch', default=0.0, type=float,
                        help='generate noisy batch based on add-noise-in principle or not')
    parser.add_argument('-dropout', default=0.4, type=float, help='Dropout between layers')
    parser.add_argument('-edge-dropout', default=0.5, type=float, help='Dropout between layers')
    parser.add_argument('-node-dropout', default=0.5, type=float, help='Dropout between layers')
    parser.add_argument('-early-stop', default=100, type=int, help='early stop epoch number')
    parser.add_argument('-eval-epoch', default=5, type=int, help='average number of epochs for evaluation')
    parser.add_argument('-random-seed', default=5417, type=int,
                        help='random seed used for model initialization and negative sample generation')
    parser.add_argument('-size-opt-clus', default=0, type=int,
                        help='whether conduct size optimized clustering prediction'
                             'this saves GPU memory sizes but consumes more training time)'
                             'when expecting a large number of small sets, set this option to be 0;'
                             'when expecting a small number of huge sets, set this option to be 1')
    parser.add_argument('-max-clus-num', default=-1, type=int, help='maximum cluster number, -1 means auto-infer')
    parser.add_argument('-max-K', default=-1, type=int, help='maximum cluster number, -1 means auto-infer')
    parser.add_argument('-T', default=1.0, type=int, help='temperature scaling, 1.0 means no scaling')
    parser.add_argument('-sip-suffix', default="", type=str, help='suffix for sip negative training')

    # Device options
    parser.add_argument('-device-id', default=-1, type=int, help='device to use for iterate data, -1 means cpu')

    # Model saving/loading options
    parser.add_argument('-save-dir', default="./snapshots/", type=str, help='location to save models')
    parser.add_argument('-load-model', default="", type=str, help='path to loaded model')
    parser.add_argument('-snapshot', default="", type=str, help='path to model snapshot')
    parser.add_argument('-tune-result-file', default="tune_prefix", type=str, help='path to save all tuning results')

    # Other options
    parser.add_argument('-remark', default='', help='reminder of this run')
    parser.add_argument('-exp-id', default='-1', type=int, help='experiment id, -1 means random hyper-parameter search')
    parser.add_argument('-f', type=str, default="", help='placeholder for debugging in IPython')

    # pair-prediction model
    parser.add_argument('-semantic-class-name', default='us_states', help='name of semantic class')
    parser.add_argument('-supervision-source', default='train-cold.set', help='name of supervision sources')

    try:
        args = parser.parse_args()
        print_args(args)
    except:
        parser.error("Unable to parse arguments")

    # Update Device information
    if args.device_id == -1:
        args.device = torch.device("cpu")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
        args.device = torch.device("cuda:0")

    # Update Tensorboard logging
    if args.mode == "train":
        args.comment = '_{}'.format(args.remark)
    elif args.mode == "tune":
        args.comment = "_{}".format(args.tune_result_file)
    else:
        args.comment = ""

    # Model snapshot saving
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    if args.mode == "train":
        args.save_dir = os.path.join(args.save_dir, current_time + '_' + args.remark)
    elif args.mode == "tune":
        args.save_dir = os.path.join(args.save_dir, current_time + '_' + args.tune_result_file)

    if args.max_K == -1:
        args.max_K = None

    args.use_pair_feature = (args.use_pair_feature != 0)

    return args


