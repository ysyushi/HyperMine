import itertools
from utils import load_embedding, Results, Metrics, save_model, load_model, load_directional_supervision, load_element_pairs, load_pair_features
from model import PointNet, PairNet, TripletNet, PairNetWithPairFeatures, TripletNetWithPairFeatures
from trainer import train_triplet_epoch
from datasets import Triplets, DirectionalTriplets
from options import read_options
import numpy as np
import torch
import random
from evaluator import evaluate_synonym_ranking_prediction
from losses import TripletLoss
from tensorboardX import SummaryWriter
from test_semantic_classes import obtain_semantic_classes
from predictor import pair_prediction, pair_prediction_with_pair_feature


if __name__ == '__main__':
    args = read_options()

    # Add TensorBoard Writer
    writer = SummaryWriter(log_dir=None, comment=args.comment)

    # Initialize random seed
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if args.device_id != -1:
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_printoptions(precision=9)
    torch.set_num_threads(1)

    # Load command line options
    options = vars(args)
    writer.add_text('Text', 'Hyper-parameters: {}'.format(options), 0)

    # Load supervision pairs and convert to dict
    f_supervision = options["supervision_file"]
    train_hyper2hypo, train_hypo2hyper = load_directional_supervision(f_supervision)

    # Load embedding files and word <-> index map
    f_embed = options["embed_file"]
    embedding, index2word, word2index, vocab_size, embed_dim = load_embedding(f_embed)
    print("=== Finish loading embedding ===")
    options["embedding"] = embedding
    options["index2word"] = index2word
    options["word2index"] = word2index
    options["vocabSize"] = vocab_size
    options["embedSize"] = embed_dim

    # Load pair features
    if options["use_pair_feature"]:
        print("!!! Using pair features")
        f_pair_feature_key = options["pair_feature_prefix"] + "edge.keys.tsv"
        f_pair_feature_value = options["pair_feature_prefix"] + "edge.values.npy"
        pair_features = load_pair_features(f_pair_feature_key, f_pair_feature_value)

    # Construct testing set
    f_pred = options["pred_pairs_file_in"]
    print("!!! Loading term pairs for prediction from: {}".format(f_pred))
    pred_pairs = load_element_pairs(f_pred, with_label=False)
    print("Number of term pairs for prediction: {}".format(len(pred_pairs)))

    # Construct model skeleton
    cuda = options["device_id"] != -1
    if options["use_pair_feature"]:
        point_net = PointNet(options)
        pair_net = PairNetWithPairFeatures(options, point_net)
        model = TripletNetWithPairFeatures(pair_net)
    else:
        point_net = PointNet(options)
        pair_net = PairNet(options, point_net)
        model = TripletNet(pair_net)
    if cuda:
        model.cuda()

    # Load pre-trained model
    model_path = options["snapshot"]
    model.load_state_dict(torch.load(model_path))
    print(model)

    # Conduct pair prediction and dump results to file
    if options["use_pair_feature"]:
        prediction_scores = pair_prediction_with_pair_feature(model.pair_net, pred_pairs, pair_features, cuda, options,
                                                             batch_size=10000)
    else:
        prediction_scores = pair_prediction(model.pair_net, pred_pairs, cuda, options, batch_size=10000)
    f_res = options["pred_pairs_file_out"]
    print("!!! Saving prediction results to: {}".format(f_res))
    with open(f_res, "w") as fout:
        for term_pair, score in zip(pred_pairs, prediction_scores):
            fout.write("{}\t{}\t{}\n".format(term_pair[0], term_pair[1], -1.0 * score))
