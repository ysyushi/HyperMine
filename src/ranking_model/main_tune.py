import itertools
from utils import load_embedding, Results, Metrics, save_model, load_model, load_directional_supervision, load_element_pairs, load_pair_features
from model import PointNet, PairNet, TripletNet, PairNetWithPairFeatures, TripletNetWithPairFeatures
from trainer import train_triplet_epoch
from datasets import Triplets, DirectionalTriplets, DirectionalTripletsWithPairFeature
from options import read_options
import numpy as np
import torch
import random
from evaluator import evaluate_synonym_ranking_prediction, evaluation_main
from losses import TripletLoss
from tensorboardX import SummaryWriter
from test_semantic_classes import obtain_semantic_classes
from predictor import pair_prediction, pair_prediction_with_pair_feature
from sklearn.model_selection import ParameterSampler, ParameterGrid


def sample_hyperparameters(exp_id=0, num=100, random_state=None):
    """ Sample hyperparameters

    :param exp_id: 0 means random search, otherwise, search according to exp_id
    :return:
    """

    if exp_id == 0:
        space = {
            'h1': [64, 128],
            'h2': [32, 64],
            'modelName': ["ltdl_bilinearDIAG+PTF"],
            'batch_size': [50, 100],
            'lr': [0.1, 0.01, 0.001],
            'pt_name': ['ld', 'd'],
            'edge_dropout': [0.1, 0.3, 0.5, 0.7],
            'node_dropout': [0.1, 0.3, 0.5, 0.7],
            'epochs': [200]
        }

        sampler = ParameterSampler(space, n_iter=num, random_state=random_state)

        for params in sampler:
            yield params


def run(train_loader, test_pairs, options):
    # Construct model
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
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=options["lr"], weight_decay=options["weight_decay"])
    loss_fn = TripletLoss(options["margin"])

    best_overall_metric = 0  # a single value
    best_metrics = None  # a dictionary
    best_epoch = 0
    save_model(model, options["save_dir"], 'best', 0)  # save the initial first model
    for epoch in range(options["epochs"]):
        epoch_loss, non_zero_triplet_ratio = train_triplet_epoch(train_loader, model, loss_fn, optimizer, cuda,
                                                                 use_pair_feature=options["use_pair_feature"])
        print("Epoch: {}, train-loss: {:06.4f}, non_zero_triplet_ratio: {:06.4f}, ".format(epoch, epoch_loss,
                                                                                          non_zero_triplet_ratio))
        if epoch % options["eval_epoch"] == 0 and epoch != 0:
            if options["use_pair_feature"]:
                prediction_score = pair_prediction_with_pair_feature(model.pair_net, test_pairs, pair_features, cuda, options, batch_size=10000)
            else:
                prediction_score = pair_prediction(model.pair_net, test_pairs, cuda, options, batch_size=10000)
            test_triplets = []
            for term_pair, score in zip(test_pairs, prediction_score):
                test_triplets.append((term_pair[0], term_pair[1], -1.0 * score))
            metrics = evaluation_main(test_triplets)
            if metrics["all"] >= best_overall_metric:
                best_overall_metric =metrics["all"]
                best_epoch = epoch
                best_metrics = metrics
                save_model(model, options["save_dir"], 'best', epoch)  # save the initial first model

    return best_overall_metric, best_epoch, best_metrics


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

    # Construct training set and training data loader
    if options["use_pair_feature"]:
        print("!!! Using pair features")
        f_pair_feature_key = options["pair_feature_prefix"] + "edge.keys3.tsv"
        f_pair_feature_value = options["pair_feature_prefix"] + "edge.values3.scaled.npy"
        pair_features = load_pair_features(f_pair_feature_key, f_pair_feature_value)
        train_data = DirectionalTripletsWithPairFeature(options["embedding"], train_hyper2hypo, pair_features)
    else:
        train_data = DirectionalTriplets(options["embedding"], train_hyper2hypo)
    print("=== Finish constructing dataset ===")
    print("Number of training hyposets: {}".format(len(train_data)))
    kwargs = {'num_workers': 1, 'pin_memory': True} if options["device_id"] != -1 else {}
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=options["batch_size"], shuffle=True,
                                               drop_last=False)

    # Construct testing set
    f_test = options["test_pairs_file"]
    test_pairs = load_element_pairs(f_test, with_label=False)
    print("Number of testing term pairs: {}".format(len(test_pairs)))

    # Start model tunning
    results = Results('./results/tune_{}.txt'.format(args.comment))
    metrics = Metrics()
    for hp in sample_hyperparameters(num=200):
        for m in hp:
            options[m] = hp[m]  # update hyper-parameters

        options["pt"] = {
            "name": hp["pt_name"],
            "dropout": hp["edge_dropout"]
        }

        best_overall_metric, best_epoch, best_metrics = run(train_loader, test_pairs, options)
        metrics.metrics = best_metrics
        results.save_metrics(hp, metrics)
