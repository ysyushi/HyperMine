import torch
import torch.nn as nn
import zoo
import math


# Initializing weights
def initializeWeights(moduleList, itype):
    assert itype == 'xavier', 'Only Xavier initialization supported'

    for moduleId, module in enumerate(moduleList):
        if hasattr(module, '_modules') and len(module._modules) > 0:
            # Iterate again
            initializeWeights(module, itype)
        else:
            # Initialize weights
            name = type(module).__name__
            # If linear or embedding
            if name == 'Embedding' or name == 'Linear':
                fanIn = module.weight.data.size(0)
                fanOut = module.weight.data.size(1)

                factor = math.sqrt(2.0/(fanIn + fanOut))
                weight = torch.randn(fanIn, fanOut) * factor
                module.weight.data.copy_(weight)

            # Check for bias and reset
            if hasattr(module, 'bias') and hasattr(module.bias, 'data'):
                module.bias.data.fill_(0.0)


def initialize_single_module_weights(module):
    name = type(module).__name__
    # If linear or embedding
    if name == 'Embedding' or name == 'Linear':
        fanIn = module.weight.data.size(0)
        fanOut = module.weight.data.size(1)

        factor = math.sqrt(2.0 / (fanIn + fanOut))
        weight = torch.randn(fanIn, fanOut) * factor
        module.weight.data.copy_(weight)

    # Check for bias and reset
    if hasattr(module, 'bias') and hasattr(module.bias, 'data'):
        module.bias.data.fill_(0.0)


class PointNet(nn.Module):
    def __init__(self, params):
        super(PointNet, self).__init__()
        self.transform, self.last_dim = zoo.selectModel_PointNet(params)
        initializeWeights([self.transform], 'xavier')

    def forward(self, x, normalize=True):
        """

        :param x: a tensor of size (batch_size, embed_dim)
        :param normalize:
        :return: a tensor of size (batch_size, params[h2])
        """
        return self.transform(x)


class PairNet(nn.Module):
    def __init__(self, params, point_net):
        super(PairNet, self).__init__()
        self.point_net = point_net
        self.dist = zoo.selectModel_PairNet(params, point_net.last_dim)
        if params["modelName"].split("_")[1] in ["bilinearDIAG", "bilinearFULL"]:
            initialize_single_module_weights(self.dist)

    def forward(self, x1, x2):
        """

        :param x1: a tensor of size (batch_size, embed_dim)
        :param x2: a tensor of size (batch_size, embed_dim)
        :return: a tensor of size (batch_size, 1)
        """
        return self.dist(self.point_net(x1), self.point_net(x2))


class TripletNet(nn.Module):
    def __init__(self, pair_net):
        super(TripletNet, self).__init__()
        self.pair_net = pair_net

    def forward(self, x1, x2, x3):
        output1 = self.pair_net(x1, x2)
        output2 = self.pair_net(x1, x3)
        return output1, output2


class PairNetWithPairFeatures(nn.Module):
    def __init__(self, params, point_net):
        super(PairNetWithPairFeatures, self).__init__()
        self.point_net = point_net
        self.dist = zoo.selectModel_PairNet(params, point_net.last_dim, params["pair_feature_dim"], params["pt"])
        if params["modelName"].split("_")[1] in ["bilinearDIAG", "bilinearFULL", "bilinearDIAG+PF", "PF",
                                                 "bilinearDIAG+PTF"]:
            initialize_single_module_weights(self.dist)

    def forward(self, x1, x2, r):
        """

        :param x1: a tensor of size (batch_size, embed_dim)
        :param x2: a tensor of size (batch_size, embed_dim)
        :param r: a tensor of size (batch_size, pair_feature_dim)
        :return: a tensor of size (batch_size, 1)
        """
        return self.dist(self.point_net(x1), self.point_net(x2), r)


class TripletNetWithPairFeatures(nn.Module):
    def __init__(self, pair_net):
        super(TripletNetWithPairFeatures, self).__init__()
        self.pair_net = pair_net

    def forward(self, x1, x2, x3, r1, r2):
        output1 = self.pair_net(x1, x2, r1)
        output2 = self.pair_net(x1, x3, r2)
        return output1, output2