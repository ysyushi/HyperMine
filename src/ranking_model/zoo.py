import torch.nn as nn


# identity module
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inTensor):
        return inTensor


class DiagonalBilinear(nn.Module):
    def __init__(self, dim):
        super(DiagonalBilinear, self).__init__()
        self.W_diag = nn.Linear(dim, 1, bias=False)

    def forward(self, x1, x2):
        return self.W_diag(x1 * x2)


class DiagonalBilinearWithPairFeature(nn.Module):
    def __init__(self, node_dim, pair_dim):
        super(DiagonalBilinearWithPairFeature, self).__init__()
        self.W_diag = nn.Linear(node_dim, 1, bias=False)
        self.V = nn.Linear(pair_dim, 1, bias=True)

    def forward(self, x1, x2, r):
        return self.W_diag(x1 * x2) + self.V(r)


class OnlyPairFeature(nn.Module):
    def __init__(self, pair_dim):
        """

        :param pair_dim: pair feature dimension
        :type pair_dim: int
        """
        super(OnlyPairFeature, self).__init__()
        self.V = nn.Linear(pair_dim, 1, bias=True)

    def forward(self, x1, x2, r):
        return self.V(r)


class DiagonalBilinearWithPairTransformFeature(nn.Module):
    def __init__(self, node_dim, pair_dim, pt):
        """

        :param node_dim: node feature dimension
        :type node_dim: int
        :param pair_dim: pair feature dimension
        :type pair_dim: int
        :param pt: pair transform options
        :type pt: dict
        """
        super(DiagonalBilinearWithPairTransformFeature, self).__init__()
        self.W_diag = nn.Linear(node_dim, 1, bias=False)
        self.V = nn.Linear(pair_dim, 1, bias=True)
        if pt["name"] == "d":  # only dropout
            self.transform = nn.Dropout(pt["dropout"])
        elif pt["name"] == "ld":  # linear nn and then dropout
            self.transform = nn.Sequential(
                nn.Linear(pair_dim, pair_dim),
                nn.Dropout(pt["dropout"])
            )

    def forward(self, x1, x2, r):
        return self.W_diag(x1 * x2) + self.V(self.transform(r))


def selectModel_PointNet(params):
    if params['modelName'].split("_")[0] == "l":
        transform = nn.Sequential(
            nn.Linear(params["embedSize"], params['h1'], bias=False)
        )
        last_dim = params['h1']
    elif params['modelName'].split("_")[0] == "lt":
        transform = nn.Sequential(
            nn.Linear(params["embedSize"], params['h1'], bias=False),
            nn.Tanh()
        )
        last_dim = params['h1']
    elif params['modelName'].split("_")[0] == "lr":
        transform = nn.Sequential(
            nn.Linear(params["embedSize"], params['h1'], bias=False),
            nn.ReLU()
        )
        last_dim = params['h1']
    elif params['modelName'].split("_")[0] == "ltl":
        transform = nn.Sequential(
            nn.Linear(params["embedSize"], params['h1'], bias=False),
            nn.Tanh(),
            nn.Linear(params["h1"], params['h2'], bias=False),
        )
        last_dim = params['h2']
    elif params['modelName'].split("_")[0] == "ltlt":
        transform = nn.Sequential(
            nn.Linear(params["embedSize"], params['h1'], bias=False),
            nn.Tanh(),
            nn.Linear(params["h1"], params['h2'], bias=False),
        )
        last_dim = params['h2']
    elif params['modelName'].split("_")[0] == "lrl":
        transform = nn.Sequential(
            nn.Linear(params["embedSize"], params['h1'], bias=False),
            nn.ReLU(),
            nn.Linear(params["h1"], params['h2'], bias=False),
        )
        last_dim = params['h2']
    elif params['modelName'].split("_")[0] == "ltdl":
        transform = nn.Sequential(
            nn.Linear(params["embedSize"], params['h1'], bias=False),
            nn.Tanh(),
            nn.Dropout(params["node_dropout"]),
            nn.Linear(params["h1"], params['h2'], bias=False),
        )
        last_dim = params['h2']
    elif params['modelName'].split("_")[0] == "lrdl":
        transform = nn.Sequential(
            nn.Linear(params["embedSize"], params['h1'], bias=False),
            nn.ReLU(),
            nn.Dropout(params["node_dropout"]),
            nn.Linear(params["h1"], params['h2'], bias=False),
        )
        last_dim = params['h2']
    else:
        transform = Identity()
        last_dim = params['embedSize']
    return transform, last_dim


def selectModel_PairNet(params, point_net_last_dim=None, pair_feature_dim=None, pt=None):
    if params['modelName'].split("_")[1] == "l1":
        dist = nn.PairwiseDistance(p=1, keepdim=True)
    if params['modelName'].split("_")[1] == "l2":
        dist = nn.PairwiseDistance(p=2, keepdim=True)
    if params['modelName'].split("_")[1] == "cosine":
        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-08)
        dist = lambda x1, x2: 1.0 - cos_sim(x1, x2).unsqueeze_(1)
    if params['modelName'].split("_")[1] == "bilinearDIAG":
        assert point_net_last_dim is not None
        dist = DiagonalBilinear(point_net_last_dim)
    if params['modelName'].split("_")[1] == "bilinearFULL":
        assert point_net_last_dim is not None
        dist = nn.Bilinear(point_net_last_dim, point_net_last_dim, 1, bias=False)
    if params['modelName'].split("_")[1] == "bilinearDIAG+PF":
        assert point_net_last_dim is not None and pair_feature_dim is not None
        dist = DiagonalBilinearWithPairFeature(point_net_last_dim, pair_feature_dim)
    if params['modelName'].split("_")[1] == "bilinearDIAG+PTF":
        assert point_net_last_dim is not None and pair_feature_dim is not None and pt is not None
        dist = DiagonalBilinearWithPairTransformFeature(point_net_last_dim, pair_feature_dim, pt)
    if params['modelName'].split("_")[1] == "PF":
        assert pair_feature_dim is not None
        dist = OnlyPairFeature(pair_feature_dim)
    return dist
