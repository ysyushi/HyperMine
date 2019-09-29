import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def _distance_based_loss(self, positive_scores, negative_scores):
        return F.relu(positive_scores + self.margin - negative_scores)

    def _strength_based_loss(self, positive_scores, negative_scores):
        return F.relu(negative_scores + self.margin - positive_scores)

    def forward(self, positive_scores, negative_scores, size_average=True):
        # losses = self._strength_based_loss(positive_scores, negative_scores)
        losses = self._distance_based_loss(positive_scores, negative_scores)
        zeros = (losses.detach() == 0.0).sum().item()
        if size_average:
            return losses.mean(), zeros
        else:
            return losses.sum(), zeros
