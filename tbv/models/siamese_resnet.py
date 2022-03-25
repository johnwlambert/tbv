"""
Copyright (c) 2021
Argo AI, LLC, All Rights Reserved.

Notice: All information contained herein is, and remains the property
of Argo AI. The intellectual and technical concepts contained herein
are proprietary to Argo AI, LLC and may be covered by U.S. and Foreign
Patents, patents in process, and are protected by trade secret or
copyright law. This work is licensed under a CC BY-NC-SA 4.0 
International License.

Originating Authors: John Lambert
"""

import pdb

import numpy as np
import torch
from torch import nn

from tbv.models.resnet_factory import ResNetConvBackbone


def contrastive_loss(y_c: torch.Tensor, pred_dists: torch.Tensor, margin: int = 1) -> torch.Tensor:
    """
    Compute the similarities in the separation loss (4) by
    computing average pairwise similarities between points
    in the embedding space.
            element-wise square, element-wise maximum of two tensors.
            Contrastive loss also defined in:
            -	"Dimensionality Reduction by Learning an Invariant Mapping"
                            by Raia Hadsell, Sumit Chopra, Yann LeCun
    Args:
        y_c: Indicates if pairs share the same semantic class label or not
        pred_dists: Distances in the embeddding space between pairs.

    Returns:
        tensor representing contrastive loss values.
    """
    N = pred_dists.shape[0]

    # corresponds to "d" in the paper. If same class, pull together.
    # Zero loss if all same-class examples have zero distance between them.
    pull_losses = y_c * torch.pow(pred_dists, 2)
    # corresponds to "k" in the paper. If different class, push apart more than margin
    # if semantically different examples have distances are in [0,margin], then there WILL be loss
    zero = torch.zeros(N)
    device = y_c.device
    zero = zero.to(device)
    # if pred_dists for non-similar classes are <1, then incur loss >0.
    clamped_dists = torch.max(margin - pred_dists, zero)
    push_losses = (1 - y_c) * torch.pow(clamped_dists, 2)
    return torch.mean(pull_losses + push_losses)


def paired_euclidean_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Compute the distance in the semantic alignment loss (3) by
    computing average pairwise distances between *already paired*
    points in the embedding space.
    Note this is NOT computed between all possible pairs. Rather, we
    compare i'th vector of X vs. i'th vector of Y (i == j always).

    Args:
        X: Pytorch tensor of shape (N,D) representing N embeddings of dim D
        Y: Pytorch tensor of shape (N,D) representing N embeddings of dim D
    Returns:
        dists: Pytorch tensor of shape (N,) representing distances between fixed pairs
    """
    device = X.device
    N, D = X.shape
    assert Y.shape == X.shape
    eps = 1e-08 * torch.ones((N, 1))
    eps = eps.to(device)  # make sure in same memory (CPU or CUDA)
    # compare i'th vector of x vs. i'th vector of y (i == j always)
    diff = torch.pow(X - Y, 2)

    affinities = torch.sum(diff, dim=1, keepdim=True)
    # clamp the affinities to be > 1e-8 ?? Unclear why the authors do this...
    affinities = torch.max(affinities, eps)
    return torch.sqrt(affinities)


def smooth_triplet_loss(embeddings: torch.Tensor, triplet_idxs: np.ndarray) -> torch.Tensor:
    """
    N. N. Vo and J. Hays. Localizing and orienting street views
    using overhead imagery. ECCV 16.

    K. Sohn. Improved deep metric learning with multi-class npair
    loss objective. NIPS, 2016

    A. Hermans, L. Beyer, and B. Leibe. In defense of the
    triplet loss for person re-identification. arXiv, 2017.

    Ref:
            https://arxiv.org/pdf/1803.03310.pdf
            https://github.com/lugiavn/generalization-dml/blob/master/nams.py

    d is the squared Euclidean distance (negative dot
    product can also be used instead).

    Normalize the image feature to have unit magnitude and
    then scale it by 4,
    """
    # dist from anchors to negatives
    an_dists = paired_euclidean_dists(embeddings, a, n)
    # dist from anchors to positives
    ap_dists = pair_dists(embeddings, a, p)
    return torch.log(1 + torch.exp(an_dists - ap_dists))


def test_smooth_triplet_loss():
    """ """
    pass


class SiameseTripletResnet(nn.Module):
    def __init__(self, num_layers: int, pretrained: bool) -> None:
        """ """
        super(SiameseTripletResnet, self).__init__()
        self.net = ResNetConvBackbone(num_layers, pretrained)

    def forward(self, x_a: torch.Tensor, x_p: torch.Tensor, x_n: torch.Tensor) -> torch.Tensor:
        """anchor, positive, negative"""
        pdb.set_trace()

        x_a = self.net(x_a)
        x_p = self.net(x_p)
        x_n = self.net(x_n)

        loss = smooth_triplet_loss()

        # do some inference here, based on some thresholded distance
        output = x_a, x_p, x_n
        return output, loss


class SiameseContrastiveResnet(nn.Module):
    def __init__(self, num_layers: int, pretrained: bool) -> None:
        """ """
        super(SiameseContrastiveResnet, self).__init__()
        self.net = ResNetConvBackbone(num_layers, pretrained)

    def forward(self, x: torch.Tensor, xstar: torch.Tensor, y_c: torch.Tensor):
        """y_c: whether they belong to the same class

        Args:
            x: NCHW tensor
            xstar: NCHW tensor
            y_c: (N,) tensor
        """
        # get back (N,512) embeddings
        x = self.net(x)
        xstar = self.net(xstar)

        # TODO: should we normalize the embeddings first?

        # pass in the corresponding embeddings
        # inputs must be (N,D)
        pred_dists = paired_euclidean_distance(x, xstar)
        loss = contrastive_loss(y_c, pred_dists)

        return pred_dists, loss
