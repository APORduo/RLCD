import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from typing import Tuple
import torch.distributed as dist
import numpy as np

def cdr(prob1, prob2,y1=None,y2=None,):

    if y1 is None:

        p_label1 = torch.argmax(prob1, dim=1)
        p_label2 = torch.argmax(prob2, dim=1)
    else:
        p_label1 = y1
        p_label2 = y2

    same_pred_mask = p_label1.eq(p_label2)
    
    p_label_all = torch.cat([p_label1, p_label2], dim=0)
    unique_labels = torch.unique(p_label1[same_pred_mask])
    unique_labels = torch.unique(p_label_all)
    #unique_labels = torch.arange(0,prob1.size(1)).cuda()
    a1 = torch.sum(prob1, dim=0)
    H1 = torch.mm(prob1.T, prob1)
    H1_norm = (H1.T / a1).T
    
    a2 = torch.sum(prob2, dim=0)
    H2 = torch.mm(prob2.T, prob2)
    H2_norm = (H2.T / a2).T
    
  
    H_mat = torch.mm(H1_norm, H2_norm.T)

    transfer_loss = (1 - torch.diagonal(H_mat)[unique_labels].mean())# add is good, aircraft will move towards base

    return transfer_loss 


    

    
    
class ConfidenceBasedSelfTrainingLoss(nn.Module):
    """
    Self training loss that adopts confidence threshold to select reliable pseudo labels from
    `Pseudo-Label : The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks (ICML 2013)
    <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.664.3543&rep=rep1&type=pdf>`_.

    Args:
        threshold (float): Confidence threshold.

    Inputs:
        - y: unnormalized classifier predictions.
        - y_target: unnormalized classifier predictions which will used for generating pseudo labels.

    Returns:
         A tuple, including
            - self_training_loss: self training loss with pseudo labels.
            - mask: binary mask that indicates which samples are retained (whose confidence is above the threshold).
            - pseudo_labels: generated pseudo labels.

    Shape:
        - y, y_target: :math:`(minibatch, C)` where C means the number of classes.
        - self_training_loss: scalar.
        - mask, pseudo_labels :math:`(minibatch, )`.

    """

    def __init__(self, threshold: float):
        super(ConfidenceBasedSelfTrainingLoss, self).__init__()
        self.threshold = threshold
        self.tmp = 0.07

    def forward(self, y, y_target):
        if len(y.size()) == 1:
            y = y.unsqueeze(0)
            y_target = y_target.unsqueeze(0)
        elif len(y.size()) < 1:
            return torch.Tensor([0]).cuda()
            
        y = y / self.tmp

        teacher_out = F.softmax(y_target.detach() / self.tmp, dim=-1)
        confidence, pseudo_labels = teacher_out.max(dim=1)
        mask = (confidence > self.threshold).float()
        self_training_loss = torch.sum(-teacher_out * F.log_softmax(y, dim=-1), dim=-1) * mask
       
        return self_training_loss.mean(),mask, pseudo_labels
        #return self_training_loss, mask, pseudo_labels




def entropy(predictions: torch.Tensor, reduction='none') -> torch.Tensor:
    r"""Entropy of prediction.
    The definition is:

    .. math::
        entropy(p) = - \sum_{c=1}^C p_c \log p_c

    where C is number of classes.

    Args:
        predictions (tensor): Classifier predictions. Expected to contain raw, normalized scores for each class
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``

    Shape:
        - predictions: :math:`(minibatch, C)` where C means the number of classes.
        - Output: :math:`(minibatch, )` by default. If :attr:`reduction` is ``'mean'``, then scalar.
    """
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H
def entropy_loss(input_:torch.Tensor):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def invariance_loss(x: Tensor, y: Tensor) -> Tensor:
    """Returns VICReg invariance loss.

    Args:
        x:
            Tensor with shape (batch_size, ..., dim).
        y:
            Tensor with shape (batch_size, ..., dim).
    """
    return F.mse_loss(x, y)


def sinkhorn(
    probabilities: Tensor,
    iterations: int = 3,
    gather_distributed: bool = False,
) -> Tensor:
    """Runs sinkhorn normalization on the probabilities as described in [0].

    Code inspired by [1].

    - [0]: Masked Siamese Networks, 2022, https://arxiv.org/abs/2204.07141
    - [1]: https://github.com/facebookresearch/msn

    Args:
        probabilities:
            Probabilities tensor with shape (batch_size, num_prototypes).
        iterations:
            Number of iterations of the sinkhorn algorithms. Set to 0 to disable.
        gather_distributed:
            If True then features from all gpus are gathered during normalization.
    Returns:
        A normalized probabilities tensor.

    """
    if iterations <= 0:
        return probabilities

    world_size = 1
    if gather_distributed and dist.is_initialized():
        world_size = dist.get_world_size()

    num_targets, num_prototypes = probabilities.shape
    probabilities = probabilities.T
    sum_probabilities = torch.sum(probabilities)
    if world_size > 1:
        dist.all_reduce(sum_probabilities)
    probabilities = probabilities / sum_probabilities

    for _ in range(iterations):
        # normalize rows
        row_sum = torch.sum(probabilities, dim=1, keepdim=True)
        if world_size > 1:
            dist.all_reduce(row_sum)
        probabilities /= row_sum
        probabilities /= num_prototypes

        # normalize columns
        probabilities /= torch.sum(probabilities, dim=0, keepdim=True)
        probabilities /= num_targets

    probabilities *= num_targets
    return probabilities.T

def variance_loss(x: Tensor, eps: float = 0.0001) -> Tensor:
    """Returns VICReg variance loss.

    Args:
        x:
            Tensor with shape (batch_size, ..., dim).
        eps:
            Epsilon for numerical stability.
    """
    std = torch.sqrt(x.var(dim=0) + eps)
    loss = torch.mean(F.relu(1.0 - std))
    return loss

