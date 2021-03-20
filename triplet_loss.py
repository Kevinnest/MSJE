from __future__ import print_function
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
class TripletLoss(object):
  """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid). 
  Related Triplet Loss theory can be found in paper 'In Defense of the Triplet 
  Loss for Person Re-Identification'."""
  def __init__(self, device, margin=None):
    self.margin = margin
    self.device = device
    if margin is not None:
      self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    else:
      self.ranking_loss = nn.SoftMarginLoss()

  def __call__(self, dist_ap, dist_an):

    with torch.cuda.device(self.device):
      y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1)).cuda()
    if self.margin is not None:
      loss = self.ranking_loss(dist_an, dist_ap, y)
    else:
      loss = self.ranking_loss(dist_an - dist_ap, y)
    return loss

def normalize(x, axis=-1):

  x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
  return x


def euclidean_dist(x):
  """
  Args:
    x: pytorch Variable, with shape [m, d]
    y: pytorch Variable, with shape [n, d]
  Returns:
    dist: pytorch Variable, with shape [m, n]
  """
  size = x.size(0) // 2
  x1 = x[:size].cuda()
  x2 = x[size:].cuda()
  # m, n = x.size(0), y.size(0)
  xx = torch.pow(x1, 2).sum(1, keepdim=True).expand(size, size)
  # import pdb; pdb.set_trace()
  yy = torch.pow(x2, 2).sum(1, keepdim=True).expand(size, size).t()
  dist = xx + yy
  dist.addmm_(1, -2, x1, x2.t())
  dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
  return dist


def hard_example_mining(dist_mat, return_inds=False):

  assert len(dist_mat.size()) == 2
  assert dist_mat.size(0) == dist_mat.size(1)
  N = dist_mat.size(0)

  dist_mat = torch.cat((dist_mat, dist_mat.t()))
  label = list(range(0, N))
  # label.extend(label)
  label = np.array(label)
  # label = torch.tensor(label).cuda().long()
  label = torch.tensor(label).cuda().long()

  # shape [N, N]
  is_pos = label.expand(N, N).eq(label.expand(N, N).t())
  is_pos = torch.cat((is_pos, is_pos))
  is_neg = label.expand(N, N).ne(label.expand(N, N).t())
  is_neg = torch.cat((is_neg, is_neg))

  # `dist_ap` means distance(anchor, positive)
  # both `dist_ap` and `relative_p_inds` with shape [N, 1]
  dist_ap, relative_p_inds = torch.max(
    dist_mat[is_pos].contiguous().view(2 * N, -1), 1, keepdim=True)
  # `dist_an` means distance(anchor, negative)
  # both `dist_an` and `relative_n_inds` with shape [N, 1]
  dist_an, relative_n_inds = torch.min(
    dist_mat[is_neg].contiguous().view(2 * N, -1), 1, keepdim=True)
  # shape [N]
  dist_ap = dist_ap.squeeze(1)
  dist_an = dist_an.squeeze(1)

  if return_inds:
    # shape [N, N]
    ind = (label.new().resize_as_(label)
           .copy_(torch.arange(0, N).long())
           .unsqueeze( 0).expand(N, N))
    # shape [N, 1]
    p_inds = torch.gather(
      ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
    n_inds = torch.gather(
      ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
    # shape [N]
    p_inds = p_inds.squeeze(1)
    n_inds = n_inds.squeeze(1)
    return dist_ap, dist_an, p_inds, n_inds

  return dist_ap, dist_an


def global_loss(tri_loss, global_feat, normalize_feature=False):

  if normalize_feature:
    global_feat = normalize(global_feat, axis=-1)
  # shape [N, N]
  dist_mat = euclidean_dist(global_feat)
  dist_ap, dist_an = hard_example_mining(
    dist_mat, return_inds=False)
  loss = tri_loss(dist_ap, dist_an)
  return loss, dist_ap, dist_an, dist_mat