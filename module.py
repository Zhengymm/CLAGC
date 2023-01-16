import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv


def pos_sampling(q, q_aug, thred):
    # evaluate clustering performance, q: cluster distribution
    y_pred = q.argmax(1)
    yaug_pred = q_aug.argmax(1)

    intra_pos = []
    inter_pos = []

    intra_confid_mask = q.ge(thred)
    inter_confid_mask = q_aug.ge(thred)

    intra_confid_id = torch.where(intra_confid_mask)[0]  # tensor
    inter_confid_id = torch.where(inter_confid_mask)[0]

    for k in range(np.max(y_pred.cpu().numpy()) + 1):
        intra_same_id = torch.where(y_pred == k)[0]
        inter_same_id = torch.where(yaug_pred == k)[0]

        if thred > 0:
            intra_pos_confid = np.intersect1d(intra_same_id.numpy(), intra_confid_id.numpy())  # 取交集
            intra_pos.append(torch.tensor(intra_pos_confid))
            inter_pos_confid = np.intersect1d(inter_same_id.numpy(), inter_confid_id.numpy())  # 取交集
            inter_pos.append(torch.tensor(inter_pos_confid))
        else:
            intra_pos.append(intra_same_id)
            inter_pos.append(inter_same_id)

    intra_mask = torch.eye(y_pred.shape[0], device=q.device)
    inter_mask = torch.eye(y_pred.shape[0], device=q.device)

    for i in range(y_pred.shape[0]):
        intra_mask[i, intra_pos[y_pred[i]].tolist()] = 1
        inter_mask[i, inter_pos[y_pred[i]].tolist()] = 1

    return intra_mask, inter_mask


def contrastive_loss_batch(z1, z2, temperature=1):
    batch_size = 2000
    z1 = F.normalize(z1, dim=-1, p=2)
    z2 = F.normalize(z2, dim=-1, p=2)
    # device = z1.device
    # pos_mask = torch.eye(z1.size(0), dtype=torch.float32)
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    f = lambda x: torch.exp(x / temperature)
    indices = torch.arange(0, num_nodes)
    losses = []

    # neg_mask = 1 - pos_mask

    for i in range(num_batches):
        mask = indices[i * batch_size:(i + 1) * batch_size]
        # intra_sim = f(torch.mm(z1[mask], z1.t()))  # [B, N]
        intra_sim_11 = f(torch.mm(z1[mask], z1.t()))  # [B, N]
        intra_sim_22 = f(torch.mm(z2[mask], z2.t()))  # [B, N]
        inter_sim = f(torch.mm(z1[mask], z2.t()))  # [B, N]

        loss_12 = -torch.log(
            inter_sim[:, i * batch_size:(i + 1) * batch_size].diag()
            / (intra_sim_11.sum(1) + inter_sim.sum(1)
               - intra_sim_11[:, i * batch_size:(i + 1) * batch_size].diag()))
        loss_21 = -torch.log(
            inter_sim[:, i * batch_size:(i + 1) * batch_size].diag()
            / (intra_sim_22.sum(1) + inter_sim.sum(1)
               - intra_sim_22[:, i * batch_size:(i + 1) * batch_size].diag()))
        losses.append(loss_12+loss_21)

    return torch.cat(losses)


def contrastive_cross_view(h1, h2, temperature=1):
    z1 = F.normalize(h1, dim=-1, p=2)
    z2 = F.normalize(h2, dim=-1, p=2)
    f = lambda x: torch.exp(x / temperature)
    intra_sim = f(torch.mm(z1, z1.t()))
    inter_sim = f(torch.mm(z1, z2.t()))  # 视图间同一节点做正样本

    # loss = -torch.log(inter_sim.diag() / (intra_sim.sum(dim=-1) + inter_sim.sum(dim=-1) - intra_sim.diag()))

    # 改变pos_mask, 以利用伪标签进行约束
    pos_mask = torch.eye(z1.size(0), dtype=torch.float32).to(z1.device)

    neg_mask = 1 - pos_mask
    pos = (inter_sim * pos_mask).sum(1)  # pos <=> between_sim.diag()
    neg = (intra_sim * neg_mask).sum(1)  # neg <=> refl_sim.sum(1) - refl_sim.diag()

    loss = -torch.log(pos / (inter_sim.sum(1) + neg))  # inter_sim.sum(1) = (inter_sim*neg_mask).sum(1) + pos

    return loss


def contrastive_cross_view_hard_batch(z1, z2, q, q_aug, thred, temperature=1):
    batch_size = 2000
    z1 = F.normalize(z1, dim=-1, p=2)
    z2 = F.normalize(z2, dim=-1, p=2)
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    f = lambda x: torch.exp(x / temperature)
    indices = torch.arange(0, num_nodes)
    total_losses = []

    intra_pos_mask, inter_pos_mask = pos_sampling(q, q_aug, thred)

    for i in range(num_batches):
        mask = indices[i * batch_size:(i + 1) * batch_size]
        intra_sim = f(torch.mm(z1[mask], z1.t()))  # [B, N]
        inter_sim = f(torch.mm(z1[mask], z2.t()))  # [B, N]

        intra_pos_sim = intra_sim.masked_fill(~intra_pos_mask[mask].bool(), 0)
        intra_pos = intra_pos_sim.sum(1)
        intra_neg = (intra_sim - intra_pos_sim).sum(1)

        pos = intra_pos - intra_sim[:, i * batch_size:(i + 1) * batch_size].diag() \
              + inter_sim[:, i * batch_size:(i + 1) * batch_size].diag()

        neg = intra_neg + inter_sim.sum(1) - inter_sim[:, i * batch_size:(i + 1) * batch_size].diag()

        batch_loss = -torch.log(pos / (pos + neg))
        total_losses.append(batch_loss)

    return torch.cat(total_losses)


class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, x):
        return self.net(x)


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden,
                 activation, base_model=GCNConv):
        super(Encoder, self).__init__()
        self.base_model = base_model

        self.gcn1 = base_model(in_channels, hidden)
        self.gcn2 = base_model(hidden, out_channels)

        self.activation = nn.PReLU() if activation == 'prelu' else nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.activation(self.gcn1(x, edge_index))
        x = self.gcn2(x, edge_index)
        return x


class Contra(Module):
    def __init__(self,
                 encoder,
                 hidden_size,
                 projection_size,
                 projection_hidden_size,
                 n_cluster,
                 v=1):
        super().__init__()

        # backbone encoder
        self.encoder = encoder

        # projection layer for representation contrastive
        self.rep_projector = MLP(hidden_size, projection_size, projection_hidden_size)
        # t-student cluster layer for clustering
        self.cluster_layer = nn.Parameter(torch.Tensor(n_cluster, hidden_size), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.v = v

    def kl_cluster(self, z1: torch.Tensor, z2: torch.Tensor):
        q1 = 1.0 / (1.0 + torch.sum(torch.pow(z1.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q1 = q1.pow((self.v + 1.0) / 2.0)  # q1 n*K
        q1 = (q1.t() / torch.sum(q1, 1)).t()

        q2 = 1.0 / (1.0 + torch.sum(torch.pow(z2.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q2 = q2.pow((self.v + 1.0) / 2.0)
        q2 = (q2.t() / torch.sum(q2, 1)).t()

        return q1, q2

    def forward(self, feat, adj):
        h = self.encoder(feat, adj)
        z = self.rep_projector(h)

        return h, z

