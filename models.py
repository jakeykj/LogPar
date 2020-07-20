import torch
from torch import nn


class NonnegativeSigmoid(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, x):
        return 2 / (1 + torch.exp(-self.gamma * x)) - 1


class LogisticPARAFAC2(nn.Module):
    def __init__(self,
                 num_visits,
                 num_feats,
                 rank,
                 alpha,
                 gamma,
                 is_projector=False):
        super().__init__()
        self.num_pts = len(num_visits)
        self.num_visits = num_visits
        self.num_feats = num_feats
        self.rank = rank
        self.gamma = gamma
        self.alpha = alpha

        self.U = nn.Parameter(torch.rand(self.num_pts, max(num_visits), rank))
        self.S = nn.Parameter(torch.rand(self.num_pts, rank) * 30 / rank)
        self.V = nn.Parameter(torch.rand(num_feats, rank))
        self.Phi = nn.Parameter(torch.rand(rank, rank), requires_grad=False)

        self.sigmoid = NonnegativeSigmoid(gamma)

        for i, num_visit in enumerate(num_visits):
            self.U.data[i, num_visit:] = 0
        
        if not is_projector:
            self.update_phi()

    def forward(self, pids):
        out = torch.einsum('ptr,pr,fr->ptf', self.U[pids], self.S[pids], self.V)
        out = self.sigmoid(out)
        return out

    def projection(self):
        self.U.data = self.U.data.clamp(min=0, max=self.alpha)
        self.S.data = self.S.data.clamp(min=0, max=self.alpha)
        self.V.data = self.V.data.clamp(min=0, max=self.alpha)

    def update_phi(self):
        if self.rank <= 200:  # use GPU with small ranks
            self.Phi.data = (torch.transpose(self.U.data, 1, 2) @ self.U.data).mean(dim=0)
        else:  # use CPU to avoid insufficient VRAM error
            Phi = (torch.transpose(self.U.data.cpu(), 1, 2) @ self.U.data.cpu()).mean(dim=0)
            self.Phi.data = Phi.to(self.Phi.data.device)
    
    def uniqueness_regularization(self, pids):
        U = self.U[pids]
        reg = torch.norm(torch.transpose(U, 1, 2) @ U - self.Phi.unsqueeze(0)) ** 2
        return reg / pids.shape[0]


class PULoss(nn.Module):
    def __init__(self, prior, gamma=0.5,
                 base_loss=nn.BCELoss(reduction='none')):
        super().__init__()
        if not 0 < prior < 1:
            raise ValueError('class prior for the nnPU loss should be between 0 and 1, '
                             f'but {prior} was given.')
        self.prior = prior
        self.gamma = gamma
        self.base_loss = base_loss

    def forward(self, input, target, masks=None):
        if masks is None:
            masks = torch.ones_like(input)
        if masks.shape[-1] == 1:
            masks = masks.repeat(1, 1, target.shape[-1])

        positive, unlabeled = ((target == 1) & (masks == 1)).float(), ((target == 0) & (masks == 1)).float()
        n_positive = target.sum()
        n_ublabeled = masks.sum() - n_positive

        loss_positive = masks * self.base_loss(input, torch.ones_like(input))
        loss_unlabeled = masks * self.base_loss(input, torch.zeros_like(input))

        positive_risk = (self.prior * positive * loss_positive).sum() / n_positive
        negative_risk = ((unlabeled / n_ublabeled - self.prior * positive / n_positive) * loss_unlabeled).sum()

        if negative_risk.item() < 0:
            objective = positive_risk
            out = -self.gamma * negative_risk
        else:
            objective = positive_risk + negative_risk
            out = objective

        return objective, out


class SmoothnessConstraint(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def forward(self, X, seq_len, deltas, norm_p=1):
        L = torch.zeros(X.shape[1], X.shape[1]+1)
        L[:, :-1] = torch.eye(X.shape[1])
        L[:, 1:] += -1 * torch.eye(X.shape[1])
        L = L.unsqueeze(0).repeat(seq_len.shape[0], 1, 1)
        L[torch.arange(seq_len.shape[0]), (seq_len-1).long()] = 0
        L = L[:, :-1, :-1]
        L = L.to(X.device)
        smoothness_mat = torch.exp(-self.beta * deltas[:, 1:].unsqueeze(2)) * (L @ X)
        smoothness = (smoothness_mat).norm(p=norm_p, dim=1) ** norm_p
        return smoothness.sum()
