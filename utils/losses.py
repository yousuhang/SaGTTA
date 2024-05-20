import torch
from torch import nn


class CrossEntropyLossWeighted(nn.Module):
    """
    Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
    vector of shape (batch_size,).
    """

    def __init__(self, n_classes=3):
        super(CrossEntropyLossWeighted, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.n_classes = n_classes

    def one_hot(self, targets):
        targets_extend = targets.clone()
        targets_extend.unsqueeze_(1)  # convert to Nx1xHxW
        one_hot = torch.FloatTensor(targets_extend.size(0), self.n_classes, targets_extend.size(2),
                                    targets_extend.size(3)).zero_().to(targets.device)
        one_hot.scatter_(1, targets_extend, 1)

        return one_hot

    def forward(self, inputs, targets):
        one_hot = self.one_hot(targets)

        # size is batch, nclasses, 256, 256
        weights = 1.0 - torch.sum(one_hot, dim=(2, 3), keepdim=True) / torch.sum(one_hot)
        one_hot = weights * one_hot

        loss = self.ce(inputs, targets).unsqueeze(1)  # shape is batch, 1, 256, 256
        loss = loss * one_hot

        return torch.sum(loss) / (torch.sum(weights) * targets.size(0) * targets.size(1))

class ContourRegularizationLoss(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2 * d + 1)

    def forward(self, x):
        # x is the probability maps
        C_d = self.max_pool(x) + self.max_pool(-1*x) # size is batch x 1 x h x w

        loss = torch.norm(C_d, p=2, dim=(2, 3)).mean()
        return loss


class NuclearNorm(nn.Module):
    def __init__(self, k=4):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=k)

    def forward(self, x):
        # x is probabilities
        x = self.max_pool(x) # size is batch x n_classes x h x w
        x = x.permute(0, 2, 3, 1) # batch x h x w x n_classes
        x = x.flatten(1, 2) # batch x hw x n_classes
        loss = torch.norm(x, p='nuc', dim=(1, 2)).mean()
        return loss

class Cos_Sim(nn.Module):
    def __init__(self, dim=1, eps=1e-7):
        super().__init__()
        self.cos_sim = nn.CosineSimilarity(dim=dim, eps=eps)

    def forward(self, x, y):
        # x , y are input images/vectors
        cos_sim = self.cos_sim(x,y).mean()
        return cos_sim