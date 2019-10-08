import math as m
import torch
import torch.nn


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def silverman_rule_of_thumb(N: int):
    return torch.pow(4/(3*N), torch.tensor([0.4], device=device)).to(device)


def euclidean_norm_squared(X: torch.Tensor, dim=None) -> torch.Tensor:
    return torch.sum(torch.pow(X, 2), dim=dim).to(device)


def phi_sampling(s, D):
    return torch.pow(1.0 + 4.0*s/(2.0*D-3), -0.5).to(device)


def cw_distance(Z: torch.Tensor):
    D = Z.shape[1]
    N = Z.shape[0]
    y = silverman_rule_of_thumb(N)

    K = (1 / (2 * D - 3))

    A1 = euclidean_norm_squared(torch.sub(Z.unsqueeze(0), Z.unsqueeze(1)), dim=2).to(device)
    A = (1 / (N ** 2)) * torch.sum(1 / torch.sqrt(y + K * A1)).to(device)

    B1 = euclidean_norm_squared(Z, dim=1).to(device)
    B = (2 / N) * torch.sum((1 / torch.sqrt(y + 0.5 + K * B1))).to(device)

    return ((1 / torch.sqrt(1 + y)) + A - B).to(device)


class CWSample(torch.nn.Module):
    def forward(self, X: torch.Tensor, Y: torch.Tensor, y=None):
        D = X.shape[1]
        N = X.shape[0]

        if y is None:
            y = silverman_rule_of_thumb(N)

        T = 1.0 / (2.0 * N * torch.sqrt(m.pi * y))

        A0 = euclidean_norm_squared(torch.sub(X.unsqueeze(0), X.unsqueeze(1)), dim=2).to(device)
        A = torch.sum(phi_sampling(A0 / (4 * y).to(device), D)).to(device)

        B0 = euclidean_norm_squared(torch.sub(Y.unsqueeze(0), Y.unsqueeze(1)), dim=2).to(device)
        B = torch.sum(phi_sampling(B0 / (4 * y).to(device), D)).to(device)

        C0 = euclidean_norm_squared(torch.sub(X.unsqueeze(0), Y.unsqueeze(1)), dim=2).to(device)
        C = torch.sum(phi_sampling(C0 / (4 * y).to(device), D)).to(device)

        return torch.mean(T * (A + B - 2 * C)).to(device)
