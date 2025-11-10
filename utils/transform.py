import torch

def data_transform(X, rescaled=False):
    if rescaled:
        X = 2 * X - 1.
    return X


def inverse_data_transform(X, rescaled=False):
    if rescaled:
        X = (X + 1.) / 2.
    return torch.clamp(X, 0.0, 1.0)