"""
Losses
"""
from utils import *
import pytorch_msssim


def check_shape(x, y):
    assert x.shape == y.shape, 'shape of tensors must be the same!'
    assert x.stride() == y.stride(), 'strides of tensors must be the same!'
    assert x.ndim == y.ndim == 5, 'inputs are expected to have 5D ([N, C, T, H, W])'


def mse(x, y):
    """
    Compute the per-frame MSE loss
    """
    check_shape(x, y)
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    y = y.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    return F.mse_loss(x, y, reduction='none').mean(dim=2)


def l1(x, y):
    """
    Compute the per-frame L1 loss
    """
    check_shape(x, y)
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    y = y.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    return F.l1_loss(x, y, reduction='none').mean(dim=2)


def psnr(x, y, v_max=1.):
    """
    Compute the per-frame PSNR
    """
    return 10 * torch.log10((v_max ** 2) / (mse(x, y) + 1e-9))


def ssim(x, y, v_max=1., win_size=11):
    """
    Compute the per-frame SSIM
    """
    check_shape(x, y)
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
    y = y.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
    return pytorch_msssim.ssim(x, y, v_max, win_size=win_size, size_average=False).view(N, T)


def ms_ssim(x, y, v_max=1., win_size=11):
    """
    Compute the per-frame MS-SSIM
    """
    check_shape(x, y)
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
    y = y.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
    return pytorch_msssim.ms_ssim(x, y, v_max, win_size=win_size, size_average=False).view(N, T)


def compute_loss(name, x, y):
    assert x.ndim == 5 and y.ndim == 5, 'inputs are expected to have 5D ([N, C, T, H, W])'
    x, y = x.float(), y.float()
    if name == 'mse':
        return mse(x, y)
    elif name == 'l1':
        return l1(x, y)
    elif name == 'ssim':
        return 1. - ssim(x, y)
    elif name == 'ssim_3x3':
        return 1. - ssim(x, y, win_size=3)
    elif name == 'ssim_5x5':
        return 1. - ssim(x, y, win_size=5)
    elif name == 'ssim_7x7':
        return 1. - ssim(x, y, win_size=7)
    elif name == 'ms-ssim':
        return 1. - ms_ssim(x, y)
    elif name == 'ms-ssim_3x3':
        return 1. - ms_ssim(x, y, win_size=3)
    elif name == 'ms-ssim_5x5':
        return 1. - ms_ssim(x, y, win_size=5)
    elif name == 'ms-ssim_7x7':
        return 1. - ms_ssim(x, y, win_size=7)
    else:
        raise ValueError


def compute_metric(name, x, y):
    assert x.ndim == 5 and y.ndim == 5, 'inputs are expected to have 5D ([N, C, T, H, W])'
    x, y = x.float(), y.float()
    if name == 'mse':
        return mse(x, y)
    elif name == 'l1':
        return l1(x, y)
    elif name == 'psnr':
        return psnr(x, y)
    elif name == 'ssim':
        return ssim(x, y)
    elif name == 'ms-ssim':
        return ms_ssim(x, y)
    else:
        raise ValueError


def compute_regularization(name, model):
    raise ValueError