"""
Patch utils
"""
from .utils import *


def vidx_to_pidx(vidx, vidx_max, pidx_max):
    """
    Video indexes to patch indexes
    """
    assert vidx.ndim == 2 and vidx.shape[1] == 3
    assert len(vidx_max) == 3
    assert len(pidx_max) == 3
    assert all(pidx_max[d] % vidx_max[d] == 0 for d in range(3))
    scales = [pidx_max[d] // vidx_max[d] for d in range(3)]
    pidx_t, pidx_h, pidx_w = [scales[d] * vidx[:, d][:, None] + torch.arange(scales[d], device=vidx.device)[None, :] for d in range(3)]
    pidx = torch.stack([pidx_t[:, :, None, None].expand([vidx.shape[0],] + scales), 
                        pidx_h[:, None, :, None].expand([vidx.shape[0],] + scales),
                        pidx_w[:, None, None, :].expand([vidx.shape[0],] + scales)], dim=-1).view(-1, 3)
    return pidx


def video_to_patch(video, patch_size):
    """
    Convert videos to 3D patches
    """
    assert video.ndim == 5
    assert all(video.shape[d + 1] % patch_size[d] == 0 for d in range(3))
    n_patch_t, n_patch_h, n_patch_w = (video.shape[d + 1] // patch_size[d] for d in range(3))
    patch = video.view(-1, n_patch_t, patch_size[0], n_patch_h, patch_size[1], n_patch_w, patch_size[2], video.shape[-1])\
                .permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, patch_size[0], patch_size[1], patch_size[2], video.shape[-1])
    return patch


def patch_to_video(patch, video_size):
    """
    Convert videos to 3D patches
    """
    assert patch.ndim == 5
    assert all(video_size[d] % patch.shape[d + 1] == 0 for d in range(3))
    n_patch_t, n_patch_h, n_patch_w = (video_size[d] // patch.shape[d + 1] for d in range(3))
    video = patch.view(-1, n_patch_t, n_patch_h, n_patch_w, patch.shape[1], patch.shape[2], patch.shape[3], patch.shape[-1])\
                .permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(-1, video_size[0], video_size[1], video_size[2], patch.shape[-1])
    return video


def _compute_pixel_idx_nd(n, idx, idx_max, sizes, padding, clipped=True, return_mask=True):
    assert idx.ndim == 2 and idx.shape[1] == n
    assert len(idx_max) == n
    assert all(sizes[d] % idx_max[d] == 0 for d in range(n))
    patch_sizes = [sizes[d] // idx_max[d] for d in range(n)]
    patch_sizes_padded = [patch_sizes[d] + padding[d] * 2 for d in range(n)]
    px_idx = [idx[:, d][:, None] * patch_sizes[d] - padding[d] + torch.arange(patch_sizes_padded[d], device=idx.device)[None, :] for d in range(n)]
    px_idx_clipped = [torch.clip(px_idx[d], 0, sizes[d] - 1) for d in range(n)] if clipped else px_idx
    if return_mask:
        idx_pad_mask = [(px_idx[d] >= 0) * (px_idx[d] < sizes[d]) for d in range(n)]
        return px_idx_clipped, idx_pad_mask
    else:
        return px_idx_clipped


def compute_pixel_idx_1d(idx, idx_max, sizes, padding, clipped=True, return_mask=True):
    """
    Get 1D pixel indexes.
    """
    return _compute_pixel_idx_nd(1, idx, idx_max, sizes, padding, clipped=clipped, return_mask=return_mask)


def compute_pixel_idx_3d(idx, idx_max, sizes, padding, clipped=True, return_mask=True):
    """
    Get 3D pixel indexes.
    """
    return _compute_pixel_idx_nd(3, idx, idx_max, sizes, padding, clipped=clipped, return_mask=return_mask)


def compute_paddings(output_patchsize=(1, 120, 120), scales=((1, 5, 5), (1, 4, 4), (1, 3, 3), (1, 2, 2)),
                     kernel_sizes=((0, 1, 1), (0, 1, 1), (0, 1, 1), (0, 1, 1)), 
                     depths=(3, 3, 3, 1), resize_methods='trilinear'):
    """
    Compute the required padding sizes.
    This is just an approximation to the padding required. The exact way is to calculate the boundary coordinates of the patches.
    Length of returned padding == len(scales) + 1 == len(padding_per_layer) + 1 == len(depths) + 1
    """
    assert len(scales) == len(depths) == len(kernel_sizes)    
    paddings = np.zeros(3, dtype=np.int32)
    scales = np.array(scales)
    kernel_sizes = np.array(kernel_sizes).clip(min=1)
    paddings_reversed = []
    for i in reversed(range(len(scales))):
        assert np.all((kernel_sizes[i] - 1) % 2 == 0)
        assert np.all(scales[i] >= 1.), 'scales must be positive'
        assert np.all(output_patchsize % scales[i] == 0), 'output_patchsize must be divisble by scales'
        paddings += depths[i] * (kernel_sizes[i] - 1) // 2
        if resize_methods == 'trilinear':
            paddings_reversed.append(paddings.tolist())
            paddings = np.round(paddings / scales[i]).astype(np.int32) + (scales[i] > 1.)
        elif resize_methods == 'nearest':
            paddings_reversed.append(paddings.tolist())
            paddings = np.round(paddings / scales[i]).astype(np.int32)
        elif resize_methods in ['conv1x1', 'conv3x3']:
            paddings_reversed.append(paddings.tolist())
            paddings = np.ceil(paddings / scales[i]).astype(np.int32) + (scales[i] > 1.)
            if resize_methods == 'conv1x1':
                pass
            elif resize_methods == 'conv3x3':
                paddings += np.array([0, 1, 1])
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    paddings_reversed.append(paddings.tolist())
    return list(reversed(paddings_reversed))