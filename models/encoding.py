"""
Encoding
"""
from .utils import *
from .layers import *
from .patch_utils import *


"""
Index to coordinate
"""
class NormalizedCoordinate(nn.Module):
    def __init__(self, align_corners=False):
        super().__init__()
        self.align_corners = align_corners

    def extra_repr(self):
        s = 'align_corners={align_corners}'
        return s.format(**self.__dict__)

    def normalize_index(self, x: torch.Tensor, xmax: float, align_corners: bool=True):
        if xmax == 1.:
            return x * 0.
        elif align_corners:
            step = 2. / (xmax - 1)
            return -1. + x * step
        else:
            step = 2. / xmax
            return -1. + step / 2. + x * step

    def forward(self, l: torch.IntTensor, L: int):
        return self.normalize_index(l, float(L), self.align_corners)


class UnNormalizedCoordinate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, l: torch.IntTensor, L: int):
        return l


"""
Frequency Encoding
"""
class FrequencyEncoding1D(nn.Module):
    def __init__(self, B=1.25, L=192):
        super().__init__()
        self.B = B
        self.L = L
        self.register_buffer('base', (self.B ** torch.arange(self.L)) * torch.pi, persistent=False)

    def extra_repr(self):
        s = 'B={B}, L={L}'
        return s.format(**self.__dict__)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 2
        index_base = x[:, :, None] * self.base[None, None, :]
        enc = torch.cat([torch.sin(index_base), torch.cos(index_base)], dim=-1)
        return enc


class FrequencyEncoding(nn.Module):
    def __init__(self, C=256, Bt=1.25, Lt=192, Bs=1.25, Ls=192, use_t=True):
        super().__init__()
        self.C = C
        self.encoding_h = FrequencyEncoding1D(B=Bs, L=Ls)
        self.encoding_w = FrequencyEncoding1D(B=Bs, L=Ls)        
        self.encoding_t = FrequencyEncoding1D(B=Bt, L=Lt) if use_t else None
        self.linear = nn.Linear(2 * (Lt if use_t else 0) + 2 * 2 * Ls, C)

    def extra_repr(self):
        s = 'C={C}'
        return s.format(**self.__dict__)

    def forward(self, coor_t: torch.Tensor, coor_h: torch.Tensor, coor_w: torch.Tensor):
        assert coor_t.ndim == coor_h.ndim == coor_w.ndim == 2
        N, T = coor_t.shape
        _, H = coor_h.shape
        _, W = coor_w.shape

        enc = [self.encoding_h(coor_h).view(N, 1, H, 1, -1).expand(N, T, H, W, -1)]
        enc = enc + [self.encoding_w(coor_w).view(N, 1, 1, W, -1).expand(N, T, H, W, -1)]
        if self.encoding_t is not None:
            enc = enc + [self.encoding_t(coor_t).view(N, T, 1, 1, -1).expand(N, T, H, W, -1)]

        return self.linear(torch.concat(enc, dim=-1))


"""
Learned encoding
"""
def interpolate3D(grid: torch.Tensor, coor_wht: torch.Tensor, align_corners: bool=False):
    """
    Inputs:
        grid: input tensor with shape [T_grid, H_grid, W_grid, C].
        coor: coordinates with shape [N, T, H, W, 3]. In (x, y, t)/(w, h, t) order and normalized in [-1., 1.].
    Output:
        a tensor with shape [N, T, H, W, C]
    """
    _, _, _, C = grid.shape
    N, T, H, W, _ = coor_wht.shape
    # [T_grid, H_grid, W_grid, C] -> [1, C, T_grid, H_grid, W_grid]
    grid = grid.permute(3, 0, 1, 2).unsqueeze(0)
    # [N, T, H, W, 3] -> [1, N * T, H, W, 3]
    coor_wht = coor_wht.view(1, N * T, H, W, 3)
    # 'bilinear' with 5D input is actually 'trilinear' in grid_sample
    return F.grid_sample(grid, coor_wht, mode='bilinear', padding_mode='border', align_corners=align_corners).view(C, N, T, H, W).permute(1, 2, 3, 4, 0)


class GridEncodingBase(nn.Module):
    """
    K: Kernel size [K_t, K_h, K_w].
    C: Number of output channels.
    grid_size: First level grid's size.
    """
    def __init__(self, K=(1, 2, 2), C=128, grid_size=[120, 9, 16, 64], grid_level=3, grid_level_scale=[2., 1., 1., .5], init_scale=1e-3, align_corners=True):
        super().__init__()
        self.K = K
        self.C = C
        self.grid_sizes = []
        self.grid_level = grid_level
        self.grid_level_scale = grid_level_scale
        self.init_scale = init_scale
        self.align_corners = align_corners

        # Weights (saved in 2D to prevent converted by .to(channels_last))
        self.grids = nn.ModuleList()
        for i in range(self.grid_level):
            T_grid_i, H_grid_i, W_grid_i, C_grid_i = tuple((int(grid_size[d] / (self.grid_level_scale[d] ** i)) for d in range(4)))
            self.grid_sizes.append((T_grid_i, H_grid_i, W_grid_i, self.K[0], self.K[1], self.K[2], C_grid_i))
            self.grids.append(FeatureGrid((T_grid_i * H_grid_i * W_grid_i, self.K[0] * self.K[1] * self.K[2] * C_grid_i), init_scale=self.init_scale))

        # Linear
        self.linear = nn.Linear(sum([C_em for _, _, _, _, _, _, C_em in self.grid_sizes]), self.C)

    def extra_repr(self):
        s = 'C={C}, grid_sizes={grid_sizes}, grid_level={grid_level}, grid_level_scale={grid_level_scale}, '\
            'init_scale={init_scale}, align_corners={align_corners}'
        return s.format(**self.__dict__)

    def forward(self, coor_t: torch.FloatTensor, coor_h: Optional[torch.FloatTensor]=None, coor_w: Optional[torch.FloatTensor]=None):
        """        
        Inputs:
            coor_t/coor_h/coor_w: coordinates with shape [N, T/H/W]
        Output:
            a tensor with shape [N, T, H, W, K_t * K_h * K_w, C]
        """
        assert coor_t.ndim == 2 and (coor_h is None or coor_h.ndim == 2) and (coor_w is None or coor_w.ndim == 2)
        assert (coor_h is None) == (coor_w is None)
        N, T = coor_t.shape
        H = coor_h.shape[1] if coor_h is not None else 1
        W = coor_w.shape[1] if coor_w is not None else 1

        # Sampling coordinates
        if H > 1 and W > 1:
            coor_t = coor_t[:, :, None, None, None].expand(N, T, H, W, 1)
            coor_h = coor_h[:, None, :, None, None].expand(N, T, H, W, 1)
            coor_w = coor_w[:, None, None, :, None].expand(N, T, H, W, 1)
            coor_wht = torch.cat([coor_w, coor_h, coor_t], dim=-1) # (x, y, t)/(w, h, t) order
        else:
            coor_t = coor_t.view(N, T, 1, 1, 1)
            coor_wht = F.pad(coor_t, (2, 0)) # (x, y, t)/(w, h, t) order

        enc = []

        for i in range(self.grid_level):
            T_grid_i, H_grid_i, W_grid_i, K0, K1, K2, C_grid_i = self.grid_sizes[i]

            # Interpolate in all dimenions
            weight_i = self.grids[i]().view(T_grid_i, H_grid_i, W_grid_i, K0 * K1 * K2 * C_grid_i)
            enc_i = interpolate3D(grid=weight_i, coor_wht=coor_wht, align_corners=self.align_corners)
            enc.append(enc_i.contiguous().view(N, T * H * W * K0 * K1 * K2, C_grid_i))

        return self.linear(torch.concat(enc, dim=-1)).view(N, T, H, W, math.prod(self.K) * self.C)


class GridEncoding(GridEncodingBase):
    def __init__(self, C=128, grid_size=[120, 9, 16, 64], grid_level=3, grid_level_scale=[2., 1., 1., .5], init_scale=1e-3, align_corners=True):
        super().__init__(K=(1, 1, 1), C=C, grid_size=grid_size, grid_level=grid_level, 
                         grid_level_scale=grid_level_scale, 
                         init_scale=init_scale, align_corners=align_corners)


class TemporalLocalGridEncoding(GridEncodingBase):
    def __init__(self, K=(1, 2, 2), C=128, grid_size=[10, 64], grid_level=3, grid_level_scale=[1., 1.], init_scale=1e-3, align_corners=True):
        super().__init__(K=K, C=C, grid_size=(grid_size[0], 1, 1, grid_size[1]), grid_level=grid_level, 
                         grid_level_scale=(grid_level_scale[0], 1, 1, grid_level_scale[1]), 
                         init_scale=init_scale, align_corners=align_corners)

    def forward(self, coor_t: torch.FloatTensor, coor_h: Optional[torch.FloatTensor]=None, coor_w: Optional[torch.FloatTensor]=None):
        assert coor_t.ndim == 2 and (coor_h is None or coor_h.ndim == 2) and (coor_w is None or coor_w.ndim == 2)
        assert (coor_h is None) == (coor_w is None)
        N, T = coor_t.shape
        H = coor_h.shape[1] if coor_h is not None else 1
        W = coor_w.shape[1] if coor_w is not None else 1
        return super().forward(coor_t, coor_h, coor_w).view(N, T, H, W, self.K[0], self.K[1], self.K[2], self.C)


"""
Encoder
"""
class PositionalEncoder(nn.Module):
    """
    PositionalEncoder. It can be used for the input grid to the network, or the HiNeRV's hierarchical encoding layer.
    """    
    def __init__(self, scale, channels, cfg=None):
        super().__init__()
        self.scale = scale
        self.channels = channels

        # Coordinate/Encoding type
        if cfg:
            self.coor_type, self.enc_type = cfg['type'].split('+')
        else:
            self.coor_type, self.enc_type = 'none', 'none'
        # Coordinate
        if self.coor_type == 'normalized':
            self.coor = NormalizedCoordinate(align_corners=cfg['align_corners'])
        elif self.coor_type == 'unnormalized':
            self.coor = UnNormalizedCoordinate()
        elif self.coor_type == 'none':
            self.coor = None
        else:
            raise NotImplementedError

        # Encoding
        if self.enc_type == 'frequency':
            self.enc = FrequencyEncoding(C=self.channels, Bt=cfg['Bt'], Lt=cfg['Lt'], Bs=cfg['Bs'], Ls=cfg['Ls'], use_t=not cfg['no_t'])
            self.f_type = 'global'
        elif self.enc_type == 'local_frequency':
            self.enc = FrequencyEncoding(C=self.channels, Bt=cfg['Bt'], Lt=cfg['Lt'], Bs=cfg['Bs'], Ls=cfg['Ls'], use_t=not cfg['no_t'])
            self.f_type = 'local'
        elif self.enc_type == 'grid':
            self.enc = GridEncoding(C=self.channels, grid_size=cfg['grid_size'], grid_level=cfg['grid_level'], grid_level_scale=cfg['grid_level_scale'],
                                       init_scale=cfg['grid_init_scale'], align_corners=cfg['align_corners'])
            self.f_type = 'global'
        elif self.enc_type == 'temp_local_grid':
            self.enc = TemporalLocalGridEncoding(K=self.scale, C=self.channels, grid_size=cfg['grid_size'], grid_level=cfg['grid_level'], grid_level_scale=cfg['grid_level_scale'],
                                                 init_scale=cfg['grid_init_scale'], align_corners=cfg['align_corners'])
            self.f_type = 'temp_local'
        elif self.enc_type == 'none':
            self.enc, self.f_type = None, None
        else:
            raise NotImplementedError

    def extra_repr(self):
        s = 'scale={scale}, channels={channels}, coor_type={coor_type}, enc_type={enc_type}'
        return s.format(**self.__dict__)

    def compute_global_encoding(self, x: torch.Tensor, idx: torch.IntTensor, idx_max: tuple[int, int, int],
                                size: tuple[int, int, int], scale: tuple[int, int, int], padding: tuple[int, int, int]):
        N, T, H, W, C = x.shape

        # Compute the global voxels coordinates
        px_idx, px_mask = compute_pixel_idx_3d(idx, idx_max, size, padding, clipped=False, return_mask=True)
        px_mask_3d = px_mask[0][:, :, None, None, None] \
                        * px_mask[1][:, None, :, None, None] \
                        * px_mask[2][:, None, None, :, None]
        coor_t, coor_h, coor_w = tuple(self.coor(idx, s) for idx, s in zip(px_idx, size))

        # Encode
        enc = px_mask_3d * self.enc(coor_t, coor_h, coor_w).view(N, T, H, W, C)

        return x + enc

    def compute_local_encoding(self, x: torch.Tensor, idx: torch.IntTensor, idx_max: tuple[int, int, int],
                               size: tuple[int, int, int], scale: tuple[int, int, int], padding: tuple[int, int, int]):
        N, T, H, W, C = x.shape

        # Compute the local voxel indexes
        px_idx, px_mask = compute_pixel_idx_3d(idx, idx_max, size, padding, clipped=False, return_mask=True)
        px_mask_3d = (px_mask[0][:, :, None, None, None]
                      * px_mask[1][:, None, :, None, None]
                      * px_mask[2][:, None, None, :, None])
        lpx_idx = tuple(px_idx[d] % scale[d] for d in range(3))
        coor_t, coor_h, coor_w = tuple(self.coor(idx, s) for idx, s in zip(lpx_idx, size))

        # Encode
        enc = px_mask_3d * self.enc(coor_t, coor_h, coor_w).view(N, T, H, W, C)

        return x + enc

    def compute_temp_local_encoding(self, x: torch.Tensor, idx: torch.IntTensor, idx_max: tuple[int, int, int],
                                    size: tuple[int, int, int], scale: tuple[int, int, int], padding: tuple[int, int, int]):
        N, T, H, W, C = x.shape

        # This implementation is only correct with no temporal upsampling
        assert scale[0] == 1

        # Compute the global voxel coordinates before upscaling
        pre_size = (size[0] // scale[0],)
        pre_padding = (int(math.ceil(padding[0] / scale[0])),)
        pre_idx = compute_pixel_idx_1d(idx[:, 0:1], idx_max[0:1], pre_size, pre_padding, clipped=False, return_mask=False)
        coor_t = self.coor(pre_idx[0], pre_size[0])

        # Compute the local voxel indexes
        px_idx, px_mask = compute_pixel_idx_3d(idx, idx_max, size, padding, clipped=False, return_mask=True)
        px_mask_3d = (px_mask[0][:, :, None, None, None]
                      * px_mask[1][:, None, :, None, None]
                      * px_mask[2][:, None, None, :, None])
        lpx_idx = tuple(px_idx[d] % scale[d] for d in range(3))

        # Compute the encoding indexes
        enc_idx = tuple(torch.arange(scale[d], device=x.device) for d in range(3))

        # Encoding
        M = tuple(lpx_idx[d][:, :, None] == enc_idx[d][None, None, :] for d in range(3))
        M_3d = (M[0][:, :, None, None, :, None, None] *
                M[1][:, None, :, None, None, :, None] *
                M[2][:, None, None, :, None, None, :])
        local_enc = self.enc(coor_t)
        local_enc_masked = px_mask_3d * torch.matmul(M_3d.view(N, T, H * W, scale[0] * scale[1] * scale[2]).to(local_enc.dtype),
                                                     local_enc.view(N, T, scale[0] * scale[1] * scale[2], C)).view_as(x)

        return x + local_enc_masked

    def forward(self, x: Optional[torch.Tensor], idx: torch.IntTensor, idx_max: tuple[int, int, int],
                size: tuple[int, int, int], scale: tuple[int, int, int], padding: tuple[int, int, int]):
        """ 
        Inputs:
            x: input tensor with shape [N, T1, H1, W1, C]
            idx: patch index tensor with shape [N, 3]
            idx_max: list of 3 ints. Represents the range of patch indexes.
            size: list of 3 ints. Represents the size of the fulle video. It does not have to be the same as the input size, as the input can be a patch from the full video.
            scale: list of 3 ints. Represents the scale factor. This will be used to compute the output size.
            padding: list of 3 ints. Represents the padding size. This will be used to compute the output size.
        Outputs:
            a tensor with shape [N, T2, H2, W2, C]
        """
        assert x is None or x.ndim == 5, x.shape
        assert idx.ndim == 2 and idx.shape[1] == 3, idx.shape
        assert len(idx_max) == 3
        assert len(scale) == 3
        assert len(size) == 3
        assert len(padding) == 3

        if x is None:
            x = torch.zeros((1,) + tuple(size[d] // idx_max[d] + 2 * padding[d] for d in range(3)) + (1,), device=idx.device)

        if self.f_type is None:
            x = x
        elif self.f_type == 'global':
            x = self.compute_global_encoding(x, idx, idx_max, size, scale, padding)
        elif self.f_type == 'local':
            x = self.compute_local_encoding(x, idx, idx_max, size, scale, padding)
        elif self.f_type == 'temp_local':
            x = self.compute_temp_local_encoding(x, idx, idx_max, size, scale, padding)
        else:
            raise NotImplementedError

        return x