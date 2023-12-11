"""
Quantization
"""
from .utils import *
from models.layers import FeatureGrid


_quant_target_cls = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, FeatureGrid)


def _is_quant_target(model, name, module):
    return name.startswith(model.bitstream_prefix) and not name.startswith(model.no_quant_prefix) and isinstance(module, _quant_target_cls)


def _ste(x):
    """
    Straight-through estimator.
    """
    return (x.round() - x).detach() + x


def _quantize_ste(x, n, axis=None):
    """
    Per-channel & symmetric quantization with STE.
    """
    quant_range = 2. ** n - 1.
    x_max = abs(x).max(dim=axis, keepdim=True)[0] if axis is not None else abs(x).max()
    x_scale = 2 * x_max / quant_range + 1e-6
    x_q = _ste(x / x_scale).clamp(-2**(n - 1), 2**(n - 1) - 1)
    return x_q, x_scale


class QuantNoise(nn.Module):
    """
    Quant-Noise with optional STE.
    """
    def __init__(self, bitwidth, noise_ratio, ste, axis):
        super().__init__()
        self.register_buffer('bitwidth', torch.tensor(bitwidth, dtype=torch.float32))
        self.register_buffer('noise_ratio', torch.tensor(noise_ratio, dtype=torch.float32))
        self.ste = ste
        self.axis = axis

    def extra_repr(self):
        s = 'ste={ste}, axis={axis}'
        return s.format(**self.__dict__)

    def forward(self, x):
        if self.training:
            x_q, x_scale = _quantize_ste(x, self.bitwidth, self.axis)
            x_q = x_q if self.ste else x_q.detach()
            x_qr = x_q.to(x.dtype) * x_scale
            mask = (torch.rand_like(x) > self.noise_ratio).to(x.dtype)
            return x * mask + x_qr * (1. - mask)
        else:
            return x


def init_quantization(args, logger, model):
    """
    Initialize quantization for the model.
    """
    model = unwrap_model(model)

    with torch.no_grad():
        for k, v in model.named_modules():
            if _is_quant_target(model, k, v):
                if args.debug:
                    logger.info(f'     Set quatization: {k}.weight')
                quant_layer = QuantNoise(bitwidth=8, noise_ratio=0., ste=False, axis=compute_best_quant_axis(v.weight))
                quant_layer.to(list(v.parameters())[0].device)
                torch.nn.utils.parametrize.register_parametrization(v, 'weight', quant_layer)


def set_quantization(args, logger, model, quant_level, quant_noise, quant_ste):
    """
    Set quantization for the model.
    """
    model = unwrap_model(model)

    with torch.no_grad():
        for k, v in model.named_modules():
            if isinstance(v, QuantNoise):
                if args.debug:
                    logger.info(f'     Update quatization: {k}.weight')
                v.bitwidth.copy_(quant_level)
                v.noise_ratio.copy_(quant_noise)
                v.ste = quant_ste


def _quant_tensor(args, logger, x, quant_level):
    """
    Quantize a tensor.
    """
    axis = compute_best_quant_axis(x)
    with torch.no_grad():
        x_q, x_scale = _quantize_ste(x, quant_level, axis)
        x_q = x_q.to(torch.int32)
        x_qr = x_q.to(x.dtype) * x_scale

    if args.debug:
        logger.info(f'     Shape: {x.shape}')
        logger.info(f'     quant axis: {axis}')

    # Return the quantisied tensors (both int/float) and config
    meta = {
        'axis': axis,
        'scale': x_scale.half()
    }

    return x_q, x_qr, meta


def quant_model(args, logger, model, quant_level):
    """
    Quantize the full model.
    """
    model = unwrap_model(model)

    # Get the quantisation targets
    excluded_keys = set()

    for k in model.state_dict().keys():
        if k.startswith(model.no_quant_prefix):
            excluded_keys.add(k)

    # Quantize
    qr_state_dict = copy.deepcopy(model.state_dict())
    q_state_dict = {}
    q_config = {'quant_level': quant_level}
    if quant_level == 32:
        pass
    elif quant_level <= 16:
        for k, v in model.state_dict().items():
            if not k in excluded_keys and v.ndim > 1 and torch.is_floating_point(v):
                q_state_dict[k], qr_state_dict[k], q_config[k] = \
                    _quant_tensor(args, logger, v, quant_level)
    else:
        raise ValueError

    # Load quantisied state dict
    model.load_state_dict(qr_state_dict)

    # Ruturn configs
    return q_state_dict, q_config