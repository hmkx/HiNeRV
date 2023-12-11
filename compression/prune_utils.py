"""
Prunning
"""
from .utils import *


_prunning_target_cls = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)


def _is_pruning_target(model, name, module):
    return name.startswith(model.bitstream_prefix) and not name.startswith(model.no_prune_prefix) and isinstance(module, _prunning_target_cls)


class PruningMask(nn.Module):
    """
    Pruning mask.
    """
    def forward(self, x):
        if not hasattr(self, 'mask'):
            self.register_buffer('mask', torch.ones_like(x, dtype=torch.bool))
        return self.mask * x


def get_sparsity(model):
    """
    Compute sparsity of the model. Note that, only the parameters that can be pruned will be counted as zeros.
    """
    model = unwrap_model(model)
    zeros = 0
    total = 0
    with torch.no_grad():
        for k, v in model.named_modules():
            if k.startswith(model.bitstream_prefix):
                if isinstance(v, PruningMask):
                    zeros += (v.mask == 0.).sum().item()
                else:
                    total += sum([p.numel() for p in v.parameters(recurse=False)])
    return zeros, total


def init_pruning(args, logger, model):
    """
    Initialize the pruning masks.
    """
    model = unwrap_model(model)
    with torch.no_grad():
        for k, v in model.named_modules():
            if _is_pruning_target(model, k, v):
                if args.debug:
                    logger.info(f'     Init pruning: {k}.weight')
                nn.utils.parametrize.register_parametrization(v, 'weight', PruningMask())


def set_pruning(args, logger, model, ratio, weight):
    """
    Set pruning for the model.
    """
    model = unwrap_model(model)
    with torch.no_grad():
        parameters_masks = []
        zeros, total = get_sparsity(model)
        target_amount = int(total * ratio)

        if ratio == 0. or target_amount == zeros:
            return zeros, total

        for k, v in model.named_modules():
            if _is_pruning_target(model, k, v):
                if args.debug:
                    logger.info(f'     Update pruning: {k}.weight')
                for p in v.parametrizations['weight']:
                    if isinstance(p, PruningMask):
                        parameters_masks.append((v.weight, p.mask))

        # Compute new mask
        scores = nn.utils.parameters_to_vector([abs(torch.where(m, v / (m.sum() ** weight + 1e-6), torch.inf)) for v, m in parameters_masks])
        masks = nn.utils.parameters_to_vector([m for _, m in parameters_masks])
        thres = torch.sort(scores)[0][target_amount - 1]

        # Update mask
        masks = masks * (scores > thres)
        pointer = 0
        for _, m in parameters_masks:
            num_param = m.numel()
            m.copy_(masks[pointer:pointer + num_param].view_as(m))
            pointer += num_param

        # Compute and return the new sparsity
        zeros, total = get_sparsity(model)
        return zeros, total