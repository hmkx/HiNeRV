"""
HiNeRV Compression
"""
from utils import *
import compression
from compression import prune_utils, quant_utils, codec_utils
from compression.prune_utils import set_pruning, get_sparsity, PruningMask
from compression.quant_utils import set_quantization, QuantNoise


def initial_parametrizations(args, logger, model):
    prune_utils.init_pruning(args, logger, model)
    quant_utils.init_quantization(args, logger, model)


def fix_parametrizations(model):
    with torch.no_grad():
        for k, v in model.named_modules():
            if hasattr(v, 'parametrizations'):
                for w, p_list in v.parametrizations.items():
                    for p in p_list:
                        if isinstance(p, QuantNoise):
                            pass
                        elif isinstance(p, PruningMask):
                            p_list.original.mul_(p.mask)
                        else:
                            raise NotImplementedError


def set_zero(model):
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        if k.endswith('.mask'):
            v.fill_(1.)
        else:
            v.zero_()


def compress_bitstream(args, logger, accelerator, model, output_dir, quant_level):
    model = compression.utils.unwrap_model(model)

    # Save the parametrized model
    state_dict = copy.deepcopy(model.state_dict())

    # Fix parametrizations
    fix_parametrizations(model)

    # Quantize weights into integers
    quant_state_dict_int, quant_config = quant_utils.quant_model(args, logger, model, quant_level)

    # Compress bitstream
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        num_bytes = codec_utils.compress_and_save_model(args, logger, model,
                                                        output_dir, f'Q{quant_level}',
                                                        quant_state_dict_int=quant_state_dict_int,
                                                        quant_config=quant_config)
    else:
        num_bytes = 0
    num_bytes, = accelerate.utils.broadcast_object_list([num_bytes])

    # Reset parametrizations
    model.load_state_dict(state_dict)

    return num_bytes


def decompress(args, logger, accelerator, model, output_dir, quant_level):
    model = compression.utils.unwrap_model(model)

    # Decompress bitstream
    if accelerator.is_main_process:
        num_bytes = codec_utils.decompress_and_load_model(args, logger, model, output_dir, f'Q{quant_level}')
    else:
        num_bytes = 0
    num_bytes, = accelerate.utils.broadcast_object_list([num_bytes])
    state_dict = accelerate.utils.broadcast(model.state_dict())
    model.load_state_dict(state_dict)

    return num_bytes


def set_compression_args(parser):
    group = parser.add_argument_group('Model compression parameters')
    group.add_argument('--prune-ratio', default=0., type=float,
                        help='Pruning ratio for each time of pruning')
    group.add_argument('--prune-weight', default=0., type=float,
                        help='Weight for computing pruning scores')
    group.add_argument('--quant-level', default=[8], type=int, nargs='+',
                        help='Quantization level (default: 8)')
    group.add_argument('--quant-noise', default=0.9, type=float,
                        help='Quantization noise ratio (default: 0.1)')
    group.add_argument('--quant-ste', default=False, type=str_to_bool,
                        help='Quantization with ste')