"""
Entropy coding
"""
from .utils import *

import torchac
import zlib
import shutil
import tempfile


def _remove_zeros(x, mask):
    """ Remove entries from the tensor with a mask."""
    y = torch.masked_select(x.view(-1), mask.view(-1))
    return y


def _restore_zeros(y, mask):
    """ Restore entries from the tensor with a mask."""
    idx = torch.cumsum(mask.long().view(-1), dim=0)
    x = F.pad(y.view(-1), (1, 0))[idx].view_as(mask) * mask
    return x


def _arithmetic_encoding(x):
    """ Arithmetic encoding for a tensor with torchac. """    
    with torch.no_grad():
        x = x.detach().view(-1).cpu()
        sym, inverse, counts = x.unique(return_inverse=True, return_counts=True)
        inverse = inverse.to(torch.int16)
        counts = torch.concat([torch.zeros([1], dtype=torch.int64, device=counts.device), counts])
        cdf = torch.cumsum(counts, dim=0).float() / counts.sum().float()
        byte_stream = torchac.encode_float_cdf(cdf[None].repeat(math.prod(x.shape), 1), inverse, check_input_bounds=True, needs_normalization=True)        
        inverse_out = torchac.decode_float_cdf(cdf[None].repeat(math.prod(x.shape), 1), byte_stream).int()
        assert inverse_out.equal(inverse)
        x_out = sym[inverse_out.int()]
        assert x_out.equal(x)
        return sym, cdf, byte_stream


def _arithmetic_decoding(byte_stream, sym, cdf, shape):
    """ Arithmetic decoding for a tensor with torchac. """    
    with torch.no_grad():
        inverse_out = torchac.decode_float_cdf(cdf[None].repeat(math.prod(shape), 1), byte_stream).int()
        x_out = sym[inverse_out.int()].view(shape)
        return x_out


def _compress_tensor(args, logger, x, dim, output_dir, name):
    """ Compress a tensor and store its meta data. """
    assert x.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]

    if args.debug:
        logger.info(f'     Shape: {x.shape}')
        logger.info(f'     Raw size @ 8bit: {math.prod(x.shape)} bytes')
        logger.info(f'     Number of symbols: {len(x.unique())}')

    if x.numel() > 0:
        # arithmetic_encoding requires postive integers
        offset = x.min() if x.numel() > 0 else 0
        x -= offset

        if dim is None:
            sym, cdf, bits = _arithmetic_encoding(x)
            bits_length = len(bits)
        else:
            sym = []
            cdf = []
            bits = None
            bits_length = []
            x_splits = torch.split(x, 1, dim=dim)
            for x_i in x_splits:
                sym_i, cdf_i, bits_i = _arithmetic_encoding(x_i.contiguous())
                sym.append(sym_i)
                cdf.append(cdf_i)
                bits = bits + bits_i if bits is not None else bits_i
                bits_length.append(len(bits_i))

        if args.debug:
            logger.info(f'     Compressed: {len(bits)} bytes')
            logger.info(f'     Compression ratio @ 8bit: {len(bits) / (math.prod(x.shape))}')

        # meta data
        meta = {
            'dim': dim,
            'sym': sym,
            'cdf': cdf,
            'bits_length': bits_length,
            'shape': x.shape,
            'offset': offset
        }
    else:        
        # meta data
        meta = {
            'dim': None,
            'sym': None,
            'cdf': None,
            'bits_length': 0,
            'shape': None,
            'offset': None
        }

    # Either compress the tensor and return the meta data, or return the tensor
    os.makedirs(output_dir, exist_ok=True)

    # byte_stream
    if x.numel() > 0:
        with open(os.path.join(output_dir, name + '.bit'), 'wb') as f:
            f.write(bits)

    return meta


def _decompress_tensor(state_dict, output_dir, name):
    """ Decompress a tensor. """
    # Read meta file
    dim = state_dict['dim']
    sym = state_dict['sym']
    cdf = state_dict['cdf']
    bits_length = state_dict['bits_length']
    shape = state_dict['shape']
    offset = state_dict['offset']

    # Read bitstreams
    if bits_length > 0:
        with open(os.path.join(output_dir, name + '.bit'), 'rb') as f:
            byte_stream = f.read()

        if dim is None:
            x = _arithmetic_decoding(byte_stream, sym, cdf, shape)
        else:
            x_splits = []
            bits_count = 0
            for i in range(shape[dim]):
                x_splits.append(_arithmetic_decoding(byte_stream[bits_count:bits_count + bits_length[i]], sym[i], cdf[i], list(shape[:dim]) + [1] + list(shape[dim+1:])))
                bits_count += bits_length[i]
            x = torch.concat(x_splits, dim=dim)

        x += offset.to(x.device)
    else:
        x = torch.zeros([0])

    return x


def _get_target_keys(model):
    target_keys = []
    for k in model.state_dict().keys():
        if k.startswith(model.bitstream_prefix) and not 'mask' in k:
            target_keys.append(k)
    return target_keys


def _get_mask_key(state_dict, k):
    if k.endswith('.weight.original'):
        for i in range(5): # just assuming upto 5 parameterizations
            mask_k = k.replace('.weight.original', f'.weight.{i}.mask')
            if mask_k in state_dict:
                return mask_k
    return None


def compress_and_save_model(args, logger, model, output_dir, name, quant_state_dict_int, quant_config):
    """
    Compress and save the model.
    Tensors with quantization will be compressed with arithmetic coding, and the rest will be saved with torch.save.
    """
    model = unwrap_model(model)

    output_path = os.path.join(output_dir, name)
    temp_dir = tempfile.TemporaryDirectory(dir=output_dir)

    target_keys = _get_target_keys(model)
    state_dict = model.state_dict()
    state_dict_temp = {'quant_config': quant_config}

    # Try to use compress_tensor (arithmetic coding) if possible, otherwise save the state with torch.save
    for k in target_keys:
        v = state_dict[k]

        if args.debug:
            logger.info(f'Saving mask: {k}')
        mask_k = _get_mask_key(state_dict, k)
        if mask_k is not None:
            mask = state_dict[mask_k]
            state_dict_temp[k + '_mask'] = _compress_tensor(args, logger, mask.to(torch.int8), None, temp_dir.name, k + '_mask')
        else:
            mask = None

        if args.debug:
            logger.info(f'Saving state: {k}')
        if quant_config is not None and k in quant_config:
            # Only use coding with the quantized tensors, since they have limited number of symbols
            quant_v = quant_state_dict_int[k] 
            quant_v = _remove_zeros(quant_v, mask) if mask is not None else quant_v
            state_dict_temp[k] = _compress_tensor(args, logger, quant_v, None, temp_dir.name, k)
        else:
            v = _remove_zeros(v, mask) if mask is not None else v
            state_dict_temp[k] = v.half() if v.dtype == torch.float32 else v

    ckpt_path = os.path.join(temp_dir.name, 'model.pth.tar')
    compress_ckpt_path = os.path.join(temp_dir.name, 'model_compressed.pth.tar')

    # Save the state_dict
    torch.save(state_dict_temp, ckpt_path)

    # Compress the state_dict
    with open(ckpt_path, mode="rb") as f_in:
        with open(compress_ckpt_path, mode="wb") as f_out:
            f_out.write(zlib.compress(f_in.read(), zlib.Z_NO_COMPRESSION))
    os.remove(ckpt_path)

    # Archive the full folder
    shutil.make_archive(output_path, 'zip', temp_dir.name)
    temp_dir.cleanup()

    logger.info(f'Model bitstream saved in {output_path}.zip')
    
    return os.path.getsize(output_path + '.zip')


def decompress_and_load_model(args, logger, model, output_dir, name):
    """
    Decompress and load the model.
    """
    model = unwrap_model(model)

    output_path = os.path.join(output_dir, name)
    temp_dir = tempfile.TemporaryDirectory(dir=output_dir)

    target_keys = _get_target_keys(model)
    state_dict_temp = {}
    state_dict_out = {}
    model_state_dict = model.state_dict()

    ckpt_path = os.path.join(temp_dir.name, 'model.pth.tar')
    compress_ckpt_path = os.path.join(temp_dir.name, 'model_compressed.pth.tar')

    shutil.unpack_archive(output_path + '.zip', temp_dir.name)

    # Deompress the state_dict
    with open(compress_ckpt_path, mode="rb") as f_in:
        with open(ckpt_path, mode="wb") as f_out:
            f_out.write(zlib.decompress(f_in.read()))

    # Load the state_dict
    state_dict_temp = torch.load(ckpt_path)
    quant_config = state_dict_temp['quant_config']

    for k in target_keys:
        if args.debug:
            logger.info(f'Loading state: {k}')
        
        device, dtype = model_state_dict[k].device, model_state_dict[k].dtype

        if (k + '_mask') in state_dict_temp:
            mask = _decompress_tensor(state_dict_temp[k + '_mask'], temp_dir.name, k + '_mask').to(torch.bool).to(device)
        else:
            mask = None

        if k in quant_config:
            quant_v = _decompress_tensor(state_dict_temp[k], temp_dir.name, k).to(dtype).to(device)
            quant_v = _restore_zeros(quant_v, mask) if mask is not None else quant_v
            v = quant_v.float() * quant_config[k]['scale'].to(quant_v.device).float()
        else:
            v = state_dict_temp[k].to(dtype).to(device)
            v = _restore_zeros(v, mask) if mask is not None else v
        state_dict_out[k] = v

    # Cleanup the full folder
    temp_dir.cleanup()

    # Load weight
    missing_keys, unexpected_keys = model.load_state_dict(state_dict_out, strict=False)

    logger.info(f'Model restored from {output_path}.zip')
    if args.debug:
        logger.info(f'Missing keys: {missing_keys}')
        logger.info(f'Unexpected keys: {unexpected_keys}')
        
    return os.path.getsize(output_path + '.zip')