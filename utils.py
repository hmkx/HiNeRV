import os
import shutil
import argparse
import yaml
import copy
import logging
import time
import datetime
import uuid

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo as dynamo

import torchvision

import timm
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs

import accelerate

from models import model_cls


"""
Common settings
"""
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
dynamo.config.verbose = False
#dynamo.logging.set_loggers_level(logging.FATAL)
# These are required to make the code work for some cases in torch v2.1
dynamo.config.automatic_dynamic_shapes = False
dynamo.config.optimize_ddp = False


"""
Tools
"""
def get_ckpt_id():
    return time.strftime('%d-%m-%y_%H%M', time.localtime(time.time())) + '-' + str(uuid.uuid4()).split('-')[0]


def str_to_bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise ValueError


def parse_args(parser):

    def set_model_args(parser):
        args_config, remaining = parser.parse_known_args()
        model_cls[args_config.model].set_args(parser)

    # Set script specific args
    _, remaining = parser.parse_known_args()

    # Set model specific args
    set_model_args(parser)

    # Parse both script/model specific args
    args = parser.parse_args()

    return args


def get_output_path(args):
    if args.exp_name:
        exp_name = args.exp_name
    else:
        exp_name = '-'.join([
            args.dataset_name.replace('/', '_').replace(',', '_'),
            args.model,
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            str(uuid.uuid4()).split('-')[0]
        ])
    output_dir = os.path.join(args.output, exp_name) if len(exp_name) else args.output
    output_dir, = accelerate.utils.broadcast_object_list([output_dir], 0)
    return output_dir


def unwrap_model(model):
    model = model._orig_mod if hasattr(model, '_orig_mod') else model # For compiled models
    model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model # For DDP models
    return model


"""
Accerlator & logger
"""
def get_accelerator_logger(args):
    logger = logging.getLogger(__name__)
    accelerator = accelerate.Accelerator(gradient_accumulation_plugin=accelerate.utils.GradientAccumulationPlugin(num_steps=args.grad_accum, adjust_scheduler=False))

    # Prepare output dir & save configurations
    output_dir = get_output_path(args)
    if accelerator.is_main_process:
        os.makedirs(output_dir)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(yaml.safe_dump(args.__dict__, default_flow_style=False))
    accelerator.wait_for_everyone()

    # Log setting
    if accelerator.num_processes > 1:
        formatter = logging.Formatter('[%(asctime)s] ' + 'Rank - ' + str(accelerator.process_index) + ' %(levelname)s - %(message)s')
    else:
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(filename=os.path.join(output_dir, f'rank_{accelerator.process_index}.txt'))
    file_handler.setFormatter(formatter)
    if accelerator.is_main_process:
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)

    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    # World setting
    logger.info(f'Torch version: {torch.__version__}')
    logger.info(f'World size: {accelerator.num_processes}')
    logger.info(f'Rank: {accelerator.process_index}    Local rank: {accelerator.local_process_index}    Device: {accelerator.device}')
    if accelerator.device.type == 'cuda':
        logger.info(f'Device name: {torch.cuda.get_device_name(accelerator.device)}')

    return accelerator, logger, output_dir


"""
Optimizer & scheduler
"""
def get_optimizer_scheduler(args, logger, accelerator, model, loader, param_group_fn=None):

    args = copy.deepcopy(args)

    if args.auto_lr_scaling:
        relative_batch_size = args.batch_size * accelerator.num_processes * args.grad_accum \
                                * math.prod(loader.dataset.patch_size[1:]) / math.prod(loader.dataset.video_size[1:])
        args.lr *= relative_batch_size
        args.warmup_lr *= relative_batch_size
        args.min_lr *= relative_batch_size
        logger.info(f'Autoscale learning rates:')
        logger.info(f'         lr - {args.lr:.2e}')
        logger.info(f'         warmup_lr - {args.warmup_lr:.2e}')
        logger.info(f'         min_lr - {args.min_lr:.2e}')

    args.sched_on_updates = True # This is to match the accelerator option 'step_scheduler_with_optimizer=True'

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(args), param_group_fn=param_group_fn)
    scheduler, _ = create_scheduler_v2(optimizer, **scheduler_kwargs(args), updates_per_epoch=len(loader))
    return optimizer, scheduler


def set_optimizer_args(parser):
    group = parser.add_argument_group('Optimizer & scheduler parameters')
    group.add_argument('--opt', default='sgd', type=str, help='Optimizer')
    group.add_argument('--opt-eps', default=None, type=float, help='Optimizer Epsilon')
    group.add_argument('--opt-betas', default=None, type=float, nargs='+', help='Optimizer Betas (default: None, use opt default)')
    group.add_argument('--momentum', type=float, default=0.9, help='Optimizer momentum (default: 0.9)')
    group.add_argument('--weight-decay', type=float, default=0.0, help='weight decay (default: 0.0)')
    group.add_argument('--max-norm', type=float, default=None, help='Maximum gradient norm for clipping')
    group.add_argument('--norm-type', type=float, default=2.0, help='Gradient norm type for clipping')
    group.add_argument('--sched', default='cosine', type=str, help='LR scheduler')
    group.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    group.add_argument('--warmup-epochs', type=int, default=30, help='epochs to warmup LR, if scheduler supports')
    group.add_argument('--warmup-lr', type=float, default=5e-5, help='warmup learning rate')
    group.add_argument('--min-lr', type=float, default=5e-6, help='lower lr bound for cyclic schedulers that hit 0 (5e-6)')
    group.add_argument('--auto-lr-scaling', type=str_to_bool, default=False,
                       help='enable linear lr scaling with patch training/distributed training/gradient accumulation, where the base batch size is 1 (in frames).')


"""
Others
"""
class CheckpointManager:
    def __init__(self, logger, accelerator, output_dir, metric_key):
        self.logger = logger
        self.accelerator = accelerator
        self.output_dir = output_dir
        self.best_metric_key = metric_key
        self.best_metric = -np.inf
        self.n_checkpoints = 3
        self.checkpoints = []

    def _save_meta(self, path, data):
        with open(os.path.join(path, 'meta.yml'), 'w') as file:
            yaml.dump(data, file)

    def _load_meta(self, path):
        with open(os.path.join(path, 'meta.yml'), 'r') as file:
            data = yaml.safe_load(file)
        return data

    def save(self, epoch, metrics):
        # Save the new checkpoint
        if self.accelerator.is_main_process:
            new_path = os.path.join(self.output_dir, f'checkpoint_{epoch}')
            self.accelerator.save_state(output_dir=new_path)
            self._save_meta(new_path, {'epoch': epoch, 'metric': metrics})

            self.checkpoints.append(new_path)
            while len(self.checkpoints) > self.n_checkpoints:
                shutil.rmtree(self.checkpoints.pop(0))

            # Save the best checkpoint
            if metrics[self.best_metric_key] > self.best_metric:
                self.best_metric = metrics[self.best_metric_key]
                best_path = os.path.join(self.output_dir, 'checkpoint_best')
                self.accelerator.save_state(output_dir=best_path)
                self._save_meta(best_path, {'epoch': epoch, 'metric': metrics})
                self.logger.info(f'Saving new best checkpoint: {best_path}')

        self.accelerator.wait_for_everyone()

    def load(self, path):
        self.accelerator.load_state(path)
        return self._load_meta(path)['epoch']


class BestMetricTracker:
    def __init__(self, logger, accelerator, output_dir, size_metric_keys, save_on_main_process=True):
        assert isinstance(size_metric_keys, dict) and all(isinstance(v, list) and len(v) == 2 for _, v in size_metric_keys.items())
        self.logger = logger
        self.accelerator = accelerator
        self.output_dir = output_dir
        self.size_metric_keys = size_metric_keys
        self.size = {}
        self.best_metric = {}
        self.save_on_main_process = save_on_main_process

    def update(self, key, size, metrics):
        path = os.path.join(self.output_dir, f'{key}.txt')
        size_key, best_metric_key = self.size_metric_keys[key]

        if best_metric_key not in self.best_metric or metrics[self.best_metric_key] > self.best_metric[best_metric_key]:
            self.size[key] = size
            self.best_metric[key] = copy.deepcopy(metrics)

            out_str = ''
            out_list = [[size_key], [f'{size:.4f}']]
            for k, v in self.best_metric[key].items():
                out_str += f'{k}: {v:.4f} '
                out_list[0].append(k)
                out_list[1].append(f'{v:.4f}')
            if size_key == 'size':
                self.logger.info(f'***        {out_str} @ {size/10**6:.2f}M')
            elif size_key == 'bpp':
                self.logger.info(f'***        {out_str} @ {size:.4f}bpp')
            else:
                self.logger.info(f'***        {out_str} @ {size:.4f}')

            if self.accelerator.is_main_process or not self.save_on_main_process:
                os.makedirs(self.output_dir, exist_ok=True)
                np.savetxt(path, out_list, fmt='%s')

        self.accelerator.wait_for_everyone()