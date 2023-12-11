"""
HiNeRV Training Script
"""

from utils import *
from datasets import create_dataset, create_loader, set_dataset_args
from hinerv_tasks import VideoRegressionTask, set_task_args
from hinerv_compress import *

from deepspeed.profiling.flops_profiler import get_model_profile

parser = argparse.ArgumentParser()


# Script
group = parser.add_argument_group('Script parameters')
group.add_argument('--model', default='HiNeRV', type=str, help='Model to build, e.g. HiNeRV.')

group.add_argument('--epochs', type=int, default=300, help='Number of epochs.')
group.add_argument('--eval-epochs', type=int, default=30, help='Number of epochs for every evaluation.')
group.add_argument('--log-epochs', type=int, default=30, help='Number of epochs for every output logging.')

group.add_argument('--prune-epochs', default=0, type=int, help='Number of epochs with pruning.')
group.add_argument('--prune-warmup-epochs', default=0, type=int, help='Number of warmup epochs with pruning.')
group.add_argument('--prune-lr-scale', default=1., type=float, help='Learning rate scale with pruning.')

group.add_argument('--quant-epochs', default=0, type=int, help='Number of epochs with QAT.')
group.add_argument('--quant-warmup-epochs', default=0, type=int, help='Number of warmup epochs with QAT.')
group.add_argument('--quant-lr-scale', default=0.1, type=float, help='Learning rate scale with QAT')

group.add_argument('--output', default='output', type=str, help='Path to output folder.')
group.add_argument('--exp-name', default=None, type=str, help='Suffix of the output folder.')
group.add_argument('--debug', action='store_true', default=False, help='Set debug mode and watch additional outputs.')
group.add_argument('--eval-only', action='store_true', default=False, help='Skip the training stage.')

group.add_argument('--resume', default='', type=str, help='Checkpoint path for resuming training.')
group.add_argument('--bitstream', default='', type=str, help='Bitstream path for evaluation.')
group.add_argument('--bitstream-q', default='', type=str, help='Bitstream quantization level for evaluation.')

group.add_argument('--grad-accum', type=int, default=1, help='Number of gradient accumuluation steps.')

group.add_argument('--profile', action='store_true', default=False, help='Run profiler.')
group.add_argument('--seed', type=int, default=0, help='Random seed.')


# Dataset
set_dataset_args(parser)

# Task
set_task_args(parser)

# Optimizer & scheduler
set_optimizer_args(parser)

# Compression
set_compression_args(parser)


def get_stage_optimizer_scheduler(args, logger, accelerator, model, loader, stage):
    args = copy.deepcopy(args)
    if stage == 'main':
        args.epochs = max(args.epochs, 1)
    elif stage == 'prune':
        args.lr *= args.prune_lr_scale
        args.warmup_lr *= args.prune_lr_scale
        args.min_lr *= args.prune_lr_scale
        args.epochs = max(args.prune_epochs, 1)
        args.warmup_epochs = args.prune_warmup_epochs
    elif stage == 'quant':
        args.lr *= args.quant_lr_scale
        args.warmup_lr *= args.quant_lr_scale
        args.min_lr *= args.quant_lr_scale
        args.epochs = max(args.quant_epochs, 1)
        args.warmup_epochs = args.quant_warmup_epochs
    else:
        raise ValueError
    logger.info(f'Create optimizer & scheduler for {stage} stage')
    return get_optimizer_scheduler(args, logger, accelerator, model, loader)


def train_step(args, logger, suffix, epoch, model, loader, optimizer, scheduler, task, accelerator, log_output):
    start_time = time.time()
    model.train()

    accum_loss = None
    accum_metrices = None
    reduced_loss = None
    reduced_metrics = None
    counts = 0
    samples = 0

    num_updates = epoch * len(loader)
    scheduler.step(epoch=epoch)
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        # Train step
        with accelerator.accumulate(model):
            inputs, _, outputs, loss, metrics = task.step(model, loader, batch)
            mean_loss = loss.mean()
            mean_metrics = {k: v.mean() for k, v in metrics.items()}

            accelerator.backward(mean_loss)
            if accelerator.sync_gradients and args.max_norm:
                accelerator.unscale_gradients(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm, args.norm_type)

            optimizer.step()
            scheduler.step_update(num_updates=num_updates)
            optimizer.zero_grad()
            num_updates += 1

        if i == 0:
            accum_loss = mean_loss.detach()
            accum_metrices = {k: mean_metrics[k].detach() for k in mean_metrics}
        else:
            accum_loss = accum_loss + mean_loss.detach()
            accum_metrices = {k: accum_metrices[k] + mean_metrics[k].detach() for k in mean_metrics}

        counts += 1
        samples += inputs['batch_size'] * accelerator.num_processes

        if (i + 1) % (len(loader) // 4) == 0 or (i + 1) == len(loader):
            # Accumulate loss and metrics globally
            reduced_loss = accelerator.reduce(accum_loss, reduction='mean')
            reduced_metrics = accelerator.reduce(accum_metrices, reduction='mean')
            # Logging loss & metrics
            log_msg = f'Train' + ('' if not suffix else f' ({suffix})') +  f' - Epoch {epoch} [{i + 1}/{len(loader)}]'
            log_msg = log_msg + f'    lr: {np.mean([group["lr"] for group in optimizer.param_groups]):.2e}'
            log_msg = log_msg + f'    img/s: {samples / (time.time() - start_time):.2f}'
            log_msg = log_msg + f'    loss: {reduced_loss.item() / counts:.4f}'
            for k, v in reduced_metrics.items():
                log_msg = log_msg + f'    {k}: {v.item() / counts:.4f}'
            logger.info(log_msg)

        # Logging outputs
        if log_output:
            task.log_eval(dir_name=f'{epoch}_{suffix.lower()}', inputs=inputs, outputs=outputs, metrics=metrics)

    return reduced_loss.item() / counts, {k: v.item() / counts for k, v in reduced_metrics.items()}


def eval_step(args, logger, suffix, epoch, model, loader, task, accelerator, log_output):
    start_time = time.time()
    model.eval()

    accum_loss = None
    accum_metrices = None
    reduced_loss = None
    reduced_metrics = None
    counts = 0
    samples = 0

    for i, batch in enumerate(loader):
        # Eval step
        with torch.no_grad():
            inputs, _, outputs, loss, metrics = task.step(model, loader, batch)
            mean_loss = loss.mean()
            mean_metrics = {k: v.mean() for k, v in metrics.items()}

        if i == 0:
            accum_loss = mean_loss
            accum_metrices = {k: mean_metrics[k] for k in mean_metrics}
        else:
            accum_loss = accum_loss + mean_loss
            accum_metrices = {k: accum_metrices[k] + mean_metrics[k] for k in mean_metrics}

        counts += 1
        samples += inputs['batch_size'] * accelerator.num_processes

        if (i + 1) % (len(loader) // 4) == 0 or (i + 1) == len(loader):
            # Accumulate loss and metrics globally
            reduced_loss = accelerator.reduce(accum_loss, reduction='mean')
            reduced_metrics = accelerator.reduce(accum_metrices, reduction='mean')
            # Logging loss & metrics
            log_msg = f'Eval' + ('' if not suffix else f' ({suffix})') +  f' - [{i + 1}/{len(loader)}]'
            log_msg = log_msg + f'    img/s: {samples / (time.time() - start_time):.2f}'
            log_msg = log_msg + f'    loss: {reduced_loss.item() / counts:.4f}'
            for k, v in reduced_metrics.items():
                log_msg = log_msg + f'    {k}: {v.item() / counts:.4f}'
            logger.info(log_msg)

        # Logging outputs
        if log_output:
            task.log_eval(dir_name=f'{epoch}' + ('' if not suffix else f'_{suffix.lower()}'), inputs=inputs, outputs=outputs, metrics=metrics)

    return reduced_loss.item() / counts, {k: v.item() / counts for k, v in reduced_metrics.items()}


def do_eval(args, epoch, num_epochs): 
    return (epoch + 1) % args.eval_epochs == 0 or (epoch + 1) == num_epochs


def do_log(args, epoch, num_epochs):
    return (args.log_epochs > 0 and (epoch + 1) % args.log_epochs == 0) or \
           (args.log_epochs == -1 and (epoch + 1) == num_epochs)


def main():
    start_time = datetime.datetime.now()

    # Set & store args
    args = parse_args(parser)

    # Set accelerator, logger, output dir
    accelerator, logger, output_dir = get_accelerator_logger(args)
    logger.info(f'Output dir: {output_dir}')

    # Set seed
    logger.info(f'Set seed: {args.seed}')
    accelerate.utils.set_seed(args.seed, device_specific=True)

    # Task setting
    train_task = VideoRegressionTask(args, logger, accelerator, root=os.path.join(output_dir, 'train_output'), 
                                     training=True, enable_log_eval=False)
    eval_task = VideoRegressionTask(args, logger, accelerator, root=os.path.join(output_dir, 'eval_output'),
                                    training=False, enable_log_eval=args.log_eval)

    # Optimize steps
    if accelerator.state.dynamo_plugin.backend != accelerate.utils.DynamoBackend.NO:
        # Some precision issues with max-autotune exist, so the default cfg is used here.
        train_task.step = torch.compile(train_task.step) #, **accelerator.state.dynamo_plugin.to_kwargs())
        eval_task.step = torch.compile(eval_task.step) #, **accelerator.state.dynamo_plugin.to_kwargs())

    # Datasets & Loaders
    logger.info(f'Create training dataset & loader: {os.path.join(args.dataset, args.dataset_name)}')
    train_dataset = create_dataset(logger=logger, args=args, training=True)
    train_loader = create_loader(logger=logger, args=args, training=True, dataset=train_dataset)

    logger.info(f'Create evaluation dataset & loader: {os.path.join(args.dataset, args.dataset_name)}')
    eval_dataset = create_dataset(logger=logger, args=args, training=False)
    eval_loader = create_loader(logger=logger, args=args, training=False, dataset=eval_dataset)

    # Model
    logger.info(f'Create model: {args.model}')
    model = model_cls[args.model].build_model(args, logger, train_task.parse_input(train_loader, next(iter(train_loader)))[0])
    logger.info(f'Model info:')
    logger.info(model)

    # Compute number of parameters & MACs
    logger.info(f'Number of parameters:')
    num_params = model.get_num_parameters()
    for k, v in num_params.items():
        logger.info(f'    {k}: {v / 10**6:.2f}M')

    # Compute MACs
    with torch.no_grad():
        model.eval()
        _, macs, _ = get_model_profile(model=model, 
                                       args=[eval_task.parse_input(eval_loader, next(iter(eval_loader)))[0]],
                                       print_profile=False, detailed=False, warm_up=1, as_string=False)
        macs /= args.eval_batch_size
    logger.info(f'MACs: {macs / 10 ** 9 :.2f}G')

    # Pruning & quanization
    initial_parametrizations(args, logger, model)

    # Place model & loaders
    model, train_loader, eval_loader = accelerator.prepare(model, train_loader, eval_loader)

    # Optimizer & scheduler
    logger.info(f'Create scheduler: {args.sched}')
    opt_sch, prune_opt_sch, quant_opt_sch = [get_stage_optimizer_scheduler(args, logger, accelerator, model, train_loader, stage) for stage in ['main', 'prune', 'quant']]

    # Place optimizer & scheduler
    opt_sch, prune_opt_sch, quant_opt_sch = accelerator.prepare(*opt_sch), accelerator.prepare(*prune_opt_sch), accelerator.prepare(*quant_opt_sch)

    # Restoring training state
    checkpoint_manager = CheckpointManager(logger, accelerator, os.path.join(output_dir, 'checkpoints'), args.eval_metric[0])
    if args.resume:
        logger.info(f'Restore from training state: {args.resume}')
        epoch = checkpoint_manager.load(args.resume) + int(not args.eval_only)
    else:
        epoch = 0

    # Restore from bitstream
    if args.bitstream:
        logger.info(f'Restore model weights from bitstream')
        num_bytes = decompress(args, logger, accelerator, model, os.path.join(args.bitstream, 'bitstreams'), int(args.bitstream_q))
        bits_per_pixel = num_bytes * 8 / np.prod(eval_dataset.video_size)

        logger.info(f'Compressed model size: {num_bytes / 10**6:.2f}MB')
        logger.info(f'Bits Per Pixel (BPP): {bits_per_pixel:.4f}')


    if not args.eval_only:
        # Start training
        logger.info(f'Start training for {args.epochs + args.prune_epochs + args.quant_epochs} epochs.')
        logger.info(f'    Number of training epochs: {args.epochs}')
        logger.info(f'    Number of pruning fine-tuning epochs: {args.prune_epochs}')
        logger.info(f'    Number of quant fine-tuning epochs: {args.quant_epochs}')

        best_metrics = BestMetricTracker(logger, accelerator, os.path.join(output_dir, 'results'), 
                                        {**{k: ['size', args.eval_metric[0]] for k in ['full', 'pruned', 'qat']},
                                         **{f'Q{quant_level}': ['bpp', args.eval_metric[0]] for quant_level in sorted(args.quant_level, reverse=True)}})
        zeros, total = get_sparsity(model)

        # Main Training loop
        if epoch < args.epochs:
            logger.info(f'Start main training for {args.epochs} epochs.')
        while epoch < args.epochs:
            # Training
            train_step(args, logger, '', epoch, model, train_loader, opt_sch[0], opt_sch[1], train_task, accelerator, False)

            # Evaluation
            if do_eval(args, epoch, args.epochs):
                _, metrics = eval_step(args, logger, '', epoch, model, eval_loader, eval_task, accelerator,
                                       do_log(args, epoch, args.epochs))
                best_metrics.update('full', num_params['All'], metrics)
                checkpoint_manager.save(epoch, metrics)

            epoch += 1


        # Pruning fine-tuning loop
        if epoch < args.epochs + args.prune_epochs:
            logger.info(f'Start pruning fine-tuning for {args.prune_epochs} epochs.')
            zeros, total = set_pruning(args, logger, model, args.prune_ratio, args.prune_weight)
            logger.info(f'Number of pruned parameters: {zeros}    total: {total}')
            logger.info(f'Sparsity: {zeros / total:.4f}')

        while epoch < args.epochs + args.prune_epochs:
            prune_epoch = epoch - args.epochs
            train_step(args, logger, 'Pruning', prune_epoch, model, train_loader, prune_opt_sch[0], prune_opt_sch[1], train_task, accelerator, False)
            
            # Evaluation
            if (prune_epoch + 1) % args.eval_epochs == 0 or (prune_epoch + 1) == args.prune_epochs:
                _, metrics = eval_step(args, logger, 'Pruning', epoch, model, eval_loader, eval_task, accelerator,
                                       do_log(args, prune_epoch, args.prune_epochs))
                best_metrics.update('pruned', num_params['All'], metrics)
                checkpoint_manager.save(epoch, metrics)

            epoch += 1


        # QAT fine-tuning loop
        if epoch < args.epochs + args.prune_epochs + args.quant_epochs:
            logger.info(f'Start quant fine-tuning for {args.quant_epochs} epochs.')
            logger.info(f'Quantization level: {sorted(args.quant_level, reverse=True)}')
            logger.info(f'Quantization noise: {args.quant_noise}')

        # Set QAT configs
        set_quantization(args, logger, model, min(args.quant_level), args.quant_noise, args.quant_ste)

        while epoch < args.epochs + args.prune_epochs + args.quant_epochs:
            quant_epoch = epoch - args.epochs - args.prune_epochs
            train_step(args, logger, 'QAT', quant_epoch, model, train_loader, quant_opt_sch[0], quant_opt_sch[1], train_task, accelerator, False)

            # Evaluation
            if do_eval(args, quant_epoch, args.quant_epochs):
                qat_state_dict = copy.deepcopy(model.state_dict())
                for quant_level in sorted(args.quant_level, reverse=True):
                    # Compress bitstream
                    logger.info(f'Compress model weights into bitstream')
                    logger.info(f'***  Quant level: {quant_level}bits')
                    logger.info(f'***  Sparsity: {zeros / total:.4f}')

                    num_bytes = compress_bitstream(args, logger, accelerator, model, os.path.join(output_dir, 'bitstreams'), quant_level)
                    bits_per_pixel = num_bytes * 8 / np.prod(eval_dataset.video_size)

                    logger.info(f'Compressed model size: {num_bytes / 10**6:.2f}MB')
                    logger.info(f'Bits Per Pixel (BPP): {bits_per_pixel:.4f}')

                    # Set model weights to zero for ensuring the correctness
                    set_zero(model)

                    # Decompress bitstream
                    logger.info(f'Decompress model weights from bitstream')
                    decompress(args, logger, accelerator, model, os.path.join(output_dir, 'bitstreams'), quant_level)

                    # Evaluation
                    _, metrics = eval_step(args, logger, f'Q{quant_level}', epoch, model, eval_loader, eval_task, accelerator,
                                           do_log(args, quant_epoch, args.quant_epochs))
                    best_metrics.update(f'Q{quant_level}', bits_per_pixel, metrics)

                    if quant_level == min(args.quant_level): # use the least bitwidth for checkpointing metric
                        checkpoint_manager.save(epoch, metrics)

                    # Restore from the checkpoint
                    model.load_state_dict(qat_state_dict)

            epoch += 1

        # Complete training
        train_time = datetime.datetime.now() - start_time

        logger.info(f'Training completed in: {train_time}')
        logger.info(f'Output are located in: {output_dir}')

    else:
        # Evaluation
        _, metrics = eval_step(args, logger, None, epoch, model, eval_loader, eval_task, accelerator, True)


if __name__ == '__main__':
    main()