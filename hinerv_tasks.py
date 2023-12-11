"""
HiNeRV Task
"""
from utils import *
from losses import compute_loss, compute_metric, compute_regularization


class VideoRegressionTask:
    def __init__(self, args, logger, accelerator, root, training=True, enable_log_eval=False):
        self.root = root
        self.accelerator = accelerator
        self.loss_cfg = [1.0, args.loss[0]] if len(args.loss) == 1 else args.loss
        self.metric_cfg = args.train_metric if training else args.eval_metric
        self.reg_cfg = args.reg
        self.training = training
        self.enable_log_eval = enable_log_eval

        logger.info(f'VideoRegressionTask:')
        logger.info(f'     Root: {self.root}')
        logger.info(f'     Losses: {self.loss_cfg}    Metrics: {self.metric_cfg}    Regularization: {self.reg_cfg}')
        logger.info(f'     Training: {self.training}')
        logger.info(f'     Log evaulation: {self.enable_log_eval}')

    def parse_input(self, loader, batch):
        """
        Parse the input to the model during training/evaluation step
        """
        idx, x = batch
        assert idx.ndim == 2, 'idx should have 2 dimensions with shape [N, 3], where each row is the 3D patch coordinate'
        assert x.ndim == 5,  'x should have 5 dimensions with shape [N, C, T, H, W], where each sample is a 3D patch'

        input = {
            'x': x if self.training else None,
            'idx': idx, 
            'idx_max': loader.dataset.num_patches,
            'batch_size': x.shape[0] * loader.dataset.patch_size[0] / (loader.dataset.num_patches[1] * loader.dataset.num_patches[2]), # in the number of images
            'video_size': loader.dataset.video_size,
            'patch_size': loader.dataset.patch_size
        }

        target = x

        return input, target

    def parse_output(self, loader, batch):
        """
        Parse the output from the model during training/evaluation step
        """
        output = batch
        output = output.contiguous(memory_format=torch.contiguous_format)

        return output

    def compute_loss(self, x, y, model=None):
        total_loss = None
        for i in range(len(self.loss_cfg) // 2):
            weight = float(self.loss_cfg[i * 2])
            loss_type = self.loss_cfg[i * 2 + 1]
            loss = weight * compute_loss(loss_type, x, y)
            total_loss = total_loss + loss if total_loss is not None else loss
        for i in range(len(self.reg_cfg) // 2):
            weight = float(self.reg_cfg[i * 2])
            reg_type = self.reg_cfg[i * 2 + 1]
            loss = weight * compute_regularization(reg_type, model)
            total_loss = total_loss + loss if total_loss is not None else loss
        return total_loss

    def compute_metrics(self, x, y):
        metrics = {}
        for metric_type in self.metric_cfg:
            metrics[metric_type] = compute_metric(metric_type, x, y)
        return metrics

    def step(self, model, loader, batch):
        inputs, targets = self.parse_input(loader, batch)
        outputs = self.parse_output(loader, model(inputs))
        loss = self.compute_loss(outputs, targets, model)
        metrics = self.compute_metrics(outputs, targets)
        return inputs, targets, outputs, loss, metrics

    def log_eval(self, dir_name, inputs, outputs, metrics):
        """
        Log the evaluation outputs
        """
        if not self.enable_log_eval:
            return

        # Use the first metric
        metric = metrics[self.metric_cfg[0]]

        N, C, T, H, W = outputs.shape
        _, H_img, W_img = inputs['video_size']
        T_patch, H_patch, W_patch = inputs['patch_size']

        assert H_img == H_patch and W_img == W_patch, 'Only full image output is supported'

        idx = inputs['idx'].cpu()
        outputs = outputs.detach().cpu()
        metric = metric.detach().cpu()

        output_dir = os.path.join(self.root, dir_name)

        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)

        self.accelerator.wait_for_everyone()

        for n in range(N):

            t, h, w = idx[n].numpy()
            patch_idx = 1 + t * T_patch

            for dt in range(T):
                img_id = f'{patch_idx + dt:04d}_{metric[n, dt].numpy():.2f}'
                img = outputs[n, :, dt, :, :].float().cpu()
                torchvision.utils.save_image(img, os.path.join(output_dir, img_id + '.png'), 'png')


def set_task_args(parser):
    group = parser.add_argument_group('Task parameters')
    group.add_argument('--loss', default='mse', type=str, nargs='+', help='Loss (default: "mse")')
    group.add_argument('--reg', default=[], type=str, nargs='+', help='Regularization')
    group.add_argument('--train-metric', default=['psnr'], type=str, nargs='+', help='Metric (default: "psnr")')
    group.add_argument('--eval-metric', default=['psnr', 'ms_ssim'], type=str, nargs='+', help='Metric (default: "psnr")')
    group.add_argument('--log-eval', type=str_to_bool, default=True, help='Log the output during evaluation.')