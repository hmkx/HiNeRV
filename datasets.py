"""
Datasets & loaders
"""

from utils import *


class VideoDataset(torch.utils.data.Dataset):
    """
    The dataset class for loading videos. Each dataset instance loads all video frames in a folder.
    It optionally performs frame cropping and resizing, and loading video frames in 3D patches.
    It also implemented caching for fast random patch loading. Caching can be set to 'image'/'patch' level.
    """
    def __init__(self, logger, root, name, crop=[-1, -1], resize=[-1, -1], patch_size=[1, -1, -1], cached='none'):
        self.logger = logger
        self.root = os.path.expanduser(root)
        self.name = name
        self.img_paths = sorted([f for f in os.listdir(os.path.join(self.root, self.name)) if not f.startswith(".")])

        self.raw_size = self.get_raw_size()
        self.crop = tuple(crop[d] if crop[d] != -1 else self.raw_size[d] for d in range(2))
        self.resize = tuple(resize[d] if resize[d] != -1 else self.crop[d] for d in range(2))

        self.video_size = (len(self.img_paths), self.resize[0], self.resize[1])
        self.patch_size = tuple(patch_size[d] if patch_size[d] != -1 else self.video_size[d] for d in range(3))

        assert all(self.video_size[d] % self.patch_size[d] == 0 for d in range(3))
        self.num_patches = tuple(self.video_size[d] // self.patch_size[d] for d in range(3))

        assert cached in ['none', 'image', 'patch']
        self.cached = cached
        self.load_cache()

        self.logger.info(f'VideoDataset:')
        self.logger.info(f'     root: {self.root}    name: {self.name}    number of images: {len(self.img_paths)}')
        self.logger.info(f'     video_size: {self.video_size}    patch_size: {self.patch_size}    num_patches: {self.num_patches}')
        self.logger.info(f'     cached: {self.cached}')

    def load_cache(self):
        """
        Caching the images/patches.
        """
        if self.cached == 'image' or self.cached == 'patch':
            self.image_cached = self.load_all_images()
        else:
            self.image_cached = None

        if self.cached == 'patch':
            self.patch_cached = self.load_all_patches()
            self.image_cached = None
        else:
            self.patch_cached = None

        if self.cached == 'patch':
            self.image_cached = None

    def get_raw_size(self):
        """
        Get the original video frame size, i.e., before cropping/resizing.
        This assume that all frames have the same size.
        """
        img = torchvision.io.read_image(os.path.join(self.root, self.name, self.img_paths[0]))
        return img.shape[1:3]

    def load_image(self, idx):
        """
        For loading single image (not cached).
        """
        assert isinstance(idx, int)
        img = torchvision.io.read_image(os.path.join(self.root, self.name, self.img_paths[idx]))
        img = torchvision.transforms.functional.center_crop(img, self.crop)
        img = torchvision.transforms.functional.resize(img, self.resize, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True)
        return img

    def load_patch(self, idx):
        """
        For loading single 3D patch (not cached).
        """
        assert isinstance(idx, tuple) and len(idx) == 3
        patches = []
        h = idx[1] * self.patch_size[1]
        w = idx[2] * self.patch_size[2]
        for dt in range(self.patch_size[0]):
            t = idx[0] * self.patch_size[0] + dt
            image = self.image_cached[t] if self.image_cached is not None else self.load_image(t)
            patch = image[:, None, h: h + self.patch_size[1], w: w + self.patch_size[2]]
            patches.append(patch)
        return torch.concatenate(patches, dim=1)

    def load_all_images(self):
        images = {}
        for t in range(self.video_size[0]):
            images[t] = self.load_image(t)
        return images

    def load_all_patches(self):
        patches = {}
        for t in range(self.num_patches[0]):
            for h in range(self.num_patches[1]):
                for w in range(self.num_patches[2]):
                    patches[(t, h, w)] = self.load_patch((t, h, w))
        return patches

    def get_image(self, idx):
        """
        For getting single image (either cached or not).
        """
        assert isinstance(idx, int)
        return self.image_cached[idx] if self.image_cached is not None else self.load_image(idx)

    def get_patch(self, idx):
        """
        For getting single 3D patch (either cached or not).
        """
        assert isinstance(idx, tuple) and len(idx) == 3
        return self.patch_cached[idx] if self.patch_cached is not None else self.load_patch(idx)

    def __len__(self):
        return math.prod(self.num_patches)

    def __getitem__(self, idx):
        assert isinstance(idx, int)
        idx_thw = (idx // (self.num_patches[1] * self.num_patches[2]),
                    (idx % (self.num_patches[1] * self.num_patches[2])) // self.num_patches[2],
                    (idx % (self.num_patches[1] * self.num_patches[2])) % self.num_patches[2])
        patch = self.get_patch(idx_thw)
        return torch.tensor(idx_thw, dtype=int), torch.clone(patch).float() / 255.


def create_dataset(args, logger, training):
    """
    Create the dataset instance. Only apply the patch configuration for trainset.
    """
    return VideoDataset(logger, root=args.dataset, name=args.dataset_name,
                        crop=args.crop_size, resize=args.input_size,
                        patch_size=args.patch_size if training else [args.patch_size[0], -1, -1],
                        cached=args.cached)


def create_loader(args, logger, training, dataset):
    """
    Create the dataset loader.
    """
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.eval_batch_size if not training else args.batch_size,
        shuffle=training,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        persistent_workers=True,
        drop_last=True,
        prefetch_factor=2
    )
    return loader


def set_dataset_args(parser):
    group = parser.add_argument_group('Dataset parameters')
    group.add_argument('--dataset', default='~/Data', type=str, help='root path of datasets')
    group.add_argument('--dataset-name', default='Bunny', type=str, help='dataset name. dataset/dataset_name should be the path for image folder.')
    group.add_argument('--crop-size', default=[-1, -1], type=int, nargs='+', help='crop size (before resizing to the input)')    
    group.add_argument('--input-size', default=[-1, -1], type=int, nargs='+', help='input size (scaling that apply after cropping)')
    group.add_argument('--patch-size', default=[1, -1, -1], type=int, nargs='+', help='patch size (apply after cropping and scaling)')
    group.add_argument('--cached', type=str, default='none', help='cache setting for the datasets')
    group.add_argument('--batch-size', type=int, default=1, help='Training batch size')
    group.add_argument('--eval-batch-size', type=int, default=1, help='Evaluation batch size')
    group.add_argument('--workers', type=int, default=2, help='Number of workers for dataloader (default: 2)')
    group.add_argument('--pin-mem', type=str_to_bool, default=True, help='Pin memory for dataloader')