import sys
import logging
import argparse
from pathlib import Path
import cv2
import yaml
import torch
import numpy as np
from tqdm import tqdm
from models.yolo import Model
from utils.callbacks import Callbacks
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.torch_utils import intersect_dicts, select_device
from utils.general import labels_to_class_weights, init_seeds, \
    check_dataset, check_img_size, check_requirements, check_file,\
    check_yaml, check_suffix, colorstr, set_logging


FILE = Path(__file__).resolve()
sys.path.append(FILE.parents[0].as_posix())
LOGGER = logging.getLogger(__name__)
TYPE = 'ALL'


def parser(known=False):
    args = argparse.ArgumentParser()
    args.add_argument('--data_cfg', type=str,
                      default='config/data_cfg.yaml', help='dataset config file path')
    args.add_argument('--batch-size', type=int, default=4, help='batch size')
    args.add_argument('--cache', type=str, nargs='?', const='ram',
                      help='--cache images in "ram" (default) or "disk"')
    args.add_argument('--device', default='',
                      help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args.add_argument('--workers', type=int, default=8,
                      help='maximum number of dataloader workers')
    args = args.parse_known_args()[0] if known else args.parse_args()

    with open(Path('config') / 'train_cfg.yaml') as f:
        temp_args: dict = yaml.safe_load(f)

    keys = list(temp_args.keys())
    already_keys = list(args.__dict__.keys())

    for key in keys:
        if key not in already_keys:
            args.__setattr__(key, temp_args[key])

    return args


def train(hyp, args, device, callbacks):
    [epochs, batch_size, pretrained_path,
     data_cfg, model_cfg, resume, workers] = args.epochs, \
        args.batch_size, args.weights, \
        args.data_cfg, args.model_cfg, \
        args.resume, args.workers

    # Hyper parameters
    if isinstance(hyp, str):
        with open(hyp) as f:
            hyp: dict = yaml.safe_load(f)  # load hyper parameter dict
    LOGGER.info(colorstr('Hyper parameters: ') +
                ', '.join(f'{k}={v}' for k, v in hyp.items()))

    """
    ===============================
        Config
    ===============================
    """
    init_seeds(0)

    data_dict = check_dataset(data_cfg)
    train_path = data_dict['train']
    num_class = int(data_dict['num_class'])  # number of classes
    class_name = data_dict['names']

    """
    ===============================
        Model
    ===============================
    """
    check_suffix(pretrained_path, '.pt')
    use_pretrained = pretrained_path.endswith('.pt')
    check_point = None
    if use_pretrained:
        check_point = torch.load(
            pretrained_path, map_location=device)  # load checkpoint

        # create model
        model = Model(model_cfg or check_point['model'].yaml, ch=3, nc=num_class, anchors=hyp.get(
            'anchors')).to(device)
        exclude = ['anchor'] if (model_cfg or hyp.get(
            'anchors')) and not resume else []  # exclude keys
        # checkpoint state_dict as FP32
        csd = check_point['model'].float().state_dict()
        csd = intersect_dicts(csd, model.state_dict(),
                              exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(
            f'Transferred {len(csd)}/{len(model.state_dict())} items from {pretrained_path}')  # report
    else:
        # create model
        model = Model(model_cfg, ch=3, nc=num_class,
                      anchors=hyp.get('anchors')).to(device)

    start_epoch = 0

    # Image sizes
    grid_size = max(int(model.stride.max()), 32)
    # verify img_size is gs-multiple
    img_size = check_img_size(args.img_size, grid_size, floor=grid_size * 2)

    # Train Loader
    train_loader, dataset = create_dataloader(
        train_path, img_size, batch_size, grid_size, hyp=hyp, augment=True, workers=workers, prefix=colorstr('Train: '))

    max_label_class = int(np.concatenate(dataset.labels, 0)[:, 0].max())
    num_batches = len(train_loader)
    assert max_label_class < num_class, \
        'Label class {} exceeds num_class={} in {}. Possible class labels are 0-{}'.format(
            max_label_class,
            num_class,
            data_cfg,
            num_class - 1
        )

    if not resume:
        labels = np.concatenate(dataset.labels, 0)

        # Anchors
        if not args.noautoanchor:
            check_anchors(dataset, model=model,
                          thr=hyp['anchor_t'], imgsz=img_size)
        model.half().float()  # pre-reduce anchor precision

    callbacks.run('on_pretrain_routine_end')

    # Model parameters
    model.nc = num_class  # attach number of classes to model
    model.hyp = hyp  # attach hyper parameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, num_class).to(
        device) * num_class  # attach class weights
    model.names = class_name

    for epoch in range(start_epoch, epochs):
        path_images = 'imgs_augment/mosaic_mixup/images'
        plot_bar = enumerate(train_loader)
        plot_bar = tqdm(plot_bar, total=num_batches)
        for i, (img_batch, targets, paths, _) in plot_bar:
            # img = img_batch.
            for img in img_batch:
                # print(paths[0])
                img = img.permute(1, 2, 0)
                img = img.cpu().detach().numpy()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                filename = paths[0].split('/')[-1].split('.')[0]

                outputname = filename + '_Mosaic_AutoAug_' + \
                    TYPE + str(epoch) + '.jpg'
                path_image = path_images + "/" + outputname

                cv2.imwrite(path_image, img)

            path_labels = path_image.replace(
                'images', 'labels').replace('jpg', 'txt')
            labels = targets[:, 1:].cpu().detach().numpy()
            with open(path_labels, 'w') as f:
                for label in labels:
                    label = tuple(label)
                    f.write(('%g ' * len(label)).rstrip() % label + '\n')
            f.close()


def main(args, callbacks=Callbacks()):

    set_logging()
    print(colorstr('Train: ') +
          ', '.join(f'{k}={v}' for k, v in vars(args).items()))

    # Check requirements
    check_requirements(requirements=FILE.parent /
                       'requirements.txt', exclude=['thop'])

    args.data_cfg = check_file(args.data_cfg)
    args.model_cfg = check_yaml(args.model_cfg)
    args.hyp = check_yaml(args.hyp)
    assert len(args.model_cfg) or len(
        args.weights), 'either --cfg or --weights must be specified'

    # DDP mode
    device = select_device(args.device, batch_size=args.batch_size)
    print(device)

    train(args.hyp, args, device, callbacks)


if __name__ == "__main__":
    main(args=parser())
