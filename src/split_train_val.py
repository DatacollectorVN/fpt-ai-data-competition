""" 
Source: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#preparing-the-dataset
usage: partition_dataset.py [-h] [-i IMAGEDIR] [-o OUTPUTDIR] [-r RATIO] [-x]

Partition dataset of images into training and testing sets

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGEDIR, --imageDir IMAGEDIR
                        Path to the folder where the image dataset is stored. If not specified, the CWD will be used.
  -o OUTPUTDIR, --outputDir OUTPUTDIR
                        Path to the output folder where the train and test dirs should be created. Defaults to the same directory as IMAGEDIR.
  -r RATIO, --ratio RATIO
                        The ratio of the number of test images over the total number of images. The default is 0.1.
  -x, --xml             Set this flag if you want the xml annotation files to be processed and copied over.
"""
import os
from tqdm import tqdm
from shutil import copyfile
import argparse
import math
import random


def iterate_dir(source, dest, ratio):
    source = source.replace('\\', '/')
    dest = dest.replace('\\', '/')
    train_dir_img = os.path.join(dest, 'images', 'train')
    test_dir_img = os.path.join(dest, 'images', 'val')
    train_dir_label = os.path.join(dest, 'labels', 'train')
    test_dir_label = os.path.join(dest, 'labels', 'val')

    if not os.path.exists(train_dir_img):
        os.makedirs(train_dir_img)
    if not os.path.exists(test_dir_img):
        os.makedirs(test_dir_img)
    if not os.path.exists(train_dir_label):
        os.makedirs(train_dir_label)
    if not os.path.exists(test_dir_label):
        os.makedirs(test_dir_label)

    images = [f for f in os.listdir(source)]

    num_images = len(images)
    num_test_images = math.ceil(ratio*num_images)

    print('Process with val dataset')
    for i in tqdm(range(num_test_images)):
        idx = random.randint(0, len(images)-1)
        filename = images[idx]
        copyfile(os.path.join(source, filename),
                 os.path.join(test_dir_img, filename))
        copyfile(os.path.join(source.replace('images', 'labels'), filename.split('.')[0] + '.txt'),
                 os.path.join(test_dir_label, filename.split('.')[0] + '.txt'))
        images.remove(images[idx])

    print('Process with train dataset')
    for filename in tqdm(images):
        copyfile(os.path.join(source, filename),
                 os.path.join(train_dir_img, filename))
        copyfile(os.path.join(source.replace('images', 'labels'), filename.split('.')[0] + '.txt'),
                 os.path.join(train_dir_label, filename.split('.')[0] + '.txt'))


def main():

    # Initiate argument parser
    parser = argparse.ArgumentParser(description="Partition dataset of images into training and testing sets",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--imageDir', default='./data/dataset/images/val',
        help='Path to the folder where the image dataset is stored. If not specified, the CWD will be used.',
        type=str,
    )
    parser.add_argument(
        '-o', '--outputDir', default='./data/val_new',
        help='Path to the output folder where the train and test dirs should be created. '
             'Defaults to the same directory as IMAGEDIR.',
        type=str,
    )
    parser.add_argument(
        '-r', '--ratio',
        help='The ratio of the number of test images over the total number of images. The default is 0.1.',
        default=0.5,
        type=float)
    args = parser.parse_args()

    if args.outputDir is None:
        args.outputDir = args.imageDir

    # Now we are ready to start the iteration
    iterate_dir(args.imageDir, args.outputDir, args.ratio)


if __name__ == '__main__':
    main()
    print('DONE')
