'''
Source: https://gist.github.com/BIGBALLON/cb6ab73f6aaaa068ab6756611bb324b2
'''

from PIL import Image, ImageOps
import glob
import os
import shutil

# Params
IMAGE_DIR = './data/dataset/images/public_test'
LABEL_DIR = './data/dataset/labels/public_test'
IMAGE_SIZE = (1280, 720)


def main():
    if os.path.exists(IMAGE_DIR + '_padding'):
        shutil.rmtree(IMAGE_DIR + '_padding')
    os.makedirs(IMAGE_DIR + '_padding')
    if os.path.exists(LABEL_DIR + '_padding'):
        shutil.rmtree(LABEL_DIR + '_padding')
    os.makedirs(LABEL_DIR + '_padding')
    for image_file in glob.glob(os.path.join(IMAGE_DIR, '*.jpg')):
        img = Image.open(image_file)
        label_file = image_file.replace('images', 'labels')
        label_file = label_file.replace('jpg', 'txt')
        img_new, annos_list = resize_with_padding(
            img, label_file, IMAGE_SIZE)
        print('\rSucceesed with file:', image_file.split(
            '/')[-1], end=' ', flush=True)
        img_new.save(IMAGE_DIR + '_padding/' + image_file.split('/')
                     [-1].split('.')[0] + '_PADDING.jpg', quality=80)
        with open(LABEL_DIR + '_padding/' + label_file.split('/')[-1].split('.')[0] + '_PADDING.txt', "w") as outfile:
            outfile.write("\n".join(line for line in annos_list))


def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width -
               pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def resize_with_padding(img, anno, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width -
               pad_width, delta_height - pad_height)
    f = open(anno, 'r')
    lines = f.readlines()
    annos_list = []
    for line in lines:
        obj = line.rstrip('\n').split(' ')
        obj_new = '{} {} {} {} {}'.format(
            int(obj[0]), (float(obj[1])*img.size[0]+pad_width)/expected_size[0], (float(obj[2])*img.size[1]+pad_height) /
            expected_size[1], float(
                obj[3])*img.size[0]/expected_size[0], float(obj[4])*img.size[1]/expected_size[1])
        annos_list.append(obj_new)

    return ImageOps.expand(img, padding), annos_list


if __name__ == "__main__":
    main()
    print('DONE')
