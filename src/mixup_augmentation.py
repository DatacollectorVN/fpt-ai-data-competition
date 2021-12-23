import random
import cv2
import os
import glob
import numpy as np
from tqdm import tqdm

# Parrameters
LABEL_DIR = './imgs_augment/flip/labels'
IMG_DIR = './imgs_augment/flip/images'
NUMBER_IMAGES = 800
TYPE_TRANSFORM = 'MIXUP'


def main():
    img_paths, annos = get_dataset(LABEL_DIR, IMG_DIR)
    for index in tqdm(range(NUMBER_IMAGES)):
        img_list = []
        anno_list = []
        idxs = random.sample(range(len(annos)), 2)
        for i, idx in enumerate(idxs):
            img = img_paths[idx]
            img_annos = annos[idx]
            img = cv2.imread(img)
            img_list.append(img)
            anno_list.append(img_annos)
        img_mixup, anno_mixup = mixup(
            img_list[0], anno_list[0], img_list[1], anno_list[1])
        bbox_image1 = []
        for obj_list in anno_list[0]:
            obj = obj_list.rstrip('\n').split(' ')
            bbox_image1.append([int(obj[0]), float(obj[1]),
                                float(obj[2]), float(obj[3]), float(obj[4])])
        bbox_image2 = []
        for obj_list in anno_list[1]:
            obj = obj_list.rstrip('\n').split(' ')
            bbox_image2.append([int(obj[0]), float(obj[1]),
                                float(obj[2]), float(obj[3]), float(obj[4])])
        bbox_list = []
        for box1 in bbox_image1:
            for box2 in bbox_image2:
                bbox_list.append(intersection_over_union(box1[1:], box2[1:]))
        if max(bbox_list) < 0.05:
            # Get random string code: '7b7ad245cdff75241935e4dd860f3bad'
            letter_code = random_chars(8)
            file_name = img_paths[idxs[0]].split(
                '/')[-1].rsplit('.', 1)[0]
            cv2.imwrite("imgs_augment/mixup/images/{}_{}_{}.jpg".format(file_name, TYPE_TRANSFORM,
                        letter_code), img_mixup, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with open("imgs_augment/mixup/labels/{}_{}_{}.txt".format(file_name, TYPE_TRANSFORM, letter_code), "w") as outfile:
                outfile.write("\n".join(line for line in anno_mixup))
        else:
            pass


def mixup(im, labels, im2, labels2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels


def get_dataset(label_dir, img_dir):
    '''
    Params:
        - label_dir <type: list>: Path to label include annotation of images
        - img_dir <type: list>: Path to folder contain images
    Return <type: list>: List of images path and labels
    '''
    img_paths = []
    labels = []
    for label_file in glob.glob(os.path.join(label_dir, '*.txt')):
        label_name = label_file.split('/')[-1].rsplit('.', 1)[0]
        f = open(label_file, 'r')
        obj_lists = f.readlines()
        img_path = os.path.join(img_dir, f'{label_name}.jpg')

        boxes = []
        for obj in obj_lists:
            if obj.endswith('\n'):
                boxes.append(obj[:-1])
            else:
                boxes.append(obj)
        if not boxes:
            continue
        img_paths.append(img_path)
        labels.append(boxes)
    return img_paths, labels


def random_chars(number_char):
    # Get random string code: '7b7ad245cdff75241935e4dd860f3bad'
    letter_code = 'abcdefghijklmnopqrstuvwxyz0123456789'
    return ''.join(random.choice(letter_code) for _ in range(number_char))


def xywh2xyxy(x, y, w, h):
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return x1, y1, x2, y2


def intersection_over_union(boxes_preds, boxes_labels):
    # https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/metrics/iou.py
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
    Returns:
        tensor: Intersection over union for all examples
    """

    box1_xc1 = boxes_preds[0]
    box1_yc1 = boxes_preds[1]
    box1_w = boxes_preds[2]
    box1_h = boxes_preds[3]
    box1_x1, box1_y1, box1_x2, box1_y2 = xywh2xyxy(
        box1_xc1, box1_yc1, box1_w, box1_h)
    box2_xc1 = boxes_labels[0]
    box2_yc1 = boxes_labels[1]
    box2_w = boxes_labels[2]
    box2_h = boxes_labels[3]
    box2_x1, box2_y1, box2_x2, box2_y2 = xywh2xyxy(
        box2_xc1, box2_yc1, box2_w, box2_h)

    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)

    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


if __name__ == '__main__':
    main()
    print('DONE ✅')
