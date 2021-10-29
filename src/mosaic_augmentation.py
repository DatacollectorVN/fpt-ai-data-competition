'''Source: https://github.com/jason9075/opencv-mosaic-data-aug'''

import random
import cv2
import os
import glob
import numpy as np

# Parrameters
OUTPUT_SIZE = (720, 1280)  # Height, Width
SCALE_RANGE = (0.4, 0.6)  # if height or width lower than this scale, drop it.
FILTER_TINY_SCALE = 1 / 100
LABEL_DIR = '../data/labels/train'
IMG_DIR = '../data/images/train'
NUMBER_IMAGES = 200


def main():
    img_paths, annos = get_dataset(LABEL_DIR, IMG_DIR)
    for index in range(NUMBER_IMAGES):
        idxs = random.sample(range(len(annos)), 4)
        new_image, new_annos, path = update_image_and_anno(img_paths, annos,
                                                           idxs,
                                                           OUTPUT_SIZE, SCALE_RANGE,
                                                           filter_scale=FILTER_TINY_SCALE)

        # Get random string code: '7b7ad245cdff75241935e4dd860f3bad'
        letter_code = random_chars(32)
        file_name = path.split('/')[-1].rsplit('.', 1)[0]
        cv2.imwrite("../imgs_augment/mosaic/images/{}_MOSAIC_{}.jpg".format(file_name,
                    letter_code), new_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print('Successed {}/{} with {}'.format(index+1, NUMBER_IMAGES, file_name))
        annos_list = []
        for anno in new_annos:
            width = anno[3] - anno[1]
            height = anno[4] - anno[2]
            x_center = anno[1] + width/2
            y_center = anno[2] + height/2
            obj = '{} {} {} {} {}'.format(
                anno[0], x_center, y_center, width, height)
            annos_list.append(obj)
        with open("../imgs_augment/mosaic/labels/{}_MOSAIC_{}.txt".format(file_name, letter_code), "w") as outfile:
            outfile.write("\n".join(line for line in annos_list))


# def main():
#     img_paths, annos = get_dataset(LABEL_DIR, IMG_DIR)
#     print('Processing...')
#     for index in range(NUMBER_IMAGES):
#         idxs = random.sample(range(len(annos)), 4)
#         new_image, new_annos, path = update_image_and_anno(img_paths, annos,
#                                                            idxs,
#                                                            OUTPUT_SIZE, SCALE_RANGE,
#                                                            filter_scale=FILTER_TINY_SCALE)

#         # Get random string code: '7b7ad245cdff75241935e4dd860f3bad'
#         # letter_code = random_chars(32)
#         file_name, letter_code = path.split(
#             '/')[-1].rsplit('.', 1)[0].split('_HISTOGRAM_')
#         cv2.imwrite("imgs_augment/mosaic/images/{}_MOSAIC_{}.jpg".format(file_name,
#                     letter_code), new_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
#         print('Successed {}/{} with {}'.format(index+1, NUMBER_IMAGES, file_name))
#         annos_list = []
#         for anno in new_annos:
#             width = anno[3] - anno[1]
#             height = anno[4] - anno[2]
#             x_center = anno[1] + width/2
#             y_center = anno[2] + height/2
#             obj = '{} {} {} {} {}'.format(
#                 anno[0], x_center, y_center, width, height)
#             annos_list.append(obj)
#         with open("imgs_augment/mosaic/labels/{}_MOSAIC_{}.txt".format(file_name, letter_code), "w") as outfile:
#             outfile.write("\n".join(line for line in annos_list))


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
        for obj_list in obj_lists:
            obj = obj_list.rstrip('\n').split(' ')
            xmin = float(obj[1]) - float(obj[3])/2
            ymin = float(obj[2]) - float(obj[4])/2
            xmax = float(obj[1]) + float(obj[3])/2
            ymax = float(obj[2]) + float(obj[4])/2

            boxes.append([int(obj[0]), xmin, ymin, xmax, ymax])
        if not boxes:
            continue
        img_paths.append(img_path)
        labels.append(boxes)
    return img_paths, labels


def update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale=0.):
    '''
    Params:
        - all_img_list <type: list>: list of all images
        - all_annos <type: list>: list of all annotations of specific image
        - idxs <type: list>: index of image in list
        - output_size <type: tuple>: size of output image (Height, Width)
        - scale_range <type: tuple>: range of scale image
        - filter_scale <type: float>: the condition of downscale image and bounding box
    Return:
        - output_img <type: narray>: image after resize
        - new_anno <type: list>: list of new annotation after scale
        - path[0] <type: string>: get the name of image file
    '''
    output_img = np.zeros([output_size[0], output_size[1], 3], dtype=np.uint8)
    scale_x = scale_range[0] + \
        random.random() * (scale_range[1] - scale_range[0])
    scale_y = scale_range[0] + \
        random.random() * (scale_range[1] - scale_range[0])
    divid_point_x = int(scale_x * output_size[1])
    divid_point_y = int(scale_y * output_size[0])

    new_anno = []
    path_list = []
    for i, idx in enumerate(idxs):
        path = all_img_list[idx]
        path_list.append(path)
        img_annos = all_annos[idx]
        img = cv2.imread(path)
        if i == 0:  # top-left
            img = cv2.resize(img, (divid_point_x, divid_point_y))
            output_img[:divid_point_y, :divid_point_x, :] = img
            for bbox in img_annos:
                xmin = bbox[1] * scale_x
                ymin = bbox[2] * scale_y
                xmax = bbox[3] * scale_x
                ymax = bbox[4] * scale_y
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])
        elif i == 1:  # top-right
            img = cv2.resize(
                img, (output_size[1] - divid_point_x, divid_point_y))
            output_img[:divid_point_y, divid_point_x:output_size[1], :] = img
            for bbox in img_annos:
                xmin = scale_x + bbox[1] * (1 - scale_x)
                ymin = bbox[2] * scale_y
                xmax = scale_x + bbox[3] * (1 - scale_x)
                ymax = bbox[4] * scale_y
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])
        elif i == 2:  # bottom-left
            img = cv2.resize(
                img, (divid_point_x, output_size[0] - divid_point_y))
            output_img[divid_point_y:output_size[0], :divid_point_x, :] = img
            for bbox in img_annos:
                xmin = bbox[1] * scale_x
                ymin = scale_y + bbox[2] * (1 - scale_y)
                xmax = bbox[3] * scale_x
                ymax = scale_y + bbox[4] * (1 - scale_y)
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])
        else:  # bottom-right
            img = cv2.resize(
                img, (output_size[1] - divid_point_x, output_size[0] - divid_point_y))
            output_img[divid_point_y:output_size[0],
                       divid_point_x:output_size[1], :] = img
            for bbox in img_annos:
                xmin = scale_x + bbox[1] * (1 - scale_x)
                ymin = scale_y + bbox[2] * (1 - scale_y)
                xmax = scale_x + bbox[3] * (1 - scale_x)
                ymax = scale_y + bbox[4] * (1 - scale_y)
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])

    # Remove bounding box small than scale of filter
    if 0 < filter_scale:
        new_anno = [anno for anno in new_anno if
                    filter_scale < (anno[3] - anno[1]) and filter_scale < (anno[4] - anno[2])]

    return output_img, new_anno, path_list[0]


def random_chars(number_char):
    # Get random string code: '7b7ad245cdff75241935e4dd860f3bad'
    letter_code = 'abcdefghijklmnopqrstuvwxyz0123456789'
    return ''.join(random.choice(letter_code) for _ in range(number_char))


if __name__ == '__main__':
    main()
    print('DONE ✅')
