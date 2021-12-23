'''Source: 
https://github.com/whynotw/rotational-data-augmentation-yolo
https://learnopencv.com/image-rotation-and-translation-using-opencv/
'''

import numpy as np
import cv2
from glob import glob
import os
from tqdm import tqdm

# Params
angle_interval = 10
ratio = 80
dirname_input_label = './imgs_augment/flip/labels'
dirname_input_image = './imgs_augment/flip/images'
output_path = './imgs_augment/rotate'


def xywh2xyxy(label, height_image, width_image):
    x_left = int((float(label[1]) - float(label[3])/2.) * width_image)
    x_right = int((float(label[1]) + float(label[3])/2.) * width_image)
    y_top = int((float(label[2]) - float(label[4])/2.) * height_image)
    y_bottom = int((float(label[2]) + float(label[4])/2.) * height_image)
    return label[0], x_left, y_top, x_right, y_bottom


def xyxy2xywh(coord, height_image, width_image):
    x_center = (float(coord[1]) + float(coord[3]))/2. / width_image
    y_center = (float(coord[2]) + float(coord[4]))/2. / height_image
    width = (float(coord[3]) - float(coord[1])) / width_image
    height = (float(coord[4]) - float(coord[2])) / height_image
    return coord[0], x_center, y_center, width, height


def main():
    dirname_output_image = os.path.join(output_path, "images")
    dirname_output_label = os.path.join(output_path, "labels")

    image_names = glob(dirname_input_image + "/*")
    print(f"Processing... with {len(image_names)} images")
    count = 1
    for image_name0 in tqdm(image_names):
        count += 1
        label_name = os.path.join(dirname_input_label, os.path.splitext(
            os.path.basename(image_name0))[0]+".txt")
        with open(label_name, "r") as f0:
            labels0 = f0.read().splitlines()

        image0 = cv2.imread(image_name0)
        height_image0, width_image0 = image0.shape[:2]
        coords = []
        for label in labels0:
            label = label.split()
            coord = xywh2xyxy(label, height_image0, width_image0)
            coords.append([coord[0], coord[1], coord[2], coord[3], coord[4]])

        for angle in [angle_interval, 360-angle_interval]:
            image_name = os.path.join(dirname_output_image, os.path.splitext(
                os.path.basename(image_name0))[0] + "_%03d" % angle + ".jpg")
            label_name = os.path.join(dirname_output_label, os.path.splitext(
                os.path.basename(image_name0))[0] + "_%03d" % angle + ".txt")

            center = int(width_image0/2), int(height_image0/2)
            scale = 1.
            matrix = cv2.getRotationMatrix2D(center, angle, scale)
            image = cv2.warpAffine(
                image0, matrix, (width_image0, height_image0))

            file_label = open(label_name, "w")
            for coord in coords:
                category, x_left0, y_top0, x_right0, y_bottom0 = coord
                points0 = np.array([[x_left0, y_top0, 1.],
                                    [x_left0, y_bottom0, 1.],
                                    [x_right0, y_top0, 1.],
                                    [x_right0, y_bottom0, 1.]])
                points = np.dot(matrix, points0.T).T
                x_left = int(min(p[0] for p in points))
                x_right = int(max(p[0] for p in points))
                y_top = int(min(p[1] for p in points))
                y_bottom = int(max(p[1] for p in points))
                x_left, x_right = np.clip(
                    [x_left, x_right], 0, width_image0)
                y_top, y_bottom = np.clip(
                    [y_top, y_bottom], 0, height_image0)
                x_left, x_right = int(x_left*1.0075), int(x_right*0.9925)
                y_top, y_bottom = int(y_top*1.0075), int(y_bottom*0.9925)
                label = xyxy2xywh(
                    [category, x_left, y_top, x_right, y_bottom], height_image0, width_image0)
                min_label = min(label[1:])
                if min_label > 0:
                    file_label.write(
                        " ".join([str(l) for l in label]) + "\n")
                else:
                    continue
            cv2.imwrite(image_name, image, [cv2.IMWRITE_JPEG_QUALITY, 75])


if __name__ == '__main__':
    main()
    print('\nDONE')
