import cv2
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

# Params
IMAGE_DIR = './data/dataset/images/train_new_below_1.7'
OUTPUT_DIR = './output'
file_name_list = []
height_list = []
width_list = []
ratio_list = []
focus_list = []


def main():
    count = 0
    height, width = (0, 0)
    print('Processing...')
    for image_file in glob.glob(os.path.join(IMAGE_DIR, '*.jpg')):
        image = cv2.imread(image_file)
        focus_measure = cv2.Laplacian(image, cv2.CV_64F).var()
        ratio_wh = image.shape[1]/image.shape[0]
        height += image.shape[0]
        width += image.shape[1]
        file_name_list.append(image_file.split('/')[-1])
        height_list.append(image.shape[0])
        width_list.append(image.shape[1])
        ratio_list.append(round(ratio_wh, 3))
        focus_list.append(round(focus_measure, 3))
        if image.shape[0] != 720 or image.shape[1] != 1280 or round(ratio_wh, 1) != 1.8:
            print(
                "Image: {} - Shape: {} - Ratio: {:.3f}".format(image_file.split('/')[-1], image.shape, ratio_wh))
        count += 1
    print("Count of different image: ", count)
    print("The average of height = {:.3f}".format(
        height/len(glob.glob(os.path.join(IMAGE_DIR, '*.jpg')))))
    print("The average of width = {:.3f}".format(
        width/len(glob.glob(os.path.join(IMAGE_DIR, '*.jpg')))))
    print("The ratio of width and height: {:.3f}".format(width/height))
    header_dict = {'file_name': file_name_list, 'height': height_list, 'width': width_list,
                   'ratio': ratio_list, 'focus_measure': focus_list}
    df = pd.DataFrame(header_dict)
    # df.to_csv(OUTPUT_DIR + '/' + IMAGE_DIR.split('/')
    #           [-1] + '_information.csv', index=False)
    df['ratio'].value_counts().sort_index().plot(
        kind='barh')
    plt.title('The ratio of {} dataset'.format(IMAGE_DIR.split('/')[-1]))
    plt.show()


if __name__ == '__main__':
    main()
    print('DONE')
