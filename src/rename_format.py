import os

# Params
# ./data/NO_INCORRECT_MASK.v1-nim-flipmosaic.yolov5pytorch
DATA_PATH = './data/NO_MASK_TRAIN_SIZE.v1-nm_aug.yolov5pytorch'
TYPE_AUGMENTATION = 'NM_AUG'  # Ex: MOSAIC


def main():
    print('Processing...')
    images = os.listdir(DATA_PATH + '/train/images')
    for image_aug in images:
        old_path = os.path.join(DATA_PATH + '/train/images', image_aug)
        new_name = image_aug.replace('jpg.rf.', TYPE_AUGMENTATION + '_')
        new_path = os.path.join(DATA_PATH + '/train/images', new_name)
        os.rename(old_path, new_path)
    print('✅ DONE with images')

    labels = os.listdir(DATA_PATH + '/train/labels')
    for label_aug in labels:
        old_path = os.path.join(DATA_PATH + '/train/labels', label_aug)
        new_name = label_aug.replace('jpg.rf.', TYPE_AUGMENTATION + '_')
        new_path = os.path.join(DATA_PATH + '/train/labels', new_name)
        os.rename(old_path, new_path)
    print('✅ DONE with labels')


if __name__ == '__main__':
    main()
