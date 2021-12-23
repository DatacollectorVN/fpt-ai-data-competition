import os
import shutil

# Params
DATASET = 'public_test_padding'
LABEL_DIR = './data/dataset/labels/'
IMAGE_DIR = './data/dataset/images/'


def main():
    print('Processing data...')
    labels = os.listdir(LABEL_DIR + DATASET)

    mask_types = ['no_mask', 'mask', 'incorrect_mask']
    for idx, mask_type in enumerate(mask_types):
        name_list = []
        for label in labels:
            f = open(os.path.join(LABEL_DIR + DATASET, label), 'r')
            for line in f.readlines():
                if line[0] == str(idx):
                    name_list.append(f.name)
        name_list = list(dict.fromkeys(name_list))
        print('There have {} images contain category {} - type {}'.format(
            len(name_list), idx, mask_type))

        os.makedirs(LABEL_DIR + mask_type)
        for name in name_list:
            new_path = name.replace(DATASET, mask_type)
            shutil.copyfile(name, new_path)
        print('✅ DONE with {} labels'.format(mask_type))

        os.makedirs(IMAGE_DIR + mask_type)
        for name in name_list:
            path = name.replace('labels', 'images')
            path = path.replace('txt', 'jpg')
            new_path = path.replace(DATASET, mask_type)
            shutil.copyfile(path, new_path)
        print('✅ DONE with {} images'.format(mask_type))
        print('-------------------------------')


if __name__ == '__main__':
    main()
