import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--labels_path', help='Path of labels to count annotations',
                    default='data/labels')
args = parser.parse_args()
labels_path = args.labels_path


def main():
    label_types = os.listdir(labels_path)
    for label_type in label_types:
        label_nm, label_m, label_im = (0, 0, 0)
        print('Processing {} folder...'.format(label_type))
        files_list = os.listdir(os.path.join(labels_path, label_type))
        for anno in files_list:
            f = open(os.path.join(labels_path, label_type, anno), 'r')
            lines = f.readlines()
            for line in lines:
                assert line[0] == '0' or line[0] == '1' or line[0] == '2', "Wrong format annotation file name is {}".format(
                    anno)
                if line[0] == '0':
                    label_nm += 1
                elif line[0] == '1':
                    label_m += 1
                elif line[0] == '2':
                    label_im += 1
        print('[{}]: no_mask have {} - mask have {} - incorrect mask have {} annotations'.format(
            label_type.upper(), label_nm, label_m, label_im))


if __name__ == '__main__':
    main()
    print('DONE ✅')
