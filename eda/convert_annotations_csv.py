import numpy as np
import pandas as pd 
import yaml
import os
import cv2
from tqdm import tqdm
import sys
sys.path.insert(1, "../src")
from utils import rescale, convert_xywh_to_xyxy
from pathlib import Path

'''
reduant annotation in train: 10.35.17.117_01_20210709172855792_MOTION_DETECTION.txt
'''
FILE_TRAIN_CONFIG = os.path.join("..", "config", "eda_cfg.yaml")
with open(FILE_TRAIN_CONFIG) as file:
    params = yaml.load(file, Loader = yaml.FullLoader)

if __name__ == "__main__":
    annots_train = os.listdir(params["train_annotation"])
    # remove the reduant 
    annots_train.remove("10.35.17.117_01_20210709172855792_MOTION_DETECTION.txt")
    annots_val = os.listdir(params["val_annotation"])
    annots_test = os.listdir(params["test_annotation"])

    print(f"Convert train")
    img_files_lst = []
    for i, annot_train in tqdm(enumerate(annots_train), total = len(annots_train)):
        img_id = Path(annot_train).stem
        img_file = img_id + ".jpg"
        img = cv2.imread(os.path.join(params["train"], img_file))
        img_annots = np.loadtxt(os.path.join(params["train_annotation"], annot_train))
        classes_id, bboxes = rescale(img_annots, img)
        bboxes = convert_xywh_to_xyxy(bboxes)
        img_file_lst = [img_file for _ in classes_id]
        img_files_lst.extend(img_file_lst)
        for j, bbox in enumerate(bboxes):
            if (j == 0) & (i == 0):
                row = [classes_id[j], bbox[0], bbox[1], bbox[2], bbox[3]]
            else:
                sub_row = [classes_id[j], bbox[0], bbox[1], bbox[2], bbox[3]]
                row = np.vstack([row, sub_row]).astype(int)
    df = pd.DataFrame(row, columns = ["class_id", "x_min", "y_min", "x_max", "y_max"])
    df["img_file"] = img_files_lst
    df.to_csv(params["train_annotation_csv"], index = False)
    print(f"Number of annotations in train = {df.shape[0]}")

    print(f"Convert val")
    img_files_lst = []
    for i, annot_val in tqdm(enumerate(annots_val), total = len(annots_val)):
        img_id = Path(annot_val).stem
        img_file = img_id + ".jpg"
        img = cv2.imread(os.path.join(params["val"], img_file))
        img_annots = np.loadtxt(os.path.join(params["val_annotation"], annot_val))
        classes_id, bboxes = rescale(img_annots, img)
        bboxes = convert_xywh_to_xyxy(bboxes)
        img_file_lst = [img_file for _ in bboxes]
        img_files_lst.extend(img_file_lst)
        for j, bbox in enumerate(bboxes):
            if (j == 0) & (i == 0):
                row = [classes_id[j], bbox[0], bbox[1], bbox[2], bbox[3]]
            else:
                sub_row = [classes_id[j], bbox[0], bbox[1], bbox[2], bbox[3]]
                row = np.vstack([row, sub_row]).astype(int)
    df = pd.DataFrame(row, columns = ["class_id", "x_min", "y_min", "x_max", "y_max"])
    df["img_file"] = img_files_lst
    df.to_csv(params["val_annotation_csv"], index = False)
    print(f"Number of annotations in val = {df.shape[0]}")

    print(f"Convert test")
    img_files_lst = []
    for i, annot_test in tqdm(enumerate(annots_test), total = len(annots_test)):
        img_id = Path(annot_test).stem
        img_file = img_id + ".jpg"
        img = cv2.imread(os.path.join(params["test"], img_file))
        img_annots = np.loadtxt(os.path.join(params["test_annotation"], annot_test))
        classes_id, bboxes = rescale(img_annots, img)
        bboxes = convert_xywh_to_xyxy(bboxes)
        img_file_lst = [img_file for _ in bboxes]
        img_files_lst.extend(img_file_lst)
        for j, bbox in enumerate(bboxes):
            if (j == 0) & (i == 0):
                row = [classes_id[j], bbox[0], bbox[1], bbox[2], bbox[3]]
            else:
                sub_row = [classes_id[j], bbox[0], bbox[1], bbox[2], bbox[3]]
                row = np.vstack([row, sub_row]).astype(int)
    df = pd.DataFrame(row, columns = ["class_id", "x_min", "y_min", "x_max", "y_max"])
    df["img_file"] = img_files_lst
    df.to_csv(params["test_annotation_csv"], index = False)
    print(f"Number of annotations in test = {df.shape[0]}")