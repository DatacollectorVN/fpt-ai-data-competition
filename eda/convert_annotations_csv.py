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

FILE_TRAIN_CONFIG = os.path.join("..", "config", "eda_cfg.yaml")
with open(FILE_TRAIN_CONFIG) as file:
    params = yaml.load(file, Loader = yaml.FullLoader)

if __name__ == "__main__":
    annots_train = os.listdir(params["train_annotation"])
    annots_val = os.listdir(params["val_annotation"])
    annots_test = os.listdir(params["test_annotation"])

    print(f"Convert train")
    img_files_lst = []
    for i, annot_train in tqdm(enumerate(annots_train[:]), total = len(annots_train[:])):
        img_id = Path(annot_train).stem
        img_file = img_id + ".jpg"
        img = cv2.imread(os.path.join(params["train"], img_file))
        img_annots = np.loadtxt(os.path.join(params["train_annotation"], annot_train))
        if img_annots.size == 0:
            classes_id = np.array([])
            bboxes = np.array([])
            img_file_lst = [img_file]
            img_files_lst.extend(img_file_lst)
        else:
            classes_id, bboxes = rescale(img_annots, img)
            bboxes = convert_xywh_to_xyxy(bboxes)
            img_file_lst = [img_file for _ in classes_id]
            img_files_lst.extend(img_file_lst)
        
        if bboxes.size == 0:
            row_empty = [None, None, None, None, None]
            row = np.vstack([row, row_empty])
        else:
            for j, bbox in enumerate(bboxes):
                if (j == 0) & (i == 0):
                    row = np.array([classes_id[j], bbox[0], bbox[1], bbox[2], bbox[3]]).astype(int)
                else:
                    sub_row = np.array([classes_id[j], bbox[0], bbox[1], bbox[2], bbox[3]]).astype(int)
                    row = np.vstack([row, sub_row])
    df = pd.DataFrame(row, columns = ["class_id", "x_min", "y_min", "x_max", "y_max"])
    df["img_file"] = img_files_lst
    df.to_csv(params["train_annotation_csv"], index = False)
    print(f"Number of annotations in train = {df.shape[0]}")

    print(f"Convert val")
    img_files_lst = []
    for i, annot_train in tqdm(enumerate(annots_val[:]), total = len(annots_val[:])):
        img_id = Path(annot_train).stem
        img_file = img_id + ".jpg"
        img = cv2.imread(os.path.join(params["val"], img_file))
        img_annots = np.loadtxt(os.path.join(params["val_annotation"], annot_train))
        if img_annots.size == 0:
            classes_id = np.array([])
            bboxes = np.array([])
            img_file_lst = [img_file]
            img_files_lst.extend(img_file_lst)
        else:
            classes_id, bboxes = rescale(img_annots, img)
            bboxes = convert_xywh_to_xyxy(bboxes)
            img_file_lst = [img_file for _ in classes_id]
            img_files_lst.extend(img_file_lst)
        
        if bboxes.size == 0:
            row_empty = [None, None, None, None, None]
            row = np.vstack([row, row_empty])
        else:
            for j, bbox in enumerate(bboxes):
                if (j == 0) & (i == 0):
                    row = np.array([classes_id[j], bbox[0], bbox[1], bbox[2], bbox[3]]).astype(int)
                else:
                    sub_row = np.array([classes_id[j], bbox[0], bbox[1], bbox[2], bbox[3]]).astype(int)
                    row = np.vstack([row, sub_row])
    df = pd.DataFrame(row, columns = ["class_id", "x_min", "y_min", "x_max", "y_max"])
    df["img_file"] = img_files_lst
    df.to_csv(params["val_annotation_csv"], index = False)
    print(f"Number of annotations in val = {df.shape[0]}")

    '''
    print(f"Convert test")
    img_files_lst = []
    for i, annot_train in tqdm(enumerate(annots_test[:]), total = len(annots_test[:])):
        img_id = Path(annot_train).stem
        img_file = img_id + ".jpg"
        img = cv2.imread(os.path.join(params["test"], img_file))
        img_annots = np.loadtxt(os.path.join(params["test_annotation"], annot_train))
        if img_annots.size == 0:
            classes_id = np.array([])
            bboxes = np.array([])
            img_file_lst = [img_file]
            img_files_lst.extend(img_file_lst)
        else:
            classes_id, bboxes = rescale(img_annots, img)
            bboxes = convert_xywh_to_xyxy(bboxes)
            img_file_lst = [img_file for _ in classes_id]
            img_files_lst.extend(img_file_lst)
        
        if bboxes.size == 0:
            row_empty = [None, None, None, None, None]
            row = np.vstack([row, row_empty])
        else:
            for j, bbox in enumerate(bboxes):
                if (j == 0) & (i == 0):
                    row = np.array([classes_id[j], bbox[0], bbox[1], bbox[2], bbox[3]]).astype(int)
                else:
                    sub_row = np.array([classes_id[j], bbox[0], bbox[1], bbox[2], bbox[3]]).astype(int)
                    row = np.vstack([row, sub_row])
    df = pd.DataFrame(row, columns = ["class_id", "x_min", "y_min", "x_max", "y_max"])
    df["img_file"] = img_files_lst
    df.to_csv(params["test_annotation_csv"], index = False)
    print(f"Number of annotations in test = {df.shape[0]}")
    '''