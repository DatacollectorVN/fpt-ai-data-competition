import numpy as np
import pandas as pd 
import yaml
import os
import cv2
from tqdm import tqdm
import sys
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.insert(1, "../src")
import argparse
from pathlib import Path

FILE_TRAIN_CONFIG = os.path.join("..", "config", "eda_cfg.yaml")
with open(FILE_TRAIN_CONFIG) as file:
    params = yaml.load(file, Loader = yaml.FullLoader)

def main(dataset="train", draw="boxplot"):
    if dataset == "train":
        mode = "train_annotation"
        annots = os.listdir(params[mode])
    elif dataset == "val":
        mode = "val_annotation"
        annots = os.listdir(params[mode])
    elif dataset == "test":
        mode = "test_annotation"
        annots = os.listdir(params[mode])
    else:
        print(f"invalid value of dataset. dataset should be 'train', 'val' or 'test'")
        return
    
    print(f"Compute bboxes area in {dataset}")
    rows = []
    for i, annot_train in tqdm(enumerate(annots[:]), total = len(annots[:])):
        img_id = Path(annot_train).stem
        img_file = img_id + ".jpg"
        img = cv2.imread(os.path.join(params[dataset], img_file))
        img_annots = np.loadtxt(os.path.join(params[mode], annot_train))
        if img_annots.size == 0: # image without annotation.
            continue
        else:
            h, w = img.shape[:2]
            if len(img_annots.shape) == 1: # 1 ndarray
                img_annots = np.expand_dims(img_annots, axis = 0)
            for img_annot in img_annots:
                class_id = int(img_annot[0])
                w_norm = img_annot[3]
                h_norm = img_annot[4]
                ratio_norm = w_norm / h_norm
                area_norm = w_norm * h_norm
                w = w_norm * w
                h = h_norm * h
                ratio = w / h
                area = w * h
                rows.append([class_id, area_norm, area,ratio_norm, ratio])
    df = pd.DataFrame(np.array(rows), columns = ["class_id", "bbox_area_norm", "bbox_area", "ratio_norm", "ratio"])

    # draw
    if draw == "boxplot":
        sns.boxplot(data = df, x = "class_id", y = "bbox_area_norm")
        plt.savefig(os.path.join("..", "images", "bboxes", f"bboxes_area_norm_{draw}_{dataset}.png"))

        sns.boxplot(data = df, x = "class_id", y = "bbox_area")
        plt.savefig(os.path.join("..", "images", "bboxes", f"bboxes_area_{draw}_{dataset}.png"))
    elif draw == "barplot":
        df = df.groupby(by = "class_id").mean().reset_index()
        sns.barplot(data = df, x = "class_id", y = "bbox_area_norm")
        plt.xlabel("Mean normalize area")
        plt.savefig(os.path.join("..", "images", "bboxes", f"bboxes_area_norm_{draw}_{dataset}.png"))

        sns.barplot(data = df, x = "class_id", y = "bbox_area")
        plt.xlabel("Mean area")
        plt.savefig(os.path.join("..", "images", "bboxes", f"bboxes_area_{draw}_{dataset}.png"))   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", dest = "dataset", 
                        default = "train")
    parser.add_argument("--draw", dest = "draw", 
                        default = "boxplot")
    args = parser.parse_args() 
    main(dataset = args.dataset, draw = args.draw)