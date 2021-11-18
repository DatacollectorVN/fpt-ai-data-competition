import numpy as np 
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import argparse
import shutil
import sys
sys.path.insert(1, "../src")
from utils import crop, increase_brightness, detect_darkness

def main(path_imgs, path_annotations, path_imgs_new, path_annotations_new):
    img_ids = os.listdir(path_imgs)
    for img_id in tqdm(img_ids, total = len(img_ids)):
        img = cv2.imread(os.path.join(path_imgs, img_id))           
        img_annots = np.loadtxt(os.path.join(path_annotations, Path(img_id).stem + ".txt"))
        imgs_crop = crop(img, img_annots)
        is_darkness = detect_darkness(imgs_crop)
        if is_darkness:
            img_new = increase_brightness(img, 25)
            cv2.imwrite(filename = os.path.join(path_imgs_new, Path(img_id).stem + "_bright_" + ".jpg"), 
                        img = img_new)
            np.savetxt(os.path.join(path_annotations_new, Path(img_id).stem + "_bright_" + ".txt"), 
                       img_annots)
        cv2.imwrite(filename = os.path.join(path_imgs_new, img_id), img = img)
        np.savetxt(os.path.join(path_annotations_new, Path(img_id).stem + ".txt"), 
                   img_annots)   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_imgs", dest = "path_imgs", 
                        default = "../dataset-datacomp/dataset-relabel/images/train/")
    parser.add_argument("--path_annots", dest = "path_annots",
                        default = "../dataset-datacomp/dataset-relabel/labels/train/")
    parser.add_argument("--path_imgs_new", dest = "path_imgs_new", 
                        default = "../dataset-datacomp/dataset-increase-brightness/images/train/")  
    parser.add_argument("--path_annots_new", dest = "path_annots_new", 
                        default = "../dataset-datacomp/dataset-increase-brightness/labels/train/")  
    args = parser.parse_args()
    main(args.path_imgs, args.path_annots, args.path_imgs_new, args.path_annots_new)