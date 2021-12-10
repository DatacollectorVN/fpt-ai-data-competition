import numpy as np 
import cv2
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import sys
sys.path.insert(1, "../src")
from utils import save_annotations, enhence_face, img_enhence_face

def main(path_imgs, path_annotations, path_imgs_new, path_annotations_new):
    img_ids = os.listdir(path_imgs)
    for img_id in tqdm(img_ids, total = len(img_ids)):
        img = cv2.imread(os.path.join(path_imgs, img_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)           
        img_annots = np.loadtxt(os.path.join(path_annotations, Path(img_id).stem + ".txt"))
        enhence_face_lst = enhence_face(img, img_annots)
        img_after = img_enhence_face(img, enhence_face_lst)
        img_after = Image.fromarray(img_after)
        img_after.save(os.path.join(path_imgs_new, Path(img_id).stem + "_enhance_" + ".jpg"))
        save_annotations(img_annots, Path(img_id).stem + ".txt", path_annotations_new)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_imgs", dest = "path_imgs", 
                        default = "../dataset-datacomp/dataset-relabel/images/train/")
    parser.add_argument("--path_annots", dest = "path_annots",
                        default = "../dataset-datacomp/dataset-relabel/labels/train/")
    parser.add_argument("--path_imgs_new", dest = "path_imgs_new", 
                        default = "../dataset-datacomp/dataset-enhance-face/images/train/")  
    parser.add_argument("--path_annots_new", dest = "path_annots_new", 
                        default = "../dataset-datacomp/dataset-enhance-face/labels/train/")  
    args = parser.parse_args()
    main(args.path_imgs, args.path_annots, args.path_imgs_new, args.path_annots_new)