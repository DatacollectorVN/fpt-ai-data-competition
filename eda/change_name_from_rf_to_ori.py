import re
import os
import yaml
import sys
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm

pattern = re.compile(pattern =r"\d+_") 
FILE_TRAIN_CONFIG = os.path.join("..", "config", "eda_cfg.yaml")
with open(FILE_TRAIN_CONFIG) as file:
    params = yaml.load(file, Loader = yaml.FullLoader)

def main(dataset):
    if dataset == "train":
        mode = "train_annotation"
    elif dataset == "val":
        mode = "val_annotation"

    img_ids = os.listdir(os.path.join(params[dataset]))
    for img_id in tqdm(img_ids, total = len(img_ids)):
        img_id_new = re.match(pattern, img_id, flags = 0).group()
        img_id_new = img_id_new[:len(img_id_new)-1]
        img_annots_file = os.path.join(params[mode], Path(img_id).stem + ".txt")
        
        # copy images
        shutil.copy(src = os.path.join(params[dataset], img_id),
                    dst = os.path.join(params[dataset], img_id_new + ".jpg"))
        
        # copy annotations
        shutil.copy(src = img_annots_file,
                    dst = os.path.join(params[mode], img_id_new + ".txt"))
        
        # remove the old file
        os.remove(os.path.join(params[dataset], img_id))
        os.remove(img_annots_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", dest = "dataset", 
                        default = "train")
    args = parser.parse_args() 
    main(dataset = args.dataset)
