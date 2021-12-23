import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import yaml
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(1, "../src")
from utils import variance_of_laplacian, crop, plot_multi_imgs
FILE_TRAIN_CONFIG = os.path.join("..", "config", "eda_cfg.yaml")
with open(FILE_TRAIN_CONFIG) as file:
    params = yaml.load(file, Loader = yaml.FullLoader)

def change_brightness(img, alpha, beta):
    img_new = np.asarray(alpha*img + beta, dtype = np.uint8)   # cast pixel values to int
    img_new[img_new>255] = 255
    img_new[img_new<0] = 0
    return img_new

img_id = "649.jpg"
img = cv2.imread(os.path.join(params["train"], img_id))
img = change_brightness(img, 1, 40)
cv2.imshow("test", img)
cv2.waitKey(0)