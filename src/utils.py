import numpy as np
import cv2
import pandas as pd
import sys
import streamlit as st
import os
import matplotlib.pyplot as plt
import re

def rescale(img_annot, img):
    '''
    Args:
        + img_annot: (ndarray) contains the labels and normalize offset value of bboxes.
                     img_annot with shape (num_bboxes, 5) with 5 represent to (class_id, x_center, y_center, width, height)
        + img: (ndarray) RGB image with shape (H, W, C)
    Return:
        + classes_id: (ndarray): contains the labels of classes.
        + bboxes: (ndarray): offset value of bboxes (rescale) with format(x_center, y_center, width, height)
    '''
    if len(img_annot.shape) == 1: # 1 ndarray
        img_annot = np.expand_dims(img_annot, axis = 0)
    classes_id = img_annot[:, 0].astype(np.int32)
    bboxes = img_annot[:, 1:]
    h, w = img.shape[:2]
    scale = np.hstack([w, h, w, h]).astype(np.float32)
    bboxes = bboxes * scale
    return classes_id, bboxes 

def convert_xywh_to_xyxy(bboxes):
    '''
    Args:
        + bboxes: (ndarray) contains offset value of bboxes (rescale) with format(x_center, y_center, width, height)
                  with shape (num_bboxes, 4)
    return:
        + bboxes: (ndarray) contains offset value of bboxes (rescale) with format(x_min, y_min, x_max, y_max)
    '''
    bboxes_lst = []
    for bbox in bboxes:
        x_center, y_center, width, height = bbox
        x_min = x_center - (width / 2)
        x_max = x_center + (width / 2)
        y_min = y_center - (height / 2)
        y_max = y_center + (height / 2)
        bbox_lst = [x_min, y_min, x_max, y_max]
        bboxes_lst.append(bbox_lst)
    return np.array(bboxes_lst)

def draw_bbox(img, img_bboxes, img_classes_name, classes_name, color, thickness=5):
    img_draw = img.copy()
    for i, img_bbox in enumerate(img_bboxes):
        img_draw = cv2.rectangle(img_draw, pt1 = (int(img_bbox[0]), int(img_bbox[1])), 
                                 pt2 = (int(img_bbox[2]), int(img_bbox[3])), 
                                 color = color[classes_name.index(img_classes_name[i])],
                                 thickness = thickness) 
        cv2.putText(img_draw,
                    text = img_classes_name[i].upper(),
                    org = (int(img_bbox[0]), int(img_bbox[1]) - 5),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 0.6,
                    color = (255, 255, 255),
                    thickness = 1, lineType = cv2.LINE_AA)    
    return img_draw

def filter_classes_id(df, classes_id_lst):
    str_condition = ""
    for i, class_id in enumerate(classes_id_lst):
        if i != len(classes_id_lst) - 1:
            str_condition += f"class_id == {class_id}|"
        else:
            str_condition += f"class_id == {class_id}"
    df = df[df.eval(str_condition)]
    return df

def mode_index(df, img_ids, start, stop, params, dataset_mode_lst):
    if (start ==0) and (stop ==0):
        st.write("Choose the values")
    elif start > stop:
        st.write("The value of start index must smaller than stop index")
    else:
       for i, img_id in enumerate(img_ids[start:stop]):
            if dataset_mode_lst[0] == "Train":
               img = cv2.imread(os.path.join(params["train"], img_id))
            elif dataset_mode_lst[0] == "Val":
               img = cv2.imread(os.path.join(params["val"], img_id))
            else:
               img = cv2.imread(os.path.join(params["test"], img_id))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            df_img_id = df[df["img_file"] == img_id]
            classes_id = df_img_id["class_id"].values.astype(int).tolist()
            classes_id_unique = np.unique(np.array(classes_id))
            bboxes = df_img_id[["x_min", "y_min", "x_max", "y_max"]].values.tolist()
            st.write(f"**Image**: {img_id}")
            st.write(f"**Index**: {i + start}")
            st.write(f"**All bboxes**: {len(classes_id)}")
            st.write(f"**Number of unique classes**: {len(classes_id_unique)}")
            if -1e5 < classes_id[0] < 1e5 : # advoid the cases without annotations
                img = draw_bbox(img, bboxes, [params["names"][i] for i in classes_id], params["names"], params["colors"], 2)
            st.image(img)

def mode_file_name(df, img_file_name, params, dataset_mode_lst):
    if dataset_mode_lst[0] == "Train":
        img = cv2.imread(os.path.join(params["train"], img_file_name))
    elif dataset_mode_lst[0] == "Val":
        img = cv2.imread(os.path.join(params["val"], img_file_name))
    else:
        img = cv2.imread(os.path.join(params["test"], img_file_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    df_img_id = df[df["img_file"] == img_file_name]
    bboxes = df_img_id[["x_min", "y_min", "x_max", "y_max"]].values.tolist()
    classes_id = df_img_id["class_id"].values.tolist()
    classes_id_unique = np.unique(np.array(classes_id))
    st.write(f"**Image**: {img_file_name}")
    st.write(f"**All bboxes**: {len(classes_id)}")
    st.write(f"**Number of unique classes**: {len(classes_id_unique)}")
    if -1e5 < classes_id[0] < 1e5 : # advoid the cases without annotations
        img = draw_bbox(img, bboxes, [params["names"][i] for i in classes_id], params["names"], params["colors"], 2)
    st.image(img)

def get_normalize(params, mode):
    ''' 
    Args: 
        + params: (dict) Default Parameters, Loading from config yaml file.
        + mode: (str) Dataset mode (train, val, test)    
    '''
    
    # df must contain 5 columns (class_id, x_min, y_min, x_max, y_max, img_file)
    if mode == "train":
        df = pd.read_csv(params["train_annotation_csv"])
    elif mode == "val":
        df = pd.read_csv(params["val_annotation_csv"])
    elif mode == "test":
        df = pd.read_csv(params["test_annotation_csv"])
    else:
        print(f"Invalid mode")
        return

    img_ids = df["img_file"].values
    for i, img_id in enumerate(img_ids):
        height, width = cv2.imread(os.path.join(params[mode], img_id)).shape[:2]
        if i == 0:
            rows = np.array([width, height, width, height])
        else:
            row = np.array([width, height, width, height])
            rows = np.vstack([rows, row])
    
def crop(img, img_annots):
    classes_id, bboxes = rescale(img_annots, img)
    bboxes = convert_xywh_to_xyxy(bboxes)
    imgs_crop = []
    for i, bbox in enumerate(bboxes):
        img_content = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]
        img_crop = {f"img" : img_content, "class_id" : classes_id[i], "bbox" : bbox}
        imgs_crop.append(img_crop)
    return imgs_crop

def variance_of_laplacian(img):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(img, cv2.CV_64F).var()

# https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
# https://github.com/albumentations-team/albumentations/issues/67
def change_brightness(img, alpha, beta):
    img_new = np.asarray(alpha*img + beta, dtype = np.uint8)   # cast pixel values to int
    img_new[img_new>255] = 255
    img_new[img_new<0] = 0
    return img_new

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v)) 
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def plot_multi_imgs(imgs, # 1 batchs contain multiple images
                    cols = 2, size = 10, # size of figure
                    is_rgb = True, title = None, cmap = "gray",
                    img_size = None): # set img_size if you want (width, height)
    rows = (len(imgs) // cols) + 1
    fig = plt.figure(figsize = (size *  cols, size * rows))
    for i , img in enumerate(imgs):
        if img_size is not None:
            img = cv2.resize(img, img_size)
        fig.add_subplot(rows, cols, i + 1) # add subplot int the the figure
        plt.imshow(img, cmap = cmap) # plot individual image
    plt.suptitle(title)

def bright_scorce_bbox(bbox_img):
    # Convert color space to LAB format and extract L channel
    L, A, B = cv2.split(cv2.cvtColor(bbox_img, cv2.COLOR_BGR2LAB))
    # Normalize L channel by dividing all pixel values with maximum pixel value
    L = L/255 
    return np.mean(L)

def detect_darkness(imgs_crop):
    for img_crop in imgs_crop:
        bright_score = bright_scorce_bbox(img_crop["img"])
        if bright_score <= 0.3:
            return True
    return False

def save_annotations(img_annotations, file_save, path):
    pattern = re.compile(pattern =r"0.\d+")
    if len(img_annotations.shape) ==  1:
        img_annotations = np.expand_dims(img_annotations, axis = 0)
    img_classes = img_annotations[:, 0].astype(np.int32)
    with open(os.path.join(path, file_save), 'w') as output:
        for i in range(len(img_classes)):
            img_class = img_classes[i]
            output.write(str(img_class) + " ")
            img_annotation = str(img_annotations[i])
            matches = pattern.finditer(img_annotation)
            num_matches = len([_ for _ in pattern.finditer(img_annotation)])
            assert num_matches == 4, f"Error with num_matche = {num_matches}"
            for j, match in enumerate(matches):
                if j == 3:
                    output.write(str(match.group()) + "\n")
                else:
                    output.write(str(match.group()) + " ")