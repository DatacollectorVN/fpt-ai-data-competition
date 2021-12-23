import numpy as np
import cv2
import sys
import streamlit as st
import os

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
            classes_id = df_img_id["class_id"].values.tolist()
            classes_id_unique = np.unique(np.array(classes_id))
            bboxes = df_img_id[["x_min", "y_min", "x_max", "y_max"]].values.tolist()
            st.write(f"**Image**: {img_id}")
            st.write(f"**Index**: {i + start}")
            st.write(f"**All bboxes**: {len(classes_id)}")
            st.write(f"**Number of unique classes**: {len(classes_id_unique)}")
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
    img = draw_bbox(img, bboxes, [params["names"][i] for i in classes_id], params["names"], params["colors"], 2)
    st.image(img)
