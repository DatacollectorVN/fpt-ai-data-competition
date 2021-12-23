import streamlit as st
import pandas as pd
import yaml
import os
import sys
sys.path.insert(1, "../src")
from utils import filter_classes_id, mode_index, mode_file_name
from pathlib import Path

FILE_TRAIN_CONFIG = os.path.join("..", "config", "eda_cfg.yaml")
with open(FILE_TRAIN_CONFIG) as file:
    params = yaml.load(file, Loader = yaml.FullLoader)

option_dataset = ["Train", "Val", "Test"]
option_classes = ["No mask", "Mask", "Incorrect mask", "All"]
CLASSES_NAME = ["No mask", "Mask", "Incorrect mask"]

def main():
    st.header("**FPT-AI-COMPETITIONS-SHOW-ANNOTATIONS**")
    st.write("by Kos Nhan")
    dataset_mode_lst = st.sidebar.multiselect("Choose dataset (only one)", options = option_dataset)
    class_mode_lst = st.sidebar.multiselect("Choose class", options = option_classes)
    mode = st.sidebar.selectbox("Choose mode", options = ["INDEX", "IMAGE'S FILE NAME"], index = 0)
    if mode == "INDEX":
        start_index, stop_index = st.sidebar.columns(2)
        start = start_index.number_input("None", min_value = 0, value = 0)
        stop = stop_index.number_input("None", min_value = 0, value = 1)
    else:
        img_file_name = st.sidebar.text_input("Enter image's file name", value = "NONE")

    if (dataset_mode_lst == []) or (class_mode_lst == []):
        return
    elif len(dataset_mode_lst) > 1:
        return
    
    # dataset
    if "Train" in dataset_mode_lst:
        df = pd.read_csv(params["train_annotation_csv"])
    elif "Val" in dataset_mode_lst:
        df = pd.read_csv(params["val_annotation_csv"])
    else:
        df = pd.read_csv(params["test_annotation_csv"]) 

    # classes
    if "All" not in class_mode_lst:
        df = filter_classes_id(df, [CLASSES_NAME.index(i) for i in class_mode_lst])
    
    img_ids = df["img_file"].unique().tolist()
    st.write(f"**Number of annotations**= {df.shape[0]} with {len(img_ids)} images")
    if mode == "INDEX":
        st.dataframe(df)
        mode_index(df, img_ids, start, stop, params, dataset_mode_lst)
    else:
        if img_file_name == "NONE":
            st.write("Please enter image's file name")
        else:
            df = df[df["img_file"] == img_file_name]
            if df.shape[0] == 0:
                st.write("Wrong dataset mode")
                return
            st.dataframe(df)
            mode_file_name(df, img_file_name, params, dataset_mode_lst)

if __name__ == "__main__":
    main()
    
    
