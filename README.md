# FPT-AI-DATA-COMPETITION
This repository clone from [FPT-AI](https://github.com/fsoft-ailab/Data-Competition)

Update:
+ src/: contains all utils files for EDA.
+ eda/: contains all files for EDA **(NOTE: don't commit notebook file)**
+ download_data.sh: file bash for downloading the dataset. 

## 1. Create virtual environment:
```bash
conda create -n fptai python=3.7
conda activate fptai
```

## 2. Clone this repository:
```bash
git clone https://github.com/DatacollectorVN/fpt-ai-data-competition.git
```

## 3. Install required packages: 
```bash 
pip install -r requirements.txt
```

## 4. Download the standard and additional data after processing:

`bash download_data_standard_add.sh`

## 5. Baseline: [Update soon]

## 6. Train:

- On Google Colab: **(Note: Make a copy in drive)**
<a href="https://colab.research.google.com/drive/18VZqW9X2bI2Os28BhIyE4YqkFC9FKRrf?usp=sharing" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- On server:
```bash
python train.py --batch-size 32 --device 0 --name <version_name> 
```
*Note*: Change the number of epochs to 70 in config/train_cfg.yaml

## 7. Evaluation:
```bash
python val.py --weights results/train/<version_name>/weights/best.pt  --task test --name <version_name> --batch-size 64 --device 0
                                                                             val
                                                                             train
```
* Results are saved at `results/evaluate/<task>/<version_name>`.

## 8. Prediction:
* Results are saved at `<save_dir>`.
```bash
python detect.py --weights results/train/<version_name>/weights/best.pt --source <path_to_folder> --dir <save_dir> --device 0
```
*Note*: 
- <path_to_folder>: folder contain images to predict (Usually ./dataset/public_test)
- <save_dir>: path to save images predict