# FPT-AI-DATA-COMPETITION
This repository clone from [FPT-AI](https://github.com/fsoft-ailab/Data-Competition)
Update:
+ src/: contains all utils files for EDA.
+ eda/: contains all files for EDA **(NOTE: don't commit notebook file)**
+ download_data.sh: file bash for downloading the dataset. 

1. Create virtual environment.
```bash
conda create -n fptai python=3.7
conda activate fptai
```

2. clone this repository.

3. Install required packages. 
```bash 
pip install -r requirements.txt
```

4. Download the standard and additional data after processing.
`bash download_data_standard_add.sh`
