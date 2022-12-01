# Challenging Conceptual Transformers

## Project structure

```
.
├── ct
│   ├── data
│   │    - keep each dataset in separate file
│   └── models
│        - keep each model in separate file
│        - concept transformer module will be implemented in some file(s) in this folder? 
├── checkpoints
├── logs
│    - tensorboard/wandb logs
├── notebooks
│    - data analysis, model predictions analysis
├── plots
│    - plots for the report
├── trained_models
├── readme.md
└── .gitignore
```

## Data

### aPY

Download annotations and aYahoo test dataset from: https://vision.cs.uiuc.edu/attributes/
Download aPascal train/validation data from: http://host.robots.ox.ac.uk/pascal/VOC/voc2008/index.html#devkit
(Direct dwnld link for aPascal: http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar)
VOC2008 documentation downloaded from http://host.robots.ox.ac.uk/pascal/VOC/voc2008/devkit_doc_21-Apr-2008.pdf
(We probably do not need it, enough info in attribute_data/README.md file)

## Report

https://www.overleaf.com/8966194915vvrcgtbpmpdn

## gcp environment installation

Using Deep Learning VM on google cloud.
zone: europe-west1-c
series: N1
machine type: n1-highmem-2 (2 vCPU, 13 GB memory) (default)
gpu: NVIDIA T4
framework: PyTorch 1.12 (CUDA 11.3) (Don't forget to confirm you want to install GPU drivers)
boot disk: Standard Persistent Disk (default) - 200 GB

```
conda create -n dd2412 python=3.10
conda activate dd2412
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install pytorch_lightning
conda install pandas
<!-- conda install -c conda-forge scikit-learn -->
```