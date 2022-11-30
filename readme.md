# Project structure

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


# Data

## aPY

Download annotations and aYahoo test dataset from: https://vision.cs.uiuc.edu/attributes/
Download aPascal train/validation data from: http://host.robots.ox.ac.uk/pascal/VOC/voc2008/index.html#devkit
(Direct dwnld link for aPascal: http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar)
VOC2008 documentation downloaded from http://host.robots.ox.ac.uk/pascal/VOC/voc2008/devkit_doc_21-Apr-2008.pdf
(We probably do not need it, enough info in attribute_data/README.md file)
