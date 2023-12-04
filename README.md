# Floodwater Classification with Digital Elevation Models and Sen1Floods11

For my Data Science MSc project at the University of Bristol I chose to combine my knowledge of the issues facing hydrological models from my geography degree with some machine learning techniques. Identifying flooded land from satellite imagery has much potential to improve real time understanding of flood events, as well as to provide better constraints for numerical models. My final report can be found [here](https://github.com/aeroniemi/floodwater-classification/blob/a58d17f0f7bfecfdd6e23139ee8db5bdf95982d8/report.pdf). 

Parts of the code are adapted from the methods used by Iselborn, et al. although no code is copied from this source in the current version. Their code can be found [here](https://github.com/DFKI-Earth-And-Space-Applications/Flood_Mapping_Feature_Space_Importance). Previously this project was aimed as a fork of their code with added features. The forked version can be found by looking back at the commit history. 

## Requirements

This project depends on:

- v1.1 of Sen1Floods11
- Python (>3.10)
- Conda
- A wide array of Python packages

## Usage

### Downloading the data

You can download the Sen1Floods11 dataset from a AWS S3 bucket. Detailled(ish) instructions for how to do this are available [here](https://github.com/DFKI-Earth-And-Space-Applications/Flood_Mapping_Feature_Space_Importance/blob/main/src/sen1floods11/README.md)

### File/folder structure

Importantly, you must have the following file structure once downloaded:

```
./code/ - contains the code in this repository
./downloaded_data/ - contains the sen1floods11 data
```

Inside the latter folder you should have:

```
./downloaded_data/label/
./downloaded_data/S1Hand/
./downloaded_data/S1Weak/
./downloaded_data/S1WeakLabel/
./downloaded_data/S2Hand/
./downloaded_data/S2Weak/
./downloaded_data/S2WeakLabel/
./downloaded_data/split/
./downloaded_data/flood_bolivia_data.csv
./downloaded_data/flood_test_data.csv
./downloaded_data/flood_train_data.csv
./downloaded_data/flood_val_data.csv
```

The CSV files contain lists of images for each of the train/test/val/bolivia splits, and must be named as above

Additionally, you must create the following folders:

```
./data_cache/ - contains the cached npz files with feature spaces
./metrics_outputs/ - contains the output file for metrics.py
./model_outputs/ - contains npz files with model outputs
./output_images/ - contains images produced by the image export tasks
```

### Creating feature space caches

Before training a model you need to create a feature space cache. This is a file that contains the data for that feature space transformed arrays that are easy and quick to import to python. 
``python ./generate_dataset.py "[FEATURE_SPACE]"``

### Training a model

``python ./train_models.py [FEATURE_SPACE] [MODEL]``

Alternatively the feature space can be made at the same time:

``python ./gen_and_train.py [FEATURE_SPACE] [MODEL]``

### Create metrics file

Metrics can be exported for all models run at once by running:
``python ./metrics.py``

## The Report

Detailled definitions of how this was used for the project can be found in the report, which will be linked here after submission.
