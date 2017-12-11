# Chest_Xray_Classification

## Instructions for evironment setup
This is designed to run on Ubuntu 16.04

If using conda to control enviroment (recommended), the install is separate from the below


### Install through apt-get
imagemagick

python 3.6


### Install through pip
pillow

sklearn

scipy

numpy

wheel

pillow

tensorflow

keras

h5py

## Download Data

https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community

Download the images gzip files in the box application linked on that page. Unzip the images into the project root directory so that the images sit inside of a folder called "images"

## Prepare Data

Run the mod_files.sh script to correct the files that are not in grayscale.

Create a data directory and train, test, and validation directories within them. Run the create_data_structure.py script to create test/validation/train splits on 80%/10%/10% of your data and handle the rest of the directory creation

## Run and test syste

The train_cnn_model.py script will load in the data and train it. You may need to adjust the image size and batch size for the network to not segfault or core dump due to a lack of memory.

The test_cnn_model.py script will use the data in the data/test directory to generate a confusion matrix to evaluate the system that you have run.
