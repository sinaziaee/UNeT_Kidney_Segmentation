# Kidney Image Segmentation with 2D U-Net

In medical imaging, accurate segmentation of kidney structures is crucial for various diagnostic and therapeutic purposes. One popular approach for automated kidney segmentation is utilizing convolutional neural networks (CNNs), particularly the U-Net architecture.

The U-Net architecture, known for its success in biomedical image segmentation tasks, employs an encoder-decoder framework with skip connections to preserve spatial information. Specifically, the 2D U-Net variant is well-suited for segmenting kidney structures in two-dimensional medical images. The architecture of UNET is depicted below:

### U-Net Architecture

<!-- <img src='./images/UNET_architecture.png' width='100%' > -->

![The architecture](https://raw.githubusercontent.com/sinaziaee/kidney_segmentation/master/images/UNET_architecture.png)

In this project we will work on a variation of UNET architecture, called EffUNet, where the encoder is a efficient net model. The purpose of this project is to calculate the uncertainty maps of the model in inference to find the anomalies in kidneys, including tumors and cysts. We will train our model on a healthy dataset of kidneys and use the trained model on an unhealthy dataset of kidneys to infer and detect tumors. The first phase is detection of kidneys in images.
The training is on two A6000 GPUS on a parallel mode and took approximately 20 Gigabytes of GPU from each machine.

# How to Run the Project
# Phase 1

## Step 1 (Download Datasets)
First start by downloading the datasets from [TCIA](https://wiki.cancerimagingarchive.net/display/public/pancreas-ct#225140405a525c7710d147e8bfc6611f18577bb7) for healthy kidneys and [kits19](https://github.com/neheller/kits19) , [kits23](https://github.com/neheller/kits23) for unhealthy datasets.

## Step 2 (Pre-Process)

## Step 3 (Model Creation)

## Step 4 (Model Training)

## Step 5 (Model Inference)

# How to Run the Project:
# Phase 2

## Step 1 (Run MonteCarlo Inference)