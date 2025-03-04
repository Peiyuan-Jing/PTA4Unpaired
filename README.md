# Pretext Task Adversarial Learning for Unpaired Low-field to Ultra High-field MRI Synthesis

> A deep learning framework for unpaired high-field MRI synthesis. 🚀  

---

## 📖 Table of Contents
- [Abstract](#Abstract)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)

---
![PTA-Task](fig1_miccai.png) 
## Abstract
Given the scarcity and cost of high-field MRI, the synthesis of high-field MRI from low-field MRI holds significant potential when there is limited data for training downstream tasks (e.g. segmentation). Low-field MRI often suffers from a reduced signal-to-noise ratio (SNR) and spatial resolution compared to high-field MRI. However, synthesizing high-field MRI data presents challenges. These involve aligning image features across domains while preserving anatomical accuracy and enhancing fine details. To address these challenges, we propose a Pretext Task Adversarial (PTA) learning framework for high-field MRI synthesis from low-field MRI data. The framework comprises three processes: (1) The slice-wise gap perception (SGP) network aligns the slice inconsistencies of low-field and high-field datasets based on contrastive learning. (2) The local structure correction (LSC) network extracts local structures by restoring the locally rotated and masked images. (3) The pretext task-guided adversarial training process introduces additional supervision and incorporates a discriminator to improve image realism. Extensive experiments on low-field to ultra high-field task demonstrate the effectiveness of our method, achieving state-of-the-art performance (16.892 in FID, 1.933 in IS, and 0.324 in MS-SSIM). This enables the generation of high-quality high-field-like MRI data from low-field MRI data to augment training datasets for downstream tasks.

---

## Installation
Clone the repository:
```bash
git clone https://github.com/xxx.git
```
Install the required packages:
```bash
pip install -r requirements.txt
```

---
## Dataset&Preparation
Run the code
```bash preprocessing_images.py ```
to process images in the proper format for PTA_hybrid and getting the masks.
Then prepare two directories to host training images from domain A '/path/to/data/trainA' and from domain B '/path/to/data/trainB' respectively. You can train the model with the dataset flag '--dataroot /path/to/data'.
Similarly, you need to prepare two directories:'/path/to/data/testA' and '/path/to/data/testB' during test time. All data are suggested to converting to 'png' format. 

---
## Usage

To train the model(hybrid-version):
```bash 
python train.py --dataroot /media/USER_PATH/npy --name test  --model pta_hybrid --display_id -1 --checkpoints_dir /media/USER_PATH/output/repo_test --load_size=224 --n_epochs 100 --batch_size 16 --input_nc 1 --output_nc 1 --n_epochs_decay 50  --preprocess resize --save_epoch_freq=5 --netG=hybrid  --dataset_mode unaligned_mask --gpu_ids 0
```
Notable arguments include:

- **dataroot**: Datapath to your folder.
- **model**: Which model structure you want to train.
- **load_size**: The shape of your image.
- **net_G**: Which generator you want to use.
- **preprocess**: Which preprocess method want to apply.
- **dataset_mode**: For hybrid model, unaligned mask; for others unaligned.
More details on data preparation can be found in the [dataset documentation](dataset/README.md).
---

## Model Architecture
![PTA-Model](fig2_miccai.png) 

---

## Results

---

## Usage

---

## Dataset

---

## Model Architecture

---

## Results
