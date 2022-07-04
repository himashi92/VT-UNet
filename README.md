# VT-UNet
This repo contains the supported pytorch code and configuration files to reproduce 3D medical image segmentaion results of [VT-UNet](https://arxiv.org/pdf/2111.13300.pdf). 

![VT-UNet Architecture](img/vt_unet.png?raw=true)

Our previous Code for A Volumetric Transformer for Accurate 3D Tumor Segmentation can be found iside version 1 folder.

# VT-UNet: A Robust Volumetric Transformer for Accurate 3D Tumor Segmentation

Parts of codes are borrowed from [nn-UNet](https://github.com/MIC-DKFZ/nnUNet).

## System requirements
This software was originally designed and run on a system running Ubuntu.

## Dataset Preparation

- Create a folder under VTUNet as DATASET
- Download MSD BraTS dataset (http://medicaldecathlon.com/) and put it under DATASET/vtunet_raw/vtunet_raw_data
- Rename folder as Task03_tumor
- Move dataset.json file to Task03_tumor

## Pre-trained weights

- Download swin-T pretrained weights : https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth
- Add it under pretrained_ckpt folder.

## Create Environment variables

vi ~/.bashrc

- export vtunet_raw_data_base="/home/VTUNet/DATASET/vtunet_raw/vtunet_raw_data"
- export vtunet_preprocessed="/home/VTUNet/DATASET/vtunet_preprocessed"
- export RESULTS_FOLDER_VTUNET="/home/VTUNet/DATASET/vtunet_trained_models"

source ~/.bashrc

## Environment setup

Create a virtual environment 
- virtualenv -p /usr/bin/python3.8 venv
- source venv/bin/activate

Install torch
- pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

Install other dependencies
- pip install -r requirements.txt

## Preprocess Data

cd VTUNet

pip install -e .

- vtunet_convert_decathlon_task -i /home/VTUNet/DATASET/vtunet_raw/vtunet_raw_data/Task03_tumor
- vtunet_plan_and_preprocess -t 3

## Train Model

cd vtunet
- CUDA_VISIBLE_DEVICES=0 nohup vtunet_train 3d_fullres vtunetTrainerV2_vtunet_tumor 3 0 &> small.out &
- CUDA_VISIBLE_DEVICES=0 nohup vtunet_train 3d_fullres vtunetTrainerV2_vtunet_tumor_base 3 0 &> base.out &

## Test Model

cd /home/VTUNet/DATASET/vtunet_raw/vtunet_raw_data/vtunet_raw_data/Task003_tumor/
- CUDA_VISIBLE_DEVICES=0 vtunet_predict -i imagesTs -o inferTs/vtunet_tumor -m 3d_fullres -t 3 -f 0 -chk model_best -tr vtunetTrainerV2_vtunet_tumor  
- python vtunet/inference_tumor.py vtunet_tumor

## Trained model Weights
- [VT-UNet-S](https://drive.google.com/drive/folders/1t7RTwHNwAqh2fiIqUpFTGTG0FHY8k3pQ?usp=sharing) - (fold 0 only)
- VT-UNet-B (To be updated)

## Acknowledgements

This repository makes liberal use of code from [open_brats2020](https://github.com/lescientifik/open_brats2020), [Swin Transformer](https://github.com/microsoft/Swin-Transformer), [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer), [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet), [nnUNet](https://github.com/MIC-DKFZ/nnUNet) and [nnFormer](https://github.com/282857341/nnFormer)

## References

* [Medical segmentation decathlon](http://medicaldecathlon.com/)

## Citing VT-UNet
```bash
    @misc{peiris2021volumetric,
      title={A Volumetric Transformer for Accurate 3D Tumor Segmentation}, 
      author={Himashi Peiris and Munawar Hayat and Zhaolin Chen and Gary Egan and Mehrtash Harandi},
      year={2021},
      eprint={2111.13300},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
    }
```
