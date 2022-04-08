# Improved version of the code base for VT-UNet will be released soon !!

# VT-UNet
This repo contains the supported pytorch code and configuration files to reproduce 3D medical image segmentaion results of [VT-UNet](https://arxiv.org/pdf/2111.13300.pdf). 


![VT-UNet Architecture](img/vt_unet.png?raw=true)

## Environment
Prepare an environment with python=3.8, and then run the command "pip install -r requirements.txt" for the dependencies.

## Data Preparation
- For experiments we used four datasets:
    - BRATS 2021 : http://braintumorsegmentation.org/
    - MSD BRATS, LIVER, PANCREAS : http://medicaldecathlon.com/

- File structure
    ```
     BRATS2021
      |---Data
      |   |--- RSNA_ASNR_MICCAI_BraTS2021_TrainingData
      |   |   |--- BraTS2021_00000
      |   |   |   |--- BraTS2021_00000_flair...
      |   
      |              
      |   
      |
     VT-UNet
      |---train.py
      |---test.py
      |---pretrained_ckpt
      |---saved_model
      ...
    ```

## Pre-Trained Weights
- Swin-T: https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
- Download Swin-T pre-trained weights and add it under pretrained_ckpt folder

## Pre-Trained Base Model For BraTS 2021
- VT-UNet-B: https://drive.google.com/file/d/1tLpEfyKgQ8xgvM3D1Aqi8aD3ILVB3du7/view?usp=sharing
- Download VT-UNet-B pre-trained model and add it under saved_model folder before running test.py

## Train/Test
- Train : Run the train script on BraTS 2021 Training Dataset with Base model Configurations. 
```bash
python train.py --cfg configs/vt_unet_base.yaml --num_classes 3 --epochs 350
```

- Test : Run the test script on BraTS 2021 Training Dataset. 
```bash
python test.py --cfg configs/vt_unet_base.yaml --num_classes 3
```

## Acknowledgements
This repository makes liberal use of code from [open_brats2020](https://github.com/lescientifik/open_brats2020), [Swin Transformer](https://github.com/microsoft/Swin-Transformer), [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer) and [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)

## References
* [BraTS 2021](http://braintumorsegmentation.org/)
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



