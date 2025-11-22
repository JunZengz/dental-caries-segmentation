# When CNNs Outperform Transformers and Mambas: Revisiting Deep Architectures for Dental Caries Segmentation

## Overview
This repository contains the source code for When CNNs Outperform Transformers and Mambas: Revisiting Deep Architectures for Dental Caries Segmentation

## Qualitative results
<p align="center">
<img src="imgs/Qualitative_results.jpeg" alt> 
  <em> Figure 3: Qualitative Examples of Dental Caries Segmentation on the DC1000 Dataset</em>
  </p>



## Create Environment
```
conda create -n caries-seg python==3.9.0
conda activate caries-seg
```

## Install Dependencies
```    
pip install -r requirements.txt
cd selective_scan && pip install .
```

## Download Dataset
Download DC1000 dataset from [Google Drive](https://drive.google.com/file/d/1UABfWMw7Vvd2KC1xyAOlAS3OXNKPtMED/view?usp=drive_link).
Move it to the `data` directory.

## Train
```
python train.py 
```

## Test
```
python test.py
```

## Citation
Please cite our paper if you find the work useful:
```
@article{ghimire2025caries,
  title={When CNNs Outperform Transformers and Mambas: Revisiting Deep Architectures for Dental Caries Segmentation},
  author={Ghimire, Aashish and Zeng, Jun and Paudel, Roshan and Tomar, Nikhil Kumar and Nayak, Deepak Ranjan and Nalla, Harshith Reddy and Jha, Vivek and Reynolds, Glenda and Jha, Debesh},
  journal={arXiv preprint arXiv:2511.14860},
  year={2025}
}
```

## Acknowledgment
This project is built upon DoubleUnet ([paper](https://arxiv.org/abs/2006.04868), [code](https://github.com/DebeshJha/2020-CBMS-DoubleU-Net)), ColonSegNet ([paper](https://arxiv.org/abs/2011.07631), [code](https://github.com/DebeshJha/ColonSegNet)), U-Net ([paper](https://arxiv.org/abs/1505.04597), [code](https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/tree/master)), ResUNet++ ([paper](https://arxiv.org/abs/1911.07067), [code](https://github.com/DebeshJha/ResUNetPlusPlus)), VMUNet ([paper](https://arxiv.org/abs/2402.02491), [code](https://github.com/JCruan519/VM-UNet)), VMUNetV2 ([paper](https://arxiv.org/abs/2403.09157), [code](https://github.com/nobodyplayer1/VM-UNetV2)), Mamba-UNet ([paper](https://arxiv.org/abs/2402.05079), [code](https://github.com/ziyangwang007/Mamba-UNet)), RMAMamba ([paper](https://arxiv.org/abs/2502.18232), [code](https://github.com/JunZengz/RMAMamba)), PVTFormer ([paper](https://arxiv.org/abs/2401.09630), [code](https://github.com/DebeshJha/PVTFormer)), TransRUPNet ([paper](https://arxiv.org/abs/2306.02176), [code](https://github.com/DebeshJha/TransRUPNet)), TransNetR ([paper](https://arxiv.org/abs/2303.07428), [code](https://github.com/DebeshJha/TransNetR)), RSAFormer ([paper](https://www.sciencedirect.com/science/article/pii/S0010482524003524), [code](https://github.com/JunZengz/RSAFormer)). 
We gratefully acknowledge the authors for their excellent work and for sharing their code in the open-source community. 

## Contact

Please contact zeng.cqupt@gmail.com for any further questions.
