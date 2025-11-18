# When CNNs Outperform Transformers and Mambas: Revisiting Deep Architectures for Dental Caries Segmentation

## Overview

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
```

## Contact

Please contact zeng.cqupt@gamil.com for any further questions.
