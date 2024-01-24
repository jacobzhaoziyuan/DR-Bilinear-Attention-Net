<div align="center">

# BIRA-NET: BILINEAR ATTENTION NET FOR DIABETIC RETINOPATHY GRADING

[![ICIP2019](https://img.shields.io/badge/arXiv-1905.06312-blue)](https://arxiv.org/abs/1905.06312)
[![ICIP2019](https://img.shields.io/badge/Conference-ICIP2019-green)](https://ieeexplore.ieee.org/document/8803074))

</div>

Pytorch implementation of our method for ICIP2019 paper: "BIRA-NET: BILINEAR ATTENTION NET FOR DIABETIC RETINOPATHY GRADING". 

## Abstract

Diabetic retinopathy (DR) is a common retinal disease that leads to blindness. For diagnosis purposes, DR image grading aims to provide automatic DR grade classification, which is not addressed in conventional research methods of binary DR image classification. Small objects in the eye images, like lesions and microaneurysms, are essential to DR grading in medical imaging, but they could easily be influenced by other objects. To address these challenges, we propose a new deep learning architecture, called BiRA-Net, which combines the attention model for feature extraction and bilinear model for fine-grained classification. Furthermore, in considering the distance between different grades of different DR categories, we propose a new loss function, called grading loss, which leads to improved training convergence of the proposed approach. Experimental results are provided to demonstrate the superior performance of the proposed approach.

<p align="center">
<img src="./pic/QQ20190426-212322@2x.png" width="550">
</p>


### Installation

The code was tested with Anaconda and Python 2.7. After installing the Anaconda environment:

Clone the repo:

``

```
git clone https://github.com/jfzhang95/pytorch-deeplab-xception.git
cd pytorch-deeplab-xception
```

Install dependencies:

`tensorboardX`

`Pytorch==4.1`

`Numpy`

Download dataset:

Link: https://pan.baidu.com/s/1nz9lMQtgjJbyzjbf22DWyg   password: hqjm 

## Training

Fellow commands below to train the model:

```
usage: train.py [-h] [--batch-size 30]
            [--epochs 10] [--lr 0.0002]
            [--no-cuda] [--seed 1] [--save-epoch 5]
            [--weight-decay 1e-8] [--image-size 610]
```


## Citation
If you find the codebase useful for your research, please cite the paper:
```
@INPROCEEDINGS{zhao2019bira,
  author={Zhao, Ziyuan and Zhang, Kerui and Hao, Xuejie and Tian, Jing and Heng Chua, Matthew Chin and Chen, Li and Xu, Xin},
  booktitle={2019 IEEE International Conference on Image Processing (ICIP)}, 
  title={BiRA-Net: Bilinear Attention Net for Diabetic Retinopathy Grading}, 
  year={2019},
  volume={},
  number={},
  pages={1385-1389},
  keywords={Diabetes;Feature extraction;Retina;Retinopathy;Training},
  doi={10.1109/ICIP.2019.8803074}}
```

