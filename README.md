# ACNN for Text-Independent Speaker Recognition

Official implementation of 
 - **Adaptive Convolutional Neural Network for Text-Independent Speaker Recognition** <br>
by Seong-Hu Kim, Yong-Hwa Park @ Human Lab, Mechanical Engineering Department, KAIST<br>
[![Interspeech](https://img.shields.io/badge/InterSpeech2021-inproceedings-orange)](https://www.isca-speech.org/archive/pdfs/interspeech_2021/kim21_interspeech.pdf) <br>

Accepted paper in [InterSpeech 2021](https://www.interspeech2021.org/).

This code was written mainly with reference to [baseline code](https://github.com/Jungjee/RawNet).

## Adaptive Convolutional Neural Network Module
We use two scaling maps, which are frequency and time domain, to each axis for the adaptive kernel in the ACNN module.  The adaptive kernel is created by element-wise multiplication of each output channel of the content-invariant kernel with the scaling matrix. The structure of proposed ACNN module for speaker recognition is shown as follows.

<img src="./pretrained_model/ACNN_module.png" width="700">

This module is applied to VGG-M and ResNet for text-independent speaker recognition. 

## Requirements and versions used
- pytorch >= 1.4.0
- pytorchaudio >= 0.4.0
- numpy >= 1.18

## Dataset
We used Voxceleb1 dataset in this paper. You can download the dataset by reffering to [Voxceleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html). All data should be gathered in one folder and you set the dataset directories in 'train_model.yaml'.

## Training
You can train and save model in `exps` folder by running:
```shell
python train_model.py
```
You need to adjust the training parameters in yaml before training.

#### Results:

Network              | Top-1 (%) |  Top-1 (%) | EER (%) | C_det (%) |
---------------------|-----------|------------|---------|-----------|
Adaptive VGG-M (N=18)| 86.51     | 95.31      | 5.68    | 0.510     |
Adaptive ResNet18 (N=18)| 85.84     | 95.29      | 6.18    | 0.589     |

## Pretrained models
There are pretrained models in 'pretrained_model'. The example code for verification using the pretrained models is not provided separately.

## Citation

    @inproceedings{kim21_interspeech,
      author={Seong-Hu Kim and Yong-Hwa Park},
      title={{Adaptive Convolutional Neural Network for Text-Independent Speaker Recognition}},
      year=2021,
      booktitle={Proc. Interspeech 2021},
      pages={66--70},
      doi={10.21437/Interspeech.2021-65}
    }


Please contact Seong-Hu Kim at seonghu.kim@kaist.ac.kr for any query.

