# Interp-Parts

Code repository for our paper "Interpretable and Accurate Fine-grained Recognition via Region Grouping" in CVPR 2020 (*Oral Presentation*).

![](demo.jpg)

[[Project Page]](https://www.biostat.wisc.edu/~yli/cvpr2020-interp/)  [[Paper]](https://arxiv.org/abs/2005.10411)

The repository includes full training and evaluation code for CelebA and CUB-200-2011 datasets.


## Dependencies

* p7zip (used for uncompression)
* Python 3
* Pytorch 1.4.0+
* OpenCV-Python
* Numpy
* Scipy
* MatplotLib
* Scikit-learn

## Dataset

### CelebA

You will need to download both aligned and unaligned face images (JPEG format) in CelebA dataset at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html. Make sure your ```data/celeba``` folder is structured as follows:

```
├── img_align_celeba.zip
├── img_celeba.7z.001
├── ...
├── img_celeba.7z.014
└── annotation.zip
```

We provide a bash script to unpack all images. You can use
```sh
cd ./data/celeba
sh data_processing.sh
```
It might take more than 30 minutes to uncompress all the data.

### CUB 200

For Caltech-UCSD Birds-200-2011 (CUB-200) dataset, you will need to manually download the dataset at http://www.vision.caltech.edu/visipedia/CUB-200-2011.html and uncompress the tgz file into ```data/cub200```. Make sure your ```data/cub200``` folder is structured as follows:

```
├── CUB_200_2011/
|   ├── images/
|   ├── parts/
|   ├── attributes/
├── train_test_split.txt
├── ...
```

## CelebA Example

Helper for training parameters:

```sh
cd src/celeba
python train.py --config-help
```
### Training (Unaligned CelebA from SCOPS)
Training (You can specify the desired settings in celeba_res101.json):

```sh
cd src/celeba
python train.py --config ../../celeba_res101.json
```

The code will create three folders for model checkpoints (./checkpoint), log files (./log) and tensorboard logs (./tensorboard_log).

### Visualization and Evaluation (Unaligned CelebA from SCOPS)
Visualization of the results (assuming a ResNet 101 model trained with 9 parts):

```sh
cd src/celeba
python visualize.py --load celeba_res101_p9
```
The code will create a new folder (./visualization) for output images (25 by default).

Evaluating interpretability using part localization (assuming a ResNet101 model trained with 9 parts):

```sh
cd src/celeba
python eval_interp.py --load celeba_res101_p9
```
This should reproduce our results in Table 2.

Evaluating accuracy (assuming a ResNet101 model trained with 9 parts):

```sh
cd src/celeba
python eval_acc.py --load celeba_res101_p9
```
This will report the classification accuracy  (mean class accuracy) on the test set of SCOPS split.

### Reproduce Results in Table 1 (Aligned CelebA)
Training (You need to change the split to *accuracy* in celeba_res101.json):

```sh
cd src/celeba
python train.py --config ../../celeba_res101.json
```

Evaluation:
```sh
cd src/celeba
python eval_acc.py --load celeba_res101_p9
```

## CUB-200 Example

Helper for training parameters:

```sh
cd src/cub200
python train.py --config-help
```
### Training
Training (You can specify the desired settings in celeba_res101.json. The default configuration is slightly different from the paper to reduce GPU memory usage, so that the code can run on a single advanced graphic card.):

```sh
cd src/cub200
python train.py --config ../../cub_res101.json
```

The code will create three folders for model checkpoints (./checkpoint), log files (./log) and tensorboard logs (./tensorboard_log). 

### Visualization and Evaluation
Visualization of the results (assuming a ResNet 101 model trained with 5 parts):

```sh
cd src/cub200
python visualize.py --load cub_res101_p5
```
The code will create a new folder (./visualization) for output images (25 by default).

Evaluating interpretability using part localization (assuming a ResNet101 model trained with 5 parts):

```sh
cd src/cub200
python eval_interp.py --load cub_res101_p5
```

Evaluating accuracy (assuming a ResNet101 model trained with 5 parts):

```sh
cd src/cub200
python eval_acc.py --load cub_res101_p5
```
This will report the classification accuracy on the test set of CUB-200.

## References
If you are using our code, please consider citing our paper.
```
@InProceedings{Huang_2020_CVPR,
author = {Huang, Zixuan and Li, Yin},
title = {Interpretable and Accurate Fine-grained Recognition via Region Grouping},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

If you are using CelebA dataset, please cite
```
@inproceedings{liu2015faceattributes,
 title = {Deep Learning Face Attributes in the Wild},
 author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
 booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
 month = {December},
 year = {2015}
}
```

If you are using CUB-200 dataset, please cite
```
@techreport{WahCUB_200_2011,
Title = {{The Caltech-UCSD Birds-200-2011 Dataset}},
Author = {Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S.},
Year = {2011}
Institution = {California Institute of Technology},
Number = {CNS-TR-2011-001}
}
```
