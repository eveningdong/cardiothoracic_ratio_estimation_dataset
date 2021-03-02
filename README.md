# Cardiothoracic Ratio (CTR) Estimation Dataset

## Description
Extended Wingspan Cardiothoracic Ratio (CTR) Estimation Dataset used in the paper [Unsupervised Domain Adaptation for Automatic Estimation of Cardiothoracic Ratio](https://link.springer.com/chapter/10.1007/978-3-030-00934-2_61) and [Neural Architecture Search for Adversarial Medical Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-030-00934-2_61). The extension includes additional 38 patients and chest organ segmenation masks for all patients.

The dataset contains 259 patients' chest X-ray images, chest organ segmentation masks, and key points for CTR estimation.

## Full Dataset
The full dataset can be downloaded vis this [link](https://drive.google.com/file/d/1EgcxMyMUgRm4VvodGDK9w_HJbL1jCW1o/view?usp=sharing).

### Data Structure
Each patient has an independent folder with the folder name as his ID. In each folder, there are `original.dcm`, `left_lung.png`, `right_lung.png`, `heart.png`, and one `key_points.txt`.

In `key_points.txt`, you will see `name,x,y`. `name` is the organ name, and `x` and `y` are the normalized pixel coordinates for the `width` and `height`. `(x=0, y=0)` stands for the upper left corner and `(x=1, y=1)` stands for the lower right corner.

## Downsampled Dataset
The downsampled dataset with fixed resolution `512 x 512` is provided within the repository. In addition, another popular chest organ segmentation dataset Japanese Society of Radiological Technology([JSRT](https://jsrt.or.jp/)) is provided here as a comparison. 

### Data Structure
Two datasets `wingspan`  and `jsrt` have the same structure. Each dataset has two folders `png` and `mask`, where `mask` has three sub-folders `left_lung`, `right_lung` and `heart`.

## Python API
Python functions for CTR estimation is provided in `utils.py`.

## Citation
If you find the work and dataset are useful in your research, please cite:

@inproceedings{dong2018unsupervised,
  title={Unsupervised domain adaptation for automatic estimation of cardiothoracic ratio},
  author={Dong, Nanqing and Kampffmeyer, Michael and Liang, Xiaodan and Wang, Zeya and Dai, Wei and Xing, Eric},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={544--552},
  year={2018},
  organization={Springer}
}

@inproceedings{dong2019neural,
  title={Neural architecture search for adversarial medical image segmentation},
  author={Dong, Nanqing and Xu, Min and Liang, Xiaodan and Jiang, Yiliang and Dai, Wei and Xing, Eric},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={828--836},
  year={2019},
  organization={Springer}
}
