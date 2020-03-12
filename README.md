# CSRNet-Sliced Wasserstein Distance(SWD)-pytorch  

This is the PyTorch Project based on the repo by leeyehoo/CSRNet-pytorch for [CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes](https://arxiv.org/abs/1802.10062) in CVPR 2018.
in this Project we explore the used of Sliced Wasserstein Distance as a loss function in the task of crowd counting. 

## Datasets
ShanghaiTech Dataset: [Google Drive](https://drive.google.com/open?id=16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI)

## Prerequisites
We strongly recommend Anaconda as the environment.

Python: 3.5	

PyTorch: 1.3.0

CUDA: >9.2
## Ground Truth

Please follow the `make_dataset.ipynb` or `make_dataset.py`to generate the ground truth. It shall take some time to generate the dynamic ground truth. Note you need to generate your own json file. (to correct the paths in the existing ones)

## Training Process

Try `python train_swd.py train.json val.json 0 0` to start training process.

## Validation
In this repo there are a number of validation and anlysis scripts to help you validate and visualize the output of the netowrk.
1) `val.py` to get results of single network, and create grephs of network.
2) `choose_best_val.py` to compare a dir of network and get the network with the best result based on MEA and SWD.
3) `val_two_models.py` to get graphs that compare density maps outputs of the two networks

## Results

ShanghaiA MAE: 75.69 [Google Drive](https://drive.google.com/open?id=1P7u3Ox0WJEt43WySzXwmHTuUiUguYd-9)
ShanghaiB MAE: 12.4 [Google Drive](https://drive.google.com/open?id=1fq0__0ZYunCEOOpv2SmT3vMQkg1SGFgS)

## References

If you find the CSRNet-SWD useful, please cite our repo. Thank you!

