# Fu-TransHNet
This paper presents a novel deep learning fusion method called Fu-TransHNet, which aims to improve the accuracy of colon polyp segmentation.

# Configuration
We adopt NVIDIA GTX3090 for GPU acceleration training

Pytorch = 2.1.2+cu121

Python = 3.8.19

# Model Overview
![1742888947498](https://github.com/user-attachments/assets/982405b0-2934-48a0-981e-dbd80e48fda3)

# Experiments
GPUs with a memory capacity of 4GB or more are adequate for this experiment.
GPU-accelerated training with NVIDIA GeForce RTX 4090 D

1.Preparing necessary data:

Dataset: CVC-ClinicDB, CVC-ColonDB, CVC-EndoScene, ETIS-LaribPolypDB and Kvasir
- Download the dataset and put the unzipped data into . /data.
- In accordance with (https://github.com/ChenxiLuo-code1/Fu-TransHNet), run process_train.py and process_test.py to process the train and test data,
producing `data_{train, val, test}.npy` and `mask_{train, val, test}.npy`.

2.Training:
- run `train_isic.py`；Need to change the default save path or other hyperparameters.
- Save the model parameter file `*.pth`.
- Model checkpoints have been given: “w3-Trans109.pth”

3.Testing:
- run `test_isic.py`；Need to change the default save path or other hyperparameters.
