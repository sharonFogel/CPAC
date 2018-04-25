# Clustering-driven Deep Embedding with Pairwise Constraints (CPAC)

### Overview

This repository contains a pytorch implementation of our [paper](https://arxiv.org/abs/1803.08457) "Clustering-driven Deep Embedding with Pairwise Constraints". Our algorithm performs non-parametric clustering using a siamese neural network. 

### Citation
If you find our code is useful in your researches, please consider citing:

	@article{fogel2018clustering,
	  title={Clustering-driven Deep Embedding with Pairwise Constraints},
	  author={Fogel, Sharon and Averbuch-Elor, Hadar and Goldberger, Jacov and Cohen-Or, Daniel},
	  journal={arXiv preprint arXiv:1803.08457},
	  year={2018}
	}

### Dependencies
1. [CUDA](https://developer.nvidia.com/cuda-downloads)

2. [cudnn](https://developer.nvidia.com/cudnn)

3. [Python 2.7](https://www.python.org/downloads/)

4. [Pytorch](http://pytorch.org)

5. Additional Python libraries: numpy, sklearn, matplotlib,

### Train model

In order to train the net on a specific dataset (for example USPS) you can run:
   ```bash
   $ python main_CPAC.py --input USPS
   ```
The name of the dataset should be the beginning of the hdf5 file ending with "4torch.h5" (in this case the name of the file will be "USPS4torch.h5". Datasets should be saved in the directory named datasets. We uploaded the datasets USPS, CMUPIE and FRGC. It is possible to train on your own dataset by creating a new hdf5 file with the dataset samples and labels.

You can also change other hyper parameters for model training, such as learning rate, autoencoder dimensions, etc.

