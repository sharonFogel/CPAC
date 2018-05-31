import numpy as np
import torch
import os
import h5py
from torchvision import transforms
from pretrained_model import *
from torch.utils.data.dataset import Dataset

class New_Dataset(Dataset):
    def __init__(self, name, transform, rgb=True, use_vgg19=False):
        file_path_dir = os.path.join(os.getcwd(),'..', 'datasets')
        self.name = name+'_VGG19'*use_vgg19
        file_path = file_path_dir+'/{dataset_name}'.format(dataset_name=name)+use_vgg19*'_VGG19_'+'4torch.h5'
        if not os.path.isfile(file_path):
            file_path =file_path_dir+ '/{dataset_name}4torch.h5'.format(dataset_name=name)
            rgb=True
        h5_file = h5py.File(file_path, 'r')
        self.train_data = h5_file['data']
        self.train_data = self.train_data.value
        if self.train_data.shape[1] == 3:
            if rgb:
                self.train_data = np.moveaxis(self.train_data, 1, -1)
            else:
                gray = [0.3, 0.59, 0.11]
                if name=='YTF' or name=='YTFFULL':
                    gray = [1, 0, 0]
                gray = np.asarray(gray)
                dim_array = np.ones((1, self.train_data.ndim), int).ravel()
                given_axis = 1
                dim_array[given_axis] = -1
                gray_reshaped = gray.reshape(dim_array)
                self.train_data = np.sum(self.train_data * gray_reshaped, 1)
                self.train_data = np.expand_dims(self.train_data, 3)
        else:
            self.train_data = np.moveaxis(self.train_data, 1, -1)
            if rgb and not use_vgg19:
                self.train_data.repeat(1, 3, 1, 1)
        self.train_labels = h5_file['labels'].value.astype(int)
        y_true_unique = np.unique(self.train_labels)
        self.train_labels = np.nonzero(self.train_labels[:, None] == y_true_unique)[1]
        self.transform = transform
        file_path = file_path_dir+'/{dataset_name}'.format(dataset_name=name) + use_vgg19 * '_VGG19_' + '4torch.h5'
        if not os.path.isfile(file_path):
            #vgg19 file doesn't exist, need to create it:
            self.transform = transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.ToPILImage(),
                                                transforms.Resize(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])
                                                ])
            if self.train_data.shape[3]==1:
                train_data_rgb = np.zeros([self.train_data.shape[0], self.train_data.shape[1],self.train_data.shape[2],3])
                self.train_data = np.squeeze(self.train_data)
                train_data_rgb[:,:,:,0] = self.train_data
                train_data_rgb[:,:,:,0] = self.train_data
                train_data_rgb[:,:,:,0] = self.train_data
                self.train_data = train_data_rgb
            dataset = return_deep_fetures(self)
            hdf5_file = h5py.File(file_path, 'w')
            hdf5_file.create_dataset('data', data=dataset.train_data.numpy())
            hdf5_file.create_dataset('labels', data=dataset.train_labels)
        if use_vgg19:
            self.transform = None
    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, idx):
        data_idx = self.train_data[idx]
        if self.transform:
            data_idx = self.transform(data_idx)
        return data_idx, self.train_labels[idx]
