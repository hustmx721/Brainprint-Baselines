import sys
import numpy as np
from sklearn.model_selection import train_test_split
import os
import h5py
from utils import *
from .unaligned_data_loader import UnalignedDataLoader


def dataset_read(src_file_path, tar_file_path, src_data_name, tar_data_name, batch_size):
    S = {}
    T = {}
    T_val = {}


    # Load source data
    src_data_path = os.path.join(src_file_path, src_data_name)
    dataset = h5py.File(src_data_path, 'r')
    src_data = np.array(dataset['data'])
    if len(np.shape(src_data)) < 4:
        src_data = np.expand_dims(src_data, axis=1)
    src_label = np.array(dataset['label'])
    print('>>> total --> Src Data:{} Src Label:{}'.format(src_data.shape, src_label.shape))

    # Load target data
    tar_data_path = os.path.join(tar_file_path, tar_data_name)
    dataset = h5py.File(tar_data_path, 'r')
    tar_data = np.array(dataset['data'])
    if len(np.shape(tar_data)) < 4:
        tar_data = np.expand_dims(tar_data, axis=1)
    tar_label = np.array(dataset['label'])
    print('>>> total --> Target Data:{} Target Label:{}'.format(tar_data.shape, tar_label.shape))

    # Normalize data
    src_data, tar_data = normalize_v2(src_data, tar_data)

    # Convert to torch tensors
    src_data = torch.from_numpy(src_data).float()
    src_label = torch.from_numpy(src_label).long()
    tar_data = torch.from_numpy(tar_data).float()
    tar_label = torch.from_numpy(tar_label).long()

    # Split target data into validation and test sets
    tar_val_data, tar_test_data, tar_val_label, tar_test_label = train_test_split(tar_data, tar_label, test_size=0.2,
                                                                                  random_state=42)

    print('Data and label prepared!')
    print('>>> Validation Data:{} Validation Label:{} '.format(tar_val_data.shape, tar_val_label.shape))
    print('>>> Test Data:{} Test Label:{} '.format(tar_test_data.shape, tar_test_label.shape))
    print('>>> Train Data:{} Train Label:{}'.format(src_data.shape, src_label.shape))
    print('----------------------')

    S['data'] = src_data
    S['labels'] = src_label

    T['data'] = tar_test_data
    T['labels'] = tar_test_label

    T_val['data'] = tar_val_data
    T_val['labels'] = tar_val_label

    train_loader = UnalignedDataLoader()
    train_loader.initialize(S, T, T_val, batch_size, batch_size, batch_size)
    dataset = train_loader.load_data()
    return dataset
