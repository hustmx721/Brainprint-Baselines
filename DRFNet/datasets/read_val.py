import numpy as np
import os
import h5py
import torch
from sklearn.model_selection import train_test_split
from utils import normalize_v2
import torch.utils.data
def dataset_read(src_file_path, tar_file_path, src_data_name, tar_data_name, batch_size):
    # Load source (training) data
    src_data_path = os.path.join(src_file_path, src_data_name)
    dataset = h5py.File(src_data_path, 'r')
    src_data = np.array(dataset['data'])
    if len(np.shape(src_data)) < 4:
        src_data = np.expand_dims(src_data, axis=1)
    src_label = np.array(dataset['label'])
    print('>>> total --> Src Data:{} Src Label:{}'.format(src_data.shape, src_label.shape))

    # Load target (test) data
    tar_data_path = os.path.join(tar_file_path, tar_data_name)
    dataset = h5py.File(tar_data_path, 'r')
    tar_data = np.array(dataset['data'])
    if len(np.shape(tar_data)) < 4:
        tar_data = np.expand_dims(tar_data, axis=1)
    tar_label = np.array(dataset['label'])
    print('>>> total --> Tar Data:{} Tar Label:{}'.format(tar_data.shape, tar_label.shape))

    # Normalize data
    src_data, tar_data = normalize_v2(src_data, tar_data)

    # Convert to torch tensors
    src_data = torch.from_numpy(src_data).float()
    src_label = torch.from_numpy(src_label).long()
    tar_data = torch.from_numpy(tar_data).float()
    tar_label = torch.from_numpy(tar_label).long()

    # Split source data into training and validation sets
    src_train_data, src_val_data, src_train_label, src_val_label = train_test_split(src_data, src_label, test_size=0.2, random_state=42)

    print('Data and label prepared!')
    print('>>> Validation Data:{} Validation Label:{} '.format(src_val_data.shape, src_val_label.shape))
    print('>>> Test Data:{} Test Label:{} '.format(tar_data.shape, tar_label.shape))
    print('>>> Train Data:{} Train Label:{}'.format(src_train_data.shape, src_train_label.shape))
    print('----------------------')

    # Create DataLoader for each dataset
    train_dataset = torch.utils.data.TensorDataset(src_train_data, src_train_label)
    val_dataset = torch.utils.data.TensorDataset(src_val_data, src_val_label)
    test_dataset = torch.utils.data.TensorDataset(tar_data, tar_label)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# Example model class
class ExampleModel(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(ExampleModel, self).__init__()
        self.fc = torch.nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

def train_and_evaluate(train_loader, val_loader, test_loader, model_class, input_size, num_classes, batch_size, **model_kwargs):
    model = model_class(input_size, num_classes, **model_kwargs)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(10):  # Example epoch number, adjust as needed
        model.train()
        for data, label in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

    # Validation loop
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for data, label in val_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            val_total += label.size(0)
            val_correct += (predicted == label).sum().item()

    val_accuracy = 100 * val_correct / val_total
    print(f'Validation Accuracy: {val_accuracy:.2f}%')

    # Testing loop
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data, label in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            test_total += label.size(0)
            test_correct += (predicted == label).sum().item()

    test_accuracy = 100 * test_correct / test_total
    print(f'Test Accuracy: {test_accuracy:.2f}%')

# Usage example
# src_file_path = 'path/to/source'
# tar_file_path = 'path/to/target'
# src_data_name = 'source_data.h5'
# tar_data_name = 'target_data.h5'
# batch_size = 32
# input_size = 128  # Adjust based on your dataset
# num_classes = 10  # Adjust based on your dataset
#
# train_loader, val_loader, test_loader = dataset_read(src_file_path, tar_file_path, src_data_name, tar_data_name, batch_size)
# train_and_evaluate(train_loader, val_loader, test_loader, ExampleModel, input_size, num_classes, batch_size)
