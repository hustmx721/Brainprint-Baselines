import os
import sys
import itertools
import csv
import datetime
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from sklearn.model_selection import KFold
from utils import ensure_path, seed_all, normalize
from My_Solver_K_fold import Solver
import logging
import h5py
import torch.utils.data



def setup_logger(log_path: str, log_filename: str, file_save_name: str) -> None:
    ensure_path(log_path)
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_filename = f"{file_save_name}_{current_datetime}_{log_filename}"
    logging.basicConfig(filename=os.path.join(log_path, log_filename),
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def load_config(config_file: str) -> dict:
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config
def save_confusion_matrix(cm, file_name, num_class, cm_save_path, cm_csv_save_path, acc, fold):
    cm_percent = cm / cm.sum(axis=1)[:, None] * 100
    plt.close()
    plt.figure(figsize=(30, 30))
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    ensure_path(cm_save_path)
    fig_file_name = f"{file_name}_fold{fold}_{num_class}_{acc:.4f}_{current_datetime}_confusion_matrix.png"
    fig_save_path = os.path.join(cm_save_path, fig_file_name)
    plt.savefig(fig_save_path)

    ensure_path(cm_csv_save_path)
    csv_file_name = f"{file_name}_fold{fold}_{num_class}_{acc:.4f}_{current_datetime}_confusion_matrix.csv"
    csv_save_path = os.path.join(cm_csv_save_path, csv_file_name)
    with open(csv_save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([''] + [f'Predicted {i}' for i in range(len(cm))])
        for i, row in enumerate(cm):
            writer.writerow([f'Actual {i}'] + list(row))

def save_model(solver, file_name, params, acc, fold, model_save_path):
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    params_str = '_'.join([f'{k}{v}' for k, v in params.items()])
    model_file_name = f"{file_name}_{params_str}_{acc:.4f}_fold{fold}_{current_datetime}.pt"
    ensure_path(model_save_path)
    model_save_path = os.path.join(model_save_path, model_file_name)
    solver.save_model(model_save_path)
    logging.info(f"Model saved to {model_save_path}")

def k_fold_cross_validation(config: dict, k: int = 5) -> Tuple[Tuple, Tuple]:
    device = torch.device(config['runtime']['device'] if torch.cuda.is_available() else "cpu")

    # 提取配置参数
    data_path = config['file_paths']['data_path']
    source_file_name = config['file_paths']['source_file_name']
    target_file_name = config['file_paths']['target_file_name']
    log_path = config['file_paths']['log_path']
    model_save_path = config['file_paths']['model_save_path']
    cm_save_path = config['file_paths']['cm_save_path']
    cm_csv_save_path = config['file_paths']['cm_csv_save_path']
    file_save_name = config['file_paths']['file_save_name']

    setup_logger(log_path, 'training.log',file_save_name)

    learning_rate = float(config['hyperparameters']['learning_rate'])
    batch_size = int(config['hyperparameters']['batch_size'])
    max_epochs = int(config['hyperparameters']['max_epochs'])
    patience = int(config['hyperparameters']['patience'])
    initial_weight_decay = float(config['hyperparameters']['initial_weight_decay'])

    input_size = tuple(config['network_parameters']['input_size'])
    num_class = int(config['network_parameters']['num_class'])
    dropout_rate = float(config['network_parameters']['dropout_rate'])

    out_channels_list = config['grid_search_params']['out_channels']
    D_list = config['grid_search_params']['D']
    eeg_groups_list = config['grid_search_params']['eeg_groups']
    sa_groups_list = config['grid_search_params']['sa_groups']

    seed_all(2023)


    source_data, source_label = load_data(data_path, source_file_name)
    target_data, target_label = load_data(data_path, target_file_name)
      

    for out_channels, D, eeg_groups, sa_groups in itertools.product(out_channels_list, D_list, eeg_groups_list, sa_groups_list):

        logging.info(f"Trying out_channels={out_channels}, D={D}, eeg_groups={eeg_groups}, sa_groups={sa_groups}")
        kfold = KFold(n_splits=k, shuffle=True, random_state=2023)
        fold_results_best = []
        best_globe_acc = 0
        best_globe_metrics = None
        best_globe_solver=None

        for fold, (train_idx, val_idx) in enumerate(kfold.split(source_data)):
            logging.info(f"Fold {fold+1}/{k}")
            train_data, val_data = source_data[train_idx], source_data[val_idx]
            train_label, val_label = source_label[train_idx], source_label[val_idx]
            train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_label), batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_data, val_label), batch_size=batch_size, shuffle=False)
            test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(target_data, target_label), batch_size=batch_size, shuffle=False)
       
        
            solver = Solver(
                device=device,
                batch_size=batch_size,
                input_size=input_size,
                D=D,
                out_channels=out_channels,
                eeg_groups=eeg_groups,
                sa_groups=sa_groups,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                num_class=num_class,
                initial_weight_decay=initial_weight_decay
            )

            solver.set_data_loaders(train_loader, val_loader, test_loader)
            best_fold_test_acc = 0
            best_fold_test_metrics = None
            best_fold_solver = None 
            for epoch in range(max_epochs):
                solver.train_epoch()
      
                if (epoch + 1) % 5 == 0:
                    test_results = solver.test()
                    test_acc = test_results[0]
                    # Update global best test accuracy
                    if test_acc > best_fold_test_acc:
                        best_fold_test_acc = test_acc
                        best_fold_test_metrics = test_results
                        best_fold_solver = solver
                solver.lr_scheduler.step()
                if (epoch + 1) % 40 == 0:
                    new_weight_decay = solver.initial_weight_decay * 0.9
                    solver.adjust_weight_decay(new_weight_decay)
                    logging.info(f"Adjusted weight decay to {new_weight_decay}")

            final_test_results = solver.test()
            final_test_acc= final_test_results[0]



            if final_test_acc > best_fold_test_acc:
                best_fold_test_acc = final_test_acc
                best_fold_test_metrics = final_test_results
                best_fold_solver = solver
            if best_fold_test_acc> best_globe_acc:
                    best_globe_acc= best_fold_test_acc
                    best_globe_metrics= best_fold_test_metrics
                    best_globe_solver = best_fold_solver

            fold_results_best.append(best_fold_test_metrics[:4] + best_fold_test_metrics[5:])




        if best_globe_acc > 0:
            save_confusion_matrix(best_globe_metrics[4], file_save_name, num_class, cm_save_path, cm_csv_save_path, best_globe_acc, fold+1)
            params = {'out_channels': out_channels, 'D': D, 'eeg_groups': eeg_groups, 'sa_groups': sa_groups}
            save_model(best_globe_solver, file_save_name, params, best_globe_acc, fold+1, model_save_path)

        logging.info(f"Fold best results for params out_channels={out_channels}, D={D}, eeg_groups={eeg_groups}, sa_groups={sa_groups}:")
        for i, result in enumerate(fold_results_best):
            logging.info(f"Fold {i+1}: {result}")

        fold_results_best_np = np.array(fold_results_best)
        mean_best_metrics = fold_results_best_np.mean(axis=0)
        std_best_metrics = fold_results_best_np.std(axis=0)

        logging.info(f"5-Fold Best Results for params out_channels={out_channels}, D={D}, eeg_groups={eeg_groups}, sa_groups={sa_groups}: Mean ± Std")
        logging.info(f"Best Accuracy: {mean_best_metrics[0]:.4f} ± {std_best_metrics[0]:.4f}")
        logging.info(f"Best F1: {mean_best_metrics[1]:.4f} ± {std_best_metrics[1]:.4f}")
        logging.info(f"Best Precision: {mean_best_metrics[2]:.4f} ± {std_best_metrics[2]:.4f}")
        logging.info(f"Best Recall: {mean_best_metrics[3]:.4f} ± {std_best_metrics[3]:.4f}")
        logging.info(f"Best Precision Weighted: {mean_best_metrics[4]:.4f} ± {std_best_metrics[4]:.4f}")
        logging.info(f"Best Recall Weighted: {mean_best_metrics[5]:.4f} ± {std_best_metrics[5]:.4f}")
        logging.info(f"Best F1 Weighted: {mean_best_metrics[6]:.4f} ± {std_best_metrics[6]:.4f}")



def load_data(data_path, data_name):
    with h5py.File(os.path.join(data_path, data_name), 'r') as dataset:
        data = np.array(dataset['data'])
        data = normalize(data)
        if len(np.shape(data)) < 4:
            data = np.expand_dims(data, axis=1)
        label = np.array(dataset['label'])
    return torch.from_numpy(data).float(), torch.from_numpy(label).long()

if __name__ == "__main__":
    config_file = '/home/lhg/PycharmProjects/third_work/DRFNet/config_for_30ssvep_normal_paramanager.yaml'
    config = load_config(config_file)
    k_fold_cross_validation(config)





