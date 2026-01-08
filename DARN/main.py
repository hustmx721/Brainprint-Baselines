
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
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import KFold
from utils import ensure_path, seed_all, normalize
from My_Solver_K_fold import Solver
import logging
import h5py
import torch.utils.data
from thop import profile
import time, copy
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity



PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def setup_logger(log_path: str, log_filename: str, file_save_name: str) -> None:
    ensure_path(log_path)
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_filename = f"{file_save_name}_{current_datetime}_{log_filename}"
    logging.basicConfig(filename=os.path.join(log_path, log_filename),
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def expand_param_range(config: dict) -> dict:

    params = config['grid_search_params']
    if 'out_channels_range' in params:
        start, end, step = params['out_channels_range']
        config['grid_search_params']['out_channels'] = list(range(start, end + 1, step))
        
        del config['grid_search_params']['out_channels_range']
    if 'D_range' in params:
        start, end, step = params['D_range']
        config['grid_search_params']['D'] = list(range(start, end + 1, step))
        del config['grid_search_params']['D_range']
    if 'eeg_groups_range' in params:
        start, end, step = params['eeg_groups_range']
        config['grid_search_params']['eeg_groups'] = list(range(start, end + 1, step))
        del config['grid_search_params']['eeg_groups_range']
    for param in ['out_channels', 'D', 'eeg_groups']:
        if param not in config['grid_search_params']:
            logging.warning(f"Parameter '{param}' not found in grid_search_params. Using default value.")
            config['grid_search_params'][param] = [64] if param == 'out_channels' else [2]
    return config

def load_config(config_file: str) -> dict:
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    config = expand_param_range(config)
    return config



def save_model(solver, file_name, params, acc, fold, model_save_path):
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    params_str = '_'.join([f'{k}{v}' for k, v in params.items()])
    model_file_name = f"{file_name}_{params_str}_{acc:.6f}_fold{fold}_{current_datetime}.pt"
    ensure_path(model_save_path)
    model_save_path = os.path.join(model_save_path, model_file_name)
    solver.save_model(model_save_path)
    logging.info(f"Model saved to {model_save_path}")


def check_parameter_constraint(out_channels: int, D: int, eeg_groups: int) -> bool:
    return (out_channels * D) % (8 * eeg_groups) == 0


def generate_parameter_combinations(config: dict) -> List[dict]:
    out_channels_list = config['grid_search_params']['out_channels']
    D_list = config['grid_search_params']['D']
    eeg_groups_list = config['grid_search_params']['eeg_groups']
    valid_combinations = []
    for out_channels, D, eeg_groups in itertools.product(
            out_channels_list, D_list, eeg_groups_list):
        if check_parameter_constraint(out_channels, D, eeg_groups):
            valid_combinations.append({
                'out_channels': out_channels,
                'D': D,
                'eeg_groups': eeg_groups
            })
        else:
            logging.info(f"Skipping invalid parameter combination: out_channels={out_channels}, "
                         f"D={D}, eeg_groups={eeg_groups} - Does not satisfy constraint")
    return valid_combinations


def k_fold_cross_validation(config: dict, k: int = 5) -> Tuple[Dict[str, Any], Dict[str, float]]:
    device = torch.device(config['runtime']['device'] if torch.cuda.is_available() else "cpu")
    data_path = config['file_paths']['data_path']
    source_file_name = config['file_paths']['source_file_name']
    target_file_name = config['file_paths']['target_file_name']
    file_save_name = config['file_paths']['file_save_name']
    output_root = config['file_paths'].get('output_root', '')
    if output_root:  
        output_root_full = os.path.join(PROJECT_ROOT, output_root)
        model_save_path = os.path.join(output_root_full, "models")
        log_path = os.path.join(output_root_full, "log")
        cm_save_path = os.path.join(output_root_full, "confusion_matrix")
        cm_csv_save_path = os.path.join(output_root_full, "confusion_matrix_csv")
    else:
        model_save_path = config['file_paths']['model_save_path']
        log_path = config['file_paths']['log_path']
        cm_save_path = config['file_paths']['cm_save_path']
        cm_csv_save_path = config['file_paths']['cm_csv_save_path']

    ensure_path(model_save_path)
    ensure_path(log_path)
    ensure_path(cm_save_path)
    ensure_path(cm_csv_save_path)
    setup_logger(log_path, 'training.log', file_save_name)
    learning_rate = float(config['hyperparameters']['learning_rate'])
    batch_size = int(config['hyperparameters']['batch_size'])
    max_epochs = int(config['hyperparameters']['max_epochs'])
    patience = int(config['hyperparameters']['patience'])
    initial_weight_decay = float(config['hyperparameters']['initial_weight_decay'])
    input_size = tuple(config['network_parameters']['input_size'])
    num_class = int(config['network_parameters']['num_class'])
    dropout_rate = float(config['network_parameters']['dropout_rate'])
    seed_all(2025)
    source_data, source_label = load_data(data_path, source_file_name)
    target_data, target_label = load_data(data_path, target_file_name)
    best_global_acc = 0
    best_global_metrics = None
    best_global_params = None
    parameter_combinations = generate_parameter_combinations(config)
    if not parameter_combinations:
        logging.error("No valid parameter combinations found! Please check your grid_search_params in config.")
        return None, None
    result_csv_path = os.path.join(log_path, f"{file_save_name}_grid_search_results.csv")
    with open(result_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['out_channels', 'D', 'eeg_groups',
                      'mean_accuracy', 'std_accuracy', 'mean_f1', 'std_f1',
                      'mean_precision', 'std_precision', 'mean_recall', 'std_recall',
                      'mean_precision_weighted', 'std_precision_weighted',
                      'mean_recall_weighted', 'std_recall_weighted',
                      'mean_f1_weighted', 'std_f1_weighted',
                      'mean_kappa', 'std_kappa', 'mean_mcc', 'std_mcc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for param_set in parameter_combinations:
            out_channels = param_set['out_channels']
            D = param_set['D']
            eeg_groups = param_set['eeg_groups']
            logging.info(f"Trying out_channels={out_channels}, D={D}, eeg_groups={eeg_groups}")
            kfold = KFold(n_splits=k, shuffle=True, random_state=2023)
            fold_results = []
            fold_best_results = []  
            for fold, (train_idx, val_idx) in enumerate(kfold.split(source_data)):
                logging.info(f"Fold {fold+1}/{k}") 
                train_data, val_data = source_data[train_idx], source_data[val_idx]
                train_label, val_label = source_label[train_idx], source_label[val_idx]
                train_loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(train_data, train_label), 
                    batch_size=batch_size, shuffle=True
                )
                val_loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(val_data, val_label), 
                    batch_size=batch_size, shuffle=False
                )
                test_loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(target_data, target_label), 
                    batch_size=batch_size, shuffle=False
                )
                best_val_acc = 0
                best_test_metrics = None
                best_test_cm = None
                best_model_state = None
                count = 0  
                solver = Solver(
                    device=device,
                    batch_size=batch_size,
                    input_size=input_size,
                    D=D,
                    out_channels=out_channels,
                    eeg_groups=eeg_groups,
                    dropout_rate=dropout_rate,
                    learning_rate=learning_rate,
                    num_class=num_class,
                    initial_weight_decay=initial_weight_decay
                )
                solver.set_data_loaders(train_loader, val_loader, test_loader)
                for epoch in range(max_epochs):
                    solver.train_epoch()
                    if (epoch + 1) % 5 == 0:
                        val_metrics = solver.validate()
                        val_acc = val_metrics[0]  
                        test_results = solver.test()
                        test_acc, test_f1, test_precision, test_recall, test_cm, test_prec_w, test_rec_w, test_f1_w, test_kappa, test_mcc = test_results
                        logging.info(f"Mid_Test Metrics - Accuracy: {test_acc:.6f}, F1: {test_f1:.6f}, Precision: {test_precision:.6f}, Recall: {test_recall:.6f}, "
                                    f"Precision Weighted: {test_prec_w:.6f}, Recall Weighted: {test_rec_w:.6f}, F1 Weighted: {test_f1_w:.6f}")
                        logging.info(f"New Metrics - Cohen's Kappa: {test_kappa:.6f}, MCC: {test_mcc:.6f}")
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            count = 0
                            best_test_metrics = (test_acc, test_f1, test_precision, test_recall, test_prec_w, test_rec_w, test_f1_w, 
                                                test_kappa, test_mcc)
                            best_test_cm = test_cm
                            best_model_state = copy.deepcopy(solver.myModel.state_dict())
                        else:
                            count += 1
                        if count >= patience:
                            logging.info(f"Early stopping triggered at epoch {epoch+1}. Stopping training.")
                            break
                    solver.lr_scheduler.step()
                    if (epoch + 1) % 40 == 0:
                        new_weight_decay = solver.initial_weight_decay * 0.9
                        solver.adjust_weight_decay(new_weight_decay)
                        logging.info(f"Adjusted weight decay to {new_weight_decay}")
                if best_test_metrics is not None:
                    fold_best_results.append(best_test_metrics)
                else:
                    test_results = solver.test()
                    test_acc, test_f1, test_precision, test_recall, test_cm, test_prec_w, test_rec_w, test_f1_w, test_kappa, test_mcc = test_results
                    best_test_metrics = (test_acc, test_f1, test_precision, test_recall, test_prec_w, test_rec_w, test_f1_w, 
                                        test_kappa, test_mcc)
                    fold_best_results.append(best_test_metrics)
                    best_test_cm = test_cm
                    best_model_state = copy.deepcopy(solver.myModel.state_dict())
                params = {'out_channels': out_channels, 'D': D, 'eeg_groups': eeg_groups}
                if best_model_state is not None:
                    solver.myModel.load_state_dict(best_model_state)
                save_model(solver, file_save_name, params, best_test_metrics[0], fold+1, model_save_path)
            logging.info(f"5-Fold CV Results for this param set - Individual Folds")
            for fold_idx, metrics in enumerate(fold_best_results, 1):
                logging.info(f"Fold {fold_idx}: Accuracy: {metrics[0]:.6f}, F1: {metrics[1]:.6f}, "
                            f"Precision: {metrics[2]:.6f}, Recall: {metrics[3]:.6f}, "
                            f"Precision Weighted: {metrics[4]:.6f}, Recall Weighted: {metrics[5]:.6f}, "
                            f"F1 Weighted: {metrics[6]:.6f}, "
                            f"Cohen's Kappa: {metrics[7]:.6f}, MCC: {metrics[8]:.6f}")
            fold_metrics_np = np.array(fold_best_results)
            mean_metrics = fold_metrics_np.mean(axis=0)
            std_metrics = fold_metrics_np.std(axis=0)           
            logging.info(f"5-Fold CV Results for this param set - Mean ± Std")
            logging.info(f"Accuracy: {mean_metrics[0]:.6f} ± {std_metrics[0]:.6f}, F1: {mean_metrics[1]:.6f} ± {std_metrics[1]:.6f}, "
                        f"Precision: {mean_metrics[2]:.6f} ± {std_metrics[2]:.6f}, Recall: {mean_metrics[3]:.6f} ± {std_metrics[3]:.6f}")
            logging.info(f"Precision Weighted: {mean_metrics[4]:.6f} ± {std_metrics[4]:.6f}, Recall Weighted: {mean_metrics[5]:.6f} ± {std_metrics[5]:.6f}, "
                        f"F1 Weighted: {mean_metrics[6]:.6f} ± {std_metrics[6]:.6f}")
            logging.info(f"Cohen's Kappa: {mean_metrics[7]:.6f} ± {std_metrics[7]:.6f}, MCC: {mean_metrics[8]:.6f} ± {std_metrics[8]:.6f}")
            writer.writerow({
                'out_channels': out_channels,
                'D': D,
                'eeg_groups': eeg_groups,
                'mean_accuracy': f"{mean_metrics[0]:.6f}",
                'std_accuracy': f"{std_metrics[0]:.6f}",
                'mean_f1': f"{mean_metrics[1]:.6f}",
                'std_f1': f"{std_metrics[1]:.6f}",
                'mean_precision': f"{mean_metrics[2]:.6f}",
                'std_precision': f"{std_metrics[2]:.6f}",
                'mean_recall': f"{mean_metrics[3]:.6f}",
                'std_recall': f"{std_metrics[3]:.6f}",
                'mean_precision_weighted': f"{mean_metrics[4]:.6f}",
                'std_precision_weighted': f"{std_metrics[4]:.6f}",
                'mean_recall_weighted': f"{mean_metrics[5]:.6f}",
                'std_recall_weighted': f"{std_metrics[5]:.6f}",
                'mean_f1_weighted': f"{mean_metrics[6]:.6f}",
                'std_f1_weighted': f"{std_metrics[6]:.6f}",
                'mean_kappa': f"{mean_metrics[7]:.6f}",
                'std_kappa': f"{std_metrics[7]:.6f}",
                'mean_mcc': f"{mean_metrics[8]:.6f}",
                'std_mcc': f"{std_metrics[8]:.6f}"
            })
            if mean_metrics[0] > best_global_acc:
                best_global_acc = mean_metrics[0]
                best_global_params = {
                    'out_channels': out_channels,
                    'D': D,
                    'eeg_groups': eeg_groups
                }
                best_global_metrics = {
                    'accuracy': mean_metrics[0],
                    'f1': mean_metrics[1],
                    'precision': mean_metrics[2],
                    'recall': mean_metrics[3],
                    'precision_weighted': mean_metrics[4],
                    'recall_weighted': mean_metrics[5],
                    'f1_weighted': mean_metrics[6],
                    'kappa': mean_metrics[7],
                    'mcc': mean_metrics[8]
                }
    logging.info(f"Best Global Parameters: out_channels={best_global_params['out_channels']}, "
                f"D={best_global_params['D']}, eeg_groups={best_global_params['eeg_groups']}, "
               )
    logging.info(f"Best Global Metrics - Accuracy: {best_global_metrics['accuracy']:.6f}, "
                f"F1: {best_global_metrics['f1']:.6f}, Precision: {best_global_metrics['precision']:.6f}, "
                f"Recall: {best_global_metrics['recall']:.6f}, Precision Weighted: {best_global_metrics['precision_weighted']:.6f}, "
                f"Recall Weighted: {best_global_metrics['recall_weighted']:.6f}, F1 Weighted: {best_global_metrics['f1_weighted']:.6f}, "
                f"Cohen's Kappa: {best_global_metrics['kappa']:.6f}, MCC: {best_global_metrics['mcc']:.6f}")
    return best_global_params, best_global_metrics


def load_data(data_path, data_name):
    with h5py.File(os.path.join(data_path, data_name), 'r') as dataset:
        data = np.array(dataset['data'], dtype=np.float32)
        if len(np.shape(data)) < 4:
            data = np.expand_dims(data, axis=1)
        data = normalize(data)
        label = np.array(dataset['label'])
        electrodes_to_extract = [32, 14, 13, 12, 33, 29, 15, 16, 17, 30, 28, 20, 19, 18, 27, 21, 22, 26, 25, 24]  # dataset==datset A
        # electrodes_to_extract = [12, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24, 25, 26, 27, 61, 44, 62, 29, 30, 31] # dataset==datset B
        electrodes_to_extract = np.array(electrodes_to_extract) - 1
        data = data[:, :, electrodes_to_extract, :]
    return torch.from_numpy(data).float(), torch.from_numpy(label).long()


if __name__ == "__main__":
    config_file = os.path.join(PROJECT_ROOT, 'config.yaml')
    config = load_config(config_file)
    best_params, best_metrics = k_fold_cross_validation(config)
    logging.info(f"Best Parameters: {best_params}")
    logging.info(f"Best Metrics: {best_metrics}")
