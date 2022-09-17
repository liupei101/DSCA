import sys
import os.path as osp
import numpy as np
import pandas as pd
import random
import h5py
import torch
from torch import Tensor


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print('[setup] seed: {}'.format(seed))

def setup_device(no_cuda, cuda_id, verbose=True):
    device = 'cpu'
    if not no_cuda and torch.cuda.is_available():
        device = 'cuda' if cuda_id < 0 else 'cuda:{}'.format(cuda_id)
    if verbose:
        print('[setup] device: {}'.format(device))

    return device

# worker_init_fn = seed_worker
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# generator = g
def seed_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def collect_tensor(collector, y, y_hat):
    if collector['y'] is None:
        collector['y'] = y
    else:
        collector['y'] = torch.cat([collector['y'], y], dim=0)

    if collector['y_hat'] is None:
        collector['y_hat'] = y_hat
    else:
        collector['y_hat'] = torch.cat([collector['y_hat'], y_hat], dim=0)
    
    return collector

def to_patient_data(df, at_column='patient_id'):
    df_gps = df.groupby('patient_id').groups
    df_idx = [i[0] for i in df_gps.values()]
    return df.loc[df_idx, :]

def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)
    
    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
    
    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)

def coord_discretization(wsi_coord: Tensor):
    """
    Coordinate Discretization.
    If the value of coordinates is too large (such as 100,000), it will need super large space 
    when computing the positional embedding of patch.
    """
    x, y = wsi_coord[:, 0].tolist(), wsi_coord[:, 1].tolist()
    sorted_x, sorted_y = sorted(list(set(x))), sorted(list(set(y))) # remove duplicates and then sort
    xmap, ymap = {v:i for i, v in enumerate(sorted_x)}, {v:i for i, v in enumerate(sorted_y)}
    nx, ny = [xmap[v] for v in x], [ymap[v] for v in y]
    res = torch.tensor([nx, ny], dtype=wsi_coord[0].dtype, device=wsi_coord[0].device)
    return res.T

def to_relative_coord(wsi_coord: Tensor):
    ref_xy, _ = torch.min(wsi_coord, dim=-2)
    top_xy, _ = torch.max(wsi_coord, dim=-2)
    rect = top_xy - ref_xy
    ncoord = wsi_coord - ref_xy
    # print("To relative coordinates:", ref_xy, rect)
    return ncoord, ref_xy, rect

def rearrange_coord(wsi_coords, offset_coord=[1, 0], discretization=False):
    """
    wsi_coord (list(torch.Tensor)): list of all patch coordinates of one WSI.
    offset_coord (list): it is set as [1, 0] by default, which means putting WSIs horizontally.
    """
    assert isinstance(wsi_coords, list)
    ret = []
    off_coord = torch.tensor([offset_coord], dtype=wsi_coords[0].dtype, device=wsi_coords[0].device)
    top_coord = -1 * off_coord
    for coord in wsi_coords:
        if discretization:
            coord = coord_discretization(coord)
        new_coord, ref_coord, rect = to_relative_coord(coord)
        new_coord = top_coord + off_coord + new_coord
        top_coord = top_coord + off_coord + rect
        ret.append(new_coord)
    return ret

##################################################################
#
#                     Functionality: I/O
# 
##################################################################
def print_config(config, print_to_path=None):
    if print_to_path is not None:
        f = open(print_to_path, 'w')
    else:
        f = sys.stdout
    
    print("**************** MODEL CONFIGURATION ****************", file=f)
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val), file=f)
    print("**************** MODEL CONFIGURATION ****************", file=f)
    
    if print_to_path is not None:
        f.close()

def print_metrics(metrics, print_to_path=None):
    if print_to_path is not None:
        f = open(print_to_path, 'w')
    else:
        f = sys.stdout
    
    print("**************** MODEL METRICS ****************", file=f)
    for key in sorted(metrics.keys()):
        val = metrics[key]
        for v in val:
            cur_key = key + '/' + v[0]
            keystr  = "{}".format(cur_key) + (" " * (20 - len(cur_key)))
            valstr  = "{}".format(v[1])
            if isinstance(v[1], list):
                valstr = "{}, avg/std = {:.5f}/{:.5f}".format(valstr, np.mean(v[1]), np.std(v[1]))
            print("{} -->   {}".format(keystr, valstr), file=f)
    print("**************** MODEL METRICS ****************", file=f)
    
    if print_to_path is not None:
        f.close()

def read_datasplit_npz(path: str):
    data_npz = np.load(path)
    
    pids_train = [str(s) for s in data_npz['train_patients']]
    pids_val   = [str(s) for s in data_npz['val_patients']]
    if 'test_patients' in data_npz:
        pids_test = [str(s) for s in data_npz['test_patients']]
    else:
        pids_test = None
    return pids_train, pids_val, pids_test

def read_coords(path: str, dtype: str = 'torch'):
    r"""Read patch coordinates from path.

    Args:
        path (string): Read data from path.
        dtype (string): Type of return data, default `torch`.
    """
    assert dtype in ['numpy', 'torch']

    with h5py.File(path, 'r') as hf:
        nfeats = hf['coords'][:]

    if isinstance(nfeats, np.ndarray) and dtype == 'torch':
        return torch.from_numpy(nfeats)
    else:
        return nfeats 

def read_nfeats(path: str, dtype: str = 'torch'):
    r"""Read node features from path.

    Args:
        path (string): Read data from path.
        dtype (string): Type of return data, default `torch`.
    """
    assert dtype in ['numpy', 'torch']
    ext = osp.splitext(path)[1]

    if ext == '.h5':
        with h5py.File(path, 'r') as hf:
            nfeats = hf['features'][:]
    elif ext == '.pt':
        nfeats = torch.load(path, map_location=torch.device('cpu'))
    else:
        raise ValueError(f'not support {ext}')

    if isinstance(nfeats, np.ndarray) and dtype == 'torch':
        return torch.from_numpy(nfeats)
    elif isinstance(nfeats, Tensor) and dtype == 'numpy':
        return nfeats.numpy()
    else:
        return nfeats

def save_prediction(pids, y_true, y_pred, save_path):
    r"""Save surival prediction.

    Args:
        y_true (Tensor or ndarray): true labels.
        y_pred (Tensor or ndarray): predicted values.
        save_path (string): path to save.

    If it is a discrete model:
        y: [B, 2] (col1: y_t, col2: y_c)
        y_hat: [B, BINS]
    else:
        y: [B, 1]
        y_hat: [B, 1]
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.numpy()
    
    print(y_pred.shape, y_true.shape)
    if y_true.shape[1] == 1:
        y_pred = np.squeeze(y_pred)
        y_true = np.squeeze(y_true)
        df = pd.DataFrame({'patient_id': pids, 'pred': y_pred, 'true': y_true}, columns=['patient_id', 'true', 'pred'])
    elif y_true.shape[1] == 2:
        bins = y_pred.shape[1]
        y_t, y_e = y_true[:, [0]], 1 - y_true[:, [1]]
        survival = np.cumprod(1 - y_pred, axis=1)
        risk = np.sum(survival, axis=1, keepdims=True)
        arr = np.concatenate((y_t, y_e, risk, survival), axis=1) # [B, 3+BINS]
        df = pd.DataFrame(arr, columns=['t', 'e', 'risk'] + ['surf_%d' % (_ + 1) for _ in range(bins)])
        df.insert(0, 'patient_id', pids)
    df.to_csv(save_path, index=False)