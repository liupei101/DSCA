#coding=utf-8
import sys
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import KFold, train_test_split


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    print('[setup] seed: {}'.format(seed))

def to_patient_data(df, at_column='patient_id'):
    df_gps = df.groupby('patient_id').groups
    df_idx = [i[0] for i in df_gps.values()]
    ret = df.loc[df_idx, :]
    ret = ret.reset_index(drop=True)
    return ret

def main(dataset, seed):
    seed_everything(seed)
    df = pd.read_csv('../data_split/{}/{}_path_full.csv'.format(dataset, dataset))
    df = to_patient_data(df)
    sample_idx = np.arange(len(df))
    # 5-fold
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    fold = 0
    for train_index, test_index in kf.split(sample_idx):
        # split train_index into train/val
        train_index, val_index = train_test_split(train_index, test_size=0.2, random_state=seed)
        # train/val/test
        train_ids = list(df.loc[train_index, 'patient_id'])
        val_ids = list(df.loc[val_index, 'patient_id'])
        test_ids = list(df.loc[test_index, 'patient_id'])
        # save patients to files
        np.savez('../data_split/{}/{}-seed{}-fold{}.npz'.format(dataset, dataset, seed, fold),
               train_patients=train_ids, val_patients=val_ids, test_patients=test_ids)
        fold += 1

# python3 data_split.py nlst/tcga_brca 42
if __name__ == '__main__':
    dataset = sys.argv[1]
    seed = int(sys.argv[2])
    main(dataset, seed)