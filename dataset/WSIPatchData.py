import pandas as pd
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import read_nfeats, read_coords
from utils import to_patient_data, rearrange_coord


class SurvLabelTransformer(object):
    """
    SurvLabelTransformer: create label of survival data for model training.
    """
    def __init__(self, path_label, column_t='t', column_e='e', verbose=True):
        super(SurvLabelTransformer, self).__init__()
        self.path_label = path_label
        self.column_t = column_t
        self.column_e = column_e
        self.column_label = None
        self.full_data = pd.read_csv(path_label, dtype={'patient_id': str, 'pathology_id': str})
        
        self.pat_data = to_patient_data(self.full_data, at_column='patient_id')
        self.min_t = self.pat_data[column_t].min()
        self.max_t = self.pat_data[column_t].max()
        if verbose:
            print('[surv label] at patient level')
            print('\tmin/avg/median/max time = {}/{:.2f}/{}/{}'.format(self.min_t, 
                self.pat_data[column_t].mean(), self.pat_data[column_t].median(), self.max_t))
            print('\tratio of event = {}'.format(self.pat_data[column_e].sum() / len(self.pat_data)))

    def to_continuous(self, column_label='y'):
        print('[surv label] to continuous')
        self.column_label = [column_label]

        label = []
        for i in self.pat_data.index:
            if self.pat_data.loc[i, self.column_e] == 0:
                label.append(-1 * self.pat_data.loc[i, self.column_t])
            else:
                label.append(self.pat_data.loc[i, self.column_t])
        self.pat_data.loc[:, column_label] = label
        
        return self.pat_data

    def to_discrete(self, bins=4, column_label_t='y_t', column_label_c='y_c'):
        """
        based on the quartiles of survival time values (in months) of uncensored patients.
        see Chen et al. Multimodal Co-Attention Transformer for Survival Prediction in Gigapixel Whole Slide Images
        """
        print('[surv label] to discrete, bins = {}'.format(bins))
        self.column_label = [column_label_t, column_label_c]

        # c = 1 -> censored/no event, c = 0 -> uncensored/event
        self.pat_data.loc[:, column_label_c] = 1 - self.pat_data.loc[:, self.column_e]

        # discrete time labels
        df_events = self.pat_data[self.pat_data[self.column_e] == 1]
        _, qbins = pd.qcut(df_events[self.column_t], q=bins, retbins=True, labels=False)
        qbins[0] = self.min_t - 1e-5
        qbins[-1] = self.max_t + 1e-5

        discrete_labels, qbins = pd.cut(self.pat_data[self.column_t], bins=qbins, retbins=True, labels=False, right=False, include_lowest=True)
        self.pat_data.loc[:, column_label_t] = discrete_labels.values.astype(int)

        return self.pat_data

    def collect_slide_info(self, pids, column_label=None):
        if column_label is None:
            column_label = self.column_label

        sel_pids, pid2sids, pid2label = list(), dict(), dict()
        for pid in pids:
            sel_idxs = self.full_data[self.full_data['patient_id'] == pid].index
            if len(sel_idxs) > 0:
                sel_pids.append(pid)
                pid2sids[pid] = list(self.full_data.loc[sel_idxs, 'pathology_id'])
                
                pat_idx = self.pat_data[self.pat_data['patient_id'] == pid].index[0]
                pid2label[pid] = list(self.pat_data.loc[pat_idx, column_label])
            else:
                print('[warning] patient {} not found!'.format(pid))

        return sel_pids, pid2sids, pid2label


class WSIPatchDataset(Dataset):
    r"""Dataset class that loads patches from WSI for survival prediction (at slide level).

    Args:
        patient_ids (list): A list of patients that are included in the dataset.
        path_label (string): Path of survival data that gives `patient_id`, `t`, `e` of each 
            slide. Only support CSV file. 
        path_patchx20 (string): Path of patch feature at 20x magnification. 
        path_patchx5 (string): Path of patch feature at 5x magnification. 
    """
    def __init__(self, patient_ids, path_patchx20: str, path_patchx5: str, path_coordx5:str, 
        path_label: str, label_discrete:bool=False, bins_discrete:int=4, feat_format:str='pt'):
        super(WSIPatchDataset, self).__init__()
        self.path_patchx20 = path_patchx20
        self.path_patchx5  = path_patchx5
        self.path_coordx5  = path_coordx5
        self.feat_format   = feat_format

        SurvLabel = SurvLabelTransformer(path_label)
        if label_discrete:
            self.label_column = ['y_t', 'y_c']
            SurvLabel.to_discrete(bins=bins_discrete, column_label_t=self.label_column[0], column_label_c=self.label_column[1])
        else:
            self.label_column = ['y']
            SurvLabel.to_continuous(column_label=self.label_column[0])

        self.pids, self.pid2sids, self.pid2label = SurvLabel.collect_slide_info(patient_ids)

        self.summary()

    def summary(self):
        print(f"Class WSIPatchDataset: #Patients = {len(self.pids)}")

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, index):
        pid = self.pids[index]
        sids = self.pid2sids[pid]

        feats_x20, feats_x5, coors_x5 = [], [], []
        for sid in sids:
            fpath_px20 = osp.join(self.path_patchx20, sid + '.' + self.feat_format)
            fpath_px5  = osp.join(self.path_patchx5,  sid + '.' + self.feat_format)
            fpath_cx5  = osp.join(self.path_coordx5,  sid + '.h5')

            feats_x20.append(read_nfeats(fpath_px20, dtype='torch'))
            feats_x5.append(read_nfeats(fpath_px5,  dtype='torch'))
            coors_x5.append(read_coords(fpath_cx5, dtype='torch'))
        coors_x5 = rearrange_coord(coors_x5, discretization=True)

        feats_x20 = torch.cat(feats_x20, dim=0).to(torch.float)
        feats_x5  = torch.cat(feats_x5,  dim=0).to(torch.float)
        coors_x5  = torch.cat(coors_x5,  dim=0).to(torch.int32)
        assert coors_x5.shape[0] == feats_x5.shape[0]
        assert feats_x20.shape[0] == 16 * feats_x5.shape[0]

        y = torch.Tensor(self.pid2label[pid]).to(torch.float)

        return feats_x20, feats_x5, coors_x5, y


class WSIOnePatchDataset(Dataset):
    r"""Dataset class that loads patches (only support a single magnification) from WSI for 
    survival prediction (at patient level).

    Args:
        patient_ids (list): A list of patients that are included in the dataset.
        path_label (string): Path of survival data that gives `patient_id`, `t`, `e` of each 
            slide. Only support CSV file. 
        path_patch (string): Path of patch feature. 
        path_ref_coord (string): Path of reference coordinates.
    """
    def __init__(self, patient_ids, path_patch: str, path_ref_coord: str, path_label: str, label_discrete:bool=False, 
        bins_discrete:int=4, feat_format:str='pt', num_sampling:int=-1):
        super(WSIOnePatchDataset, self).__init__()
        self.path_patch = path_patch
        self.path_ref_coord = path_ref_coord
        self.feat_format = feat_format
        self.num_sampling = num_sampling

        SurvLabel = SurvLabelTransformer(path_label)
        if label_discrete:
            self.label_column = ['y_t', 'y_c']
            SurvLabel.to_discrete(bins=bins_discrete, column_label_t=self.label_column[0], column_label_c=self.label_column[1])
        else:
            self.label_column = ['y']
            SurvLabel.to_continuous(column_label=self.label_column[0])

        self.pids, self.pid2sids, self.pid2label = SurvLabel.collect_slide_info(patient_ids)

        self.summary()

    def summary(self):
        print(f"Class WSIOnePatchDataset: #Patients = {len(self.pids)}, Patch Sampling = {self.num_sampling}")

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, index):
        pid = self.pids[index]
        sids = self.pid2sids[pid]

        feats = []
        ref_coord = []
        for sid in sids:
            fpath_patch = osp.join(self.path_patch, sid + '.' + self.feat_format)
            fpath_refco = osp.join(self.path_ref_coord,  sid + '.h5')
            nfeats = read_nfeats(fpath_patch, dtype='torch')
            coords = read_coords(fpath_refco, dtype='torch')
            # if sampling
            if self.num_sampling > 0:
                if coords.shape[0] == nfeats.shape[0]:
                    idxs = torch.randperm(nfeats.shape[0])[:min(self.num_sampling,nfeats.shape[0])].long()
                    nfeats = nfeats[idxs, :]
                    coords = coords[idxs, :]
                else:
                    raise RuntimeError('For sampling patches, the length of nfeats and coords must be equal.')

            feats.append(nfeats)
            ref_coord.append(coords)
        ref_coord = rearrange_coord(ref_coord, discretization=True)

        feats = torch.cat(feats, dim=0).to(torch.float)
        ref_coord = torch.cat(ref_coord, dim=0).to(torch.int32)
        y = torch.Tensor(self.pid2label[pid]).to(torch.float)
        assert feats.shape[0] % ref_coord.shape[0] == 0
        # force to have a consistent behaviour for two dataset classes
        place_temp1 = torch.Tensor([0])
        return feats, place_temp1, ref_coord, y
