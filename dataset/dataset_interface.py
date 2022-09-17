from .WSIPatchData import WSIPatchDataset
from .WSIPatchData import WSIOnePatchDataset


def prepare_dataset(patient_ids, cfg, magnification='x5_x20'):
    assert magnification in ['x5', 'x20', 'x5_x20']
    if magnification == 'x5_x20':
        path_patchx20 = cfg['path_patchx20']
        path_patchx5 = cfg['path_patchx5']
        path_coordx5 = cfg['path_coordx5']
        path_label = cfg['path_label']
        label_discrete = cfg['label_discrete']
        bins_discrete = cfg['bins_discrete']
        feat_format = cfg['feat_format']
        dataset = WSIPatchDataset(
            patient_ids, path_patchx20, path_patchx5, path_coordx5, path_label,
            label_discrete=label_discrete,
            bins_discrete=bins_discrete,
            feat_format=feat_format,
        )
    else:
        if magnification == 'x20' and cfg['emb_backbone'] == 'identity':
            path_ref_coord = cfg['path_coordx20']
            print("[info] got emb_backbone=identity and 20x magnification, so read patch coordinates from x20")
        else:
            path_ref_coord = cfg['path_coordx5']
            print("[info] read patch coordinates from x5")
        path_patch = cfg['path_patchx5'] if magnification == 'x5' else cfg['path_patchx20']
        path_label = cfg['path_label']
        label_discrete = cfg['label_discrete']
        bins_discrete = cfg['bins_discrete']
        feat_format = cfg['feat_format']
        num_sampling = cfg['num_patch_sampling']
        dataset = WSIOnePatchDataset(
            patient_ids, path_patch, path_ref_coord, path_label,
            label_discrete=label_discrete,
            bins_discrete=bins_discrete,
            feat_format=feat_format,
            num_sampling=num_sampling,
        )

    return dataset
