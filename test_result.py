import argparse
import os
import albumentations as A
import pandas as pd
import torch
import numpy as np

from utils.metrics import dice_coef_multiclass, segmentation_score_stats
from utils.utils import natural_sort


def is_site(sites, name):
    for site in sites:
        if site in name:
            return True

    return False


@torch.no_grad()
def compute_metrics_one_step(pred, seg, all_classes):
    metrics_dic = {}
    # print(pred.shape)
    # compute dice coefficients
    batch_dice_coef = dice_coef_multiclass(seg, pred, all_classes)
    # print(batch_dice_coef)
    # compute class-wise
    for i, coef in enumerate(batch_dice_coef.T):
        metrics_dic["ds_class_{:d}".format(i)] = torch.tensor(coef)

    # compute sample-wise mean (w/o background)
    metrics_dic["ds"] = torch.tensor(np.nanmean(batch_dice_coef[:, 1:], axis=1))

    return metrics_dic


parser = argparse.ArgumentParser(description="Test script")

# Checkpoint arguments
parser.add_argument('--prediction_path',
                    default='/home/syou/nerve/logs_spinal_cord_site1/bilinear_1em6_torch_t/adaptives/20000/bilinear_1em6_torch_transpose_sopt_a_1',
                    type=str)

# Model arguments
parser.add_argument('--n_classes', default=3, type=int)

# Dataset arguments
parser.add_argument('--dataset_mode', choices=['spinalcord', 'heart', 'prostate'], default='spinalcord', type=str)
parser.add_argument('--dataroot', default='/media/SpinalCord_nodepth_test_npy', type=str)
parser.add_argument('--target_sites', default='3')

opt = parser.parse_args()

if opt.dataset_mode == 'prostate':
    opt.target_sites = ['site-' + site_nbr for site_nbr in opt.target_sites.split(',')]
else:
    opt.target_sites = ['site' + site_nbr for site_nbr in opt.target_sites.split(',')]

# Extract predictions and gt segments
predictions_path = os.path.join(opt.prediction_path, "predictions")
gts_path = os.path.join(opt.dataroot)

predictions_flist = np.array(natural_sort([f for f in os.listdir(predictions_path) if is_site(opt.target_sites, f)]))


patient_roots = np.unique(np.array(natural_sort(['-'.join(x.split('-')[:2]) for x in predictions_flist])))


print(patient_roots)

# Compute scores for the predictions
metrics = None

for patient in patient_roots:

    patient_f_slices = np.array(natural_sort([f for f in predictions_flist if patient in f]))
    all_preds = torch.Tensor()
    all_gts = torch.Tensor()

    for slice in patient_f_slices:
        pred = np.load(os.path.join(predictions_path, slice)).astype(np.float64)
        gt = np.load(os.path.join(opt.dataroot, slice))

        # Resize gt
        gt = A.Resize(256, 256)(image=gt, mask=gt)["mask"]

        # Make tensors
        gt = torch.from_numpy(gt).to(torch.long).unsqueeze(0)
        pred = torch.from_numpy(pred).to(torch.long).unsqueeze(0)

        # Aggregate predictions and gts
        all_preds = torch.cat([all_preds, pred])
        all_gts = torch.cat([all_gts, gt])

    all_classes = np.arange(opt.n_classes)


    all_gts = all_gts.unsqueeze(0)
    all_preds = all_preds.unsqueeze(0)

    # Compute metrics on the 3D volume
    if metrics is None:
        metrics = compute_metrics_one_step(all_preds, all_gts, all_classes)
    else:
        for k, v in compute_metrics_one_step(all_preds, all_gts, all_classes).items():
            metrics[k] = torch.cat((metrics[k], v))
df = pd.DataFrame(metrics)
df.to_csv(f'{opt.prediction_path}/tta_result.csv')
segmentation_score_stats(metrics)
