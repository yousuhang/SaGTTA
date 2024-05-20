import argparse
import os
import json

from sagtta import SOptTTA_GC


def get_source_free_domain_adaptaion_options(parser):
    ## Experiment Specific/home/syou/nerve/OptTTA/main_sopttta_gc.py
    parser.add_argument('--checkpoints_source_free_da', default = '/home/syou/sagtta_experiments/logs_spinal_cord_site1/bilinear_1em6_torch_t/trial1', type=str)
    parser.add_argument('--checkpoints_source_segmentor', default = '/home/syou/sagtta_experiments/logs_spinal_cord_site1/bilinear_1em6_torch_t', type=str)
    parser.add_argument('--candidate_policy_file', default = None, type = str)
    parser.add_argument('--sal1_train_path', default = None)
    parser.add_argument('--sal2_train_path', default = None)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--bilinear', default = False, type = bool)
    parser.add_argument('--inter_steps', default=3, type=int)
    ## Sizes
    parser.add_argument('--crop_size', default=256, type=int)
    parser.add_argument('--ncolor_channels', default=1, type=int)
    parser.add_argument('--n_classes', default=3, type=int)

    ## Datasets
    parser.add_argument('--dataroot', default = '/home/syou/SpinalCord_test_npy', type=str) # default='../datasets_slices/spinal_cord_no_depth_interpolation/train'
    parser.add_argument('--target_sites', default = '3')

    ## Networks
    parser.add_argument("--n_steps", default=1000, type=int)
    parser.add_argument("--alpha_0", default=1, type=float)
    parser.add_argument("--alpha_1", default=1, type=float)
    parser.add_argument("--alpha_2", default=1, type=float)
    parser.add_argument("--alpha_3", default=1, type=float)
    parser.add_argument("--beta", default=1, type=float)
    parser.add_argument("--n_augs", default=5, type=int)
    parser.add_argument("--bs", default=12, type=int)

    parser.add_argument("--k", default=128, type=int)
    parser.add_argument("--sp_selection_metric", default="All", type=str, choices=("All, BN, Ent"))


    opt = parser.parse_args()
    opt.gpu_id = 'cuda:%s'%opt.gpu_id

    opt.target_sites = ['site'+ site_nbr for site_nbr in opt.target_sites.split(',')]
    return opt




def ensure_dirs(checkpoints_dir):
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    if not os.path.exists(os.path.join(checkpoints_dir, 'visuals')):
        os.makedirs(os.path.join(checkpoints_dir, 'visuals'))

    if not os.path.exists(os.path.join(checkpoints_dir, 'predictions')):
        os.makedirs(os.path.join(checkpoints_dir, 'predictions'))

    if not os.path.exists(os.path.join(checkpoints_dir, 'uncertainties')):
        os.makedirs(os.path.join(checkpoints_dir, 'uncertainties'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Source Free Adaptation to test time Image.')
    opt_style = get_source_free_domain_adaptaion_options(parser)
    opt_dict = vars(opt_style)
    ensure_dirs(opt_style.checkpoints_source_free_da)
    with open(f'{opt_style.checkpoints_source_free_da}/argument.json', "w") as outfile:
        json.dump(opt_dict, outfile)
    trainer = SOptTTA_GC(opt_style)
    trainer.launch()
