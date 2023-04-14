import argparse
import os

from opttta import OptTTA


def ensure_dirs(checkpoints_dir):
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    if not os.path.exists(os.path.join(checkpoints_dir, 'visuals')):
        os.makedirs(os.path.join(checkpoints_dir, 'visuals'))

    if not os.path.exists(os.path.join(checkpoints_dir, 'predictions')):
        os.makedirs(os.path.join(checkpoints_dir, 'predictions'))



def get_source_free_domain_adaptaion_options(parser):
    ## Experiment Specific
    parser.add_argument('--checkpoints_source_free_da', default = '/home/syou/OptTTA/bilinear_1em6_torch_t/adaptives/20000/bilinear_1em6_torch_transpose_opptta', type=str)
    parser.add_argument('--checkpoints_source_segmentor', default = '/home/syou/OptTTA/bilinear_1em6_torch_t', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--bilinear', default = False, type = bool)
    ## Sizes
    parser.add_argument('--crop_size', default=256, type=int)
    parser.add_argument('--ncolor_channels', default=1, type=int)
    parser.add_argument('--n_classes', default=3, type=int)

    ## Datasets
    parser.add_argument('--dataroot', default = '/home/syou/OptTTA_tf/SpinalCord_nodepth_test_npy', type=str) # default='../datasets_slices/spinal_cord_no_depth_interpolation/train'
    parser.add_argument('--target_sites', default = '3')

    ## Networks
    parser.add_argument("--n_steps", default=1000, type=int)
    parser.add_argument("--alpha_1", default=1, type=float)
    parser.add_argument("--alpha_2", default=1, type=float)
    parser.add_argument("--n_augs", default=6, type=int)
    parser.add_argument("--bs", default=16, type=int)

    parser.add_argument("--k", default=128, type=int)
    parser.add_argument("--sp_selection_metric", default="All", type=str, choices=("All, BN, Ent"))


    opt = parser.parse_args()
    opt.gpu_id = 'cuda:%s'%opt.gpu_id
    if opt.dataset_mode == 'prostate':
        opt.target_sites = ['site-'+ site_nbr for site_nbr in opt.target_sites.split(',')]
    else:
        opt.target_sites = ['site'+ site_nbr for site_nbr in opt.target_sites.split(',')]
    return opt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Source Free Adaptation to test time Image.')
    opt_style = get_source_free_domain_adaptaion_options(parser)
    ensure_dirs(opt_style.checkpoints_source_free_da)
    trainer = OptTTA(opt_style)
    trainer.launch()