import argparse
import json
import os

from grid_search import GridSearchEvaluator


def get_source_segmentor_options(parser):
    ## Experiment Specific
    parser.add_argument('--checkpoints_dir', default='/media/logs_spinal_cord_site1/bilinear_1em6_torch_t', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--resume_iter', default=200000)
    parser.add_argument('--continue_train', action='store_true')
    parser.add_argument('--use_bilinear', default=False, type=bool)
    parser.add_argument('--statistic_path', default='/media/logs_spinal_cord_site1/bilinear_1em6_torch_t/eval_gamma_', type=str)
    parser.add_argument('--gamma_value', default=-1, type = float)
    parser.add_argument('--grid_search_set_path', default='./grid_search/para_sets.json', type=str)
    ## Sizes
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--crop_size', default=256, type=int)
    parser.add_argument('--ncolor_channels', default=1, type=int)
    parser.add_argument('--n_classes', default=3, type=int)
    parser.add_argument('--volumne_index', default=0, type = int)
    ## Datasets
    parser.add_argument('--dataroot', default='/media/SpinalCord_nodepth_test_npy',
                        type=str)  # default='../datasets_slices/spinal_cord/train',
    parser.add_argument('--source_sites', default='3', type=str)  # default='1,2,3'
    parser.add_argument('--dataset_mode', choices=['spinalcord', 'heart', 'prostate', 'skinlesion'],
                        default='spinalcord')  # default='spinalcord'
    parser.add_argument('--n_dataloader_workers', default=4)
    parser.add_argument('--data_ratio', type=float, default=1.)
    parser.add_argument('--target_UB', action='store_true',
                        help="whether we train on upper bound. We take 3/10 of the data for training")

    ## optimizer
    parser.add_argument("--lr", default=0.00001, type=float)

    ## display
    parser.add_argument("--total_nimgs", default=251000, type=int)
    parser.add_argument("--save_freq", default=50000, type=int)
    parser.add_argument("--evaluation_freq", default=10000, type=int)
    parser.add_argument("--print_freq", default=480, type=int)
    parser.add_argument("--display_freq", default=10000, type=int)
    parser.add_argument("--save_visuals", default=True, type=bool)

    ## test mode
    parser.add_argument("--test", action="store_true", help="whether we enter in test mode")

    opt = parser.parse_args()
    opt.gpu_id = 'cuda:%s' % opt.gpu_id
    opt.source_sites = ['site' + site_nbr for site_nbr in opt.source_sites.split(',')]

    if opt.test:
        opt.continue_train = True
    # opt.statistic_path = opt.statistic_path + '_volume_{:02}.csv'.format(opt.volumne_index+1)
    return opt
# CUDA_VISIBLE_DEVICES

def ensure_dirs(checkpoints_dir):
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    if not os.path.exists(os.path.join(checkpoints_dir, 'console_logs')):
        os.makedirs(os.path.join(checkpoints_dir, 'console_logs'))

    if not os.path.exists(os.path.join(checkpoints_dir, 'tf_logs')):
        os.makedirs(os.path.join(checkpoints_dir, 'tf_logs'))

    if not os.path.exists(os.path.join(checkpoints_dir, 'saved_models')):
        os.makedirs(os.path.join(checkpoints_dir, 'saved_models'))

    if not os.path.exists(os.path.join(checkpoints_dir, 'visuals')):
        os.makedirs(os.path.join(checkpoints_dir, 'visuals'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Segmentor on Source Images')
    opt_style = get_source_segmentor_options(parser)
    print(opt_style.checkpoints_dir)
    print(opt_style.resume_iter)
    ensure_dirs(opt_style.checkpoints_dir)
    trainer = GridSearchEvaluator(opt_style)
    trainer.launch()