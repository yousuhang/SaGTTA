import itertools
import json

import torch
import os
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from tqdm import tqdm

from data_prep.dataset_construction import GenericVolumeDataset
from network.unet import UNet
from transformations.transformation import Gamma, GaussianBlur, Contrast, RandomResizeCropV2, Brightness
from utils.iter_counter import IterationCounter
from utils.logger import Logger
from utils.metric_tracker import MetricTracker
from utils.metrics import dice_coef_multiclass




class GridSearchEvaluator():
    def __init__(self, opt):
        self.opt = opt

    def initialize(self):

        self.test_set = GenericVolumeDataset(self.opt.dataroot, self.opt.source_sites,
                                             phase='test')
        # if self.opt.test:
        self.test_dataloader = DataLoader(self.test_set,
                                          batch_size=1,
                                          shuffle=False,
                                          drop_last=False,
                                          num_workers=4
                                          )
        #
        print('Length of test dataset: ', len(self.test_dataloader))



        ## initialize the models
        self.model = UNet(self.opt.ncolor_channels, self.opt.n_classes, bilinear=self.opt.use_bilinear)
        self.mod = [Gamma(),  GaussianBlur(), Contrast(), Brightness()]
        self.resize_img = RandomResizeCropV2()
        self.param_names = ['Gamma', 'GaussianBlur','Contrast',  'Scale_X', 'Scale_Y','#Brightness']
        ## load models if needed
        self.load_models(self.opt.resume_iter)

        ## use gpu
        if self.opt.use_gpu:
            self.model = self.model.to(self.opt.gpu_id)
            self.mod = [aug.to(self.opt.gpu_id) for aug in self.mod]
            self.resize_img = self.resize_img.to(self.opt.gpu_id)
            ## augmentation_parameter list
        self.grid_search_paras = self.load_models(self.opt.grid_search_set_path)
        self.gamma_list = np.linspace(self.grid_search_paras['gamma'][0],self.grid_search_paras['gamma'][1],self.grid_search_paras['gamma'][2])
        self.gaussian_list = np.linspace(self.grid_search_paras['gaussian'][0],self.grid_search_paras['gaussian'][1],self.grid_search_paras['gaussian'][2])
        self.contrast_list = 1+ np.linspace(self.grid_search_paras['contrast'][0],self.grid_search_paras['contrast'][1],self.grid_search_paras['contrast'][2])
        self.brightness_list = np.linspace(self.grid_search_paras['brightness'][0],self.grid_search_paras['brightness'][1],self.grid_search_paras['brightness'][2])
        self.scale_x = np.linspace(self.grid_search_paras['scale_x'][0],self.grid_search_paras['scale_x'][1],self.grid_search_paras['scale_x'][2])
        self.scale_y = np.linspace(self.grid_search_paras['scale_y'][0],self.grid_search_paras['scale_y'][1],self.grid_search_paras['scale_y'][2])
        self.param_list = list(
            itertools.product(
                self.gamma_list,
                self.gaussian_list,
                self.contrast_list,
                self.brightness_list,
                self.scale_x,
                self.scale_y
            ))
        ## metrics
        self.dice_coef = dice_coef_multiclass

        # visualizations
        self.iter_counter = IterationCounter(self.opt)
        self.visualizer = Logger(self.opt)
        self.metric_tracker = MetricTracker()

    def load_models(self, epoch):
        checkpoints_dir = self.opt.checkpoints_dir
        weights = torch.load(os.path.join(checkpoints_dir, 'saved_models', 'Segmentor_%s.pth' % epoch),
                             map_location='cpu')
        self.model.load_state_dict(weights)


    def freeze_weigths(self, net):
        for param in net.parameters():
            param.requires_grad = False

    @staticmethod
    def load_json(path):
        with open(path) as f_in:
            return json.load(f_in)
    @torch.no_grad()
    def compute_metrics_one_step(self, pred, seg, all_classes, metrics_dic):

        # metrics_dic = {}

        # compute dice coefficients
        batch_dice_coef = self.dice_coef(seg, pred, all_classes)
        # compute class-wise
        for i, coef in enumerate(batch_dice_coef.T):
            metrics_dic["ds_class_{:d}".format(i)] = coef[0]  # torch.tensor(coef)

        # compute sample-wise mean (w/o background)
        metrics_dic["ds"] = np.nanmean(batch_dice_coef[:, 1:], axis=1)[
            0]  # torch.tensor(np.nanmean(batch_dice_coef[:, 1:], axis=1))

        return metrics_dic

    @torch.no_grad()
    def eval(self, param_list):
        self.model.eval()
        metric_list = []
        for test_it, (test_imgs, test_segs) in enumerate(self.test_dataloader):
            test_pred = torch.Tensor()

            for i in np.arange(test_imgs.shape[1] // self.opt.batch_size + 1):
                metric_dict = {}
                test_img_unpack = test_imgs[0][self.opt.batch_size * i:self.opt.batch_size * (i + 1)]
                if self.opt.use_gpu:
                    test_img_unpack = test_img_unpack.to(self.opt.gpu_id)
                for j, aug in enumerate(self.mod):
                    test_img_unpack = aug(test_img_unpack, param_list[j])
                    metric_dict[self.param_names[j]] = round(param_list[j], 4)
                    # test_segs_unpack = test_segs[0][self.opt.batch_size*i:self.opt.batch_size*(i+1)]
                test_img_unpack, affine = self.resize_img.test(test_img_unpack, v=[param_list[j+1],param_list[j+2]])
                affine = affine.detach()
                metric_dict[self.param_names[j+1]] = round(param_list[j+1], 4)
                metric_dict[self.param_names[j + 2]] = round(param_list[j + 2], 4)
                inv_affine = self.resize_img.invert_affine(affine)
                    # test_segs_unpack = test_segs_unpack.to(self.opt.gpu_id)
                predict_unpack = self.model(test_img_unpack)

                predict_unpack, inv_affine = self.resize_img.test(predict_unpack, affine=inv_affine)
                predict_unpack = torch.argmax((predict_unpack).softmax(dim=1), dim=1)
                test_pred = torch.cat([test_pred, predict_unpack.detach().cpu()])
            test_pred = test_pred.unsqueeze(0)
            # print(test_pred.shape)
            all_classes = np.arange(self.opt.n_classes)
            metrics = self.compute_metrics_one_step(test_pred, test_segs, all_classes, metric_dict)
            metrics['vol_index'] = test_it
            # metrics[param_name] = param_list
            metric_list.append(metrics)
        metric_df = pd.DataFrame(metric_list)

        return metric_df  # metirc_dict

    def launch(self):
        self.initialize()
        dict_list = []
        dict_all_list = []
        for param in tqdm(self.param_list):
            metric_dict = self.eval(list(param))
            metric_dict_new = {}
            for key in metric_dict.keys()[:-5]:
                metric_dict_new[key] = metric_dict[key][0]
            metric_dict_new['ds_mean'] = np.mean(metric_dict['ds'])
            metric_dict_new['ds_std'] = np.std(metric_dict['ds'])
            dict_list.append(metric_dict)
            dict_all_list.append(metric_dict_new)
        df = pd.concat(dict_list)  # pd.DataFrame(dict_list)
        df2 = pd.DataFrame(dict_all_list)
        df.to_csv(self.opt.statistic_path + str(self.opt.gamma_value) + '.csv')
        df2.to_csv(self.opt.statistic_path+ str(self.opt.gamma_value) + '_all.csv')