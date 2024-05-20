import torch.utils.data as data
import os
import torch
import numpy as np
import albumentations as A
from data_prep.image_transformations import get_transform
from utils.utils import natural_sort


class GenericDataset(data.Dataset):
    def __init__(self, rootdir, sites, phase='train', split_train=False, seed=0, batch_size=0):
        self.batch_size = batch_size
        img_query = 'image'
        seg_query = 'mask'
        self.rootdir = rootdir
        self.sites = sites
        self.all_imgs = np.array(
            natural_sort([f for f in os.listdir(rootdir) if img_query in f and self.is_site(f)]))

        self.all_segs = np.array(natural_sort([f for f in os.listdir(rootdir) if seg_query in f and self.is_site(f)]))

        assert len(self.all_imgs) == len(self.all_segs)
        # data augmentations
        self.augmenter = A.Compose(get_transform(phase))
        np.random.seed(seed)
        self.indeces = np.random.choice(range(1, 11), 8, replace=False)  # split train and valid to 4:1
        if split_train:
            train_brain = ['sc' + str(i).zfill(2) for i in self.indeces]
            self.sampled_train_imgs = [i for i in self.all_imgs for j in train_brain if j in i]
            self.sampled_train_segs = [i for i in self.all_segs for j in train_brain if j in i]
            self.sampled_valid_imgs = [i for i in self.all_imgs if i not in self.sampled_train_imgs]
            self.sampled_valid_segs = [i for i in self.all_segs if i not in self.sampled_train_segs]

        else:
            self.sampled_train_idx = np.arange(len(self.all_imgs))
            self.sampled_train_imgs = self.all_imgs
            self.sampled_train_segs = self.all_segs

        if phase == 'train' or phase == 'test':

            self.all_imgs = np.sort(self.sampled_train_imgs)
            self.all_segs = np.sort(self.sampled_train_segs)

        elif phase == 'val':  # chose validation data from the training set

            self.all_imgs = np.sort(self.sampled_valid_imgs)
            self.all_segs = np.sort(self.sampled_valid_segs)
        else:
            raise Exception('Unrecognized phase.')

    def get_indeces(self):
        return self.indeces

    def is_site(self, name):
        for site in self.sites:
            if site in name:
                return True

        return False

    def __getitem__(self, index):
        if self.batch_size != 0:
            index = index % len(self.all_imgs)

        img = np.load(os.path.join(self.rootdir, self.all_imgs[index])).astype(np.float32)
        seg = np.load(os.path.join(self.rootdir, self.all_segs[index]))

        transformed = self.augmenter(image=img, mask=seg)

        img = transformed['image']
        img = 2 * torch.from_numpy(img).to(torch.float32).unsqueeze(0) - 1

        seg = transformed['mask']
        seg = torch.from_numpy(seg).to(torch.long)

        return img, seg

    def __len__(self):
        return max(len(self.all_imgs), self.batch_size)


class GenericVolumeDataset(data.Dataset):
    def __init__(self, rootdir, sites, phase='train'):

        img_query = 'image'
        seg_query = 'mask'

        self.rootdir = rootdir
        self.sites = sites

        all_imgs = natural_sort([f for f in os.listdir(rootdir) if img_query in f and self.is_site(f)])
        all_segs = natural_sort([f for f in os.listdir(rootdir) if seg_query in f and self.is_site(f)])

        assert len(all_imgs) == len(all_segs)

        self.grouped_imgs = {}
        self.grouped_segs = {}

        # group the slices based on patient
        for img, seg in zip(all_imgs, all_segs):

            assert img.split('-')[:2] == seg.split('-')[:2]
            patient_name = '-'.join(img.split('-')[:2])

            if patient_name in self.grouped_imgs:
                self.grouped_imgs[patient_name].append(img)
                self.grouped_segs[patient_name].append(seg)

            else:
                self.grouped_imgs[patient_name] = [img]
                self.grouped_segs[patient_name] = [seg]

        # data augmentations
        self.augmenter = A.Compose(get_transform(phase))

        self.all_imgs = [v for _, v in self.grouped_imgs.items()]
        self.all_segs = [v for _, v in self.grouped_segs.items()]

    def is_site(self, name):
        for site in self.sites:
            if site in name:
                return True

        return False

    def __getitem__(self, index):
        img_names = self.all_imgs[index]
        seg_names = self.all_segs[index]

        out_imgs = []
        out_segs = []

        for i_n, s_n in zip(img_names, seg_names):
            img = np.load(os.path.join(self.rootdir, i_n)).astype(np.float32)
            seg = np.load(os.path.join(self.rootdir, s_n))

            transformed = self.augmenter(image=img, mask=seg)

            img = transformed['image']
            img = 2 * torch.from_numpy(img).to(torch.float32).unsqueeze(0) - 1

            seg = transformed['mask']
            seg = torch.from_numpy(seg).to(torch.long)

            out_imgs.append(img)
            out_segs.append(seg)

        out_imgs = torch.stack(out_imgs)
        out_segs = torch.stack(out_segs)

        return out_imgs, out_segs

    def __len__(self):
        return len(self.grouped_imgs)
