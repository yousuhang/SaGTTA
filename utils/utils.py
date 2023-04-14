import os
import re

import torch

COLORS = 2*torch.tensor([[0, 0, 0], [197,17,17], [239,125,14], [246,246,88], [237,84,186], [113,73,30]], dtype=float)/255.0 - 1.0

def clip_gradient(optimizer):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                torch.nn.utils.clip_grad_norm_(param, 1.0, norm_type=2.0)

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]
def natural_sort(items):
    new_items = items.copy()
    new_items.sort(key=natural_keys)
    return new_items

def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def overlay_segs(img, seg, alpha=0.2):
    """
    imgs should be in range [-1, 1] and shape C x H x W or B x C x H x W.
    seg should have integer range with same spatial shape as H x W or B x H x W.
    """
    img = img.detach().cpu()
    seg = seg.detach().cpu()

    assert img.size()[-2:] == seg.size()[-2:]
    if img.size()[-3] == 1:
        img = img.repeat([3, 1, 1]) if img.dim() == 3 else img.repeat([1, 3, 1, 1])

    mask = (seg != 0)

    if seg.dim() == 3:
        mask = mask.unsqueeze(1)

    color_seg = COLORS[seg].permute([2, 0, 1]) if seg.dim() == 2 else COLORS[seg].permute([0, 3, 1, 2])

    ### colors from 0 and 1
    color_seg = color_seg.clamp(-1, 1) * 0.5 + 0.5
    img = img.clamp(-1, 1) * 0.5 + 0.5

    merged = mask * (alpha * color_seg + (1 - alpha) * img) + (~mask) * img

    return merged
def getcolorsegs(seg):
    seg = seg.detach().cpu()
    color_seg = COLORS[seg].permute([2, 0, 1]) if seg.dim()==2 else COLORS[seg].permute([0, 3, 1, 2])
    color_seg = color_seg.clamp(-1, 1) * 0.5 + 0.5
    return color_seg