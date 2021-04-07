'''
This module approximates the OBS Color Key Filter [1] using PyTorch
It also contains a method to sample the parameters for this filter

[1] https://github.com/obsproject/obs-studio/blob/4ff3d6b300559c8f7a78a131413f19b7f3199642/plugins/obs-filters/data/color_key_filter_v2.effect
'''

from os.path import basename

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from .base_filter import BaseFilter
from .utils import linear_srgb, rgb2hexstr


class OBSColorKey(BaseFilter):
    def __init__(self, args, img, gt_mask, gt_ignore=None):
        super().__init__()
        self.device = args.device
        self.n_sample = args.n_sample
        self.color_space = 'Linear RGB'

        self.in_img = img.convert('RGB')
        img = np.array(self.in_img)
        img = torch.from_numpy(img).to(self.device).float()
        # color space conversion
        img = linear_srgb(img/255.0)
        # to BxCxHxW
        self.img = img.permute(2, 0, 1).unsqueeze(0)

        self.gt_mask = torch.from_numpy(gt_mask).to(self.device).float().unsqueeze(0)
        if gt_ignore is None:
            self.gt_ignore = None
        else:
            self.gt_ignore = torch.from_numpy(gt_ignore).to(self.device).unsqueeze(0)

        self.sample_parameters()

    def sample_parameters(self):
        self.key_all = torch.randint(0, 256, (self.n_sample, 3), dtype=torch.float, device=self.device)/255.0
        self.sim_all = torch.randint(0, 1000, (self.n_sample,), dtype=torch.float, device=self.device)/1000.0
        self.smo_all = torch.randint(0, 1000, (self.n_sample,), dtype=torch.float, device=self.device)/1000.0

    def batch_filter(self, key, similarity, smooth):
        n = key.shape[0]
        imgs = self.img.expand(n, -1, -1, -1)

        if key.ndim == 2:
            key = key.view(n, -1, 1, 1)
        else:
            key = key.view(1, -1, 1, 1)
        similarity = similarity.view(-1, 1, 1)
        smooth = similarity.view(-1, 1, 1)

        # BxCxHxW
        d = imgs - key
        d = (d**2).sum(1).sqrt()

        # BxHxW
        masks = (d - similarity).clamp(min=0)
        masks = (masks / smooth).clamp(0, 1)

        return masks

    def test_range(self, i_start, i_end):
        this_batch_size = i_end - i_start
        masks = self.batch_filter(
            self.key_all[i_start: i_end],
            self.sim_all[i_start: i_end],
            self.smo_all[i_start: i_end],
        )
        
        # L1 loss
        loss = (masks - self.gt_mask).abs()
        if self.gt_ignore is not None:
            ignore_mask = self.gt_ignore.expand(this_batch_size, -1, -1)
            loss[ignore_mask] = 0
        loss = loss.view(this_batch_size, -1)
        loss = loss.sum(1)

        return loss

    def export_para(self, i):
        opt_key = (255.0*self.key_all[i]).byte().cpu().numpy().tolist()
        opt_key_hex = rgb2hexstr(opt_key)
        opt_para = {
            'filter': basename(__file__[:-3]),
            'color space': self.color_space,
            'keying channels': [0, 1, 2],
            'key': opt_key,
            'key in RGB': opt_key_hex,
            'similarity': round(1000.0*self.sim_all[i].item()),
            'smooth': round(1000.0*self.smo_all[i].item()),
        }
        return opt_para

    def get_result(self, i):
        # note that if i is a torch.tensor type, [i] will not return an array
        if isinstance(i, torch.Tensor):
            i = i.item()
        mask = self.batch_filter(
            self.key_all[[i]],
            self.sim_all[[i]],
            self.smo_all[[i]],
        )
        mask[self.gt_ignore] = 0.0
        out_mask = (255*mask[0]).byte().cpu().numpy()

        # note that masking should be done in the RGB space
        out_img = np.uint8(np.array(self.in_img, np.float32)*mask[0].unsqueeze(2).cpu().numpy())

        return out_img, out_mask