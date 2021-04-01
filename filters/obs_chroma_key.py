'''
This module approximates the OBS Chroma Key Filter [1] using PyTorch
It also contains a method to sample the parameters for this filter

[1] https://github.com/obsproject/obs-studio/blob/4ff3d6b300559c8f7a78a131413f19b7f3199642/plugins/obs-filters/data/chroma_key_filter_v2.effect

'''

from os.path import basename

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from .base_filter import BaseFilter
from .utils import linear_srgb, rgb2hexstr


class OBSChromaKey(BaseFilter):
    color_cvt = [
        [-0.100644, -0.338572,  0.439216],
        [0.439216, -0.398942, -0.040274],
    ]
    color_cvt_bias = 0.501961 

    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.n_sample = args.n_sample
        self.color_space = 'OBS YCbCr'

        self.in_img = Image.open(args.input).convert('RGB')
        img = np.array(self.in_img)
        img = torch.from_numpy(img).to(self.device).float()
        # to CxHxW
        img = img.permute(2, 0, 1)
        _, self.h, self.w = img.shape

        # color space conversion
        img = linear_srgb(img/255.0)
        img = img.view(3, -1) # flatten the spatial dimensions
        img = torch.mm(torch.tensor(self.color_cvt, device=self.device), img)
        img += self.color_cvt_bias

        # to BxCxHxW
        img = img.view(1, 2, self.h, self.w)

        # unfold for 3x3 color sampling
        img = F.pad(img, (1, 1, 1, 1), mode='reflect')
        # to BxCx9x(HxW)
        self.img = F.unfold(img, 3).view(1, 2, 9, self.h*self.w)

        gt = np.array(Image.open(args.gt_mask).convert('L'))
        gt_ignore = np.array(Image.open(args.gt_ignore).convert('L'))

        self.gt = torch.from_numpy(gt).to(self.device).float().unsqueeze(0)
        self.gt_ignore = torch.from_numpy(gt_ignore).to(self.device).unsqueeze(0) < 128

        self.sample_parameters()

    def sample_parameters(self):
        self.key_rgb_all = torch.randint(0, 256, (self.n_sample, 3), dtype=torch.float, device=self.device)
        self.key_all = torch.mm(self.key_rgb_all, torch.tensor(self.color_cvt, device=self.device).t())
        self.key_all = self.key_all/255.0 + self.color_cvt_bias

        self.sim_all = torch.randint(0, 1000, (self.n_sample,), dtype=torch.float, device=self.device)/1000.0
        self.smo_all = torch.randint(0, 1000, (self.n_sample,), dtype=torch.float, device=self.device)/1000.0


    def batch_filter(self, key, similarity, smooth):
        n = key.shape[0]
        imgs = self.img.expand(n, -1, -1, -1)

        if key.ndim == 2:
            key = key.view(n, -1, 1, 1)
        else:
            key = key.view(1, -1, 1, 1)

        # BxCx9x(HxW)
        d = imgs - key
        d = (d**2).sum(1).sqrt()
        # Bx9x(HxW)
        d = d.mean(1)
        # Bx(HxW)

        masks = d - similarity.unsqueeze(1)
        masks = (masks / smooth.unsqueeze(1)).clamp(0, 1) ** 1.5

        masks = masks.view(n, self.h, self.w)

        return masks

    def test_range(self, i_start, i_end):
        this_batch_size = i_end - i_start
        masks = self.batch_filter(
            self.key_all[i_start: i_end],
            self.sim_all[i_start: i_end],
            self.smo_all[i_start: i_end],
        )
        
        # L1 loss
        loss = (masks - self.gt).abs()
        ignore_mask = self.gt_ignore.expand(this_batch_size, -1, -1)
        loss[ignore_mask] = 0
        loss = loss.view(this_batch_size, -1)
        loss = loss.sum(1)

        return loss

    def export_para(self, i):
        opt_key = self.key_rgb_all[i].byte().cpu().numpy().tolist()
        opt_key_hex = rgb2hexstr(opt_key)
        opt_para = {
            'filter': basename(__file__[:-3]),
            'color space': self.color_space,
            'keying channels': [1, 2],
            'key': opt_key,
            'key in RGB': opt_key_hex,
            'similarity': round(1000.0*self.sim_all[i].item()),
            'smooth': round(1000.0*self.smo_all[i].item()),
            'spill': 'Not implemented. A general rule of thumb might be setting it to 1 or a small value if not using a green screen.'
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