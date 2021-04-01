'''
A simple version of chroma key
'''

from os.path import basename

import numpy as np
from PIL import Image

import torch

from .base_filter import BaseFilter
from .utils import rgb2hexstr


class BasicChromaKey(BaseFilter):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.n_sample = args.n_sample
        self.color_space = args.color_space
        if self.color_space == 'YCbCr':
            self.ch_sel = [1, 2]
        elif self.color_space == 'RGB':
            self.ch_sel = [0, 1, 2]
        elif self.color_space == 'HSV':
            self.ch_sel = [0, 1]
        else:
            raise ValueError(f'Unsupported color space: {self.color_space}')

        self.in_img = Image.open(args.input)
        img = np.array(self.in_img.convert(self.color_space))
        img = torch.from_numpy(img).to(self.device).float()
        # to BxCxHxW
        self.img = img.permute(2, 0, 1).unsqueeze(0)
        self.img_ch = self.img[:, self.ch_sel]

        gt = np.array(Image.open(args.gt_mask).convert('L'))
        gt_ignore = np.array(Image.open(args.gt_ignore).convert('L'))

        self.gt = torch.from_numpy(gt).to(self.device).float().unsqueeze(0)
        self.gt_ignore = torch.from_numpy(gt_ignore).to(self.device).unsqueeze(0) < 128

        self.sample_parameters()

    def sample_parameters(self):
        self.key_all = torch.randint(0, 256, (self.n_sample, len(self.ch_sel)), dtype=torch.float, device=self.device)

        max_tol = 256 # the true max is np.sqrt(n_channel*255^2), but that is not useful

        self.tol_a_all = torch.randint(0, max_tol, (self.n_sample,), dtype=torch.float, device=self.device)
        self.tol_b_all = torch.randint(0, max_tol, (self.n_sample,), dtype=torch.float, device=self.device)
        sel = self.tol_b_all < self.tol_a_all
        self.tol_b_all[sel] = self.tol_a_all[sel]

    @staticmethod
    def filter(imgs, key, tol_a, tol_b):
        n = imgs.shape[0]
        if key.ndim == 2:
            key = key.view(n, -1, 1, 1)
        else:
            key = key.view(1, -1, 1, 1)

        d = imgs - key
        d = (d**2).sum(1).sqrt()

        masks = torch.ones_like(d)

        if tol_a.numel() == 1 and tol_b.numel() == 1:
            sel_a = d < tol_a
            sel_b = d < tol_b
            masks[sel_a] = 0.0
            sel = (~sel_a & sel_b)
            masks[sel] = (d[sel] - tol_a)/(tol_b - tol_a)
        else:
            tol_a = tol_a.view(n, 1, 1).expand(d.shape)
            tol_b = tol_b.view(n, 1, 1).expand(d.shape)
            sel_a = d < tol_a
            sel_b = d < tol_b
            masks[sel_a] = 0.0
            sel = (~sel_a & sel_b)
            masks[sel] = (d[sel] - tol_a[sel])/(tol_b[sel] - tol_a[sel])

        return masks

    def test_range(self, i_start, i_end):
        this_batch_size = i_end - i_start
        imgs = self.img_ch.expand(i_end - i_start, -1, -1, -1)
        masks = self.filter(
            imgs,
            self.key_all[i_start: i_end],
            self.tol_a_all[i_start: i_end],
            self.tol_b_all[i_start: i_end],
        )
        
        # L1 loss

        loss = (masks - self.gt).abs()
        ignore_mask = self.gt_ignore.expand(this_batch_size, -1, -1)
        loss[ignore_mask] = 0
        loss = loss.view(this_batch_size, -1)
        loss = loss.sum(1)

        return loss

    def export_para(self, i):
        opt_key = self.key_all[i].byte().cpu().numpy().tolist()
        if self.color_space == 'YCbCr':
            opt_key_padded = (255, *opt_key)
        elif self.color_space == 'RGB':
            opt_key_padded = opt_key
        elif self.color_space == 'HSV':
            opt_key_padded = (*opt_key, 255)
        opt_key_rgb = Image.new(self.color_space, (1, 1), color=opt_key_padded).convert('RGB').getpixel((0, 0))
        opt_key_hex = rgb2hexstr(opt_key_rgb)
        opt_para = {
            'filter': basename(__file__[:-3]),
            'color space': self.color_space,
            'keying channels': self.ch_sel,
            'key': opt_key,
            'key in RGB': opt_key_hex,
            'tol_a': self.tol_a_all[i].item(),
            'tol_b': self.tol_b_all[i].item(),
        }
        return opt_para

    def get_result(self, i):
        mask = self.filter(
            self.img_ch,
            self.key_all[i],
            self.tol_a_all[i],
            self.tol_b_all[i],
        )
        mask[self.gt_ignore] = 0.0
        out_mask = (255*mask[0]).byte().cpu().numpy()

        # note that masking should be done in the RGB space
        out_img = np.uint8(np.array(self.in_img, np.float32)*mask[0].unsqueeze(2).cpu().numpy())

        return out_img, out_mask