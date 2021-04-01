import argparse, json

from os import makedirs
from os.path import join, dirname

import numpy as np
from PIL import Image

from tqdm import trange
import torch

import filters

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='samples/VR.jpg',
        help='input image')
    parser.add_argument('--gt-mask', type=str, default='samples/VR_gt_mask.png',
        help='ground truth mask')
    parser.add_argument('--gt-ignore', type=str, default='samples/VR_gt_ignore.png',
        help='ground truth mask')
    parser.add_argument('--filter', type=str, default='OBSChromaKey',
        help='select the filter algorithm')
    parser.add_argument('--color-space', type=str, default='YCbCr', choices=['YCbCr', 'RGB', 'HSV'],
        help='apply chroma key in which color space (for BasicChromaKey)')
    parser.add_argument('--n-sample', type=int, default=10000,
        help='test how many sets of parameters in total')
    parser.add_argument('--batch-size', type=int, default=20,
        help='test how many sets of parameters at a time (pick the largest that can fit your GPU memory)')
    parser.add_argument('--out-para', type=str, default='out/VR_para.json',
        help='output optimal parameters')
    parser.add_argument('--out-mask', type=str, default='out/VR_mask.png',
        help='output mask')
    parser.add_argument('--out-img', type=str, default='out/VR_out.jpg',
        help='output image')
    parser.add_argument('--device', type=str, default='cuda',
        help='whether to use CUDA to speed up')
    parser.add_argument('--seed', type=int, default=0,
        help='random seed for the search')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    img = Image.open(args.input)
    gt_mask = np.array(Image.open(args.gt_mask).convert('L'))
    if args.gt_ignore:
        gt_ignore = np.array(Image.open(args.gt_ignore).convert('L')) < 128
    else:
        gt_ignore = None
        
    with torch.no_grad():
        if args.seed is not None:
            torch.manual_seed(args.seed)

        fil = vars(filters)[args.filter](args, img, gt_mask, gt_ignore)

        loss_all = torch.empty(args.n_sample, dtype=torch.float, device=args.device)

        n_iter = int(np.ceil(args.n_sample/args.batch_size))
        for i in trange(n_iter):
            i_start = i*args.batch_size
            i_end = min((i + 1)*args.batch_size, args.n_sample)
            loss = fil.test_range(i_start, i_end)
            loss_all[i_start: i_end] = loss

        min_idx = loss_all.min(0)[1]

        opt_para = fil.export_para(min_idx)
        makedirs(dirname(args.out_para), exist_ok=True)
        json.dump(opt_para, open(args.out_para, 'w'), indent=4)

        out_img, out_mask = fil.get_result(min_idx)
        makedirs(dirname(args.out_img), exist_ok=True)
        Image.fromarray(out_img).save(args.out_img)

        makedirs(dirname(args.out_mask), exist_ok=True)
        Image.fromarray(out_mask).save(args.out_mask)

if __name__ == '__main__':
    main()
