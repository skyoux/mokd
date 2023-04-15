
import random
from PIL import ImageFilter, ImageOps
import numpy as np
from numpy.random import randint

import torch

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def drop_rand_patches(X, X_rep=None, max_drop=0.3, max_block_sz=0.25, tolr=0.05):
    #######################
    # X_rep: replace X with patches from X_rep. If X_rep is None, replace the patches with Noise
    # max_drop: percentage of image to be dropped
    # max_block_sz: percentage of the maximum block to be dropped
    # tolr: minimum size of the block in terms of percentage of the image size
    #######################
    
    C, H, W = X.size()
    n_drop_pix = np.random.uniform(0, max_drop)*H*W
    mx_blk_height = int(H*max_block_sz)
    mx_blk_width = int(W*max_block_sz)
    
    tolr = (int(tolr*H), int(tolr*W))
    
    total_pix = 0
    while total_pix < n_drop_pix:
        
        # get a random block by selecting a random row, column, width, height
        rnd_r = randint(0, H-tolr[0])
        rnd_c = randint(0, W-tolr[1])
        rnd_h = min(randint(tolr[0], mx_blk_height)+rnd_r, H) #rnd_r is alread added - this is not height anymore
        rnd_w = min(randint(tolr[1], mx_blk_width)+rnd_c, W)
        
        if X_rep is None:
            X[:, rnd_r:rnd_h, rnd_c:rnd_w] = torch.empty((C, rnd_h-rnd_r, rnd_w-rnd_c), dtype=X.dtype, device='cuda').normal_()
        else:
            X[:, rnd_r:rnd_h, rnd_c:rnd_w] = X_rep[:, rnd_r:rnd_h, rnd_c:rnd_w]    
         
        total_pix = total_pix + (rnd_h-rnd_r)*(rnd_w-rnd_c)

    return X
