import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
from matplotlib.pyplot import imread, imsave
import matplotlib.pyplot as plt
from skimage.transform import resize
import time
import sys
from tqdm import tqdm
import math
import glob

import torch
from torch import optim
from torch_radon import Radon

sys.path.append('../')
sys.path.append('../../')
from admm_utils import *
from models import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
#%%
def run(f_name, specific_result_dir, noise_sigma, num_iter, rho, sigma_0, L):
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    cmap = 'jet' # gray
    img = imread(f_name)
    if img.shape[-1] <= 4:
        img = img[:, :, 0]
    if img.dtype == 'uint8':
        img = img.astype('float32') / 255  # scale to [0, 1]
    elif img.dtype == 'float32':
        img = img.astype('float64')
    else:
        raise TypeError()
    img = np.clip(resize(img, (256, 256)), 0, 1)
    imsave(specific_result_dir + 'true.png', img, cmap=cmap)
    image_size = img.shape[-1]
    img_true = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).type(dtype)  # (1,1,image_size,image_size)

    n_angles = 100
    seeds = 501
    noise_level = 50
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    det_count = int(np.ceil(np.sqrt(2) * image_size))
    radon = Radon(image_size, angles, det_count=det_count)
    sino = radon.forward(torch.tensor(img).type(dtype)).detach().cpu().numpy()

    b_true = torch.from_numpy(sino).unsqueeze(0).unsqueeze(0).type(dtype)
    torch.manual_seed(seeds)
    b = torch.poisson(b_true*noise_level)/noise_level
    b = torch.clamp(b, 0, 10e5)  # sino>0
    imsave(specific_result_dir + 'sino_noise_free.png', b_true[0, 0].cpu().numpy(), cmap=cmap)
    imsave(specific_result_dir + 'sino.png', b[0, 0].cpu().numpy(), cmap=cmap)

    def fn(x):
        return torch.norm(radon.forward(x) - b) ** 2 / 2

    G = skip(1, 1,
            num_channels_down=[8, 16, 32,64,64],
            num_channels_up=[8, 16, 32,64,64],  # [16, 32, 64, 128, 128],
            num_channels_skip=[0,0,0,0,0],
            filter_size_up=3, filter_size_down=3, filter_skip_size=1,
            upsample_mode='bilinear',  # downsample_mode='avg',
            need1x1_up=False,
            need_sigmoid=True,
            need_bias=True, pad='reflection',
            act_fun='ELU').type(dtype)

    #%%
    z = torch.randn((1, 1, image_size, image_size)).type(dtype)

    x = (G(z)*0.5+0.5).clone().detach()
    scaled_lambda_ = torch.zeros_like(x, requires_grad=False).type(dtype)

    # since we use exact minimization over x, we don't need the grad of x
    x.requires_grad = False
    z.requires_grad = False

    opt_z = optim.AdamW(G.parameters(), lr=L, weight_decay=1e-4, amsgrad=True)

    sigma_0 = torch.tensor(sigma_0).type(dtype)
    Gz = G(z)*0.5+0.5

    record = {"psnr_gt": [],
              "ssim_gt": [],
              "mse_gt": [],
              "fidelity_loss": [],
              }

    results = None
    # 迭代训练过程
    for t in range(num_iter):
        # for x
        with torch.no_grad():
            x = tv_prox(Gz.detach() - scaled_lambda_, noise_sigma)

        # for z (GD)
        opt_z.zero_grad()
        Gz = G(z)*0.5+0.5
        AGz = radon.forward(Gz)
        loss_z = torch.norm(b-AGz) ** 2 / 2 + (rho / 2) * torch.norm(x - Gz + scaled_lambda_) ** 2
        loss_z.backward()
        nn.utils.clip_grad_norm_(G.parameters(), max_norm=2.0, norm_type=2)
        opt_z.step()

        # for dual var(lambda)
        with torch.no_grad():
            Gz = (G(z)*0.5+0.5).detach()
            x_Gz = x - Gz
            scaled_lambda_.add_(sigma_0 * x_Gz)

        if results is None:
            results = Gz.detach()
        else:
            results = results * 0.99 + Gz.detach() * 0.01
            # results = Gz.detach()
        psnr_gt = peak_signal_noise_ratio(img_true.cpu().numpy(), results.cpu().numpy())
        ssim_gt = structural_similarity(img_true[0,0].cpu().numpy(), results[0,0].cpu().numpy())
        mse_gt = np.mean((img_true.cpu().numpy() - results.cpu().numpy()) ** 2)
        fidelity_loss = fn(results.cuda()).detach()
        
        if (t + 1) % 1000 == 0:
            imsave(specific_result_dir + '%d_PSNR_%.2f_SSIM_%.2f.png' % (t, psnr_gt,ssim_gt), results[0,0].cpu().numpy(), cmap=cmap)

        record["psnr_gt"].append(psnr_gt)
        record["ssim_gt"].append(ssim_gt)
        record["mse_gt"].append(mse_gt)
        record["fidelity_loss"].append(fidelity_loss.item())

        if (t + 1) % 500 == 0:
            print('Img %d Iteration %5d PSRN_gt: %.2f SSIM: %.2f MSE_gt: %e' % (f_num, t + 1, psnr_gt,ssim_gt, mse_gt))
    np.savez(specific_result_dir + 'record', **record)

#%%
import shutil,glob
dataset_dir = 'data_ct/'
results_dir = 'data_ct/results/ct_reconst_DIP_tv_baseline/'

f_name_list = sorted(glob.glob('../../data_ct/lodopab.png'))
for f_num, f_name in enumerate(f_name_list):
    for ns in [4,2]:
        for L in [0.0005]:
            specific_result_dir = results_dir+str(f_num)+'-tv'+str(ns)+'-lr'+str(L)+'/'
            if os.path.exists(specific_result_dir):
                shutil.rmtree(specific_result_dir)
            os.makedirs(specific_result_dir)
            run(f_name=f_name,
                specific_result_dir=specific_result_dir,
                noise_sigma=ns,
                num_iter=40000,
                rho=1,
                sigma_0=1,
                L=L)
