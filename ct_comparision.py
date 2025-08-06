import matplotlib
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from torch_radon import Radon, RadonFanbeam
from skimage.metrics import peak_signal_noise_ratio

torch.set_grad_enabled(False)
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from skimage.transform import radon
from matplotlib.pyplot import imread, imsave
from torch_radon.solvers import cg, cgne, Landweber
device = torch.device("cuda")
#%% phantom
img = shepp_logan_phantom()

# ct
img = imread('data_ct/lodopab.png')[:,:,0]
print(img.shape,img.max())
if img.dtype == 'uint8':
    img = img.astype('float32') / 255  # scale to [0, 1]
elif img.dtype == 'float32':
    img = img.astype('float64')
else:
    raise TypeError()

image_size = img.shape[0]
torch_x = torch.from_numpy(img).type(torch.float).cuda()

n_angles = 100
seeds = 501
noise_level = 50
cmap = 'jet'       # gray
#%% FBP, Landweber,CGNE, Shearlet
angles = np.linspace(0, np.pi, n_angles, endpoint=False)
det_count = int(np.ceil(np.sqrt(2)*image_size))
radon = Radon(image_size, angles,det_count=det_count)

sino_true = radon.forward(torch_x)
plt.figure()
plt.imshow(sino_true.cpu().numpy(),cmap=cmap)
plt.colorbar()
plt.savefig('sino_noise_free.png',dpi=600)


torch.manual_seed(seeds)
sino = torch.poisson(sino_true*noise_level)/noise_level
sino = torch.clamp(sino,0,10e5)    # sino>0
plt.figure()
plt.imshow(sino.cpu().numpy(),cmap=cmap)
plt.colorbar()
plt.savefig('sino.png',dpi=600)

plt.figure()
plt.imshow((sino-sino_true).abs().cpu().numpy(),cmap=cmap)
plt.colorbar()
plt.savefig('noise.png',dpi=600)

# FBP
filtered_sino = radon.filter_sinogram(sino, "ram-lak")
fbp = radon.backprojection(filtered_sino)

# Landweber
landweber = Landweber(radon)
# start with a solution guess which is all zeros
guess = torch.zeros_like(torch_x)
# estimate the step size using power iteration
alpha = landweber.estimate_alpha(image_size, device)*0.95
landweber_rec = landweber.run(guess, sino, alpha, iterations=100)

# for i in [50,100,200,300]:
#     landweber_rec = landweber.run(guess, sino, alpha,iterations=i)
#     landweber_rec = torch.clamp(landweber_rec,0,1).cpu().numpy()
#     print(f'Landweber-{i}:{peak_signal_noise_ratio(torch_x.cpu().numpy(), landweber_rec)}')
#
#     guess = torch.zeros_like(torch_x)
#     cgne_rec = cgne(radon, guess, sino, max_iter=i)
#     cgne_rec = torch.clamp(cgne_rec, 0, 1).cpu().numpy()
#     print(f'CGNE-{i}:{peak_signal_noise_ratio(torch_x.cpu().numpy(), cgne_rec)}')


# CGNE
guess = torch.zeros_like(torch_x)
# guess = fbp
cgne_rec = cgne(radon, guess, sino, max_iter=30)

fbp = torch.clamp(fbp,0,1).cpu().numpy()
landweber_rec = torch.clamp(landweber_rec,0,1).cpu().numpy()
cgne_rec = torch.clamp(cgne_rec,0,1).cpu().numpy()
torch_x = torch_x.cpu().numpy()
print(f'fbp:{peak_signal_noise_ratio(torch_x,fbp)}')
print(f'Landweber:{peak_signal_noise_ratio(torch_x,landweber_rec)}')
print(f'CGNE:{peak_signal_noise_ratio(torch_x,cgne_rec)}')
#%%
from matplotlib.pyplot import imsave
imgs = [torch_x,fbp,landweber_rec,cgne_rec]
titles = ['x','fbp','landweber_rec','cgne_rec']
fig, axs = plt.subplots(2, 2,figsize=(8,8))
ax = axs.flatten()
for i in range(4):
    ax[i].imshow(imgs[i],cmap=cmap)
    # print(torch_x.shape,imgs[i].shape)
    ax[i].set_title(titles[i]+f': {peak_signal_noise_ratio(torch_x,imgs[i]).round(3)}')
    ax[i].axis('off')

plt.savefig('ct_reconst_other_methods.png',dpi=600)

