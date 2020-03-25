import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--dataset_name", type=str, default="monet2photo", help="name of the dataset")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
opt = parser.parse_args()

os.makedirs("test_images/%s" % opt.dataset_name, exist_ok=True)

criterion_GAN = torch.nn.MSELoss()


input_shape = (opt.channels, opt.img_height, opt.img_width)


G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
D_A = Discriminator(input_shape)

G_BA_checkpoint = torch.load("saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, opt.epoch), map_location={'cuda:0': 'cpu'})
D_A_checkpoint = torch.load("saved_models/%s/D_A_%d.pth" % (opt.dataset_name, opt.epoch), map_location={'cuda:0': 'cpu'})

G_BA.load_state_dict(G_BA_checkpoint)
D_A.load_state_dict(D_A_checkpoint)


G_BA.eval()
D_A.eval()

# Image transformations
transforms_ = [
    transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Test data loader
val_dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True, mode="test"),
    batch_size=5,
    shuffle=True,
    num_workers=1,
)


# ----------
#  Testing
# ----------

for i, batch in enumerate(val_dataloader):
	real_B = Variable(batch["B"].type(torch.Tensor))
	fake_A = G_BA(real_B)

	real_B = make_grid(real_B, nrow=5, normalize=True)
	fake_A = make_grid(fake_A, nrow=5, normalize=True)

	image_grid = torch.cat((real_B, fake_A), 1)
	save_image(image_grid, "test_images/%s/%s.png" % (opt.dataset_name, i), normalize=False)

