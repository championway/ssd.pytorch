import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import cv2

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from pix2pix_models import *
from pix2pix_datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from ssd import build_ssd

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="mnist", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=50, help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=100, help='interval between model checkpoints')
opt = parser.parse_args()
print(opt)

os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()
criterion_bbx = torch.nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height//2**4, opt.img_width//2**4)

# Initialize generator and discriminator
generator = GeneratorUNet(in_channels=1, out_channels=1)
discriminator = Discriminator(in_channels=1)

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()
    criterion_bbx.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load('saved_models/%s/generator_%d.pth' % (opt.dataset_name, opt.epoch)))
    discriminator.load_state_dict(torch.load('saved_models/%s/discriminator_%d.pth' % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders
transforms_A = [ transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
transforms_B = [ transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                transforms.ToTensor() ]

inv_normalize = transforms.Compose([ 
                transforms.Normalize(mean = [ 0., 0., 0. ],
                                             std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                             std = [ 1., 1., 1. ]),
                ])

dataloader = DataLoader(ImageDataset(transform_=transforms_A),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

val_dataloader = DataLoader(ImageDataset(transform_=transforms_A, mode='test'),
                            batch_size=1, shuffle=True, num_workers=1)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs, bbx = next(iter(val_dataloader))
    real = Variable(imgs.type(Tensor))
    fake = generator(real)
    img_sample = torch.cat((real.data, fake.data), -1)
    save_image(img_sample, 'images/%s/%s.png' % (opt.dataset_name, batches_done), nrow=5, normalize=True)

net = build_ssd('train', 300, 2)
net.load_weights('/home/arg_ws3/ssd.pytorch/weights/subt_real/subt_real_290.pth')
if torch.cuda.is_available():
    net = net.cuda()
def predtictSSD(img):
    # Preprocessing
    image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    image = image/10.
    x = cv2.resize(image, (300, 300)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)

    #SSD Forward Pass
    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()
    y = net(xx)
    scale = torch.Tensor(image.shape[1::-1]).repeat(2)
    detections = y.data # torch.Size([1, 4, 200, 5]) --> [batch?, class, object, coordinates]
    objs = []
    predict_score = 0
    pt = torch.tensor([0, 0, 0, 0])
    for i in range(detections.size(1)): # detections.size(1) --> class size
        for j in range(5):  # each class choose top 5 predictions
            if detections[0, i, j, 0].numpy() > predict_score and detections[0, i, j, 0].numpy() > 0.5:
                predict_score = detections[0, i, j, 0].numpy()
                score = detections[0, i, j, 0]
                pt = (detections[0, i, j,1:]).cpu()
    print(pt)
    return pt.cuda()

# ----------
#  Training
# ----------

prev_time = time.time()

dataloader_mnist = torch.utils.data.DataLoader(
    datasets.MNIST('../../data/mnist', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])),
    batch_size=opt.batch_size, shuffle=True)
imgs = next(iter(dataloader))
for epoch in range(opt.epoch, opt.n_epochs):
    for i, (img, bbx) in enumerate(dataloader):
        # Model inputs
        real_src_img = Variable(img.type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_src_img.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_src_img.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        fake_trg_img = generator(real_src_img)
        pred_trg_img = fake_trg_img.cpu().squeeze(0).detach()
        pred_trg_img = inv_normalize(pred_trg_img)
        pred_trg_img = pred_trg_img.mul(10000).clamp(0, 10000).to(torch.int16).permute(1, 2, 0)
        pred_trg_img = np.array(pred_trg_img)
        pred_trg_img = pred_trg_img.astype(np.uint16)
        #print("-----", pred_trg_img[120][120:150])
        #print("==")
        #pred_trg_img = pred_trg_img[...,::-1]
        #bbx = torch.tensor(bbx)
        bbx = torch.tensor(bbx)
        bbx = torch.DoubleTensor(bbx).cuda()
        pred_bbx = predtictSSD(pred_trg_img).view(-1, 4).double()
        loss_bbx = criterion_bbx(bbx, pred_bbx).float()
        pred_fake = discriminator(fake_trg_img, real_src_img)
        loss_GAN = criterion_GAN(pred_fake, valid)
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_trg_img, real_src_img)
        #print(loss_pixel, loss_GAN, loss_bbx)

        # Total loss
        loss_G = loss_GAN*0.8 + lambda_pixel * loss_pixel * 0.8 + loss_bbx*100
        #loss_G = loss_GAN

        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(fake_trg_img.detach(), real_src_img)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(fake_trg_img.detach(), real_src_img)
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s" %
                                                        (epoch, opt.n_epochs,
                                                        i, len(dataloader),
                                                        loss_D.item(), loss_G.item(),
                                                        loss_pixel.item(), loss_GAN.item(),
                                                        time_left))

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            #save_image(fake_trg_img.data[:1], 'images/%d.png' % batches_done, nrow=5, normalize=True)
            sample_images(batches_done)


    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), 'saved_models/%s/generator_%d.pth' % (opt.dataset_name, epoch))
        torch.save(discriminator.state_dict(), 'saved_models/%s/discriminator_%d.pth' % (opt.dataset_name, epoch))
torch.save(generator.state_dict(), 'saved_models/%s/generator.pth' % (opt.dataset_name))
torch.save(discriminator.state_dict(), 'saved_models/%s/discriminator.pth' % (opt.dataset_name))