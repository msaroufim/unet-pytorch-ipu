import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from eval import eval_net
# from unet import UNet
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import CarvanaDataset
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

import poptorch
import time

dir_img = 'data/imgs/'
dir_mask = 'data/masks/'
dir_checkpoint = 'checkpoints/'

poptorch.setLogLevel(1)
opts = poptorch.Options()
opts.deviceIterations(4)
opts.replicationFactor(1)
n_ipus = 4
mem_prop = {f'IPU{i}': 0.1 for i in range(n_ipus)}
opts.setAvailableMemoryProportion(mem_prop)
opts.setExecutionStrategy(poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement))

# # How many IPUs to replicate over.


# opts.randomSeed(42)
# Distributed execution opts.Distributed.configureProcessId(process_id, num_processes)

# Use pipelined training 
# https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/overview.html#pipeline-annotator


class DoubleConv2(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='nearest', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class TrainingModelWithLoss(torch.nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        if self.n_classes > 1:
            print("Using nn.CrossEntropyLoss()")
            self.loss = nn.CrossEntropyLoss()
        else:
            print("Using nn.BCEWithLogitsLoss()")
            self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, true_masks=None):
        poptorch.Block.useAutoId()
        #Actual forward propagation code
        with poptorch.Block(ipu_id=0):
            # Fix to deal with Region bug
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)

        with poptorch.Block(ipu_id=1):
            x5 = self.down4(x4)

        with poptorch.Block(ipu_id=2):
            x = self.up1(x5, x4)

        with poptorch.Block(ipu_id=3):
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
            mask_pred = torch.rand(true_masks.size())
            print(mask_pred)
            print(mask_pred.type())
            print(mask_pred.size())
            print(mask_pred.sum())
            masks_pred = logits

            if true_masks is not None:
                return masks_pred, self.loss(mask_pred, true_masks)
            return masks_pred

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              scaleX=0.5,
              scaleY=0.5):

    dataset = CarvanaDataset(dir_img, dir_mask, scaleX, scaleY)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALEX_{scaleX}_SCALEY_{scaleY}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling X: {scaleX}
        Images scaling Y: {scaleY}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)

    tic = time.time()
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                print(f"imgs batch shape {imgs.shape}")
                true_masks = batch['mask']
                print(f"True masks batch shape {true_masks.shape}")

                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 # if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                masks_pred, loss = net(imgs, true_masks)

                print(masks_pred)
                print(loss)
                # print(a)
                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        # writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        # writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    # val_score = eval_net(net, val_loader, device)
                    # scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        print("classes more than 1")
                        # logging.info('Validation cross entropy: {}'.format(val_score))
                        # writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        print("1 class")
                        # logging.info('Validation Dice Coeff: {}'.format(val_score))
                        # writer.add_scalar('Dice/test', val_score, global_step)

                    # writer.add_images('images', imgs, global_step)
                    # if net.n_classes == 1:
                    #     writer.add_images('masks/true', true_masks, global_step)
                    #     writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    toc = time.time()
    duration = (toc - tic) 
    logging.info(f'Training time {duration}')
    throughput = (n_train * epochs) / duration
    logging.info(f'Throughput {throughput} im/sec')
    writer.close()

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-x', '--scaleX', dest='scaleX', type=float, default=0.5,
                        help='Downscaling factor of the images X')
    parser.add_argument('-y', '--scaleY', dest='scaleY', type=float, default=0.5,
                        help='Downscaling factor of the images Y')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=-0.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    # net = UNet(n_channels=3, n_classes=1, bilinear=False) # This used to be true but unsupoorted op

    net = TrainingModelWithLoss(n_channels=3, n_classes=1, bilinear=False)

    # logging.info(f'Network:\n'
    #             f'\t{net.n_channels} input channels\n'
    #             f'\t{net.n_classes} output channels (classes)\n'
    #             f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling'
    net.half()
    net = poptorch.trainingModel(net,opts)

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    # net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  scaleX=args.scaleX,
                  scaleY=args.scaleY)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)