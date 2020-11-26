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
from unet import UNet
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import CarvanaDataset
from torch.utils.data import DataLoader, random_split
import poptorch
import time

dir_img = 'data/imgs/'
dir_mask = 'data/masks/'
dir_checkpoint = 'checkpoints/'

# # # Things to add
opts = poptorch.Options()
# # # Device "step"
opts.deviceIterations(1)
# opts.setExecutionStrategy(poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement))

# # How many IPUs to replicate over.
# opts.replicationFactor(4)

# opts.randomSeed(42)
# Distributed execution opts.Distributed.configureProcessId(process_id, num_processes)

# Use pipelined training 
# https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/overview.html#pipeline-annotator
class TrainingModelWithLoss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self._model = model
        self.n_channels=model.n_channels
        self.n_classes=model.n_classes
        self.bilinear=model.bilinear
        if self.n_classes > 1:
            print("Using nn.CrossEntropyLoss()")
            self.loss = nn.CrossEntropyLoss()
        else:
            print("Using nn.BCEWithLogitsLoss()")
            self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, true_masks=None):
        mask_pred = torch.rand(true_masks.size())
        # print(mask_pred)
        # print(mask_pred.type())
        # print(mask_pred.size())
        # print(mask_pred.sum())

        masks_pred = self._model(x)[0]
        # print(masks_pred)
        # print(masks_pred.type())
        # print(masks_pred.size())
        # print(masks_pred.sum())

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

    # Make batch size batch_size * opts.device_iterations
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
    for epoch in range(epochs):
        first = True
        time_t = 0
        imgs_n = 0
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch[0].half()
                true_masks = batch[1].half()
​
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
​
                if(first):
                    masks_pred, loss = net(imgs, true_masks)
                    first = False
                else:
                    tic = time.time()
                    masks_pred, loss = net(imgs, true_masks)
                    time_a = (time.time() - tic)
                    time_t += time_a
                    imgs_n += imgs.size()[0]
                    print("Type of run 2 Tput: " + str(imgs.size()[0]/time_a))

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
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
    net = UNet(n_channels=3, n_classes=1, bilinear=True) # This used to be true but unsupoorted op

    net = TrainingModelWithLoss(net)

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
