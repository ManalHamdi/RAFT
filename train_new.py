from __future__ import print_function, division
import wandb
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from raft import RAFT
import evaluate
import datasets
import core.sequence_handling_utils as seq_utils
from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000
GB = 10**9
def sequence_loss(flow_preds, flow_gt, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

def disimilarity_loss(image_batch, template_batch, flow_predictions, gamma):
    '''
    image1_batch: [B, N, H, W] [B,N,H, W]
    template: [B, N, H, W]
    flow_predictions: list len iters, type tensor [B, N, 2, H, W]
    '''
    partial_loss = 0
    total_iter = len(flow_predictions)
    for itr in range(0, total_iter):
        warped_img_batch = seq_utils.warp_batch(image_batch, flow_predictions[itr]) # [B, N, H, W]
        mse_loss = torch.nn.MSELoss(reduction='mean')
        similarity = mse_loss(warped_img_batch, template_batch) # [B, N]  disimililarity of batch in iteration i
        partial_loss += similarity * gamma ** (total_iter - itr - 1)
        
    return partial_loss / total_iter

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, epochs=args.num_steps, steps_per_epoch=767,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter()

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0
            
    def add_image(self, description, image):
        self.writer.add_image(description, image.view(1, image.shape[0], image.shape[1]), dataformats='CHW')
        
    def add_scalar(self, description, x, y):
        self.writer.add_scalar(description, x, y)
        
    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):
    wandb.init(project="test-project", entity="manalteam")
    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.to("cuda:0")
    model.train()

    if args.stage != 'chairs':
        model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    VAL_FREQ = 10 #5000
    add_noise = False

    should_keep_training = True
    while should_keep_training:
        loss_epoch = 0
        for i_batch, data_blob in enumerate(train_loader):
            #
            
            optimizer.zero_grad()
            image_batch, template_batch = [x.to("cuda:0") for x in data_blob] # old [B, C, H, W] new [B, N, H, W], [B, N, H, W]
            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image_batch = (image_batch + stdv * torch.randn(*image_batch.shape).cuda()).clamp(0.0, 255.0)
                template_batch = (template_batch + stdv * torch.randn(*template_batch.shape).cuda()).clamp(0.0, 255.0)

            flow_predictions = model(image_batch, template_batch, iters=args.iters)
           
            '''
            if (i_batch % 300 == 0):
                warped_img_batch = seq_utils.warp_batch(image_batch, flow_predictions[len(flow_predictions)-1]) # [B, N, H, W]
                wandb.log({"Training original image from epoch"+str(total_steps): wandb.Image(image_batch[0,3,:,:]), 
                           "Training template from epoch"+str(total_steps): wandb.Image(template_batch[0,3,:,:]),
                           "Training warped image from epoch"+str(total_steps): wandb.Image(warped_img_batch[0,3,:,:])})
            '''
            # list of flow estimations with length iters, and each item of the list is [B, 2, H, W]   new [B, N, 2, H, W]  
            batch_loss = disimilarity_loss(image_batch, template_batch, flow_predictions, args.gamma) # -- loss in batch
            
            #loss, metrics = sequence_loss(flow_predictions, flow_predictions, args.gamma)
            scaler.scale(batch_loss).backward()
           
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            loss_epoch += batch_loss.item() / len(train_loader)
            
        wandb.log({"Training Loss": loss_epoch})
        print("Epoch:", total_steps, "training loss", loss_epoch)
        # VALIDATION
        results = {}
        for val_dataset in args.validation:
            if val_dataset == 'chairs':
                results.update(evaluate.validate_chairs(model.module))
            elif val_dataset == 'sintel':
                results.update(evaluate.validate_sintel(model.module))
            elif val_dataset == 'kitti':
                results.update(evaluate.validate_kitti(model.module))
            elif val_dataset == 'acdc':
                results.update(evaluate.validate_acdc(model.module, args))
            wandb.log({"Validation Loss": results['acdc']})

            # Log every epoch
            #if total_steps % VAL_FREQ == VAL_FREQ - 1: # log after a number of epochs
            PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
            torch.save(model.module.state_dict(), PATH)

            #logger.write_dict(results)
            model.train()
            if args.stage != 'chairs':
                model.module.freeze_bn()
    
            total_steps += 1 # Num of epochs

            if total_steps > args.num_steps:
                should_keep_training = False
                break
        

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--dataset_folder', type=str)
    parser.add_argument('--max_seq_len', type=int, default=35)
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)