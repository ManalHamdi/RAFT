import losses as Losses
import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
import torch.nn as nn
import datasets
import flow_vis
from utils import flow_viz
from utils import frame_utils
import core.sequence_handling_utils as seq_utils

from raft import RAFT
from utils.utils import InputPadder, forward_interpolate


@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results

@torch.no_grad()
def validate_acdc(model, args, mode, epoch, iters=2):
    ''' Perform validation using ACDC processed dataset '''
    cuda_to_use = "cuda:" + str(args.gpus[0])
    model.eval()
    val_dataset = datasets.ACDCDataset(args.dataset_folder, mode, args.max_seq_len, args.add_normalisation)

    out_list = []
    total_loss, total_error, total_spa_loss, total_temp_loss = 0, 0, 0, 0
    for val_id in range(0, len(val_dataset)):
        image_batch, template_batch, patient_slice_id_batch = val_dataset[val_id]
        image_batch = image_batch[None].to(cuda_to_use)
        template_batch = template_batch[None].to(cuda_to_use)
        flow_predictions1, flow_predictions2 = model(image_batch, template_batch, iters=iters, test_mode=True) #[B, 2, H, W]
        batch_loss_dict = Losses.disimilarity_loss(image_batch, template_batch, [patient_slice_id_batch],
                                           flow_predictions1, flow_predictions2, 
                                           epoch=epoch, mode="validation", 
                                           i_batch=2, args=args) # -- loss in batch

        total_loss += batch_loss_dict["Total"].item() / len(val_dataset)
        total_error += batch_loss_dict["Error"].item() / len(val_dataset)
    val_dict = {"Total": total_loss,
                "Error": total_error}
    print("Validation ACDC: %f" % (total_loss))
    print("Validation Error ACDC: %f" % (total_error))
    return val_dict

@torch.no_grad()
def log_gifs_test(image_batch, template_batch, temp_pred, flow_pred_fwd, flow_pred_bwd, patient_name, add_normalisation):
    total_iter = len(flow_pred_fwd)
    b, s, _, h, w = flow_pred_fwd[0].shape
    iters = 2
    mode = 'testing'
    # img pred consec
    img_consec = seq_utils.warp_batch(temp_pred[:,0:s-1,:,:], flow_pred_bwd[iters-1][:,1:s,:,:,:], gpu=args.gpus[0])
    i1 = img_consec.permute(1, 0, 2, 3).cpu().detach().numpy()[::2,:,:,:]
    # error consec
    error = torch.abs(img_consec - image_batch[:,1:s,:,:])
    i2 = error.permute(1, 0, 2, 3).cpu().detach().numpy()[::2,:,:,:]
    if (add_normalisation):
        scale = 255
    else:
        scale = 1
    
    wandb.log({"GIF " + mode + " " + patient_name + " ": [
                                                    wandb.Video(i1*scale, fps=2, caption="Consecutive Image Pred" , format="gif"),
                                                    wandb.Video(i2*scale, fps=2, caption="Consecutive Error" , format="gif")]})
@torch.no_grad()
def log_additional_flow(flow_pred_fwd, flow_pred_bwd, patient_name):
    mode = 'testing'
    iters = len(flow_pred_fwd)
    flow_1_2 = flow_pred_fwd[iters-1][0,1,:,:,:] + flow_pred_bwd[iters-1][0,2,:,:,:]
    flow_1_6 = flow_pred_fwd[iters-1][0,1,:,:,:] + flow_pred_bwd[iters-1][0,6,:,:,:]
    
    flow_for = flow_vis.flow_to_color(flow_pred_fwd[iters-1][0,2,:,:,:].permute(1,2,0).cpu().detach().numpy())

    print("mine, orig", flow_1_2.shape, flow_pred_fwd[iters-1][0,2,:,:,:].shape)
    
    print("Shape all", flow_for.shape)
    flow_1_2 = flow_vis.flow_to_color(flow_1_2.permute(1,2,0).cpu().detach().numpy())
    flow_1_6 = flow_vis.flow_to_color(flow_1_6.permute(1,2,0).cpu().detach().numpy())
    wandb.log({patient_name + " "+ mode + " Images ": [wandb.Image(flow_1_2, caption=patient_name + " Flow 1->2"),
                                                      wandb.Image(flow_1_6, caption=patient_name + " Flow 1->6")]})

@torch.no_grad()
def test_acdc(args):
    print("We are in!")
    wandb.init(project="test-project", entity="manalteam")
    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    model.load_state_dict(torch.load(args.restore_ckpt, map_location=torch.device('cpu')), strict=False)
    cuda_to_use = "cuda:" + str(args.gpus[0])
    model.to(cuda_to_use)
    model.eval()
    mode = 'testing'
    iters = 2

    test_dataset = datasets.ACDCDataset(args.dataset_folder, 'testing', args.max_seq_len, args.add_normalisation)
    
    all_seq_error = 0
    for test_id in range(0, len(test_dataset)):
        print(test_id)
        image_batch, template_batch, patient_slice_id_batch = test_dataset[test_id] # [B, S, H, W]
        image_batch = image_batch[None].to(cuda_to_use)
        template_batch = template_batch[None].to(cuda_to_use)
        flow_pred_fwd, flow_pred_bwd = model(image_batch, template_batch, iters=iters, test_mode=True) #[B, 2, H, W]

        # Here we have sequences, it should become all combination of pairs
        _, len_s, h, w = image_batch.shape
        
        template_prime = seq_utils.warp_batch(image_batch, flow_pred_fwd[iters-1], gpu=args.gpus[0])
        img_pred = seq_utils.warp_batch(template_prime, flow_pred_bwd[iters-1], gpu=args.gpus[0])
        #if (patient_slice_id_batch == ''):
        Losses.log_images(image_batch[0,2,:,:], img_pred[0,2,:,:], template_batch[0,2,:,:], template_prime[0,2,:,:], 
                       flow_pred_fwd, flow_pred_bwd, 2, patient_slice_id_batch, mode)
        Losses.log_gifs(image_batch, img_pred, 
                         template_batch, template_prime, 
                         flow_pred_fwd, flow_pred_bwd, 
                         patient_slice_id_batch, mode, args.add_normalisation)
        log_gifs_test(image_batch, template_batch, template_prime, 
                          flow_pred_fwd, flow_pred_bwd, patient_slice_id_batch, args.add_normalisation)
        log_additional_flow(flow_pred_fwd, flow_pred_bwd, patient_slice_id_batch)
        
        
        error_seq = 0
        for i in range(0, len_s):
            # Ii->n to all images
            template_i_prime = template_prime[0,i,:,:].repeat(len_s,1,1).repeat(1,1,1,1) # [H, W] --> [B, S, H, W]
            image_prime = seq_utils.warp_batch(template_i_prime, flow_pred_bwd[iters-1], gpu=args.gpus[0])
            l1_loss = torch.nn.L1Loss()
            error_seq += l1_loss(image_prime, image_batch)
                
        error_seq /= len_s
        print("Error of seq ", patient_slice_id_batch, "made of all pairs", error_seq)
        all_seq_error += error_seq
    all_seq_error /= len_s     
    print("Error of All sequences is ", all_seq_error)
        
        

@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--dataset_folder', type=str)
    parser.add_argument('--max_seq_len', type=int, default=35)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--add_normalisation', action='store_true')

    args = parser.parse_args()
    '''
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()
    '''
    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)

        elif args.dataset == 'sintel':
            validate_sintel(model.module)

        elif args.dataset == 'kitti':
            validate_kitti(model.module)
        elif args.dataset == 'acdc':
            test_acdc(args)

