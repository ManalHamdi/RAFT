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
import yaml
import experiment

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
    val_dataset = datasets.ACDCDataset(args.dataset_folder, mode, args.max_seq_len, args.model, args.add_normalisation)

    out_list = []
    total_loss, img_total_error, tmp_total_error, total_spa_loss, total_temp_loss = 0, 0, 0, 0, 0
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
        img_total_error += batch_loss_dict["Img Error"] / len(val_dataset)
        tmp_total_error += batch_loss_dict["Temp Error"].item() / len(val_dataset)
    val_dict = {"Total": total_loss,
                "Img Error": img_total_error,
                "Tmp Error": tmp_total_error}
    print("Validation ACDC: %f" % (total_loss))
    print("Validation Img Error ACDC: %f" % (img_total_error))
    print("Validation Tmp Error ACDC: %f" % (tmp_total_error))
    return val_dict

@torch.no_grad()
def log_gifs_test(image_batch, template_batch, temp_pred, flow_pred_fwd, flow_pred_bwd, patient_name, args):
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
    if (args.add_normalisation):
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
def log_test(im1, im2, im2_p, err, flow, patient_name, img1_idx, img2_idx):
    ''' Logging images, not GIF 
    '''
    flow = flow_vis.flow_to_color(flow.permute(1,2,0).cpu().detach().numpy())
    
    log_name = "Consecutive Frames " + patient_name + " from "+ str(img1_idx) + " to " + str(img2_idx)
    
    wandb.log({log_name: [wandb.Image(im1, caption=patient_name + " Image"+str(img1_idx)), 
                         wandb.Image(im2, caption=patient_name + " Image"+str(img2_idx)),
                         wandb.Image(im2_p, caption=patient_name + " Image"+str(img2_idx)+" predicted"),
                         wandb.Image(err, caption=patient_name + " Error"+str(img2_idx)+" "+str(img2_idx)+"'"),
                         wandb.Image(flow, caption=patient_name + " Flow"+str(img1_idx)+"->"+str(img2_idx))]})
    
@torch.no_grad()
def compute_avg_pair_error_pair(model, test_dataset, args):
    avg_pair_err, pair_count = 0, 0
    patient_log = False
    
    idx_log_1 = [0, 6, 13]
    l1_loss = torch.nn.L1Loss()
    
    for seq_id in range(0, len(test_dataset)):
        seq_original, _, patient_name = test_dataset[seq_id]
        if (patient_name == 'patient113_z_3' or patient_name == 'patient102_z_1' or patient_name == 'patient123_z_3'):
            patient_log = True
        cuda_to_use = "cuda:" + str(args.gpus[0])
        seq_len, h, w = seq_original.shape
        
        for frame_id in range(0, seq_len): # Flow frame_id -> i:  1-1 1-2 1-3 / 2-1 2-2 2-3 /
            '''
            if (frame_id % 5 != 0):
                continue
            '''
            seq1 = seq_original[frame_id,:,:].repeat(seq_len, 1, 1)
            flow_pred_fwd, _ = model(seq1[None].to(cuda_to_use), seq_original[None].to(cuda_to_use), 
                                     iters=args.iters, test_mode=True)
            seq_pred = seq_utils.warp_batch(seq1[None].to(cuda_to_use), flow_pred_fwd[args.iters-1], gpu=args.gpus[0])
            for i in range(0, seq_len):
                '''
                if (i != frame_id):
                    continue
                '''
                l1_loss_none = torch.nn.L1Loss(reduction='none')
                err = l1_loss_none(seq_original[i,:,:].to(cuda_to_use), seq_pred[0,i,:,:])
                avg_pair_err += l1_loss(seq_original[i,:,:].to(cuda_to_use), seq_pred[0,i,:,:])
                pair_count += 1
                
                can_log = frame_id in idx_log_1 and patient_log
                # Flow 0 -> 1 Flow 6 -> 7 Flow 13 -> 14
                log_consc = can_log and i == frame_id+1
                # Flow 0 -> 4 Flow 6 -> 14 Flow 13 -> 5
                log_none_consc = can_log and ((frame_id == 0 and i == 4) 
                                           or (frame_id == 6 and i == 14) 
                                           or (frame_id == 13 and i == 5))
                if (log_consc or log_none_consc): # Log consecutive or none consec
                    log_test(seq1[i,:,:], seq_original[i,:,:], seq_pred[0, i,:,:], 
                             err, flow_pred_fwd[args.iters-1][0,i,:,:,:], patient_name, frame_id, i) # Flow frame_id -> i
                                    
        patient_log = False
    avg_pair_err /= pair_count
    return avg_pair_err

@torch.no_grad()
def compute_avg_pair_error_group(model, test_dataset, args):
    avg_pair_err, pair_count = 0, 0
    l1_loss = torch.nn.L1Loss()
    patient_log = False
    idx_log_1 = [0, 6, 13]
    cuda_to_use = "cuda:" + str(args.gpus[0])
    for seq_id in range(0, len(test_dataset)):
        seq, tmp_seq, patient_name = test_dataset[seq_id]
        seq_len, h, w = seq.shape
        flow_pred_fwd, flow_pred_bwd = model(seq[None].to(cuda_to_use), tmp_seq[None].to(cuda_to_use), 
                                             iters=args.iters, test_mode=True)
        if (patient_name == 'patient113_z_3' or patient_name == 'patient102_z_1' or patient_name == 'patient123_z_3'):
            patient_log = True
        
        for im1_id in range(0, seq_len):
            '''
            if (im1_id % 5 != 0):
                continue
            '''
            # Construct a seq with frame im1_id repeated
            im1 = seq[im1_id,:,:][None].to(cuda_to_use)
            for im2_id in range(0, seq_len): # Flow im1_id -> im2_id
                '''
                if (im1_id != im2_id):
                    continue
                '''
                flow1_tmp = flow_pred_fwd[args.iters-1][0,im1_id,:,:,:]
                temp_p = seq_utils.warp_seq(im1, flow1_tmp.repeat(1,1,1,1), gpu=args.gpus[0]) 
                
                im2 = seq[im2_id,:,:][None].to(cuda_to_use)
                flow2 = flow_pred_bwd[args.iters-1][0,im2_id,:,:,:][None]
                #im2_p = seq_utils.warp_seq(temp_p, flow2, gpu=args.gpus[0])
                ''' seq_utils.warp_seq(IM1, flow1_tmp+flow2) CHECK IF SAME'''
                im2_p = seq_utils.warp_seq(im1, flow1_tmp[None]+flow2)
                #assert torch.equal(seq_utils.warp_seq(im1, flow1_tmp[None]+flow2), im2_p)
                
                l1_loss_none = torch.nn.L1Loss(reduction='none')
                err = l1_loss_none(im2, im2_p)
                
                avg_pair_err += l1_loss(im2, im2_p)
                pair_count += 1
                
                can_log = im1_id in idx_log_1 and patient_log
                log_consc = can_log and im2_id == im1_id+1
                log_none_consc = can_log and ((im1_id == 0 and im2_id == 4)
                                              or (im1_id == 6 and im2_id == 14)
                                              or (im1_id == 13 and im2_id == 5))
                if (log_consc or log_none_consc):
                    flow1_2 = flow1_tmp+flow2
                    log_test(im1, im2, im2_p, err, flow1_2[0,:,:,:], patient_name, im1_id, im2_id)
                    
        patient_log = False
    avg_pair_err /= pair_count
    return avg_pair_err

@torch.no_grad()
def evaluate_acdc_test(args):
    f = open(args.output_file, "a")
    f.write(args.name + "\n")

    wandb.init(project="test-project", entity="manalteam")
    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    
    if args.restore_ckpt is not None:
        f.write("Loading Checkpoint: " + str(args.restore_ckpt) + "\n")
        ckpt = torch.load(args.restore_ckpt)
        model.module.load_state_dict(ckpt['model'], strict=True)
    
    model.eval()
    mode = 'testing'
    cuda_to_use = "cuda:" + str(args.gpus[0])
    
    test_dataset = datasets.ACDCDataset(args.dataset_folder, 'testing', args.max_seq_len, args.model, args.add_normalisation)
    assert args.model == 'group' or args.model == 'pair'
    if (args.model == 'pair'):
        avg_pair_err = compute_avg_pair_error_pair(model, test_dataset, args)
    elif(args.model == 'group'):
        avg_pair_err = compute_avg_pair_error_group(model, test_dataset, args)

    wandb.log({"Average pair error": avg_pair_err})
            
@torch.no_grad()
def test_acdc(args):
    f = open(args.output_file, "a")
    f.write(args.name + "\n")

    wandb.init(project="test-project", entity="manalteam")
    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    if args.restore_ckpt is not None:
        f.write("Loading Checkpoint: " + str(args.restore_ckpt) + "\n")
        ckpt = torch.load(args.restore_ckpt)
        epoch = ckpt['epoch'] + 1
        model.module.load_state_dict(ckpt['model'], strict=True)
    
    model.eval()
    mode = 'testing'
    cuda_to_use = "cuda:" + str(args.gpus[0])
    iters = 2

    test_dataset = datasets.ACDCDataset(args.dataset_folder, 'testing', args.max_seq_len, args.model, args.add_normalisation)
    
    all_seq_error = 0
    all_pair_error = 0
    count_pair = 0
    Img_err = 0
        
    for test_id in range(0, len(test_dataset)):
        image_batch, template_batch, patient_slice_id_batch = test_dataset[test_id] # [B, S, H, W]
        image_batch = image_batch[None].to(cuda_to_use)
        template_batch = template_batch[None].to(cuda_to_use)
        flow_pred_fwd, flow_pred_bwd = model(image_batch, template_batch, iters=iters, test_mode=True) #[B, 2, H, W]
        
        _, len_s, h, w = image_batch.shape
        
        template_prime = seq_utils.warp_batch(image_batch, flow_pred_fwd[iters-1], gpu=args.gpus[0])
        img_pred = seq_utils.warp_batch(template_batch, flow_pred_bwd[iters-1], gpu=args.gpus[0])
        img_prime = seq_utils.warp_batch(template_prime, flow_pred_bwd[iters-1], gpu=args.gpus[0])
        
        l1_loss = torch.nn.L1Loss()
        Img_err += l1_loss(image_batch, img_prime)
        
        if (test_id % 5 == 0):
            '''LOGGING'''
            Losses.log_images(image_batch[0,3,:,:], img_pred[0,3,:,:], template_batch[0,3,:,:], template_prime[0,3,:,:], 
                           flow_pred_fwd, flow_pred_bwd, 3, patient_slice_id_batch, mode)
            wandb.log({patient_slice_id_batch + " "+ mode + " Images ": 
                                           [wandb.Image(img_prime[0,3,:,:], caption=patient_slice_id_batch + " Image Prime")]})
            Losses.log_gifs(image_batch, img_pred, 
                             template_batch, template_prime, 
                             flow_pred_fwd, flow_pred_bwd, 
                             patient_slice_id_batch, mode, args.add_normalisation)
            i2 = img_prime.permute(1, 0, 2, 3).cpu().detach().numpy()[::2,:,:,:]
            wandb.log({"GIF " + mode + " " + patient_slice_id_batch + " ": 
                       [wandb.Video(i2, fps=2, caption="Image Prime" , format="gif")]})

            log_gifs_test(image_batch, template_batch, template_prime, 
                              flow_pred_fwd, flow_pred_bwd, patient_slice_id_batch, args)
            #log_additional_flow(flow_pred_fwd, flow_pred_bwd, patient_slice_id_batch)

        this_seq_pair_err = 0
        
        this_seq_count_pair = 0
        for i in range(0, len_s):
            # img i --> temp' i --> image_prime (all)
            template_i_prime = template_prime[0,i,:,:].repeat(len_s,1,1).repeat(1,1,1,1) # [H, W] --> [B, S, H, W]
            image_prime = seq_utils.warp_batch(template_i_prime, flow_pred_bwd[iters-1], gpu=args.gpus[0])
            for s in range(0, len_s):
                all_pair_error += l1_loss(image_prime[0,s,:,:], image_batch[0,s,:,:])
                this_seq_pair_err += l1_loss(image_prime[0,s,:,:], image_batch[0,s,:,:])
                count_pair += 1
                this_seq_count_pair += 1
        
        assert this_seq_count_pair == len_s * len_s
        
        this_seq_pair_err /= (len_s * len_s)
        all_seq_error += this_seq_pair_err
        f.write("Average Error of seq for "+str(patient_slice_id_batch)+" made of all pairs "+ str(this_seq_pair_err.item())+"\n")
    all_seq_error /= len(test_dataset)
    all_pair_error /= count_pair
    Img_err /= len(test_dataset)
    wandb.log({"Average pair error per sequence": all_seq_error})
    wandb.log({"Average pair error": all_pair_error})
    wandb.log({"Average sequence error": Img_err})
    
    f.write("---------------------------------------------\n")
    f.write("Average pair error per sequence "+str(all_seq_error.item())+ "\n")
    f.write("Average pair error "+str(all_pair_error.item()) + "\n")
    f.write("Average sequence error "+str(Img_err.item()) + "\n")
    f.write("*********************************************\n")
    f.write("*********************************************\n")
    f.close
        

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
    with open("config_eval.yml", "r") as stream:
        try:
            d = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='EvalGroupwiseFull', help="name of experiment from config.yml")
    args = parser.parse_args()
    config = experiment.Experiment(d[args.experiment]) 
    
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help="name of experiment being evaluated")
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
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--model', type=str, default='group')

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()
    '''

    with torch.no_grad():
        if config.dataset == 'chairs':
            validate_chairs(model.module)

        elif config.dataset == 'sintel':
            validate_sintel(model.module)

        elif config.dataset == 'kitti':
            validate_kitti(model.module)
        elif config.dataset == 'acdc':
            #test_acdc(config)
            evaluate_acdc_test(config)

