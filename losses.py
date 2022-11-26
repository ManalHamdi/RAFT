import torch
import sys
sys.path.append('core')
import core.sequence_handling_utils as seq_utils
import wandb
import flow_vis
import numpy as np 

import torch.nn.functional as F

def temporal_grad_central(data):
    data = data.permute(3, 1, 2, 0)
    padding_t = (1, 1, 0, 0)
    pad_t = F.pad(data, padding_t, mode='replicate')
    grad_t = pad_t[..., 2:] - 2 * data + pad_t[..., :-2]
    grad_t = grad_t.permute(3, 1, 2, 0)
    return grad_t

def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy

def log_images(img_gt, img_pred, temp_gt, temp_pred, flow_forward, flow_backward, index_slice, patient_name, mode):
    total_iter = len(flow_forward)
    flow_for = flow_vis.flow_to_color(flow_forward[total_iter-1][0,index_slice,:,:,:].permute(1,2,0).cpu().detach().numpy())
    
    if (img_pred is not None and flow_backward is not None):
        error = torch.abs(img_pred - img_gt)
        flow_back = flow_vis.flow_to_color(flow_backward[total_iter-1][0,index_slice,:,:,:].permute(1,2,0).cpu().detach().numpy())
        wandb.log({patient_name + " "+ mode + " Images ": [wandb.Image(img_gt, caption=patient_name + " Image GT"), 
                                                      wandb.Image(img_pred, caption=patient_name + " Image Pred"),
                                                      wandb.Image(error, caption=patient_name + " Img Error"),
                                                      wandb.Image(temp_gt, caption=patient_name + " Template GT"),
                                                      wandb.Image(temp_pred, caption=patient_name + " Template Pred"),
                                                      wandb.Image(flow_for, caption=patient_name + " Forward Flow"),
                                                      wandb.Image(flow_back, caption=patient_name + " BackwardFlow")]})

    
    else:
        wandb.log({patient_name + " "+ mode + " Images ": [wandb.Image(img_gt, caption=patient_name + " Image GT"), 
                                                      wandb.Image(temp_gt, caption=patient_name + " Template GT"),
                                                      wandb.Image(temp_pred, caption=patient_name + " Template Pred"),
                                                      wandb.Image(flow_for, caption=patient_name + " Forward Flow")]})

def log_gifs(img_gt, img_pred, temp_gt, temp_pred, flow_forward, flow_backward, patient_name, mode, add_normalisation):
    total_iter = len(flow_forward)
    b, s, _, h, w = flow_forward[0].shape
    
    i1 = img_gt.permute(1, 0, 2, 3).cpu().detach().numpy()[::2,:,:,:] #s, c, h, w
    i3 = temp_gt.permute(1, 0, 2, 3).cpu().detach().numpy()[::2,:,:,:]
    i4 = temp_pred.permute(1, 0, 2, 3).cpu().detach().numpy()[::2,:,:,:]
    all_flows_fwd = np.empty([s, 3, h, w])
    if (img_pred is not None and flow_backward is not None):
        i2 = img_pred.permute(1, 0, 2, 3).cpu().detach().numpy()[::2,:,:,:]
        all_flows_bwd = np.empty([s, 3, h, w])
    
    for frame_idx in range(0, s):
        flow_for = flow_vis.flow_to_color(flow_forward[total_iter-1][0,frame_idx,:,:,:].permute(1, 2, 0).cpu().detach().numpy())
        all_flows_fwd[frame_idx,:,:,:] =  np.transpose(flow_for, (2,0,1))
        if (img_pred is not None and flow_backward is not None):
            flow_bwd = flow_vis.flow_to_color(flow_backward[total_iter-1][0,frame_idx,:,:,:].permute(1, 2, 0).cpu().detach().numpy())
            all_flows_bwd[frame_idx,:,:,:] =  np.transpose(flow_bwd, (2,0,1))
        frame_idx += 1
        
    i5 = all_flows_fwd
    if (img_pred is not None and flow_backward is not None):
        i6 = all_flows_bwd
        error = torch.abs(img_pred - img_gt)
        i7 = error.permute(1, 0, 2, 3).cpu().detach().numpy()[::2,:,:,:]

    if (add_normalisation):
        scale = 1
    else:
        scale = 1
    
    if (img_pred is not None and flow_backward is not None):
        wandb.log({"GIF " + mode + " " + patient_name + " ": [wandb.Video(i1*scale, fps=2, caption="Image gt" , format="gif"),
                                                          wandb.Video(i2*scale, fps=2, caption="Image Pred" , format="gif"),
                                                          wandb.Video(i3*scale, fps=2, caption="Template gt" , format="gif"),
                                                          wandb.Video(i4*scale, fps=2, caption="Template pred" , format="gif"),
                                                          wandb.Video(i5*scale, fps=2, caption="Forward Flow" , format="gif"),
                                                          wandb.Video(i6*scale, fps=2, caption="Backward Flow" , format="gif"),
                                                          wandb.Video(i7*scale, fps=2, caption="Img Error" , format="gif")]})
    else:
        wandb.log({"GIF " + mode + " " + patient_name + " ": [wandb.Video(i1*scale, fps=2, caption="Image gt" , format="gif"),
                                                          wandb.Video(i3*scale, fps=2, caption="Template gt" , format="gif"),
                                                          wandb.Video(i4*scale, fps=2, caption="Template pred" , format="gif"),
                                                          wandb.Video(i5*scale, fps=2, caption="Forward Flow" , format="gif")]})
                                                          
    
def disimilarity_loss(img_gt, temp_gt, patient_slice_id_gt, flow_forward, flow_backward, epoch, mode, i_batch, args):
    '''
    image1_batch: [B, N, H, W] 
    template: [B, N, H, W]
    flow_predictions: list len iters, type tensor [B, N, 2, H, W]
    flow1 + imge = temp_generated --forward
    flow2 + temp = img_generated --backward
    '''
    partial_loss = 0
    total_iter = len(flow_forward)
    
    partial_temp_error = 0
    partial_img_error = 0
    
    partial_spatial_loss = 0
    partial_temporal_loss = 0
    partial_photo_loss = 0
    should_log = False
    for itr in range(0, total_iter):
        temp_pred = seq_utils.warp_batch(img_gt, flow_forward[itr], gpu=args.gpus[0]) # [B, N, H, W]
        l1_loss = torch.nn.L1Loss()
        partial_temp_error += l1_loss(temp_pred, temp_gt) 
        Charbonnier_Loss = CharbonnierLoss()
        photo_loss = Charbonnier_Loss(temp_pred, temp_gt) #[B, N] loss batch in iteration i
                
        if (args.model == 'group'):
            img_pred = seq_utils.warp_batch(temp_gt, flow_backward[itr], gpu=args.gpus[0]) # [B, N, H, W]
            partial_img_error += l1_loss(img_pred, img_gt)
            photo_loss += Charbonnier_Loss(img_pred, img_gt)
            
            if (args.composed_flows):
                #photo_loss = 0
                seq_len = img_gt.shape[1]
                idx_rnd = torch.randperm(seq_len)
                seq_img_rdn = torch.zeros_like(img_gt)
                comp_flow = torch.zeros_like(flow_backward[itr]) # {Ii, Ij} => FFi + BFj = Ij',  [B, N, 2, H, W]
                for i in range(0, seq_len):
                    seq_img_rdn[0,i,:,:] = img_gt[0,idx_rnd[i],:,:]
                    comp_flow[0,i,:,:,:] = flow_forward[itr][0,i,:,:,:] + flow_backward[itr][0,idx_rnd[i],:,:,:]
                img_pred_comp = seq_utils.warp_batch(img_gt, comp_flow, gpu=args.gpus[0]) # [B, N, H, W]
                photo_loss += args.beta_comp * Charbonnier_Loss(img_pred_comp, seq_img_rdn)
            
        partial_photo_loss += photo_loss 
        
        if (mode == 'training'):
            Spatial_Loss = SpatialSmooth(grad=1, boundary_awareness=True)
            spatial_loss = Spatial_Loss(flow_forward[itr][0,:,:,:,:], img_gt) 
            Temporal_Loss = TemporalSmooth(mode="forward", grad=1)
            temporal_loss = Temporal_Loss(flow_forward[itr][0,:,:,:,:]) 
            
            if (args.model == 'group'):
                spatial_loss += Spatial_Loss(flow_backward[itr][0,:,:,:,:], temp_gt)
                temporal_loss += Temporal_Loss(flow_backward[itr][0,:,:,:,:])
                #s = False
                '''
                if (args.composed_flows):
                    spatial_loss += Spatial_Loss(comp_flow[0,:,:,:,:], seq_img_rdn) 
                    temporal_loss += Temporal_Loss(comp_flow[0,:,:,:,:]) 
                '''
            partial_spatial_loss += spatial_loss
            partial_temporal_loss += temporal_loss 
            
            partial_loss += (photo_loss * args.beta_photo + 
                             spatial_loss * args.beta_spatial + 
                             temporal_loss * args.beta_temporal) * args.gamma ** (total_iter - itr - 1)
        else:
            partial_loss += (photo_loss * args.beta_photo) * args.gamma ** (total_iter - itr - 1)
        
        # The size of the batch is 1 sequence, so we only have 1 patient slice
        if (itr == total_iter - 1 and mode == "training" and (patient_slice_id_gt[0] == "patient020_z_4" 
                                                           or patient_slice_id_gt[0] == "patient065_z_0" 
                                                           or patient_slice_id_gt[0] =="patient096_z_17")):
            should_log = True
        elif(itr == total_iter - 1 and mode == "validation" and (patient_slice_id_gt[0] == "patient124_z_8" 
                                                              or patient_slice_id_gt[0] == "patient137_z_0" 
                                                              or patient_slice_id_gt[0] == "patient150_z_3")):  
            should_log = True
        else:
            should_log = False
        
        if (should_log):
            if (args.model == 'group'):
                # Log slice 3 of this patient
                log_images(img_gt[0,3,:,:], img_pred[0,3,:,:], temp_gt[0,3,:,:], temp_pred[0,3,:,:], 
                           flow_forward, flow_backward, 3, patient_slice_id_gt[0], mode)
                log_gifs(img_gt, img_pred, 
                         temp_gt, temp_pred, 
                         flow_forward, flow_backward, 
                         patient_slice_id_gt[0], mode, args.add_normalisation)
            else:
                log_images(img_gt[0,3,:,:], None, temp_gt[0,3,:,:], temp_pred[0,3,:,:], 
                           flow_forward, None, 3, patient_slice_id_gt[0], mode)
                log_gifs(img_gt, None, temp_gt, temp_pred, 
                         flow_forward, None, 
                         patient_slice_id_gt[0], mode, args.add_normalisation)
            
    loss_dict = {"Total": partial_loss / total_iter,
                 "Photometric": partial_photo_loss / total_iter,
                 "Spatial": partial_spatial_loss / total_iter,
                 "Temporal": partial_temporal_loss / total_iter,
                 "Img Error": partial_img_error / total_iter,
                 "Temp Error": partial_temp_error / total_iter}
    
    return loss_dict  
            
class SpatialSmooth(torch.nn.Module):
    def __init__(self, grad, boundary_awareness):
        super(SpatialSmooth, self).__init__()
        assert grad in (1, 2)
        self.loss = SpatialSmoothForward(grad=grad, boundary_awareness=boundary_awareness)

    def forward(self, flow, image=None):
        return self.loss(flow, image)


class SpatialSmoothForward(torch.nn.Module):
    def __init__(self, grad, boundary_awareness, boundary_alpha=10):
        super(SpatialSmoothForward, self).__init__()
        self.grad = grad
        self.boundary_awareness = boundary_awareness
        self.boundary_alpha = boundary_alpha

    def forward(self, flow, image=None): #[B, 2, H, W]
        dx, dy = gradient(flow)
        if self.grad == 1:
            final_x, final_y = dx.abs(), dy.abs()  

        elif self.grad == 2:
            dx2, dxdy = gradient(dx)
            dydx, dy2 = gradient(dy)
            final_x, final_y = dx2.abs(), dy2.abs()

        if self.boundary_awareness:
            img_dx, img_dy = gradient(image)
            weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * self.boundary_alpha)
            weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * self.boundary_alpha)
            loss_x = weights_x * final_x / 2.
            loss_y = weights_y * final_y / 2.
        else:
            loss_x = final_x / 2.
            loss_y = final_y / 2.

        return loss_x.mean() / 2. + loss_y.mean() / 2.
    
class TemporalSmooth(torch.nn.Module):
    def __init__(self, mode, grad):
        super(TemporalSmooth, self).__init__()
        assert mode in ('forward', 'central')
        assert grad in (1, 2)
        self.mode = mode
        self.grad = grad

    def forward(self, flow):
        dt = flow[1:, ...] - flow[:-1, ...] if self.mode == 'forward' else temporal_grad_central(flow) 
        # Seq without last img - Seq without first img 
        if self.grad == 2:
            dt = dt[1:, ...] - dt[:-1, ...] if self.mode == 'forward' else temporal_grad_central(dt)

        eps = 1e-6
        dt = torch.sqrt(dt**2 + eps)
        return dt.mean()/2

class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-6, alpha=0.45):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.alpha = alpha

    def forward(self, x, y):
        diff = x - y
        square = torch.conj(diff) * diff
        if square.is_complex():
            square = square.real
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.pow(square + self.eps, exponent=self.alpha))
        return loss