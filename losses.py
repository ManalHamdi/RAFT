import torch
import sys
sys.path.append('core')
import core.sequence_handling_utils as seq_utils
import wandb
import flow_vis

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

def disimilarity_loss(img_gt, temp_gt, flow_forward, flow_backward, epoch, mode, i_batch, args):
    '''
    image1_batch: [B, N, H, W] [B,N,H, W]
    template: [B, N, H, W]
    flow_predictions: list len iters, type tensor [B, N, 2, H, W]
    flow1 + imge = temp_generated --forward
    flow2 + temp = img_generated --backward
    '''
    partial_loss = 0
    total_iter = len(flow_forward)
    partial_error = 0
    partial_spatial_loss = 0
    partial_temporal_loss = 0
    
    for itr in range(0, total_iter):
        temp_pred = seq_utils.warp_batch(img_gt, flow_forward[itr]) # [B, N, H, W]
        img_pred = seq_utils.warp_batch(temp_gt, flow_backward[itr]) # [B, N, H, W]
        
        Charbonnier_Loss = CharbonnierLoss()
        photo_loss = Charbonnier_Loss(temp_pred, temp_gt) + Charbonnier_Loss(img_pred, img_gt) #[B, N] loss of batch in iteration i
        
        Spatial_Loss = SpatialSmooth(grad=1, boundary_awareness=False)
        spatial_loss = Spatial_Loss(flow_forward[itr][0,:,:,:,:], img_gt) + Spatial_Loss(flow_backward[itr][0,:,:,:,:], temp_gt)
        partial_spatial_loss += spatial_loss
        
        Temporal_Loss = TemporalSmooth(mode="forward", grad=1)
        temporal_loss = Temporal_Loss(flow_forward[itr][0,:,:,:,:]) + Temporal_Loss(flow_backward[itr][0,:,:,:,:])
        partial_temporal_loss += temporal_loss 
        
        partial_loss += (photo_loss * args.beta_photo + spatial_loss * args.beta_spatial + temporal_loss * args.beta_temporal) * args.gamma ** (total_iter - itr - 1)
        '''
        if (mode == "training"):
            wandb.log({"Training Spatial Loss": spatial_loss})
            wandb.log({"Training Temporal Loss": spatial_loss})
        elif (mode == "validation"):
            wandb.log({"Validation Spatial Loss": spatial_loss})
            wandb.log({"Validation Temporal Loss": spatial_loss})
        '''
        l1_loss = torch.nn.L1Loss()
        partial_error += l1_loss(temp_pred, temp_gt) + l1_loss(img_pred, img_gt)
        if (itr == total_iter-1 and (mode == "training" and i_batch % 700 == 0) or (mode == "validation" and i_batch % 200 == 0)):
            flow_for = flow_vis.flow_to_color(flow_forward[itr][0,3,:,:,:].permute(1,2,0).cpu().detach().numpy())
            flow_back = flow_vis.flow_to_color(flow_backward[itr][0,3,:,:].permute(1,2,0).cpu().detach().numpy())
            
            wandb.log({mode + " Images": [wandb.Image(img_gt[0,3,:,:], caption="Image GT"), 
                                          wandb.Image(img_pred[0,3,:,:], caption="Image Pred"),
                                          wandb.Image(temp_gt[0,3,:,:], caption="Template GT"),
                                          wandb.Image(temp_pred[0,3,:,:], caption="Template Pred"),
                                          wandb.Image(flow_for, caption="Forward Flow"),
                                          wandb.Image(flow_back, caption="BackwardFlow"),]})
          
    return partial_loss / total_iter, partial_error / total_iter, partial_spatial_loss / total_iter, partial_temporal_loss / total_iter

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