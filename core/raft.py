import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8
import sequence_handling_utils as seq_utils

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4
        '''
        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False
        '''
        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0 [B, C, H, W]"""
        B, C, H, W = img.shape
        coords0 = coords_grid(B, H//8, W//8, device=img.device)
        coords1 = coords_grid(B, H//8, W//8, device=img.device) # [B, 2, H//8, W//8]

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1 #[B, 2, H//8, W//8]

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)
    
    def predict_flow(self, image_batch, template_batch, iters=12, flow_init=None, upsample=True, test_mode=False):
        batch_size, seq_size, h, w = image_batch.shape #batch size is num_seq = 1
        
        assert batch_size == 1, "ERROR: batch size is not 1 sequence."
        
        image_batch = image_batch.view(seq_size, 1, h, w)  #[N, 1, H, W]
        template_batch = template_batch.view(seq_size, 1, h, w)  #[N, 1, H, W]
        
        image_batch = 2 * (image_batch / 255.0) - 1.0
        template_batch = 2 * (template_batch / 255.0) - 1.0

        image_batch = image_batch.contiguous()
        template_batch = template_batch.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image_batch, template_batch])        

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image_batch)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image_batch) # [B,2,H//8,W//8]

        if flow_init is not None:
            coords1 = coords1 + flow_init # [B,2,H//8,W//8]
        
        flow_predictions = []  
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume  #[B,2,H//8,W//8]

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow
        
            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0) #flow_up [N, 2, H, W]
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up.view(1, seq_size, 2, h, w))
        
        return flow_predictions #list with length of iters, 1 element is [1, N, 2, H, W]
        
    def forward(self, image_batch, template_batch, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames B: batch size N: length of sequence
            image and template batch are tensors [B, N, H, W] // instead of [B, C, H, W]
            Returns list of flow estimations with length iters, and each item of the list is [B, N, 2, H, W]// [B, 2, H, W]
        """
        template_model = nn.DataParallel(seq_utils.TemplateFormer(), device_ids=self.args.gpus)
        b, n, h, w = image_batch.shape
        if (self.args.learn_temp):
            template_batch = template_model(image_batch.float())
            template_batch = template_batch.repeat(1,n, 1, 1)
            cuda_to_use = "cuda:" + str(self.args.gpus[0])
            image_batch, template_batch = image_batch.to(cuda_to_use), template_batch.to(cuda_to_use)
            
            
        flow_pred1 = self.predict_flow(image_batch, template_batch, iters=12, 
                                       flow_init=None, upsample=True, test_mode=False) #[B, N, 2, H, W]
        flow_pred2 = None
        if (self.args.model == 'group'):
            flow_pred2 = self.predict_flow(template_batch, image_batch, iters=12, 
                                           flow_init=None, upsample=True, test_mode=False) #[B, N, 2, H, W]
        return flow_pred1, flow_pred2, template_batch
        