'''
@author ManalHamdi
'''
import torch
import random
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import torch.nn as nn

def generate_image(img_batch, template_batch, flow1, flow2):
    '''img and temp [N, H, W]
       flow [N, 2, H, W]
       flow1 img->temp
       flow2 temp->img'''
    seq_len, h, w = img_batch.shape
    
    idx1, idx2 = random.sample(range(0, seq_len), 2)
    img1 = img_batch[idx1,:,:].view(1,h,w)
    img2 = img_batch[idx2,:,:].view(1,h,w)
    flow1 = flow1[len(flow1)-1]
    flow2 = flow2[len(flow1)-1]
    # convert image 1 to 2
    temp1_g = warp_seq(img1, flow1[idx1,:,:,:].view(1,2,h,w))
    img2_g = warp_seq(temp1_g, flow2[idx2,:,:,:].view(1,2,h,w))
    return img2, img2_g
    
def generate_template(frame_seq, mode):
    if (mode == "avg"):
        if (len(frame_seq.shape) == 4):               
            '''Input [B, N, H, W] Returns tensor [N, H, W]'''
            return frame_seq.mean(dim=0)
        elif (len(frame_seq.shape) == 5):             
            '''Input [B, N, C, H, W] Returns tensor [B, C, H, W]'''
            return frame_seq.mean(dim=1)
        elif (len(frame_seq.shape) == 3):             
            '''Input [N, H, W] Returns tensor [H, W]'''
            return frame_seq.mean(dim=0)
    elif (mode == "pca"):
        return construct_template_pca(frame_seq)
    elif (mode == "learn"):
        return TemplateFormer()(frame_seq)
    else:
        print("This mode", mode, "is not supported for template generation.")

def construct_template_pca(seq):
    '''seq is a tensor with shape [N, H, W]'''
    s_len, h, w = seq.shape
    seq = seq.numpy() #[N, H, W]
    # Flatten the images in the seq 
    seq_flat = np.zeros((s_len, h*w)) # [5000, 15]

    for i in range(0, s_len):
        seq_flat[i] = seq[i,:,:].flatten()
    seq_flat = np.transpose(seq_flat) #[h*w, s], each col represents an image
    sc = StandardScaler()
    seq_flat = sc.fit_transform(seq_flat) # standarize and scale
    
    #pca = PCA(n_components=lvl_conf, svd_solver='full')
    pca = PCA(n_components=1)
    
    seq_flat_transformed = pca.fit_transform(seq_flat)
    template = seq_flat_transformed.reshape(h, w)
    return torch.tensor(template)
    #return seq_flat_transformed
    
def warp_seq(x, flo, gpu=0):
    """
    @author: Jiazhen Pan
    warp a sequence of images/tensor (im) back to template, according to the optical flow
    x: [N, H, W] (im2)   
    flo: [N, 2, H, W] flow
    """
    cuda_to_use = "cuda:" + str(gpu)
    N, H, W = x.size() #[5, 224, 256]
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1) # [H, W]
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)  # [H, W]
    xx = xx.view(1, 1, H, W).repeat(N, 1, 1, 1)  # [N, 1, H, W]
    yy = yy.view(1, 1, H, W).repeat(N, 1, 1, 1)  # [N, 1, H, W]
    grid = torch.cat((xx, yy), 1).float() # [N, 2, H, W] true

    mask = torch.ones(x.size(), dtype=x.dtype) # [N, H, W]
    if x.is_cuda:
        grid = grid.to(cuda_to_use)
        mask = mask.to(cuda_to_use)

    flo = torch.flip(flo, dims=[1])

    vgrid = grid + flo #[N, 2, H, W]
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1) #[N, H, W, 2]
    output = torch.nn.functional.grid_sample(x.view(N, 1, H, W).double(), vgrid.double(), align_corners=True) # [N, H, W]
    mask = torch.nn.functional.grid_sample(mask.view(N, 1, H, W).double(), vgrid.double(), align_corners=True)
    
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return (output * mask).view(N, H, W)
    
def warp_batch(batch_seq, batch_flo, gpu=0):
    """
    warp a batch of sequences of images/tensor (im) back to template, according to the batch of optical flow
    batch_seq: [B, N, H, W] (im2)
    batch_flo: [B, N, 2, H, W] flow
        """
    warped_seq_list = []
    for b in range(0, batch_seq.shape[0]):
        seq = batch_seq[b, :, :, :] #[N, H, W]
        flo_seq = batch_flo[b, :, :, :, :] 
        warped_seq = warp_seq(seq, flo_seq, gpu) # [N, H, W]
        warped_seq_list.append(warped_seq)
    warped_batch = torch.stack(warped_seq_list, dim=0) # [B, N, H, W]
    return warped_batch

class TemplateFormer(nn.Module):
    def __init__(self, ch_num=[64, 32, 1], seq_group=18, circular=3, average_init=True):
        super(TemplateFormer, self).__init__()
        self.conv1 = nn.Conv1d(1, 1, (3, 1, 1), 1, 1)
        self.seq_group = seq_group
        layers = []
        ch_num = [1] + ch_num
        for i, ch in enumerate(ch_num):
            conv = nn.Conv1d(ch, ch_num[i+1], (circular*2+1, 1, 1), 1, (circular, 0, 0))
            if average_init:
                torch.nn.init.constant_(conv.weight, 1/(circular*2+1))
            layers.append(conv)
            layers.append(nn.ReLU())
            if i == len(ch_num) - 2:
                # layers = layers[:-1]
                break
        self.seq = nn.Sequential(*layers)
        self.linear = nn.Linear(seq_group, 1)
        if average_init:
            torch.nn.init.constant_(self.linear.weight, 1/seq_group)

    def form_seq_group(self, x):
        seq_len = x.shape[1]
        node = [int(np.ceil(i*seq_len/self.seq_group)) for i in range(self.seq_group)]
        group = []
        for i in range(len(node)-1):
            group.append(x[:, node[i]:node[i+1], ...].mean(dim=1, keepdim=True))
            if i == len(node)-2:
                group.append(x[:, node[i+1]:, ...])
        return torch.cat(group, dim=1)

    def forward(self, x):
        group = self.form_seq_group(x)
        group = self.seq(group[:, None, ...])
        group = nn.ReLU()(self.linear(group[0].permute(0, 2, 3, 1)).permute(0, 3, 1, 2))

        return group/group.max()*255