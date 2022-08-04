'''
@author ManalHamdi
'''
import torch

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
    else:
        print("This mode", mode, "is not supported for template generation.")
    
def warp_seq(x, flo):
    """
    @author: Jiazhen Pan
    warp a sequence of images/tensor (im) back to template, according to the optical flow
    x: [N, H, W] (im2)   
    flo: [N, 2, H, W] flow
    """
    N, H, W = x.size() #[5, 224, 256]
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1) # [H, W]
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)  # [H, W]
    xx = xx.view(1, 1, H, W).repeat(N, 1, 1, 1)  # [N, 1, H, W]
    yy = yy.view(1, 1, H, W).repeat(N, 1, 1, 1)  # [N, 1, H, W]
    grid = torch.cat((xx, yy), 1).float() # [N, 2, H, W] true

    mask = torch.ones(x.size(), dtype=x.dtype) # [N, H, W]
    if x.is_cuda:
        grid = grid.cuda()
        mask = mask.cuda()

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
    
def warp_batch(batch_seq, batch_flo):
    """
    warp a batch of sequences of images/tensor (im) back to template, according to the batch of optical flow
    batch_seq: [B, N, H, W] (im2)
    batch_flo: [B, N, 2, H, W] flow
        """
    warped_seq_list = []
    for b in range(0, batch_seq.shape[0]):
        seq = batch_seq[b, :, :, :] #[N, H, W]
        flo_seq = batch_flo[b, :, :, :, :] 
        warped_seq = warp_seq(seq, flo_seq) # [N, H, W]
        warped_seq_list.append(warped_seq)
    warped_batch = torch.stack(warped_seq_list, dim=0) # [B, N, H, W]
    return warped_batch
