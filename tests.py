import core.sequence_handling_utils as seq_utils
import torch
import train_new as trainer

N = 10
B = 6
C = 3
H = 5
W = 4

def disimilarity_loss_same_batch_success():
    '''
    image_batch: [B, N, H, W]
    template: [B, N, H, W]
    flow_predictions: list len iters, type tensor [B, N, 2, H, W]
    '''
    image_batch = torch.ones(B, N, H, W)
    template_batch = torch.ones(B, N, H, W)
    flow = torch.zeros(B, N, 2, H, W)
    flow_predictions = [flow, flow, flow]
    loss = trainer.disimilarity_loss(image_batch, template_batch, flow_predictions, gamma=0.2)
    if(loss == 0):
        print("disimilarity_loss_same_batch_success passed.")
    else:
        print("disimilarity_loss_same_batch_success failed. Expected the loss of warped seq with null flow to be 0, got instead", loss)


disimilarity_loss_same_batch_success()