import torch.nn.functional as F
import torch

def coord_loss(pred_coords, target_coords, loss_mode="mse"):
    """
    pred_coords, target_coords: (B, 2) tensors, x,y normalized to [0, 1]
    loss_mode: "mse" or "mae"
    """
    if loss_mode == "mse":
        return F.mse_loss(pred_coords, target_coords)
    elif loss_mode == "mae":
        return F.l1_loss(pred_coords, target_coords)
    else:
        raise ValueError(f"Unknown loss : {loss_mode}")

