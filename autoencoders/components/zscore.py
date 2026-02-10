"""
This module contains the ZScore class, which is a component for normalizing data using the z-score method. The z-score normalization transforms the data to have a mean of 0 and a standard deviation of 1.

Kindly refer to this paper for an example of how ZScore normalization is used in the context of medical autoencoders:https://pmc.ncbi.nlm.nih.gov/articles/PMC6758567/


"""
import torch
import torch.nn as nn


class ZScore(nn.Module):
  def __init__(self, eps = 1e-6):
    super().__init__()
    self.eps = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Accept (B,H,W) or (B,C,H,W)
    if x.ndim == 3:
      x = x.unsqueeze(1)  # (B,1,H,W)

    B = x.shape[0]
    x_out = x.clone()

    for b in range(B):
      xb = x[b]                 # (C,H,W)
      mask = (xb != 0)

      if mask.sum() < 10:
        mean = xb.mean()
        std = xb.std().clamp_min(self.eps)
      else:
        vals = xb[mask]
        mean = vals.mean()
        std = vals.std().clamp_min(self.eps)

      x_out[b] = (xb - mean) / std
      

    return x_out
  
  