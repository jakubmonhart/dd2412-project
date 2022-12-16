"""
Concept Transformer is implemented here.
"""

import math
import torch
from torch import nn
import torch.nn.functional as F

from .attention import CrossAttention


class ConceptTransformer(nn.Module):
  """
  Only non-spatial for now (enough for MNIST dataset).
  """

  def __init__(self, n_concepts, dim, n_classes, att_pool=True, num_heads=2, is_spatial=False, allow_d=False):
    """
    num_heads is not specified in the text, from the code, it looks like 2 is used for MNIST experiment
    """
    super().__init__()
    self.is_spatial = is_spatial
    self.concepts = nn.Parameter(torch.zeros(1, n_concepts, dim), requires_grad=True) # First dimension is 1 to make it broadcastable over batches
    # Initialization of the parameter is not described in the text, only in the code in Appendix C (TODO - should we try other options)
    nn.init.trunc_normal_(self.concepts, std=1.0 / math.sqrt(dim))
    self.cross_attention = CrossAttention(dim=dim, out_dim=n_classes, num_heads=num_heads)
    
    if att_pool and not is_spatial:
      self.token_attention_pool = nn.Linear(dim, 1)
    else:
      self.token_attention_pool = None


  def forward(self, x):
    
    if self.token_attention_pool:
      # Pool over patches (taken from the Appendix C of the paper) - TODO - why not use simple mean here?
      token_attn = F.softmax(self.token_attention_pool(x), dim=1).transpose(-1, -2)
      x = torch.matmul(token_attn, x)
    out, attn = self.cross_attention(x, self.concepts) # out.shape = [B, P, n_classes] - P=1 for mnist experiment
    if not self.is_spatial:
      out = out.squeeze(1) # Squeeze over patches (there is only one for global concepts)
    else:
      out = out.mean(1)
    return out, attn

