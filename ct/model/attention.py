from torch import nn


class CrossAttention(nn.Module):
  """
  Attention (self-attention) obtained from timm: github.com:rwightman/pytorch-image-models

  Both inputs into CrossAttention will have the same dimension: dim_embedding.
  dim_embedding is the output dimension of the backbone (in all experiments (CCT or VIT)), we can therefore assume both
  inputs will have the same last dimension (last entry in the shape array).

  Modified to use distinct values for q and k, v -> cross-attention (actually the basic attention from 'Attention is all you need' ?).
  Only modified thing is done by splitting self.qkv into self.q and self.kv and corresponding operations in the forawd method.

  This is almost identical to what authors of Concept Transformer - is that bad? This naturally comes from extending Attention
  from implementation of CCT ...
  """

  def __init__(self, dim, out_dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
    super().__init__()
    self.num_heads = num_heads
    head_dim = dim // self.num_heads
    self.scale = head_dim ** -0.5

    self.q = nn.Linear(dim, dim, bias=False) # TODO - do we want bias here?
    self.k = nn.Linear(dim, dim, bias=False)
    self.v = nn.Linear(dim, dim, bias=False)
    self.attn_drop = nn.Dropout(attention_dropout)
    self.proj = nn.Linear(dim, out_dim)
    self.proj_drop = nn.Dropout(projection_dropout)

  def forward(self, x, y):
    """
    Inputs:
      x - output of backbone (processed image patches for all of our experiments?) Nx is usually num of visual patches
      y - concepts - Ny is number of concepts
    """
    # print(x.shape, y.shape, '\n\n')
    B, Nx, D = x.shape
    By, Ny, Dy = y.shape 

    assert By == 1 # We are using the same concept for each image in the batch, 
                   # batch dimension is broadcasted in attn computation 
    
    assert D == Dy
    q = self.q(x).reshape(B, Nx, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3) # [B, NH, Nx, DH]
    k = self.k(y).reshape(1, Ny, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3) # [1, NH, Ny, DH]
    v = self.v(y).reshape(1, Ny, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3) # [1, NH, Ny, DH]

    attn = (q @ k.transpose(-2, -1)) * self.scale # [B, NH, Nx, Ny]
    attn = attn.softmax(dim=-1) # [B, NH, Nx, Ny]
    attn = self.attn_drop(attn)

    x = (attn @ v) # [B, NH, Nx, DH]
    x = x.transpose(1, 2) # [B, Nx, NH, DH]
    x = x.reshape(B, Nx, D) # [B, Ny, D]
    x = self.proj(x) # [B, Ny, out_dim] - logits over classes (separerely for each patch). For decision over multiple patches, average must be taken.
    x = self.proj_drop(x)

    return x, attn # Attention needs to be returned to compute the explanation loss


class SelfAttention(nn.Module):
  """
  Obtained from timm: github.com:rwightman/pytorch-image-models

  Renamed to SelfAttention (originaly Attention), as such name is more fitting.
  """

  def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
    super().__init__()
    self.num_heads = num_heads
    head_dim = dim // self.num_heads
    self.scale = head_dim ** -0.5

    self.qkv = nn.Linear(dim, dim * 3, bias=False)
    self.attn_drop = nn.Dropout(attention_dropout)
    self.proj = nn.Linear(dim, dim)
    self.proj_drop = nn.Dropout(projection_dropout)

  def forward(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x