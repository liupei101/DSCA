#coding=utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention, Nystromformer

from .model_utils_extra import *
from utils import to_relative_coord


@torch.no_grad()
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()

def topk_keep_order(score, k):
    """
    The results won't destroy the original order.

    score: [B, N].
    """
    _, sorted_idx = torch.topk(score, k)
    idx, _ = torch.sort(sorted_idx)
    new_score = torch.gather(score, -1, idx)
    return new_score, idx

def generate_mask(idxs, n):
    sz = list(idxs.size())
    res = torch.zeros(sz[:-1] + [n]).bool().to(idxs.device)
    res.scatter_(-1, idxs, True)
    return res

def extend_mask(mask, scale=4):
    """
    mask = [..., N]
    """
    if scale == 1:
        return mask

    mask_size = list(mask.size())
    tmp = mask.unsqueeze(-1)
    tensor_ex = torch.ones(mask_size + [scale * scale]).to(mask.device)
    res = tensor_ex * tmp
    res = res.view(mask_size[:-1] + [-1]).bool()
    return res

def square_seq(x):
    """
    x: [1, N, d] -> [1, L^2, d]
    """
    B, N = x.shape[0], x.shape[1]
    L = int(np.ceil(np.sqrt(N)))
    len_padding = L * L - N
    x = torch.cat([x, x[:, :len_padding, :]], dim=1) # square [1, L^2, 512]
    x = x.reshape(B, L, L, -1)
    return x

def square_align_seq(x1, x2, scale=4):
    """
    x1: [1, 16N, d1], level = 1
    x2: [1,   N, d2], level = 2
    """
    B, N = x2.shape[0], x2.shape[1]
    D1, D2 = x1.shape[2], x2.shape[2]
    L = int(np.ceil(np.sqrt(N)))
    len_padding = L * L - N

    x2 = torch.cat([x2, x2[:, :len_padding, :]], dim=1) # [1, L^2, 512]
    x1 = torch.cat([x1, x1[:, :(scale*len_padding), :]], dim=1) # [1, (4L)^2, 512]

    x2 = x2.reshape(B, L, L, -1)
    # spatial alignment
    x1 = x1.view(B, L, L, scale, scale, -1)
    x1 = x1.transpose(3, 4).reshape(B, L, L*scale, scale, -1)
    x1 = x1.transpose(2, 3).reshape(B, -1, L*scale, D1)

    return x1, x2

def sequence2square(x, s):
    """
    [B, N, C] -> [B*(N/s^2), C, s, s]
    """
    size = x.size()
    assert size[1] % (s * s) == 0
    L = size[1] // (s * s)
    x = x.view(-1, s, s, size[2])
    x = x.permute(0, 3, 1, 2)
    return x, L

def square2sequence(x, L):
    """
    [B*L, C, s, s] -> [B, L*s*s, c]
    """
    size = x.size()
    assert size[0] % L == 0
    x = x.view(size[0], size[1], -1)
    x = x.transpose(2, 1).view(size[0]//L, -1, size[1])
    return x

def posemb_sincos_2d(y, x, dim, device, dtype, temperature=10000):
    """
    Adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py#L12
    """
    # y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

def compute_pe(coord: torch.Tensor, ndim=384, step=1, device='cpu', dtype=torch.float):
    # coord: [B, N, 2]
    assert coord.shape[0] == 1
    coord = coord.squeeze(0)
    ncoord, ref_xy, rect = to_relative_coord(coord)
    assert rect[0] % step == 0 and rect[1] % step == 0
    y = torch.div(ncoord[:, 1], step, rounding_mode='floor')
    x = torch.div(ncoord[:, 0], step, rounding_mode='floor')
    PE = posemb_sincos_2d(y, x, ndim, device, dtype) # [N, ndim]
    PE = PE.unsqueeze(0) # [1, N, ndim]
    return PE

def make_conv1d_layer(in_dim, out_dim, kernel_size=3, spatial_conv=True):
    conv1d_ksize = kernel_size if spatial_conv else 1
    p = (conv1d_ksize - 1) // 2
    return Conv1dPatchEmbedding(in_dim, out_dim, conv1d_ksize, stride=1, padding=p)

#####################################################################################
#
#    Functions/Classes for Patch Embedding, intended to
#    1. aggregate patches in a small field into regional features using 1D/2D Conv
#    2. reduce feature dimension
#
#####################################################################################
def make_embedding_layer(backbone:str, args):
    """
    backbone: ['conv1d', 'gapool', 'avgpool', 'capool', 'identity']
    """
    if backbone == 'conv1d':
        layer = Conv1dPatchEmbedding(args.in_dim, args.out_dim, args.ksize, stride=1, padding=(args.ksize-1)//2)
    elif backbone == 'gapool':
        layer = GAPoolPatchEmbedding(args.in_dim, args.out_dim, args.scale, args.dw_conv, args.ksize)
    elif backbone == 'avgpool':
        layer = AVGPoolPatchEmbedding(args.in_dim, args.out_dim, args.scale, args.dw_conv, args.ksize)
    elif backbone == 'capool':
        layer = CAPoolPatchEmbedding(args.in_dim, args.out_dim, args.scale, args.dw_conv, args.ksize)
    elif backbone == 'sconv':
        layer = SquareConvPatchEmbedding(args.in_dim, args.out_dim, args.scale, args.dw_conv, args.ksize)
    elif backbone == 'identity':
        layer = IdentityPatchEmbedding(args.in_dim, args.out_dim)
    else:
        raise NotImplementedError(f'{backbone} has not implemented.')
    return layer


class IdentityPatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(IdentityPatchEmbedding, self).__init__()
        if in_dim == out_dim:
            self.layer = nn.Identity()
        else:
            self.layer = nn.Sequential(
                nn.Linear(in_dim, out_dim), 
                nn.LayerNorm(out_dim), 
                nn.ReLU(),
            )

    def forward(self, x):
        out = self.layer(x)
        return out


class AVGPoolPatchEmbedding(nn.Module):
    """
    head layer (FC/Conv2D) + pooling Layer (avg pooling) for patch embedding.

    ksize = 1 -> head layer = FC
    ksize = 3 -> head layer = Conv2D

    Patch data with shape of [B, N, C]
    if scale = 1, then apply Conv2d with stride=1
        [B, N, C] -> [B, C, N] --conv1d--> [B, C', N]
    elif scale = 2/4, then apply Conv2d with stride=2
        [B, N, C] -> [B*(N/s^2), C, s, s] --conv2d--> [B*(N/s^2), C, 1, 1] -> [B, N/s^2, C]
    """
    def __init__(self, in_dim, out_dim, scale:int=4, dw_conv=False, ksize=3, stride=1):
        super(AVGPoolPatchEmbedding, self).__init__()
        assert scale == 4, 'It only supports for scale = 4'
        assert ksize == 1 or ksize == 3, 'It only supports for ksize = 1 or 3'
        self.scale = scale
        self.stride = stride
        if scale == 4:
            # Conv2D on the grid of 4 x 4: stride=2 + ksize=3 or stride=1 + ksize=1/3
            assert (stride == 2 and ksize == 3) or (stride == 1 and (ksize == 1 or ksize == 3)), \
                'Invalid stride or kernel_size when scale=4'
            if dw_conv:
                self.conv = SeparableConvBlock(in_dim, out_dim, ksize, stride, norm=False)
            else:
                self.conv = nn.Conv2d(in_dim, out_dim, ksize, stride, padding=(ksize-1)//2)
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise NotImplementedError()

        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: [B, N ,C]
        """
        x, L = sequence2square(x, self.scale) # [B*N/16, C, 4, 4]
        x = self.conv(x) # [B*N/16, C, 4/s, 4/s]
        x = square2sequence(x, L) # [B, N/(s*s), C]
        x = self.norm(x)
        x = self.act(x)
        x, L = sequence2square(x, self.scale//self.stride) # [B*N/16, C, 4/s, 4/s]
        x = self.pool(x) # [B*N/16, C, 1, 1]
        x = square2sequence(x, L) # [B, N/16, C]
        return x


class GAPoolPatchEmbedding(nn.Module):
    """
    head layer (FC/Conv2D) + pooling Layer (global-attention pooling) for patch embedding.

    ksize = 1 -> head layer = FC
    ksize = 3 -> head layer = Conv2D

    Global Attention Pooling for patch data with shape of [B, N, C].
    [B, N, C] -> [B, N/(scale^2), C']
    """
    def __init__(self, in_dim, out_dim, scale:int=4, dw_conv:bool=False, ksize=3):
        super(GAPoolPatchEmbedding, self).__init__()
        assert scale == 4, 'It only supports for scale = 4'
        assert ksize == 1 or ksize == 3, 'It only supports for ksize = 1 or 3'
        self.scale = scale
        if dw_conv:
            self.conv = SeparableConvBlock(in_dim, out_dim, ksize, 1, norm=False)
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, ksize, 1, padding=(ksize-1)//2)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU(inplace=True)
        self.pool = GAPool(out_dim, out_dim, 0.0)

    def forward(self, x):
        # conv2d (strid=1) embedding (spatial continuity)
        x, L = sequence2square(x, self.scale) # [B*N/(s^2), C, s, s]
        x = self.conv(x) # [B*N/(s^2), C, s, s]
        x = square2sequence(x, L) # [B, N, C]
        x = self.norm(x)
        x = self.act(x)
        
        # gapool
        sz = x.size() # [B, N, C]
        x = x.view(-1, self.scale*self.scale, sz[2]) # [B*N/(scale^2), scale*scale, C]
        x, x_attn = self.pool(x) # [B*N/(scale^2), C]
        x = x.view(sz[0], -1, sz[2]) # [B, N/(scale^2), C]
        return x


class CAPoolPatchEmbedding(nn.Module):
    """
    head layer (FC/Conv2D) + pooling Layer (cross-attention pooling) for patch embedding.

    ksize = 1 -> head layer = FC
    ksize = 3 -> head layer = Conv2D

    Patch Embedding guided by x5 patches
    """
    def __init__(self, in_dim, out_dim, scale:int=4, dw_conv:bool=True, ksize=3):
        super(CAPoolPatchEmbedding, self).__init__()
        self.scale = scale
        assert scale != 1, "Please pass a scale larger than 1 for capool."
        # Conv1D-ksize_1 (= FC layer) for x5 patches to make dimension equal to the x20 patches 
        self.conv_patch_x5 = Conv1dPatchEmbedding(in_dim, out_dim, 1, norm=False, activation=False)
        # Conv2D for x20 patches
        assert ksize == 1 or ksize == 3, 'It only supports for ksize=1/3 for embedding layer at scale=4'
        if dw_conv:
            self.conv = SeparableConvBlock(in_dim, out_dim, ksize, 1, norm=False)
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, ksize, 1, padding=(ksize-1)//2)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU(inplace=True)
        self.cross_att_pool = MultiheadAttention(embed_dim=out_dim, num_heads=4)
        
    def forward(self, x20, x5):
        # firstly reduce the dimension of x5
        x5 = self.conv_patch_x5(x5)

        # conv2d (strid=1) embedding (spatial continuity)
        x20, L = sequence2square(x20, self.scale) # [B*N/(s^2), C, s, s]
        x20 = self.conv(x20) # [B*N/(s^2), C', s, s]
        x20 = square2sequence(x20, L) # [B, 16N, C']
        x20 = self.norm(x20)
        x20 = self.act(x20)
        
        assert x5.shape[1] == L # N == L
        assert x20.shape[2] == x5.shape[2]
        # x5 patch guided pooling
        #[B, 16N, C]->[B*L, 16, C]->[16, B*L, C]
        x20 = x20.view(-1, self.scale*self.scale, x20.shape[2]).transpose(0, 1) 
        # [B, N, C]->[B*N, 1, C]->[1, B*L, C]
        x5 = x5.view(-1, 1, x5.shape[2]).transpose(0, 1)
        
        # x: [1, B*L, C], x_attn: [B*L, num_heads, 1, 16]
        x, x_attn = self.cross_att_pool(x5, x20, x20)
        # [B*L, num_heads, 1, 16] -> [B, L, num_heads, 1, 16] -> [B, L, num_heads, 16]
        x_attn = x_attn.view(-1, L, x_attn.shape[1], x_attn.shape[2]*x_attn.shape[3])
        x5 = x5.view(-1, L, x5.shape[2])
        x = x.squeeze().view(-1, L, x20.shape[2])
        return x, x_attn, x5


class SquareConvPatchEmbedding(nn.Module):
    """
    Conv for patch data with shape of [B, N, C].
    The convolution is paticularly applied to the squared sequences (Refer to TransMIL).

    We use [conv2d + avgpool] as its architecture as same as the ConvPatchEmbedding layer to 
    keep the total number of learnable parameters same.

    [B, N, C] -> [B, C, L, L] --conv2d--> [B, C', L/2, L/2] --avgpool--> [B, C', L/4, L/4]
    """
    def __init__(self, in_dim, out_dim, scale:int=4, dw_conv=False, ksize=3):
        super(SquareConvPatchEmbedding, self).__init__()
        assert scale == 4, 'It only used in x20 magnification'
        if dw_conv:
            self.conv = SeparableConvBlock(in_dim, out_dim, ksize, 2, norm=False)
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, ksize, 2, padding=(ksize-1)//2)
        self.pool = nn.AvgPool2d(2, 2)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: [B, N ,C]
        """
        H = x.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        x = torch.cat([x, x[:,:add_length,:]], dim = 1)

        # [B, L*L, C]
        B, _, C = x.shape
        cnn_feat = x.transpose(1, 2).view(B, C, _H, _W) # [B, C, L, L]
        x = self.conv(cnn_feat) # [B, C', L/2, L/2]
        _, C, _H, _W = x.shape
        x = x.flatten(2).transpose(1, 2) # [B, L/2*L/2, C']

        # [B, L/2*L/2, C']
        x = self.norm(x)
        x = self.act(x)
        cnn_feat = x.transpose(1, 2).view(B, C, _H, _W) # [B, C', L/2, L/2]
        x = self.pool(cnn_feat) # [B, C', L/4, L/4]
        x = x.flatten(2).transpose(1, 2) # [B, L/2*L/2, C']
        return x

###################################################################################
#
#    Functions/Classes for capturing region denpendency by
#    1. Transformer: may be suitable for dense relations
#    2. SimTransformer: may be suitable for sparse relations (rubost to noises)
#    3. Conv1D/Conv2D: is used for relations of spatial neighbours
#
###################################################################################
def make_transformer_layer(backbone:str, args):
    """
    [B, N, C] --Transformer--> [B, N, C]

    Transformer/Nystromformer: for long range dependency building.
    Conv1D/Conv2D: for short range dependency building.
    """
    if backbone == 'Transformer':
        patch_encoder_layer = nn.TransformerEncoderLayer(
            args.d_model, args.nhead, dim_feedforward=args.d_model, 
            dropout=args.dropout, activation='relu', batch_first=True
        )
        patch_transformer = nn.TransformerEncoder(patch_encoder_layer, num_layers=args.num_layers)
    elif backbone == 'Nystromformer':
        patch_transformer = Nystromformer(
            dim=args.d_model, depth=args.num_layers, heads=args.nhead,
            attn_dropout=args.dropout
        )
    elif backbone == 'Conv1D':
        patch_transformer = Conv1dPatchEmbedding(
            args.d_model, args.d_out, args.ksize, 1, padding=(args.ksize - 1) // 2,
            norm=True, dw_conv=args.dw_conv, activation=True
        )
    elif backbone == 'Conv2D':
        patch_transformer = Conv2dPatchEmbedding(
            args.d_model, args.d_out, args.ksize, 1, padding=(args.ksize - 1) // 2,
            norm=True, dw_conv=args.dw_conv, activation=True
        )
    elif backbone == 'SimTransformer':
        patch_transformer = SimTransformer(
            args.d_model, proj_qk_dim=args.d_out, proj_v_dim=args.d_out, 
            epsilon=args.epsilon
        )
    elif backbone == 'Identity':
        patch_transformer = nn.Identity()
    else:
        raise NotImplementedError(f'{backbone} has not implemented.')
    return patch_transformer


class Conv1dPatchEmbedding(nn.Module):
    """Conv1dPatchEmbedding"""
    def __init__(self, in_dim, out_dim, conv1d_ksize, stride=1, padding=0, 
        norm=True, dw_conv=False, activation=False):
        super(Conv1dPatchEmbedding, self).__init__()
        if dw_conv:
            self.conv = nn.Conv1d(in_dim, out_dim, conv1d_ksize, stride, padding, group=out_dim)
        else:
            self.conv = nn.Conv1d(in_dim, out_dim, conv1d_ksize, stride, padding)
        if norm:
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = None
        if activation:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        """x: [B, N, C]"""
        x = x.transpose(2, 1) # [B, C, N]
        x = self.conv(x) # [B, C', N]
        x = x.transpose(2, 1) # [B, N, C']
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class Conv2dPatchEmbedding(nn.Module):
    """Conv2dPatchEmbedding
    Conv2dPatchEmbedding: sequences to square and make 2d conv.
    """
    def __init__(self, in_dim, out_dim, conv2d_ksize, stride=1, padding=0, 
        norm=True, dw_conv=True, activation=True):
        super(Conv2dPatchEmbedding, self).__init__()
        if dw_conv:
            self.conv = SeparableConvBlock(in_dim, out_dim, conv2d_ksize, stride, norm=False)
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, conv2d_ksize, stride, padding)
        if norm:
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = None
        if activation:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        """x: [B, N, C]"""
        squ_x = square_seq(x) # [B, L, L, C]
        squ_x = squ_x.permute(0, 3, 1, 2) # [B, C, L, L]
        squ_x = self.conv(squ_x) # [B, C, L, L]
        x = squ_x.flatten(2).transpose(2, 1) # [B, L*L, C]
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SimTransformer(nn.Module):
    def __init__(self, in_dim, proj_qk_dim=None, proj_v_dim=None, epsilon=None):
        """
        in_dim: the dimension of input.
        proj_qk_dim: the dimension of projected Q, K.
        proj_v_dim: the dimension of projected V.
        topk: number of patches with highest attention values.
        """
        super(SimTransformer, self).__init__()
        self._markoff_value = 0
        self.epsilon = epsilon
        if proj_qk_dim is None:
            proj_qk_dim = in_dim
        if proj_v_dim is None:
            proj_v_dim = in_dim
        self.proj_qk = nn.Linear(in_dim, proj_qk_dim, bias=False)
        nn.init.xavier_uniform_(self.proj_qk.weight)
        self.proj_v = nn.Linear(in_dim, proj_v_dim, bias=False)
        nn.init.xavier_uniform_(self.proj_v.weight)
        self.norm = nn.LayerNorm(proj_v_dim)

    def forward(self, x):
        q, k, v = self.proj_qk(x), self.proj_qk(x), self.proj_v(x)
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        attention = torch.matmul(q_norm, k_norm.transpose(-1, -2))
        if self.epsilon is not None:
            mask = (attention > self.epsilon).detach().float()
            attention = attention * mask + self._markoff_value * (1 - mask)
        out = torch.matmul(attention, v)
        out = self.norm(out)
        return out
