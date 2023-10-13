import math
from telnetlib import PRAGMA_HEARTBEAT
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from einops import rearrange 


class TargetEmbedding(nn.Module):
    def __init__(self, num_labels, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        self.targetEmbedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_labels , embedding_dim=d_model, padding_idx=0),
            nn.Linear(d_model, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.targetEmbedding(t)
        return emb

class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c1 = nn.Sequential(nn.GELU(),
        nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1, padding_mode='reflect'))
        self.c2 = nn.Sequential(nn.GELU(),
        nn.Conv2d(in_ch, in_ch, 5, stride=2, padding=2, padding_mode='reflect'))
    def forward(self, x,  target_emb):
        x = self.c1(x)+self.c2(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.t = nn.Sequential(nn.GELU(), nn.GroupNorm(16, in_ch),
        nn.ConvTranspose2d(in_ch, in_ch, 5, 2, 2, 1))

    def forward(self, x,  target_emb):
        _, _, H, W = x.shape
        x = self.t(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.norm = nn.GroupNorm(16,in_ch)
        
        self.to_kv=nn.Conv2d(in_ch,in_ch*2,1)

        self.out = nn.Sequential(nn.GroupNorm(16,in_ch), 
        nn.GELU(),
        nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
                                )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        q_scale = int(C) ** (-0.5)
        kv = self.to_kv(x).chunk(2,dim=1)
        k,v = map(lambda t: rearrange(t, 'b c x y -> b c (x y)'),kv)
        q = F.softmax(k,dim=-2)
        k = F.softmax(k,dim=-1)
        #q = F.sigmoid(k)
        #k = F.sigmoid(k)
        q = q*q_scale
        context = torch.einsum('b d n, b e n -> b d e',k,v)
        assert list(context.shape) == [B, C, C]
        out = torch.einsum('b d e, b d n -> b e n',context,q)

        assert list(out.shape) == [B, C, H*W]
        out = rearrange(out,'b c (i j) -> b c i j', i=H ,j=W)
        out = self.out(out)
        return x + out

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, attn=True):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.GELU(),
            nn.GroupNorm(16, in_ch),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, padding_mode='reflect'),
        )

        self.target_proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(tdim, out_ch),
        )

        self.block2 = nn.Sequential(
            nn.GELU(),
            nn.GroupNorm(16, out_ch),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, padding_mode='reflect'),
            
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()

    def forward(self, x ,  target):
        h = self.block1(x)
        h += self.target_proj(target)[:, :, None, None]
        h = self.block2(h)
        B,C,H,W = h.size()
        h = nn.LayerNorm([C,H,W],device=h.device)(h)
        h = h + self.shortcut(x)
        h = self.attn(h)

        return h

class GCT(nn.Module):

    def __init__(self, num_channels, tdim, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu
        self.target_proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(tdim, num_channels),
        )
    def forward(self, x):
        b, c, h, w = x.shape
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3), keepdim=True) +
                         self.epsilon).pow(0.5) * self.alpha 
            norm = self.gamma / \
                (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)

        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha 
            norm = self.gamma / \
                (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate

class Generator(nn.Module):
    def __init__(self, num_target, ch, ch_mult, num_res_blocks):
        super().__init__()
        tdim = ch * 4
        self.target_embedding = TargetEmbedding(num_target, ch, tdim)
        self.head = nn.Sequential(
            nn.Conv2d(3, ch, kernel_size=5, padding=2, padding_mode='reflect'),
            nn.GELU(),
            nn.GroupNorm(16, ch)
            
        )  
        self.downblocks = nn.ModuleList()
        chs = [ch]  
        now_ch = ch

        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(in_ch=now_ch, out_ch=out_ch, tdim=tdim))
                now_ch = out_ch

                chs.append(now_ch)

            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, attn=True),
            ResBlock(now_ch, now_ch, tdim, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult

            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, attn=False))
                now_ch = out_ch

            if i != 0:
                self.upblocks.append(UpSample(now_ch))

        assert len(chs) == 0

        self.se = GCT(num_channels=now_ch,tdim=tdim)
        
        self.tail = nn.Sequential(
            nn.GroupNorm(16, now_ch),
            
            nn.GELU(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1, padding_mode='reflect'),
        )
        
    def weight_init(self):
        for m in self.modules():
            if isinstance(m,torch.nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    init.constant_(m.bias.data,0.1)
            elif isinstance(m,torch.nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,torch.nn.Linear):
                m.weight.data.normal_(0,0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, target):
        
        targetemb = self.target_embedding(target)
        h = self.head(x)
        hs = [h]

        for layer in self.downblocks:
            h = layer(h,  targetemb)
            hs.append(h)
        for layer in self.middleblocks:
            h = layer(h,  targetemb)
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h,  targetemb)

        h = self.se(h)
        h = self.tail(h)


        assert len(hs) == 0
        return (torch.tanh(h) + 1) / 2
