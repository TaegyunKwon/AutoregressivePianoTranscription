import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def checkpointByPass(f, *args):
    return f(*args)


class RMSNorm(nn.Module):
    def __init__(self, eps = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        var = x.pow(2).mean(dim = -1, keepdim=True)
        return x* torch.rsqrt(var+ self.eps)


class ResBlock(nn.Module):
    def __init__(self, module, size, prenorm = True, dropoutProb =0.0):
        super().__init__()
        self.module = module
        self.norm = RMSNorm()

        # LayerScale
        self.scale = nn.Parameter(torch.ones(size)*1e-2)
        self.dropout = nn.Dropout(dropoutProb)

    def forward(self, x, *args):
        return x + self.dropout(self.module(self.norm(x), *args))*self.scale

class SelfAttnWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        shape = x.shape
        x = x.flatten(0,1)
        result, _ = self.module(x,x,x)

        result = result.unflatten(0, shape[:2])
        return result




class LearnableSpatialPositionEmbedding(nn.Module):
    def __init__(self, embedSize, coordDim, gamma = 10.0, dropoutProb = 0.0):
        super().__init__()

        self.gamma = gamma
        self.proj = nn.Linear(coordDim, embedSize)

        self.mlp = nn.Sequential(
                nn.Linear(embedSize, 4*embedSize),
                nn.GELU(),
                nn.Dropout(dropoutProb),
                nn.Linear(4*embedSize, embedSize))


        self.dropout = nn.Dropout(dropoutProb)
        self._reset_parameters()


    def _reset_parameters(self):
        nn.init.normal_(self.proj.weight, std = 1/self.gamma)
        nn.init.uniform_(self.proj.bias, a = -math.pi, b = math.pi)

    """
    arguments:
        indices [nBatch, nDimCoordinates]
    """
    def forward(self, *coords):
        device = self.proj.weight.device
        coords = torch.meshgrid(coords, indexing="ij")
        coord = torch.stack(coords, dim = -1)

        phi = self.proj(coord.float())

        z = torch.cos(phi)/ math.sqrt(phi.shape[-1]/2)
        z = self.mlp(z)

        return z

    def forwardWithCoordVec(self, coord):
        device = self.proj.weight.device

        phi = self.proj(coord.float())

        z = torch.cos(phi)/ math.sqrt(phi.shape[-1]/2)
        z = self.mlp(z)

        return z

class LearnableSpatialPositionEmbedding1D(nn.Module):
    def __init__(self, embedSize, gamma = 10.0, dropoutProb = 0.0):
        super().__init__()

        self.gamma = gamma
        self.proj = nn.Linear(1, embedSize)

        self.mlp = nn.Sequential(
                nn.Linear(embedSize, 4*embedSize),
                nn.GELU(),
                nn.Dropout(dropoutProb),
                nn.Linear(4*embedSize, embedSize))


        self.dropout = nn.Dropout(dropoutProb)
        self._reset_parameters()


    def _reset_parameters(self):
        nn.init.normal_(self.proj.weight, std = 1/self.gamma)
        nn.init.uniform_(self.proj.bias, a = -math.pi, b = math.pi)

    def forward(self, coord):
        # coord: [N] or [B, N]
        if coord.dim() == 1:
            coord = coord.unsqueeze(-1) # [N, 1]
        elif coord.dim() == 2:
            coord = coord.unsqueeze(-1) # [B, N, 1]
        
        phi = self.proj(coord.float())

        z = torch.cos(phi)/ math.sqrt(phi.shape[-1]/2)
        z = self.mlp(z)

        return z


"""
The customized MHA layer using the approximated attention
adapted from the pytorch implementation
"""
class MultiHeadAttentionKernel(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout = 0., k_dim = None, v_dim = None, fourierSize = 32, kernel = "fourier", hiddenFactor = 1):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kernel = kernel


        hiddenSize = math.ceil(hiddenFactor*embed_dim)

        # make sure hiddenSize to be divisible by num_heads
        self.head_dim = int(math.ceil(hiddenSize/num_heads))
        hiddenSize = self.head_dim*num_heads



        if k_dim is None:
            k_dim = embed_dim

        if v_dim is None:
            v_dim = embed_dim


        self.fourierSize = fourierSize


        self.q_proj_weight = nn.Parameter(torch.empty(( embed_dim, hiddenSize)))
        self.k_proj_weight = nn.Parameter(torch.empty(( k_dim, hiddenSize)))
        self.v_proj_weight = nn.Parameter(torch.empty(( v_dim, hiddenSize)))
        self.out_proj = nn.Linear(hiddenSize, embed_dim)

        if kernel is not None:
            self.gamma = nn.Parameter(torch.tensor(1.0))
            self.norm = RMSNorm()

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj_weight)
        nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)

    def forward(self, query, key = None, value = None):
        if key == None:
            key = query

        if value == None:
            value = key

        q = query@self.q_proj_weight
        k = key@self.k_proj_weight
        v = value@self.v_proj_weight

        # split into heads
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = q.transpose(-2,-3)
        k = k.unflatten(-1, (self.num_heads, self.head_dim))
        k = k.transpose(-2,-3)
        v = v.unflatten(-1, (self.num_heads, self.head_dim))
        v = v.transpose(-2,-3)

        if self.kernel is not None:
            raise NotImplementedError
        else:
            fetched = F.scaled_dot_product_attention(q,k,v)

        fetched = fetched.transpose(-2,-3).flatten(-2,-1)

        result = self.out_proj(fetched)

        return result
     

class BasicBlock(nn.Module):
    def __init__(self,
            inputSize,
            num_heads,
            fourierSize,
            hiddenFactor = 2,
            hiddenFactorAttn = 1,
            approxKernels = [None,
                None,
                None, None],
            dropoutProb = 0.0):
        super().__init__()

        fnnHiddenSize = int(math.ceil(inputSize*hiddenFactor))


        self.mhaBlockF = ResBlock(
                MultiHeadAttentionKernel(
                    inputSize,
                    num_heads=num_heads,
                    fourierSize = fourierSize,
                    kernel = approxKernels[0],
                    hiddenFactor = hiddenFactorAttn),
                size = inputSize,
                dropoutProb = dropoutProb
                )

        self.fnnBlockF  = ResBlock(
                nn.Sequential(
                    nn.Linear(inputSize, fnnHiddenSize),
                    nn.GELU(),
                    nn.Dropout(dropoutProb),
                    nn.Linear(fnnHiddenSize, inputSize),
                    ),
                size = inputSize,
                dropoutProb = dropoutProb
                )

    def forward(self, x, mem = None, crossAttn = False):
        # x: [B, T, F, C]

        inShape = x.shape

        crossAttn = True 

        if mem is None:
            mem = x
            crossAttn = False

        h = x

        h = self.mhaBlockF(h, mem)
        h = self.fnnBlockF(h)



        outShape = h.shape
        assert inShape == outShape
        return h


class TransformerEncoder(nn.Module):
    def __init__(self, 
            in_channels,
            pos_embed_init_gamma,
            num_heads,
            fourier_size = 16,
            hidden_factor = 2,
            hidden_factor_attn = 1,
            dropout = 0.0,
            num_layers= 4,
            pos_T = False
            ):
        super().__init__()
        self.pos_T = pos_T

        self.posEmbedBuilderAttnTF = LearnableSpatialPositionEmbedding1D(
                in_channels,
                gamma = pos_embed_init_gamma,
                dropoutProb=dropout)


        self.posEmbedBuilderAttnTE = LearnableSpatialPositionEmbedding1D(
                in_channels,
                gamma = pos_embed_init_gamma, dropoutProb = dropout)


        encoderLayers = [BasicBlock( in_channels,
            num_heads = num_heads,
            fourierSize = fourier_size,
            dropoutProb = dropout,
            hiddenFactor = hidden_factor,
            hiddenFactorAttn = hidden_factor_attn,
            ) for i in range(num_layers)]

        self.encoderLayers = nn.ModuleList(encoderLayers)
        self.maintain_F = True
        self.out_shape = 'BPCT'

    def forward(self, x, outputIndices):
        if self.training:
            checkpoint = torch.utils.checkpoint.checkpoint
        else:
            checkpoint = checkpointByPass
        # x: [B, C, T, F]
        # out: [B, T, P, D]

        # change to [B, T, F, C] shape
        h = x.permute(0, 2, 3, 1)
        nT = h.shape[1]

        # append 1 time and 1 frequency aggregation track
        h = F.pad(h, (0,0,1,0, 1, 0))

        ################ transformer encoders
        coord_F = torch.arange(h.shape[-2], device = x.device).float()
        outputIndices =  outputIndices.float()
        posEmbed = self.posEmbedBuilderAttnTF(coord_F)
        posEmbedTgt = self.posEmbedBuilderAttnTE(outputIndices)
        # posEmbed: [F, D] -> [1, 1, F, D]
        posEmbed = posEmbed.unsqueeze(0).unsqueeze(0)
        # posEmbedTgt: [P, D] -> [1, 1, P, D]
        posEmbedTgt = posEmbedTgt.unsqueeze(0).unsqueeze(0)

        posEmbedTgt = posEmbedTgt.repeat(h.shape[0], h.shape[1], 1, 1)


        h = h + posEmbed
        hTarget = posEmbedTgt

        hAll = torch.cat( [h, hTarget], dim = -2)

        for l in self.encoderLayers:
            hAll = checkpoint(l, hAll)

        h, hTarget = hAll.split([h.shape[-2], hTarget.shape[-2]], dim = -2)

        # print(hTarget.std(), hTarget.mean())

        # remove the t=0 pooling track
        # [N, T, P, D]
        hTarget = hTarget[..., 1:, :, :]

        return hTarget