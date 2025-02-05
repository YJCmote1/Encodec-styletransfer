import torch
import torch.nn as nn
import functools
from torch.nn import init
from torch.optim import lr_scheduler
import modules as m
import quantization as qt
import math
import numpy as np
import typing as tp
from utils import _check_checksum, _linear_overlap_add, _get_checkpoint_url



def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            # lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            lr_l = 0.3 ** max(0, epoch + opt.epoch_count - opt.n_epochs)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=()):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

#新建Codecencoder
# class AudioEncoder(nn.Module):
#     def __init__(self,
#                  encoder: m.SEANetEncoder,
#                  decoder: m.SEANetDecoder,
#                  quantizer: qt.ResidualVectorQuantizer,
#                  target_bandwidths: tp.List[float],
#                  sample_rate: int,
#                  channels: int,
#                  normalize: bool = False,
#                  segment: tp.Optional[float] = None,
#                  overlap: float = 0.01,
#                  name: str = 'unset'):
#         super().__init__()
#         self.decoder = decoder
#         self.sample_rate = sample_rate
#         self.segment = segment
#         self.overlap = overlap
#         self.normalize = normalize
#         self.encoder = encoder
#         self.encoder = encoder
#         self.quantizer = quantizer
#         self.decoder = decoder
#         self.bandwidth: tp.Optional[float] = None
#         self.frame_rate = math.ceil(self.sample_rate / np.prod(self.encoder.ratios)) #75
#     @property
#     def segment_length(self) -> tp.Optional[int]:
#         if self.segment is None:
#             return None
#         return int(self.segment * self.sample_rate)
    
#     @property
#     def segment_stride(self) -> tp.Optional[int]:
#         segment_length = self.segment_length
#         if segment_length is None:
#             return None
#         return max(1, int((1 - self.overlap) * segment_length))

#     def encode(self, x: torch.Tensor) -> tp.List[EncodedFrame]:
#             """Given a tensor `x`, returns a list of frames containing
#             the discrete encoded codes for `x`, along with rescaling factors
#             for each segment, when `self.normalize` is True.

#             Each frames is a tuple `(codebook, scale)`, with `codebook` of
#             shape `[B, K, T]`, with `K` the number of codebooks.
#             """
#             assert x.dim() == 3
#             _, channels, length = x.shape
#             assert channels > 0 and channels <= 2
#             segment_length = self.segment_length 
#             if segment_length is None: #segment_length = 1*sample_rate
#                 segment_length = length
#                 stride = length
#             else:
#                 stride = self.segment_stride  # type: ignore
#                 assert stride is not None

#             encoded_frames: tp.List[EncodedFrame] = []
#             for offset in range(0, length, stride): # shift windows to choose data
#                 frame = x[:, :, offset: offset + segment_length]
#                 encoded_frames.append(self._encode_frame(frame))
#             return encoded_frames

#     def _encode_frame(self, x: torch.Tensor) -> EncodedFrame:
#         length = x.shape[-1] # tensor_cut or original
#         duration = length / self.sample_rate
#         assert self.segment is None or duration <= 1e-5 + self.segment

#         if self.normalize:
#             mono = x.mean(dim=1, keepdim=True)
#             volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
#             scale = 1e-8 + volume
#             x = x / scale
#             scale = scale.view(-1, 1)
#         else:
#             scale = None

#         emb = self.encoder(x) # [2,1,10000] -> [2,128,32]
#         #TODO: Encodec Trainer的training
#         if self.training:
#             return emb,scale
#         codes = self.quantizer.encode(emb, self.frame_rate, self.bandwidth)
#         codes = codes.transpose(0, 1)
#         # codes is [B, K, T], with T frames, K nb of codebooks.
#         return codes, scale
    
#     def decode(self, encoded_frames: tp.List[EncodedFrame]) -> torch.Tensor:
#         """Decode the given frames into a waveform.
#         Note that the output might be a bit bigger than the input. In that case,
#         any extra steps at the end can be trimmed.
#         """
#         segment_length = self.segment_length
#         if segment_length is None:
#             assert len(encoded_frames) == 1
#             return self._decode_frame(encoded_frames[0])

#         frames = [self._decode_frame(frame) for frame in encoded_frames]
#         return _linear_overlap_add(frames, self.segment_stride or 1)

#     def _decode_frame(self, encoded_frame: EncodedFrame) -> torch.Tensor:
#         codes, scale = encoded_frame
#         if self.training:
#             emb = codes
#         else:
#             codes = codes.transpose(0, 1)
#             emb = self.quantizer.decode(codes)
#         out = self.decoder(emb)
#         if scale is not None:
#             out = out * scale.view(-1, 1, 1)
#         return out

#     @staticmethod
#     def _get_model(target_bandwidths: tp.List[float],
#                    sample_rate: int = 24_000,
#                    channels: int = 1,
#                    causal: bool = True,
#                    model_norm: str = 'weight_norm',
#                    audio_normalize: bool = False,
#                    segment: tp.Optional[float] = None,
#                    name: str = 'unset',
#                    ratios=[8, 5, 4, 2]):
#         encoder = m.SEANetEncoder(channels=channels, norm=model_norm, causal=causal,ratios=ratios)
#         decoder = m.SEANetDecoder(channels=channels, norm=model_norm, causal=causal,ratios=ratios)
#         n_q = int(1000 * target_bandwidths[-1] // (math.ceil(sample_rate / encoder.hop_length) * 10)) # int(1000*24//(math.ceil(24000/320)*10))
#         quantizer = qt.ResidualVectorQuantizer(
#             dimension=encoder.dimension,
#             n_q=n_q,
#             bins=1024,
#         )
#         model = AudioEncoder(
#             encoder,
#             decoder,
#             quantizer,
#             target_bandwidths,
#             sample_rate,
#             channels,
#             normalize=audio_normalize,
#             segment=segment,
#             name=name,
#         )
#         return model

       



#将adaattn适用于音频
class AdaAttN(nn.Module):

    def __init__(self, in_planes, max_sample=256, key_planes=None):
        super(AdaAttN, self).__init__()
        if key_planes is None:
            key_planes = in_planes
        self.f = nn.Conv1d(key_planes, key_planes, 1)
        self.g = nn.Conv1d(key_planes, key_planes, 1)
        self.h = nn.Conv1d(in_planes, in_planes, 1)
        self.sm = nn.Softmax(dim=-1)
        self.max_sample = max_sample

    def forward(self, content, style, content_key, style_key, seed=None):
        # 输入维度假定为 [B, C, T]
        F = self.f(content_key)  # [B, C, T_c]
        G = self.g(style_key)    # [B, C, T_s]
        H = self.h(style)        # [B, C, T_s]

        # Flatten style for sampling
        B, C, T_s = G.size()
        G = G.view(B, C, T_s).permute(0, 2, 1)  # [B, T_s, C]
        if T_s > self.max_sample:
            if seed is not None:
                torch.manual_seed(seed)
            index = torch.randperm(T_s).to(content.device)[:self.max_sample]
            G = G[:, index, :]
            style_flat = H[:, :, index].permute(0, 2, 1)  # [B, max_sample, C]
        else:
            style_flat = H.permute(0, 2, 1)  # [B, T_s, C]

        # Compute attention map
        B, C, T_c = F.size()
        F = F.permute(0, 2, 1)  # [B, T_c, C]
        S = torch.bmm(F, G)  # Attention map [B, T_c, T_s]
        S = self.sm(S)       # Normalize attention map

        # Compute mean and std
        mean = torch.bmm(S, style_flat)  # [B, T_c, C]
        std = torch.sqrt(
            torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2)
        )  # [B, T_c, C]

        # Reshape back to [B, C, T_c]
        mean = mean.permute(0, 2, 1)  # [B, C, T_c]
        std = std.permute(0, 2, 1)    # [B, C, T_c]
        return std * mean_variance_norm(content) + mean


class Transformer(nn.Module):

    def __init__(self, in_planes, key_planes=None, shallow_layer=False):
        super(Transformer, self).__init__()
        self.attn_adain_4_1 = AdaAttN(in_planes=in_planes, key_planes=key_planes)
        self.attn_adain_5_1 = AdaAttN(in_planes=in_planes,
                                        key_planes=key_planes + 512 if shallow_layer else key_planes)
        #self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True) #音频任务
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        #self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))
        self.merge_conv = nn.Conv1d(in_planes, in_planes, (3, 3)) #音频任务

    def forward(self, content4_1, style4_1, content5_1, style5_1,
                content4_1_key, style4_1_key, content5_1_key, style5_1_key, seed=None):
        return self.merge_conv(self.merge_conv_pad(
            self.attn_adain_4_1(content4_1, style4_1, content4_1_key, style4_1_key, seed=seed) +
            self.upsample5_1(self.attn_adain_5_1(content5_1, style5_1, content5_1_key, style5_1_key, seed=seed))))


class Decoder(nn.Module):
    def __init__(self, skip_connection_3=False):
        super(Decoder, self).__init__()
        self.decoder_layer_1 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        )
        self.decoder_layer_2 = nn.Sequential(
            nn.Conv1d(256 + 256 if skip_connection_3 else 256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=3, padding=1)  # 输出 1 通道（单声道音频）
        )

    def forward(self, cs, c_adain_3_feat=None):
        cs = self.decoder_layer_1(cs)
        if c_adain_3_feat is None:
            cs = self.decoder_layer_2(cs)
        else:
            cs = self.decoder_layer_2(torch.cat((cs, c_adain_3_feat), dim=1))
        return cs
    